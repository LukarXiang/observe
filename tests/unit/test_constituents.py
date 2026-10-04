from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import job_status, main
from observe.data.constituents import combine_csi800, constituents_at, membership_records, normalize_constituents, update_constituents
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.jobs import Jobs
from observe.runs import file_sha

START, END = date(2024, 1, 2), date(2024, 1, 5)
STOCKS = [f'600{k:03d}.SH' for k in range(800)]


def response(index, day):
    stocks = STOCKS[:300] if index == '000300.SH' else STOCKS[300:]
    return pd.DataFrame({'updateDate': str(day), 'code': ['sh.' + i[:6] for i in stocks], 'code_name': 'stock'})


class Source:
    def __init__(self, fault = None): self.calls, self.fault = [], fault
    @contextmanager
    def session(self): yield self
    def index_constituents(self, index, day):
        self.calls.append((index, day)); frame = response(index, day)
        if index != '000905.SH' or self.fault is None: return frame
        if self.fault == 'empty': return frame.iloc[:0]
        if self.fault == 'missing': return frame.iloc[:-1]
        if self.fault == 'duplicate': frame.loc[1, 'code'] = frame.loc[0, 'code']
        if self.fault == 'schema': return frame.drop(columns = 'updateDate')
        if self.fault == 'future': frame['updateDate'] = '2026-10-01'
        if self.fault == 'stale': frame['updateDate'] = '2023-01-01'
        if self.fault == 'date': frame.loc[0, 'updateDate'] = 'broken'
        if self.fault == 'mixed': frame.loc[0, 'updateDate'] = '2024-01-01'
        if self.fault == 'unknown': frame.loc[0, 'code'] = 'sh.603999'
        if self.fault == 'overlap': frame.loc[0, 'code'] = 'sh.600000'
        if self.fault == 'none': return None
        if self.fault == 'network': raise TimeoutError('offline')
        return frame


@pytest.fixture
def store(tmp_path):
    st = Store(tmp_path); days = pd.date_range('2024-01-01', END)
    calendar = pd.DataFrame({'date': days.date, 'is_open': days.dayofweek < 5})
    master = pd.DataFrame({'instrument': ['000300.SH', '000905.SH', '000906.SH', *STOCKS], 'kind': ['index'] * 3 + ['stock'] * 800})
    st.publish(st.write_batch({'calendar': {'all': st.write_partition('calendar', 'all', calendar)}, 'instruments': {'all': st.write_partition('instruments', 'all', master)}}))
    return st


def test_historical_archive_atomic_and_repeated_without_network(store, capsys):
    root = store.root; sid = store.snapshot('before'); snapshot_sha = file_sha(root / 'snapshots' / f'{sid}.json'); source = Source()
    result = update_constituents(root, START, END, ['000906.SH', '000300.SH'], source = source)
    assert result['status'] == 'published' and result['requested_rows'] == 4400
    assert len(source.calls) == 8                     # 800 和 300 共用供应商响应
    frame = store.load('index_constituents'); assert not frame.strict_usable.any()
    assert constituents_at(root, '000906.SH', START)['total'] == 800
    assert constituents_at(root, '000906.SH', START, sid)['total'] == 0
    assert constituents_at(root, '000906.SH', '2024-01-06')['total'] == 0  # 不延续到未请求日
    assert file_sha(root / 'snapshots' / f'{sid}.json') == snapshot_sha
    before = store.published_path.read_bytes(); repeat = Source('network')
    again = update_constituents(root, START, END, ['000906.SH', '000300.SH'], source = repeat)
    assert again['status'] == 'unchanged' and again['reused_snapshots'] == 8 and not repeat.calls
    assert store.published_path.read_bytes() == before
    main(['--root', str(root), 'data', 'constituents-at', '--date', str(START), '--limit', '2'])
    assert json.loads(capsys.readouterr().out)['total'] == 800
    client = TestClient(create_app(root)); response_ = client.get('/api/data/constituents', params = {'index': '000906.SH', 'date': str(START), 'limit': 2})
    assert response_.status_code == 200 and len(response_.json()['data']) == 2 and not response_.json()['strict_evidence']
    assert client.get('/api/data/constituents', params = {'index': 'today', 'date': str(START)}).status_code == 400
    jid = Jobs(root).submit('constituents_update', {'start': str(START), 'end': str(END)})
    Jobs(root).run_next()                           # 已归档任务通过实际子进程完成，不联网
    assert Jobs(root).get(jid)['status'] == 'success'
    assert client.post('/api/jobs', json = {'kind': 'constituents_update', 'params': {'start': str(START), 'end': str(END), 'indices': ['000001.SH']}}).status_code == 400


@pytest.mark.parametrize('fault', ['empty', 'missing', 'duplicate', 'schema', 'future', 'stale', 'date', 'mixed', 'unknown', 'overlap', 'none', 'network'])
def test_bad_provider_data_does_not_publish_partial_union(store, fault):
    before = store.published_path.read_bytes()
    result = update_constituents(store.root, START, END, source = Source(fault))
    assert result['status'] == 'rejected' and result['issues'] and result['batch_id'] is None
    assert store.published_path.read_bytes() == before and constituents_at(store.root, '000906.SH', START)['total'] == 0
    assert job_status('constituents_update', result) == 'failed'
    assert json.loads(Path(result['report_file']).read_text())['status'] == 'rejected'
    if fault not in ('network', 'none'):
        assert result['requests'][-1]['raw_file']
        # 重叠名单各自完整，合并时拒绝；其余异常在单份名单校验时拒绝。
        assert result['requests'][-1]['status'] == ('validated' if fault == 'overlap' else 'rejected')


def test_date_gate_union_and_frozen_full_lists():
    a = normalize_constituents(response('000300.SH', START), '000300.SH', START, STOCKS, 'a')
    b = normalize_constituents(response('000905.SH', START), '000905.SH', START, STOCKS, 'b')
    union = combine_csi800(a, b)
    assert len(membership_records(union, '000906.SH', [START])) == 800
    with pytest.raises(ValueError, match = '缺查询日期'): membership_records(union, '000906.SH', [END])
    wrong = union.copy(); wrong['strict_usable'] = 'False'
    with pytest.raises(ValueError, match = '严格历史'): membership_records(wrong, '000906.SH', [START])
    wrong = union.copy(); wrong['source_date'] = END
    with pytest.raises(ValueError, match = '晚于'): membership_records(wrong, '000906.SH', [START])
    wrong = b.copy(); wrong['source_date'] = date(2024, 1, 1)
    with pytest.raises(ValueError, match = '日期不同'): combine_csi800(a, wrong)


def test_force_refresh_preserves_other_dates_and_old_snapshot(store):
    update_constituents(store.root, START, END, source = Source()); sid = store.snapshot('original'); old = store.load('index_constituents', sid)
    class Rename(Source):
        def index_constituents(self, index, day):
            f = response(index, day); f['code_name'] = 'renamed'; return f
    result = update_constituents(store.root, START, START, force = True, source = Rename())
    assert result['status'] == 'published'
    current = store.load('index_constituents')
    pd.testing.assert_frame_equal(current[current.date != START].reset_index(drop = True), old[old.date != START].reset_index(drop = True))
    pd.testing.assert_frame_equal(store.load('index_constituents', sid), old)


def test_writer_lock_and_master_checked_before_network(store):
    source = Source()
    with operation_lock(store.root, DATA_WRITER):
        with pytest.raises(RuntimeError, match = '锁被占用'): update_constituents(store.root, START, END, source = source)
    with pytest.raises(ValueError, match = '日历不完整'): update_constituents(store.root, '2023-12-31', END, source = source)
    assert not source.calls
    with pytest.raises(ValueError, match = '主键重复'):
        frame = normalize_constituents(response('000300.SH', START), '000300.SH', START, STOCKS, 'a')
        store.write_partition('index_constituents', '2024', pd.concat([frame, frame.iloc[:1]]))


def test_login_diagnostics_do_not_pollute_json_output(tmp_path, capsys):
    class Client:
        def login(self):
            print('login success!'); return type('Response', (), {'error_code': '0'})()
        def logout(self): print('logout success!')
    with BaoStock(tmp_path, Client()).session(): pass
    output = capsys.readouterr()
    assert output.out == '' and 'login success!' in output.err and 'logout success!' in output.err


def test_session_and_publish_failure_preserve_state_and_leave_report(store, monkeypatch):
    class Offline(Source):
        @contextmanager
        def session(self): raise ConnectionError('login failed'); yield
    before = store.published_path.read_bytes()
    result = update_constituents(store.root, START, END, source = Offline())
    assert result['status'] == 'rejected' and result['issues'][0]['rule'] == 'source_session_failed'
    assert store.published_path.read_bytes() == before
    def fail(*args): raise OSError('disk full')
    monkeypatch.setattr(Store, 'publish', fail)
    with pytest.raises(OSError, match = 'disk full'): update_constituents(store.root, START, END, source = Source())
    assert store.published_path.read_bytes() == before
    reports = [json.loads(p.read_text()) for p in (store.root / 'audits/constituents').glob('*.json')]
    assert any(r['status'] == 'failed' and 'disk full' in r['error'] for r in reports)
