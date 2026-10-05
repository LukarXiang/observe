"""指数入库公开服务：主键审计、整批发布、失败留证、幂等和旧快照保护。"""
from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import main, job_status, run_kind
from observe.data.indices import IndexUpdateConfig, index_bars, update_indices
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.jobs import Jobs
from observe.runs import file_sha

START, END = date(2024, 1, 1), date(2024, 1, 7)
DAYS = list(pd.bdate_range(START, END).date)


def response(code = 'sh.000300', days = DAYS):
    return pd.DataFrame({'date': [str(d) for d in days], 'code': code, 'open': '100', 'high': '102', 'low': '99',
                         'close': '101', 'preclose': '100', 'volume': '0', 'amount': '123456'})


class Source:
    def __init__(self, frames = None): self.frames = frames or {}; self.calls = []

    @contextmanager
    def session(self): yield self

    def index_daily(self, code, start, end):
        self.calls.append((code, start, end)); result = self.frames.get(code, response(code))
        if isinstance(result, Exception): raise result
        return result


@pytest.fixture
def seeded(tmp_path):
    store = Store(tmp_path); dates = pd.date_range(START, END)
    calendar = pd.DataFrame({'date': dates.date, 'is_open': dates.dayofweek < 5})
    inst = pd.DataFrame({'instrument': ['000300.SH', '000905.SH', '600000.SH'], 'kind': ['index', 'index', 'stock'], 'name': ['300', '500', 'stock']})
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)},
                                     'instruments': {'all': store.write_partition('instruments', 'all', inst)}}))
    return store


def test_publish_idempotency_and_snapshot_isolation(seeded):
    store = seeded; root = store.root; old = store.published(); sid = store.snapshot('before')
    snapshot_sha = file_sha(root / 'snapshots' / f'{sid}.json')
    source = Source(); result = update_indices(root, START, END, source = source)
    assert result['status'] == 'published' and result['problem_count'] == 0
    assert source.calls == [('sh.000300', START, END)]
    assert result['indices']['000300.SH']['status'] == 'validated'
    pd.testing.assert_frame_equal(store.load('calendar'), store.load('calendar', sid))
    assert store.published()['tables']['instruments'] == old['tables']['instruments']
    assert index_bars(root, snapshot = sid)['total'] == 0
    assert file_sha(root / 'snapshots' / f'{sid}.json') == snapshot_sha
    assert index_bars(root)['total'] == 5
    before = store.published_path.read_bytes()
    again = update_indices(root, START, END, source = Source())
    assert again['status'] == 'unchanged' and store.published_path.read_bytes() == before
    assert len(list((root / 'batches').glob('*.json'))) == 2


@pytest.mark.parametrize('fault,rule', [
    ('missing', 'missing_day'), ('duplicate', 'duplicate_key'), ('wrong_code', 'unexpected_index'),
    ('weekend', 'unexpected_date'), ('invalid_date', 'unexpected_date'), ('zero', 'bad_price'), ('nan', 'bad_price'),
    ('infinite', 'bad_price'), ('ohlc', 'ohlc_order'), ('negative_volume', 'bad_volume_amount'),
    ('schema', 'fetch_or_schema_failed'), ('empty', 'missing_day'), ('none', 'fetch_or_schema_failed'), ('network', 'fetch_or_schema_failed'),
])
def test_rejection_retains_published_state_and_evidence(seeded, fault, rule):
    root = seeded.root; update_indices(root, START, END, source = Source()); before = seeded.published_path.read_bytes()
    frame = response()
    if fault == 'missing': frame = frame.iloc[1:]
    elif fault == 'duplicate': frame = pd.concat([frame, frame.iloc[:1]], ignore_index = True)
    elif fault == 'wrong_code': frame.loc[0, 'code'] = 'sh.600000'
    elif fault == 'weekend': frame.loc[0, 'date'] = '2024-01-06'
    elif fault == 'invalid_date': frame.loc[0, 'date'] = 'not-a-date'
    elif fault == 'zero': frame.loc[0, 'close'] = '0'
    elif fault == 'nan': frame.loc[0, 'close'] = ''
    elif fault == 'infinite': frame.loc[0, 'close'] = 'inf'
    elif fault == 'ohlc': frame.loc[0, 'high'] = '99'
    elif fault == 'negative_volume': frame.loc[0, 'volume'] = '-1'
    elif fault == 'schema': frame = frame.drop(columns = ['volume'])
    elif fault == 'empty': frame = frame.iloc[:0]
    elif fault == 'none': frame = None
    else: frame = ConnectionError('offline')
    result = update_indices(root, START, END, source = Source({'sh.000300': frame}))
    assert result['status'] == 'rejected' and result['batch_id'] is None
    assert seeded.published_path.read_bytes() == before
    assert rule in set(pd.read_csv(result['issues_file']).rule)
    assert json.loads(Path(result['report_file']).read_text(encoding = 'utf-8'))['status'] == 'rejected'
    assert job_status('data_index', result) == 'failed'


def test_batch_is_atomic_when_one_of_two_indices_fails(seeded):
    before = seeded.published_path.read_bytes()
    result = update_indices(seeded.root, START, END, ['000300.SH', '000905.SH'], Source({'sh.000905': TimeoutError('timeout')}))
    assert result['status'] == 'rejected' and result['indices']['000300.SH']['status'] == 'validated'
    assert seeded.published_path.read_bytes() == before and not seeded.load('index_1d').shape[0]


def test_refresh_preserves_other_indices_and_unrequested_dates(seeded):
    store = seeded; root = store.root
    update_indices(root, START, END, ['000300.SH', '000905.SH'], Source())
    sid = store.snapshot('two indices'); old = store.load('index_1d', sid)
    patch = response(days = DAYS[1:2]); patch.loc[0, 'close'] = '102'
    result = update_indices(root, DAYS[1], DAYS[1], source = Source({'sh.000300': patch}))
    assert result['status'] == 'published'
    current = store.load('index_1d'); changed = (current['index'] == '000300.SH') & (current.date == DAYS[1])
    assert current.loc[changed, 'close'].tolist() == [102.0]
    pd.testing.assert_frame_equal(current[~changed], old[~changed])
    assert store.load('index_1d', sid).equals(old)


def test_inputs_and_lock_checked_before_source(seeded):
    source = Source()
    with pytest.raises(ValueError, match = '未确认的指数'):
        update_indices(seeded.root, START, END, ['600000.SH'], source)
    with pytest.raises(ValueError, match = '日历不完整'):
        update_indices(seeded.root, '2023-12-31', END, source = source)
    with operation_lock(seeded.root, DATA_WRITER):
        with pytest.raises(RuntimeError, match = '锁被占用'): update_indices(seeded.root, START, END, source = source)
    assert not source.calls


def test_session_and_publication_failures_leave_reports(seeded, monkeypatch):
    class Offline(Source):
        @contextmanager
        def session(self): raise ConnectionError('login failed'); yield
    result = update_indices(seeded.root, START, END, source = Offline())
    assert result['status'] == 'rejected' and 'source_session_failed' in set(pd.read_csv(result['issues_file']).rule)
    def fail(*args): raise OSError('disk full')
    before = seeded.published_path.read_bytes(); monkeypatch.setattr(Store, 'publish', fail)
    with pytest.raises(OSError, match = 'disk full'): update_indices(seeded.root, START, END, source = Source())
    reports = [json.loads(p.read_text()) for p in (seeded.root / 'index_audits').glob('*.json')]
    assert any(r['status'] == 'failed' and 'disk full' in r['error'] for r in reports)
    assert seeded.published_path.read_bytes() == before


def test_cli_queue_api_share_validation_and_paginated_snapshot_reads(seeded, monkeypatch, capsys):
    from observe.data import indices
    monkeypatch.setattr(indices, 'BaoStock', lambda root: Source())
    root = seeded.root; sid = seeded.snapshot('no index'); client = TestClient(create_app(root))
    params = {'start': str(START), 'end': str(END)}
    job = client.post('/api/jobs', json = {'kind': 'data_index', 'params': params}).json()['job_id']
    stored = Jobs(root).get(job)
    assert json.loads(stored['params'])['indices'] == ['000300.SH']
    result = run_kind(root, 'data_index', json.loads(stored['params']))
    assert result['status'] == 'published'
    main(['--root', str(root), 'data', 'index', '--start', str(START), '--end', str(END)])
    assert json.loads(capsys.readouterr().out)['status'] == 'unchanged'
    main(['--root', str(root), 'data', 'index-bars', '--limit', '2', '--offset', '1'])
    cli = json.loads(capsys.readouterr().out)
    api = client.get('/api/indices/000300.SH/bars', params = {'limit': 2, 'offset': 1})
    assert api.json() == cli and cli['total'] == 5 and len(cli['rows']) == 2
    assert client.get('/api/indices/000300.SH/bars', params = {'snapshot': sid}).json()['total'] == 0
    assert client.get('/api/indices/000300.SH/bars', params = {'snapshot': 'missing'}).status_code == 404
    assert client.get('/api/indices/000300.SH/bars', params = {'snapshot': '../PUBLISHED'}).status_code == 400
    for extra in ({'indices': ['000300']}, {'indices': []}, {'start': '2025-01-01'}, {'unknown': 1}):
        assert client.post('/api/jobs', json = {'kind': 'data_index', 'params': {**params, **extra}}).status_code == 400
    monkeypatch.setattr(indices, 'BaoStock', lambda root: Source({'sh.000300': None}))
    with pytest.raises(SystemExit) as error: main(['--root', str(root), 'data', 'index', '--start', str(START), '--end', str(END)])
    assert error.value.code == 1


def test_invalid_config_rejected():
    for indices in ([], ['000300.SH', '000300.SH'], ['sh.000300'], ['000300']):
        with pytest.raises(ValueError): IndexUpdateConfig(start = START, end = END, indices = indices)


def test_unknown_calendar_flag_is_not_a_holiday(seeded):
    cal = seeded.load('calendar'); cal['is_open'] = cal.is_open.astype(object); cal.loc[0, 'is_open'] = None
    seeded.publish(seeded.write_batch({'calendar': {'all': seeded.write_partition('calendar', 'all', cal)}}))
    source = Source()
    with pytest.raises(ValueError, match = '开市标记缺失'): update_indices(seeded.root, START, END, source = source)
    assert not source.calls
