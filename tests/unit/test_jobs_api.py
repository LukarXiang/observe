"""任务队列与 Web 接口（模块 19）：原子领取、取消与重试、重启恢复、子进程执行、命令行与网页同时提交下载的互斥。"""
import threading
from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.data.store import Store
from observe.jobs import Jobs
from observe.cli import run_kind
from tests.unit.test_data import hold
from tests.unit.test_data import DAYS, FakeBS

DATA_WRITER = 'data-writer'


def seed(root):
    s = Store(root)
    bars = pd.DataFrame({'date': [date(2025, 6, 23), date(2025, 6, 24)], 'instrument': ['600000.SH'] * 2, 'open': [10.0, 10.2], 'high': 10.5, 'low': 9.9,
                         'close': [10.1, 10.3], 'preclose': [10.0, 10.1], 'volume': 100, 'amount': 1000.0, 'is_trading': True, 'is_st': False, 'board': 'main'})
    inst = pd.DataFrame({'instrument': ['600000.SH', '000001.SZ'], 'name': ['浦发银行', '平安银行'], 'kind': 'stock'})
    adj = pd.DataFrame({'instrument': ['600000.SH'], 'ex_date': [date(2025, 6, 24)], 'back_factor': [2.0]})
    s.publish(s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', bars)}, 'instruments': {'all': s.write_partition('instruments', 'all', inst)},
                             'adj_factors': {'all': s.write_partition('adj_factors', 'all', adj)}}))
    return s


def test_claim_is_atomic_across_threads(tmp_path):
    q = Jobs(tmp_path); ids = {q.submit('snapshot') for _ in range(20)}; got = []
    def take():
        while (j := q.claim()) is not None: got.append(j)
    t = [threading.Thread(target = take) for _ in range(4)]; [x.start() for x in t]; [x.join() for x in t]
    assert sorted(got) == sorted(ids)                                            # 每个任务恰好被领取一次


def test_cancel_retry_and_recover(tmp_path):
    q = Jobs(tmp_path); a, b = q.submit('snapshot'), q.submit('gc')
    assert q.cancel(a) and q.get(a)['status'] == 'cancelled' and not q.cancel(a)
    q.claim(); q.claim()
    with q._db() as c: c.execute("update jobs set pid = 999999 where job_id = ?", (b,))   # 进程已不存在
    assert q.recover() == [b] and q.get(b)['status'] == 'interrupted'
    r = q.retry(b); assert q.get(r)['retry_of'] == b and q.get(r)['status'] == 'queued'


def test_worker_runs_jobs_in_subprocess_and_records_failure(tmp_path):
    seed(tmp_path); q = Jobs(tmp_path); ok, bad = q.submit('snapshot', {'note': 't'}), q.submit('factor_eval')
    q.worker(once = True)
    assert q.get(ok)['status'] == 'success' and 'snapshot_id' in q.get(ok)['result']
    assert q.get(bad)['status'] == 'failed' and '尚未实现' in q.get(bad)['error']


def test_web_and_cli_download_are_mutually_exclusive(tmp_path):
    """命令行正在直接执行更新（持数据写入锁）时，网页提交的下载任务立即失败，且在联网之前就失败"""
    client = TestClient(create_app(tmp_path)); p = hold(tmp_path, DATA_WRITER)
    try:
        jid = client.post('/api/jobs', json = {'kind': 'data_update', 'params': {'start': '2025-06-23', 'end': '2025-06-24'}}).json()['job_id']
        Jobs(tmp_path).run_next()
    finally: p.wait()
    j = client.get(f'/api/jobs/{jid}').json(); assert j['status'] == 'failed' and '锁被占用' in j['error']
    assert not (tmp_path / 'raw' / 'requests.jsonl').exists()                     # 没有发出任何数据请求
    assert '锁被占用' in client.get(f'/api/jobs/{jid}/log').json()['text']


def test_cli_data_update_uses_public_service_without_nested_writer_lock(tmp_path, monkeypatch):
    from observe.data import update as update_module
    from observe.data.sources.baostock import BaoStock
    monkeypatch.setattr(update_module, 'BaoStock', lambda root: BaoStock(root, FakeBS(DAYS)))
    result = run_kind(tmp_path, 'data_update', {'start': str(DAYS[0]), 'end': str(DAYS[0]), 'no_actions': True})
    assert result['status'] == 'published' and result['verified_days'] == 1


def test_api_reads_published_data(tmp_path):
    seed(tmp_path); c = TestClient(create_app(tmp_path))
    assert c.get('/api/data/status').json()['tables']['bars_1d']['rows'] == 2
    assert c.get('/api/data/coverage').json()[0]['n_days'] == 2
    assert [x['instrument'] for x in c.get('/api/instruments', params = {'q': '浦发'}).json()] == ['600000.SH']
    raw = c.get('/api/instruments/600000.SH/bars').json(); adj = c.get('/api/instruments/600000.SH/bars', params = {'price': 'adj'}).json()
    assert [x['close'] for x in raw] == [10.1, 10.3] and [x['close_adj'] for x in adj] == [None, 20.6]   # 首个复权事件前无可信基准，保持缺失
    assert c.post('/api/jobs', json = {'kind': 'nope'}).status_code == 400


def test_api_audit_status_is_explicit_and_batch_bound(tmp_path):
    seed(tmp_path); c = TestClient(create_app(tmp_path))
    assert c.get('/api/data/issues').json()['status'] == 'not_audited'
    run_kind(tmp_path, 'data_audit', {})
    passed = c.get('/api/data/issues').json(); assert passed['status'] == 'passed' and passed['batch_id'] == Store(tmp_path).published()['batch_id']
    s = Store(tmp_path); s.publish(s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', s.load('bars_1d', parts = ['2025']))}}))
    assert c.get('/api/data/issues').json()['status'] == 'expired'
