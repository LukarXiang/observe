"""本机 Web 接口（模块 19）：只读查询走 DuckDB 读已发布分区；写数据一律提交任务，由工作进程执行。只监听 127.0.0.1。"""
import json
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..data.store import Store
from ..jobs import KINDS, SUPPORTED, Jobs


class JobIn(BaseModel):
    kind: str
    params: dict = {}


def _records(df): return json.loads(df.to_json(orient = 'records', date_format = 'iso', force_ascii = False))


def create_app(root):
    root = Path(root); store, jobs = Store(root), Jobs(root); app = FastAPI(title = 'observe')

    def files(table, state):
        return [(root / v['file']).as_posix() for v in state['tables'].get(table, {}).values()]

    def query(table, sql, params = (), state = None):
        f = files(table, state or store.published())
        if not f: return pd.DataFrame()
        return duckdb.connect().execute(sql.replace('{t}', f"read_parquet({f!r})"), list(params)).df()

    @app.get('/api/data/status')
    def status():
        pub = store.published(); return {'batch_id': pub['batch_id'], 'published_at': pub.get('published_at'),
                                         'tables': {t: {'partitions': len(v), 'rows': sum(x['rows'] for x in v.values())} for t, v in pub['tables'].items()}}

    @app.get('/api/data/coverage')
    def coverage():
        df = query('bars_1d', "select year(date) as year, count(distinct date) as n_days, count(*) as n_rows, count(*) filter (where is_trading) as trading_rows, "
                              "count(distinct instrument) as n_instruments from {t} group by 1 order by 1")
        return _records(df)

    @app.get('/api/data/daily')
    def daily():
        df = query('bars_1d', "select date, count(*) as n_rows, count(*) filter (where is_trading) as n_trading, count(*) filter (where is_st) as n_st "
                              "from {t} group by 1 order by 1")
        return _records(df)

    @app.get('/api/data/snapshots')
    def snapshots(): return sorted((p.stem for p in (root / 'snapshots').glob('*.json')), reverse = True)

    @app.get('/api/data/issues')
    def issues(batch: str | None = None):
        """默认读最近一次 observe data audit 的重审结果（按当前规则），没有时读已发布批次的审计结果"""
        latest = root / 'coverage' / 'audit_latest.csv'; b = batch or store.published()['batch_id']; p = root / 'batches' / f'{b}.issues.csv'
        src, path = ('audit_latest', latest) if batch is None and latest.exists() else (f'batch {b}', p)
        return {'source': src, 'rows': _records(pd.read_csv(path)) if path.exists() else []}

    @app.get('/api/instruments')
    def instruments(q: str = '', limit: int = 50):
        df = query('instruments', "select * from {t} where instrument ilike ? or name ilike ? order by instrument limit ?", (f'%{q}%', f'%{q}%', limit)); return _records(df)

    @app.get('/api/instruments/{inst}/bars')
    def bars(inst: str, start: str = '1990-01-01', end: str = '2099-12-31', price: str = 'raw'):
        state = store.published()
        df = query('bars_1d', "select * from {t} where instrument = ? and date between ? and ? order by date", (inst, start, end), state = state)
        if price == 'adj' and len(df):
            from ..data.prices import with_adjusted
            adj = query('adj_factors', "select * from {t} where instrument = ?", (inst,), state = state)
            cov = query('adj_coverage', "select * from {t} where instrument = ?", (inst,), state = state)
            df = with_adjusted(df, adj, cov if len(cov) else None)
        return _records(df)

    @app.get('/api/instruments/{inst}/actions')
    def actions(inst: str):
        state = store.published(); return _records(query('corp_actions', "select * from {t} where instrument = ? order by ex_date", (inst,), state = state))

    @app.get('/api/jobs')
    def list_jobs(limit: int = 50): return jobs.list(limit)

    @app.post('/api/jobs')
    def submit(j: JobIn):
        if j.kind not in KINDS: raise HTTPException(400, f'未知任务种类 {j.kind}')
        if j.kind not in SUPPORTED: raise HTTPException(400, f'任务种类 {j.kind} 尚未实现')
        return {'job_id': jobs.submit(j.kind, j.params)}

    @app.get('/api/jobs/{jid}')
    def get_job(jid: str):
        r = jobs.get(jid)
        if r is None: raise HTTPException(404, '任务不存在')
        return r

    @app.get('/api/jobs/{jid}/log')
    def log(jid: str, offset: int = 0):
        r = get_job(jid); p = Path(r['log_path'])
        if not p.exists(): return {'offset': offset, 'text': ''}
        b = p.read_bytes(); return {'offset': len(b), 'text': b[offset:].decode('utf-8', 'replace')}

    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid: str): return {'cancelled': jobs.cancel(jid)}

    @app.post('/api/jobs/{jid}/retry')
    def retry(jid: str):
        try: return {'job_id': jobs.retry(jid)}
        except ValueError as e: raise HTTPException(400, str(e)) from e

    dist = Path(__file__).resolve().parents[3] / 'web' / 'dist'
    if dist.exists(): app.mount('/', StaticFiles(directory = dist, html = True), name = 'web')
    return app
