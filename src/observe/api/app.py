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
        """审计明细和元数据必须来自同一个已提交产物。"""
        b = batch or store.published()['batch_id']
        path = root / 'audits'
        match = []
        for m in path.glob('*.json'):
            try:
                info = json.loads(m.read_text(encoding = 'utf-8'))
                if info.get('batch_id') == b and info.get('audit_id'): match.append(info)
            except (OSError, ValueError): continue
        info = max(match, key = lambda x: x.get('audited_at', '')) if match else None
        csv = path / f"{info['audit_id']}.issues.csv" if info else None
        complete = bool(info and csv.exists())
        rows = _records(pd.read_csv(csv)) if complete else []
        status = 'not_audited' if not info else ('problem' if not complete or rows else 'passed')
        if not info:
            latest = path / 'latest.json'
            try:
                old = json.loads(latest.read_text(encoding = 'utf-8'))
                if old.get('batch_id') != b: status = 'expired'
            except (OSError, ValueError): pass
        return {'source': f"audit {info['audit_id']}" if info else f'batch {b}', 'batch_id': b, 'status': status, 'rows': rows}

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
