"""本机 Web 接口（模块 19）：只读查询走 DuckDB 读已发布分区；写数据一律提交任务，由工作进程执行。只监听 127.0.0.1。"""
import json
from datetime import date as Date
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..data.store import Store
from ..jobs import KINDS, SUPPORTED, Jobs
from ..runs import REPO, RunRegistry


class JobIn(BaseModel):
    kind: str
    params: dict = {}


class ExprIn(BaseModel):
    expr: str = Field(min_length = 1, max_length = 10000)


class DoctorIn(BaseModel):
    model_config = {'extra': 'forbid'}
    snapshot: str | None = None
    config: dict | None = None
    verify_files: bool = False


def _records(df): return json.loads(df.to_json(orient = 'records', date_format = 'iso', force_ascii = False))


def create_app(root):
    root = Path(root); store, jobs = Store(root), Jobs(root); app = FastAPI(title = 'observe'); registry = RunRegistry(root / 'runs'); registry.index()

    def read_result(fn, *args, **kwargs):
        try: return fn(*args, **kwargs)
        except FileNotFoundError as exc: raise HTTPException(404, str(exc)) from exc
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc

    @app.get('/api/doctor')
    def doctor_get(snapshot: str | None = None, verify_files: bool = False):
        from ..doctor import doctor
        return doctor(root, snapshot = snapshot, verify_files = verify_files)

    @app.post('/api/doctor')
    def doctor_post(body: DoctorIn):
        from ..doctor import doctor
        return doctor(root, **body.model_dump())

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

    @app.get('/api/data/financial-history')
    def financial_history(snapshot: str, fields: str, instruments: str, decision_time: str, mode: str = 'strict', lag_days: int | None = None, table: str = 'financial_quarterly'):
        from ..data.financial_history import query_financial_history
        result = read_result(query_financial_history, root, snapshot, fields.split(','), instruments.split(','), decision_time, mode, lag_days, archive_table = table)
        return {'coverage': result.coverage, 'data': _records(result.data)}

    @app.get('/api/strategies')
    def strategies(status: str | None = None, limit: int = Query(100, ge = 1, le = 1000), offset: int = Query(0, ge = 0)):
        def read():
            latest = json.loads((root / 'catalog/strategies/latest.json').read_text(encoding = 'utf-8'))
            path = root / 'catalog/strategies' / latest['catalog_id'] / 'catalog.json'
            records = json.loads(path.read_text(encoding = 'utf-8'))
            if status: records = [r for r in records if r['status'] == status]
            return {'catalog_id': latest['catalog_id'], 'total': len(records), 'data': records[offset:offset + limit]}
        return read_result(read)

    @app.get('/api/data/constituents')
    def constituents(index: str, date: Date, snapshot: str | None = None, limit: int = Query(1000, ge = 1, le = 1000), offset: int = Query(0, ge = 0)):
        from ..data.constituents import constituents_at
        return read_result(constituents_at, root, index, date, snapshot, limit, offset)

    @app.get('/api/data/issues')
    def issues(batch: str | None = None):
        """审计明细和元数据必须来自同一个已提交产物；返回审计范围（scope / input_range），增量通过不显示成全快照通过。"""
        from ..data.audit import audit_status
        from ..data.update import default_rules
        try: rules = default_rules()
        except Exception: rules = None                                          # 规则加载失败 → rules_unavailable，不绕过指纹校验
        r = audit_status(root, batch or store.published()['batch_id'], rules)
        return {**r, 'rows': _records(r['rows']) if len(r['rows']) else []}

    @app.get('/api/instruments')
    def instruments(q: str = '', limit: int = 50):
        df = query('instruments', "select * from {t} where instrument ilike ? or name ilike ? order by instrument limit ?", (f'%{q}%', f'%{q}%', limit)); return _records(df)

    @app.get('/api/indices/{index}/bars')
    def index_bars(index: str, start: str = '1990-01-01', end: str = '2099-12-31', snapshot: str | None = None,
                   limit: int = Query(1000, ge = 1, le = 10000), offset: int = Query(0, ge = 0)):
        from ..data.indices import index_bars as read_index
        return read_result(read_index, root, index, start, end, snapshot, limit, offset)

    @app.get('/api/instruments/{inst}/bars')
    def bars(inst: str, start: str = '1990-01-01', end: str = '2099-12-31', price: str = 'raw'):
        state = store.published()
        df = query('bars_1d', "select * from {t} where instrument = ? and date between ? and ? order by date", (inst, start, end), state = state)
        if price == 'adj' and len(df):
            from ..data.prices import with_adjusted
            adj = query('adj_factors', "select * from {t} where instrument = ?", (inst,), state = state)
            cov = query('adj_coverage', "select * from {t} where instrument = ?", (inst,), state = state)
            df = with_adjusted(df, adj, cov)
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
        return read_result(lambda: {'job_id': jobs.submit(j.kind, j.params)})

    @app.get('/api/factors')
    def factors(factor_set: str = 'daily_basic_v1'):
        from ..features import load_factor_set
        if factor_set not in ('daily_basic_v1', 'daily_intraday_v1'): raise HTTPException(400, '未知内置因子集')
        return load_factor_set(REPO / 'configs/factor_sets' / f'{factor_set}.yaml')['factors']

    @app.post('/api/factors/validate')
    def validate_factor(body: ExprIn):
        from ..factors import parse
        p = read_result(parse, body.expr); return {'expr': body.expr, 'fields': sorted(p.fields), 'lookback': p.lookback}

    @app.get('/api/factors/{name}/evaluation')
    def factor_evaluation(name: str, run: str):
        from ..artifacts import run_detail
        reports = read_result(run_detail, root, run)['reports']; report = reports.get('factor_diagnostics') or reports.get('factor_eval')
        if report is None and reports.get('report'): report = reports['report']['factor']['diagnostics']
        if report is None or name not in report.get('factors', {}): raise HTTPException(404, '实验没有该因子的评价')
        return report['factors'][name]

    @app.get('/api/runs')
    def runs(limit: int = Query(50, ge = 1, le = 1000), offset: int = Query(0, ge = 0), kind: str | None = None, status: str | None = None):
        return registry.list(limit, offset, kind, status)

    @app.get('/api/runs/compare')
    def compare(ids: str):
        from ..artifacts import compare_runs
        return read_result(compare_runs, root, ids.split(','))

    @app.get('/api/runs/{rid}')
    def run(rid: str):
        from ..artifacts import run_detail
        return read_result(run_detail, root, rid)

    @app.get('/api/runs/{rid}/verify')
    def verify(rid: str, recursive: bool = True):
        from ..integrity import verify_run
        return read_result(verify_run, root, rid, recursive)

    @app.get('/api/runs/{rid}/{table}')
    def run_table(rid: str, table: str, limit: int = Query(100, ge = 1, le = 5000), offset: int = Query(0, ge = 0),
                  start: Date | None = None, end: Date | None = None, date: Date | None = None, instrument: str | None = None, model: str | None = None, scenario: str = 'base', benchmark: str | None = None,
                  sort_by: str | None = None, descending: bool = False):
        from ..artifacts import run_table as read_table
        return read_result(read_table, root, rid, table, limit, offset, date or start, date or end, instrument, model, scenario, benchmark, sort_by, descending)

    @app.get('/api/runs/{rid}/{table}/csv')
    def export_run_table(rid: str, table: str, start: Date | None = None, end: Date | None = None, date: Date | None = None,
                         instrument: str | None = None, model: str | None = None, scenario: str = 'base', benchmark: str | None = None,
                         sort_by: str | None = None, descending: bool = False):
        from ..artifacts import TableCSV
        stream = read_result(TableCSV, root, rid, table, start = date or start, end = date or end, instrument = instrument,
                             model = model, scenario = scenario, benchmark = benchmark, sort_by = sort_by, descending = descending)
        return StreamingResponse(stream, media_type = 'text/csv; charset=utf-8',
                                 headers = {'Content-Disposition': f'attachment; filename="{table}.csv"'}, background = BackgroundTask(stream.close))

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
