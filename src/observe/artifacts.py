"""实验产物的只读查询：CLI 与 API 共用；大表通过 DuckDB 按条件分页。"""
import csv
import io
import json
from datetime import date
from pathlib import Path

import duckdb

from .replay import _read
from .runs import resolve_run

TABLES = {'universe': 'decision_date', 'factors': 'date', 'labels': 'decision_date', 'split_plan': None, 'predictions': 'decision_date',
          'factor_daily': 'date', 'factor_groups': 'date', 'equity': 'date', 'orders': 'decision_date', 'fills': 'date', 'positions': 'date',
          'cash_events': 'date', 'receivables': 'date', 'benchmark_daily': 'date'}
TABLES.update(scores = 'decision_date', targets = 'decision_date', signal_coverage = 'date')


def run_detail(root, run):
    out = resolve_run(root, run); status = _read(out / 'status.json'); config = _read(out / 'config.json') if (out / 'config.json').exists() else None
    return {'run_id': status['run_id'], 'output': str(out), 'status': status, 'config': config,
            'limitations': _read(out / 'limitations.json') if (out / 'limitations.json').exists() else status.get('limitations', []),
            'reports': {n: _read(out / f'{n}.json') for n in ('report', 'model_eval', 'factor_eval', 'factor_diagnostics', 'model_comparison', 'benchmark_eval', 'cache', 'metrics', 'trading', 'paired_eval') if (out / f'{n}.json').exists()},
            'artifacts': sorted(_read(out / 'manifest.json').get('files', {})) if (out / 'manifest.json').exists() else []}


def table_path(root, run, table, model = 'ridge', scenario = 'base'):
    out = resolve_run(root, run); name = 'positions_daily' if table == 'positions' else table
    for suffix in ('parquet', 'json'):
        p = out / f'{name}.{suffix}'
        if p.is_file(): return p
    kind = _read(out / 'config.json').get('kind')
    if kind in ('experiment', 'strategy'):
        subruns = _read(out / 'subruns.json')
        if table in ('universe', 'factors', 'labels', 'predictions', 'split_plan'): child = subruns['research']
        else:
            if kind == 'strategy' and model == 'ridge': model = _read(out / 'config.json')['config']['implementation']
            children = [x for x in subruns['backtests'] + subruns.get('benchmarks', []) if x['model'] == model and x['scenario'] == scenario]
            if not children: raise ValueError(f'实验没有 {model} / {scenario} 的账本产物')
            child = children[0]
        return table_path(root, child['output'], table, model, scenario)
    raise FileNotFoundError(f'实验 {out.name} 没有产物 {table}')


def _open_query(root, run, table, start = None, end = None, instrument = None, model = None, scenario = 'base', benchmark = None,
                sort_by = None, descending = False):
    """分页和导出共用筛选 / 排序 SQL；字段来自实际 schema，值使用绑定参数。"""
    if table not in TABLES: raise ValueError(f'未知实验表 {table}')
    start, end = (date.fromisoformat(str(v)) if v is not None else None for v in (start, end))
    if start is not None and end is not None and start > end: raise ValueError('start 不得晚于 end')
    if descending and sort_by is None: raise ValueError('降序必须指定 sort_by')
    p = table_path(root, run, table, model or 'ridge', scenario); reader = 'read_parquet' if p.suffix == '.parquet' else 'read_json_auto'
    source = f'{reader}(?)'; params, clauses = [str(p)], []
    db = duckdb.connect()
    try:
        columns = [r[0] for r in db.execute(f'describe select * from {source}', params).fetchall()]
        day = TABLES[table]
        # 空 JSON 数组无 schema；不存在的日期 / 证券筛选仍返回空结果。
        empty_json = p.suffix == '.json' and columns == ['json'] and db.execute(f'select count(*) from {source}', params).fetchone()[0] == 0
        if sort_by is not None and sort_by not in columns: raise ValueError(f'未知排序字段 {sort_by}')
        quote = lambda c: '"' + c.replace('"', '""') + '"'
        for value, op in ((start, '>='), (end, '<=')):
            if value is not None and not empty_json:
                if day not in columns: raise ValueError(f'{table} 没有可筛选的日期字段')
                clauses.append(f'{quote(day)} {op} ?'); params.append(value.isoformat())
        if instrument is not None and not empty_json:
            if 'instrument' not in columns: raise ValueError(f'{table} 没有证券字段')
            clauses.append('instrument = ?'); params.append(instrument)
        if model is not None and 'model_id' in columns:
            clauses.append('model_id = ?'); params.append(model)
        if 'scenario' in columns:
            clauses.append('scenario = ?'); params.append(scenario)
        if benchmark is not None and not empty_json:
            if 'benchmark_id' not in columns: raise ValueError(f'{table} 没有基准字段')
            clauses.append('benchmark_id = ?'); params.append(benchmark)
        where = ' where ' + ' and '.join(clauses) if clauses else ''
        base = f'select * from {source}{where}'
        natural = [c for c in (day, 'model_id', 'scenario', 'benchmark_id', 'instrument', 'factor', 'quantile', 'order_id', 'fill_id', 'split_id') if c and c in columns]
        # 所有剩余列用于打破并列，跨页排序稳定；完全相同的行没有可观察差异。
        order = list(dict.fromkeys(([sort_by] if sort_by else []) + natural + columns))
        suffix = ' order by ' + ','.join(quote(c) + (' desc' if c == sort_by and descending else ' asc') + ' nulls last' for c in order)
        return db, base, suffix, params, [] if empty_json else columns
    except Exception:
        db.close(); raise


def run_table(root, run, table, limit = 100, offset = 0, start = None, end = None, instrument = None, model = None, scenario = 'base', benchmark = None,
              sort_by = None, descending = False):
    if not 1 <= limit <= 5000 or offset < 0: raise ValueError('limit 必须在 1–5000 内，offset 不得为负数')
    db, base, order, params, columns = _open_query(root, run, table, start, end, instrument, model, scenario, benchmark, sort_by, descending)
    with db:
        total = db.execute(f'select count(*) from ({base})', params).fetchone()[0]
        frame = db.execute(base + order + ' limit ? offset ?', params + [limit, offset]).df()
    return {'total': int(total), 'limit': limit, 'offset': offset, 'rows': json.loads(frame.to_json(orient = 'records', date_format = 'iso', force_ascii = False))}


class TableCSV:
    """打开查询后逐批读取，API 响应前完成参数验证；close 可重复调用。"""
    def __init__(self, root, run, table, **filters):
        self.db, base, order, params, self.columns = _open_query(root, run, table, **filters)
        self.header = False; self.rows = 0; self.closed = False
        try: self.db.execute(base + order, params)
        except Exception: self.close(); raise

    def __iter__(self): return self

    def __next__(self):
        if self.closed: raise StopIteration
        try:
            buf = io.StringIO(newline = ''); writer = csv.writer(buf)
            if not self.header:
                self.header = True
                if self.columns: writer.writerow(self.columns)
                return buf.getvalue()
            rows = self.db.fetchmany(4096)
            if not rows: self.close(); raise StopIteration
            writer.writerows(rows); self.rows += len(rows)
            return buf.getvalue()
        except BaseException:
            self.close(); raise

    def close(self):
        if not self.closed: self.closed = True; self.db.close()


def export_table(root, run, table, output, **filters):
    """导出全部筛选行；目标独占创建，失败仅移除本次创建的未完成文件。"""
    stream = TableCSV(root, run, table, **filters); output = Path(output); created = False
    try:
        with output.open('x', encoding = 'utf-8', newline = '') as handle:
            created = True
            for chunk in stream: handle.write(chunk)
        return {'output': str(output.resolve()), 'table': table, 'rows': stream.rows, 'columns': stream.columns}
    except BaseException:
        if created: output.unlink(missing_ok = True)
        raise
    finally: stream.close()


def compare_runs(root, runs):
    """并列展示已存指标；不同设计不冒充正式成对证据。正式逐日差值来自 paired / experiment 的 model_comparison。"""
    if len(runs) < 2: raise ValueError('至少指定两个实验')
    details = [run_detail(root, r) for r in runs]; rows = []
    for d in details:
        st, doc, reports = d['status'], d['config'] or {}, d['reports']; valid = st['status'] in ('success', 'success_limited')
        rows.append({'run_id': d['run_id'], 'kind': st.get('kind'), 'status': st['status'], 'snapshot_id': doc.get('snapshot_id'), 'evidence': st.get('evidence'),
                     'metrics': reports.get('metrics') if valid else None, 'model_eval': reports.get('model_eval') if valid else None,
                     'report': reports.get('report') if valid else None, 'limitations': d['limitations']})
    return {'runs': rows, 'same_snapshot': len({r['snapshot_id'] for r in rows}) == 1,
            'note': '已保存指标的并列表；区间、模型、组合或执行配置不一致时不能据此声称有增量，正式比较使用冻结预测的成对评价'}
