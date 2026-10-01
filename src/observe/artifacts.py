"""实验产物的只读查询：CLI 与 API 共用；大表通过 DuckDB 按条件分页。"""
import json

import duckdb

from .replay import _read
from .runs import resolve_run

TABLES = {'universe': 'decision_date', 'factors': 'date', 'labels': 'decision_date', 'split_plan': None, 'predictions': 'decision_date',
          'factor_daily': 'date', 'factor_groups': 'date', 'equity': 'date', 'orders': 'decision_date', 'fills': 'date', 'positions': 'date',
          'cash_events': 'date', 'receivables': 'date', 'benchmark_daily': 'date'}


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
    if _read(out / 'config.json').get('kind') == 'experiment':
        subruns = _read(out / 'subruns.json')
        if table in ('universe', 'factors', 'labels', 'predictions', 'split_plan'): child = subruns['research']
        else:
            children = [x for x in subruns['backtests'] + subruns.get('benchmarks', []) if x['model'] == model and x['scenario'] == scenario]
            if not children: raise ValueError(f'实验没有 {model} / {scenario} 的账本产物')
            child = children[0]
        return table_path(root, child['output'], table, model, scenario)
    raise FileNotFoundError(f'实验 {out.name} 没有产物 {table}')


def run_table(root, run, table, limit = 100, offset = 0, start = None, end = None, instrument = None, model = None, scenario = 'base', benchmark = None):
    if table not in TABLES: raise ValueError(f'未知实验表 {table}')
    if not 1 <= limit <= 5000 or offset < 0: raise ValueError('limit 必须在 1–5000 内，offset 不得为负数')
    p = table_path(root, run, table, model or 'ridge', scenario); reader = 'read_parquet' if p.suffix == '.parquet' else 'read_json_auto'
    source = f'{reader}(?)'; params, clauses = [str(p)], []
    with duckdb.connect() as db:
        columns = {r[0] for r in db.execute(f'describe select * from {source}', params).fetchall()}
        day = TABLES[table]
        absent_filter = ((start is not None or end is not None) and day not in columns) or (instrument is not None and 'instrument' not in columns)
        if absent_filter and db.execute(f'select count(*) from {source}', params).fetchone()[0] == 0:
            return {'total': 0, 'limit': limit, 'offset': offset, 'rows': []}      # 全现金 / 无成交 JSON 没有可推断字段，仍是可筛选的空结果
        for value, op in ((start, '>='), (end, '<=')):
            if value is not None:
                if day not in columns: raise ValueError(f'{table} 没有可筛选的日期字段')
                clauses.append(f'"{day}" {op} ?'); params.append(value)
        if instrument is not None:
            if 'instrument' not in columns: raise ValueError(f'{table} 没有证券字段')
            clauses.append('instrument = ?'); params.append(instrument)
        if model is not None and 'model_id' in columns:
            clauses.append('model_id = ?'); params.append(model)
        if 'scenario' in columns:
            clauses.append('scenario = ?'); params.append(scenario)
        if benchmark is not None:
            if 'benchmark_id' not in columns: raise ValueError(f'{table} 没有基准字段')
            clauses.append('benchmark_id = ?'); params.append(benchmark)
        where = ' where ' + ' and '.join(clauses) if clauses else ''
        total = db.execute(f'select count(*) from {source}{where}', params).fetchone()[0]
        order = [c for c in (day, 'model_id', 'scenario', 'benchmark_id', 'instrument', 'factor', 'quantile', 'order_id', 'fill_id', 'split_id') if c and c in columns]
        sql = f'select * from {source}{where}' + (' order by ' + ','.join(f'"{c}"' for c in dict.fromkeys(order)) if order else '') + ' limit ? offset ?'
        frame = db.execute(sql, params + [limit, offset]).df()
    return {'total': int(total), 'limit': limit, 'offset': offset, 'rows': json.loads(frame.to_json(orient = 'records', date_format = 'iso', force_ascii = False))}


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
