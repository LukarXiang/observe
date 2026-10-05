"""只读验收财务归档、冻结查询和导入前分区引用；可保存新的验证凭据。"""
import argparse
from pathlib import Path

import duckdb

from observe.data.financial_history import query_financial_history
from observe.data.financials import TABLES, read_financial_archive
from observe.data.store import KEYS, Store
from observe.runs import canonical, environment, file_sha, write_json


def verify(root, snapshot, baseline_snapshot):
    store = Store(root); state = store.state(snapshot); baseline = store.state(baseline_snapshot)
    tables = (*TABLES, 'financial_fields', 'financial_availability'); counts = {}
    with duckdb.connect() as db:
        for table in tables:
            parts = state['tables'].get(table, {})
            if not parts: raise ValueError(f'缺少财务表 {table}')
            rel = db.read_parquet([str(store.root / v['file']) for v in parts.values()], union_by_name = True)
            db.register('checked', rel)
            count = db.execute('select count(*) from checked').fetchone()[0]
            keys = ','.join('"' + column.replace('"', '""') + '"' for column in KEYS[table])
            duplicate = db.execute(f'select count(*) from (select {keys} from checked group by {keys} having count(*) > 1)').fetchone()[0]
            expected = sum(v['rows'] for v in parts.values())
            if count != expected or duplicate: raise ValueError(f'{table}: 行数或来源主键不一致')
            counts[table] = {'rows': count, 'partitions': len(parts), 'duplicate_keys': duplicate}
        strict_rows = db.execute('select count(*) from checked where strict_usable').fetchone()[0]
        if strict_rows: raise ValueError('最终整理来源不应有严格可用记录')
    unchanged = all(state['tables'].get(t) == parts for t, parts in baseline['tables'].items())
    if not unchanged: raise ValueError('导入前已有表的分区引用发生变化')
    strict = query_financial_history(root, snapshot, ['net_profit_ytd', 'total_assets'], ['000333.SZ', '600519.SH'], '2024-06-28 16:00')
    exploratory = query_financial_history(root, snapshot, ['net_profit_ytd', 'total_assets'], ['000333.SZ', '600519.SH'],
                                         '2024-06-28 16:00', mode = 'exploratory', lag_days = 120)
    archive = read_financial_archive(root, 'financial_annual', snapshot, ['000333.SZ'], '2023-01-01', '2023-12-31',
                                    ['instrument', 'report_period', 'net_profit_ytd', 'source_sha256', 'source_row'])
    if len(strict.data) or not len(exploratory.data) or not len(archive): raise ValueError('真实财务查询未符合严格/探索/归档契约')
    return canonical({'status': 'ok', 'snapshot': snapshot, 'baseline_snapshot': baseline_snapshot, 'batch_id': state['batch_id'],
                      'old_partition_references_unchanged': unchanged, 'tables': counts, 'strict_usable_rows': strict_rows,
                      'strict_query': strict.coverage, 'exploratory_query': exploratory.coverage, 'annual_sample': archive.to_dict('records'),
                      'environment': environment(), 'baseline_snapshot_sha256': file_sha(store.root / 'snapshots' / f'{baseline_snapshot}.json')})


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--root', default = 'data'); parser.add_argument('--snapshot', required = True)
    parser.add_argument('--baseline-snapshot', required = True); parser.add_argument('--output')
    args = parser.parse_args()
    if args.output and Path(args.output).exists(): raise FileExistsError(args.output)
    result = verify(args.root, args.snapshot, args.baseline_snapshot)
    if args.output: write_json(args.output, result)
    import json
    print(json.dumps(result, ensure_ascii = False, indent = 2))


if __name__ == '__main__': main()
