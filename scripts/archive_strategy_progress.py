"""Archive catalog coverage and local strategy runs without changing experiments."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import secrets

from observe.data.store import Store
from observe.runs import file_sha, write_json


def read(path): return json.loads(path.read_text(encoding = 'utf-8'))


def archive(root, note):
    root = Path(root); latest = read(root / 'catalog/strategies/latest.json')
    catalog = root / 'catalog/strategies' / latest['catalog_id'] / 'catalog.json'
    records = read(catalog); runs = []
    identities = {(r['path'], r['bytes_sha256']): r['strategy_id'] for r in records}
    for config in sorted((root / 'runs').glob('*/config.json')):
        doc = read(config)
        if doc.get('kind') != 'strategy': continue
        output = config.parent; status = read(output / 'status.json')
        report = read(output / 'report.json') if (output / 'report.json').exists() else {}
        source = doc.get('source', {}); source_sha = source.get('bytes_sha256')
        sid = source.get('strategy_id') or identities.get((source.get('path'), source_sha))
        runs.append({'run_id': output.name, 'strategy_id': sid,
                     'source_sha256': source_sha, 'implementation': doc['config'].get('implementation', doc['config'].get('name')),
                     'snapshot': doc.get('snapshot_id'), 'start': doc['config']['start'], 'end': doc['config']['end'],
                     'status': status['status'], 'reproduction': status.get('reproduction'),
                     'results': report.get('results', []), 'report': str(output / 'report.json'),
                     'manifest_present': (output / 'manifest.json').exists()})
    coverage = []
    for record in records:
        local = [r for r in runs if r['strategy_id'] == record['strategy_id'] and r['source_sha256'] == record['bytes_sha256']]
        successful = [r for r in local if r['status'] in ('success', 'success_limited')]
        matched = [r for r in successful if (r.get('reproduction') or {}).get('result') == 'match']
        coverage.append({'strategy_id': record['strategy_id'], 'path': record['path'], 'source_sha256': record['bytes_sha256'],
                         'catalog_status': record['status'], 'review_status': record['review_status'],
                         'fidelity': record['fidelity'], 'gaps': record['gaps'], 'next_step': record['next_step'],
                         'duplicate_of': record['duplicate_of'], 'implementations': record['implementations'],
                         'local_run_ids': [r['run_id'] for r in local], 'local_backtested': bool(successful),
                         'local_reproduction_match': bool(matched), 'original_strategy_complete': False})
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
    destination = root / 'catalog/strategies/progress' / stamp
    destination.mkdir(parents = True, exist_ok = False)
    summary = {'checkpoint': stamp, 'note': note, 'catalog_id': latest['catalog_id'], 'catalog_sha256': file_sha(catalog),
               'published_batch': Store(root).published()['batch_id'], 'sources': len(coverage),
               'catalog_status_counts': dict(Counter(r['catalog_status'] for r in coverage)),
               'manually_reviewed': sum(r['review_status'] == '人工审查完成' for r in coverage),
               'locally_backtested_sources': sum(r['local_backtested'] for r in coverage),
               'locally_reproduced_sources': sum(r['local_reproduction_match'] for r in coverage),
               'local_strategy_runs': len(runs),
               'caveat': 'Static classification is not manual review; local status is not a fresh integrity check; all implementations remain component/approximate evidence.'}
    write_json(destination / 'summary.json', summary)
    write_json(destination / 'sources.json', coverage); write_json(destination / 'runs.json', runs)
    lines = ['# 策略研究进度', '', f'阶段：`{stamp}`；{note}', '',
             f'来源 {len(coverage)}；人工审查 {summary["manually_reviewed"]}；本机已回测来源 {summary["locally_backtested_sources"]}；本机复现匹配来源 {summary["locally_reproduced_sources"]}。', '',
             '全量逐来源状态见同目录 `sources.json`，逐实验结果见 `runs.json`。静态分类不代表人工审查，成功状态不代表本次重新核验完整性，近似/组件不代表原策略完整复现。', '',
             '| 本机实验 | 实现 | 区间 | 状态 | 基础净收益 | 最大回撤 | 离线复现 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for run in runs:
        base = next((r for r in run['results'] if r.get('scenario') == 'base'), {})
        values = base.get('metrics') or {}
        fmt = lambda key: f'{values[key]:.4%}' if values.get(key) is not None else '未登记'
        lines.append(f'| {run["run_id"]} | {run["implementation"]} | {run["start"]}..{run["end"]} | {run["status"]} | {fmt("total_return")} | {fmt("max_drawdown")} | {(run.get("reproduction") or {}).get("result", "未登记")} |')
    (destination / 'progress.md').write_text('\n'.join(lines) + '\n', encoding = 'utf-8')
    write_json(root / 'catalog/strategies/progress/latest.json', {'checkpoint': stamp, 'directory': str(destination),
               'summary_sha256': file_sha(destination / 'summary.json')})
    return {**summary, 'output': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--root', default = 'data'); parser.add_argument('--note', required = True)
    args = parser.parse_args(); print(json.dumps(archive(args.root, args.note), ensure_ascii = False))
