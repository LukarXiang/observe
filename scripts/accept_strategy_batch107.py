"""Accept two complete source reviews while keeping ETF trading gates closed."""
import argparse
import ast
import json
from pathlib import Path
import shutil

import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch107 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch107/20261011-formal-source-acceptance-v1')
RECEIPT = Path('docs/handoff/2026-10-11-batch107-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零七批ETF动量与均线-正式研究验收.md')
SPECS = ('docs/handoff/batch107-y2023-027-source-review.yaml', 'docs/handoff/batch107-y2024b-061-source-review.yaml')


def reviews():
    result = {}
    for k, name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == study.SOURCES[k] and doc['sha256'] == study.SOURCE_SHA[k] and
            doc['definitions'] == (13 if k == 0 else 12) and
            doc['formal_source_research_acceptance'] is False and doc['original_strategy_complete'] is False,
            'Source identity or review scope differs')
        result[doc['source']] = {'review_status': '人工审查完成', 'status': '待数据',
            'scope': doc['scope'], 'rules': doc['rules'], 'gaps': doc['gaps'], 'next_step': doc['next'],
            'implementations': [], 'original_strategy_complete': False,
            'fidelity': '完整原文/原函数诊断；27三原生基金最长29行回归和排名参考，61原24池缺失；原交易0',
            'future_information_risks': ['固定标的事后选择；当前供应商原生历史不证明平台复权与当时可交易身份']}
    return result


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and len(selected) == 2, 'Catalog scope differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and set(previous) == {r['path'] for r in new}, 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated or missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
            {k: v for k, v in before.items() if k not in allowed} == {k: v for k, v in row.items() if k not in allowed}, 'Static identity changed')
        require(all(row[k] == v for k, v in allowed.items()), 'Literal source rules changed')
        require(row['original_strategy_complete'] is False and row['implementations'] == [] and row['status'] == '待数据', 'Fabricated trading acceptance')
    require(sum(r['review_status'] == '人工审查完成' for r in new) == 300, 'Reviewed count differs')
    return {'sources': 695, 'reviewed': 300, 'remaining': 395, 'unrelated_sources_unchanged': 693}


def components():
    path = study.DIRECTORY/'checked.json'
    proof = read(path)
    require(proof['status'] == 'ok_component_stage' and proof['saved_exact_readback'] is True and
        proof['formal_source_research_acceptance'] is False and proof['new_trading_backtests'] == 0 and
        proof['offline_differences'] == 0 and proof['fresh_network_blocked_processes'] == 2, 'Component scope differs')
    for name, sha in proof['evidence_sha256'].items(): require(file_sha(name) == sha, 'Component evidence changed: '+name)
    require(proof['summary']['original61_actual_pool_windows'] == 0 and
        proof['summary']['original61_pool_reduced'] is False and proof['summary']['complete_original_signals'] == 0,
        'Missing original61 inputs were substituted')
    summary = proof['summary']
    require((summary['native_windows'], summary['common_ranking_windows'], summary['diagnostic_cases'],
        summary['independent_order_differences']) == (11663, 3170, 30, 0), 'Longest verified scope differs')
    scales = read('data/staging/strategies-batch107/20261011-gap-scale-identity-v1/checked.json')
    require(scales['status'] == 'ok_operand_identity_only' and scales['master399306'][0]['kind'] == 'index' and
        len(scales['large_native_price_jumps_gt50pct']) == 3 and
        scales['registry_scan']['matches_for_23_funds'] == [] and scales['repaired_prices'] == 0,
        'Raw scale or original index identity was silently changed')
    for name, sha in scales['input_sha256'].items(): require(file_sha(name) == sha, 'Scale evidence input changed')
    copied = read(study.DIRECTORY/'copied-tests.json')
    require(copied['local_tests'] == copied['isolated_tests'] == 32, 'Required local/isolated checks differ')
    for label in ('local', 'isolated'):
        command = read(study.DIRECTORY/(label+'-tests.json'))
        origins = json.loads(command['stdout'].splitlines()[0])['actual_imports']
        root = Path(command['cwd']).resolve()
        require(command['returncode'] == 0 and len(origins) == 4 and
            all(Path(p).is_relative_to(root) for p in origins), 'Actual import origin differs')
    for k in (0, 1):
        require(file_sha(study.DIRECTORY/'sources'/f'{k}.source') ==
            file_sha(Path('repo/量化策略源代码')/study.SOURCES[k]) == study.SOURCE_SHA[k], 'Complete original source changed')
    return {k: proof[k] for k in ('summary', 'protection', 'fresh_network_blocked_processes',
        'offline_differences', 'new_trading_backtests', 'new_strategy_parent_runs')}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal stage exists')
    component = components()
    study.runtime_binding(study.DIRECTORY)
    old_pointer = read('data/catalog/strategies/latest.json')
    parent = read(study.PARENT)
    require(old_pointer['catalog_id'] == parent['progress']['catalog_id'], 'Formal parent catalog changed')
    old = read(Path(old_pointer['directory'])/'catalog.json')
    selected = reviews()
    before = Path('src/observe/strategy_catalog.py').read_bytes()
    proposed = catalog_program(before, selected)
    DIRECTORY.mkdir(parents=True)
    (DIRECTORY/'strategy_catalog.before.py').write_bytes(before)
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(proposed)
    ast.parse(proposed)
    originals = dict(REVIEWS)
    try:
        REVIEWS.update(selected)
        digests = []
        for label in ('candidate', 'replay'):
            root = DIRECTORY/label
            target = root/'catalog/strategies/implementation-evidence.json'
            target.parent.mkdir(parents=True)
            shutil.copyfile(study.DIRECTORY/'implementation-evidence.before.json', target)
            generated = catalog_strategies(root, 'repo/量化策略源代码')
            folder = Path(generated['output'])
            candidate_checks(old, read(folder/'catalog.json'), selected)
            digests.append({n: file_sha(folder/n) for n in ('catalog.json', 'summary.json', 'catalog.parquet')})
            if label == 'candidate': candidate = generated
        require(digests[0] == digests[1], 'Fresh candidate catalog bytes differ')
    finally:
        REVIEWS.clear()
        REVIEWS.update(originals)
    paths = [*SPECS, str(study.DIRECTORY/'checked.json'), str(study.PARENT), str(study.PROBE),
        'scripts/accept_strategy_batch107.py', 'tests/unit/test_batch107_acceptance.py',
        str(study.DIRECTORY/'copied-tests.json'),
        'data/staging/strategies-batch107/20261011-gap-scale-identity-v1/checked.json']
    save(DIRECTORY/'prepared.json', {'component': component, 'previous_catalog': old_pointer,
        'candidate': candidate, 'catalog_bytes': digests[0],
        'old_code_sha256': file_sha(DIRECTORY/'strategy_catalog.before.py'),
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py'),
        'evidence_sha256': {p: file_sha(p) for p in paths}})
    return {'status': 'candidate_verified_not_published', 'catalog_id': candidate['catalog_id']}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists')
    doc = read(DIRECTORY/'prepared.json')
    selected = reviews()
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Prepared evidence changed')
    require(components() == doc['component'] and REPORT.exists(), 'Formal result/report missing or changed')
    require(file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
        all(REVIEWS.get(n) == v for n, v in selected.items()), 'Runtime source review differs')
    tests = read(DIRECTORY/'tests.json')
    require(tests['returncode'] == 0 and '39 passed' in tests['stdout'] and tests['network_blocked'] is True, 'Formal checks missing')
    folder = Path(doc['candidate']['output'])
    old = read(Path(doc['previous_catalog']['directory'])/'catalog.json')
    candidate_checks(old, read(folder/'catalog.json'), selected)
    for name, sha in doc['catalog_bytes'].items(): require(file_sha(folder/name) == sha, 'Candidate changed')
    with operation_lock('data', DATA_WRITER):
        require(read('data/catalog/strategies/latest.json') == doc['previous_catalog'], 'Catalog pointer changed')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']
        dest.mkdir(exist_ok=False)
        for name, sha in doc['catalog_bytes'].items():
            shutil.copyfile(folder/name, dest/name)
            require(file_sha(dest/name) == sha, 'Published bytes differ')
        _atomic_json(Path('data/catalog/strategies/latest.json'), {'catalog_id': dest.name,
            'directory': str(dest), 'sha256': file_sha(dest/'catalog.json')})
        progress = archive('data', 'Batch107 two full ETF sources accepted;longest native29-row regressions/three-fund rankings and32 local+32 copied/39 formal tests;original ETF trades blocked')
        require((progress['manually_reviewed'], progress['locally_backtested_sources'],
            progress['locally_reproduced_sources'], progress['local_strategy_runs']) == (300, 12, 12, 87), 'Actual progress differs')
        paths = [*doc['evidence_sha256'], str(REPORT), str(DIRECTORY/'prepared.json'), str(DIRECTORY/'tests.json'),
            'src/observe/strategy_catalog.py', str(dest/'catalog.json'), str(Path(progress['output'])/'summary.json')]
        save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
            'original_strategy_complete': False, 'library_goal_complete': False, 'progress': progress,
            'components': doc['component'], 'source_sha256': list(study.SOURCE_SHA),
            'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
            'evidence_sha256': {p: file_sha(p) for p in paths},
            'review_method': 'Full original byte-bound sources;literal diagnostics;full saved operands/independent scalar regression;fresh offline and copied import checks;no independent agent review'})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 300, 'remaining': 395}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'publish'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), ensure_ascii=False, allow_nan=False))
