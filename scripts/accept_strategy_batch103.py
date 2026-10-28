"""Accept ETF6/81 source research, preserving all original trade blockers and frozen inputs."""
import argparse
import ast
import json
from pathlib import Path
import shutil

import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import reference_strategy_batch103 as price
from scripts import reference_strategy_batch103_emotion as volume
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch103/20261011-formal-source-acceptance')
CHECK = Path('data/staging/strategies-batch103/20261011-formal-validation-v3')
RECEIPT = Path('docs/handoff/2026-10-11-batch103-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零三批实盘ETF与随机平局动量-正式研究验收.md')
SPECS = ('docs/handoff/batch103-y2023-006-formal-source-review.yaml',
         'docs/handoff/batch103-y2025-081-formal-source-review.yaml')


def reviews():
    selected = {}
    for number, name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == price.study.SOURCES[number] and doc['source_sha256'] == price.study.SOURCE_SHA[number] and
            doc['formal_acceptance'] is False and doc['original_strategy_complete'] is False and
            doc['economic_variant_approved'] is False, 'Original source/spec identity differs')
        selected[doc['source']] = {'review_status': '人工审查完成', 'status': '待数据',
            'scope': doc['review_scope'], 'rules': doc['rules'], 'fidelity': doc['allowed_evidence'],
            'gaps': doc['data_gaps']+doc['defects'], 'next_step': doc['next'],
            'implementations': [], 'original_strategy_complete': False}
    return selected


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and set(selected) == set(price.study.SOURCES), 'Catalog scope differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and set(previous) == {r['path'] for r in new}, 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated or missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and before.get('duplicate_of') is None and row.get('duplicate_of') is None,
                'Prior review or independent-source mapping differs')
        require({k: v for k, v in row.items() if k not in allowed} == {k: v for k, v in before.items() if k not in allowed},
                'Static source identity changed')
        require(all(row[k] == v for k, v in allowed.items()) and row['status'] == '待数据' and
            not row['original_strategy_complete'] and row['implementations'] == [], 'Rule or trading claim differs')
    require(sum(r['review_status'] == '人工审查完成' for r in old) == 290 and
        sum(r['review_status'] == '人工审查完成' for r in new) == 292, 'Manual count differs')
    return {'sources': 695, 'reviewed': 292, 'remaining': 403, 'changed': sorted(selected), 'new_independent_sources': 2}


def offline_checks(root, count):
    proof = read(root/'offline-verification.json')
    require(proof['status'] == 'ok' and not proof['differences'] and proof['fresh_network_blocked_processes'] == 2 and
        len(proof['outputs_sha256']) == count, 'Two-worker scope differs')
    for command in proof['commands']:
        require(command['returncode'] == 0 and file_sha(command['log']) == command['log_sha256'], 'Worker did not succeed')
    for name, sha in proof['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(name.replace('/compute/', '/replay/')), 'Actual replay bytes differ')
    return proof


def components():
    # Call this before catalog-code publication: the historical graph binds the old code byte SHA.
    price.validate(); volume.validate(); phase = read(volume.PHASE)
    require(phase['status'] == 'ok_stage_only' and not phase['formal_source_research_acceptance'] and
        phase['progress']['manually_reviewed'] == 290, 'Historical stage scope differs')
    for name, sha in phase['evidence_sha256'].items(): require(file_sha(name) == sha, 'Historical stage changed')
    offline_checks(price.DIRECTORY, 4); offline_checks(volume.DIRECTORY, 3)
    guard = read(CHECK/'checked.json')
    require(guard['status'] == 'ok_formal_ready' and guard['tests'] == {'local': 48, 'isolated': 48} and
        guard['saved_readback']['price']['saved_cells_checked'] == 633290 and
        guard['saved_readback']['volume']['windows'] == 48437, 'Fresh formal checks incomplete')
    require(len(guard['actual_imports']) == 5 and all(Path(x).is_relative_to((CHECK/'isolated').resolve())
        for x in guard['actual_imports'].values()), 'Actual copied imports differ')
    for name, sha in guard['evidence_sha256'].items(): require(file_sha(name) == sha, 'Formal evidence changed: '+name)
    for name, sha in guard['copied_file_sha256'].items(): require(file_sha(CHECK/'isolated'/name) == sha, 'Copied evidence changed')
    for tag in ('local', 'isolated'):
        command = read(CHECK/(tag+'-command.json'))
        require(command['returncode'] == 0 and file_sha(CHECK/(tag+'.log')) == command['log_sha256'], 'Formal tests did not succeed')
    prices = read(price.DIRECTORY/'compute/summary.json'); heat = read(volume.DIRECTORY/'compute/summary.json')
    require(prices['real14_row_price_windows'] == 28078 and prices['momentum31_row_price_windows'] == 17157 and
        heat['windows'] == 48437 and prices['new_trading_backtests'] == heat['new_trading_backtests'] == 0,
        'Longest reference scope differs')
    actual = Store('data').published(); baseline = read(volume.DIRECTORY/'baseline.json')
    require(actual['batch_id'] == phase['progress']['published_batch'] and actual['tables'] == baseline['published']['tables'] and
        file_sha(Path('data/snapshots')/(price.SNAPSHOT+'.json')) == price.SNAPSHOT_SHA, 'Actual publication/snapshot changed')
    return {'price_reference': prices, 'native_volume_and_explicit_float_reference': heat,
        'original_diagnostics': 45, 'dtype_diagnostics': 2, 'tests': guard['tests'],
        'complete_saved_readback': guard['saved_readback'], 'offline_output_differences': 0,
        'original_platform_history_dtype_proven': False, 'new_trading_backtests': 0,
        'original_strategy_complete': False}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); parent = read(price.PARENT); active = read('data/catalog/strategies/latest.json')
    require(active['catalog_id'] == parent['progress']['catalog_id'] and
        file_sha('src/observe/strategy_catalog.py') == parent['evidence_sha256']['src/observe/strategy_catalog.py'], 'Accepted parent catalog changed')
    old = read(Path(active['directory'])/'catalog.json'); before = Path('src/observe/strategy_catalog.py').read_bytes()
    DIRECTORY.mkdir(parents=True); (DIRECTORY/'strategy_catalog.py.before').write_bytes(before)
    shutil.copyfile(volume.DIRECTORY/'baseline.json', DIRECTORY/'baseline.json')
    save(DIRECTORY/'source-reviews.json', selected)
    save(DIRECTORY/'source-AST.json', {str(n): {node.name: ast.dump(node, include_attributes=False) for node in price.study.tree(n).body
        if isinstance(node, ast.FunctionDef)} for n in (0, 1)})
    REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(price.DIRECTORY/'implementation-evidence.before.json', target)
        proposed = catalog_strategies(root, 'repo/量化策略源代码'); folder = Path(proposed['output'])
        candidate_checks(old, read(folder/'catalog.json'), selected)
        digests.append({name: file_sha(folder/name) for name in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': candidate = proposed
    require(digests[0] == digests[1], 'Candidate replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, str(volume.PHASE), str(price.PARENT), str(price.DIRECTORY/'prepared.json'),
        str(volume.DIRECTORY/'prepared.json'), str(CHECK/'checked.json'), str(DIRECTORY/'source-reviews.json'),
        str(DIRECTORY/'source-AST.json'), 'scripts/accept_strategy_batch103.py', 'tests/unit/test_batch103_acceptance.py']
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': candidate, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {name: file_sha(name) for name in paths},
        'previous_code_sha256': file_sha(DIRECTORY/'strategy_catalog.py.before'),
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'catalog_id': candidate['catalog_id'], 'not_published': True}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists'); doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Prepared evidence changed')
    require(REPORT.is_file(), 'Final report missing')
    before = (DIRECTORY/'strategy_catalog.py.before').read_bytes(); proposed = (DIRECTORY/'strategy_catalog.proposed.py').read_bytes()
    require(proposed == catalog_program(before, selected) and file_sha(DIRECTORY/'strategy_catalog.proposed.py') == doc['proposed_code_sha256'],
            'Proposed code bytes changed')
    candidate = Path(doc['candidate']['output']); previous = read(Path(doc['previous_catalog']['directory'])/'catalog.json')
    candidate_checks(previous, read(candidate/'catalog.json'), selected)
    for name, sha in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == sha, 'Candidate bytes changed')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        source_code = Path('src/observe/strategy_catalog.py')
        require(read(latest) == doc['previous_catalog'] and source_code.read_bytes() == before and
            file_sha(source_code) == doc['previous_code_sha256'], 'Actual publication antecedent changed')
        # The live old graph is validated while holding the lock, before replacing its catalog module.
        require(components() == doc['components'], 'Bound components changed')
        proof = read(DIRECTORY/'protection.json'); base = read(DIRECTORY/'baseline.json')
        require(proof['status'] == 'ok' and proof['old_controls_unchanged'] == len(base['controls']) and
            proof['old_partition_metadata_unchanged'] == len(base['partitions']) and
            proof['baseline_sha256'] == file_sha(DIRECTORY/'baseline.json'), 'Actual old-asset protection incomplete')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name, sha in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == sha, 'Published bytes differ')
        old_latest, old_progress = latest.read_bytes(), progress_latest.read_bytes()
        try:
            source_code.write_bytes(proposed); REVIEWS.update(selected)
            require(file_sha(source_code) == doc['proposed_code_sha256'], 'Final catalog code differs')
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch103 source research accepted:ETF6/81 original rules,45 callbacks,48437 native/float heat and28078/17157 price references;trade inputs blocked')
            require((progress['manually_reviewed'], progress['locally_backtested_sources'], progress['locally_reproduced_sources'], progress['local_strategy_runs']) ==
                (292, 12, 12, 87), 'Formal counts differ')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), str(source_code): file_sha(source_code),
                str(DIRECTORY/'protection.json'): file_sha(DIRECTORY/'protection.json'), str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and read(progress_latest)['checkpoint'] == progress['checkpoint'] and
                all(file_sha(name) == sha for name, sha in bound.items()), 'Final binding/navigation differs')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress, 'components': doc['components'],
                'source_sha256': list(price.study.SOURCE_SHA), 'evidence_sha256': bound, 'protection': proof,
                'new_independent_sources': 2, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'review_method': 'Full originals;fresh local/copied48 tests and complete saved-value checks;two workers per reference;no independent agent review'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress)
            if source_code.read_bytes() == proposed: source_code.write_bytes(before)
            raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 292, 'remaining': 403}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'protect', 'publish'))
    args = parser.parse_args(); print(json.dumps(protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action](), ensure_ascii=False))
