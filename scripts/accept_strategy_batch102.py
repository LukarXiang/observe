"""Accept two source studies with exact duplicate mapping and no fictitious trades."""
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
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts import reference_strategy_batch102 as study
from scripts import verify_strategy_batch102_stage as stage_check
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch102/20261011-formal-source-acceptance')
CHECK = Path('data/staging/strategies-batch102/20261011-formal-validation')
RECEIPT = Path('docs/handoff/2026-10-11-batch102-final-review.json')
PHASE = Path('docs/handoff/2026-10-11-batch102-reference-stage.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零二批宏观周期与ETF调试版-正式研究验收.md')
SPECS = ('docs/handoff/batch102-y2024b-048-source-review.yaml', 'docs/handoff/batch102-y2025-058-source-review.yaml')


def reviews():
    selected = {}
    for n, name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == study.original.SOURCES[n] and doc['source_sha256'] == study.original.SOURCE_SHA[n] and
            doc['formal_acceptance'] is False and doc['original_strategy_complete'] is False and
            doc['economic_variant_approved'] is False, 'Original rule identity differs')
        selected[doc['source']] = {'review_status': '人工审查完成', 'status': '重复版本' if n == 0 else '待数据',
            'scope': doc['review_scope'], 'rules': doc['rules'], 'fidelity': doc['allowed_evidence'],
            'gaps': doc['data_gaps']+doc['defects'], 'next_step': doc['next'], 'implementations': [], 'original_strategy_complete': False}
    return selected


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and set(selected) == set(study.original.SOURCES), 'Catalog scope differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and set(previous) == {r['path'] for r in new}, 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated/missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
            {k: v for k, v in row.items() if k not in allowed} == {k: v for k, v in before.items() if k not in allowed}, 'Static identity changed')
        require(all(row[k] == v for k, v in allowed.items()) and row['original_strategy_complete'] is False and
            row['implementations'] == [], 'Reviewed rules/trading claim differ')
        if row['path'] == study.original.SOURCES[0]:
            require(row['status'] == before['status'] == '重复版本' and row['duplicate_of'] == before['duplicate_of'] == 'S-d6db552a3293', 'Macro duplicate identity lost')
    require(sum(r['review_status'] == '人工审查完成' for r in old) == 288 and
        sum(r['review_status'] == '人工审查完成' for r in new) == 290, 'Manual count differs')
    return {'sources': 695, 'reviewed': 290, 'remaining': 405, 'changed': sorted(selected), 'new_independent_sources': 1}


def components():
    study.validate(); phase = read(PHASE)
    require(phase['status'] == 'ok_reference_stage' and not phase['formal_source_research_acceptance'] and
        not phase['formal_trading_acceptance'], 'Historical stage scope differs')
    for name, sha in phase['evidence_sha256'].items(): require(file_sha(name) == sha, 'Historical phase changed: '+name)
    proof = read(study.DIRECTORY/'offline-verification.json')
    require(proof['status'] == 'ok' and proof['differences'] == [] and proof['fresh_network_blocked_processes'] == 2 and
        all(c['returncode'] == 0 and file_sha(c['log']) == c['log_sha256'] for c in proof['commands']), 'Fresh worker terminal state differs')
    for name, sha in proof['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(Path(name.replace('/compute/', '/replay/'))), 'Replay output differs')
    old_guard = read(stage_check.DIRECTORY/'checked.json')
    require(old_guard['status'] == 'ok_stage_only' and old_guard['tests'] == {'local': 20, 'isolated': 20} and
        old_guard['readback']['all_saved_cells_checked'], 'Historical full readback missing')
    for name, sha in old_guard['copied_file_sha256'].items(): require(file_sha(stage_check.DIRECTORY/'isolated'/name) == sha, 'Historical copied byte changed')
    fresh = read(CHECK/'checked.json')
    require(fresh['status'] == 'ok_formal_ready' and fresh['tests'] == {'local': 34, 'isolated': 34} and
        fresh['supplemental_diagnostics'] == 11 and fresh['saved_readback'] == old_guard['readback'] and
        fresh['diagnostic_replay_differences'] == 0, 'Fresh formal validation incomplete')
    require(len(fresh['actual_imports']) == 6 and
        all(Path(name).is_relative_to((CHECK/'isolated').resolve()) for name in fresh['actual_imports'].values()), 'Actual copied import scope differs')
    for tag in ('local', 'isolated', 'diagnostic'):
        command = read(CHECK/(tag+'-command.json'))
        log = CHECK/('isolated.log' if tag == 'isolated' else 'local-tests.log' if tag == 'local' else 'diagnostic-worker.log')
        require(command['returncode'] == 0 and file_sha(log) == command['log_sha256'], 'Formal command did not succeed')
    for name, sha in fresh['evidence_sha256'].items(): require(file_sha(name) == sha, 'Formal validation evidence changed: '+name)
    for name, sha in fresh['copied_file_sha256'].items(): require(file_sha(CHECK/'isolated'/name) == sha, 'Fresh copied byte changed')
    base = read(study.DIRECTORY/'dependencies.json')
    require(base['published_batch'] == Store('data').published()['batch_id'] and len(base['ETF_candidates_in_selected_registries']) == 4 and
        len(base['ETF_missing_in_selected_registries']) == 13, 'Actual inventory scope changed')
    summary = read(study.DIRECTORY/'compute/summary.json')
    require((summary['PMI_windows'], summary['index_complete120_EOD_rows'], summary['annual109_month_windows'], summary['ETF20_row_windows']) ==
        (213, 5161, 153, 11095) and summary['new_trading_backtests'] == 0 and summary['diagnostics'] == 33, 'Reference scope differs')
    return {'longest_references': summary, 'original_diagnostics': 33, 'supplemental_diagnostics': 11,
        'local_tests': 34, 'isolated_tests': 34, 'saved_cells_per_readback': phase['validation']['saved_cells_per_readback'],
        'offline': 'match/0', 'new_trading_backtests': 0, 'original_strategy_complete': False}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); parent = read(study.PARENT); active = read('data/catalog/strategies/latest.json')
    require(active['catalog_id'] == parent['progress']['catalog_id'] and
        file_sha('src/observe/strategy_catalog.py') == parent['evidence_sha256']['src/observe/strategy_catalog.py'], 'Formal parent catalog changed')
    old = read(Path(active['directory'])/'catalog.json'); before = Path('src/observe/strategy_catalog.py').read_bytes()
    DIRECTORY.mkdir(parents=True); (DIRECTORY/'strategy_catalog.py.before').write_bytes(before)
    shutil.copyfile(study.DIRECTORY/'baseline.json', DIRECTORY/'baseline.json')
    save(DIRECTORY/'source-reviews.json', selected)
    save(DIRECTORY/'source-AST.json', {str(n): {node.name: ast.dump(node, include_attributes=False) for node in study.original.tree(n).body
        if isinstance(node, ast.FunctionDef)} for n in (0, 1)})
    REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(study.DIRECTORY/'implementation-evidence.before.json', target)
        proposed = catalog_strategies(root, 'repo/量化策略源代码'); folder = Path(proposed['output'])
        candidate_checks(old, read(folder/'catalog.json'), selected)
        digests.append({name: file_sha(folder/name) for name in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': candidate = proposed
    require(digests[0] == digests[1], 'Candidate byte replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, str(PHASE), str(study.PARENT), str(study.DIRECTORY/'prepared.json'), str(CHECK/'checked.json'),
        str(DIRECTORY/'source-reviews.json'), str(DIRECTORY/'source-AST.json'), 'scripts/accept_strategy_batch102.py',
        'tests/unit/test_batch102_acceptance.py']
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': candidate, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {name: file_sha(name) for name in paths},
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'catalog_id': candidate['catalog_id'], 'not_published': True}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists'); doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Prepared evidence changed')
    require(components() == doc['components'] and REPORT.is_file(), 'Formal report/result differs')
    before = (DIRECTORY/'strategy_catalog.py.before').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == catalog_program(before, selected) and
        file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
        all(REVIEWS.get(name) == value for name, value in selected.items()), 'Proposed runtime rules differ')
    candidate = Path(doc['candidate']['output']); candidate_checks(read(Path(doc['previous_catalog']['directory'])/'catalog.json'), read(candidate/'catalog.json'), selected)
    for name, sha in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == sha, 'Candidate byte changed')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        require(read(latest) == doc['previous_catalog'], 'Catalog pointer changed')
        proof = read(DIRECTORY/'protection.json'); base = read(DIRECTORY/'baseline.json')
        require(proof['status'] == 'ok' and proof['old_controls_unchanged'] == len(base['controls']) and
            proof['old_partition_metadata_unchanged'] == len(base['partitions']), 'Old-asset protection incomplete')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name, sha in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == sha, 'Published byte differs')
        old_latest, old_progress = latest.read_bytes(), progress_latest.read_bytes()
        try:
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch102 source research accepted:macro48 exact duplicate/ETF58 rules;44 diagnostics/longest numerical references match0;original trades blocked')
            require((progress['manually_reviewed'], progress['locally_backtested_sources'], progress['locally_reproduced_sources'], progress['local_strategy_runs']) ==
                (290, 12, 12, 87), 'Formal counts differ')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), 'src/observe/strategy_catalog.py': file_sha('src/observe/strategy_catalog.py'),
                str(DIRECTORY/'protection.json'): file_sha(DIRECTORY/'protection.json'), str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and read(progress_latest)['checkpoint'] == progress['checkpoint'] and
                all(file_sha(name) == sha for name, sha in bound.items()), 'Final binding/navigation differs')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress, 'components': doc['components'],
                'source_sha256': list(study.original.SOURCE_SHA), 'evidence_sha256': bound, 'protection': proof,
                'new_independent_sources': 1, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'review_method': 'Full source rules;fresh local/copied34 tests and all saved-value checks;two longest reference workers and supplemental callback replay;no independent agent review'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress); raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 290, 'remaining': 405}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'protect', 'publish'))
    args = parser.parse_args(); print(json.dumps(protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action](), ensure_ascii=False))
