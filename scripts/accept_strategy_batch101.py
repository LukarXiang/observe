"""Accept original source research and longest references without admitting missing trades."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts import reference_strategy_batch101 as study
from scripts import verify_strategy_batch101_reference as verification
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch101/20261011-formal-source-acceptance')
RECEIPT = Path('docs/handoff/2026-10-11-batch101-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零一批全天候ES与期货贴水-正式研究验收.md')
SPECS = ('docs/handoff/batch101-y2024a-072-source-review.yaml', 'docs/handoff/batch101-y2024b-036-source-review.yaml')


def reviews():
    selected = {}
    for number, name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == study.original.SOURCES[number] and doc['source_sha256'] == study.original.SOURCE_SHA[number] and
                doc['formal_acceptance'] is False and doc['original_strategy_complete'] is False and
                doc['economic_variant_approved'] is False, 'Frozen source rule identity differs')
        selected[doc['source']] = {'review_status': '人工审查完成', 'status': '待数据',
            'scope': doc['review_scope'], 'rules': doc['rules'], 'fidelity': doc['allowed_evidence'],
            'gaps': doc['data_gaps']+doc['defects'], 'next_step': doc['next'], 'implementations': [],
            'original_strategy_complete': False}
    return selected


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and len(selected) == 2, 'Catalog scope differs')
    previous = {row['path']: row for row in old}
    require(len(previous) == len({row['path'] for row in new}) == 695 and set(previous) == {row['path'] for row in new}, 'Catalog identities differ')
    require({row['path'] for row in new if row != previous[row['path']]} == set(selected), 'Unrelated/missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
                {key: value for key, value in row.items() if key not in allowed} ==
                {key: value for key, value in before.items() if key not in allowed}, 'Static identity changed')
        require(all(row[key] == value for key, value in allowed.items()), 'Literal rules differ')
    require(sum(row['review_status'] == '人工审查完成' for row in new) == 288, 'Reviewed count differs')
    return {'sources': 695, 'reviewed': 288, 'remaining': 407, 'changed': sorted(selected)}


def components():
    study.validate(); proof = read(study.DIRECTORY/'offline-verification.json')
    require(proof['status'] == 'match' and proof['differences'] == 0 and proof['network_blocked'] is True and
            proof['reference_rows_per_mode'] == 30924 and proof['native_windows_per_policy'] == 15462,
            'Longest fresh reference replay incomplete')
    for mode in ('study', 'replay'):
        completed = read(study.DIRECTORY/mode/'completed.json')
        command = read(study.DIRECTORY/mode/'command.json')
        require(completed['output_sha256'] == proof['output_sha256'] and completed['mode'] == mode and
                completed['prepared_sha256'] == file_sha(study.DIRECTORY/'prepared.json') and command['returncode'] == 0 and
                command['timeout1800'] is False, 'Actual terminal worker differs')
        for name, digest in proof['output_sha256'].items(): require(file_sha(study.DIRECTORY/mode/name) == digest, 'Reference output changed')
    frame = pd.read_parquet(study.DIRECTORY/'study/ES-reference.parquet')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(study.DIRECTORY/'replay/ES-reference.parquet'), check_exact=True)
    require(read(study.DIRECTORY/'study/diagnostics.json') ==
            read('data/staging/strategies-batch101/20261011-source-preread-check/diagnostics.json'), 'Original diagnostics drifted')
    guard = read(verification.DIRECTORY/'checked.json')
    require(guard['status'] == 'ok_reference_phase' and guard['local_passed'] == guard['isolated_passed'] == 50 and
            guard['skips'] == 0 and guard['full_readback']['all_saved_ES_values_checked'] == 30924 and
            all(command['returncode'] == 0 for command in guard['commands']), 'Complete validation scope differs')
    for name, digest in {**guard['implementation_sha256'], **guard['evidence_sha256']}.items():
        require(file_sha(name) == digest, 'Validated bytes changed: '+name)
    isolation = read(verification.DIRECTORY/'isolated/isolation.json'); root = Path(isolation['cwd'])
    require(len(isolation['actual_imports']) == 6 and all(Path(p).is_relative_to(root) for p in isolation['actual_imports'].values()) and
            isolation['copied_full_readback'] == guard['full_readback'], 'Copied readback/import scope differs')
    for name, digest in isolation['copied_sha256'].items(): require(file_sha(root/name) == digest, 'Copied evidence changed')
    base = read(study.inputs.DIRECTORY/'dependencies.json')
    require(Store('data').published()['batch_id'] == base['published_batch'] and
            base['funds_without_candidate_in_selected_registries'] == ['510170.XSHG', '511260.XSHG'] and
            base['bars_1m_registered'] is False and base['futures_tables_registered'] == [], 'Actual dependency scope differs')
    return {'reference_rows': 30924, 'native_windows_per_policy': 15462, 'original_diagnostics': 32,
        'offline': 'match/0', 'local_tests': 50, 'isolated_tests': 50, 'saved_ES_readback': guard['full_readback'],
        'new_trading_backtests': 0, 'original_strategy_complete': False}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); active = read('data/catalog/strategies/latest.json')
    parent = read(study.inputs.PARENT)
    require(active['catalog_id'] == parent['progress']['catalog_id'], 'Accepted parent catalog differs')
    old = read(Path(active['directory'])/'catalog.json'); before = Path('src/observe/strategy_catalog.py').read_bytes()
    require(file_sha('src/observe/strategy_catalog.py') == parent['evidence_sha256']['src/observe/strategy_catalog.py'], 'Parent source catalog changed')
    DIRECTORY.mkdir(parents=True); (DIRECTORY/'strategy_catalog.py.before').write_bytes(before)
    shutil.copyfile(study.inputs.DIRECTORY/'baseline.json', DIRECTORY/'baseline.json')
    save(DIRECTORY/'source-reviews.json', selected); REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(study.inputs.DIRECTORY/'implementation-evidence.before.json', target)
        proposed = catalog_strategies(root, 'repo/量化策略源代码'); folder = Path(proposed['output'])
        candidate_checks(old, read(folder/'catalog.json'), selected)
        digests.append({name: file_sha(folder/name) for name in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': candidate = proposed
    require(digests[0] == digests[1], 'Candidate catalog byte replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, str(study.inputs.PARENT), str(study.inputs.DIRECTORY/'prepared.json'),
        str(study.DIRECTORY/'prepared.json'), str(study.DIRECTORY/'offline-verification.json'),
        str(verification.DIRECTORY/'checked.json'), 'scripts/accept_strategy_batch101.py',
        'tests/unit/test_batch101_acceptance.py', str(DIRECTORY/'source-reviews.json'),
        'data/staging/strategies-batch101/20261011-reference-policy-boundaries/checked.json',
        'data/staging/strategies-batch101/20261011-extended-registry-search/checked.json',
        'docs/handoff/2026-10-11-batch101-preread-stage.json']
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': candidate, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {name: file_sha(name) for name in paths},
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'catalog_id': candidate['catalog_id'], 'not_published': True}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, digest in doc['evidence_sha256'].items(): require(file_sha(name) == digest, 'Prepared evidence changed')
    require(components() == doc['components'] and REPORT.is_file(), 'Formal report/result differs')
    before = (DIRECTORY/'strategy_catalog.py.before').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == catalog_program(before, selected) and
            file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
            all(REVIEWS.get(name) == value for name, value in selected.items()), 'Runtime proposed rules differ')
    candidate = Path(doc['candidate']['output'])
    candidate_checks(read(Path(doc['previous_catalog']['directory'])/'catalog.json'), read(candidate/'catalog.json'), selected)
    for name, digest in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == digest, 'Candidate bytes differ')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        require(read(latest) == doc['previous_catalog'], 'Actual catalog pointer changed')
        require(Store('data').published()['batch_id'] == read(study.inputs.DIRECTORY/'dependencies.json')['published_batch'], 'Actual data root changed')
        proof = read(DIRECTORY/'protection.json'); baseline = read(DIRECTORY/'baseline.json')
        require(proof['status'] == 'ok' and proof['old_controls_unchanged'] == len(baseline['controls']) and
                proof['old_partition_metadata_unchanged'] == len(baseline['partitions']), 'Old-asset protection incomplete')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name, digest in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == digest, 'Published bytes differ')
        old_latest, old_progress = latest.read_bytes(), progress_latest.read_bytes()
        try:
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch101 source research accepted:15462 native120 ES windows/two explicit references/32 original diagnostics;seven-fund and actual futures trading blocked')
            require((progress['manually_reviewed'], progress['locally_backtested_sources'],
                progress['locally_reproduced_sources'], progress['local_strategy_runs']) == (288, 12, 12, 87), 'Formal counts differ')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), 'src/observe/strategy_catalog.py': file_sha('src/observe/strategy_catalog.py'),
                str(DIRECTORY/'protection.json'): file_sha(DIRECTORY/'protection.json'), str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and
                    read(progress_latest)['checkpoint'] == progress['checkpoint'] and
                    all(file_sha(name) == digest for name, digest in bound.items()), 'Final identity/navigation differs')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress,
                'components': doc['components'], 'source_sha256': list(study.original.SOURCE_SHA),
                'evidence_sha256': bound, 'protection': proof, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'review_method': 'Full original rules;two fresh network-blocked reference workers;all saved ES readback local/copied;no independent agent review'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress); raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 288, 'remaining': 407}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'protect', 'publish'))
    args = parser.parse_args()
    print(json.dumps(protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action](), ensure_ascii=False))
