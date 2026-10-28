"""Publish two reviewed sources after checking actual frozen arithmetic evidence."""
import argparse
import json
from pathlib import Path
import pprint
import shutil

import pandas as pd
import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch98 as stage
from scripts import probe_strategy_batch98 as probes
from scripts import diagnose_strategy_batch98 as boundary
from scripts import recover_strategy_batch98_diagnostics as recovery
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch98/20261011-formal-source-acceptance')
SPECS = ('docs/handoff/batch98-y2021-063-source-review.yaml',
         'docs/handoff/batch98-y2023-058-source-review.yaml')
REPORT = Path('docs/tasks/2026-10-11-第九十八批牛熊指标与北向RSRS-正式研究验收.md')
RECEIPT = Path('docs/handoff/2026-10-11-batch98-final-review.json')
GUARD = Path('data/staging/strategies-batch98/20261011-guard-validation-v2/checked.json')


def validation_commands():
    interpreter = sys_executable()
    return [[str(Path(interpreter).with_name('ruff')), 'check', 'src', 'tests',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
        'scripts/research_strategy_batch98.py', 'scripts/probe_strategy_batch98.py',
        'scripts/diagnose_strategy_batch98.py', 'scripts/accept_strategy_batch98.py',
        'scripts/verify_strategy_batch98.py', 'scripts/recover_strategy_batch98_diagnostics.py'],
        [interpreter, '-m', 'pytest', '-q', 'tests/unit/test_batch98_research.py',
         'tests/unit/test_batch98_acceptance.py', 'tests/unit/test_batch98_rsrs_boundary.py',
         'tests/unit/test_batch98_json_recovery.py', 'tests/unit/test_research_metadata.py'], ['git', 'diff', '--check']]


def reviews():
    result = {}
    for i, spec in enumerate(SPECS):
        row = yaml.safe_load(Path(spec).read_text())
        require(row['source'] == stage.SOURCE_PATHS[i] and row['source_sha256'] == stage.SOURCE_SHA[i]
                and row['original_strategy_complete'] is False and row['economic_variant_approved'] is False,
                'Review spec identity/scope differs')
        result[row['source']] = {'review_status': '人工审查完成', 'scope': row['review_scope'],
            'status': '待数据', 'implementations': [], 'original_strategy_complete': False,
            'fidelity': row['allowed_evidence'], 'rules': row['rules'],
            'gaps': row['data_gaps']+row['defects'], 'next_step': row['next']}
    return result


def candidate_checks(old, current, selected):
    require(len(old) == len(current) == 695, 'Source count differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == 695 and len({r['path'] for r in current}) == 695 and
            {r['path'] for r in current} == set(previous), 'Source identities differ')
    require({r['path'] for r in current if r != previous[r['path']]} == set(selected),
            'Unrelated or missing catalog changes')
    for row in current:
        if row['path'] not in selected: continue
        before = previous[row['path']]; allowed = selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
                {k:v for k, v in row.items() if k not in allowed} ==
                {k:v for k, v in before.items() if k not in allowed}, 'Static source identity changed')
        require(all(row[k] == v for k, v in allowed.items()), 'Review rules differ')
    require(sum(r['review_status'] == '人工审查完成' for r in current) == 282,
            'Reviewed count differs')
    return {'sources': 695, 'changed': sorted(selected), 'reviewed': 282, 'remaining': 413}


def catalog_program(before, selected):
    marker = b'REVIEWS = {\n'; require(before.count(marker) == 1, 'Insertion marker differs')
    entries = pprint.pformat(selected, sort_dicts=False, width=100)[1:-1]
    return before.replace(marker, marker+(entries+',\n').encode(), 1)


def diagnostic_checks(doc):
    rows = {r['case']: r for r in doc['cases']}
    require(len(rows) == len(doc['cases']) == doc['case_count'] == 27, 'Diagnostic scope differs')
    require(rows['threshold_strict']['value'] == [0, 0, 1] and
            rows['interval_inclusive']['value'] == [0, 1, 1, 0] and
            rows['hold_signal_resets_count']['value'] == [1, 1, 1, 1, 1, 0], 'Threshold hand checks differ')
    require(rows['quantile17_always_nan']['value'] == {'nonfinite': 'nan'} and
            rows['first_RSRS_zero_no_stock_history']['value'] == 0. and
            rows['second_RSRS_nan_shared_state']['value'] == {
                'score': {'nonfinite': 'nan'}, 'shared_length': 4, 'history_calls': ['B']},
            'Original RSRS state differs')
    require(rows['same_target_set_blocks_RSRS_exit']['value'] == [None, []] and
            len(rows['rank_excludes_top3_selects17']['value']) == 17 and
            rows['stat_std_field_is_median']['value']['stdRet'] == 2., 'Ranking/exit/stat defects differ')
    require(rows['turnover3001_duplicate_boundary']['value']['duplicated'] == 2,
            'Pagination duplicate evidence differs')
    expected = {'kernel_empty_periods': 'ValueError', 'single_signal_summary_name_error': 'NameError',
        'quantile29_current_integer_index_error': 'KeyError', 'date_namespace_unbound': 'NameError',
        'boll_empty_error': 'IndexError', 'nan_passes_filter_then_missing_fund_keyerror': 'KeyError'}
    require({n:r['error'] for n, r in rows.items() if r['error']} == expected,
            'Missing or unexpected diagnostic exception')
    for method in ('MA', 'BBANDS'):
        first = rows['inversion_'+method+'_close_kernel']['value']
        other = rows['inversion_'+method+'_kernel_close']['value']
        require(first['close_MARK'] == first['kernel_MARK'] and
                other['close_MARK'] != other['kernel_MARK'], 'Inversion fixture lacks crossing')
    require(rows['full_asset_per_new_stock_incremental_intents']['value'] == [None,
        [['buy', 'A', 500.], ['buy', 'B', 500.], ['redeem', '420002.OF', 100.], ['sell', '420002.OF', 0]]],
        'Original incremental cash intents differ')
    return {'cases': 27, 'synthetic_only': True, 'hand_cases': 'match', 'orders_or_fills': 0}


def components():
    require(file_sha(stage.PARENT) == stage.PARENT_SHA, 'Parent changed')
    prepared = read(stage.DIRECTORY/'prepared.json')
    for n, digest in prepared['implementation_sha256'].items():
        require(file_sha(stage.DIRECTORY/'implementation'/n) == digest, 'Frozen stage changed')
        if n != 'src/observe/strategy_catalog.py': require(file_sha(n) == digest, 'Helper changed')
    require(prepared['parent_sha256'] == stage.PARENT_SHA and
            file_sha(stage.DIRECTORY/'index.parquet') == prepared['operand_sha256'], 'Operand/parent differs')
    frame, index = stage.index_operand('data')
    require(index == prepared['index'] and file_sha('data/snapshots/'+stage.SNAPSHOT+'.json') == stage.SNAPSHOT_SHA,
            'Current input bytes/logical content/calendar differ')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(stage.DIRECTORY/'index.parquet'), check_exact=True)
    for number, name in enumerate(stage.SOURCE_PATHS):
        require(file_sha(Path('repo/量化策略源代码')/name) ==
                file_sha(stage.DIRECTORY/'sources'/f'{number}.source') == stage.SOURCE_SHA[number], 'Source changed')
    require(stage.inventory() == read(stage.DIRECTORY/'dependencies.json'), 'Dependencies differ')
    work = recovery.DIRECTORY
    repaired = read(work/'prepared.json')
    require(repaired['recovery_antecedent']['original_prepared_sha256'] == file_sha(stage.DIRECTORY/'prepared.json')
            and repaired['source_inventory'] == prepared['source_inventory'] and repaired['index'] == prepared['index']
            and file_sha(work/'index.parquet') == prepared['operand_sha256'], 'Recovery input/scope differs')
    for name in recovery.FILES:
        require(file_sha(name) == file_sha(work/'implementation'/name) == repaired['implementation_sha256'][name],
                'Recovery implementation differs')
    representation = read(work/'representation-verification.json')
    require(representation['status'] == 'ok' and representation['representation_changes'] == 1
            and representation['nonranking_diagnostics_unchanged'] == 26 and representation['ranking_elements'] == 17
            and representation['economic_rules_changed'] is False, 'Recovery representation proof differs')
    for name, digest in representation['original_and_recovery_sha256'].items():
        require(file_sha(name) == digest, 'Representation artifact changed')
    offline = read(work/'offline-verification.json')
    require(offline['status'] == 'match' and offline['differences'] == 0 and offline['files'] == 3 and
            offline['fresh_process'] and offline['network_blocked'] and
            offline['prepared_sha256'] == file_sha(work/'prepared.json') and
            set(offline['outputs_sha256']) == set(stage.OUTPUTS), 'Offline proof differs')
    for mode in ('study', 'replay'):
        folder = work/(mode+'-command'); command = read(folder/'command.json')
        require(command['command'] == [sys_executable(), '-B', '-m', 'scripts.recover_strategy_batch98_diagnostics',
            'worker', '--mode', mode] and command['returncode'] == 0 and
            file_sha(folder/'stdout.log') == command['stdout_sha256'] and
            file_sha(folder/'stderr.log') == command['stderr_sha256'], 'Actual worker command differs')
        out = json.loads((folder/'stdout.log').read_text().splitlines()[-1])
        require(out['outputs_sha256'] == offline['outputs_sha256'] and out['status'] == 'ok', 'Worker output differs')
        for name, digest in offline['outputs_sha256'].items():
            require(file_sha(work/mode/name) == digest, 'Worker artifact changed')
    data = pd.read_parquet(work/'study/index-components.parquet')
    pd.testing.assert_frame_equal(data, pd.read_parquet(work/'replay/index-components.parquet'), check_exact=True)
    summary = read(work/'study/summary.json')
    require(len(data) == summary['rows'] == 5280 and summary['OLS_windows'] == 5263 and
            all(r['marks_differences'] == 0 for r in summary['comparison_price_leg'].values()) and
            all(v < 1e-10 for v in summary['OLS_max_absolute_errors'].values()) and
            summary['new_trading_backtests'] == 0, 'Long reference scope/arithmetic differs')
    diag = diagnostic_checks(read(work/'study/diagnostics.json'))
    guard = read(GUARD)
    require(guard['status'] == 'ok' and guard['isolated_passed'] == 13 and guard['isolated_skipped'] == 0
            and [c['command'] for c in guard['commands']] == validation_commands()
            and all(c['returncode'] == 0 for c in guard['commands']), 'Validation guard differs')
    for name, digest in guard['evidence_sha256'].items(): require(file_sha(name) == digest, 'Guard evidence changed')
    for name, digest in guard['implementation_sha256'].items():
        require(file_sha(name) == digest, 'Guard implementation changed')
    isolation = read(GUARD.parent/'isolated.json')
    require(isolation['tests_passed'] == 13 and isolation['skips'] == 0 and
            len(isolation['copied_sha256']) == guard['copied_files'] and
            len(isolation['actual_imports']) == 5 and
            all(Path(p).is_relative_to(Path(isolation['cwd'])) for p in isolation['actual_imports'].values()),
            'Isolated scope/import origins differ')
    for name, digest in isolation['copied_sha256'].items():
        require(file_sha(Path(isolation['cwd'])/name) == digest, 'Isolated copy changed')
    require(read(probes.DIRECTORY/'summary.json')['requests'] == 3, 'Probe scope differs')
    extra = read(boundary.DIRECTORY/'offline.json')
    require(extra['status'] == 'match' and extra['cases'] == 2 and extra['differences'] == 0
            and extra['network_blocked'] and extra['fresh_process'] and
            extra['implementation_sha256'] == file_sha('scripts/diagnose_strategy_batch98.py') and
            extra['source_sha256'] == stage.SOURCE_SHA[1] and
            file_sha(boundary.DIRECTORY/'study.json') == file_sha(boundary.DIRECTORY/'replay.json') == extra['output_sha256'],
            'Perfect-fit exception evidence differs')
    require([r['exclude_below_minus_point4'] for r in read(boundary.DIRECTORY/'study.json')['cases']] == [False, True],
            'Perfect-fit finite filter boundary differs')
    for mode in ('study', 'replay'):
        command = read(boundary.DIRECTORY/(mode+'-command.json'))
        require(command['returncode'] == 0 and command['command'] == [sys_executable(), '-B', '-m',
                'scripts.diagnose_strategy_batch98', 'worker', '--mode', mode] and
                file_sha(boundary.DIRECTORY/(mode+'.log')) == command['stdout_sha256'] and
                file_sha(boundary.DIRECTORY/(mode+'.stderr')) == command['stderr_sha256'], 'Boundary command differs')
    return {'index_rows': 5280, 'OLS_windows': 5263, 'comparison_price_leg': summary['comparison_price_leg'],
            'diagnostics': diag, 'perfect_fit_boundary_cases': 2, 'offline': 'match/0', 'new_trading_backtests': 0}


def sys_executable():
    import sys
    return sys.executable


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); parent = read(stage.PARENT)['progress']
    old = read(Path('data/catalog/strategies')/parent['catalog_id']/'catalog.json')
    active = read('data/catalog/strategies/latest.json')
    require(active['catalog_id'] == parent['catalog_id'] and active['sha256'] == parent['catalog_sha256'],
            'Active catalog identity differs')
    before = (stage.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == before, 'Catalog changed before preparation')
    DIRECTORY.mkdir(parents=True, exist_ok=False); save(DIRECTORY/'source-reviews.json', selected)
    REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag
        target = root/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
        shutil.copyfile(stage.DIRECTORY/'implementation-evidence.before.json', target)
        candidate = catalog_strategies(root, 'repo/量化策略源代码')
        output = Path(candidate['output']); candidate_checks(old, read(output/'catalog.json'), selected)
        digests.append({n: file_sha(output/n) for n in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': proposed = candidate
    require(digests[0] == digests[1], 'Candidate byte replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, 'scripts/accept_strategy_batch98.py', 'tests/unit/test_batch98_acceptance.py',
        str(GUARD), str(stage.DIRECTORY/'prepared.json'), str(stage.DIRECTORY/'dependencies.json'),
        str(recovery.DIRECTORY/'offline-verification.json'), str(recovery.DIRECTORY/'prepared.json'),
        str(recovery.DIRECTORY/'representation-verification.json'), str(probes.DIRECTORY/'summary.json'),
        str(probes.DIRECTORY/'prepared.json'), str(boundary.DIRECTORY/'offline.json'),
        str(boundary.DIRECTORY/'study-command.json'), str(boundary.DIRECTORY/'replay-command.json'),
        str(DIRECTORY/'source-reviews.json')]
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': proposed, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {n: file_sha(n) for n in paths},
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'not_published': True, 'catalog': proposed['catalog_id']}


def publish():
    require(not RECEIPT.exists(), 'Formal receipt exists')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, digest in doc['evidence_sha256'].items(): require(file_sha(name) == digest, 'Prepared proof changed')
    require(components() == doc['components'] and REPORT.is_file(), 'Formal evidence/report differs')
    before = (stage.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == catalog_program(before, selected)
            and file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256']
            and all(REVIEWS.get(n) == r for n, r in selected.items()), 'Proposed implementation differs')
    old = read(Path(doc['previous_catalog']['directory'])/'catalog.json')
    candidate = Path(doc['candidate']['output']); candidate_checks(old, read(candidate/'catalog.json'), selected)
    for name, digest in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == digest, 'Candidate differs')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        require(read(latest) == doc['previous_catalog'], 'Catalog navigation changed')
        old_latest = latest.read_bytes(); old_progress = progress_latest.read_bytes()
        protection = protect('data', stage.DIRECTORY)
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']
        require(not dest.exists(), 'Published destination exists'); dest.mkdir()
        for name, digest in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == digest, 'Frozen publication changed')
        try:
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch98 bull/bear63+northRSRS58 source research accepted;5280index days/5263OLS/27+2diagnostics/offline match0;original trades blocked')
            require((progress['manually_reviewed'], progress['locally_reproduced_sources'], progress['local_strategy_runs']) == (282, 12, 87), 'Counts differ')
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and
                    read(progress_latest)['checkpoint'] == progress['checkpoint'], 'Final navigation differs')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), 'src/observe/strategy_catalog.py': file_sha('src/observe/strategy_catalog.py'),
                str(stage.DIRECTORY/'protection.json'): file_sha(stage.DIRECTORY/'protection.json'),
                str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            for name, digest in bound.items(): require(file_sha(name) == digest, 'Final readback failed')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress,
                'components': doc['components'], 'source_sha256': list(stage.SOURCE_SHA), 'evidence_sha256': bound,
                'protection': protection, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'review_method': 'Full-source review, independent scalar arithmetic, fresh offline workers and local/isolated tests; no independent agent review claimed'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress); raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 282, 'remaining': 413}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['prepare', 'publish'])
    args = parser.parse_args(); print(json.dumps({'prepare': prepare, 'publish': publish}[args.action](), ensure_ascii=False))
