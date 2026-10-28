"""Accept the original trend notebook's source and paper-state research only."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import review_strategy_batch92 as stage
from scripts import verify_strategy_batch91 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch92/20261010-trend-definition-acceptance')
STAGE_CHECKPOINT = stage.DIRECTORY/'stage-checkpoint.json'
STAGE_CHECKPOINT_SHA = 'db3e192793add15bd16785740283ac2c8b724c27c9ee3e086dade9d63f5dcfdb'
ORIGINAL_STAGE_BINDING = stage.binding
SOURCE_NAME = stage.SOURCE.relative_to('repo/量化策略源代码').as_posix()
RECEIPT = Path('docs/handoff/2026-10-10-batch92-verification.json')
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch92.py',
    'tests/unit/test_batch92_acceptance.py', *stage.FILES)
NEW_CASES = ('test_frozen_stage_scope_and_nontrading_identity',
    'test_changed_stage_evidence_is_rejected[index-input.parquet]',
    'test_changed_stage_evidence_is_rejected[independent-full-signal-audit.json]',
    'test_changed_stage_evidence_is_rejected[demo-coverage-audit.json]',
    'test_changed_stage_scope_is_rejected', 'test_adapter_rejects_unexpected_directory',
    'test_adapter_preserves_upstream_evidence[baseline.json]',
    'test_adapter_preserves_upstream_evidence[input-binding.json]',
    'test_catalog_review_preserves_original_rules_without_trades',
    'test_full_junit_rejects_subset_duplicates_and_skips',
    'test_selection_environment_is_removed', 'test_unrelated_baseline_changes_are_rejected',
    'test_repeated_catalog_is_rejected_before_writes', 'test_original_binding_capture_prevents_recursion',
    'test_required_command_contract_rejects_empty_or_substituted', 'test_stage_control_mutation_before_acceptance_is_rejected',
    'test_new_run_inventory_is_rejected_before_acceptance_publication', 'test_isolation_candidate_must_match_tested_manifest')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'isolation-baseline.json', 'reference-replay.json', 'source-reviews/review.json',
    'offline-catalog.json', 'offline-verification.json', 'isolated-verification.json', 'stage-protection-baseline.json'}


def stage_evidence():
    require(file_sha(STAGE_CHECKPOINT) == STAGE_CHECKPOINT_SHA, 'Stage checkpoint changed')
    doc = read(STAGE_CHECKPOINT)
    require(doc['status'] == 'component_outputs_and_unit_checks_archived' and
        not doc['formal_stage_acceptance'] and not doc['formal_source_acceptance'] and
        doc['source_sha256'] == stage.SOURCE_SHA and doc['new_requests'] == doc['new_trading_backtests'] == 0 and
        doc['unit_tests_passed'] == 13 and doc['offline']['rows'] == 7181 and
        doc['offline']['status'] == 'match' and doc['offline']['differences'] == 0 and
        not doc['net_asset_value_computed'], 'Stage scope changed')
    require(all(file_sha(n) == sha for n, sha in doc['implementation_sha256'].items()), 'Stage implementation changed')
    require(all(file_sha(stage.DIRECTORY/n) == sha for n, sha in doc['evidence_sha256'].items()), 'Stage evidence changed')
    source = stage.tree(stage.DIRECTORY)
    review = read(stage.DIRECTORY/'source-definition-review.json')
    require(review['definitions'] == stage.definitions(source) and review['complete_lines_read'] == 807 and
        not review['top_level_executed'] and review['new_trading_backtests'] == 0, 'Original definition scope changed')
    return doc


def binding(directory=None):
    stage_evidence()
    result = {'stage_checkpoint_file': str(STAGE_CHECKPOINT), 'stage_checkpoint_sha256': STAGE_CHECKPOINT_SHA,
        'snapshot': stage.SNAPSHOT, 'source_sha256': stage.SOURCE_SHA, 'not_a_backtest': True,
        'new_requests': 0, 'new_trading_backtests': 0}
    if directory is not None: require(read(directory/'input-binding.json') == result, 'Acceptance input binding changed')
    return result


def preflight(root, directory):
    bound = binding(); ORIGINAL_STAGE_BINDING(stage.DIRECTORY)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    index = read('data/metadata/index.json'); baseline = read(index['implementation_files'])
    require(all(file_sha(r['path']) == r['sha256'] for r in baseline), 'Isolation baseline code changed')
    checkpoint(root, directory, 'Batch92 source/component acceptance starts; no executable index or added trades')
    save(directory/'input-binding.json', bound)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    save(directory/'isolation-baseline.json', {'metadata_index': index, 'code': baseline,
        'metadata_index_sha256': file_sha('data/metadata/index.json'),
        'basis': 'SHA-verified preceding metadata code; committed HEAD equivalence not claimed'})
    stage_protection(root, directory)
    return {'status': 'ok', 'baseline_code_files': len(baseline), 'new_trading_backtests': 0}


def stage_ancestry(directory):
    old = read(stage.DIRECTORY/'baseline.json'); current = read(directory/'baseline.json')
    configs = lambda doc: {n for n in doc['controls'] if Path(n).parts[0] == 'runs' and Path(n).name == 'config.json'}
    require(old['published'] == current['published'] and
        configs(old) == configs(current) and
        all(current['controls'].get(n) == sha for n, sha in old['controls'].items()) and
        all(current['partitions'].get(n) == item for n, item in old['partitions'].items()),
        'Stage-to-acceptance old assets changed')
    return {'status': 'ok', 'stage_baseline_sha256': file_sha(stage.DIRECTORY/'baseline.json'),
        'acceptance_baseline_sha256': file_sha(directory/'baseline.json'),
        'old_control_SHA_bindings_retained': len(old['controls']), 'old_partition_stats_retained': len(old['partitions']),
        'method': 'Compare stage controls against freshly hashed acceptance controls; finish rehashes these via protect'}


def stage_protection(_root, directory):
    binding(directory); output = directory/'stage-protection-baseline.json'
    require(not output.exists(), 'Stage protection already archived')
    result = stage_ancestry(directory); save(output, result); return result


def frozen_operand_binding(directory):
    require(directory == stage.DIRECTORY, 'Unexpected stage directory')
    binding()
    parent = read(stage.PARENT); checked = read(parent['checked_state_file'])
    catalog_sha = checked['implementation_sha256']['src/observe/strategy_catalog.py']
    require(file_sha(DIRECTORY/'strategy_catalog.before.py') == catalog_sha, 'Prior catalog adapter operand changed')
    original_sha = stage.file_sha
    adapted = lambda p: catalog_sha if Path(p) == Path('src/observe/strategy_catalog.py') else original_sha(p)
    # Only the expected historical catalog code identity is adapted. All accepted91 evidence remains checked.
    with patch.object(stage, 'file_sha', adapted):
        bound = ORIGINAL_STAGE_BINDING(directory)
    for item in bound['index_partitions']:
        frame = pd.read_parquet(item['file'])
        require(fingerprint(frame) == item['dataframe_fingerprint'] and len(frame) == item['registered_rows'],
            'Index logical fingerprint differs')
    return bound


def supplemental_replay(frame):
    data = pd.read_parquet(stage.DIRECTORY/'index-input.parquet').set_index('date')
    audits = []
    for label, group in frame.groupby('range', sort=True):
        group = group.set_index('date'); close = data.loc[group.index, 'close']
        score = pd.Series(np.nan, index=close.index)
        for i in range(59, len(close)): score.iloc[i] = stage.independent_score(close.iloc[i-59:i+1].tolist())
        bounds = {}
        for column, q in (('lower_bound', .05), ('upper_bound', .85)):
            series = pd.Series(np.nan, index=close.index)
            for i in range(78, len(close)): series.iloc[i] = stage.independent_quantile(score.iloc[i-19:i+1].tolist(), q)
            bounds[column] = series
        flags = stage.independent_flags(pd.DataFrame({'score': score, **bounds}))
        pd.testing.assert_series_equal(score, group.score, check_names=False, check_exact=True)
        pd.testing.assert_series_equal(flags, group.flag, check_names=False, check_exact=True)
        require(pd.isna(group.paper_next_close_return.iloc[-1]) and pd.isna(group.paper_flag_times_return.iloc[-1]),
            'Last forward return was filled')
        audits.append({'range': label, 'rows': len(group), 'full_independent_flag_differences': 0,
            'difference_dates': [], 'all_score_and_quantile_inputs_independent': True})
    require(audits == read(stage.DIRECTORY/'independent-full-signal-audit.json')['rows'], 'Independent signal audit differs')
    ns = stage.namespace(stage.DIRECTORY); cases = []
    def capture(name, call, kind, **extra):
        try: call()
        except kind as exc: cases.append({'case': name, 'exception': type(exc).__name__, 'message': str(exc), 'synthetic': True, **extra})
        else: raise ValueError('Expected original compatibility failure: '+name)
    sample = pd.Series([0., 1., 0., 2.])
    capture('original_relative_score', lambda: ns['Tren_Score'](sample).calc_trend_score('opposite'), TypeError)
    price = pd.Series([1., 2.], index=pd.to_datetime(['2021-09-10', '2021-09-13']))
    capture('datetime_price_zero_label', lambda: price[0], KeyError)
    instance = ns['Tren_Score'](sample)
    require(instance.calc_trend_score('absolute') is None, 'Original class return changed')
    with patch.object(instance, 'calc_trend_score', return_value=None):
        capture('max_two_None_returns_after_injected_score_calls', instance.calc_absolute_score, TypeError,
            actual_full_mixed_branch_executed=False)
    require(cases == read(stage.DIRECTORY/'compatibility-diagnostics.json')['cases'], 'Compatibility diagnoses differ')
    # Demonstration candidate is reproduced from the frozen raw partition, never used as an index operand.
    demo = read(stage.DIRECTORY/'demo-coverage-audit.json'); path = Path(demo['original_partition'])
    require(file_sha(path) == demo['original_partition_byte_sha256'], 'Demo source bytes changed')
    raw = pd.read_parquet(path)
    require(fingerprint(raw) == demo['original_partition_logical_fingerprint'], 'Demo logical fingerprint changed')
    candidate = raw.loc[raw.instrument.eq('601388.SH') & pd.to_datetime(raw.date).between('2021-08-02', '2021-09-13')]
    candidate = candidate.sort_values('date').tail(17).reset_index(drop=True)
    frozen = pd.read_parquet(demo['candidate_file'])
    pd.testing.assert_frame_equal(candidate, frozen, check_exact=True)
    require(len(candidate) == 17 and demo['dates'] == [str(pd.Timestamp(d).date()) for d in candidate.date] and
        not demo['JQ_adjustment_and_row_semantics_proven'] and not demo['strategy_operand'], 'Demo scope changed')
    return {'status': 'match', 'differences': 0, 'independent_full_states': audits, 'compatibility_cases': cases,
        'raw_demo_candidate_rows': 17, 'JQ_demo_equivalence_proven': False, 'not_a_backtest': True}


def replay_values(adapt_binding):
    with patch.object(stage, 'binding', frozen_operand_binding if adapt_binding else ORIGINAL_STAGE_BINDING):
        frame, summary = stage.compute(stage.DIRECTORY)
    require(summary == read(stage.DIRECTORY/'component-research.json'), 'Original summary differs')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(stage.DIRECTORY/'trend-values.parquet'), check_exact=True)
    require(len(frame) == 7181 and all(r['independent_score_differences'] == 0 for r in summary['ranges']), 'Score/row scope differs')
    return {'status': 'match', 'differences': 0, 'rows': len(frame), 'summary': summary,
        'supplemental': supplemental_replay(frame), 'operand_binding_adapted': adapt_binding,
        'socket_HTTP_disabled': True, 'not_a_backtest': True, 'new_trading_backtests': 0}


def replay(root, directory):
    binding(directory)
    require(file_sha('src/observe/strategy_catalog.py') == file_sha(directory/'strategy_catalog.before.py') and
        read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Replay required before catalog edit')
    result = network_disabled(lambda: replay_values(False)); save(directory/'reference-replay.json', result)
    return {'status': 'match', 'rows': result['rows'], 'differences': 0}


def catalog(root, directory):
    require(directory == DIRECTORY, 'Unexpected acceptance directory')
    require(not (directory/'source-reviews').exists(), 'Catalog phase already archived')
    binding(directory); require(read(directory/'reference-replay.json')['status'] == 'match', 'Original replay missing')
    require(stage_ancestry(directory) == read(directory/'stage-protection-baseline.json'), 'Stage protection proof changed')
    require(read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Active catalog changed')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'; old = {r['path']: r for r in read(prior)}
    require(SOURCE_NAME in REVIEWS and old[SOURCE_NAME]['review_status'] != '人工审查完成', 'New source review missing')
    candidate_root = directory/'catalog-candidate'; candidate_root.mkdir(exist_ok=False)
    evidence = Path(root)/'catalog/strategies/implementation-evidence.json'
    require(file_sha(evidence) == file_sha(directory/'implementation-evidence.before.json'), 'Implementation evidence changed')
    target = candidate_root/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
    shutil.copyfile(evidence, target)
    candidate = catalog_strategies(candidate_root, 'repo/量化策略源代码')
    records = read(Path(candidate['output'])/'catalog.json')
    require(len(records) == len(old) == 695 and {r['path'] for r in records} == set(old) and
        {r['path'] for r in records if r != old[r['path']]} == {SOURCE_NAME}, 'Unrelated catalog changes')
    now = catalog_strategies(root, 'repo/量化策略源代码')
    require(all(file_sha(Path(now['output'])/n) == file_sha(Path(candidate['output'])/n)
        for n in ('catalog.json', 'summary.json', 'catalog.parquet')), 'Validated catalog publication differs')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    save(folder/'review.json', {'snapshot': stage.SNAPSHOT, 'catalog': now, 'not_a_backtest': True,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'sources': [{'source_path': str(stage.SOURCE), 'source_copy': str(stage.DIRECTORY/'original.source'),
            'source_sha256': stage.SOURCE_SHA, 'complete_lines_read': 807, 'review': REVIEWS[SOURCE_NAME]}]})
    return archive(root, 'Batch92 one source rule/catalog registered; formal source/component checks pending; no trades')


def validate_sources(directory):
    require(directory == DIRECTORY, 'Unexpected acceptance directory')
    binding(directory); frozen_operand_binding(stage.DIRECTORY)
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == stage.SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == 1 and
        doc['sources'][0]['review'] == REVIEWS[SOURCE_NAME] and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source review binding changed')
    row = doc['sources'][0]
    require(row['source_path'] == str(stage.SOURCE) and row['source_copy'] == str(stage.DIRECTORY/'original.source') and
        row['source_sha256'] == stage.SOURCE_SHA and row['complete_lines_read'] == 807, 'Source review identity changed')


def offline(root, directory):
    validate_sources(directory)
    result = network_disabled(lambda: replay_values(True))
    require({**result, 'operand_binding_adapted': False} == read(directory/'reference-replay.json'), 'Pre/post catalog replay differs')
    off = network_disabled(lambda: offline_catalog(root, directory)); save(directory/'offline-catalog.json', off)
    validate_catalog(root, directory)
    save(directory/'offline-verification.json', result | {'catalog_files_match': 3, 'platform_equivalence_proven': False})
    return {'status': 'match', 'differences': 0, 'rows': result['rows'], 'catalog_files_match': 3}


def test_command():
    return [*previous.test_command(), 'tests/unit/test_batch92_stage.py', 'tests/unit/test_batch92_acceptance.py']


def test_environment():
    env = dict(os.environ)
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'): env.pop(name, None)
    return env


def expected_test_cases():
    prior = previous.DIRECTORY/'checks/tests.xml'
    parent = read(stage.PARENT); checked = read(parent['checked_state_file'])
    require(file_sha(prior) == checked['evidence_sha256']['checks/tests.xml'], 'Accepted91 test baseline changed')
    stage_evidence()
    cases = [(t.get('classname'), t.get('name')) for t in ET.parse(prior).getroot().iter('testcase')]
    cases += [(t.get('classname'), t.get('name')) for t in ET.parse(stage.DIRECTORY/'checks/tests.xml').getroot().iter('testcase')]
    cases += [('tests.unit.test_batch92_acceptance', name) for name in NEW_CASES]
    require(len(cases) == len(set(cases)) == 191, 'Intended test identities changed')
    return cases


def test_coverage(junit):
    if not Path(junit).exists(): return {'complete': False, 'passed': 0}
    tree = ET.parse(junit).getroot(); cases = list(tree.iter('testcase')); suites = list(tree.iter('testsuite'))
    identities = [(t.get('classname'), t.get('name')) for t in cases]
    total = sum(int(s.get('tests', '0')) for s in suites)
    errors = sum(int(s.get('failures', '0'))+int(s.get('errors', '0')) for s in suites)
    skipped = sum(int(s.get('skipped', '0')) for s in suites)
    bad = any(any(t.find(tag) is not None for tag in ('error', 'failure', 'skipped')) for t in cases)
    expected = expected_test_cases()
    complete = (total == len(cases) == len(expected) and errors == skipped == 0 and not bad and
        len(set(identities)) == len(identities) and set(identities) == set(expected))
    return {'complete': complete, 'passed': total-errors-skipped, 'skipped': skipped, 'errors': errors,
        'files': dict(Counter(n for n, _ in identities))}


def validate_baseline(directory):
    require(all(r['path'] in FILES or file_sha(r['path']) == r['sha256']
        for r in read(directory/'isolation-baseline.json')['code']), 'Unrelated baseline implementation changed')


def isolate(_root, directory):
    validate_sources(directory); validate_baseline(directory)
    require(not (directory/'isolated-files.json').exists(), 'Isolation already archived')
    candidate = {n: file_sha(n) for n in FILES}
    destination = Path(tempfile.mkdtemp(prefix='observe-batch92-')); copied = {}
    def copy(name, sha):
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'Unsafe isolation path')
        require(file_sha(path) == sha, 'Isolated input changed: '+name)
        target = destination/path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        require(file_sha(target) == sha, 'Isolated copy differs'); copied[name] = sha
    old = read(previous.DIRECTORY/'isolated-files.json')
    baseline = read(directory/'isolation-baseline.json')
    old_code = {r['path']: r['sha256'] for r in baseline['code']}
    for name, sha in old['files'].items():
        # This mutable navigation document is not a code/test operand of this batch.
        if name == 'README.md': continue
        if name in old_code: sha = old_code[name]
        if name in FILES: sha = candidate[name]
        copy(name, sha)
    for name, sha in old_code.items(): copy(name, candidate[name] if name in FILES else sha)
    for name, sha in candidate.items(): copy(name, sha)
    for path in stage.DIRECTORY.rglob('*'):
        if path.is_file(): copy(str(path), file_sha(path))
    copy(str(stage.PARENT), file_sha(stage.PARENT)); parent = read(stage.PARENT)
    for prefix in ('receipt', 'checked_state', 'final_binding'): copy(parent[prefix+'_file'], parent[prefix+'_sha256'])
    checked = read(parent['checked_state_file'])
    for name, sha in checked['evidence_sha256'].items(): copy(str(Path(parent['accepted_directory'])/name), sha)
    for path in (directory/'strategy_catalog.before.py', previous.DIRECTORY/'isolated-files.json'):
        copy(str(path), file_sha(path))
    bound = read(stage.DIRECTORY/'input-binding.json')
    copy(bound['snapshot_file'], bound['snapshot_sha256'])
    for row in bound['index_partitions']: copy(row['file'], row['sha256'])
    demo = read(stage.DIRECTORY/'demo-coverage-audit.json')
    copy(demo['original_partition'], demo['original_partition_byte_sha256'])
    manifest = directory/'isolated-files.json'; save(manifest, {'destination': str(destination), 'files': copied,
        'baseline': 'Previous metadata SHA code, pinned prior test operands plus Batch92 candidate; not committed HEAD equivalence'})
    output = directory/'isolated-tests.log'; junit = directory/'isolated-tests.xml'
    command = test_command(); env = dict(test_environment(), PYTHONPATH=str(destination/'src')+os.pathsep+str(destination), PYTHONDONTWRITEBYTECODE='1')
    with output.open('x') as stream:
        result = subprocess.run([*command, '--junitxml='+str(junit.resolve())], cwd=destination,
            env=env, stdout=stream, stderr=subprocess.STDOUT)
    coverage = test_coverage(junit)
    unchanged = candidate == {n: file_sha(n) for n in FILES}
    proof = {'status': 'passed' if result.returncode == 0 and coverage['complete'] and unchanged else 'failed',
        'returncode': result.returncode, 'test_coverage': coverage, 'command': command,
        'copied_manifest': str(manifest), 'copied_manifest_sha256': file_sha(manifest),
        'log': str(output), 'log_sha256': file_sha(output), 'JUnit_file': str(junit), 'JUnit_sha256': file_sha(junit),
        'candidate_implementation_sha256': candidate, 'candidate_unchanged_during_test': unchanged, 'baseline_code_files': len(old_code),
        'known_commit_equivalence_claimed': False}
    save(directory/'isolated-verification.json', proof)
    require(proof['status'] == 'passed', 'Isolated tests failed; preserve all artifacts')
    return {'status': 'passed', 'tests': coverage['passed'], 'copied_files': len(copied)}


def validate_copied_candidate(proof, copied):
    require(all(copied['files'].get(n) == proof['candidate_implementation_sha256'].get(n)
        for n in FILES), 'Isolated candidate differs from tested copies')


def validate_isolation(directory):
    proof = read(directory/'isolated-verification.json')
    require(proof['status'] == 'passed' and proof['returncode'] == 0 and proof['command'] == test_command() and
        file_sha(proof['log']) == proof['log_sha256'] and file_sha(proof['JUnit_file']) == proof['JUnit_sha256'] and
        test_coverage(proof['JUnit_file'])['complete'] and
        proof['candidate_implementation_sha256'] == {n: file_sha(n) for n in FILES}, 'Isolated verification missing or stale')
    validate_baseline(directory)
    require(file_sha(proof['copied_manifest']) == proof['copied_manifest_sha256'], 'Isolation manifest changed')
    copied = read(proof['copied_manifest'])
    validate_copied_candidate(proof, copied)
    require(all(file_sha(Path(copied['destination'])/n) == sha for n, sha in copied['files'].items()), 'Isolated input changed')


def check_commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES],
        test_command(), ['git', 'diff', '--check']]


def validate_commands(checked):
    require([r['command'] for r in checked['commands']] == check_commands(), 'Required command scope changed')
    require(all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Check logs changed')


def validate_run_inventory(root, directory):
    baseline = read(directory/'baseline.json')
    expected = {n for n in baseline['controls'] if Path(n).parts[0] == 'runs' and Path(n).name == 'config.json'}
    current = {p.relative_to(root).as_posix() for p in (Path(root)/'runs').rglob('config.json')}
    require(current == expected, 'Run inventory changed before acceptance publication')


def checks(root, directory):
    validate_sources(directory); validate_catalog(root, directory); validate_isolation(directory)
    validate_run_inventory(root, directory)
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Core evidence missing')
    before = {n: file_sha(n) for n in FILES}; logs = []; folder = directory/'checks'; folder.mkdir(exist_ok=False)
    commands = check_commands()
    for i, command in enumerate(commands):
        log = folder/f'{i}.log'; executed = [*command, '--junitxml='+str((folder/'tests.xml').resolve())] if i == 1 else command
        with log.open('x') as stream: result = subprocess.run(executed, env=test_environment(), stdout=stream, stderr=subprocess.STDOUT)
        logs.append({'command': command, 'returncode': result.returncode, 'log': str(log), 'sha256': file_sha(log)})
        require(result.returncode == 0, 'Check failed; preserve log: '+str(log))
        if i == 1: require(test_coverage(folder/'tests.xml')['complete'], 'Required root test coverage missing')
    require(before == {n: file_sha(n) for n in FILES}, 'Checked code changed')
    for name in FILES:
        target = directory/'implementation-checked'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': logs, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'tests': 191}


def finish(root, directory):
    validate_sources(directory); validate_catalog(root, directory); validate_isolation(directory)
    validate_run_inventory(root, directory)
    require(stage_ancestry(directory) == read(directory/'stage-protection-baseline.json'), 'Stage protection proof changed')
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == {n: file_sha(n) for n in FILES} and
        CORE <= checked['evidence_sha256'].keys(), 'Frozen check scope changed')
    validate_commands(checked)
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
        'Checked copies changed')
    require(test_coverage(directory/'checks/tests.xml')['complete'], 'Frozen root coverage missing')
    require(off['status'] == 'match' and off['rows'] == 7181 and off['differences'] == 0 and off['catalog_files_match'] == 3 and
        off['socket_HTTP_disabled'] and off['not_a_backtest'] and off['new_trading_backtests'] == 0, 'Offline scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed')
    protected = protect(root, directory)
    validate_run_inventory(root, directory)
    progress = archive(root, 'Batch92 original trend source accepted;7181 overlapping index paper states/independent scores and quantiles offline match0; no trading backtest')
    require(progress['manually_reviewed'] == 268 and progress['locally_backtested_sources'] == 12 and progress['local_strategy_runs'] == 75,
        'Acceptance totals differ')
    save(RECEIPT, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': binding(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline_verification': off,
        'isolated_verification': read(directory/'isolated-verification.json'), 'protection': protected, 'progress': progress,
        'newly_reviewed': 1, 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
        'new_requests': 0, 'new_trading_backtests': 0, 'original_strategy_complete': False, 'not_a_backtest': True,
        'restore_required': ['Original JQ data/demo adjustment, count and paused semantics',
            'Original relative/mixed branches, plotting module and old pandas/empyrical notebook environment',
            'Any ETF mapping, execution rule or scoring change is a separate economic variant requiring confirmation and soleBook inputs']})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(RECEIPT),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': 268}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'stage_protection', 'replay', 'catalog', 'offline', 'isolate', 'checks', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()): result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
