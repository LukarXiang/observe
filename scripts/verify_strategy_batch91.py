"""Accept five source reviews; original futures data and trading remain blocked."""
import argparse
import ast
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

import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import review_strategy_batch91 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

PRIOR_DIRECTORY = Path('data/staging/strategies-batch91/20261010-corn-quarterly-acceptance')
DIRECTORY = Path('data/staging/strategies-batch91/20261010-corn-quarterly-acceptance-recovery-v3')
ORIGINAL_STAGE_BINDING = stage.binding
PRIOR_JUNIT_SHA = '485268b21a2be4c390216df89162546c19ae62265cf6a6418a6b4c1d1beea16e'
EXPECTED_TEST_FILES = {'tests.integration.test_strategy_merge': 8, 'tests.unit.test_batch70_research': 13,
    'tests.unit.test_batch91_acceptance': 17, 'tests.unit.test_batch91_stage': 10,
    'tests.unit.test_catalog_dependencies': 29, 'tests.unit.test_catalog_futures': 27,
    'tests.unit.test_catalog_schedules': 33, 'tests.unit.test_research_metadata': 23}
STAGE_RECEIPT = Path('docs/handoff/2026-10-10-batch91-stage-verification.json')
RECEIPT = Path('docs/handoff/2026-10-10-batch91-verification.json')
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch91.py',
    'tests/unit/test_batch91_acceptance.py', 'scripts/review_strategy_batch91.py', 'tests/unit/test_batch91_stage.py')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'reference-replay.json', 'supplemental-replay.json',
    'offline-verification.json', 'offline-catalog.json', 'isolated-verification.json'}


def source_checks():
    rows = read(stage.DIRECTORY/'sources-frozen.json')
    require(len(rows) == len(stage.SOURCES) == 5, 'Five-source scope changed')
    for number, row in enumerate(rows):
        tree = stage.source_tree(number, stage.DIRECTORY)
        require(row['sha256'] == stage.SOURCE_SHA[number] and row['definition_ast'] == stage.definitions(tree),
            'Original definition evidence changed')
    return rows


def binding(directory=None):
    receipt = read(STAGE_RECEIPT)
    require(receipt['status'] == 'stage_checks_passed' and not receipt['formal_acceptance'] and
        receipt['source_count'] == 5 and receipt['synthetic_cases'] == 158 and
        receipt['unique_IF_executable_bodies'] == 1 and receipt['new_trading_backtests'] == 0 and
        receipt['new_requests'] == 1 and receipt['responses'] == receipt['new_rows'] == 0, 'Stage scope changed')
    for prefix in ('stage_evidence', 'checked_state'):
        require(file_sha(receipt[prefix+'_file']) == receipt[prefix+'_sha256'], 'Stage receipt binding changed')
    evidence = read(receipt['stage_evidence_file'])
    require(receipt['implementation_sha256'] == evidence['implementation_sha256'] and
        all(file_sha(n) == sha for n, sha in evidence['implementation_sha256'].items()), 'Frozen stage implementation changed')
    require(all(file_sha(stage.DIRECTORY/n) == sha for n, sha in evidence['files'].items()), 'Frozen stage evidence changed')
    require(all(file_sha(stage.DIRECTORY/n) == sha for n, sha in receipt['additional_evidence_sha256'].items()),
        'Supplemental stage evidence changed')
    source_checks()
    result = {'snapshot': stage.SNAPSHOT, 'stage_receipt': str(STAGE_RECEIPT),
        'stage_receipt_sha256': file_sha(STAGE_RECEIPT), 'stage_evidence_sha256': receipt['stage_evidence_sha256'],
        'additional_evidence_sha256': receipt['additional_evidence_sha256'], 'new_requests': 0, 'not_a_backtest': True,
        'binding_policy': 'Original stage replay before catalog edit; after edit only prior-catalog binding adapted to SHA-verified unchanged frozen operands'}
    if directory is not None:
        require(read(directory/'input-binding.json') == result, 'Acceptance input binding changed')
    return result


def preflight(root, directory):
    bound = binding(); stage.binding(stage.DIRECTORY)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch91 acceptance starts; five sources, no admitted original8888 or IF1m/settlement')
    save(directory/'input-binding.json', bound)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    # Frozen Git metadata defines the preceding accepted code surface, including earlier uncommitted work.
    index = read('data/metadata/index.json')
    baseline = read(index['implementation_files'])
    require(all(file_sha(r['path']) == r['sha256'] for r in baseline), 'Isolation baseline code changed')
    save(directory/'isolation-baseline.json', {'metadata_index': index,
        'metadata_index_sha256': file_sha('data/metadata/index.json'), 'code': baseline,
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
        'basis': 'SHA-verified preceding metadata code snapshot, not a claim of committed HEAD equivalence'})
    return {'status': 'ok', 'new_trading_backtests': 0}


def frozen_operand_binding(directory):
    require(directory == stage.DIRECTORY, 'Unexpected stage directory')
    binding(); bound = read(directory/'input-binding.json')
    parent = read(stage.PARENT); checked = read(parent['checked_state_file'])
    previous_catalog_sha = checked['implementation_sha256']['src/observe/strategy_catalog.py']
    require(file_sha(PRIOR_DIRECTORY/'strategy_catalog.before.py') == previous_catalog_sha,
        'Prior catalog adapter operand changed')
    original_sha = stage.file_sha
    adapted_sha = lambda path: previous_catalog_sha if Path(path) == Path('src/observe/strategy_catalog.py') else original_sha(path)
    with patch.object(stage, 'file_sha', adapted_sha):
        require(ORIGINAL_STAGE_BINDING(directory) == bound, 'Original stage binding differs')
    for ins, row in bound['supplier_samples'].items():
        require(ins in ('C', 'CS') and row['status'] == 'success', 'Unexpected candidate sample')
        for item in row['files']+row['responses']:
            require(file_sha(item['file']) == item['sha256'], 'Original candidate supplier bytes changed')
        require(file_sha(directory/(ins+'-input.parquet')) == row['files'][0]['sha256'], 'Candidate operand changed')
    require(file_sha(bound['parent_review_file']) == bound['parent_review_sha256'] and
        file_sha(stage.DAILY_RECEIPT) == bound['daily_receipt_sha256'], 'Original upstream receipts changed')
    return bound


def supplemental_replay():
    receipt = read(STAGE_RECEIPT); coverage = read(stage.DIRECTORY/'candidate-coverage-audit.json')
    require(coverage == receipt['candidate_coverage'], 'Coverage receipt differs')
    path = Path(coverage['calendar_file']); require(file_sha(path) == coverage['calendar_sha256'], 'Reference calendar changed')
    calendar = pd.read_parquet(path); known = set(pd.to_datetime(calendar.date))
    opened = set(pd.to_datetime(calendar.loc[calendar.is_open, 'date']))
    candidates = {i: pd.read_parquet(stage.DIRECTORY/(i+'-input.parquet')) for i in ('C', 'CS')}
    dates = {i: set(pd.to_datetime(f.date)[pd.to_datetime(f.date).between('2014-12-22', '2026-09-29')])
        for i, f in candidates.items()}
    fmt = lambda items: [str(day.date()) for day in sorted(items)]
    require(coverage['C_only_dates'] == fmt(dates['C']-dates['CS']) and
        coverage['CS_only_dates'] == fmt(dates['CS']-dates['C']), 'Unmatched dates differ')
    values = pd.read_parquet(stage.DIRECTORY/'corn-values.parquet'); ranges = []
    for label, group in values.groupby('range', sort=True):
        common = set(group.date); reference = {d for d in opened if group.date.min() <= d <= group.date.max()}
        ranges.append({'range': label, 'common_rows': len(group), 'closed_reference_dates': fmt((common&known)-opened),
            'unknown_reference_dates': fmt(common-known), 'missing_open_reference_dates': fmt(reference-common)})
    require(ranges == coverage['ranges'] and not coverage['futures_calendar_equivalence_proven'], 'Reference calendar scope differs')
    compat = read(stage.DIRECTORY/'notebook-compatibility-review.json')
    node = next(n for n in stage.source_tree(0, stage.DIRECTORY).body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'coef' for t in n.targets))
    require(compat['coefficient_expression_ast'] == ast.dump(node.value, include_attributes=False) and
        compat['source_sha256'] == stage.SOURCE_SHA[0] and not compat['actual_statsmodels_result_observed'], 'Coefficient diagnosis scope differs')
    from types import SimpleNamespace
    try:
        eval(compile(ast.Expression(body=node.value), '<pinned-coef-compatibility>', 'eval'),
            {'result': SimpleNamespace(params=pd.Series([1., 2.], index=compat['synthetic_named_parameter_labels']))})
    except KeyError as exc:
        require(exc.args == (compat['missing_label'],) and compat['exception'] == 'KeyError', 'Coefficient error differs')
    else: raise ValueError('Expected original synthetic named-Series failure')
    engine = read(stage.DIRECTORY/'future-engine-reference.json')
    for ref in engine['references']:
        require(file_sha(ref['file']) == file_sha(ref['copy']) == ref['sha256'], 'Future engine reference changed')
    return {'status': 'match', 'differences': 0, 'reference_calendar_rows': len(calendar),
        'candidate_ranges': ranges, 'synthetic_coefficient_error': 'KeyError', 'future_engine_reference_files': len(engine['references']),
        'actual_statsmodels_executed': False, 'not_a_backtest': True}


def replay_values(adapt_binding):
    with patch.object(stage, 'binding', frozen_operand_binding) if adapt_binding else patch.object(stage, 'binding', stage.binding):
        diag = stage.diagnostics(stage.DIRECTORY)
        values, summary = stage.component(stage.DIRECTORY)
        probes = stage.probe_audit(stage.DIRECTORY)
        require(diag == read(stage.DIRECTORY/'diagnostics.json') and
            summary == read(stage.DIRECTORY/'component-research.json') and
            probes == read(stage.DIRECTORY/'probe-audit.json'), 'Stage replay summary differs')
        pd.testing.assert_frame_equal(values, pd.read_parquet(stage.DIRECTORY/'corn-values.parquet'), check_exact=True)
        for row in read(stage.DIRECTORY/'input-binding.json')['supplier_samples'].values():
            pd.testing.assert_frame_equal(stage.daily.replay_frame(row), pd.read_parquet(row['files'][0]['file']), check_exact=True)
    return {'status': 'match', 'differences': 0, 'synthetic_cases': diag['case_count'],
        'candidate_rows_overlapping_ranges': len(values), 'supplier_daily_wire_parsers': 2,
        'minute_probes': probes, 'socket_HTTP_disabled': True, 'operand_binding_adapted': adapt_binding,
        'original_arithmetic_body_changed': False, 'not_a_backtest': True}


def replay(root, directory):
    binding(directory)
    require(file_sha('src/observe/strategy_catalog.py') == file_sha(directory/'strategy_catalog.before.py') and
        read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Replay required before catalog edit')
    save(directory/'reference-replay.json', network_disabled(lambda: replay_values(False)))
    save(directory/'supplemental-replay.json', network_disabled(supplemental_replay))
    return read(directory/'reference-replay.json')


def catalog(root, directory):
    require(not (directory/'source-reviews').exists(), 'Catalog phase already archived')
    binding(directory); require(read(directory/'reference-replay.json')['status'] == 'match', 'Original replay missing')
    require(read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'),
        'Active catalog changed before registration')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'; old = {r['path']: r for r in read(prior)}
    rows = []
    for name, row in zip(stage.SOURCES, source_checks(), strict=True):
        require(name in REVIEWS and old[name]['review_status'] != '人工审查完成', 'Five new reviews missing')
        rows.append({**row, 'review': REVIEWS[name], 'formal_acceptance': 'source_research_only'})
    candidate_root = directory/'catalog-candidate'
    candidate_root.mkdir(exist_ok=False)
    evidence = Path(root)/'catalog/strategies/implementation-evidence.json'
    require(file_sha(evidence) == file_sha(directory/'implementation-evidence.before.json'), 'Implementation evidence changed')
    target = candidate_root/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
    shutil.copyfile(evidence, target)
    candidate = catalog_strategies(candidate_root, 'repo/量化策略源代码')
    records = read(Path(candidate['output'])/'catalog.json')
    require(len(records) == len(old) == 695 and {r['path'] for r in records} == set(old) and
        {r['path'] for r in records if r != old[r['path']]} == set(stage.SOURCES), 'Unrelated catalog changes')
    now = catalog_strategies(root, 'repo/量化策略源代码')
    require(all(file_sha(Path(now['output'])/n) == file_sha(Path(candidate['output'])/n)
        for n in ('catalog.json', 'summary.json', 'catalog.parquet')), 'Validated catalog publication differs')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    save(folder/'review.json', {'snapshot': stage.SNAPSHOT, 'sources': rows, 'catalog': now,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch91 five source rules/catalog registered; formal checks pending; original trading blocked')


def validate_sources(directory):
    binding(directory); doc = read(directory/'source-reviews/review.json')
    frozen_operand_binding(stage.DIRECTORY)
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'] and len(doc['sources']) == 5,
        'Original catalog/source scope changed')
    for name, original, row in zip(stage.SOURCES, source_checks(), doc['sources'], strict=True):
        require(row == {**original, 'review': REVIEWS[name], 'formal_acceptance': 'source_research_only'}, 'Source review changed')


def offline(root, directory):
    validate_sources(directory)
    post = network_disabled(lambda: replay_values(True)); supplement = network_disabled(supplemental_replay)
    require(supplement == read(directory/'supplemental-replay.json'), 'Supplemental replay differs')
    require({**post, 'operand_binding_adapted': False} == read(directory/'reference-replay.json'), 'Pre/post catalog values differ')
    off = network_disabled(lambda: offline_catalog(root, directory))
    save(directory/'offline-catalog.json', off); validate_catalog(root, directory)
    result = {**post, 'supplemental_replay': supplement, 'catalog_files_match': 3,
        'new_trading_backtests': 0, 'platform_equivalence_proven': False}
    save(directory/'offline-verification.json', result); return result


def test_command():
    return [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', '-o', 'addopts=', 'tests/unit/test_batch91_acceptance.py',
        'tests/unit/test_batch91_stage.py', 'tests/unit/test_batch70_research.py', 'tests/unit/test_catalog_futures.py',
        'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
        'tests/integration/test_strategy_merge.py']


def test_environment():
    env = dict(os.environ)
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'): env.pop(name, None)
    return env


def expected_test_cases():
    prior = PRIOR_DIRECTORY/'isolated-tests.xml'
    require(file_sha(prior) == PRIOR_JUNIT_SHA, 'Original test baseline changed')
    cases = [(t.get('classname'), t.get('name')) for t in ET.parse(prior).getroot().iter('testcase')]
    cases.extend(('tests.unit.test_batch91_acceptance', name) for name in (
        'test_partial_isolated_junit_is_rejected_before_checks[tests.unit.test_batch91_acceptance]',
        'test_partial_isolated_junit_is_rejected_before_checks[unrelated.tests]',
        'test_pytest_selection_environment_is_removed',
        'test_complete_junit_requires_every_intended_file_and_no_duplicate_cases',
        'test_baseline_helper_change_is_rejected',
        'test_adapter_preserves_upstream_accepted_evidence_checks[baseline.json]',
        'test_adapter_preserves_upstream_accepted_evidence_checks[input-binding.json]',
        'test_adapter_catalog_replay_keeps_original_binding_callable'))
    require(dict(Counter(n for n, _ in cases)) == EXPECTED_TEST_FILES and len(set(cases)) == 160,
        'Frozen intended test identities differ')
    return cases


def test_coverage(junit):
    if not Path(junit).is_file(): return {'complete': False, 'passed': 0, 'files': {}}
    tree = ET.parse(junit).getroot(); suites = list(tree.iter('testsuite'))
    cases = list(tree.iter('testcase')); identities = [(t.get('classname'), t.get('name')) for t in cases]
    counts = dict(Counter(n for n, _ in identities))
    total = sum(int(s.get('tests', '0')) for s in suites)
    skipped = sum(int(s.get('skipped', '0')) for s in suites)
    errors = sum(int(s.get('errors', '0'))+int(s.get('failures', '0')) for s in suites)
    bad_cases = any(any(t.find(tag) is not None for tag in ('skipped', 'failure', 'error')) for t in cases)
    complete = (total == len(cases) == 160 and skipped == errors == 0 and not bad_cases and
        counts == EXPECTED_TEST_FILES and len(set(identities)) == len(identities) and
        set(identities) == set(expected_test_cases()))
    return {'complete': complete, 'passed': total-skipped-errors, 'skipped': skipped, 'errors': errors, 'files': counts}


def validate_baseline(directory):
    baseline = read(directory/'isolation-baseline.json')
    require(all(r['path'] in FILES or file_sha(r['path']) == r['sha256'] for r in baseline['code']),
        'Unrelated baseline implementation changed')


def validate_isolation(directory):
    isolated = read(directory/'isolated-verification.json')
    require(isolated['status'] == 'passed' and isolated['returncode'] == 0 and isolated['skipped'] == 0 and
        isolated['command'] == test_command() and file_sha(isolated['log']) == isolated['log_sha256'] and
        file_sha(isolated['JUnit_file']) == isolated['JUnit_sha256'] and
        test_coverage(isolated['JUnit_file'])['complete'] and isolated['passed'] == 160 and
        isolated['candidate_implementation_sha256'] == {n: file_sha(n) for n in FILES},
        'Isolated verification missing or stale')
    validate_baseline(directory)
    require(file_sha(isolated['copied_manifest']) == isolated['copied_manifest_sha256'], 'Isolated copy manifest changed')
    copied = read(isolated['copied_manifest'])
    require(all(file_sha(Path(copied['destination'])/n) == sha for n, sha in copied['files'].items()),
        'Isolated copied input changed')


def recover(root, directory):
    """Retain preceding proofs; seed a new immutable recovery from frozen operands."""
    require(directory != PRIOR_DIRECTORY, 'Recovery must use a new directory')
    validate_sources(PRIOR_DIRECTORY)
    prior = read(PRIOR_DIRECTORY/'isolated-verification.json')
    require(file_sha(prior['log']) == prior['log_sha256'] and file_sha(prior['JUnit_file']) == PRIOR_JUNIT_SHA and
        file_sha(prior['copied_manifest']) == prior['copied_manifest_sha256'], 'Previous isolation evidence changed')
    files = {p.relative_to(PRIOR_DIRECTORY).as_posix(): file_sha(p) for p in PRIOR_DIRECTORY.rglob('*') if p.is_file()}
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
        'implementation-evidence.before.json', 'progress-start.json', 'isolation-baseline.json',
        'source-reviews/review.json', 'reference-replay.json', 'supplemental-replay.json'):
        out = directory/name; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PRIOR_DIRECTORY/name, out); require(file_sha(out) == files[name], 'Recovery operand copy differs')
    save(directory/'recovery-binding.json', {'prior_directory': str(PRIOR_DIRECTORY), 'prior_files_sha256': files,
        'prior_isolation_retained': True, 'prior_isolation_candidate_superseded': True,
        'fixes': ['Block pytest selection; require all160 unique intended identities and eight file counts',
            'Recheck all unrelated baseline code and copied isolated input hashes at checks and finish',
            'Preserve entire accepted90/70 binding; adapt only previous catalog code SHA'],
        'implementation_sha256': {n: file_sha(n) for n in FILES}, 'new_trading_backtests': 0})
    return {'status': 'ok', 'directory': str(directory)}


def isolate(root, directory):
    """Copy the preceding SHA-pinned code baseline plus this change into a fresh cwd."""
    validate_sources(directory); baseline = read(directory/'isolation-baseline.json')
    require(not (directory/'isolated-verification.json').exists(), 'Isolation already archived')
    destination = Path(tempfile.mkdtemp(prefix='observe-batch91-'))
    code = {r['path']: r['sha256'] for r in baseline['code']}
    candidate = {n: file_sha(n) for n in FILES}
    require({n for n, sha in code.items() if file_sha(n) != sha} <= set(FILES), 'Unrelated baseline implementation changed')
    code.update(candidate); copied = {}
    def copy(name, expected):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Isolation requires repository-relative input')
        require(file_sha(path) == expected, 'Isolated source SHA changed')
        out = destination/path; out.parent.mkdir(parents=True, exist_ok=True)
        require(not out.exists(), 'Duplicate isolated path')
        shutil.copyfile(path, out); require(file_sha(out) == expected, 'Isolated copied SHA differs')
        copied[path.as_posix()] = expected
    for name, sha in sorted(code.items()): copy(name, sha)
    review = read(directory/'source-reviews/review.json')
    originals = read(Path(review['catalog']['output'])/'catalog.json')
    for row in originals: copy('repo/量化策略源代码/'+row['path'], row['bytes_sha256'])
    require(len(originals) == 695, 'Original source isolation incomplete')
    evidence = read(stage.DIRECTORY/'stage-evidence.json')
    for name, sha in evidence['files'].items(): copy(str(stage.DIRECTORY/name), sha)
    copy(str(stage.DIRECTORY/'stage-evidence.json'), file_sha(stage.DIRECTORY/'stage-evidence.json'))
    receipt = read(STAGE_RECEIPT); copy(str(STAGE_RECEIPT), file_sha(STAGE_RECEIPT))
    for name, sha in receipt['additional_evidence_sha256'].items(): copy(str(stage.DIRECTORY/name), sha)
    coverage = read(stage.DIRECTORY/'candidate-coverage-audit.json')
    copy(coverage['calendar_file'], coverage['calendar_sha256'])
    engine = read(stage.DIRECTORY/'future-engine-reference.json')
    for row in engine['references']: copy(row['file'], row['sha256'])
    api = read(stage.DIRECTORY/'existing-apis.json'); copy(api['reference_file'], api['reference_sha256'])
    copy(str(PRIOR_DIRECTORY/'isolated-tests.xml'), PRIOR_JUNIT_SHA)
    copy(str(PRIOR_DIRECTORY/'strategy_catalog.before.py'), file_sha(PRIOR_DIRECTORY/'strategy_catalog.before.py'))
    parent = read(stage.PARENT); copy(str(stage.PARENT), file_sha(stage.PARENT))
    for prefix in ('receipt', 'checked_state', 'final_binding'): copy(parent[prefix+'_file'], parent[prefix+'_sha256'])
    upstream = read(parent['checked_state_file'])
    for name, sha in upstream['evidence_sha256'].items(): copy(str(Path(parent['accepted_directory'])/name), sha)
    copy(str(stage.DAILY_RECEIPT), file_sha(stage.DAILY_RECEIPT))
    for row in read(stage.DIRECTORY/'input-binding.json')['supplier_samples'].values():
        for item in row['files']+row['responses']:
            if item['file'] not in copied: copy(item['file'], item['sha256'])
    for ins in ('C', 'CS'):
        for name in ('result.json', 'response.bin', 'parsed.parquet'):
            path = str(stage.daily.DIRECTORY/'probes'/ins/name)
            if path not in copied: copy(path, file_sha(path))
    for name in ('README.md',): copy(name, file_sha(name))
    manifest = directory/'isolated-files.json'; save(manifest, {'destination': str(destination), 'files': copied,
        'baseline': 'Previous Git-metadata SHA snapshot plus five candidate code files; no checkout or moving user work',
        'git_head': baseline['git_head'], 'known_commit_equivalence_claimed': False})
    output = directory/'isolated-tests.log'; junit = directory/'isolated-tests.xml'
    command = test_command(); executed = [*command, '--junitxml='+str(junit.resolve())]
    env = dict(test_environment(), PYTHONPATH=str(destination/'src')+os.pathsep+str(destination), PYTHONDONTWRITEBYTECODE='1')
    with output.open('x') as stream:
        result = subprocess.run(executed, cwd=destination, env=env, stdout=stream, stderr=subprocess.STDOUT)
    coverage = test_coverage(junit)
    proof = {'status': 'passed' if result.returncode == 0 and coverage['complete'] else 'failed',
        'returncode': result.returncode, 'passed': coverage['passed'], 'skipped': coverage.get('skipped', 0), 'errors': coverage.get('errors', 0),
        'test_coverage': coverage, 'pytest_environment_selection_removed': True,
        'command': command, 'executed_command': executed, 'isolated_directory': str(destination),
        'log': str(output), 'log_sha256': file_sha(output), 'JUnit_file': str(junit), 'JUnit_sha256': file_sha(junit) if junit.exists() else None,
        'copied_manifest': str(manifest), 'copied_manifest_sha256': file_sha(manifest),
        'baseline_code_files': len(baseline['code']), 'source_files': len(originals), 'candidate_implementation_sha256': candidate,
        'known_commit_equivalence_claimed': False, 'unrelated_WIP_change_allowed': False}
    save(directory/'isolated-verification.json', proof)
    require(proof['status'] == 'passed', 'Isolated tests failed; preserve complete log and files')
    require(candidate == {n: file_sha(n) for n in FILES}, 'Candidate changed during isolation')
    return proof


def checks(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Core evidence missing')
    validate_isolation(directory)
    before = {n: file_sha(n) for n in FILES}; logs = []; folder = directory/'checks'; folder.mkdir(exist_ok=False)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES], test_command(), ['git', 'diff', '--check']]
    for number, command in enumerate(commands):
        path = folder/f'{number}.log'
        executed = [*command, '--junitxml='+str((folder/'tests.xml').resolve())] if number == 1 else command
        with path.open('x') as output: result = subprocess.run(executed, env=test_environment(), stdout=output, stderr=subprocess.STDOUT)
        logs.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, 'Check failed; preserve log: '+str(path))
        if number == 1: require(test_coverage(folder/'tests.xml')['complete'], 'Required root test coverage missing')
    require(before == {n: file_sha(n) for n in FILES}, 'Checked code changed')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': logs, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    validate_isolation(directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == {n: file_sha(n) for n in FILES} and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] ==
        [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES], test_command(),
            ['git', 'diff', '--check']], 'Frozen check scope changed')
    require(all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Check logs changed')
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
        'Checked code copies changed')
    require(test_coverage(directory/'checks/tests.xml')['complete'], 'Frozen root test coverage missing')
    recovery = read(directory/'recovery-binding.json')
    require(all(file_sha(PRIOR_DIRECTORY/n) == sha for n, sha in recovery['prior_files_sha256'].items()),
        'Prior acceptance evidence changed')
    require(off['status'] == 'match' and off['differences'] == 0 and off['synthetic_cases'] == 158 and
        off['candidate_rows_overlapping_ranges'] == 3935 and off['catalog_files_match'] == 3 and
        off['socket_HTTP_disabled'] and not off['original_arithmetic_body_changed'] and off['new_trading_backtests'] == 0,
        'Offline acceptance scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed')
    protected = protect(root, directory)
    progress = archive(root, 'Batch91 five source reviews accepted;158 original synthetic/3935 overlapping corn candidate rows and two wire parsers offline match0; original trading blocked')
    require(progress['manually_reviewed'] == 267 and progress['locally_backtested_sources'] == 12 and
        progress['local_strategy_runs'] == 75, 'Acceptance totals differ')
    save(RECEIPT, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': binding(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline_verification': off,
        'isolated_verification': read(directory/'isolated-verification.json'), 'protection': protected, 'progress': progress,
        'newly_reviewed': 5, 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
        'new_requests': 0, 'prior_stage_requests': 1, 'prior_stage_responses': 0, 'original_signal_rows': 0,
        'new_trading_backtests': 0, 'original_strategy_complete': False, 'not_a_backtest': True,
        'restore_required': ['OriginalC8888/CS8888 construction and historical daily price identity; original statsmodels/coint and notebook context',
            'Real IF quarterly300 completed1m rows/session calendars, expiry, multiplier, tick and daily settlement/margin histories',
            'Original platform long/short/positions_value and close_today defaults, rejected/partial legs and rollover lifecycle',
            'Extend soleBook futures execution only with admitted inputs; economic compatibility/atomic-order changes require separate confirmation']})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(RECEIPT),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'replay', 'catalog', 'recover', 'offline', 'isolate', 'checks', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()): result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
