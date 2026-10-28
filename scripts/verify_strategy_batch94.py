"""Accept three ETF sources as research; native references never admit trades."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import API_NAMES, CATALOG_VERSION, REVIEWS, catalog_strategies, read_source
from scripts import diagnose_strategy_batch94 as callbacks
from scripts import reference_strategy_batch94 as reference
from scripts import recover_strategy_batch94_reference as recovery
from scripts import review_strategy_batch94 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch94/20261010-etf-filters-acceptance-v5')
RECEIPT = Path('docs/handoff/2026-10-10-batch94-verification.json')
PINS = {
    str(reference.STAGE_RECEIPT): reference.STAGE_RECEIPT_SHA,
    str(callbacks.PARENT): callbacks.PARENT_SHA,
    'docs/handoff/2026-10-10-batch94-callback-stage.json':
        'b110f93d983f0bc09ffeada02b7e5d483fc947b0c37f72c7327da9cb111eff49',
}
SOURCES = ('2023年度精选策略/37.ETF动量轮动RSRS择时-魔改4-低通滤波.txt',
    '2023年度精选策略/47.动量ETF轮动-RSRS择时-卡尔曼滤波.txt',
    '2024年度精选策略2/18.动量ETF轮动-RSRS择时-卡尔曼滤波.txt')
CORRECTION = '2024年度精选策略2/67.三因子策略年化21%简单多因子策略框架.txt'
BASE_CATALOG = Path('data/catalog/strategies/83d74a59b03206fedabd/catalog.json')
BASE_CATALOG_SHA = 'b295b7318e1f925f6a9f779db82e2f9b7159e31e194ca6f5478016bb520f36a2'
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch94.py',
    'tests/unit/test_batch94_acceptance.py', *stage.FILES, *reference.FILES,
    *recovery.FILES, 'tests/unit/test_batch94_reference_transport.py', *callbacks.FILES)
TEST_FILES = ('tests/unit/test_batch94_stage.py', 'tests/unit/test_batch94_reference.py',
    'tests/unit/test_batch94_reference_recovery.py', 'tests/unit/test_batch94_reference_transport.py',
    'tests/unit/test_batch94_callbacks.py', 'tests/unit/test_batch94_acceptance.py')
EXCLUDED = 'tests/unit/test_batch94_reference_recovery.py::test_socket_methods_blocked_before_transport'
NEW_CASES = ('test_alternate_root_refused_before_reads',
    'test_historical_parent_uses_pinned_catalog_copy',
    'test_implementation_inventory_refuses_empty_subset_extra[empty]',
    'test_implementation_inventory_refuses_empty_subset_extra[subset]',
    'test_implementation_inventory_refuses_empty_subset_extra[extra]',
    'test_implementation_inventory_checks_current_and_copy[live]',
    'test_implementation_inventory_checks_current_and_copy[copy]',
    'test_junit_exact_identity_refuses_empty_subset_duplicate_and_skip',
    'test_command_gate_refuses_empty_and_substituted',
    'test_catalog_changes_are_exact_and_correction_is_text_only',
    'test_new_run_inventory_refused', 'test_replay_tables_refuse_changed_bytes',
    'test_progress_checks_exact_catalog_identity', 'test_environment_removes_selection_flags',
    'test_unrelated_baseline_code_refused', 'test_candidate_publication_uses_validated_bytes',
    'test_candidate_publication_refuses_changed_navigation',
    'test_source_reviews_preserve_nontrading_rules',
    'test_replay_evidence_requires_exact_inventory', 'test_symlink_copy_refused',
    *(f'test_new_source_values_cannot_claim_false_trading[{field}]' for field in
        ('rules', 'status', 'implementations', 'original_strategy_complete', 'bytes_sha256')),
    *(f'test_catalog_artifacts_refuse_self_consistent_false_proof[{field}]' for field in
        ('catalog_id', 'summary', 'parquet', 'output', 'ancestry')),
    'test_final_bindings_refuse_evidence_changed_during_replay',
    *(f'test_finish_binds_validated_review_after_archiving[{timing}]' for timing in
        ('normal', 'review_read', 'checked_read', 'archive')))
REPLAY_NAMES = ('rsrs-reference.parquet', 'fft-native-reference.parquet',
    'kalman-native-reference.parquet', 'literal-rsrs-capture.parquet', 'callbacks.json')
CORE = {'baseline.json', 'catalog-before.json', 'strategy_catalog.before.py', 'input-binding.json',
    'isolation-baseline.json', 'candidate.json', 'catalog-offline.json', 'replay.json', 'replay-command.json',
    'isolated-verification.json', 'isolated-files.json', 'stage-ancestry.json'}


def require_root(root):
    require(Path(root).resolve() == Path('data').resolve(), 'Only canonical data root is supported')


def regular(path):
    path = Path(path)
    require(path.is_file() and path.resolve().is_relative_to(Path.cwd().resolve()) and
        not any(p.is_symlink() for p in (path, *path.parents)), 'Unrooted or symlink evidence: '+str(path))
    return path


def read_bound(path):
    """Parse and identify the same bytes, then later refuse any replacement."""
    data = regular(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def historical_parent():
    """Verify historical catalog bytes directly; never forge a live file hash."""
    doc = read(stage.PARENT); paths = {str(stage.PARENT): file_sha(stage.PARENT)}
    require(doc['status'] == 'ok' and doc['formal_source_research_acceptance'] and
        not doc['formal_trading_acceptance'] and doc['manually_reviewed'] == 270, 'Parent scope changed')
    for key in ('receipt', 'checked_state', 'final_binding', 'formal_report', 'review_completion'):
        name = doc[key+'_file'] if key != 'formal_report' else doc[key]
        require(file_sha(regular(name)) == doc[key+'_sha256'], 'Historical parent evidence changed')
        paths[name] = doc[key+'_sha256']
    for name, sha in doc['reviewed_implementation_sha256'].items():
        actual = Path(doc['accepted_directory'])/'implementation-checked'/name if name == FILES[0] else Path(name)
        require(file_sha(regular(actual)) == sha, 'Historical parent implementation changed')
        paths[name] = sha
    require(paths == read(stage.DIRECTORY/'parent-binding.json'), 'Historical parent inventory differs')
    return paths


def stage_evidence():
    documents = []
    for name, sha in PINS.items():
        require(file_sha(regular(name)) == sha, 'Pinned stage receipt changed')
        doc = read(name); directory = Path(doc['stage_directory'])
        require(not doc['formal_source_research_acceptance'] and not doc['formal_trading_acceptance'] and
            doc['new_trading_backtests'] == 0, 'Stage became trading or formal acceptance')
        require(file_sha(regular(doc['report'])) == doc['report_sha256'] and
            file_sha(regular(doc['stage_binding'])) == doc['stage_binding_sha256'], 'Stage report/binding changed')
        bound = read(doc['stage_binding'])
        evidence = doc.get('payload_evidence_sha256') or {
            str(directory/n): digest for n, digest in bound['evidence_sha256'].items()}
        require(bool(evidence), 'Empty historical stage evidence')
        for path, digest in evidence.items():
            require(file_sha(regular(path)) == digest, 'Historical stage evidence changed')
        for path, digest in doc['implementation_sha256'].items():
            require(file_sha(regular(path)) == digest, 'Historical implementation changed')
            copy = directory/'implementation'/path
            if copy.exists(): require(file_sha(regular(copy)) == digest, 'Historical copied implementation changed')
        documents.append(doc)
    historical_parent(); reference.historical_stage(); callbacks.validate_implementations(callbacks.DIRECTORY)
    require(documents[0]['RSRS']['rows'] == 4662 and documents[0]['FFT']['windows'] == 10467 and
        documents[1]['runtime_provenance']['unmodified_reference_files'] == 70 and
        documents[2]['original_callbacks']['cases'] == 102, 'Historical numerical/callback scope changed')
    return documents


def implementation(directory, manifest=None):
    hashes = {n: file_sha(regular(n)) for n in FILES}
    if manifest is not None:
        require(isinstance(manifest, dict) and set(manifest) == set(FILES) and manifest == hashes,
            'Implementation inventory differs')
        for name, sha in manifest.items():
            require(file_sha(regular(directory/'implementation-checked'/name)) == sha, 'Checked implementation copy differs')
    return hashes


def validate_baseline(directory):
    rows = read(directory/'isolation-baseline.json')['code']
    require(bool(rows), 'Empty isolation code baseline')
    for row in rows:
        require(row['path'] in FILES[:3] or file_sha(regular(row['path'])) == row['sha256'],
            'Unrelated baseline code changed')


def ancestry(directory):
    old = read(callbacks.DIRECTORY/'baseline.json'); now = read(directory/'baseline.json')
    require(old['published'] == now['published'] and
        all(now['controls'].get(n) == sha for n, sha in old['controls'].items()) and
        all(now['partitions'].get(n) == v for n, v in old['partitions'].items()), 'Stage old assets changed')
    validate_runs(directory)
    return {'status': 'ok', 'old_controls': len(old['controls']), 'old_partitions': len(old['partitions']),
        'baseline_sha256': file_sha(directory/'baseline.json'), 'stage_baseline_sha256': file_sha(callbacks.DIRECTORY/'baseline.json')}


def prepare(root, directory):
    require_root(root); require(directory == DIRECTORY and not directory.exists(), 'Acceptance directory exists or differs')
    stage_evidence(); reference.validate_index(root)
    index = read('data/metadata/index.json'); code = read(index['implementation_files'])
    require(all(r['path'] == FILES[0] or file_sha(regular(r['path'])) == r['sha256'] for r in code),
        'Metadata baseline code changed outside candidate catalog')
    validate_catalog_code(directory)
    with operation_lock(root, DATA_WRITER):
        checkpoint(root, directory, 'Batch94 formal source research acceptance starts; no trades')
        save(directory/'input-binding.json', {'receipt_sha256': PINS, 'snapshot': '20261006-145049-e06c',
            'source_sha256': list(stage.SOURCE_SHA), 'new_trading_backtests': 0})
        save(directory/'isolation-baseline.json', {'code': code, 'metadata_index': index,
            'metadata_index_sha256': file_sha('data/metadata/index.json')})
        shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
        shutil.copyfile(FILES[0], directory/'strategy_catalog.before.py')
        save(directory/'stage-ancestry.json', ancestry(directory))
    return {'status': 'prepared', 'new_trading_backtests': 0}


def validate_changes(old, new):
    require(isinstance(old, list) and isinstance(new, list) and len(old) == len(new) == 695,
        'Catalog row inventory differs')
    old = {r['path']: r for r in old}; new = {r['path']: r for r in new}
    require(len(old) == len(new) == 695 and set(old) == set(new) and
        {n for n in new if new[n] != old[n]} == {*SOURCES, CORRECTION}, 'Unrelated catalog changes')
    changed = dict(new[CORRECTION]); changed['gaps'] = old[CORRECTION]['gaps']
    require(changed == old[CORRECTION] and new[CORRECTION]['gaps'][1:] == old[CORRECTION]['gaps'][1:] and
        new[CORRECTION]['gaps'][0] == '五项JQ原风格因子历史值/计算口径与平台截尾标准化未齐；合成standardlize加10、winsorize乘2监测桩仅核调用顺序与赋值',
        'Prior correction exceeds reviewed prose-only change')
    require(sum(r['review_status'] == '人工审查完成' for r in new.values()) == 273, 'Catalog review count differs')
    for number, name in enumerate(SOURCES):
        expected = deepcopy(old[name]); expected.update(deepcopy(REVIEWS[name]))
        if old[name]['duplicate_of']:
            expected.update(status='重复版本', next_step='共用实现但保留文件映射；核对 canonical_source 状态')
        require(new[name] == expected and new[name]['bytes_sha256'] == stage.SOURCE_SHA[number],
            'Candidate source values differ from reviewed provenance: '+name)


def validate_sources():
    rows = read(stage.DIRECTORY/'source-map.json')['sources']
    for number, (name, lines) in enumerate(zip(SOURCES, (182, 189, 189), strict=True)):
        source = Path('repo/量化策略源代码')/name
        require(file_sha(regular(source)) == stage.SOURCE_SHA[number], 'Original source bytes differ')
        text = read_source(source)[0]; tree = ast.parse(text)
        definitions = [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)]
        require(len(text.splitlines()) == rows[number]['complete_lines_read'] == lines and
            len(definitions) == 14 and definitions == rows[number]['definition_ast'], 'Complete original definitions differ')
        review = REVIEWS[name]
        require(review['review_status'] == '人工审查完成' and review['implementations'] == [] and
            review['status'] == '待数据' and not review['original_strategy_complete'], 'New review scope differs')


def validate_catalog_code(directory):
    parent = read(stage.PARENT)
    before = Path(parent['accepted_directory'])/'implementation-checked'/FILES[0]
    require(file_sha(regular(before)) == parent['reviewed_implementation_sha256'][FILES[0]],
        'Frozen prior catalog code differs')
    trees = [ast.parse(read_source(path)[0]) for path in (before, Path(FILES[0]))]
    for tree in trees:
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'REVIEWS' for t in n.targets)]
        require(len(assignments) == 1, 'Catalog review assignment differs')
        assignments[0].value = ast.Constant(value=None)
    require(ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False),
        'Catalog implementation changed beyond reviews')


def catalog(root, directory):
    require_root(root); require(directory == DIRECTORY and not (directory/'candidate.json').exists(), 'Candidate already exists or differs')
    stage_evidence(); validate_baseline(directory); ancestry(directory); validate_sources(); validate_catalog_code(directory)
    require(read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Active catalog changed')
    old = validate_catalog_ancestry(directory)
    folder = directory/'catalog-candidate'; folder.mkdir(exist_ok=False)
    target = folder/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
    shutil.copyfile(Path(root)/'catalog/strategies/implementation-evidence.json', target)
    with recovery.network_blocked(): candidate = catalog_strategies(folder, 'repo/量化策略源代码')
    validate_changes(old, read(Path(candidate['output'])/'catalog.json'))
    other = directory/'catalog-offline'; other.mkdir(exist_ok=False)
    target2 = other/'catalog/strategies/implementation-evidence.json'; target2.parent.mkdir(parents=True)
    shutil.copyfile(target, target2)
    with recovery.network_blocked(): duplicate = catalog_strategies(other, 'repo/量化策略源代码')
    require(candidate['catalog_id'] == duplicate['catalog_id'], 'Candidate repeated ID differs')
    names = ('catalog.json', 'summary.json', 'catalog.parquet')
    hashes = {n: file_sha(Path(candidate['output'])/n) for n in names}
    require(hashes == {n: file_sha(Path(duplicate['output'])/n) for n in names}, 'Repeated candidate bytes differ')
    save(directory/'catalog-offline.json', {'status': 'match', 'candidate': candidate, 'duplicate': duplicate,
        'file_sha256': hashes, 'implementation_sha256': implementation(directory)})
    save(directory/'candidate.json', candidate)
    return {'status': 'candidate_only', 'catalog_id': candidate['catalog_id'], 'active_catalog_changed': False}


def compare_table(frame, filename):
    regular(filename)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(filename), check_exact=True)


def worker(directory, output):
    require(directory == DIRECTORY and output in ('replay', 'final-replay') and sys.dont_write_bytecode and
        os.environ.get('OBSERVE_BATCH94_WORKER') == '1', 'Fresh bytecode-disabled worker required')
    stage_evidence(); reference.validate_index('data'); validate_baseline(directory)
    before = implementation(directory); target = directory/output
    require(not target.exists(), 'Fresh replay output already exists'); target.mkdir()
    with recovery.network_blocked():
        rsrs, rsummary = stage.rsrs_components(stage.DIRECTORY)
        compare_table(rsrs, stage.DIRECTORY/'rsrs-reference.parquet')
        for number in (1, 2):
            other, summary = stage.rsrs_components(stage.DIRECTORY, number)
            pd.testing.assert_frame_equal(other, rsrs, check_exact=True)
            require(summary == rsummary, 'Other source RSRS differs')
        fft, fsummary = stage.fft_components(stage.DIRECTORY)
        compare_table(fft, stage.DIRECTORY/'fft-native-reference.parquet')
        require({'RSRS': rsummary, 'FFT': fsummary} == {k: read(stage.DIRECTORY/'component-research.json')[k]
            for k in ('RSRS', 'FFT')}, 'FFT/RSRS summary differs')
        kalman, modules = reference.load_fork(recovery.DIRECTORY)
        table, ksummary = reference.kalman_components(kalman)
        capture, csummary = reference.literal_rsrs_capture()
        compare_table(table, recovery.DIRECTORY/'kalman-native-reference.parquet')
        compare_table(capture, recovery.DIRECTORY/'literal-rsrs-capture.parquet')
        previous_result = read(recovery.DIRECTORY/'component-research.json')
        require(ksummary == previous_result['Kalman'] and csummary == previous_result['actual_RSRS'] and
            modules == previous_result['loaded_modules'], 'Cached-fork reference summary differs')
        diagnostic = callbacks.diagnostics()
        require(diagnostic == read(callbacks.DIRECTORY/'diagnostics.json'), 'Actual callbacks differ')
    for frame, name in zip((rsrs, fft, table, capture), REPLAY_NAMES, strict=False):
        with (target/name).open('xb') as out: frame.to_parquet(out, index=False)
    save(target/'callbacks.json', diagnostic)
    require(before == implementation(directory), 'Replay code changed')
    result = {'status': 'match', 'differences': 0, 'RSRS_rows': len(rsrs), 'FFT_rows': len(fft),
        'Kalman_windows': len(table), 'actual_RSRS_rows': len(capture), 'callback_cases': diagnostic['case_count'],
        'network_forbidden': True, 'not_a_trading_backtest': True, 'new_trading_backtests': 0,
        'implementation_sha256': before, 'file_sha256': {n: file_sha(target/n) for n in REPLAY_NAMES}}
    require((len(rsrs), len(fft), len(table), len(capture), diagnostic['case_count']) == (4662, 10467, 11157, 4662, 102),
        'Fresh replay scope differs')
    save(directory/(output+'.json'), result)
    return result


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for n in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'): env.pop(n, None)
    return env


def replay(directory, output='replay'):
    require(directory == DIRECTORY and not (directory/(output+'-command.json')).exists(), 'Replay command already archived')
    command = [sys.executable, '-B', '-m', 'scripts.verify_strategy_batch94', 'worker', '--output', output]
    log = directory/(output+'.log'); env = environment(); env['OBSERVE_BATCH94_WORKER'] = '1'; env.pop('PYTHONPATH', None)
    with log.open('x') as stream: result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    save(directory/(output+'-command.json'), {'command': command, 'returncode': result.returncode,
        'log': str(log), 'sha256': file_sha(log)})
    require(result.returncode == 0, 'Fresh replay failed; preserve log')
    return validate_replay(directory, output)


def validate_replay(directory, output='replay'):
    result = read(directory/(output+'.json')); command = read(directory/(output+'-command.json'))
    require(command['command'] == [sys.executable, '-B', '-m', 'scripts.verify_strategy_batch94', 'worker', '--output', output]
        and command['returncode'] == 0 and file_sha(regular(command['log'])) == command['sha256'], 'Fresh replay command differs')
    require(result['status'] == 'match' and result['differences'] == result['new_trading_backtests'] == 0 and
        result['network_forbidden'] and result['not_a_trading_backtest'] and
        (result['RSRS_rows'], result['FFT_rows'], result['Kalman_windows'], result['actual_RSRS_rows'], result['callback_cases'])
        == (4662, 10467, 11157, 4662, 102) and result['implementation_sha256'] == implementation(directory) and
        set(result['file_sha256']) == set(REPLAY_NAMES), 'Replay evidence scope differs')
    require(all(file_sha(regular(directory/output/n)) == sha for n, sha in result['file_sha256'].items()),
        'Fresh replay bytes changed')
    return result


def test_command():
    return [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TEST_FILES, '--deselect', EXCLUDED]


def expected_tests():
    receipt = read('docs/handoff/2026-10-10-batch94-callback-stage.json')
    path = callbacks.DIRECTORY/'related-tests.xml'
    require(file_sha(path) == receipt['payload_evidence_sha256'][str(path)], 'Test baseline changed')
    cases = [(t.get('classname'), t.get('name')) for t in ET.parse(path).getroot().iter('testcase')]
    require(len(cases) == len(set(cases)) == 86, 'Test baseline identity differs')
    return [*cases, *(('tests.unit.test_batch94_acceptance', n) for n in NEW_CASES)]


def coverage(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    identities = [(t.get('classname'), t.get('name')) for t in cases]
    require(len(identities) == len(set(identities)) and set(identities) == set(expected_tests()) and
        all(not any(t.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for t in cases),
        'Exact JUnit coverage differs')
    return {'passed': len(cases), 'skipped': 0, 'files': dict(Counter(c for c, _ in identities))}


def check_commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES,
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        test_command(), ['git', 'diff', '--check']]


def validate_commands(commands):
    require([r['command'] for r in commands] == check_commands(), 'Exact required commands differ')
    require(all(r['returncode'] == 0 and file_sha(regular(r['log'])) == r['sha256'] for r in commands), 'Check log changed')


def checks(directory):
    require(directory == DIRECTORY and not (directory/'checks').exists(), 'Checks already archived')
    validate_baseline(directory); validate_replay(directory); before = implementation(directory)
    folder = directory/'checks'; folder.mkdir(); commands = []
    for i, command in enumerate(check_commands()):
        log = folder/f'{i}.log'
        actual = [*command, '--junitxml='+str(folder/'tests.xml')] if i == 1 else command
        with log.open('x') as stream: result = subprocess.run(actual, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
        commands.append({'command': command, 'returncode': result.returncode, 'log': str(log), 'sha256': file_sha(log)})
        require(result.returncode == 0, 'Required checks failed; preserve log')
    cov = coverage(folder/'tests.xml'); validate_commands(commands)
    require(before == implementation(directory), 'Tested code changed')
    for name in FILES:
        target = directory/'implementation-checked'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': commands,
        'implementation_sha256': before, 'coverage': cov,
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return cov


def isolate(directory):
    require(directory == DIRECTORY and not (directory/'isolated-files.json').exists(), 'Isolation already archived')
    validate_baseline(directory); candidate = implementation(directory)
    destination = Path(tempfile.mkdtemp(prefix='batch94-acceptance-isolated-', dir='/tmp'))
    copied = {}
    def copy(name, sha):
        path = regular(name); require(not path.is_absolute() and '..' not in path.parts, 'Unsafe isolation path')
        target = destination/path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        require(file_sha(path) == sha == file_sha(target), 'Isolated copy changed'); copied[str(path)] = sha
    for row in read(directory/'isolation-baseline.json')['code']:
        copy(row['path'], candidate.get(row['path'], row['sha256']))
    for name, sha in candidate.items(): copy(name, sha)
    for number in range(3): copy(str(stage.READINESS/f'{number}.source'), stage.SOURCE_SHA[number]); copy(str(stage.DIRECTORY/'sources'/f'{number}.source'), stage.SOURCE_SHA[number])
    path = callbacks.DIRECTORY/'related-tests.xml'; copy(str(path), file_sha(path))
    name = 'docs/handoff/2026-10-10-batch94-callback-stage.json'; copy(name, PINS[name])
    copy(str(BASE_CATALOG), BASE_CATALOG_SHA)
    parent = read(stage.PARENT); copy(str(stage.PARENT), file_sha(stage.PARENT))
    for key in ('receipt', 'checked_state', 'final_binding', 'formal_report', 'review_completion'):
        name = parent[key+'_file'] if key != 'formal_report' else parent[key]
        copy(name, parent[key+'_sha256'])
    for name, sha in parent['reviewed_implementation_sha256'].items():
        if name == FILES[0]: copy(str(Path(parent['accepted_directory'])/'implementation-checked'/name), sha)
    copy(str(stage.DIRECTORY/'parent-binding.json'), file_sha(stage.DIRECTORY/'parent-binding.json'))
    save(directory/'isolated-files.json', {'destination': str(destination), 'files': copied,
        'candidate_implementation_sha256': candidate, 'method': 'Verified preceding metadata code plus exact candidate, no committed-HEAD claim'})
    log = directory/'isolated-tests.log'; junit = directory/'isolated-tests.xml'
    env = environment(); env['PYTHONPATH'] = str(destination/'src')+os.pathsep+str(destination)
    with log.open('x') as stream:
        result = subprocess.run([*test_command(), '--junitxml='+str(junit.resolve())], cwd=destination,
            env=env, stdout=stream, stderr=subprocess.STDOUT)
    cov = coverage(junit)
    require(result.returncode == 0 and candidate == implementation(directory), 'Isolated tests failed or candidate changed')
    save(directory/'isolated-verification.json', {'status': 'passed', 'command': test_command(), 'returncode': result.returncode,
        'coverage': cov, 'log_sha256': file_sha(log), 'junit_sha256': file_sha(junit),
        'manifest_sha256': file_sha(directory/'isolated-files.json'), 'implementation_sha256': candidate})
    return cov


def validate_isolation(directory):
    proof = read(directory/'isolated-verification.json'); copied = read(directory/'isolated-files.json')
    require(proof['status'] == 'passed' and proof['returncode'] == 0 and proof['command'] == test_command() and
        proof['implementation_sha256'] == copied['candidate_implementation_sha256'] == implementation(directory) and
        proof['log_sha256'] == file_sha(directory/'isolated-tests.log') and
        proof['junit_sha256'] == file_sha(directory/'isolated-tests.xml') and
        proof['manifest_sha256'] == file_sha(directory/'isolated-files.json') and
        coverage(directory/'isolated-tests.xml') == proof['coverage'], 'Isolation proof changed')
    require(all(copied['files'].get(n) == sha for n, sha in implementation(directory).items()) and
        all(file_sha(Path(copied['destination'])/n) == sha for n, sha in copied['files'].items()), 'Copied candidate changed')


def validate_runs(directory):
    baseline = read(directory/'baseline.json')
    old = {n for n in baseline['controls'] if Path(n).parts[0] == 'runs' and Path(n).name == 'config.json'}
    now = {p.relative_to('data').as_posix() for p in Path('data/runs').rglob('config.json')}
    require(bool(old) and old == now, 'Run inventory changed')


def validate_candidate(directory):
    candidate = read(directory/'candidate.json'); proof = read(directory/'catalog-offline.json')
    require(candidate == proof['candidate'] and proof['status'] == 'match' and
        candidate['catalog_id'] == proof['duplicate']['catalog_id'] and set(proof['file_sha256']) ==
        {'catalog.json', 'summary.json', 'catalog.parquet'} and proof['implementation_sha256'] == implementation(directory),
        'Candidate proof changed')
    for name, sha in proof['file_sha256'].items():
        require(file_sha(regular(Path(candidate['output'])/name)) == sha ==
            file_sha(regular(Path(proof['duplicate']['output'])/name)), 'Candidate bytes changed')
    old = validate_catalog_ancestry(directory)
    validate_changes(old, read(Path(candidate['output'])/'catalog.json'))
    validate_catalog_artifacts(directory, candidate, proof['duplicate'])
    validate_sources(); validate_catalog_code(directory)
    return candidate


def validate_catalog_ancestry(directory):
    before = read(regular(directory/'catalog-before.json'))
    require(before == {'catalog_id': BASE_CATALOG.parent.name, 'directory': str(BASE_CATALOG.parent),
        'sha256': BASE_CATALOG_SHA} and file_sha(regular(BASE_CATALOG)) == BASE_CATALOG_SHA,
        'Frozen parent catalog provenance differs')
    return read(BASE_CATALOG)


def validate_catalog_artifacts(directory, candidate, duplicate):
    rows = read(Path(candidate['output'])/'catalog.json')
    identifier = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    require(isinstance(candidate['catalog_id'], str) and re.fullmatch(r'[0-9a-f]{20}', candidate['catalog_id'])
        and candidate['catalog_id'] == identifier, 'Catalog ID is not content-derived')
    summary = {'catalog_version': CATALOG_VERSION, 'catalog_id': identifier,
        'source_root': str(Path('repo/量化策略源代码').resolve()), 'files': len(rows),
        'distinct_contents': len({r['content_sha256'] for r in rows}),
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'suffix_counts': dict(Counter(Path(r['path']).suffix.lower() for r in rows)),
        'encoding_counts': dict(Counter(r['encoding'] for r in rows)),
        'api_file_counts': {a: sum(a in r['apis'] for r in rows) for a in API_NAMES},
        'reviewed_sources': [r['strategy_id'] for r in rows if r['review_status'] == '人工审查完成'],
        'caveat': '未审查文件的依赖/分类为静态候选；近似变体不会自动合并，不代表全量策略已实现'}
    expected_table = pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        for k, v in row.items()} for row in rows])
    for descriptor, role in ((candidate, 'catalog-candidate'), (duplicate, 'catalog-offline')):
        expected_output = directory/role/'catalog/strategies'/identifier
        require(descriptor == {**summary, 'output': str(expected_output)}, 'Candidate summary or output path differs')
        require(read(regular(expected_output/'summary.json')) == summary, 'Catalog summary values differ')
        pd.testing.assert_frame_equal(expected_table, pd.read_parquet(regular(expected_output/'catalog.parquet')),
            check_exact=True)


def validate_final_bindings(directory, checked, checked_sha, review_sha):
    require(file_sha(regular(directory/'checked-state.json')) == checked_sha and
        file_sha(regular(directory/'review-completion.json')) == review_sha and
        all(file_sha(regular(n)) == sha for n, sha in checked['evidence_sha256'].items()),
        'Checked evidence changed during final replay')


def publish_candidate_locked(root, directory, candidate, expected):
    """Caller holds DATA_WRITER; publish only the exact previously checked bytes."""
    require(validate_candidate(directory) == candidate, 'Candidate identity changed')
    latest = Path(root)/'catalog/strategies/latest.json'; before = read(directory/'catalog-before.json')
    require(read(latest) == before, 'Active catalog changed before publication')
    require(expected == read(directory/'catalog-offline.json')['file_sha256'], 'Frozen catalog bytes changed')
    source = Path(candidate['output']); dest = Path(root)/'catalog/strategies'/candidate['catalog_id']
    if dest.exists():
        require(all(file_sha(regular(dest/n)) == sha for n, sha in expected.items()), 'Existing catalog differs; no overwrite')
    else:
        temporary = Path(tempfile.mkdtemp(prefix='.batch94-catalog-', dir=dest.parent))
        for name, sha in expected.items():
            require(file_sha(regular(source/name)) == sha, 'Validated source catalog changed')
            shutil.copyfile(source/name, temporary/name)
            require(file_sha(temporary/name) == sha, 'Validated catalog copy changed')
        os.rename(temporary, dest)
    require(read(latest) == before and all(file_sha(regular(dest/n)) == sha == file_sha(regular(source/n))
        for n, sha in expected.items()), 'Catalog or navigation changed before switch')
    write_json(latest, {'catalog_id': candidate['catalog_id'], 'directory': str(dest), 'sha256': expected['catalog.json']})
    return {**candidate, 'output': str(dest)}


def validate_progress(progress, published, hashes):
    require(progress['catalog_id'] == published['catalog_id'] and progress['catalog_sha256'] == hashes['catalog.json'] and
        read('data/catalog/strategies/latest.json') == {'catalog_id': published['catalog_id'],
            'directory': published['output'], 'sha256': hashes['catalog.json']} and
        file_sha(Path(published['output'])/'catalog.json') == hashes['catalog.json'], 'Final catalog identity differs')


def finish(root, directory):
    require_root(root); require(directory == DIRECTORY and not RECEIPT.exists(), 'Acceptance already archived')
    stage_evidence(); reference.validate_index(root); validate_baseline(directory); validate_isolation(directory)
    checked, checked_sha = read_bound(directory/'checked-state.json')
    require(checked['status'] == 'passed' and CORE <= {Path(n).relative_to(directory).as_posix() for n in checked['evidence_sha256']},
        'Checked evidence scope differs')
    implementation(directory, checked['implementation_sha256']); validate_commands(checked['commands'])
    coverage(directory/'checks/tests.xml'); validate_replay(directory); candidate = validate_candidate(directory)
    require(all(file_sha(regular(n)) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    review, review_sha = read_bound(directory/'review-completion.json')
    require(review['status'] == 'completed' and review['implementation_sha256'] == implementation(directory) and
        {r['scope'] for r in review['reviewers']} == {'security', 'architecture', 'assumptions', 'composition', 'cascade', 'abuse'} and
        len(review['reviewers']) == 6 and all(r['status'] == 'returned' and not r['findings'] for r in review['reviewers']),
        'Final independent review incomplete')
    frozen_catalog_hashes = dict(read(directory/'catalog-offline.json')['file_sha256'])
    # A fresh child performs actual mathematics now. A self-declared old status cannot authorize publication.
    fresh = replay(directory, 'final-replay')
    require({k: v for k, v in fresh.items() if k != 'file_sha256'} ==
        {k: v for k, v in read(directory/'replay.json').items() if k != 'file_sha256'}, 'Final actual replay differs')
    with operation_lock(root, DATA_WRITER):
        validate_final_bindings(directory, checked, checked_sha, review_sha)
        stage_evidence(); reference.validate_index(root); implementation(directory, checked['implementation_sha256'])
        validate_runs(directory); require(ancestry(directory) == read(directory/'stage-ancestry.json'), 'Ancestry changed')
        require(Store(root).published() == read(directory/'baseline.json')['published'], 'Data publication changed')
        protected = protect(root, directory)
        validate_final_bindings(directory, checked, checked_sha, review_sha)
        published = publish_candidate_locked(root, directory, candidate, frozen_catalog_hashes)
        progress = archive(root, 'Batch94 three ETF source research accepted;long native FFT/Kalman andRSRS,102 synthetic callbacks;no trading')
        require((progress['manually_reviewed'], progress['locally_backtested_sources'], progress['local_strategy_runs']) == (273, 12, 75),
            'Final totals differ')
        validate_progress(progress, published, read(directory/'catalog-offline.json')['file_sha256'])
        validate_final_bindings(directory, checked, checked_sha, review_sha)
        implementation(directory, checked['implementation_sha256'])
        save(RECEIPT, {'status': 'ok', 'progress': progress, 'stage_receipts': PINS,
            'checked_state': str(directory/'checked-state.json'), 'checked_state_sha256': checked_sha,
            'review_completion': str(directory/'review-completion.json'), 'review_completion_sha256': review_sha,
            'implementation_sha256': checked['implementation_sha256'], 'catalog': published, 'fresh_replay': fresh,
            'protection': protected, 'newly_reviewed': 3, 'formal_source_research_acceptance': True,
            'formal_trading_acceptance': False, 'new_trading_backtests': 0, 'original_strategy_complete': False,
            'source_sha256': list(stage.SOURCE_SHA), 'historical_prose_correction_only': CORRECTION,
            'environment_excluded_test': EXCLUDED, 'not_a_trading_backtest': True,
            'restore_required': ['159928 plus all four original fund historical identity/events/adjustment/state/reference prices/fees',
                'Original platform library version, two open schedules andpositions/order feedback; concrete repairs need separate approval']})
        save(directory/'final-binding.json', {'receipt_sha256': file_sha(RECEIPT), 'progress': progress,
            'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 273, 'new_trading_backtests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'catalog', 'replay', 'worker', 'checks', 'isolate', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--output', choices=('replay', 'final-replay'), default='replay')
    args = parser.parse_args(); require_root(args.root)
    if args.action == 'prepare': result = prepare(args.root, DIRECTORY)
    elif args.action == 'catalog': result = catalog(args.root, DIRECTORY)
    elif args.action == 'worker': result = worker(DIRECTORY, args.output)
    elif args.action == 'replay': result = replay(DIRECTORY, args.output)
    elif args.action == 'checks': result = checks(DIRECTORY)
    elif args.action == 'isolate': result = isolate(DIRECTORY)
    else: result = finish(args.root, DIRECTORY)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
