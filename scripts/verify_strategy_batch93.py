"""Accept two financial strategy sources; final-value inventory never admits trades."""
import ast
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

from observe.data.store import Store
from observe.data.locks import DATA_WRITER, operation_lock
from observe.strategy_catalog import read_source
from observe.runs import file_sha
from observe.runs import write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import review_strategy_batch93 as stage
from scripts import verify_strategy_batch92 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch93/20261010-ffscore-threefactor-acceptance')
STAGE_CHECKPOINT = Path('docs/handoff/2026-10-10-batch93-operands-stage.json')
STAGE_CHECKPOINT_SHA = '832bdec01559405eede5102294667e955fef7dcf2f1bfffa93ef3e28a4b76cd1'
ORIGINAL_STAGE_BINDING = stage.parent_binding
RECEIPT = Path('docs/handoff/2026-10-10-batch93-verification.json')
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch93.py',
    'tests/unit/test_batch93_acceptance.py', *stage.FILES)
NEW_CASES = ('test_frozen_stage_scope_and_nontrading_identity',
    'test_changed_stage_evidence_is_rejected[financial-operands.parquet]',
    'test_changed_stage_evidence_is_rejected[operand-inventory.json]',
    'test_changed_stage_evidence_is_rejected[diagnostics.json]',
    'test_changed_stage_scope_is_rejected', 'test_adapter_rejects_unexpected_directory',
    'test_adapter_preserves_upstream_evidence[baseline.json]',
    'test_adapter_preserves_upstream_evidence[input-binding.json]',
    'test_catalog_review_preserves_original_rules_without_trades',
    'test_full_junit_rejects_subset_duplicates_and_skips',
    'test_selection_environment_is_removed', 'test_unrelated_baseline_changes_are_rejected',
    'test_repeated_catalog_is_rejected_before_writes', 'test_original_binding_capture_prevents_recursion',
    'test_required_command_contract_rejects_empty_or_substituted', 'test_stage_control_mutation_before_acceptance_is_rejected',
    'test_new_run_inventory_is_rejected_before_acceptance_publication', 'test_isolation_candidate_must_match_tested_manifest',
    'test_independent_numeric_missing_and_nonfinite_boundaries', 'test_partition_manifest_set_and_identity_are_exact',
    'test_alternate_root_rejected_before_any_reads',
    'test_candidate_publication_copies_validated_bytes_without_live_regeneration',
    'test_candidate_publication_rejects_concurrent_navigation_change',
    'test_progress_catalog_identity_rejects_same_count_competitor')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'isolation-baseline.json', 'reference-replay.json', 'source-reviews/review.json',
    'offline-catalog.json', 'offline-verification.json', 'isolated-verification.json', 'stage-protection-baseline.json',
    'acceptance-development-binding.json', 'final-catalog-development-binding.json'}


def stage_evidence():
    require(file_sha(STAGE_CHECKPOINT) == STAGE_CHECKPOINT_SHA, 'Stage checkpoint changed')
    doc = read(STAGE_CHECKPOINT)
    require(doc['status'] == 'stage_checks_passed' and
        not doc['formal_source_research_acceptance'] and not doc['formal_trading_acceptance'] and
        doc['source_sha256'] == list(stage.SOURCE_SHA) and doc['new_requests'] == doc['new_trading_backtests'] == 0 and
        doc['projected_final_financial_rows'] == 501353 and doc['strict_usable_rows'] == 0 and
        doc['tests']['passed'] == 45 and doc['offline']['cases'] == 30 and
        doc['offline']['independent_metric_cells'] == 168 and doc['offline']['differences'] == 0,
        'Stage scope changed')
    require(all(file_sha(n) == sha for n, sha in doc['implementation_sha256'].items()), 'Stage implementation changed')
    require(all(file_sha(n) == sha for n, sha in doc['evidence_sha256'].items()), 'Stage evidence changed')
    require(file_sha(doc['stage_report']) == doc['stage_report_sha256'], 'Stage report changed')
    require(file_sha(stage.READINESS/'source-readiness.json') ==
        'b5797e5927a630710e3790710f90de5da66beab2691ad596eb1b1f9158d931ff', 'Readiness changed')
    failures = read(stage.DIRECTORY/'development-failure-binding.json')
    require(all(file_sha(n) == sha for n, sha in failures.items()), 'Initial failure evidence changed')
    review = read(stage.DIRECTORY/'original-definition-review.json')
    for n, lines in enumerate((367, 135)):
        tree = stage.source_tree(n, stage.DIRECTORY); stage.source_tree(n)
        row = review['sources'][n]
        require(row['path'] == stage.SOURCES[n] and row['sha256'] == stage.SOURCE_SHA[n] and
            row['lines'] == len(read_source(stage.DIRECTORY/f'{n}.source')[0].splitlines()) == lines and
            row['definition_ast'] == [ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))], 'Original definition scope changed')
    require(not review['top_level_executed'] and not review['formal_source_acceptance'] and
        review['new_trading_backtests'] == 0, 'Original definition scope changed')
    return doc


def binding(directory=None):
    stage_evidence()
    result = {'stage_checkpoint_file': str(STAGE_CHECKPOINT), 'stage_checkpoint_sha256': STAGE_CHECKPOINT_SHA,
        'snapshot': stage.SNAPSHOT, 'source_sha256': list(stage.SOURCE_SHA), 'not_a_backtest': True,
        'new_requests': 0, 'new_trading_backtests': 0}
    if directory is not None: require(read(directory/'input-binding.json') == result, 'Acceptance input binding changed')
    return result


def preflight(root, directory):
    require_root(root)
    bound = binding(); ORIGINAL_STAGE_BINDING()
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    index = read('data/metadata/index.json'); baseline = read(index['implementation_files'])
    require(all(file_sha(r['path']) == r['sha256'] for r in baseline), 'Isolation baseline code changed')
    require(directory == DIRECTORY and not directory.exists(), 'Acceptance directory already exists or differs')
    checkpoint(root, directory, 'Batch93 independent operand/callback acceptance starts; no trades')
    save(directory/'input-binding.json', bound)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/implementation-evidence.json', directory/'implementation-evidence.before.json')
    inventory = subprocess.run(['git', 'status', '--short', '--branch', '-uall'],
        check=True, capture_output=True, text=True)
    (directory/'git-status-before.txt').write_text(inventory.stdout)
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
    require_root(_root)
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
    # Only the historical catalog is adapted; the original parent callable is captured before any patch.
    with patch.object(stage, 'file_sha', adapted):
        return ORIGINAL_STAGE_BINDING()


def validate_partition_manifest(bound, state):
    expected = {}
    for table in (*stage.FINANCIAL_TABLES, 'financial_fields', 'financial_availability'):
        for entry in state['tables'][table].values():
            expected[str(Path('data')/entry['file'])] = (table, entry['sha'], entry['rows'])
    actual = {p['file']: (p['table'], p['snapshot_logical_sha'], p['rows']) for p in bound['partitions']}
    require(len(bound['partitions']) == len(actual) == len(expected) == 147 and actual == expected,
        'Financial partition set or identity changed')


def independent_number(series):
    # Deliberately independent of parse_number; preserve the explicit vendor missing tokens.
    missing_tokens = {'', '.', 'NULL', 'null', 'NA', 'N/A', 'nan', 'NaN', '--'}
    values = []; malformed = []
    for raw in series:
        missing = pd.isna(raw) or str(raw).strip() in missing_tokens
        try: value = float(str(raw).strip()) if not missing else np.nan
        except (ValueError, TypeError): value = np.nan
        finite = np.isfinite(value)
        values.append(value if finite else np.nan); malformed.append(not missing and not finite)
    return np.array(values), np.array(malformed)


def independent_inventory():
    bound = read(stage.DIRECTORY/'input-binding.json')
    snapshot = Path('data/snapshots')/f'{stage.SNAPSHOT}.json'
    require(file_sha(snapshot) == stage.SNAPSHOT_SHA and file_sha(stage.IMPORT) == stage.IMPORT_SHA,
        'Financial snapshot/import changed')
    state = read(snapshot); validate_partition_manifest(bound, state)
    byte_manifest = read(stage.IMPORT)['partition_bytes']; by_table = {}
    for row in bound['partitions']:
        path = Path(row['file']); before = path.stat(); sha = file_sha(path); after = path.stat()
        require(sha == row['bytes_sha256'] == byte_manifest[path.relative_to('data').as_posix()] and
            before.st_size == after.st_size == row['size'] and before.st_mtime_ns == after.st_mtime_ns,
            'Financial partition bytes changed')
        by_table.setdefault(row['table'], []).append(path)
    def load(table, columns=None):
        entries = state['tables'][table]
        return pd.concat([pd.read_parquet(Path('data')/entry['file'], columns=columns)
            for _, entry in sorted(entries.items())], ignore_index=True)
    parts = []
    for table in stage.FINANCIAL_TABLES:
        part = load(table, bound['projection_columns']); part.insert(0, 'archive_table', table); parts.append(part)
    frame = pd.concat(parts, ignore_index=True); fields = load('financial_fields')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(stage.DIRECTORY/'financial-operands.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(fields, pd.read_parquet(stage.DIRECTORY/'financial-fields.parquet'), check_exact=True)
    availability = load('financial_availability', ['strict_usable', 'version_evidence'])
    require(len(availability) == 1460899 and not availability.strict_usable.any(), 'Strict coverage changed; replan')
    inventory = read(stage.DIRECTORY/'operand-inventory.json'); coverage = []
    for table in stage.FINANCIAL_TABLES:
        data = frame.loc[frame.archive_table.eq(table)]
        candidates = {**{key: (label,) for key, label in stage.AMOUNTS.items()}, **stage.RATIO_CANDIDATES}
        for key, labels in candidates.items():
            for label in labels:
                numbers, malformed = independent_number(data[label]); valid = np.isfinite(numbers)
                dictionary = fields.loc[fields.table.eq(table) & fields.column.eq(label)]
                require(len(dictionary) == 1, 'Ambiguous dictionary operand')
                coverage.append({'table': table, 'platform_dependency_candidate': key, 'column': label,
                    'rows': len(data), 'finite_values': int(valid.sum()), 'malformed_nonmissing': int(malformed.sum()),
                    'zeros': int((numbers == 0).sum()), 'positive_values': int((numbers > 0).sum()),
                    'first_report_period': data.loc[valid, 'report_period'].min(),
                    'last_report_period': data.loc[valid, 'report_period'].max(),
                    'dictionary': dictionary.iloc[0].to_dict(), 'platform_equivalence': 'not_proven', 'strict_usable_values': 0})
    require(stage.encoded(coverage) == stage.encoded(inventory['coverage']) and len(coverage) == 40,
        'Independent coverage differs')
    require(inventory['projection_rows'] == len(frame) == 501353 and inventory['field_rows'] == len(fields) == 1911 and
        inventory['table_rows'] == frame.archive_table.value_counts().to_dict() and
        inventory['matched_instrument_rows'] == int(frame.instrument.notna().sum()) == 469762 and
        inventory['availability_rows'] == len(availability) and inventory['strict_usable_rows'] == 0 and
        inventory['version_evidence'] == sorted(availability.version_evidence.unique()) and
        inventory['registered_tables'] == sorted(state['tables']) and not inventory['platform_style_factor_table_registered'],
        'Inventory scope changed')
    return {'status': 'match', 'differences': 0, 'rows': len(frame), 'coverage_records': len(coverage),
        'financial_partitions_fresh_byte_verified': 147, 'financial_partition_bytes': sum(r['size'] for r in bound['partitions']),
        'strict_usable_rows': 0, 'availability_rows': len(availability),
        'independent_parser': 'Python float with explicit missing/nonfinite policy; no stage parse_number',
        'projection': 'Direct raw Parquet reads; no Store loader or stage operand_inventory'}


def replay_values(adapt_binding):
    if adapt_binding: frozen_operand_binding(stage.DIRECTORY)
    else: ORIGINAL_STAGE_BINDING()
    inventory = independent_inventory()
    with stage.no_network(): diagnostics = stage.diagnostics(stage.DIRECTORY)
    require(stage.encoded(diagnostics) == (stage.DIRECTORY/'diagnostics.json').read_bytes(), 'Original callback replay differs')
    require(diagnostics['cases'] == 30 and diagnostics['independent_metric_cells'] == 168 and
        diagnostics['new_trading_backtests'] == diagnostics['real_platform_factors'] == 0, 'Callback scope changed')
    return {'status': 'match', 'differences': 0, 'rows': 501353, 'inventory': inventory,
        'cases': 30, 'independent_metric_cells': 168, 'operand_binding_adapted': adapt_binding,
        'socket_HTTP_disabled': True, 'not_a_backtest': True, 'new_trading_backtests': 0}


def replay(root, directory):
    require_root(root)
    require(directory == DIRECTORY and not (directory/'reference-replay.json').exists(), 'Replay already archived or unexpected directory')
    binding(directory)
    require(file_sha('src/observe/strategy_catalog.py') == file_sha(directory/'strategy_catalog.before.py') and
        read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Replay required before catalog edit')
    result = network_disabled(lambda: replay_values(False)); save(directory/'reference-replay.json', result)
    return {'status': 'match', 'rows': result['rows'], 'differences': 0}


def require_root(root):
    require(Path(root).resolve() == Path('data').resolve(), 'Only canonical data root is supported')


def publish_candidate(root, directory, candidate):
    # Freeze already validated bytes. Never regenerate from a second read of live sources.
    with operation_lock(root, DATA_WRITER):
        latest = Path(root)/'catalog/strategies/latest.json'
        require(read(latest) == read(directory/'catalog-before.json'), 'Active catalog changed before candidate publication')
        source = Path(candidate['output']); dest = Path(root)/'catalog/strategies'/candidate['catalog_id']
        names = ('catalog.json', 'summary.json', 'catalog.parquet'); frozen = {n: file_sha(source/n) for n in names}
        if dest.exists():
            require(all(file_sha(dest/n) == sha for n, sha in frozen.items()), 'Existing catalog differs; never overwrite')
        else:
            temporary = Path(tempfile.mkdtemp(prefix='.batch93-catalog-', dir=dest.parent))
            for n in names: shutil.copyfile(source/n, temporary/n)
            require(all(file_sha(temporary/n) == sha == file_sha(source/n) for n, sha in frozen.items()),
                'Validated catalog copy changed')
            os.rename(temporary, dest)
        require(read(latest) == read(directory/'catalog-before.json') and
            all(file_sha(dest/n) == sha == file_sha(source/n) for n, sha in frozen.items()),
            'Catalog/navigation changed before atomic switch')
        write_json(latest, {'catalog_id': candidate['catalog_id'], 'directory': str(dest), 'sha256': frozen['catalog.json']})
        return {**candidate, 'output': str(dest)}


def catalog(root, directory):
    require_root(root)
    require(directory == DIRECTORY, 'Unexpected acceptance directory')
    require(not (directory/'source-reviews').exists(), 'Catalog phase already archived')
    binding(directory); require(read(directory/'reference-replay.json')['status'] == 'match', 'Original replay missing')
    require(stage_ancestry(directory) == read(directory/'stage-protection-baseline.json'), 'Stage protection proof changed')
    require(read(Path(root)/'catalog/strategies/latest.json') == read(directory/'catalog-before.json'), 'Active catalog changed')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'; old = {r['path']: r for r in read(prior)}
    require(all(name in REVIEWS and old[name]['review_status'] != '人工审查完成' for name in stage.SOURCES),
        'New source review missing')
    candidate_root = directory/'catalog-candidate'; candidate_root.mkdir(exist_ok=False)
    evidence = Path(root)/'catalog/strategies/implementation-evidence.json'
    require(file_sha(evidence) == file_sha(directory/'implementation-evidence.before.json'), 'Implementation evidence changed')
    target = candidate_root/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
    shutil.copyfile(evidence, target)
    candidate = catalog_strategies(candidate_root, 'repo/量化策略源代码')
    records = read(Path(candidate['output'])/'catalog.json')
    require(len(records) == len(old) == 695 and {r['path'] for r in records} == set(old) and
        {r['path'] for r in records if r != old[r['path']]} == set(stage.SOURCES), 'Unrelated catalog changes')
    now = publish_candidate(root, directory, candidate)
    require(all(file_sha(Path(now['output'])/n) == file_sha(Path(candidate['output'])/n)
        for n in ('catalog.json', 'summary.json', 'catalog.parquet')), 'Validated catalog publication differs')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    save(folder/'review.json', {'snapshot': stage.SNAPSHOT, 'catalog': now, 'not_a_backtest': True,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'sources': [{'source_path': str(Path('repo/量化策略源代码')/name), 'source_copy': str(stage.DIRECTORY/f'{n}.source'),
            'source_sha256': stage.SOURCE_SHA[n], 'complete_lines_read': lines, 'review': REVIEWS[name]}
            for n, (name, lines) in enumerate(zip(stage.SOURCES, (367, 135), strict=True))]})
    return archive(root, 'Batch93 two source rule/catalog registered; formal research checks pending; no trades')


def validate_sources(directory):
    require(directory == DIRECTORY, 'Unexpected acceptance directory')
    binding(directory); frozen_operand_binding(stage.DIRECTORY)
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == stage.SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source review binding changed')
    for n, (name, lines) in enumerate(zip(stage.SOURCES, (367, 135), strict=True)):
        row = doc['sources'][n]
        require(row['source_path'] == str(Path('repo/量化策略源代码')/name) and
            row['source_copy'] == str(stage.DIRECTORY/f'{n}.source') and row['review'] == REVIEWS[name] and
            row['source_sha256'] == stage.SOURCE_SHA[n] and row['complete_lines_read'] == lines and
            row['review']['implementations'] == [] and not row['review']['original_strategy_complete'] and
            row['review']['status'] == '待数据', 'Source review identity changed')


def offline(root, directory):
    require_root(root)
    require(directory == DIRECTORY and not (directory/'offline-verification.json').exists(), 'Offline phase already archived or unexpected directory')
    validate_sources(directory)
    result = network_disabled(lambda: replay_values(True))
    require({**result, 'operand_binding_adapted': False} == read(directory/'reference-replay.json'), 'Pre/post catalog replay differs')
    off = network_disabled(lambda: offline_catalog(root, directory)); save(directory/'offline-catalog.json', off)
    validate_catalog(root, directory)
    save(directory/'offline-verification.json', result | {'catalog_files_match': 3, 'platform_equivalence_proven': False})
    return {'status': 'match', 'differences': 0, 'rows': result['rows'], 'catalog_files_match': 3}


def test_command():
    return [*previous.test_command(), 'tests/unit/test_batch93_stage.py', 'tests/unit/test_financials.py',
        'tests/unit/test_batch93_acceptance.py']


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
    stage_junit = Path('data/staging/strategies-batch93/20261010-operand-stage-validation/tests.xml')
    cases += [(t.get('classname'), t.get('name')) for t in ET.parse(stage_junit).getroot().iter('testcase')
        if t.get('classname') in ('tests.unit.test_batch93_stage', 'tests.unit.test_financials')]
    cases += [('tests.unit.test_batch93_acceptance', name) for name in NEW_CASES]
    require(len(cases) == len(set(cases)) == 237, 'Intended test identities changed')
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
    require_root(_root)
    validate_sources(directory); validate_baseline(directory)
    require(not (directory/'isolated-files.json').exists(), 'Isolation already archived')
    candidate = {n: file_sha(n) for n in FILES}
    destination = Path(tempfile.mkdtemp(prefix='observe-batch93-')); copied = {}
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
    for path in (directory/'strategy_catalog.before.py', previous.DIRECTORY/'isolated-files.json', STAGE_CHECKPOINT):
        copy(str(path), file_sha(path))
    bound = read(stage.DIRECTORY/'input-binding.json')
    copy(str(Path('data/snapshots')/f'{stage.SNAPSHOT}.json'), stage.SNAPSHOT_SHA)
    copy(str(stage.IMPORT), stage.IMPORT_SHA)
    for row in bound['partitions']: copy(row['file'], row['bytes_sha256'])
    for name, sha in stage_evidence()['evidence_sha256'].items(): copy(name, sha)
    report = read(STAGE_CHECKPOINT); copy(report['stage_report'], report['stage_report_sha256'])
    copy(str(stage.READINESS/'source-readiness.json'), file_sha(stage.READINESS/'source-readiness.json'))
    for name, sha in read(stage.DIRECTORY/'development-failure-binding.json').items(): copy(name, sha)
    manifest = directory/'isolated-files.json'; save(manifest, {'destination': str(destination), 'files': copied,
        'baseline': 'Previous metadata SHA code, pinned prior test operands plus Batch93 candidate; not committed HEAD equivalence'})
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
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES,
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
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
    require_root(root)
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
    return {'status': 'passed', 'tests': len(expected_test_cases())}


def validate_progress_catalog(root, directory, progress):
    catalog = read(directory/'source-reviews/review.json')['catalog']
    row = next(r for r in read(directory/'offline-catalog.json')['files'] if r['name'] == 'catalog.json')
    require(progress['catalog_id'] == catalog['catalog_id'] and progress['catalog_sha256'] == row['sha256'],
        'Final progress catalog identity differs')
    require(read(Path(root)/'catalog/strategies/latest.json') ==
        {'catalog_id': catalog['catalog_id'], 'directory': catalog['output'], 'sha256': row['sha256']} and
        file_sha(Path(catalog['output'])/'catalog.json') == row['sha256'], 'Final catalog navigation differs')


def finish(root, directory):
    require_root(root)
    with operation_lock(root, DATA_WRITER): return finish_locked(root, directory)


def finish_locked(root, directory):
    require_root(root)
    require(directory == DIRECTORY and not RECEIPT.exists() and not (directory/'final-binding-verification.json').exists(),
        'Acceptance already archived or unexpected directory')
    validate_sources(directory); validate_catalog(root, directory); validate_isolation(directory)
    validate_run_inventory(root, directory)
    require(stage_ancestry(directory) == read(directory/'stage-protection-baseline.json'), 'Stage protection proof changed')
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json')
    require(all(file_sha(n) == sha for binding_name in ('acceptance-development-binding.json', 'final-catalog-development-binding.json')
        for n, sha in read(directory/binding_name).items()),
        'Acceptance development evidence changed')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == {n: file_sha(n) for n in FILES} and
        CORE <= checked['evidence_sha256'].keys(), 'Frozen check scope changed')
    validate_commands(checked)
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
        'Checked copies changed')
    require(test_coverage(directory/'checks/tests.xml')['complete'], 'Frozen root coverage missing')
    require(off['status'] == 'match' and off['rows'] == 501353 and off['differences'] == 0 and off['catalog_files_match'] == 3 and
        off['socket_HTTP_disabled'] and off['not_a_backtest'] and off['new_trading_backtests'] == 0, 'Offline scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed')
    protected = protect(root, directory)
    validate_run_inventory(root, directory)
    progress = archive(root, 'Batch93 two financial sources research accepted;501353 raw operand rows/40 independent coverage and30 literal callbacks offline match0; strict financial0, no trades')
    require(progress['manually_reviewed'] == 270 and progress['locally_backtested_sources'] == 12 and progress['local_strategy_runs'] == 75,
        'Acceptance totals differ')
    validate_progress_catalog(root, directory, progress)
    save(RECEIPT, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': binding(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline_verification': off,
        'isolated_verification': read(directory/'isolated-verification.json'), 'protection': protected, 'progress': progress,
        'newly_reviewed': 2, 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
        'new_requests': 0, 'new_trading_backtests': 0, 'original_strategy_complete': False, 'not_a_backtest': True,
        'restore_required': ['Historical financial versions, period/units and original _1/_4/_5 definitions',
            'Historical whole security pool/end_date dtype, ST/SW_l1/IPO/paused and PB preprocessing',
            'Original five JQ style factors/identity and platform order/position/callback semantics',
            'Any formula/selection/execution correction needs a separate confirmed economic variant and soleBook inputs']})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(RECEIPT),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': 270}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'stage_protection', 'replay', 'catalog', 'offline', 'isolate', 'checks', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()): result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
