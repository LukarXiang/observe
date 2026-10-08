"""Archive the parseable-module prefix repair and its full-catalog impact."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import catalog_strategies, read_source
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch45 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

ACCEPTED = Path('data/staging/strategies-batch45/20261007-momentum-epo-trend')
RECEIPT = Path('docs/handoff/2026-10-07-batch45-verification.json')
SOURCE = '2024年度精选策略2/43.ETF动量因子评估.txt'
SOURCE_SHA = '803d0afa67862c0ab4060ed88de841463120c7d8f26989d4a68c0c5b09d2e057'
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch46.py', 'tests/unit/test_catalog_prefix.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
        'test_catalog_prefix.frozen.py', 'red-regression.json', 'source-prefix-evidence.json',
        'source-prefix.original', 'scan-impact.json', 'source-reviews/review.json', 'offline-catalog.json'}
SCAN_FIELDS = {'code_start_line', 'code_sha256', 'syntax_error', 'asset_scope', 'frequencies',
               'scheduled_times', 'schedule_detection', 'apis', 'imports', 'financial_fields',
               'called_functions', 'status', 'gaps', 'variant_candidates', 'future_information_risks'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == previous.SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch45 changed')
    result = {'snapshot': previous.SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'final_binding_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    checkpoint(root, directory, 'Batch46 parseable-module prefix regression started; no strategy run added')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile('tests/unit/test_catalog_prefix.py', directory / 'test_catalog_prefix.frozen.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    source = Path('repo/量化策略源代码') / SOURCE
    require(file_sha(source) == SOURCE_SHA, 'Research source changed')
    shutil.copyfile(source, directory / 'source-prefix.original')
    text, encoding = read_source(source); tree = ast.parse(text)
    imports = [n.lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Name) and n.func.id == 'get_price']
    pool = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'stkList' for t in n.targets))
    require(min(calls) < min(imports) and len(pool) == 4, 'Prefix diagnosis differs')
    save(directory / 'source-prefix-evidence.json', {'source': SOURCE, 'source_sha256': SOURCE_SHA,
        'encoding': encoding, 'complete_lines': len(text.splitlines()), 'full_module_parseable': True,
        'first_import_line': min(imports), 'get_price_lines': calls, 'original_pool': pool,
        'ast_parse_signature': str(inspect.signature(ast.parse)), 'ast_parse_source': inspect.getsource(ast.parse),
        'not_a_backtest': True, 'limits': ['Only scanner diagnosis; original future returns remain research labels']})
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_prefix.py', '--tb=short']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha(directory / 'test_catalog_prefix.frozen.py')})
    require(result.returncode == 1 and '5 failed, 2 passed' in result.stdout, 'Expected prefix regression differs')
    return {'status': 'ok', 'regression': '5 failed, 2 passed'}


def changes(before, after):
    prior = {row['path']: row for row in before}
    require(len(prior) == len(after) == 695 and {row['path'] for row in after} == set(prior), 'Source set changed')
    result = []
    for row in after:
        old = prior[row['path']]
        require(row.keys() == old.keys() and all(row[k] == old[k] for k in row.keys() - SCAN_FIELDS),
                'Scanner changed source identities or manual reviews')
        delta = {k: {'before': old[k], 'after': row[k]} for k in sorted(SCAN_FIELDS) if row[k] != old[k]}
        if delta: result.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
    return result


def validate_scan(root, directory):
    binding(root, directory); red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '5 failed, 2 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha(directory / 'test_catalog_prefix.frozen.py') == file_sha('tests/unit/test_catalog_prefix.py'),
        'Frozen regression changed')
    require(file_sha(directory / 'source-prefix.original') == file_sha(Path('repo/量化策略源代码') / SOURCE) == SOURCE_SHA,
        'Source diagnosis changed')
    doc = read(directory / 'scan-impact.json')
    old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    new = Path(doc['catalog']['output']) / 'catalog.json'
    require(doc['old_catalog_file'] == str(old) and doc['old_catalog_sha256'] == file_sha(old) and
            doc['catalog_sha256'] == file_sha(new) and doc['changes'] == changes(read(old), read(new)), 'Scan impact changed')
    require(read(directory / 'source-reviews/review.json') == {'catalog': doc['catalog'], 'sources': [],
            'not_a_backtest': True, 'newly_reviewed': 0}, 'Scanner-only scope changed')
    return doc


def scan(root, directory):
    binding(root, directory)
    old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    result = catalog_strategies(root, 'repo/量化策略源代码'); new = Path(result['output']) / 'catalog.json'
    delta = changes(read(old), read(new)); field_counts = Counter(k for r in delta for k in r['changes'])
    target = next(r for r in read(new) if r['path'] == SOURCE)
    require(target['code_start_line'] == 1 and target['syntax_error'] is None and 'get_price' in target['apis'] and
            target['asset_scope'] == ['ETF/基金候选'], 'Original research prefix still missing')
    save(directory / 'scan-impact.json', {'catalog': result, 'old_catalog_file': str(old),
        'old_catalog_sha256': file_sha(old), 'catalog_sha256': file_sha(new), 'changes': delta,
        'changed_sources': len(delta), 'field_counts': dict(field_counts), 'not_a_backtest': True,
        'limits': ['Static candidates are not platform execution evidence; no source or manual review changed']})
    save(directory / 'source-reviews/review.json', {'catalog': result, 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0})
    validate_scan(root, directory)
    return {'changed_sources': len(delta), 'field_counts': dict(field_counts),
        'progress': archive(root, 'Batch46 full-module prefix scan impact across695 sources frozen')}


def offline(root, directory):
    validate_scan(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch46 catalog recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = offline_catalog(root, directory)
    require(before == implementation(), 'Offline implementation changed')
    save(directory / 'offline-catalog.json', result); validate_catalog(root, directory)
    return archive(root, 'Batch46 forbidden-network full catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch46.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_prefix.py',
         'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_margin_weekly.py',
         'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategies.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    validate_scan(root, directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    validate_scan(root, directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items():
        require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch46 full-module prefix repair accepted; next66/86/43 research retained')
    save(Path('docs/handoff/2026-10-07-batch46-verification.json'), {'status': 'ok', 'snapshot': previous.SNAPSHOT,
        'progress': progress, 'checks': checked, 'scan': read(directory / 'scan-impact.json'),
        'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 0})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch46/20261007-full-module-prefix')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False))
