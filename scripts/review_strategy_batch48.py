"""Freeze the missing macro-query dependency scan and its corpus-wide impact."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch47 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

ACCEPTED = Path('data/staging/strategies-batch47/20261007-candidate-momentum')
RECEIPT = Path('docs/handoff/2026-10-07-batch47-verification.json')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch48.py', 'tests/unit/test_catalog_macro.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'test_catalog_macro.frozen.py', 'red-regression.json', 'macro-source-evidence.json',
    'scan-impact.json', 'source-reviews/review.json', 'offline-catalog.json',
    'test-correction.json', 'test_catalog_macro.corrected.py', 'initial-check-failure/failure.json'}
FIELDS = {'apis', 'gaps', 'status'}
MACRO_GAP = 'macro平台表/历史发布版本与可用时点待核实'


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == previous.SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch47 changed')
    result = {'snapshot': previous.SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'final_binding_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    checkpoint(root, directory, 'Batch48 macro query dependency regression started; no strategy run added')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile('tests/unit/test_catalog_macro.py', directory / 'test_catalog_macro.frozen.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    source_root = Path('repo/量化策略源代码'); rows = []; folder = directory / 'macro-sources'; folder.mkdir()
    for item in read(Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'):
        source = source_root / item['path']; text, encoding = read_source(source)
        if not re.search(r'\bmacro\s*\.\s*run_query\s*\(', text): continue
        require(file_sha(source) == item['bytes_sha256'] and 'macro.run_query' not in item['apis'], 'Baseline macro candidate changed')
        copied = folder / (strategy_id(item['path']) + '.source'); shutil.copyfile(source, copied)
        try:
            tree = ast.parse(text)
            calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'macro' and n.func.attr == 'run_query']
            syntax_error = None
        except SyntaxError as exc: calls = None; syntax_error = f'{exc.lineno}: {exc.msg}'
        rows.append({'path': item['path'], 'sha256': file_sha(source), 'copy': str(copied), 'encoding': encoding,
            'lines': len(text.splitlines()), 'macro_call_lines': calls, 'syntax_error': syntax_error,
            'review_status': item['review_status'], 'manual_review_added': False})
    require(len(rows) == 7, 'Macro source candidate count differs')
    save(directory / 'macro-source-evidence.json', {'sources': rows, 'not_a_backtest': True,
        'limits': ['Static call candidates and copies; not full manual rule review or macro data equivalence']})
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_macro.py', '--tb=short']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha(directory / 'test_catalog_macro.frozen.py')})
    require(result.returncode == 1 and '5 failed, 3 passed' in result.stdout, 'Expected macro regression differs')
    return {'status': 'ok', 'sources': len(rows), 'regression': '5 failed, 3 passed'}


def changes(before, after):
    prior = {row['path']: row for row in before}
    require(len(prior) == len(after) == 695 and {r['path'] for r in after} == set(prior), 'Source set changed')
    rows = []
    for item in after:
        old = prior[item['path']]
        require(item.keys() == old.keys() and all(item[k] == old[k] for k in item.keys() - FIELDS),
            'Macro scan changed source identities or manual reviews')
        delta = {k: {'before': old[k], 'after': item[k]} for k in sorted(FIELDS) if item[k] != old[k]}
        if delta:
            require([a for a in item['apis'] if a != 'macro.run_query'] == old['apis'] and
                item['apis'].count('macro.run_query') == 1 and 'macro.run_query' not in old['apis'],
                'Unrelated API change')
            require([g for g in item['gaps'] if g != MACRO_GAP] == old['gaps'] and
                item['gaps'].count(MACRO_GAP) <= 1, 'Unrelated gap change')
            require(item['status'] == old['status'] or
                (old['status'] == '待审查' and item['status'] == '待数据' and MACRO_GAP in item['gaps']),
                'Unrelated status change')
            rows.append({'path': item['path'], 'source_sha256': item['bytes_sha256'], 'changes': delta})
    return rows


def validate_scan(root, directory):
    binding(root, directory); red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '5 failed, 3 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha(directory / 'test_catalog_macro.frozen.py'),
        'Frozen regression changed')
    correction = read(directory / 'test-correction.json')
    frozen = directory / 'test_catalog_macro.frozen.py'; current = Path('tests/unit/test_catalog_macro.py')
    require(correction['original_sha256'] == file_sha(frozen) and
        correction['corrected_sha256'] == file_sha(current) == file_sha(directory / 'test_catalog_macro.corrected.py') and
        frozen.read_text(encoding='utf-8').count("assert item['status'] == '非交易研究脚本'") == 1 and
        current.read_text(encoding='utf-8') == frozen.read_text(encoding='utf-8').replace(
            "assert item['status'] == '非交易研究脚本'", "assert item['status'] == '待数据'"), 'Test-only correction changed')
    failure = read(directory / 'initial-check-failure/failure.json')
    require(failure['red_test_sha256'] == red['test_sha256'] and
        all(file_sha(r['copy']) == r['sha256'] for r in failure['files']), 'Initial check failure changed')
    for item in read(directory / 'macro-source-evidence.json')['sources']:
        require(file_sha(item['copy']) == file_sha(Path('repo/量化策略源代码') / item['path']) == item['sha256'], 'Macro source changed')
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
    delta = changes(read(old), read(new)); counts = Counter(k for r in delta for k in r['changes'])
    require(len(delta) == 7, 'Macro scanner impact differs')
    save(directory / 'scan-impact.json', {'catalog': result, 'old_catalog_file': str(old),
        'old_catalog_sha256': file_sha(old), 'catalog_sha256': file_sha(new), 'changes': delta,
        'changed_sources': len(delta), 'field_counts': dict(counts), 'not_a_backtest': True,
        'limits': ['Macro candidates require historical releases and table definitions; latest monthly data cannot prove past availability']})
    save(directory / 'source-reviews/review.json', {'catalog': result, 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0})
    validate_scan(root, directory)
    return {'changed_sources': len(delta), 'field_counts': dict(counts),
        'progress': archive(root, 'Batch48 macro dependency scan impact across695 sources frozen')}


def offline(root, directory):
    validate_scan(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch48 catalog recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = offline_catalog(root, directory)
    require(before == implementation(), 'Offline implementation changed')
    save(directory / 'offline-catalog.json', result); validate_catalog(root, directory)
    return archive(root, 'Batch48 forbidden-network macro dependency catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch48.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_macro.py',
         'tests/unit/test_catalog_prefix.py', 'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_dependencies.py',
         'tests/unit/test_catalog_margin_weekly.py', 'tests/unit/test_catalog_schedules.py',
         'tests/integration/test_strategies.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    validate_scan(root, directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-final-{k}.log'
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
    protection = protect(root, directory); progress = archive(root, 'Batch48 macro query dependency repair accepted; macro rule review remains next')
    save(Path('docs/handoff/2026-10-07-batch48-verification.json'), {'status': 'ok', 'snapshot': previous.SNAPSHOT,
        'progress': progress, 'checks': checked, 'scan': read(directory / 'scan-impact.json'),
        'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 0})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch48/20261007-macro-dependency')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False))
