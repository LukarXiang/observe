"""Freeze dominant-future and explicit CCFX dependency fixes; no trading run."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
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
from scripts import review_strategy_batch56 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch56/20261007-convertible-futures')
RECEIPT = Path('docs/handoff/2026-10-07-batch56-verification.json')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch57.py', 'tests/unit/test_catalog_futures.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'test_catalog_futures.frozen.py', 'red-regression.json', 'futures-source-evidence.json',
    'scan-impact.json', 'source-reviews/review.json', 'offline-catalog.json', 'dependency-state.json'}
FIELDS = {'apis', 'asset_scope', 'gaps', 'status'}
ASSET = '衍生品候选'
GAP = '专用资产行情与交易规则'
DOMINANT = re.compile(r'\bget_dominant_future\s*\(')
CONTRACT = re.compile(r"['\"]([A-Za-z]{1,3}\d{4}\.CCFX)['\"]")


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch56 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def source_evidence(directory, copy=False):
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    rows = []; folder = directory / 'futures-sources'
    if copy: folder.mkdir()
    for item in read(prior):
        source = Path('repo/量化策略源代码') / item['path']; text, encoding = read_source(source)
        require(file_sha(source) == item['bytes_sha256'], 'Corpus source changed')
        code = '\n'.join(text.splitlines()[item['code_start_line']-1:])
        dominant = bool(DOMINANT.search(code)); contracts = sorted(set(CONTRACT.findall(code)))
        if not dominant and not contracts: continue
        copied = folder / (strategy_id(item['path']) + '.source')
        if copy: shutil.copyfile(source, copied)
        require(file_sha(copied) == item['bytes_sha256'], 'Candidate source copy changed')
        try:
            tree = ast.parse(code)
            calls = sorted([{'line': n.lineno, 'ast': ast.dump(n, include_attributes=False)} for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == 'get_dominant_future') or
                (isinstance(n.func, ast.Attribute) and n.func.attr == 'get_dominant_future'))], key=lambda r: (r['line'], r['ast']))
            literals = sorted([{'line': n.lineno, 'value': n.value} for n in ast.walk(tree) if isinstance(n, ast.Constant) and
                isinstance(n.value, str) and re.fullmatch(r'[A-Za-z]{1,3}\d{4}\.CCFX', n.value)], key=lambda r: (r['line'], r['value']))
            syntax_error = None
        except SyntaxError as exc: calls = None; literals = None; syntax_error = f'{exc.lineno}: {exc.msg}'
        rows.append({'path': item['path'], 'copy': str(copied), 'sha256': file_sha(source), 'encoding': encoding,
            'lines': len(text.splitlines()), 'code_start_line': item['code_start_line'], 'dominant_text_candidate': dominant,
            'ccfx_text_candidates': contracts, 'dominant_calls': calls, 'ccfx_literals': literals, 'syntax_error': syntax_error,
            'review_status': item['review_status'], 'manual_review_added': False})
    require(len(rows) == 33 and sum(len(r['dominant_calls'] or []) for r in rows) == 79 and
        sum(r['syntax_error'] is not None for r in rows) == 1, 'Frozen candidate corpus differs')
    return {'sources': rows, 'not_a_backtest': True,
        'limits': ['Whole source copies and explicit call/contract candidates are not new full manual strategy reviews',
            'Text includes comments and inactive functions; AST establishes syntax, not callback reachability or actual executed assets',
            'One legacy-print source preserves raw text and syntax failure, without execution or silent syntax repair',
            'Aliases, dynamic contract construction and missing historical main mappings require separate review']}


def dependency_state(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'prior_dependency_inventory_sha256': file_sha(ACCEPTED / 'dependency-inventory.json'),
        'prior_probe_results_sha256': file_sha(ACCEPTED / 'probe-results.json'),
        'not_a_backtest': True, 'new_supplements': 0, 'new_publications': 0,
        'limits': ['Snapshot table metadata is not a fresh full partition integrity check',
            'Daily equities/indices and stock5m cannot replace futures contracts/minutes/main timeline/settlement/rules',
            'No second ledger, proxy instrument, fabricated margin/settlement or renewed failed DNS request']}


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch57 futures dependency regression started; prior turn made research progress')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile('tests/unit/test_catalog_futures.py', directory / 'test_catalog_futures.frozen.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    save(directory / 'futures-source-evidence.json', source_evidence(directory, copy=True))
    save(directory / 'dependency-state.json', dependency_state(root))
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_futures.py', '--tb=short']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'code_sha256': file_sha(directory / 'strategy_catalog.before.py'), 'test_sha256': file_sha(directory / 'test_catalog_futures.frozen.py')})
    require(result.returncode == 1 and '20 failed, 7 passed' in result.stdout, 'Expected futures regression differs')
    return {'status': 'ok', 'sources': 33, 'regression': '20 failed, 7 passed'}


def changes(before, after, evidence):
    prior = {r['path']: r for r in before}; candidates = {r['path']: r for r in evidence['sources']}; rows = []
    require(len(prior) == len(after) == 695 and {r['path'] for r in after} == set(prior), 'Source set changed')
    for item in after:
        old = prior[item['path']]
        require(item.keys() == old.keys() and all(item[k] == old[k] for k in item.keys()-FIELDS), 'Scanner changed identity/manual rules')
        delta = {k: {'before': old[k], 'after': item[k]} for k in sorted(FIELDS) if item[k] != old[k]}
        if not delta: continue
        require(item['path'] in candidates, 'Unrelated source changed'); candidate = candidates[item['path']]
        if item['apis'] != old['apis']:
            require(candidate['dominant_text_candidate'] and 'get_dominant_future' not in old['apis'] and
                item['apis'].count('get_dominant_future') == 1 and [a for a in item['apis'] if a != 'get_dominant_future'] == old['apis'], 'Unrelated API change')
        if item['asset_scope'] != old['asset_scope']:
            require(ASSET not in old['asset_scope'] and ASSET in item['asset_scope'] and
                [a for a in item['asset_scope'] if a != ASSET] == [a for a in old['asset_scope'] if a != '待人工识别'], 'Unrelated asset change')
        if item['gaps'] != old['gaps']:
            require(old['review_status'] != '人工审查完成' and GAP not in old['gaps'] and item['gaps'] == old['gaps']+[GAP], 'Unrelated gap change')
        require(item['status'] == old['status'], 'Unexpected status change')
        rows.append({'path': item['path'], 'source_sha256': item['bytes_sha256'], 'changes': delta})
    require({r['path'] for r in rows} == {r['path'] for r in evidence['sources'] if r['dominant_text_candidate']}, 'Scan missed dominant candidates')
    require(Counter(k for r in rows for k in r['changes']) == {'apis': 28, 'asset_scope': 5, 'gaps': 3}, 'Expected impact differs')
    return rows


def validate_scan(root, directory):
    binding(root, directory); red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '20 failed, 7 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha(directory / 'test_catalog_futures.frozen.py') == file_sha('tests/unit/test_catalog_futures.py'), 'Red regression changed')
    evidence = read(directory / 'futures-source-evidence.json')
    require(source_evidence(directory) == evidence, 'Candidate call/contract evidence changed')
    require(dependency_state(root) == read(directory / 'dependency-state.json'), 'Dependency state changed')
    doc = read(directory / 'scan-impact.json'); old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    new = Path(doc['catalog']['output']) / 'catalog.json'
    require(doc['old_catalog_file'] == str(old) and doc['old_catalog_sha256'] == file_sha(old) and
        doc['catalog_sha256'] == file_sha(new) and doc['changes'] == changes(read(old), read(new), evidence), 'Scan impact changed')
    require(read(directory / 'source-reviews/review.json') == {'catalog': doc['catalog'], 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0}, 'Scanner-only scope changed')


def scan(root, directory):
    binding(root, directory); old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    result = catalog_strategies(root, 'repo/量化策略源代码'); new = Path(result['output']) / 'catalog.json'
    delta = changes(read(old), read(new), read(directory / 'futures-source-evidence.json')); counts = Counter(k for r in delta for k in r['changes'])
    save(directory / 'scan-impact.json', {'catalog': result, 'catalog_sha256': file_sha(new), 'old_catalog_file': str(old),
        'old_catalog_sha256': file_sha(old), 'changes': delta, 'changed_sources': len(delta), 'field_counts': dict(counts), 'not_a_backtest': True})
    save(directory / 'source-reviews/review.json', {'catalog': result, 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0}); validate_scan(root, directory)
    return {'changed_sources': len(delta), 'field_counts': dict(counts), 'progress': archive(root, 'Batch57 dominant/CCFX dependency impact frozen across695 sources')}


def offline(root, directory):
    validate_scan(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch57 catalog attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        result = offline_catalog(root, directory); dependency = dependency_state(root)
    require(before == implementation() and dependency == read(directory / 'dependency-state.json'), 'Offline implementation/dependencies changed')
    save(directory / 'offline-catalog.json', result); validate_catalog(root, directory)
    return archive(root, 'Batch57 forbidden-network futures dependency catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch57.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_futures.py', 'tests/unit/test_catalog_bond.py',
         'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_research_queries.py', 'tests/unit/test_catalog_fund_universe.py',
         'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch56_research.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


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
    return {'status': 'passed'}


def finish(root, directory):
    validate_scan(root, directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch57 futures dependency recognition accepted; all-strategy research continues')
    save(Path('docs/handoff/2026-10-07-batch57-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'progress': progress, 'checks': checked,
        'scan': read(directory / 'scan-impact.json'), 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 0})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'offline', 'checks', 'finish']); parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch57/20261007-futures-dependency')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False))
