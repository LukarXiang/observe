"""Archive research-query classification without promoting static scans to manual review."""
import argparse
import ast
from contextlib import redirect_stdout
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
from observe.strategy_catalog import catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch47 import SNAPSHOT
from scripts import review_strategy_batch48 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

ACCEPTED = Path('data/staging/strategies-batch48/20261007-macro-dependency')
RECEIPT = Path('docs/handoff/2026-10-07-batch48-verification.json')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch49.py', 'tests/unit/test_catalog_research_queries.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'test_catalog_research_queries.frozen.py', 'test_catalog_macro.before.py', 'red-regression.json',
    'query-source-evidence.json', 'scan-impact.json', 'source-reviews/review.json', 'offline-catalog.json'}
ORDERS = {'order', 'order_value', 'order_target', 'order_target_value', 'order_target_percent'}


def implementation(): return {name: file_sha(name) for name in FILES}


def candidate(row):
    calls = row['called_functions']
    return calls is not None and not ORDERS.intersection(calls) and {n for n in calls if n.startswith('run_')} == {'run_query'}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'],
        'Accepted batch48 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'final_binding_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    checkpoint(root, directory, 'Batch49 query-only research classification regression started; no strategy run added')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile('tests/unit/test_catalog_research_queries.py', directory / 'test_catalog_research_queries.frozen.py')
    shutil.copyfile('tests/unit/test_catalog_macro.py', directory / 'test_catalog_macro.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    old = read(Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json')
    folder = directory / 'query-sources'; folder.mkdir(); rows = []
    for item in filter(candidate, old):
        source = Path('repo/量化策略源代码') / item['path']; text, encoding = read_source(source)
        require(file_sha(source) == item['bytes_sha256'], 'Query candidate changed')
        copied = folder / (strategy_id(item['path']) + '.source'); shutil.copyfile(source, copied)
        tree = ast.parse('\n'.join(text.splitlines()[item['code_start_line'] - 1:]))
        calls = sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))})
        require(calls == item['called_functions'], 'Original AST query candidates differ')
        rows.append({'path': item['path'], 'sha256': file_sha(source), 'copy': str(copied), 'encoding': encoding,
            'lines': len(text.splitlines()), 'code_start_line': item['code_start_line'], 'called_functions': calls,
            'review_status': item['review_status'], 'status': item['status'], 'manual_review_added': False})
    require(len(rows) == 15 and sum(r['review_status'] != '人工审查完成' for r in rows) == 10, 'Query candidate count differs')
    save(directory / 'query-source-evidence.json', {'sources': rows, 'not_a_backtest': True,
        'limits': ['AST call metadata only, not full source rule review or platform execution evidence']})
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_research_queries.py', '--tb=short']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha(directory / 'test_catalog_research_queries.frozen.py')})
    require(result.returncode == 1 and '5 failed, 10 passed' in result.stdout, 'Expected query regression differs')
    return {'status': 'ok', 'candidates': len(rows), 'regression': '5 failed, 10 passed'}


def changes(before, after):
    prior = {row['path']: row for row in before}
    require(len(prior) == len(after) == 695 and {r['path'] for r in after} == set(prior), 'Source set changed')
    rows = []
    for item in after:
        old = prior[item['path']]
        require(item.keys() == old.keys() and all(item[k] == old[k] for k in item.keys() - {'status'}),
            'Query classification changed identities, dependencies or manual reviews')
        if item['status'] != old['status']:
            require(candidate(old) and old['review_status'] != '人工审查完成' and item['duplicate_of'] is None and
                old['status'] in ('待数据', '暂不可复现') and item['status'] == '非交易研究脚本', 'Unrelated status change')
            rows.append({'path': item['path'], 'source_sha256': item['bytes_sha256'],
                'changes': {'status': {'before': old['status'], 'after': item['status']}}})
    return rows


def validate_scan(root, directory):
    binding(root, directory); red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '5 failed, 10 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha(directory / 'test_catalog_research_queries.frozen.py') ==
        file_sha('tests/unit/test_catalog_research_queries.py'), 'Frozen regression changed')
    old_test = (directory / 'test_catalog_macro.before.py').read_text(encoding='utf-8')
    current_test = Path('tests/unit/test_catalog_macro.py').read_text(encoding='utf-8')
    require(old_test.count("assert item['status'] == '待数据'") == 2 and current_test == old_test.replace(
        "    assert item['status'] == '待数据'\n\n\n@pytest.mark.parametrize('code'",
        "    assert item['status'] == '非交易研究脚本'\n\n\n@pytest.mark.parametrize('code'"), 'Prior macro test migration changed')
    for item in read(directory / 'query-source-evidence.json')['sources']:
        require(file_sha(item['copy']) == file_sha(Path('repo/量化策略源代码') / item['path']) == item['sha256'], 'Query source changed')
    doc = read(directory / 'scan-impact.json')
    old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; new = Path(doc['catalog']['output']) / 'catalog.json'
    require(doc['old_catalog_file'] == str(old) and doc['old_catalog_sha256'] == file_sha(old) and
        doc['catalog_sha256'] == file_sha(new) and doc['changes'] == changes(read(old), read(new)), 'Scan impact changed')
    require(read(directory / 'source-reviews/review.json') == {'catalog': doc['catalog'], 'sources': [],
        'not_a_backtest': True, 'newly_reviewed': 0}, 'Scanner-only scope changed')
    return doc


def scan(root, directory):
    binding(root, directory)
    old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    result = catalog_strategies(root, 'repo/量化策略源代码'); new = Path(result['output']) / 'catalog.json'
    delta = changes(read(old), read(new)); require(len(delta) == 10, 'Query classifier impact differs')
    save(directory / 'scan-impact.json', {'catalog': result, 'old_catalog_file': str(old),
        'old_catalog_sha256': file_sha(old), 'catalog_sha256': file_sha(new), 'changes': delta,
        'changed_sources': len(delta), 'field_counts': {'status': len(delta)}, 'not_a_backtest': True,
        'limits': ['Research status is a static candidate; dependency gaps and manual review status remain unchanged']})
    save(directory / 'source-reviews/review.json', {'catalog': result, 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0})
    validate_scan(root, directory)
    return {'changed_sources': len(delta), 'progress': archive(root, 'Batch49 query-only research status impact across695 sources frozen')}


def offline(root, directory):
    validate_scan(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch49 catalog recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = offline_catalog(root, directory)
    require(before == implementation(), 'Offline implementation changed')
    save(directory / 'offline-catalog.json', result); validate_catalog(root, directory)
    return archive(root, 'Batch49 forbidden-network query-only research catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch49.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_research_queries.py',
         'tests/unit/test_catalog_macro.py', 'tests/unit/test_catalog_prefix.py', 'tests/unit/test_catalog_bars.py',
         'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_margin_weekly.py',
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
    protection = protect(root, directory); progress = archive(root, 'Batch49 query-only classification repair accepted; macro rule review retained')
    save(Path('docs/handoff/2026-10-07-batch49-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT,
        'progress': progress, 'checks': checked, 'scan': read(directory / 'scan-impact.json'),
        'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 0})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch49/20261007-query-classification')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False))
