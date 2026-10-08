"""Freeze literal dynamic fund-universe recognition and its corpus-wide impact."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tokenize
from unittest.mock import patch

import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch51 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch51/20261007-weekly-etf-emotion')
RECEIPT = Path('docs/handoff/2026-10-07-batch51-verification.json')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch52.py', 'tests/unit/test_catalog_fund_universe.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'test_catalog_fund_universe.frozen.py', 'red-regression.json', 'universe-source-evidence.json',
    'scan-impact.json', 'source-reviews/review.json', 'offline-catalog.json', 'dependency-state.json'}
FIELDS = {'asset_scope', 'gaps', 'status'}
FUND_ASSET, FUND_GAP = 'ETF/基金候选', 'ETF行情/公司行动/执行规则'
FUND_TYPES = {'etf', 'lof', 'fund', 'open_fund'}
OLD_PROBES = Path('data/staging/strategies-batch3/20261005-dependencies')


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'],
        'Accepted batch51 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'), 'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def call_evidence(code):
    try:
        tree = ast.parse(code); nodes = list(ast.walk(tree)); mode = 'ast'
    except SyntaxError:
        nodes = []; mode = 'tokenized_legacy'
        try: tokens = list(tokenize.generate_tokens(StringIO(code).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError): return [], 'unresolved_tokenization'
        for i, token in enumerate(tokens):
            if token.type != tokenize.NAME or token.string != 'get_all_securities': continue
            if i and tokens[i-1].string in ('def', 'class'): continue
            if i+1 >= len(tokens) or tokens[i+1].string != '(': continue
            depth = 0
            for j in range(i+1, len(tokens)):
                current = tokens[j]
                if current.type == tokenize.OP:
                    if current.string == '(': depth += 1
                    elif current.string == ')': depth -= 1
                if depth == 0:
                    text = tokenize.untokenize([(t.type, t.string) for t in tokens[i:j+1]])
                    try:
                        node = ast.parse(text, mode='eval').body; node.lineno = token.start[0]; nodes.append(node)
                    except SyntaxError: pass
                    break
    rows = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, (ast.Name, ast.Attribute)): continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        if name != 'get_all_securities': continue
        value = next((k.value for k in node.keywords if k.arg == 'types'), node.args[0] if node.args else None)
        values = value.elts if isinstance(value, (ast.List, ast.Tuple, ast.Set)) else [value]
        types = sorted({n.value for n in values if isinstance(n, ast.Constant) and isinstance(n.value, str)})
        rows.append({'line': node.lineno, 'ast': ast.dump(node, include_attributes=False), 'literal_types': types,
            'fund_types': sorted(set(types) & FUND_TYPES)})
    return sorted(rows, key=lambda r: (r['line'], r['ast'])), mode


def dependency_state(root):
    store = Store(root); state = store.state(SNAPSHOT); frame = store.load_state(state, 'valuations_1d', filters=[('instrument', '=', '600519.SH')])
    old = []
    for endpoint in ('bs_profit', 'bs_pcf', 'bs_universe', 'em_shares', 'cninfo_shares', 'sz_names', 'sina_names'):
        path = OLD_PROBES / endpoint / 'result.json'; row = read(path)
        require(row['strict_usable'] is False and file_sha(row['raw_file']) == row['raw_sha256'], 'Old dependency operand changed')
        data = pd.read_parquet(row['raw_file']); require(len(data) == row['rows'] and list(data) == row['columns'], 'Old dependency profile changed')
        for response in row['wire_responses']: require(file_sha(response['file']) == response['sha256'], 'Old dependency response changed')
        old.append({'endpoint': endpoint, 'record_file': str(path), 'record_sha256': file_sha(path), 'rows': len(data),
            'raw_file': row['raw_file'], 'raw_sha256': row['raw_sha256'], 'wire_responses': row['wire_responses'], 'strict_usable': False})
    names = pd.read_parquet(next(r['raw_file'] for r in old if r['endpoint'] == 'bs_universe'))
    examples = names[names.code.isin(['sz.300420', 'sz.000755'])][['code', 'code_name']].to_dict('records')
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'shares_table_registered': 'shares' in state['tables'], 'historical_name_table_registered': 'instrument_status' in state['tables'],
        'pcf_sample': {'instrument': '600519.SH', 'rows': len(frame), 'first': str(frame.date.min()), 'last': str(frame.date.max()),
            'nonnull': int(frame.pcf_ncf_ttm.notna().sum()), 'strict_usable_rows': int(frame.strict_usable.sum()),
            'provider': sorted(frame.provider.unique()), 'version_evidence': sorted(frame.version_evidence.unique())},
        'old_dependency_files': old, 'old_20240329_name_counterexamples': examples, 'not_a_backtest': True,
        'limits': ['PCF provider-final values are available but platform equivalence and historical vintages remain unproved',
            'Quarterly amounts/events lack complete availability/unit evidence; current labels and future-name counterexamples do not supply historical pool',
            'No fresh supplement/publication; micro400 and small-value100 remain incomplete']}


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch52 literal dynamic fund pool recognition started; no new strategy run')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile('tests/unit/test_catalog_fund_universe.py', directory / 'test_catalog_fund_universe.frozen.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    old = read(Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'); rows = []; folder = directory / 'universe-sources'; folder.mkdir()
    for item in old:
        if 'get_all_securities' not in item['apis']: continue
        source = Path('repo/量化策略源代码') / item['path']; text, encoding = read_source(source)
        require(file_sha(source) == item['bytes_sha256'], 'Universe source changed')
        code = '\n'.join(text.splitlines()[item['code_start_line']-1:]); calls, mode = call_evidence(code)
        copied = folder / (strategy_id(item['path']) + '.source'); shutil.copyfile(source, copied)
        rows.append({'path': item['path'], 'copy': str(copied), 'sha256': file_sha(source), 'encoding': encoding,
            'lines': len(text.splitlines()), 'code_start_line': item['code_start_line'], 'mode': mode, 'calls': calls,
            'fund_candidate': any(r['fund_types'] for r in calls), 'manual_review_added': False})
    save(directory / 'universe-source-evidence.json', {'sources': rows, 'not_a_backtest': True,
        'limits': ['Explicit call/type candidates only; full source copies do not constitute manual strategy rule review']})
    save(directory / 'dependency-state.json', dependency_state(root))
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_fund_universe.py', '--tb=short']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'code_sha256': file_sha(directory / 'strategy_catalog.before.py'), 'test_sha256': file_sha(directory / 'test_catalog_fund_universe.frozen.py')})
    require(result.returncode == 1 and '28 failed, 22 passed' in result.stdout, 'Expected fund regression differs')
    return {'status': 'ok', 'call_candidate_sources': len(rows), 'fund_candidate_sources': sum(r['fund_candidate'] for r in rows), 'regression': '28 failed, 22 passed'}


def changes(before, after, evidence):
    prior = {r['path']: r for r in before}; candidates = {r['path'] for r in evidence['sources'] if r['fund_candidate']}
    require(len(prior) == len(after) == 695 and {r['path'] for r in after} == set(prior), 'Source set changed')
    expected_assets = {r['path'] for r in before if r['path'] in candidates and FUND_ASSET not in r['asset_scope']}; rows = []
    for item in after:
        old = prior[item['path']]
        require(item.keys() == old.keys() and all(item[k] == old[k] for k in item.keys()-FIELDS), 'Fund scanner changed identity/rules/manual review')
        delta = {k: {'before': old[k], 'after': item[k]} for k in sorted(FIELDS) if item[k] != old[k]}
        if not delta: continue
        require(item['path'] in expected_assets and item['asset_scope'] == [FUND_ASSET]+[a for a in old['asset_scope'] if a != '待人工识别'], 'Unrelated asset change')
        require(item['gaps'] == old['gaps'] or (item['gaps'].count(FUND_GAP) == 1 and [g for g in item['gaps'] if g != FUND_GAP] == old['gaps']), 'Unrelated gap change')
        require(item['status'] == old['status'] or (old['status'] == '待审查' and item['status'] == '待数据' and FUND_GAP in item['gaps']), 'Unrelated status change')
        rows.append({'path': item['path'], 'source_sha256': item['bytes_sha256'], 'changes': delta})
    require({r['path'] for r in rows} == expected_assets, 'Fund scan did not cover explicit missing pool candidates')
    return rows


def validate_scan(root, directory):
    binding(root, directory); red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '28 failed, 22 passed' in red['stdout'] and red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha(directory / 'test_catalog_fund_universe.frozen.py') == file_sha('tests/unit/test_catalog_fund_universe.py'), 'Frozen regression changed')
    evidence = read(directory / 'universe-source-evidence.json')
    for row in evidence['sources']:
        require(file_sha(row['copy']) == file_sha(Path('repo/量化策略源代码') / row['path']) == row['sha256'], 'Universe source changed')
        text, encoding = read_source(Path(row['copy'])); calls, mode = call_evidence('\n'.join(text.splitlines()[row['code_start_line']-1:]))
        require(row['calls'] == calls and row['mode'] == mode and row['encoding'] == encoding and row['lines'] == len(text.splitlines()) and
            row['fund_candidate'] == any(r['fund_types'] for r in calls) and row['manual_review_added'] is False, 'Universe call evidence changed')
    require(dependency_state(root) == read(directory / 'dependency-state.json'), 'Dependency recheck changed')
    doc = read(directory / 'scan-impact.json'); old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    new = Path(doc['catalog']['output']) / 'catalog.json'
    require(doc['old_catalog_file'] == str(old) and doc['old_catalog_sha256'] == file_sha(old) and doc['catalog_sha256'] == file_sha(new) and
        doc['changes'] == changes(read(old), read(new), evidence), 'Scan impact changed')
    require(read(directory / 'source-reviews/review.json') == {'catalog': doc['catalog'], 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0}, 'Scanner-only scope changed')


def scan(root, directory):
    binding(root, directory); old = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    result = catalog_strategies(root, 'repo/量化策略源代码'); new = Path(result['output']) / 'catalog.json'
    delta = changes(read(old), read(new), read(directory / 'universe-source-evidence.json')); counts = Counter(k for r in delta for k in r['changes'])
    save(directory / 'scan-impact.json', {'catalog': result, 'catalog_sha256': file_sha(new), 'old_catalog_file': str(old),
        'old_catalog_sha256': file_sha(old), 'changes': delta, 'changed_sources': len(delta), 'field_counts': dict(counts), 'not_a_backtest': True})
    save(directory / 'source-reviews/review.json', {'catalog': result, 'sources': [], 'not_a_backtest': True, 'newly_reviewed': 0}); validate_scan(root, directory)
    return {'changed_sources': len(delta), 'field_counts': dict(counts), 'progress': archive(root, 'Batch52 literal dynamic fund universe impact frozen across695 sources')}


def offline(root, directory):
    validate_scan(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch52 catalog attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        result = offline_catalog(root, directory); dependency = dependency_state(root)
    require(before == implementation() and dependency == read(directory / 'dependency-state.json'), 'Offline implementation/dependencies changed')
    save(directory / 'offline-catalog.json', result); validate_catalog(root, directory)
    return archive(root, 'Batch52 forbidden-network fund universe catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch52.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_fund_universe.py', 'tests/unit/test_catalog_schedules.py',
         'tests/unit/test_catalog_prefix.py', 'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_research_queries.py',
         'tests/unit/test_catalog_macro.py', 'tests/integration/test_strategies.py'], ['git', 'diff', '--check']]


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
    protection = protect(root, directory); progress = archive(root, 'Batch52 dynamic fund universe recognition accepted; original source research continues')
    save(Path('docs/handoff/2026-10-07-batch52-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'progress': progress, 'checks': checked,
        'scan': read(directory / 'scan-impact.json'), 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 0})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'offline', 'checks', 'finish']); parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch52/20261007-fund-universe')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False))
