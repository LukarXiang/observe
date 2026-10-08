"""Freeze ETF stop/weighted momentum/opening rules and historical finance API evidence."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import fingerprint
from observe.data.prices import with_adjusted
from scipy.stats import linregress
from scripts import review_strategy_batch38 as stocks
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts.review_strategy_batch39 import inputs
from scripts import review_strategy_batch41 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require


SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch41/20261006-etf-stop-weighted-open')
RECEIPT = Path('docs/handoff/2026-10-06-batch41-verification.json')
STOCK_RECEIPT = Path('docs/handoff/2026-10-06-batch38-verification.json')
SOURCES = ('2023年度精选策略/3.苦咖啡-默默赚钱系列-改.txt',
    '2023年度精选策略/40.再改进可实盘-默默赚钱系列-风险控制-增强版本-V5.0.txt',
    '2024年度精选策略1/41.人工智能强化学习DQN交易智能体（回馈社区公开训练代码）.txt',
    '聚宽2025年精选/27人工智能强化学习DQN交易智能体（回馈社区公开训练代码）.txt')
EXPECTED_SOURCE_SHA = ('e9eb4c47cdbd819d9d6200d1ecae58837ca81aa7cc78ec061eac5afd600e7cd3',
    '1e6ee2ecb16b0008eecb5f84620a7686c47eae024a7154e907a5ec601cbdbfac',
    'bf4f31552297e6ad232bb780917d926e40a27b55ca48719305f37777d29262b3',
    'bf4f31552297e6ad232bb780917d926e40a27b55ca48719305f37777d29262b3')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/ols.py'),
    ('repo/qlib', 'qlib/contrib/model/pytorch_nn.py'),
    ('repo/baostock', 'baostock/demo/demo_hs300_stocks.py'),
    ('repo/akshare', 'akshare/index/index_stock_zh.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch42.py',
    'tests/unit/test_catalog_bars.py', 'tests/unit/test_batch42_research.py'}))
CORE = previous.CORE | {'price-input.parquet', 'dqn-input.parquet', 'partition-verification.json'}
QUERIES = {
    'hs3002021': {'provider': 'baostock', 'function': 'query_hs300_stocks', 'parameters': {'date': '2021-01-04'},
        'limit': 'One dated vendor membership sample cannot prove original weekly decision-date platform pools'},
    'index_daily': {'provider': 'akshare', 'function': 'stock_zh_index_daily_em',
        'parameters': {'symbol': 'sh000001', 'start_date': '20050101', 'end_date': '20260929'},
        'limit': 'Daily index prices cannot prove include_now bars at weekly09:30'},
    'stock_minute': {'provider': 'akshare', 'function': 'stock_zh_a_hist_min_em',
        'parameters': {'symbol': '000065', 'start_date': '2026-09-28 09:30:00', 'end_date': '2026-09-29 15:00:00', 'period': '1', 'adjust': ''},
        'limit': 'Recent minute sample cannot supply missing pretrained weights or platform history defaults'}}


def implementation(): return {name: file_sha(name) for name in FILES}

def preflight(root, directory):
    checkpoint(root, directory, 'Batch42 get_bars API and four trend/DQN source reviews started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_bars.py', '--tb=no']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
        'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha('tests/unit/test_catalog_bars.py')})
    require(result.returncode == 1 and '8 failed, 4 passed' in result.stdout, 'Expected historical API regression differs')
    return {'status': 'ok', 'regression': '8 failed, 4 passed'}

def validate_scan(directory):
    before = read(directory / 'catalog-before.json')
    old = Path(before['directory']) / 'catalog.json'
    red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '8 failed, 4 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha('tests/unit/test_catalog_bars.py'), 'Red regression binding differs')
    scan_path = directory / 'api-scan.json'
    if scan_path.exists():
        doc = read(scan_path); path = Path(doc['catalog']['output']) / 'catalog.json'
        require(doc['old_catalog_file'] == str(old) and file_sha(old) == doc['old_catalog_sha256'] and
            file_sha(path) == doc['catalog_sha256'], 'Frozen scan catalogs changed')
        prior = {r['path']: r for r in read(old)}; rows = read(path); changes = []
        require(len(prior) == len(rows) == 695 and {r['path'] for r in rows} == set(prior), 'Frozen scan source set differs')
        for row in rows:
            previous_row = prior[row['path']]
            require(all(row[k] == v for k, v in previous_row.items() if k not in ('apis', 'gaps', 'status')), 'Frozen scan source/manual metadata differs')
            delta = {k: {'before': previous_row[k], 'after': row[k]} for k in ('apis', 'gaps', 'status') if row[k] != previous_row[k]}
            if delta: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
        require(changes == doc['changes'] and doc['not_a_backtest'] is True, 'Frozen scan changes differ')
    return old

def scan(root, directory):
    old_path = validate_scan(directory); old = {r['path']: r for r in read(old_path)}
    result = catalog_strategies(root, 'repo/量化策略源代码'); path = Path(result['output']) / 'catalog.json'
    rows = read(path); changes = []
    require(len(rows) == len(old) == 695 and {r['path'] for r in rows} == set(old), 'Source set differs')
    for row in rows:
        prior = old[row['path']]
        require(all(row[k] == v for k, v in prior.items() if k not in ('apis', 'gaps', 'status')), 'Scan changed source/manual metadata')
        delta = {k: {'before': prior[k], 'after': row[k]} for k in ('apis', 'gaps', 'status') if row[k] != prior[k]}
        if delta: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
    save(directory / 'api-scan.json', {'catalog': result, 'old_catalog_file': str(old_path), 'old_catalog_sha256': file_sha(old_path),
        'catalog_sha256': file_sha(path), 'changes': changes, 'not_a_backtest': True,
        'limits': ['Static calls can be in comments, strings or inactive functions; not proof of active get_bars requests']})
    return {'changed_sources': len(changes), 'catalog_id': result['catalog_id'],
        'progress': archive(root, 'Batch42 get_bars API correction across695 sources frozen')}

def start(root, directory):
    validate_scan(directory); bound = binding(root); apis = api_evidence()
    scan_doc = read(directory / 'api-scan.json')
    old_path = Path(scan_doc['catalog']['output']) / 'catalog.json'
    require(file_sha(old_path) == scan_doc['catalog_sha256'], 'API scan changed')
    save(directory / 'input-binding.json', bound); save(directory / 'existing-apis.json', apis)
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name, sha in zip(SOURCES, EXPECTED_SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require((old[name]['review_status'] == '人工审查完成') == (name == SOURCES[2]) and
            old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source changed/prior review differs')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name],
            'newly_reviewed': old[name]['review_status'] != '人工审查完成',
            'previous_review': {k: old[name][k] for k in REVIEWS[name] if k in old[name]}})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only OLS, pretrained-model separation, dated pool and index interface references; no substitute weights'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch42 three new source reviews and one DQN re-review frozen; missing pool/model/opening dependencies retained')

def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')

def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory)
    with redirect_stdout(io.StringIO()): diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch42 original trend/DQN arithmetic and state diagnoses frozen')

def worker(root, directory, endpoint):
    import akshare as ak
    import baostock as bs
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API differs')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(8)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kw):
        require(len(row['wire']) < 10, 'Response cap exceeded'); session.trust_env = False; kw['timeout'] = (8, 10)
        response = original(session, method, url, **kw); path = folder / f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status': response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()):
            if query['provider'] == 'baostock':
                with BaoStock(root).session():
                    result = getattr(bs, query['function'])(**query['parameters']); values = []
                    require(result.error_code == '0', f'Provider error: {result.error_code}/{result.error_msg}')
                    while result.next():
                        values.append(result.get_row_data()); require(len(values) <= 10000, 'Provider row cap exceeded')
                    frame = pd.DataFrame(values, columns=result.fields)
            else: frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch42_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}], limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row

def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['published'] is False, 'Probe binding differs')
        for item in [*row['files'], *row['wire']]: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] == 'sample':
            require(row['strict_usable'] is False and row['limit'] == query['limit'] and len(row['files']) == 1, 'Unproven sample admitted')
            item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(0 < len(frame) == item['rows'] <= 100000 and list(frame.columns) == item['columns'], 'Sample profile differs')
        else: require(row['status'] in ('failed', 'timeout') and not row['files'] and row.get('error'), 'Failed probe admitted data')
        rows.append(row)
    return rows

def probe(root, directory):
    binding(root, directory)
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch42', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch42 existing pool/index/minute supplementation attempts frozen; no publication')

def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch42 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch42 forbidden-network arithmetic/diagnostic/catalog byte match')

def checks(root, directory):
    before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked code changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True); require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}

def finish(root, directory):
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory); validate_sources(directory); validate_scan(directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    with redirect_stdout(io.StringIO()): require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch42 trend/DQN arithmetic accepted; original trade dependencies remain missing')
    save(Path('docs/handoff/2026-10-07-batch42-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}




def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch42.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch42_research.py',
            'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_batch41_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def binding(root, directory=None):
    receipt = read(RECEIPT); stock_receipt = read(STOCK_RECEIPT)
    require(receipt['status'] == stock_receipt['status'] == 'ok' and receipt['snapshot'] == stock_receipt['snapshot'] == SNAPSHOT,
        'Accepted receipts differ')
    stock_dir = Path(stock_receipt['reviews']['sources'][0]['source_copy']).parent.parent
    rows = {}
    for folder, rec, names in ((ACCEPTED, receipt, ('index-input.parquet', 'calendar-input.parquet')),
            (stock_dir, stock_receipt, ('price-input.parquet',))):
        for name in names:
            path = folder / name
            require(file_sha(path) == rec['checks']['evidence_sha256'][name], 'Accepted input changed')
            rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    state = Store(root).state(SNAPSHOT); partitions = []
    for table, parts in (('bars_1d', [str(y) for y in range(2021, 2027)]), ('adj_factors', None), ('adj_coverage', None)):
        entries = state['tables'][table]
        if parts is not None: require(set(parts) <= entries.keys(), 'DQN history years missing')
        for part, entry in sorted(entries.items()):
            if parts is not None and part not in parts: continue
            path = Path(root) / entry['file']; stat = path.stat()
            partitions.append({'table': table, 'part': part, 'entry': entry,
                'file': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'stock_receipt_file': str(STOCK_RECEIPT), 'stock_receipt_sha256': file_sha(STOCK_RECEIPT),
        'upstream': previous.binding(root, ACCEPTED), 'files': rows, 'partitions': partitions, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    import baostock as bs
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(bs if query['provider'] == 'baostock' else ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'queries': QUERIES, 'apis': rows}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); rows = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage'):
        entries = state['tables'].get(table, {})
        frame = store.load_state(state, table, filters=[('instrument', '=', '000065.SZ')]) if entries else pd.DataFrame()
        rows[table] = {'partitions': len(entries), 'rows': len(frame)}
    index = store.load_state(state, 'index_1d', filters=[('index', '=', '000001.SH')])
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, '000065.SZ': rows,
        'index_constituent_partitions': len(state['tables'].get('index_constituents', {})),
        '000001.SH_daily_rows': len(index),
        'model_files': sorted(str(p) for p in Path('repo').rglob('tgt_net.pt') if p.is_file()),
        'limits': ['Local snapshot inventory only; remote member archive not migrated',
            'No model loaded, no inference/trades/fills/NAV or cost scenarios computed']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); proofs = []; selected_frames = {}
    for row in bound['partitions']:
        path = Path(row['file']); before = file_sha(path); frame = pd.read_parquet(path)
        actual = fingerprint(frame); accepted = actual; restored = []
        if actual != row['entry']['sha']:
            candidate = frame.copy()
            for name in frame.select_dtypes(include=['datetime', 'datetimetz']).columns:
                if frame[name].dt.unit != 'ms': continue
                seconds = frame[name].dt.as_unit('s')
                if seconds.dt.as_unit('ms').equals(frame[name]): candidate[name] = seconds; restored.append(name)
            accepted = fingerprint(candidate)
        require(len(frame) == row['entry']['rows'] and accepted == row['entry']['sha'] and file_sha(path) == before,
            'DQN source full content differs')
        proofs.append({**row, 'sha256': before, 'actual_rows': len(frame), 'canonical_fingerprint': actual,
            'accepted_fingerprint': accepted, 'legacy_seconds_columns': restored})
        selected_frames.setdefault(row['table'], []).append(frame[frame.instrument.eq('000065.SZ')].copy())
    save(directory / 'partition-verification.json', {'status': 'ok', 'snapshot': SNAPSHOT,
        'input_binding_sha256': file_sha(directory / 'input-binding.json'), 'partitions': proofs, 'not_a_backtest': True})
    dqn = with_adjusted(pd.concat(selected_frames['bars_1d'], ignore_index=True),
        pd.concat(selected_frames['adj_factors'], ignore_index=True), pd.concat(selected_frames['adj_coverage'], ignore_index=True))
    require(len(dqn) > 7 and dqn.adjustment_status.eq('usable').all() and
        not dqn.duplicated(['date', 'instrument']).any(), 'DQN adjustment/history incomplete')
    require(not (directory / 'dqn-input.parquet').exists(), 'DQN copy exists')
    dqn.to_parquet(directory / 'dqn-input.parquet', index=False)
    for name, row in bound['files'].items():
        require(not (directory / name).exists(), 'Input copy exists'); shutil.copyfile(row['file'], directory / name)
    price = pd.read_parquet(directory / 'price-input.parquet'); profiles = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = directory / name; frame = pd.read_parquet(path)
        profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns),
            'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(directory / 'price-input.parquet'),
        'sha256': file_sha(directory / 'price-input.parquet'), 'rows': len(price), 'columns': list(price.columns),
        'first': price.date.min(), 'last': price.date.max(), 'pool': sorted(set(price.instrument)),
        'not_a_backtest': True, 'platform_equivalent': False, 'component_instrument': '000300.SH', 'profiles': profiles,
        'dqn': {'file': str(directory / 'dqn-input.parquet'),
            'sha256': file_sha(directory / 'dqn-input.parquet'), 'rows': len(dqn), 'first': dqn.date.min(), 'last': dqn.date.max(),
            'columns': list(dqn.columns), 'trading_rows': int(dqn.is_trading.sum())},
        'limits': ['Ten stock complete-traded-row arithmetic samples are not historical HS300 membership',
            'DQN close-only six returns do not prove default whole-frame dropna, tensor dtype or pretrained outputs',
            '000300 daily bars are only bull/bear operator samples, not original00000109:30 include_now inputs']})
    save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch42 full partition proofs, ten-stock operands and original000065 close operands frozen')


def input_frames(directory):
    price, excluded = stocks.inputs(directory); doc = read(directory / 'input-analysis.json'); row = doc['dqn']
    require(file_sha(row['file']) == row['sha256'], 'DQN input bytes differ')
    frame = pd.read_parquet(row['file']); proof = read(directory / 'partition-verification.json')
    require(proof['status'] == 'ok' and proof['snapshot'] == SNAPSHOT and proof['not_a_backtest'] is True and
        proof['input_binding_sha256'] == file_sha(directory / 'input-binding.json'), 'Partition proof binding differs')
    bound = read(directory / 'input-binding.json')
    require(len(proof['partitions']) == len(bound['partitions']), 'Partition proof count differs')
    for actual, expected in zip(proof['partitions'], bound['partitions'], strict=True):
        require(all(actual[k] == v for k, v in expected.items()) and actual['actual_rows'] == expected['entry']['rows'] and
            actual['accepted_fingerprint'] == expected['entry']['sha'], 'Partition proof metadata differs')
    require(len(frame) == row['rows'] and list(frame.columns) == row['columns'] and
        str(frame.date.min()) == row['first'] and str(frame.date.max()) == row['last'] and
        int(frame.is_trading.sum()) == row['trading_rows'] and set(frame.instrument) == {'000065.SZ'} and
        not frame.duplicated(['date', 'instrument']).any() and frame.adjustment_status.eq('usable').all(), 'DQN profile differs')
    frame = frame[frame.is_trading].sort_values('date').reset_index(drop=True)
    require(np.isfinite(frame[['close_adj', 'back_factor']]).all().all() and
        frame[['close_adj', 'back_factor']].gt(0).all().all(), 'DQN operands invalid')
    return price, frame, excluded, row['rows'] - len(frame)


def trend_kernel(directory):
    dumps = []; kernels = []
    for number in (0, 1):
        fn = next(n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name == 'check_stocks')
        nodes = []
        for name in ('x', 'y'):
            nodes.append(next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
        nodes.append(next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and
            isinstance(n.value.func, ast.Name) and n.value.func.id == 'linregress'))
        ratios = [next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == name for t in n.targets)) for name in ('s_fall', 's_vol_ratio')]
        condition = next(n.test for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp))
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        dumps.append(ast.dump(module) + ''.join(ast.dump(n) for n in ratios) + ast.dump(condition))
        kernels.append((compile(module, '<original-trend120>', 'exec'),
            [compile(ast.Expression(n.value), '<original-filter>', 'eval') for n in ratios],
            compile(ast.Expression(condition), '<original-trend-condition>', 'eval')))
    require(dumps[0] == dumps[1], 'Two original arithmetic kernels differ')
    return kernels[0], hashlib.sha256(dumps[0].encode()).hexdigest()


def original_trend(close, high, volume, kernel):
    require(len(close) == 120 and len(high) == 30 and len(volume) == 180 and
        np.isfinite(close).all() and np.isfinite(high).all() and np.isfinite(volume).all() and
        min(close) > 0 and min(high) > 0 and min(volume) >= 0, 'Invalid complete trend operands')
    code, ratios, condition = kernel
    ns = {'np': np, 'linregress': linregress, 'stock': 'sample',
        'attribute_history': lambda *a, **kw: {'close': np.asarray(close, dtype=float)}}
    exec(code, ns)
    fall = float(eval(ratios[0], {'high_max_30': max(high), 's_close_1': close[-1]}))
    vol = float(eval(ratios[1], {'df_vol': pd.DataFrame({'sample': volume})}).iloc[0])
    return [float(ns[n]) for n in ('slope', 'intercept', 'r_value')] + [fall, vol], [
        bool(close[-1] <= 500), bool(fall <= 1.1), bool(vol <= 1.5), bool(eval(condition, ns))]


def reference_trend(close, high, volume):
    close = list(map(float, close)); n = 120; mx = 59.5; my = math.fsum(close)/n
    sx = math.fsum((k-mx)**2 for k in range(n)); sy = math.fsum((v-my)**2 for v in close)
    cov = math.fsum((k-mx)*(v-my) for k, v in enumerate(close)); slope = cov/sx; intercept = my-slope*mx
    r = cov/math.sqrt(sx*sy) if sy else math.nan
    fall = max(high)/close[-1]; total = math.fsum(volume)
    vol = math.fsum(volume[-7:])/7/(total/180) if total else math.nan
    return [slope, intercept, r, fall, vol], [close[-1] <= 500, fall <= 1.1, vol <= 1.5,
        bool(np.divide(slope, intercept) > .005 and r > .9)]


def dqn_kernel(directory):
    dumps = []; modules = []
    for number in (2, 3):
        fn = next(n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name == 'get_action')
        nodes = [n for n in fn.body if isinstance(n, ast.Assign) and
            ((isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value == '涨跌幅') or
             (isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'dropna'))]
        require(len(nodes) == 2, 'DQN input kernel structure differs')
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])); dumps.append(ast.dump(module)); modules.append(module)
    require(dumps[0] == dumps[1], 'Duplicate DQN kernels differ')
    return compile(modules[0], '<original-dqn-close-input>', 'exec'), hashlib.sha256(dumps[0].encode()).hexdigest()


def bull_namespace(directory):
    ns = {'g': SimpleNamespace(MA=['000001.XSHG', 10], threshold=.005, is_bull=False)}
    selected(directory, 1, ('get_bull_bear_signal_minute',), ns); return ns


def diagnostics(directory):
    rows = []
    def add(name, **kw): rows.append({'case': name, **deepcopy(kw)})
    context = SimpleNamespace(previous_date='2021-01-04', current_dt='2021-01-05 09:30',
        portfolio=SimpleNamespace(positions={'old': object(), 'B': object()}, total_value=1000., cash=200.))
    for number in (0, 1):
        requests_seen = []; orders = []; logs = []
        ns = {'g': SimpleNamespace(max_hold_stock_nums=2, target_lists=[]), 'np': np, 'pd': pd,
            'linregress': linregress, 'log': SimpleNamespace(info=lambda x: logs.append(x)),
            'get_current_data': lambda: {s: SimpleNamespace(paused=False, low_limit=1., day_open=10., high_limit=1000.) for s in ('A', 'B', '688x', 'paused')},
            'get_security_info': lambda s: SimpleNamespace(display_name=s),
            'order_target_value': lambda *a: orders.append(list(a))}
        data = ns['get_current_data'](); data['paused'].paused = True; ns['get_current_data'] = lambda: data
        def members(*a, **kw):
            requests_seen.append({'args': list(a), 'kwargs': kw}); return list(data)
        ns['get_index_stocks'] = members
        values = {'A': 10.+.1*np.arange(120), 'B': 100.+1.2*np.arange(120)}
        def history(count, frequency, field, pool):
            if field == 'close': return pd.DataFrame({s: [values[s][-1]] for s in pool})
            if field == 'high': return pd.DataFrame({s: np.repeat(values[s][-1], count) for s in pool})
            return pd.DataFrame({s: np.repeat(100., count) for s in pool})
        ns['history'] = history; ns['attribute_history'] = lambda s, *a, **kw: {'close': values[s]}
        selected(directory, number, ('check_stocks', 'trade'), ns); ns['check_stocks'](context)
        add(f'trend{number}_date_and_absolute_slope_order', requests=requests_seen, targets=ns['g'].target_lists)
        ns['g'].target_lists = ['B', 'new']
        if number == 1:
            ns['get_bull_bear_signal_minute'] = lambda: setattr(ns['g'], 'is_bull', True)
            ns['trade'](context); add('bull_sells_before_return_no_buy', orders=orders)
            orders.clear(); ns['get_bull_bear_signal_minute'] = lambda: setattr(ns['g'], 'is_bull', False)
        ns['trade'](context); add(f'trend{number}_rejected_intents_sell_adjust_new', orders=orders)
    ns = bull_namespace(directory); requests_seen = []
    def bars(*a, **kw):
        requests_seen.append({'args': list(a), 'kwargs': kw}); return {'close': np.array([100.]*9+[102.])}
    ns['get_bars'] = bars; ns['get_bull_bear_signal_minute']()
    add('bull_original_request_includes_now', requests=requests_seen, is_bull=ns['g'].is_bull)
    ns['get_bars'] = lambda *a, **kw: {'close': np.array([100.]*10)}; ns['get_bull_bear_signal_minute']()
    add('bull_hysteresis_keeps_true_at_mean', is_bull=ns['g'].is_bull)
    ns['get_bars'] = lambda *a, **kw: {'close': np.array([100.]*9+[98.])}; ns['get_bull_bear_signal_minute']()
    add('bull_down_switch', is_bull=ns['g'].is_bull)
    _, code_sha = dqn_kernel(directory); code, _ = dqn_kernel(directory)
    sample = pd.DataFrame({'close': np.arange(7)+100., 'other': np.ones(7)})
    ns = {'df': sample.copy()}; exec(code, ns); add('dqn_close_input_six', dimensions=len(ns['df']), ast_sha256=code_sha)
    sample.loc[3, 'other'] = np.nan; ns = {'df': sample}; exec(code, ns)
    add('dqn_default_other_missing_reduces_input', dimensions=len(ns['df']))
    for action in (0, 1, 2):
        orders = []; logs = []
        ns = {'get_action': lambda ctx: action, 'order_target_value': lambda *a: orders.append(list(a)),
            'log': SimpleNamespace(info=lambda x: logs.append(x), debug=lambda x: None),
            'OrderStatus': SimpleNamespace(held='held'), 'g': SimpleNamespace()}
        selected(directory, 2, ('adjustment', 'order_target_value_', 'open_position', 'close_position', 'close_account'), ns)
        ctx = SimpleNamespace(portfolio=SimpleNamespace(cash=200.,
            positions={'000065.XSHE': SimpleNamespace(security='000065.XSHE')}))
        ns['adjustment'](ctx); add(f'dqn_synthetic_action{action}', orders=orders, logs=logs)
        if action == 2:
            ctx.portfolio.positions = {}
            try: ns['adjustment'](ctx)
            except KeyError as exc: add('dqn_empty_sell_fault', error=f'KeyError: {exc}')
            try: ns['close_account'](ctx)
            except AttributeError as exc: add('dqn_unscheduled_uninitialised_helper', error=f'AttributeError: {exc}')
        if action == 1:
            ns['order_target_value'] = lambda *a: SimpleNamespace(filled=1)
            add('dqn_partial_buy_true', result=ns['open_position']('000065.XSHE', 200.))
            ns['order_target_value'] = lambda *a: SimpleNamespace(status='held', filled=1, amount=100)
            add('dqn_partial_sell_false', result=ns['close_position'](SimpleNamespace(security='000065.XSHE')))
            ns['order_target_value'] = lambda *a: SimpleNamespace(status='held', filled=100, amount=100)
            add('dqn_full_held_sell_true', result=ns['close_position'](SimpleNamespace(security='000065.XSHE')))
    return {'not_a_backtest': True, 'platform_equivalent': False, 'model_loaded': False, 'cases': rows,
        'limits': ['Original selected functions with explicitly synthetic responses/actions and unchanged holdings; no network or ledger used',
            'Recorded target intents are not fills; synthetic DQN actions do not demonstrate pretrained inference']}


def compute(directory):
    validate_sources(directory); price, dqn, excluded, dqn_excluded = input_frames(directory)
    require(len(dqn) >= 7 and any(len(g) >= 180 for _, g in price.groupby('instrument')), 'Incomplete component history')
    (kernel, trend_sha), (dcode, dqn_sha) = trend_kernel(directory), dqn_kernel(directory)
    maxima = [0.]*5; boundaries = []; count = 0; passed = 0; ranges = []
    for instrument, group in price.groupby('instrument', sort=True):
        group = group.sort_values('date').reset_index(drop=True)
        for k in range(179, len(group)):
            factor = float(group.back_factor.iloc[k])
            close = group.close_adj.iloc[k-119:k+1].to_numpy()/factor
            high = group.high_adj.iloc[k-29:k+1].to_numpy()/factor
            volume = group.volume.iloc[k-179:k+1].to_numpy()
            actual, flags = original_trend(close, high, volume, kernel); expected, ref_flags = reference_trend(close, high, volume)
            for j, (a, b) in enumerate(zip(actual, expected, strict=True)):
                if math.isnan(a) and math.isnan(b): continue
                require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), 'Trend operator arithmetic differs')
                maxima[j] = max(maxima[j], abs(a-b))
            if flags != ref_flags: boundaries.append({'instrument': instrument, 'date': str(group.date.iloc[k]),
                'original': actual, 'reference': expected, 'flags': flags, 'reference_flags': ref_flags})
            count += 1; passed += all(flags)
        ranges.append({'instrument': instrument, 'windows': max(0, len(group)-179),
            'first': str(group.date.iloc[179]) if len(group) > 179 else None, 'last': str(group.date.iloc[-1])})
    dmax = 0.; examples = []
    for k in range(6, len(dqn)):
        values = dqn.close_adj.iloc[k-6:k+1].to_numpy()/float(dqn.back_factor.iloc[k])
        ns = {'df': pd.DataFrame({'close': values})}; exec(dcode, ns)
        actual = ns['df']['涨跌幅'].to_numpy(); expected = np.asarray([float(values[j])/float(values[j-1])-1 for j in range(1, 7)])
        require(len(actual) == 6 and np.isfinite(actual).all() and np.allclose(actual, expected, rtol=1e-12, atol=1e-12), 'DQN close arithmetic differs')
        dmax = max(dmax, float(np.max(np.abs(actual-expected))))
        if k in (6, len(dqn)-1): examples.append({'date': str(dqn.date.iloc[k]), 'returns': actual.tolist(), 'reference': expected.tolist()})
    index, _ = inputs(directory); ns = bull_namespace(directory); reference_state = False; mismatches = []; switches = 0
    for k in range(9, len(index)):
        values = index.close.iloc[k-9:k+1].to_numpy(dtype=float); old = ns['g'].is_bull
        ns['get_bars'] = lambda *a, **kw: {'close': values}; ns['get_bull_bear_signal_minute']()
        mean = math.fsum(values)/10
        if reference_state:
            if values[-1]*1.005 <= mean: reference_state = False
        elif values[-1] > mean*1.005: reference_state = True
        if bool(ns['g'].is_bull) != reference_state: mismatches.append(str(index.date.iloc[k]))
        switches += old != ns['g'].is_bull
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'trend': {'unique_windows': count, 'shared_by_two_sources': True, 'ranges': ranges, 'passed_all_filters': passed,
            'excluded_known_suspended_rows': excluded, 'max_differences': maxima, 'condition_boundaries': boundaries, 'ast_sha256': trend_sha},
        'dqn': {'instrument': '000065.SZ', 'unique_windows': len(dqn)-6, 'shared_by_two_duplicate_sources': True,
            'first': str(dqn.date.iloc[6]), 'last': str(dqn.date.iloc[-1]), 'excluded_known_suspended_rows': dqn_excluded,
            'max_difference': dmax, 'examples': examples, 'ast_sha256': dqn_sha, 'model_loaded': False, 'inference': False},
        'bull_operator': {'sample_instrument': '000300.SH', 'windows': len(index)-9, 'state_mismatches': mismatches, 'switches': int(switches)},
        'limits': ['Mixed120/30/180 complete traded-row stock arithmetic; not weekly pool rankings or strategy backtests',
            'Close-only DQN input arithmetic; no pretrained output, actual actions, tensor dtype or default-field contract verified',
            'Bull operator daily000300 sample does not replace original000001 include_now weekly09:30 bar'],
        'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch42/20261007-weekly-trend-dqn')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
