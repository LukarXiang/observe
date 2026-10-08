"""Freeze candle/weekly-breakout/margin-pair source and dependency evidence."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import datetime as dt
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
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts import review_strategy_batch38 as stocks
from scripts import review_strategy_batch42 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch42/20261007-weekly-trend-dqn')
RECEIPT = Path('docs/handoff/2026-10-07-batch42-verification.json')
SOURCES = ('2024年度精选策略2/37.三阳三阴战法.txt',
    '2024年度精选策略2/45.真真正正的诚意之作-2021超十倍.txt',
    '2020年度精选策略/55 中小板-中证500配对交易.txt')
EXPECTED_SOURCE_SHA = ('613bd63d6835299e162bd0086c57983ad92a746b876c6a1904e4e8b1f47703df',
    '6e33000c425cf0064144a477548c8361e970d230a26aff5f4a90eb7f383c7462',
    '4ae6210e9d2960ddd37c4c42797b71d09893208a947dde42d0bd4af998471ef0')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/wma.py'),
    ('repo/backtrader', 'backtrader/indicators/macd.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_margin_sse.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch43.py',
    'tests/unit/test_catalog_margin_weekly.py', 'tests/unit/test_batch43_research.py'}))
CORE = (previous.CORE - {'dqn-input.parquet', 'partition-verification.json'}) | {'weekly-input.parquet', 'weekly-input-verification.json'}
QUERIES = {
    'stock_minute': {'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600218',
        'start_date': '2026-09-28 09:30:00', 'end_date': '2026-09-29 15:00:00', 'period': '1', 'adjust': ''},
        'limit': 'Recent minute sample is not original all-market Tick/09:32/09:50/14:55 executions or MACD inputs'},
    'fund_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '510220', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'One fund daily sample cannot prove two-fund events, historical trading rules or borrowing'},
    'margin_detail': {'function': 'stock_margin_detail_sse', 'parameters': {'date': '20260929'},
        'limit': 'Aggregate public short-sale balances are not broker-specific availability, fees, margin or recalls'}}


def implementation(): return {name: file_sha(name) for name in FILES}

def preflight(root, directory):
    checkpoint(root, directory, 'Batch43 margin API/weekly-frequency and three candle/weekly/margin source reviews started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_margin_weekly.py', '--tb=no']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
        'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha('tests/unit/test_catalog_margin_weekly.py')})
    require(result.returncode == 1 and '12 failed, 8 passed' in result.stdout, 'Expected historical API regression differs')
    return {'status': 'ok', 'regression': '12 failed, 8 passed'}

def validate_scan(directory):
    before = read(directory / 'catalog-before.json')
    old = Path(before['directory']) / 'catalog.json'
    red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '12 failed, 8 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha('tests/unit/test_catalog_margin_weekly.py'), 'Red regression binding differs')
    scan_path = directory / 'api-scan.json'
    if scan_path.exists():
        doc = read(scan_path); path = Path(doc['catalog']['output']) / 'catalog.json'
        require(doc['old_catalog_file'] == str(old) and file_sha(old) == doc['old_catalog_sha256'] and
            file_sha(path) == doc['catalog_sha256'], 'Frozen scan catalogs changed')
        prior = {r['path']: r for r in read(old)}; rows = read(path); changes = []
        require(len(prior) == len(rows) == 695 and {r['path'] for r in rows} == set(prior), 'Frozen scan source set differs')
        for row in rows:
            previous_row = prior[row['path']]
            require(all(row[k] == v for k, v in previous_row.items() if k not in ('apis', 'frequencies', 'gaps', 'status')), 'Frozen scan source/manual metadata differs')
            delta = {k: {'before': previous_row[k], 'after': row[k]} for k in ('apis', 'frequencies', 'gaps', 'status') if row[k] != previous_row[k]}
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
        require(all(row[k] == v for k, v in prior.items() if k not in ('apis', 'frequencies', 'gaps', 'status')), 'Scan changed source/manual metadata')
        delta = {k: {'before': prior[k], 'after': row[k]} for k in ('apis', 'frequencies', 'gaps', 'status') if row[k] != prior[k]}
        if delta: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
    save(directory / 'api-scan.json', {'catalog': result, 'old_catalog_file': str(old_path), 'old_catalog_sha256': file_sha(old_path),
        'catalog_sha256': file_sha(path), 'changes': changes, 'not_a_backtest': True,
        'limits': ['Static calls can be in comments, strings or inactive functions; not proof of active borrowing or platform weekly bars']})
    return {'changed_sources': len(changes), 'catalog_id': result['catalog_id'],
        'progress': archive(root, 'Batch43 margin/weekly metadata correction across695 sources frozen')}

def start(root, directory):
    validate_scan(directory); bound = binding(root); apis = api_evidence()
    scan_doc = read(directory / 'api-scan.json')
    old_path = Path(scan_doc['catalog']['output']) / 'catalog.json'
    require(file_sha(old_path) == scan_doc['catalog_sha256'], 'API scan changed')
    save(directory / 'input-binding.json', bound); save(directory / 'existing-apis.json', apis)
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name, sha in zip(SOURCES, EXPECTED_SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and
            old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source changed/already reviewed')
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
            'use': 'Read-only WMA, MACD and public margin/fund interfaces; not platform bars, indicator seeds or actual borrowing'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch43 three new complete source reviews frozen; missing intraday/weekly/margin dependencies retained')

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
    return archive(root, 'Batch43 original candle/weekly arithmetic and state diagnoses frozen')

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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch43', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch43 existing stock-minute/fund/margin supplementation attempts frozen; no publication')

def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch43 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch43 forbidden-network arithmetic/diagnostic/catalog byte match')

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
    protection = protect(root, directory); progress = archive(root, 'Batch43 candle/weekly/margin research accepted; original trade dependencies remain missing')
    save(Path('docs/handoff/2026-10-07-batch43-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}

def worker(root, directory, endpoint):
    import akshare as ak
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
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()): frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch43_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}], limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row



def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch43.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch43_research.py',
            'tests/unit/test_catalog_margin_weekly.py', 'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_batch42_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch42 differs')
    files = {}
    for name in ('price-input.parquet', 'index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted operand changed')
        files[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': previous.binding(root, ACCEPTED), 'files': files, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'apis': rows}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); assets = ['510220.SH', '510500.SH', '600218.SH']; tables = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', assets)])
        tables[table] = {s: int(frame.instrument.eq(s).sum()) for s in assets}
    return {'snapshot': SNAPSHOT, 'requested_assets': assets, 'tables': tables,
        'snapshot_tables': sorted(state['tables']), 'index_constituent_partitions': len(state['tables'].get('index_constituents', {})),
        'ledger': {'file': 'src/observe/ledger/book.py', 'sha256': file_sha('src/observe/ledger/book.py'),
            'borrow_accounting_supported': False, 'limits': ['Book buys cash-funded positive shares and sells existing sellable shares',
                'No broker borrow inventory, margin, borrow-fee, recall or short-event obligation inputs integrated']},
        'not_a_backtest': True, 'limits': ['Daily/5m availability is not original Tick, current bars or historical full-market names',
            'Public margin aggregates cannot prove actual broker borrow eligibility or share-denominated requests']}


def completed_weeks(price, calendar):
    require(not price.duplicated(['instrument', 'date']).any(), 'Duplicate daily sample')
    cal = calendar.copy(); cal['_dt'] = pd.to_datetime(cal.date); cal['_week'] = cal['_dt'].dt.to_period('W-FRI')
    first, last = cal['_dt'].min().date(), cal['_dt'].max().date()
    sessions = cal[cal.is_open].groupby('_week', sort=True).date.apply(list).to_dict()
    rows = []; excluded = []; differences = [0.]*5
    for instrument, group in price.groupby('instrument', sort=True):
        group = group.sort_values('date').copy(); group['_week'] = pd.to_datetime(group.date).dt.to_period('W-FRI')
        stock_first = pd.to_datetime(group.date.iloc[0]).date()
        for period, sample in group.groupby('_week', sort=True):
            if period.start_time.date() < max(first, stock_first) or period.end_time.date() > last:
                excluded.append({'instrument': instrument, 'week': str(period), 'reason': 'boundary_week_not_fully_covered'}); continue
            expected = sessions.get(period, [])
            require(expected and sample.date.tolist() == expected, 'Unknown missing daily rows in completed week')
            traded = sample[sample.is_trading]
            if traded.empty:
                excluded.append({'instrument': instrument, 'week': str(period), 'reason': 'all_explicitly_suspended'}); continue
            operands = traded[['open_adj', 'high_adj', 'low_adj', 'close_adj', 'volume', 'back_factor']]
            require(np.isfinite(operands).all().all() and traded[['open_adj','high_adj','low_adj','close_adj','back_factor']].gt(0).all().all()
                and traded.volume.ge(0).all(), 'Unknown weekly operands')
            actual = [float(traded.open_adj.iloc[0]), float(traded.high_adj.max()), float(traded.low_adj.min()),
                float(traded.close_adj.iloc[-1]), float(traded.volume.sum())]
            expected_values = [float(traded.open_adj.tolist()[0]), max(map(float,traded.high_adj)), min(map(float,traded.low_adj)),
                float(traded.close_adj.tolist()[-1]), math.fsum(map(float,traded.volume))]
            for j,(a,b) in enumerate(zip(actual,expected_values,strict=True)):
                require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9), 'Weekly aggregate differs')
                differences[j] = max(differences[j],abs(a-b))
            rows.append({'instrument': instrument, 'date': str(period.end_time.date()), 'last_trade_date': str(traded.date.iloc[-1]),
                **dict(zip(('open_adj','high_adj','low_adj','close_adj','volume'),actual,strict=True)),
                'back_factor': float(traded.back_factor.iloc[-1]), 'source_daily_rows': len(sample), 'traded_rows': len(traded)})
    result = pd.DataFrame(rows)
    require(not result.empty and not result.duplicated(['instrument','date']).any(), 'No complete weekly operands')
    return result, {'excluded': excluded, 'max_aggregation_differences': differences,
        'rule': 'Calendar-complete Saturday-Friday sample weeks only; exclude first/last boundary weeks and wholly suspended weeks',
        'limits': ['Research operands only; no original include_now09:32 weekly bars or money/default fields reconstructed']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name,row in bound['files'].items():
        path = directory/name; require(not path.exists(), 'Input copy exists'); shutil.copyfile(row['file'],path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows':len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last':frame.date.max()}
    price = pd.read_parquet(directory/'price-input.parquet')
    weekly, proof = completed_weeks(price,pd.read_parquet(directory/'calendar-input.parquet'))
    path = directory/'weekly-input.parquet'; require(not path.exists(), 'Weekly input exists'); weekly.to_parquet(path,index=False)
    save(directory/'weekly-input-verification.json', {'snapshot': SNAPSHOT, 'file':str(path),'sha256':file_sha(path),
        'source_sha256': {n:file_sha(directory/n) for n in ('price-input.parquet','calendar-input.parquet')},
        'rows':len(weekly),'columns':list(weekly.columns),'not_a_backtest':True,'platform_equivalent':False,**proof})
    save(directory/'input-analysis.json', {'snapshot':SNAPSHOT, **profiles['price-input.parquet'], 'pool':sorted(set(price.instrument)),
        'not_a_backtest':True,'platform_equivalent':False,'profiles':profiles,
        'limits':['Ten stock samples are not original all-market historical pools or either fund',
            'No high_limit or money columns fabricated; pure price/volume formulas only',
            'Completed weekly aggregation does not supply partial weekly/current daily bars or intraday execution']})
    save(directory/'dependency-inventory.json',inventory(root))
    return archive(root,'Batch43 accepted long stock/calendar bytes and completed-week arithmetic operands frozen')


def inputs(directory):
    price, excluded = stocks.inputs(directory); doc = read(directory/'input-analysis.json'); profile=doc['profiles']['calendar-input.parquet']
    require(file_sha(profile['file'])==profile['sha256'], 'Calendar input differs'); calendar=pd.read_parquet(profile['file'])
    require(len(calendar)==profile['rows'] and list(calendar.columns)==profile['columns'], 'Calendar profile differs')
    proof=read(directory/'weekly-input-verification.json'); path=directory/'weekly-input.parquet'; weekly=pd.read_parquet(path)
    require(proof['snapshot']==SNAPSHOT and proof['not_a_backtest'] is True and proof['platform_equivalent'] is False and
        proof['file']==str(path) and file_sha(path)==proof['sha256'] and len(weekly)==proof['rows'] and list(weekly.columns)==proof['columns'] and
        proof['source_sha256']=={n:file_sha(directory/n) for n in ('price-input.parquet','calendar-input.parquet')}, 'Weekly binding differs')
    regenerated, check=completed_weeks(pd.read_parquet(directory/'price-input.parquet'),calendar)
    require(regenerated.equals(weekly) and all(proof[k]==v for k,v in check.items()), 'Weekly derivation differs')
    require(np.isfinite(price[['open_adj','high_adj','low_adj','close_adj','volume']]).all().all(), 'Daily operands invalid')
    return price, weekly, calendar, excluded


def compile_assignments(directory, number, function, names):
    fn=next(n for n in source_tree(directory,number).body if isinstance(n,ast.FunctionDef) and n.name==function)
    nodes=[]
    for name in names:
        found=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        require(len(found)==1,'Original arithmetic assignment ambiguous'); nodes.append(found[0])
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    return compile(module,'<original-arithmetic>','exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def candle_kernel(directory):
    return compile_assignments(directory,0,'pick_high_limit',('sum_plus_num_3','sum_plus_num_6',
        'df_min_close_30','df_max_high_30','df_max_close_30','df_min_low_30','rate_30','rate_valley'))


def original_candle(recent, past, kernel):
    require(len(recent)==6 and len(past)==30 and np.isfinite(recent[['open','close']]).all().all() and
        np.isfinite(past[['high','low','close']]).all().all(), 'Incomplete candle operands')
    ns={'df_panel_3':recent.iloc[-3:], 'df_panel_6':recent, 'df_panel_30':past}; exec(kernel,ns)
    return [int(ns['sum_plus_num_3']),int(ns['sum_plus_num_6']),float(ns['rate_30']),float(ns['rate_valley'])]


def ratio_kernel(directory):
    fn=next(n for n in source_tree(directory,1).body if isinstance(n,ast.FunctionDef) and n.name=='stock_filter')
    assignments=[n for n in fn.body if isinstance(n,ast.Assign) and
        ((isinstance(n.targets[0],ast.Name) and n.targets[0].id=='param') or isinstance(n.targets[0],ast.Subscript))]
    selection=next(n.value.func.value.slice for n in fn.body if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id=='stocks_df' and isinstance(n.value,ast.Call))
    require(len(assignments)==7,'Original weekly ratios differ')
    module=ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[]))
    return compile(module,'<original-weekly-ratios>','exec'),compile(ast.Expression(selection),'<original-weekly-condition>','eval'),hashlib.sha256((ast.dump(module)+ast.dump(selection)).encode()).hexdigest()


def original_ratios(sample, kernel):
    require(len(sample)==48,'Incomplete48-row operands')
    base=sample.iloc[:44]; gap=sample.iloc[44:47]; cur=sample.iloc[47]
    df=pd.DataFrame({'close_x':[cur.close_adj],'open':[cur.open_adj],'high':[base.high_adj.max()],
        'low':[base.low_adj.min()],'close_y':[gap.close_adj.max()],'volume_x':[cur.volume],
        'volume_y':[gap.volume.max()],'volume':[base.volume.max()]})
    ns={'check_data':df,'g':SimpleNamespace(cdh=1.35)}; exec(kernel[0],ns); flags=eval(kernel[1],ns)
    return df[['t_Raise','l_Raise','oc','range','vol_ratio','vol_r2']].iloc[0].to_numpy(dtype=float),bool(flags.iloc[0])


def reference_ratios(sample):
    base=sample.iloc[:44]; gap=sample.iloc[44:47]; cur=sample.iloc[47]
    high=max(map(float,base.high_adj)); low=min(map(float,base.low_adj))
    vals=[cur.close_adj/high,max(map(float,gap.close_adj))/high,cur.close_adj/cur.open_adj,high/low,
        float(np.divide(cur.volume,max(map(float,gap.volume)))),float(np.divide(cur.volume,max(map(float,base.volume))))]
    return vals,bool(1<vals[0]<1.35 and vals[1]<1 and vals[2]>1 and 1<vals[4]<2 and vals[5]>1)


def mean_namespace(directory):
    ns={}; selected(directory,1,('EMA_ratio',),ns); return ns


def original_means(values, ns):
    require(len(values)==10 and np.isfinite(values).all() and min(values)>0,'Incomplete weighted-mean operands')
    samples=[]
    def bars(stock,c_date,unit,count):
        frame=pd.DataFrame({'close':values[-count:]}); samples.append(frame); return frame
    ns['Pro_Get_Bars']=bars; result=ns['EMA_ratio']('sample',None,5,10)
    return [float(f.righted_num.sum()) for f in samples],result


def compute(directory):
    validate_sources(directory); price,weekly,calendar,excluded=inputs(directory)
    candle_code,candle_sha=candle_kernel(directory); ratio=ratio_kernel(directory); means=mean_namespace(directory)
    candle_max=[0.]*4; candle_bound=[]; candles=0; weekly_max=[0.]*6; week_bound=[]; weeks=0; skipped=0
    mean_max=[0.,0.]; mean_bound=[]; mean_count=0; shadow_count=0; shadow_bound=[]; ranges=[]
    shadow_ns={'g':SimpleNamespace(up_line=.4),'print':lambda *a:None}; selected(directory,1,('sell_condition',),shadow_ns)
    cal=calendar[calendar.is_open].copy(); periods=pd.to_datetime(cal.date).dt.to_period('W-FRI')
    complete_dates=sorted({str(p.end_time.date()) for p in periods if p.start_time.date()>=pd.to_datetime(calendar.date.min()).date()
        and p.end_time.date()<=pd.to_datetime(calendar.date.max()).date()})
    week_rank={day:k for k,day in enumerate(complete_dates)}
    for instrument,group in price.groupby('instrument',sort=True):
        group=group.sort_values('date').reset_index(drop=True); local=0
        for k in range(32,len(group)):
            factor=float(group.back_factor.iloc[k]); recent=group.iloc[k-5:k+1][['open_adj','close_adj']].rename(columns={'open_adj':'open','close_adj':'close'})/factor
            past=group.iloc[k-32:k-2][['high_adj','low_adj','close_adj']].rename(columns={'high_adj':'high','low_adj':'low','close_adj':'close'})/factor
            actual=original_candle(recent,past,candle_code)
            expected=[sum(c>=o for c,o in zip(recent.close.iloc[-3:],recent.open.iloc[-3:],strict=True)),
                sum(c>o for c,o in zip(recent.close,recent.open,strict=True)),
                (max(past.high)-min(past.low))/min(past.low),(max(past.close)-min(recent.close))/min(recent.close)]
            for j,(a,b) in enumerate(zip(actual,expected,strict=True)):
                require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'Candle arithmetic differs'); candle_max[j]=max(candle_max[j],abs(a-b))
            flags=lambda v:[v[0]==3,v[1]==3,1<v[2]<2.5,.15<v[3]<.4]
            if flags(actual)!=flags(expected): candle_bound.append({'instrument':instrument,'date':str(group.date.iloc[k])})
            candles+=1; local+=1
        ranges.append({'instrument':instrument,'daily_candle_windows':local,'first':str(group.date.iloc[32]),'last':str(group.date.iloc[-1])})
    for instrument,group in weekly.groupby('instrument',sort=True):
        group=group.sort_values('date').reset_index(drop=True)
        for k in range(9,len(group)):
            sample=group.iloc[k-9:k+1]; rank=week_rank[sample.date.iloc[-1]]
            if sample.date.tolist()!=complete_dates[rank-9:rank+1]: skipped+=1; continue
            values=sample.close_adj.to_numpy()/float(sample.back_factor.iloc[-1]); actual,state=original_means(values,means)
            expected=[math.fsum((j+1)*float(v) for j,v in enumerate(values[-n:]))/(n*(n+1)/2) for n in (5,10)]
            for j,(a,b) in enumerate(zip(actual,expected,strict=True)):
                require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),'Linear weighted mean differs'); mean_max[j]=max(mean_max[j],abs(a-b))
            if (state=='good')!=(expected[0]>expected[1]): mean_bound.append({'instrument':instrument,'date':sample.date.iloc[-1],'original':actual,'reference':expected})
            mean_count+=1
        for k in range(47,len(group)):
            sample=group.iloc[k-47:k+1]; rank=week_rank[sample.date.iloc[-1]]
            if sample.date.tolist()!=complete_dates[rank-47:rank+1]: skipped+=1; continue
            actual,flag=original_ratios(sample,ratio); expected,ref_flag=reference_ratios(sample)
            for j,(a,b) in enumerate(zip(actual,expected,strict=True)):
                require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'Weekly ratio differs'); weekly_max[j]=max(weekly_max[j],abs(a-b))
            if flag!=ref_flag: week_bound.append({'instrument':instrument,'date':sample.date.iloc[-1]})
            weeks+=1
        for row in group.itertuples(index=False):
            bar=pd.DataFrame({'open':[row.open_adj],'high':[row.high_adj],'low':[row.low_adj],'close':[row.close_adj]})
            shadow_ns['Pro_Get_Bars']=lambda *a,**kw:bar
            for threshold in (1.06,1.02):
                result=shadow_ns['sell_condition']('sample',None,'1w',threshold)
                width=row.high_adj-row.low_adj; upper=row.high_adj-max(row.open_adj,row.close_adj)
                expected='sell' if width!=0 and (row.open_adj/row.close_adj>threshold or upper/width>.4) else None
                if result!=expected: shadow_bound.append({'instrument':instrument,'date':row.date,'threshold':threshold})
                shadow_count+=1
    require(candles>0 and weeks>0 and mean_count>0,'Incomplete component history')
    return {'snapshot':SNAPSHOT,'not_a_backtest':True,'platform_equivalent':False,'source_sha256':list(EXPECTED_SOURCE_SHA),
        'candle':{'windows':candles,'ranges':ranges,'max_differences':candle_max,'condition_boundaries':candle_bound,'ast_sha256':candle_sha,
            'limits':['Only3/6/30price formulas with end30three rows before end6; no full pattern/high-limit or historical pool selection']},
        'weekly':{'rows':len(weekly),'first':weekly.date.min(),'last':weekly.date.max(),'ratio48_windows':weeks,
            'max_ratio_differences':weekly_max,'ratio_boundaries':week_bound,'ratio_ast_sha256':ratio[2],
            'linear_mean10_windows':mean_count,'max_mean_differences':mean_max,'mean_boundaries':mean_bound,
            'shadow_threshold_samples':shadow_count,'shadow_boundaries':shadow_bound,'skipped_windows_with_missing_complete_weeks':skipped},
        'excluded_known_suspended_daily_rows':excluded,
        'limits':['Ten stock samples only; no Tick/MACD/platform partial-week bars, borrowing, actual orders/fills/NAV/cost performance',
            'Funds55have no real arithmetic data; only explicitly synthetic zscore/action diagnoses',
            'Completed-week samples do not change original include_now intraday rules or provide a tradeable replacement']}


def diagnostics(directory):
    rows=[]
    def add(name,**kw): rows.append({'case':name,**deepcopy(kw)})
    orders=[]; requests_seen=[]
    ns={'np':np,'pd':pd,'datetime':dt,'timedelta':dt.timedelta,'help_stock':['A','B','C'],
        'log':SimpleNamespace(info=lambda *a:None),
        'get_current_data':lambda:{s:SimpleNamespace(day_open=100.,last_price=103.,high_limit=110.) for s in ('A','B','C')},
        'attribute_history':lambda *a,**kw:pd.DataFrame({'close':np.ones(10)*100.}),
        'order_value':lambda *a:orders.append(list(a))}
    def price(*a,**kw):
        requests_seen.append({'args':list(a),'kwargs':{k:str(v) if isinstance(v,dt.datetime) else v for k,v in kw.items()}})
        return pd.DataFrame({'low':[99.],'close':[100.],'high':[105.],'open':[100.],'high_limit':[110.],'money':[10000.]})
    ns['get_price']=price; selected(directory,0,('market_open',),ns)
    context=SimpleNamespace(current_dt=dt.datetime(2021,1,5,10),portfolio=SimpleNamespace(available_cash=10000.,positions={}))
    ns['market_open'](context)
    add('candle_rejected_buy_removes_and_skips_with_stale_cash',orders=orders,remaining=ns['help_stock'],requests=requests_seen)
    ns['help_stock']=['A','B','C','D']; context.current_dt=dt.datetime(2021,1,5,14,41); context.portfolio.available_cash=0.
    ns['market_open'](context); add('candle_tail_clear_skips_same_list',remaining=ns['help_stock'])
    ns['help_stock']=[]; time_requests=[]
    position=SimpleNamespace(closeable_amount=0,init_time=dt.datetime(2020,1,1),price=100.,avg_cost=100.)
    context.portfolio.positions={'A':position}; context.current_dt=dt.datetime(2021,1,5,14,40)
    ns['get_ticks']=lambda *a,**kw:time_requests.append('ticks') or {'current':[100.]}
    ns['get_price']=lambda *a,**kw:pd.DataFrame({'high':[100.], 'low':[90.], 'close':[100.]*1,'open':[100.],'high_limit':[110.]})
    ns['order_target']=lambda *a:None; ns['market_open'](context)
    add('candle1440_else_calls_ticks_with_zero_sellable',requests=time_requests)
    context.current_dt=dt.datetime(2021,1,5,14,40,1); time_requests.clear(); ns['market_open'](context)
    add('candle_after1440_checks_sellable',requests=time_requests)
    ns={'help_stock':['A'],'datetime':dt,'timedelta':dt.timedelta,
        'get_trade_days':lambda **kw:pd.bdate_range('2020-10-01','2021-01-04'),
        'get_all_securities':lambda *a:pd.DataFrame(index=['A']),
        'pick_high_limit':lambda *a:['A'], 'get_current_data':lambda:{'A':SimpleNamespace(is_st=False,paused=False)},
        'get_security_info':lambda s:SimpleNamespace(start_date=dt.date(2000,1,1))}
    selected(directory,0,('before_market_open','filter_st','filter_paused_stock','filter_stock_by_days'),ns)
    context.portfolio.positions={}; context.current_dt=dt.datetime(2021,1,5,8)
    ns['before_market_open'](context); ns['before_market_open'](context)
    add('candle_before_open_appends_duplicate_queue',queue=ns['help_stock'])
    ns['get_security_info']=lambda s:SimpleNamespace(start_date=context.current_dt.date()-dt.timedelta(days=1080 if s=='equal' else 1081))
    add('candle_listing_strict1080',accepted=ns['filter_stock_by_days'](context,['equal','older'],1080))
    flat={'pd':pd,'np':np,'timedelta':dt.timedelta,'get_price':lambda *a,**kw:pd.DataFrame({'close':[100.],'open':[99.],'high_limit':[110.],'low':[98.],'pre_close':[99.]})}
    selected(directory,0,('pick_high_limit',),flat)
    try: flat['pick_high_limit'](['600000.XSHG'],None,None,None,None,None)
    except KeyError as exc: add('candle_standard_frame_not_legacy_panel',error=f'KeyError: {exc}')
    ns={'g':SimpleNamespace(already_list=[],nb_list=[],weeks=48,security=[f'S{k}' for k in range(6)]),'np':np,'pd':pd,
        'stock_filter':lambda *a:[f'S{k}' for k in range(6)],'EMA_ratio':lambda *a:'good','sell_condition':lambda *a:None,
        'get_current_data':lambda:{f'S{k}':SimpleNamespace(last_price=100.,high_limit=110.) for k in range(6)}}
    orders=[]; ns['order_value']=lambda *a:orders.append(list(a)); selected(directory,1,('deal_stock',),ns)
    context.portfolio=SimpleNamespace(available_cash=10000.,positions={'old':object()})
    ns['deal_stock'](context); add('weekly_five_new_plus_old_not_total_cap',selected=ns['g'].already_list,attempted=ns['g'].nb_list,orders=orders)
    ns['g'].already_list=[]; ns['g'].nb_list=[]; ns['stock_filter']=lambda *a:['fail']; ns['EMA_ratio']=lambda *a:'bad'
    ns['get_current_data']=lambda:{'fail':SimpleNamespace(last_price=100.,high_limit=110.)}; orders.clear()
    ns['deal_stock'](context); ns['EMA_ratio']=lambda *a:'good'; ns['deal_stock'](context)
    add('weekly_failed_attempt_blocks_later_buy',attempted=ns['g'].nb_list,selected=ns['g'].already_list,orders=orders)
    ns={'g':SimpleNamespace(nb_list=['old']), 'datetime':dt,'pd':pd,
        'get_trade_days':lambda **kw:[], 'some_filter':lambda x:[], 'get_all_securities':lambda **kw:pd.DataFrame()}
    requests_seen=[]
    def securities(**kw): requests_seen.append(kw); return pd.DataFrame()
    ns['get_all_securities']=securities; selected(directory,1,('before_market_open',),ns)
    context.current_dt=dt.datetime(2021,1,5,8); ns['before_market_open'](context)
    add('weekly_tuesday_does_not_reset_attempts',attempted=ns['g'].nb_list,requests=requests_seen)
    context.current_dt=dt.datetime(2021,1,11,8); ns['before_market_open'](context)
    add('weekly_monday_resets_attempts',attempted=ns['g'].nb_list)
    bars_seen=[]; ns={}
    def bars(*a,**kw):
        bars_seen.append({'args':list(a),'kwargs':{k:str(v) if isinstance(v,dt.datetime) else v for k,v in kw.items()}})
        return pd.DataFrame({'close':[100.]},index=pd.MultiIndex.from_tuples([('A',0)]))
    ns['get_bars']=bars; selected(directory,1,('Pro_Get_Bars',),ns)
    frame=ns['Pro_Get_Bars'](['A'],dt.datetime(2021,1,5,9,32),'1w',48)
    add('weekly_original_current_bar_and_multiindex',requests=bars_seen,columns=list(frame.columns),codes=frame.code.tolist())
    mean_ns=mean_namespace(directory); actual,state=original_means(np.arange(10,dtype=float)+100.,mean_ns)
    add('weekly_original_linear_weight_not_ema',values=actual,state=state)
    ns={'g':SimpleNamespace(up_line=.4),'Pro_Get_Bars':lambda *a:pd.DataFrame({'open':[100.],'close':[100.],'high':[100.],'low':[100.]})}
    selected(directory,1,('sell_condition',),ns)
    add('weekly_zero_range_no_sell',result=ns['sell_condition']('A',None,'1w',1.06))
    ns={'g':SimpleNamespace(), 'np':np, 'pd':pd, 'mean':np.mean}
    selected(directory,2,('set_params','set_variables','get_signal','change_positions','handle_data','z_test'),ns)
    ns['set_params'](); ns['set_variables'](); orders=[]
    def order(kind): return lambda *a,**kw:orders.append({'kind':kind,'args':list(a),'kwargs':kw})
    for name in ('marginsec_open','marginsec_close','order_value','order_target_value'):ns[name]=order(name)
    context.portfolio=SimpleNamespace(portfolio_value=200000.)
    for z in (-2.10000001,-2.1,-2.,-1.95,0.,1.95,2.,2.1,2.10000001):
        ns['z_test']=lambda:z; add(f'margin_signal_{z}',result=ns['get_signal']())
    ns['g'].state='empty'; ns['change_positions']('buy1',context); ns['change_positions']('buy1',context)
    add('margin_repeated_buy1_additive_intents',orders=orders,state=ns['g'].state)
    orders.clear(); ns['change_positions']('mid',context); add('margin_fixed_share_close_and_unchecked_state',orders=orders,state=ns['g'].state)
    orders.clear(); ns['change_positions']('buy2',context); add('margin_buy2_uses_asset_value_as_share_count',orders=orders)
    orders.clear(); calls=[]; sequence=iter(['buy1','buy2'])
    def signal(): value=next(sequence); calls.append(value); return value
    ns['get_signal']=signal; ns['handle_data'](context,None)
    add('margin_handle_calls_twice_second_drives_orders',signals=calls,orders=orders)
    original={'g':SimpleNamespace(security1='A',security2='B',test_days=120,regression_ratio=1),'np':np,'mean':np.mean}
    selected(directory,2,('z_test',),original)
    sample=np.arange(120,dtype=float); original['attribute_history']=lambda s,*a:pd.DataFrame({'close':100.+(sample if s=='A' else sample+np.sin(sample))})
    value=original['z_test'](); spread=np.sin(sample); expected=(spread[-1]-math.fsum(map(float,spread))/120)/math.sqrt(math.fsum((float(v)-math.fsum(map(float,spread))/120)**2 for v in spread)/120)
    add('margin_synthetic120_population_std_shape',shape=list(value.shape),value=float(value[0]),reference=expected)
    original.pop('mean')
    try: original['z_test']()
    except NameError as exc:add('margin_mean_requires_platform_injection',error=f'NameError: {exc}')
    return {'not_a_backtest':True,'platform_equivalent':False,'cases':rows,
        'limits':['All responses/actions/holdings explicitly synthetic; recorded intents are not fills or cost/NAV evidence',
            'No Tick/MACD data, actual platform weekly/current bars, broker short execution or equivalent pool reconstructed']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'scan', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', default='data/staging/strategies-batch43/20261007-candle-weekly-margin')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
