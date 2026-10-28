"""Freeze intraday reversal and external price-volume factor dependencies."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import inspect
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from sklearn.decomposition import PCA

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch81 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch82/20261009-intraday-reversal-price-volume')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2022年度精选策略/64.传统反转因子的改进与实测.txt',
    '2023年度精选策略/95.【复现】高频价量相关性，意想不到的选股因子.txt',
    '2024年度精选策略2/19.【复现】高频价量相关性，意想不到的选股因子.txt')
SOURCE_SHA = ('b45d5a4be1f77cbe58ad69ea3405a5063bd22ca9be70cad4d45ecb7c449266a8',
    '24a5fc73eaf8929ec62bd773a83b4b9d81492f7b1514c2b76e372c8b2a52684c',
    '24a5fc73eaf8929ec62bd773a83b4b9d81492f7b1514c2b76e372c8b2a52684c')
REFERENCES = (('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'),
    ('repo/rqalpha', 'rqalpha/portfolio/account.py'))
QUERIES = {'minute-600519': {'function': 'stock_zh_a_hist_min_em', 'parameters': {
    'symbol': '600519', 'period': '1', 'adjust': '',
    'start_date': '2005-01-05 09:30:00', 'end_date': '2026-09-29 15:00:00'}}}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch82.py',
    'tests/unit/test_batch82_research.py', 'scripts/review_strategy_batch73.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'existing-apis.json', 'dependency-inventory.json', 'source-reviews/review.json', 'catalog-impact.json',
    'diagnostics.json', 'research-summary.json', 'probe-results.json', 'candidate-audit.json',
    'offline-verification.json', 'offline-catalog.json', 'development-api-regression/failure.json',
    'development-diagnostic-regression/failure.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch81-verification.json')
    review = Path('docs/handoff/2026-10-09-batch81-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted81 binding changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def api_evidence():
    import akshare as ak
    code = inspect.getsource(ak.stock_zh_a_hist_min_em)
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES,
        'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest(),
        'policy': 'One bounded native1m GET,8/10s,parent45s; no retry/redirect/proxy or headers archived',
        'limits': ['period1 uses ndays5; start/end only slice response, not a historical query',
            'cpv.csv factor formula and historical available_at are not present in these sources',
            'No5m, daily-close, current-index or synthetic-factor substitution; no publication']}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted evidence changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    for repository, name in REFERENCES: require((Path(repository)/name).is_file(), 'Reference missing')
    checkpoint(root, directory, 'Batch82 parent81 accepted binding verified; baseline frozen before catalog edits')
    save(directory/'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    save(directory/'existing-apis.json', api_evidence())
    return {'status': 'ok', 'controls': len(read(directory/'baseline.json')['controls']),
        'partitions': len(read(directory/'baseline.json')['partitions']), 'not_a_backtest': True}


def api_regression(root, directory):
    binding(root, directory); folder = directory/'development-api-regression'; folder.mkdir(exist_ok=False)
    names = ('src/observe/strategy_catalog.py', 'tests/unit/test_batch82_research.py')
    for name in names: shutil.copyfile(name, folder/Path(name).name)
    command = [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch82_research.py']
    log = folder/'red.log'
    with log.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 1 and '4 failed, 2 passed' in log.read_text(), 'Expected calendar detection regression absent')
    save(folder/'failure.json', {'command': command, 'returncode': result.returncode,
        'log': str(log), 'sha256': file_sha(log), 'implementation_before': {n: file_sha(n) for n in names},
        'frozen_copies': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
        'not_a_backtest': True})
    return {'status': 'expected_failure_archived', 'failed': 4, 'passed': 2}


def diagnostic_regression(root, directory):
    binding(root, directory); folder = directory/'development-diagnostic-regression'; folder.mkdir(exist_ok=False)
    names = ('scripts/review_strategy_batch82.py', 'tests/unit/test_batch82_research.py')
    for name in names: shutil.copyfile(name, folder/Path(name).name)
    command = [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch82_research.py']
    log = folder/'red.log'
    with log.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 1 and '2 failed, 34 passed' in log.read_text(), 'Expected mutable diagnostic regression absent')
    save(folder/'failure.json', {'command': command, 'returncode': result.returncode,
        'log': str(log), 'sha256': file_sha(log), 'implementation_before': {n: file_sha(n) for n in names},
        'frozen_copies': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
        'root_cause': 'diagnostics stores schedule/slippage lists by reference then mutates them in a later controlled call',
        'not_a_backtest': True})
    return {'status': 'expected_failure_archived', 'failed': 2, 'passed': 34}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    command = ['rg', '--files', '--hidden', '--no-ignore', 'repo', str(root),
        '-g', '*[cC][pP][vV]*', '-g', '!**/.git/**']
    search = subprocess.run(command, capture_output=True, text=True, check=False)
    require(search.returncode in (0, 1), 'External factor search failed')
    files = search.stdout.splitlines(); manifests = []
    for path in sorted((Path(root)/'handoff').glob('*.manifest.json')):
        doc = read(path); matches = []
        def visit(value):
            if isinstance(value, str) and re.search(r'cpv\.csv', value, re.I): matches.append(value)
            elif isinstance(value, list):
                for child in value: visit(child)
            elif isinstance(value, dict):
                for child in value.values(): visit(child)
        visit(doc)
        manifests.append({'file': str(path), 'sha256': file_sha(path), 'cpv_matches': matches})
    minutes = []
    for key, entry in state['tables'].get('bars_5m', {}).items():
        path = Path(root)/entry['file']; metadata = pq.read_metadata(path); stat = path.stat()
        require(metadata.num_rows == entry['rows'], 'Minute physical/registered rows differ')
        minutes.append({'partition': key, 'file': str(path), 'registered_rows': entry['rows'],
            'physical_rows': metadata.num_rows, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'store_fingerprint': entry['sha'], 'frequency': '5m', 'fresh_file_sha256': False})
    universe = store.load_state(state, 'minute_universe')
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'minute_partitions': minutes, 'minute_universe_rows': len(universe),
        'minute_universe_years': sorted(int(y) for y in universe.year.unique()),
        'external_factor_search': {'command': command, 'status': search.returncode, 'matches': files,
            'files': {n: file_sha(n) for n in files if Path(n).is_file()}, 'archive_member_search': manifests},
        'original_1m_endpoint_rows_admitted': 0, 'original_cpv_rows_admitted': 0,
        'strict_historical_CSI800': False, 'not_a_backtest': True,
        'limits': ['Parquet footer rows/stat checked, not a fresh full44M-row scan or byte hash',
            'Published snapshot has5m, no1m or historical members table; old remote member payload still missing',
            'Search covers actual filenames and handoff manifests, not unknown archive contents or remote platform user files',
            'Dynamic-price adjustment,1m10:30/15:00 and every_bar ordering must be proved before64 trades',
            'No default pv_corr formula/lookback/vintage/available_at in95/19; no synthetic cpv.csv replacement']}


def catalog_impact(old, current):
    require(len(old) == len(current) == 695 and set(old) == {r['path'] for r in current}, 'Catalog scope changed')
    rows = []; candidates = []
    for item in current:
        prior = old[item['path']]; fields = sorted(k for k in set(prior) | set(item) if prior.get(k) != item.get(k))
        text, _ = read_source(Path('repo/量化策略源代码')/item['path'])
        candidate = bool(re.search(r'\bget_all_trade_days\s*\(', text))
        require([a for a in item['apis'] if a != 'get_all_trade_days'] == prior['apis'], 'Unrelated APIs changed')
        require(('get_all_trade_days' in item['apis']) == candidate, 'Calendar detection differs')
        if candidate: candidates.append(item['path'])
        if item['path'] not in SOURCES: require(set(fields) <= {'apis'} and (not fields or candidate), 'Unrelated review/status changed')
        def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rows.append({'path': item['path'], 'before_sha256': digest(prior), 'after_sha256': digest(item),
            'changed_fields': fields, 'calendar_text_candidate': candidate,
            'calendar_ast_called': 'get_all_trade_days' in (item['called_functions'] or []),
            'differences': {k: {'before': prior.get(k), 'after': item.get(k)} for k in fields}})
    return {'rows': rows, 'total_sources': len(rows), 'calendar_candidates': candidates,
        'changed_sources': sum(bool(r['changed_fields']) for r in rows), 'newly_reviewed': list(SOURCES),
        'policy': 'All695 compared; source fingerprints and existing manual reviews/status unchanged'}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    require(Path(rows[1]['source_copy']).read_bytes() == Path(rows[2]['source_copy']).read_bytes(), 'Duplicate bytes differ')
    (directory/'references').mkdir(exist_ok=False)
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    save(directory/'catalog-impact.json', catalog_impact(old, current)); save(directory/'dependency-inventory.json', inventory(root))
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'distinct_source_contents': 2, 'duplicate_mapping': {SOURCES[2]: SOURCES[1]}, 'not_a_backtest': True})
    return archive(root, 'Batch82 three complete sources frozen, two contents; all695 calendar API changes archived; original inputs blocked')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 3 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/SOURCES[number]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['review'] == REVIEWS[SOURCES[number]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    for label in ('api', 'diagnostic'):
        path = directory/f'development-{label}-regression/failure.json'
        if label == 'diagnostic' and not path.exists(): continue
        failure = read(path)
        require(failure['returncode'] == 1 and file_sha(failure['log']) == failure['sha256'], 'Red regression changed')
        for name, sha in failure['frozen_copies'].items(): require(file_sha(name) == sha, 'Red copy changed')


def namespace(directory, number):
    row = read(directory/'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen function source changed')
    tree = ast.parse(read_source(Path(row['source_copy']))[0]); nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (8 if number == 0 else 6) and all(safe_function(n) for n in nodes), 'Unsafe/missing functions')
    ns = {'pd': pd, 'np': np, 'PCA': PCA, 'BytesIO': BytesIO, 'g': SimpleNamespace(),
        'log': SimpleNamespace(set_level=lambda *a: None), 'set_benchmark': lambda *a: None,
        'set_option': lambda *a: None, 'set_order_cost': lambda *a, **kw: None,
        'OrderCost': lambda **kw: kw, 'run_daily': lambda *a, **kw: None,
        'set_redeem_latency': lambda **kw: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch82-functions>', 'exec'), ns)
    return ns


def controlled_context(holdings=None, cash=100., previous_date=None):
    positions = {s: SimpleNamespace(value=float(v)) for s, v in (holdings or {}).items()}
    return SimpleNamespace(current_dt=dt.datetime(2020, 2, 5, 10, 30),
        previous_date=previous_date if previous_date is not None else pd.Timestamp('2020-02-04'),
        portfolio=SimpleNamespace(positions=positions, long_positions=positions,
            available_cash=float(cash), total_value=float(cash)+sum((holdings or {}).values())))


def diagnostic_orders(ns, ctx, mode):
    orders = []; p = ctx.portfolio
    def order(kind, stock, amount):
        orders.append([kind, stock, float(amount)])
        if mode == 'delayed': return None
        if mode == 'failed': return SimpleNamespace(status='synthetic_rejected')
        if kind == 'order_target' and amount == 0:
            if stock in p.positions: p.available_cash += p.positions.pop(stock).value
        else:
            prior = p.positions.get(stock, SimpleNamespace(value=0.)).value
            value = prior+amount if kind == 'order_value' else amount
            if value-prior > p.available_cash: return SimpleNamespace(status='synthetic_rejected')
            p.available_cash -= value-prior; p.positions[stock] = SimpleNamespace(value=value)
        return SimpleNamespace(status='synthetic_immediate')
    for kind in ('order_target', 'order_value', 'order_target_value'):
        ns[kind] = lambda s, v, k=kind: order(k, s, v)
    return orders


def captured_error(function):
    try: function()
    except (NameError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError, RuntimeError, FileNotFoundError) as exc:
        return {'type': type(exc).__name__, 'message': str(exc)}
    return None


def synthetic_factor(values=(3., 1., 2.), codes=('A', 'B', 'C')):
    index = pd.MultiIndex.from_product([[pd.Timestamp('2020-02-04')], codes], names=['date', 'code'])
    return pd.DataFrame({'pv_corr': values}, index=index)


def controlled95(directory, frame, *, factor='pv_corr', count=200, index_pool=None,
                 previous_date=None, holdings=None, mode='delayed'):
    ns = namespace(directory, 1); ctx = controlled_context(holdings, previous_date=previous_date)
    ns['set_params'](ctx); ns['g'].factor_df = frame; ns['g'].factor_name = factor; ns['g'].hold_num = count
    ns['g'].is_in_index = None if index_pool is None else 'synthetic.index'
    ns['get_index_stocks'] = lambda *a: index_pool
    orders = diagnostic_orders(ns, ctx, mode); error = captured_error(lambda: ns['TradeFunc'](ctx))
    return {'orders': orders, 'error': error, 'final_cash': ctx.portfolio.available_cash,
        'final_holdings': list(ctx.portfolio.positions), 'synthetic_only': True, 'not_a_backtest': True,
        'factor': factor, 'previous_date_type': type(ctx.previous_date).__name__, 'mode': mode}


def price_diagnostic(directory, *, array_available=False, observe_sort=False, early_calendar=False, invalid=None):
    ns = namespace(directory, 0); days = list(pd.bdate_range('2020-01-02', periods=25).date)
    ns['get_all_trade_days'] = lambda: days; ns['params_set'](); ns['g'].stock_pool = ['A', 'B', 'C']
    ctx = controlled_context(); ctx.current_dt = dt.datetime.combine(days[5 if early_calendar else -1], dt.time(10, 30))
    calls = []; observed = {}
    def get_price(pool, start, end, frequency, fields):
        calls.append({'pool': pool, 'start': start, 'end': end, 'frequency': frequency, 'fields': fields})
        values = np.ones(3) if '10:30' in start else np.array([2., 1., .5])
        if invalid == 'nan': values[0] = np.nan
        if invalid == 'zero' and '10:30' in start: values[0] = 0.
        return {'close': values}
    ns['get_price'] = get_price
    if array_available: ns['array'] = np.asarray
    if observe_sort:
        # Capture the obsolete sort and float slice without choosing an economic correction.
        class SliceObserver:
            def __getitem__(self, key):
                observed['slice'] = {'start': key.start, 'stop': key.stop, 'step': key.step}
                raise TypeError('Observed original float slice; no integer correction authorized')
        class Frame(pd.DataFrame):
            def sort_index(self, *a, **kw):
                require(kw == {'by': 'rate'}, 'Original sorting call changed')
                observed['rates'] = {s: ('NaN' if pd.isna(v) else 'Infinity' if np.isinf(v) else float(v))
                    for s, v in self['rate'].items()}
                return SimpleNamespace(index=SliceObserver())
        ns['pd'] = SimpleNamespace(DataFrame=Frame)
    with np.errstate(divide='ignore', invalid='ignore'): error = captured_error(lambda: ns['get_buy_list'](ctx))
    return {'error': error, 'price_calls': calls, 'observed': observed,
        'array_injected': array_available, 'sort_observation_adapter': observe_sort,
        'synthetic_only': True, 'not_a_backtest': True, 'original_signal_or_target_generated': False}


def diagnostics(directory):
    validate_sources(directory); rows = []; ns = namespace(directory, 0); ctx = controlled_context()
    days = list(pd.bdate_range('2020-01-02', periods=25).date); ns['get_all_trade_days'] = lambda: days
    schedules = []; costs = []; ns['run_daily'] = lambda fn, **kw: schedules.append([fn.__name__, kw])
    ns['set_order_cost'] = lambda v, **kw: costs.append([v, kw]); ns['initialize'](ctx)
    rows.append({'case': '64_defaults', 'schedules': schedules, 'costs': costs,
        'initial_clock': ns['g'].trade_clock, 'initial_if_trade': ns['g'].if_trade,
        'frequency': ns['g'].fre, 'backwindow': ns['g'].backwindow})
    ns['get_index_stocks'] = lambda code, **kw: [code, str(kw['date'])]
    ns['before_market_open'](ctx); rows.append({'case': '64_pool_date', 'arguments': ns['g'].stock_pool})
    traded = []; ns['Trade'] = lambda *a: traded.append(ns['g'].trade_clock)
    for _ in range(21): ns['market_open'](ctx)
    require(traded == [1, 10, 20], 'Original callback frequency differs')
    rows.append({'case': '64_callback_clock', 'trade_callbacks': traded, 'callbacks': 21, 'same_synthetic_day': str(ctx.current_dt.date())})
    ns = namespace(directory, 0); ns['get_all_trade_days'] = lambda: days; ns['params_set']()
    ns['Trade'] = lambda *a: (_ for _ in ()).throw(ValueError('synthetic_trade_failure'))
    error = captured_error(lambda: ns['market_open'](ctx))
    rows.append({'case': '64_trade_failure_keeps_flag', 'error': error, 'clock': ns['g'].trade_clock, 'if_trade': ns['g'].if_trade})
    for label, kwargs in (('implicit_array', {}), ('obsolete_sort', {'array_available': True}),
        ('float_slice', {'array_available': True, 'observe_sort': True}),
        ('missing_skipna', {'array_available': True, 'observe_sort': True, 'invalid': 'nan'}),
        ('zero_divisor', {'array_available': True, 'observe_sort': True, 'invalid': 'zero'}),
        ('short_calendar', {'array_available': True, 'early_calendar': True})):
        rows.append({'case': '64_'+label, **price_diagnostic(directory, **kwargs)})
    ns = namespace(directory, 0)
    rows.append({'case': '64_unused_undefined_factor', 'error': captured_error(lambda: ns['creat_hold_list'](ctx))})
    for mode in ('immediate', 'delayed', 'failed'):
        for holdings in ({}, {'OLD': 300.}, {'A': 300.}):
            ns = namespace(directory, 0); ctx = controlled_context(holdings); ns['get_buy_list'] = lambda *a: ['A', 'B']
            orders = diagnostic_orders(ns, ctx, mode); error = captured_error(lambda: ns['Trade'](ctx))
            rows.append({'case': '64_orders_'+mode+'_'+('empty' if not holdings else next(iter(holdings))),
                'orders': orders, 'error': error, 'final_cash': ctx.portfolio.available_cash,
                'final_holdings': list(ctx.portfolio.positions), 'synthetic_only': True, 'not_a_backtest': True})
    ns = namespace(directory, 0); ctx = controlled_context({'OLD': 300.}); ns['get_buy_list'] = lambda *a: []
    orders = diagnostic_orders(ns, ctx, 'delayed'); ns['Trade'](ctx)
    rows.append({'case': '64_empty_targets_sell_only', 'orders': orders})
    ns = namespace(directory, 1); ctx = controlled_context(); schedules = []; ns['run_daily'] = lambda fn, **kw: schedules.append([fn.__name__, kw])
    ns['read_file'] = lambda name: (_ for _ in ()).throw(FileNotFoundError(name))
    error = captured_error(lambda: ns['initialize'](ctx))
    rows.append({'case': '95_initialize_missing_csv', 'error': error, 'schedules': schedules.copy()})
    ns['read_file'] = lambda name: b'date,code,pv_corr\n2020-02-04,A,1\n2020-02-04,B,2\n'
    ns['initialize'](ctx)
    rows.append({'case': '95_initialize_synthetic_csv', 'schedules': schedules, 'hold_num': ns['g'].hold_num,
        'factor_name': ns['g'].factor_name, 'is_in_index': ns['g'].is_in_index,
        'rows': len(ns['g'].factor_df), 'index_names': list(ns['g'].factor_df.index.names), 'synthetic_only': True})
    for label, frame, kwargs in (
        ('ascending', synthetic_factor(), {}),
        ('date_type', synthetic_factor(), {'previous_date': dt.date(2020, 2, 4)}),
        ('absent_date', synthetic_factor(), {'previous_date': pd.Timestamp('2020-02-03')}),
        ('retained_empty_level', synthetic_factor().iloc[0:0], {}),
        ('zero_target_count', synthetic_factor(), {'count': 0}),
        ('empty_index_pool', synthetic_factor(), {'index_pool': []}),
        ('nan_under200', synthetic_factor((np.nan, 1., 2.)), {}),
        ('all_nan_under200', synthetic_factor((np.nan, np.nan, np.nan)), {}),
        ('index_subset', synthetic_factor(), {'index_pool': ['C', 'B']}),
        ('index_missing_code', synthetic_factor(), {'index_pool': ['B', 'MISSING']}),
        ('missing_column', synthetic_factor(), {'factor': 'missing'}),
        ('pca_fillzero', synthetic_factor((np.nan, 1., 2.)).assign(other=[1., 0., 2.]), {'factor': 'pca_pv'}),
        ('duplicate_codes', synthetic_factor((3., 1., 2.), ('A', 'A', 'B')), {}),
        ('over200', synthetic_factor(tuple(float(204-i) for i in range(205)), tuple(f'S{i:03d}' for i in range(205))), {})):
        rows.append({'case': '95_'+label, **controlled95(directory, frame, **kwargs)})
    for mode in ('immediate', 'delayed', 'failed'):
        rows.append({'case': '95_orders_'+mode, **controlled95(directory, synthetic_factor(), holdings={'OLD': 300.}, mode=mode)})
    for stamp in ('2008-12-31 23:59:59', '2009-01-01 00:00:00', '2009-01-01 00:00:01',
        '2011-01-01 00:00:00', '2011-01-01 00:00:01', '2013-01-01 00:00:00', '2013-01-01 00:00:01'):
        ns = namespace(directory, 1); commissions = []; slips = []; ns['datetime'] = dt
        ns['set_slippage'] = lambda v: slips.append(v); ns['FixedSlippage'] = lambda v: v
        ns['PerTrade'] = lambda **kw: kw; ns['set_commission'] = lambda v: commissions.append(v)
        ns['before_trading_start'](SimpleNamespace(current_dt=pd.Timestamp(stamp).to_pydatetime()))
        rows.append({'case': '95_cost_'+stamp, 'commissions': commissions.copy(), 'fixed_absolute_slippage': slips.copy()})
    ns.pop('datetime'); error = captured_error(lambda: ns['before_trading_start'](ctx))
    rows.append({'case': '95_implicit_datetime', 'error': error, 'slippage_called_before_failure': len(slips) == 2})
    return {'cases': rows, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'policy': 'Only frozen function definitions; top-level profile/imports not run; all CSV/prices/portfolio states synthetic; no missing real inputs replaced'}


def expected_request(case):
    require(case in QUERIES, 'Unknown minute query')
    return 'https://push2his.eastmoney.com/api/qt/stock/trends2/get', {
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58', 'ut': '7eea3edcaed734bea9cbfc24409ed989',
        'ndays': '5', 'iscr': '0', 'secid': '1.600519'}


def invoke(case):
    import akshare as ak
    q = QUERIES[case]; return getattr(ak, q['function'])(**q['parameters'])


def worker(directory, case):
    folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        target, params = expected_request(case)
        require(not row['attempts'] and method.upper() == 'GET' and url == target and kwargs.get('params') == params, 'Unexpected/repeated minute request')
        row['attempts'].append({'method': 'GET', 'url': url, 'parameters': params})
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder/'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code,
            'encoding': response.encoding}); response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'Installed API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder/'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    require(not (directory/'probe-results.json').exists(), 'Probe archive exists'); rows = []
    for case in QUERIES:
        require(not (directory/'probes'/case).exists(), 'Probe already attempted')
        cmd = [sys.executable, '-m', 'scripts.review_strategy_batch82', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            save(directory/f'worker-{case}.json', {'command': cmd, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/f'worker-{case}.json', {'command': cmd, 'status': 'timeout', 'timeout_seconds': 45})
            folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'result.json').exists():
                save(folder/'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout',
                    'attempts': [], 'wire_responses': [], 'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')})
        rows.append(read(directory/'probes'/case/'result.json'))
    require(before == implementation(), 'Implementation changed during probe')
    save(directory/'probe-results.json', {'results': rows, 'implementation_sha256': before, 'not_a_backtest': True})
    save(directory/'candidate-audit.json', verify_candidates(directory))
    return archive(root, 'Batch82 bounded native1m coverage probe archived; original CSV and history gates retained')


def verify_candidates(directory):
    require(api_evidence() == read(directory/'existing-apis.json'), 'Installed API changed')
    rows = read(directory/'probe-results.json')['results']; profiles = []
    require(len(rows) == len(QUERIES) and {r['case'] for r in rows} == set(QUERIES), 'Probe scope changed')
    for row in rows:
        require(row['query'] == QUERIES[row['case']] and row['api_sha256'] == file_sha(directory/'existing-apis.json') and
            row['strict_usable'] is False and row['published'] is False, 'Query binding/admission changed')
        target, params = expected_request(row['case'])
        require(len(row['attempts']) <= 1 and all(r == {'method': 'GET', 'url': target, 'parameters': params} for r in row['attempts']), 'Request scope changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial evidence changed')
        profile = {'case': row['case'], 'status': row['status'], 'original_rows_admitted': 0, 'complete22_day_windows': 0}
        if row['status'] in ('success', 'empty'):
            require(len(row['attempts']) == len(row['wire_responses']) == 1 and file_sha(row['file']) == row['sha256'], 'Native parse binding changed')
            calls = []
            def replay(session, method, url, **kwargs):
                require(not calls and method.upper() == 'GET' and url == target and kwargs.get('params') == params, 'Unexpected replay')
                calls.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes()
                response.encoding = wire['encoding']; response.url = url; response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay): frame = invoke(row['case'])
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(calls) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Parse scope changed')
            times = pd.to_datetime(frame['时间']); require(times.notna().all() and not times.duplicated().any(), 'Minute timestamps invalid')
            require(times.between(pd.Timestamp('2005-01-05 09:30'), pd.Timestamp('2026-09-29 15:00')).all(), 'Minute cutoff slice changed')
            endpoints = times.dt.strftime('%H:%M').isin(['10:30', '15:00'])
            grouped = times[endpoints].dt.date.value_counts(); complete = int(grouped.eq(2).sum())
            require(times.dt.date.nunique() <= 5, 'Native ndays5 coverage contract changed')
            profile.update(rows=len(frame), first=str(times.min()) if len(frame) else None,
                last=str(times.max()) if len(frame) else None, native_parser_replayed=True,
                endpoint_rows=int(endpoints.sum()), paired_endpoint_days=complete,
                complete22_day_windows=0)
        else: profile.update(native_parser_replayed=False, failure=row.get('error', 'worker deadline expired'))
        profiles.append(profile)
    return {'profiles': profiles, 'not_a_backtest': True, 'new_requests': len(rows),
        'original_input_rows_admitted': 0, 'default_cpv_formula_available': False}


def summary(directory):
    diag = diagnostics(directory); inv = read(directory/'dependency-inventory.json'); candidates = verify_candidates(directory)
    return {'diagnostic_cases': len(diag['cases']), 'source_contents': 2, 'newly_reviewed_sources': 3,
        'registered5m_rows': inv['table_rows'].get('bars_5m', 0),
        'external_factor_filename_matches': len(inv['external_factor_search']['matches']),
        'original64_factor_windows': 0, 'original95_factor_rankings': 0,
        'new_trading_backtests': 0, 'cost_backtests': [], 'not_a_backtest': True,
        'candidates': candidates, 'original_strategy_complete': False,
        'blockers': ['64 original1m22-day endpoints/dynamic adjustment and historical CSI800 absent; original sort/float slice unresolved',
            '95/19 external CSV/formula/lookback/vintage/available_at absent;10:30 trade input and date type unproved',
            'Dictionary iteration/immediate cash/partial orders/default slippage need platform evidence or confirmed variant']}


def study(root, directory):
    require(all(not (directory/n).exists() for n in ('diagnostics.json', 'research-summary.json')), 'Study archive exists')
    binding(root, directory); before = implementation(); diag = diagnostics(directory); result = summary(directory)
    require(before == implementation(), 'Implementation changed during study')
    save(directory/'diagnostics.json', diag); save(directory/'research-summary.json', {**result, 'implementation_sha256': before})
    return archive(root, 'Batch82 original callback/rank/PCA/cost/defect diagnostics archived; no full original input or trading result')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch82 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); result = summary(directory); candidates = verify_candidates(directory)
        dep = inventory(root); catalog = offline_catalog(root, directory)
    require(before == implementation() and diag == read(directory/'diagnostics.json') and
        {**result, 'implementation_sha256': before} == read(directory/'research-summary.json') and
        candidates == read(directory/'candidate-audit.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch82 forbidden-network original diagnostics/native response/inventory/three catalogs match0; originals blocked')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch82.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch82_research.py', 'tests/unit/test_batch81_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(not (directory/'checked-state.json').exists(), 'Checked archive exists'); folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for number, command in enumerate(commands()):
        path = folder/f'check-{number}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        copied = directory/'implementation-checked'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == current == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    result = read(directory/'research-summary.json')
    require(result['new_trading_backtests'] == result['original64_factor_windows'] == result['original95_factor_rankings'] == 0 and
        result['newly_reviewed_sources'] == 3 and result['source_contents'] == 2 and result['diagnostic_cases'] > 0 and
        result['not_a_backtest'] is True and result['original_strategy_complete'] is False and
        result['cost_backtests'] == [], 'Research/trading scope differs')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch82 intraday reversal and duplicate external factor sources accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch82-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'diagnostics': read(directory/'diagnostics.json'), 'research': result, 'inventory': read(directory/'dependency-inventory.json'),
        'candidates': read(directory/'candidate-audit.json'), 'catalog_impact': read(directory/'catalog-impact.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 3, 'distinct_source_contents': 2,
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'api_regression', 'diagnostic_regression', 'start', 'worker', 'probe', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
