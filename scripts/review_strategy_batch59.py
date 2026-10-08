"""Research original TD states and intraday IF dependencies, without trading."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
from fractions import Fraction
import hashlib
import inspect
from io import StringIO
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

from observe.data.store import Store, fingerprint
from observe.execution import sessions
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch52 import dependency_state as historical_dependencies
from scripts.review_strategy_batch58 import definitions, digest
from scripts import review_strategy_batch58 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch58/20261007-gold-stock')
RECEIPT = Path('docs/handoff/2026-10-07-batch58-verification.json')
SOURCES = ('2021年度精选策略/74.迪马克TD趋势反转指标.txt', '2022年度精选策略/31.再次抛砖.txt')
SOURCE_SHA = ('0402ab122e750fdf8ccee93faff4c71f25d07afd98cb59ad866ecf2a7dbfddc2',
    'b853d47a00c90780501aeb32c1bd112374628c830e0ca70a16cdbbf6058aa582')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/basicops.py'), ('repo/vnpy', 'vnpy/trader/object.py'),
    ('repo/akshare', 'akshare/futures/futures_zh_sina.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch59.py', 'tests/unit/test_batch59_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'preflight-failure-binding.json',
    'index-input.parquet', 'index-input-binding.json', 'source-reviews/review.json', 'existing-apis.json',
    'dependency-inventory.json', 'supplement-decision.json', 'td-states.parquet', 'ma-ratios.parquet',
    'component-research.json', 'diagnostics.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
SEEDS = ((0, 0), (1, 0), (1, 12), (1, 13), (-1, 0), (-1, 12), (-1, 13))
STATE_KEYS = ('setup', 'count', 'setup_high', 'setup_low')


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch58 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def index_operand(root):
    store = Store(root); state = store.state(SNAPSHOT); rows = []
    for name, entry in sorted(state['tables']['index_1d'].items()):
        path = Path(root) / entry['file']; sha = file_sha(path)
        content = pd.read_parquet(path); logical_sha = fingerprint(content)
        require(logical_sha == entry['sha'] and len(content) == entry['rows'], 'Index logical content or rows differ')
        rows.append({'partition': name, 'file': str(path), 'sha256': sha, 'content_fingerprint': logical_sha, 'rows': entry['rows']})
    frame = store.load_state(state, 'index_1d'); frame['date'] = frame.date.astype(str)
    frame = frame.sort_values(['index', 'date']).reset_index(drop=True)
    require(frame['index'].unique().tolist() == ['000300.SH'] and len(frame) == 5280 and not frame.date.duplicated().any(), 'Original index coverage differs')
    days = [str(d) for d in sessions(store.load_state(state, 'calendar')) if frame.date.min() <= str(d) <= frame.date.max()]
    require(frame.date.tolist() == days, 'Index calendar has missing or reordered rows')
    return frame, {'snapshot': SNAPSHOT, 'partitions': rows, 'rows': len(frame), 'columns': list(frame),
        'first': frame.date.min(), 'last': frame.date.max(), 'calendar_sessions': len(days), 'no_internal_missing_dates': True,
        'original_index': '000300.XSHG', 'local_index': '000300.SH', 'not_a_backtest': True,
        'limits': ['Index daily bars are original signal operands only; indices cannot be traded as IF contracts',
            'Pre-IF-listing2005 data remain arithmetic samples; no IF portfolio/execution history inferred']}


def preflight(root, directory):
    bound = binding(root); frame, operand = index_operand(root)
    checkpoint(root, directory, 'Batch59 TD/IF source review started; original index daily input checked')
    save(directory / 'input-binding.json', bound)
    failure = Path('data/staging/strategies-batch59/20261007-TD-IF-preflight-failure-01')
    save(directory / 'preflight-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    path = directory / 'index-input.parquet'; require(not path.exists(), 'Index input exists'); frame.to_parquet(path, index=False)
    save(directory / 'index-input-binding.json', {**operand, 'frozen_sha256': file_sha(path)})
    return {'status': 'ok', 'index_rows': len(frame)}


def api_evidence():
    import akshare as ak
    rows = []
    for name in ('futures_zh_minute_sina', 'futures_zh_daily_sina', 'match_main_contract'):
        fn = getattr(ak, name); source = inspect.getsource(fn)
        rows.append({'function': name, 'signature': str(inspect.signature(fn)), 'source': source,
            'sha256': hashlib.sha256(source.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'apis': rows,
        'limits': ['FewMinLine has only symbol/period; current main matching does not supply original historical ordered contracts/expiration',
            'Daily continuous IF0 would not supply actual contract rollover/minute fills/settlement/margin/fees',
            'Interface inspected only; prior DNS failure reused without repeating identical host requests']}


def supplements():
    folder = Path('data/staging/strategies-batch56/20261007-convertible-futures')
    receipt = read('docs/handoff/2026-10-07-batch56-verification.json'); source = folder / 'probe-results.json'
    require(file_sha(source) == receipt['checks']['evidence_sha256']['probe-results.json'], 'Accepted futures probe changed')
    rows = [r for r in read(source)['results'] if r['endpoint'].startswith('if')]
    require(len(rows) == 2 and all(r['status'] == 'failed' and 'NameResolutionError' in r['error'] and
        not r['files'] and not r['wire_responses'] and not r['published'] and not r['strict_usable'] for r in rows), 'Prior DNS diagnosis changed')
    for row in rows: require(read(folder / 'probes' / row['endpoint'] / 'result.json') == row, 'Prior worker changed')
    return {'prior_file': str(source), 'prior_sha256': file_sha(source), 'prior_futures_probes': rows,
        'new_requests': 0, 'new_samples': 0, 'published': False, 'not_a_backtest': True,
        'decision': 'Reuse verified DNS failures for the same futures host; changed symbols/parsers cannot repair name resolution',
        'missing': ['historical_ordered_IF_contracts_and_expiration', 'IF_09_30_and_seven_intraday_execution_prices',
            'index_intraday_current_bar_and4080one_minute_history', 'historical_contract_settlement_margin_tick_multiplier_fees']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'historical_equity_dependencies': historical_dependencies(root), 'ledger_sha256': file_sha('src/observe/ledger/book.py'),
        'not_a_backtest': True, 'limits': ['TD13day signal operands available, intraday signal/traded IF/rollover and settlement unavailable',
            'Unique Book has no proved futures contract/margin/settlement semantics; no synthetic fees or new ledger added',
            'Historical shares/PCF/names remain mandatory for micro400 and small-value100; current samples are not final completion']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = ast.parse(text)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True, 'definition_ast': definitions(tree)})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only extrema/stable mean and contract/order/interface references, not alternate execution or TD replacement'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch59 two TD/IF full source reviews and true-index operands frozen')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy']))
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Definition evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    operand = read(directory / 'index-input-binding.json')
    require(file_sha(directory / 'index-input.parquet') == operand['frozen_sha256'], 'Index operand changed')
    for row in operand['partitions']: require(file_sha(row['file']) == row['sha256'], 'Source partition changed')
    failure = read(directory / 'preflight-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure evidence scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'External preflight failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 3 passed' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red regression binding differs')


def selected(directory, number, names, ns):
    import __future__
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) for n in nodes), 'Unsafe or missing selected function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-TD-IF-functions>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns


def td_seed(close, high, low, setup, count):
    return {'setup': setup, 'count': count, 'setup_high': float(max(high[-9:-1])) if setup == -1 else 0.,
        'setup_low': float(min(low[-9:-1])) if setup == 1 else 0.}


def td_reference(close, high, low, seed):
    out = seed.copy(); first = out['setup']; c = close[-1]
    positive = [a > b for a, b in zip(close[4:], close[:-4], strict=True)]
    negative = [a < b for a, b in zip(close[4:], close[:-4], strict=True)]
    if all(negative[:-1]) and positive[-1] and not out['setup']:
        out.update(setup=-1, setup_high=float(max(high[-8:])))
    elif all(positive[:-1]) and negative[-1] and not out['setup']:
        out.update(setup=1, setup_low=float(min(low[-8:])))
    if out['setup'] == -1 and out['count'] != 13:
        out['count'] += int(c < low[-3])
    elif out['setup'] == 1 and out['count'] != 13:
        out['count'] += int(c > high[-3])
    if out['setup'] == -1 and out['count'] != 13:
        if c > out['setup_high']: out.update(setup=0, setup_high=0., count=0)
        elif all(positive): out.update(setup=1, count=0, setup_high=0., setup_low=float(min(low[-8:])))
        elif not first and all(negative): out['count'] = 0
    elif out['setup'] == 1 and out['count'] != 13:
        if c < out['setup_low']: out.update(setup=0, setup_low=0., count=0)
        elif all(negative): out.update(setup=-1, count=0, setup_low=0., setup_high=float(max(high[-8:])))
        elif not first and all(positive): out['count'] = 0
    return out


def td_original(ns, frame, seed):
    g = SimpleNamespace(n1=8, n2=13, code='000300.XSHG', today=frame.date.iloc[-1], **seed); ns['g'] = g
    views = {(field, n): frame[[field]].tail(n) for field, n in (('close', 13), ('high', 8), ('low', 8), ('high', 3), ('low', 3))}
    ns['get_price'] = lambda code, end_date, count, fields: views[(fields, count)]
    ns['calc_setup']()
    return {k: getattr(g, k) for k in STATE_KEYS}


def arithmetic(directory):
    validate_sources(directory); frame = pd.read_parquet(directory / 'index-input.parquet')
    td = selected(directory, 0, ['calc_setup'], {}); ma = selected(directory, 1, ['MA_Slope'], {'np': np,
        'g': SimpleNamespace(nMaSlope=5), 'log': SimpleNamespace(info=lambda *a: None)})
    td_rows = []; ratios = []; blocked = {'TD13': [], 'MA6': []}; boundaries = []; max_error = 0.
    for end in range(5, len(frame)):
        six = frame.iloc[end-5:end+1]; values = six.close.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values > 0).all(): blocked['MA6'].append(frame.date.iloc[end])
        else:
            bars = np.empty(6, dtype=[('close', '<f8')]); bars['close'] = values
            ma['get_bars'] = lambda *a, **kw: bars
            actual = float(ma['MA_Slope']('000300.XSHG')); reference = math.fsum(values[1:])/math.fsum(values[:-1])*100
            require(math.isfinite(actual) and math.isclose(actual, reference, rel_tol=1e-13, abs_tol=1e-12), 'MA ratio differs')
            exact = sum(map(Fraction.from_float, values[1:]))/sum(map(Fraction.from_float, values[:-1]))*100
            original_flags = [actual > 100.2, actual < 98.6]
            exact_flags = [exact > Fraction.from_float(100.2), exact < Fraction.from_float(98.6)]
            if original_flags != exact_flags: boundaries.append({'date': frame.date.iloc[end], 'original_flags': original_flags, 'exact_flags': exact_flags})
            max_error = max(max_error, abs(actual-reference))
            ratios.append({'date': frame.date.iloc[end], 'value': actual, 'independent_value': reference,
                'above_long_ratio': original_flags[0], 'below_short_ratio': original_flags[1]})
        if end < 12: continue
        window = frame.iloc[end-12:end+1]; prices = window[['close', 'high', 'low']].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or not (prices > 0).all(): blocked['TD13'].append(frame.date.iloc[end]); continue
        close, high, low = (window[k].to_numpy(dtype=float) for k in ('close', 'high', 'low'))
        for setup, count in SEEDS:
            seed = td_seed(close, high, low, setup, count); actual = td_original(td, window, seed); expected = td_reference(close, high, low, seed)
            require(actual == expected, 'TD transition differs')
            td_rows.append({'date': frame.date.iloc[end], 'seed_setup': setup, 'seed_count': count, **actual})
    summary = {'td_windows': (len(frame)-12)-len(blocked['TD13']), 'td_state_cases': len(td_rows), 'seed_states': [list(s) for s in SEEDS],
        'td_state_differences': 0, 'td_records_sha256': digest(td_rows), 'ma_windows': len(ratios), 'ma_records_sha256': digest(ratios),
        'ma_max_absolute_error': max_error, 'ratio_threshold_boundaries': boundaries, 'blocked_dates': blocked, 'not_a_backtest': True,
        'platform_equivalent': False, 'limits': ['TD states are independent seeded unit cases on actual13row index windows, not a market-open/close trading trajectory',
            'Seed bounds use the previous8rows, not recovered historical setup/holding state; no orders or portfolio returns computed',
            'MA6 uses complete end-of-day observations only as mean-ratio arithmetic; original include_now=True intraday bars not supplied',
            'Ratio flags alone are not original long signal: last1m and17*240minute mean remain missing',
            'Strict comparisons/original numeric behavior preserved; exact-fraction boundary evidence does not silently repair rules']}
    return summary, pd.DataFrame(td_rows), pd.DataFrame(ratios)


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; schedules = []; declarations = []
    log = SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, set_level=lambda *a: None)
    def order(kind):
        def call(*a, **kw): orders.append({'kind': kind, 'args': list(a), 'kwargs': kw}); return None
        return call
    def declaration(name):
        def call(*a, **kw): declarations.append({'name': name, 'args': list(a), 'kwargs': kw}); return name
        return call
    base = {'np': np, 'log': log, 'set_benchmark': lambda *a: None, 'set_option': lambda *a: None,
        'set_subportfolios': lambda *a: None, 'set_order_cost': lambda *a, **kw: None, 'set_slippage': lambda *a: None,
        'SubPortfolioConfig': declaration('SubPortfolioConfig'), 'OrderCost': declaration('OrderCost'),
        'StepRelatedSlippage': declaration('StepRelatedSlippage'), 'order': order('order'), 'order_target': order('target'),
        'order_target_value': order('value'), 'run_daily': lambda fn, **kw: schedules.append({'function': fn.__name__, **kw})}
    namespaces = []
    for number in range(2):
        names = [n.name for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef)]
        namespaces.append(selected(directory, number, names, {**base, 'g': SimpleNamespace()}))
    td, intraday = namespaces; context = SimpleNamespace(current_dt=datetime.datetime(2021, 6, 9, 9, 0),
        portfolio=SimpleNamespace(starting_cash=1000000., total_value=1000000., cash=100000., long_positions={}, short_positions={}, positions={}))
    for number, ns in enumerate(namespaces):
        declarations.clear(); schedules.clear(); ns['initialize'](context)
        require(next(r for r in declarations if r['name']=='OrderCost')['kwargs']['close_today_commission'] == (.0023 if number==0 else .0046), 'Declared fees differ')
        rows.append({'case': f'initialize_source_{number}_declarations', 'declarations': declarations.copy(), 'schedules': schedules.copy(), 'fees_not_calculated': True})
    td['attribute_history'] = lambda code, count, *a: pd.DataFrame({'close': [1200.] * count})
    hands = td['calc_hands'](context); require(hands == 9, 'TD sizing fixture differs')
    context.portfolio.total_value = 1.; require(td['calc_hands'](context) == 0, 'TD minimum hands changed'); context.portfolio.total_value = 1000000.
    rows.append({'case': 'td_index_price_sizing_and_zero_hands', 'hands': hands, 'zero_capital_hands': 0, 'synthetic_equity': True})
    td['get_future_contracts'] = lambda symbol: ['IF2106.CCFX', 'IF2107.CCFX']
    for remaining, expected in ((10, 'IF2106.CCFX'), (9, 'IF2107.CCFX')):
        td['get_security_info'] = lambda code: SimpleNamespace(end_date=context.current_dt.date()+datetime.timedelta(days=remaining))
        td['before_market_open'](context); require(td['g'].main == expected, 'TD10day boundary changed')
    rows.append({'case': 'td_main_contract_strict_ten_days', 'ten_days': 'IF2106.CCFX', 'nine_days': 'IF2107.CCFX'})
    def state(**kw):
        defaults = {'code': '000300.XSHG', 'n1': 8, 'n2': 13, 'setup': 0, 'count': 0, 'flag': 0, 'price': 0., 'p': 0.,
            'hold': '', 'main': 'IF2106.CCFX', 'sub': 'IF2107.CCFX', 'hands': 1, 'trend': 0, 'lc': False, 'setup_high': 0., 'setup_low': 0.}
        td['g'] = SimpleNamespace(**(defaults | kw)); orders.clear()
    current = 100.; td['get_current_data'] = lambda: {'000300.XSHG': SimpleNamespace(last_price=current), 'IF2107.CCFX': SimpleNamespace(last_price=115.)}
    td['get_security_info'] = lambda code: SimpleNamespace(end_date=context.current_dt.date()+datetime.timedelta(days=30))
    td['attribute_history'] = lambda code, count, *a: pd.DataFrame({'close': [110.]*count, 'high': [90.]*count, 'low': [120.]*count})
    state(setup=1, count=13); td['market_open'](context)
    require(td['g'].flag == 1 and td['g'].hold == 'IF2106.CCFX' and len(orders)==1, 'TD failed entry state differs')
    rows.append({'case': 'td_none_entry_still_updates_flag_and_hold', 'orders': orders.copy(), 'flag': td['g'].flag, 'fills': 0})
    state(setup=1, count=13, flag=1, hold='IF2106.CCFX', trend=1, price=90., p=90.); td['market_open'](context)
    require([r['kind'] for r in orders] == ['target', 'target', 'order'] and td['g'].trend == -1, 'TD double-close reversal differs')
    rows.append({'case': 'td_long_reversal_duplicates_close_intent', 'orders': orders.copy(), 'fills': 0})
    for setup, price, expected in ((1, 80., 'long'), (-1, 130., 'short')):
        state(setup=setup, count=13, flag=1, hold='IF2106.CCFX', trend=-1, price=price, p=100.); td['market_open'](context)
        require(len(orders)==1 and orders[0]['args'][-1] == expected and td['g'].hold == '', 'TD reverse close side differs')
        rows.append({'case': f'td_reverse_exit_setup_{setup}_wrong_side', 'orders': orders.copy(), 'actual_held_side': 'short' if setup==1 else 'long', 'fills': 0})
    state(setup=-1, count=13, flag=1, hold='IF2106.CCFX', trend=1, price=100., p=100.)
    td['get_security_info'] = lambda code: SimpleNamespace(end_date=context.current_dt.date()+datetime.timedelta(days=4))
    td['market_open'](context)
    require(td['g'].hold == 'IF2107.CCFX' and orders[0]['args'][-1]=='short', 'TD roll intent differs')
    rows.append({'case': 'td_roll_gap_uses_new_future_minus_index', 'gap': 15., 'orders': orders.copy(), 'price_after_callback': td['g'].price,
        'same_callback_can_reverse_after_roll': len(orders)>2, 'fills': 0})
    # Counting to13 suppresses the cancellation check within the same original callback.
    frame = pd.DataFrame({'date': ['synthetic']*13, 'close': np.arange(20., 7., -1), 'high': np.arange(20., 7., -1)+1, 'low': np.arange(20., 7., -1)-1})
    seed = {'setup': -1, 'count': 12, 'setup_high': 1., 'setup_low': 0.}
    actual = td_original(td, frame, seed); require(actual['count']==13 and actual['setup']==-1, 'TD count-before-cancel differs')
    rows.append({'case': 'td_increment_to_thirteen_skips_cancellation', 'before': seed, 'after': actual, 'synthetic_state': True})
    queries = []; intraday['attribute_history'] = lambda *a, **kw: (queries.append({'args': list(a), 'kwargs': kw}) or pd.DataFrame({'close': [1., 2.]}))
    require(intraday['getStockAvPrice']('000300.XSHG', 17)==1.5 and intraday['getCurrentPrice']('000300.XSHG')==2., 'Minute fixture differs')
    rows.append({'case': 'intraday_4080_minutes_and_last_minute_queries', 'queries': queries.copy(), 'synthetic_minutes': True})
    intraday['g'] = SimpleNamespace(nMa=17, nMaSlope=5, SlopeLong=100.2, SlopeShort=98.6,
        IF_current_month='IF2107.CCFX', IF_next_quarter='IF2109.CCFX')
    intraday['MA_Slope'] = lambda code: 98.; intraday['getCurrentPrice'] = lambda code: 100.; intraday['getStockAvPrice'] = lambda *a: 50.
    require(intraday['get_signal'](context)==-1, 'Short signal gained a price gate')
    rows.append({'case': 'intraday_short_signal_ignores_price_above_average', 'signal': -1, 'current_price': 100., 'minute_mean': 50.})
    intraday['get_signal'] = lambda context: -1; context.current_dt = datetime.datetime(2021,6,9,9,31)
    context.portfolio.long_positions = {'OLD': SimpleNamespace()}; context.portfolio.positions = context.portfolio.long_positions.copy(); orders.clear()
    intraday['handle_data'](context, None)
    require(len(orders)==1 and orders[0]['kwargs']['side']=='short' and orders[0]['args'][1]==20000., 'Opposite-side entry intent differs')
    rows.append({'case': 'intraday_can_open_short_while_long_exists', 'orders': orders.copy(), 'synthetic_orders': True})
    context.portfolio.long_positions = {'A': object(), 'B': object()}; intraday['get_days_long'] = lambda *a: 2; orders.clear()
    intraday['sell_long_pos'](context); require(len(orders)==1 and orders[0]['args'][0]=='A', 'First eligible return differs')
    rows.append({'case': 'intraday_close_helper_returns_after_first_eligible', 'orders': orders.copy(), 'remaining_eligible': ['B']})
    times = ((9,31),(10,31),(11,1),(13,5),(13,35),(14,5),(14,35)); closed = []
    intraday['get_signal'] = lambda context: 0
    for name in ('sell_all_long','sell_all_short','sell_long_pos','sell_short_pos'):
        intraday[name] = lambda context, name=name: closed.append(name)
    for hour, minute in times:
        context.current_dt = datetime.datetime(2021,6,9,hour,minute); intraday['handle_data'](context,None)
    require(closed[:2] == ['sell_all_long','sell_all_short'] and closed[2:] == ['sell_long_pos','sell_short_pos']*6, 'Intraday close routing differs')
    rows.append({'case': 'seven_intraday_times_first_all_then_non_today', 'times': [list(t) for t in times], 'calls': closed})
    intraday['get_future_contracts'] = lambda symbol: ['IF2106.CCFX','IF2107.CCFX','IF2109.CCFX']; failed = False
    try: intraday['before_market_open'](context)
    except IndexError: failed = True
    require(failed, 'Three contracts did not fail index3')
    rows.append({'case': 'intraday_contract_index_three_requires_four', 'error': 'IndexError', 'available_contracts': 3})
    portfolio = SimpleNamespace(long_positions={'IF2106.CCFX': SimpleNamespace(total_amount=2, side='long')}, short_positions={})
    context.subportfolios = [portfolio]; intraday['get_dominant_future'] = lambda symbol: 'IF2109.CCFX'
    intraday['get_current_data'] = lambda: {code: SimpleNamespace(last_price=100., low_limit=90., high_limit=110.) for code in ('IF2106.CCFX','IF2109.CCFX')}
    orders.clear(); switched = intraday['position_auto_switch'](context)
    require(len(switched)==1 and len(orders)==2, 'Unscheduled switch intent differs')
    rows.append({'case': 'unscheduled_auto_switch_reports_none_orders_as_switched', 'orders': orders.copy(), 'reported_switches': switched, 'fills': 0})
    return {'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False, 'synthetic_fixture': True,
        'limits': ['All contracts, quotes, holdings, equity, minutes and orders are explicit synthetic diagnostic fixtures',
            'Order returnsNone are not fills; wrong-side/roll/double-clear/state defects preserved, no economic repairs',
            'Callbacks and declarations are source diagnostics only; source/platform/settlement equivalence unproved']}


def study(root, directory):
    binding(root, directory)
    with redirect_stdout(StringIO()): summary, states, ratios = arithmetic(directory); diag = diagnostics(directory)
    for name, frame in (('td-states.parquet', states), ('ma-ratios.parquet', ratios)):
        path = directory / name; require(not path.exists(), 'Component file exists'); frame.to_parquet(path, index=False)
    save(directory / 'component-research.json', summary); save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch59 true-index TD seeded transitions/MA ratios and original defects archived; no futures trades')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch59 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        summary, states, ratios = arithmetic(directory); diag = diagnostics(directory); dep = inventory(root); decision = supplements(); catalog = offline_catalog(root, directory)
    for name, frame in (('td-states.parquet', states), ('ma-ratios.parquet', ratios)):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / name), check_exact=True)
    save(directory / 'component-offline.json', summary)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and dep == read(directory / 'dependency-inventory.json') and
        decision == read(directory / 'supplement-decision.json'), 'Offline evidence changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'values_sha256': {n: file_sha(directory / n) for n in ('td-states.parquet','ma-ratios.parquet')},
        'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch59 forbidden-network TD/MA/diagnostics/catalog match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch59.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch59_research.py', 'tests/unit/test_batch58_research.py',
            'tests/unit/test_catalog_futures.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    require(read(directory / 'supplement-decision.json') == supplements() and read(directory / 'existing-apis.json') == api_evidence(), 'API/decision changed')
    before = implementation(); rows = []
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json') and
        off['values_sha256'] == {n: file_sha(directory / n) for n in ('td-states.parquet','ma-ratios.parquet')}, 'Offline proof changed')
    with redirect_stdout(StringIO()): summary, states, ratios = arithmetic(directory); diag = diagnostics(directory)
    require(summary == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        inventory(root) == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json') and
        api_evidence() == read(directory / 'existing-apis.json'), 'Recomputed research changed')
    for name, frame in (('td-states.parquet', states), ('ma-ratios.parquet', ratios)):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / name), check_exact=True)
    current, operand = index_operand(root)
    require(operand == {k:v for k,v in read(directory / 'index-input-binding.json').items() if k != 'frozen_sha256'}, 'Index source binding changed')
    pd.testing.assert_frame_equal(current, pd.read_parquet(directory / 'index-input.parquet'), check_exact=True)
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch59 TD/IF research accepted; original futures remain gated by contract/minute/settlement data')
    receipt = Path('docs/handoff/2026-10-07-batch59-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'supplements': read(directory / 'supplement-decision.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch59/20261007-TD-IF')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
