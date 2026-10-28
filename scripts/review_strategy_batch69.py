"""Audit adaptive-MA/momentum futures sources, preserving input and execution gates."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
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
import requests
import talib

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts import review_strategy_batch68 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch69/20261008-adaptive-momentum-futures')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY
RECEIPT = Path('docs/handoff/2026-10-08-batch68-verification.json')
SOURCES = ('2020年度精选策略/01 AdaptiveMA自适应均线 过滤器 期货多品种模型 收益稳定增长.txt',
    '2020年度精选策略/04 趋势交易能赚钱吗？商品期货动量效应挖掘初探.txt')
SOURCE_SHA = ('2263acf4cd61f815fd61c0d21f700712e5d76ded961bca169e5f990cf41e3fec',
    'de731c1ad920d97c0994ae8b9e6fca0b53ca7071d222bd2cc1543076ab630f56')
CODE_START = (9, 9)
REFERENCES = (('repo/backtrader', 'backtrader/indicators/kama.py'),
    ('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/rqalpha', 'rqalpha/examples/turtle.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_future.py'),
    ('repo/akshare', 'akshare/futures/futures_zh_sina.py'),
    ('repo/akshare', 'akshare/futures/futures_daily_bar.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch69.py',
    'tests/unit/test_batch69_research.py'}))
QUERIES = {'rb_daily': ('futures_zh_daily_sina', {'symbol': 'RB0'}),
    'cu_daily': ('futures_zh_daily_sina', {'symbol': 'CU0'}),
    'shfe': ('get_shfe_daily', {'date': '20200102'})}
PROBE_FOLDER = 'probes-recovery-01'
PROBE_RESULTS = 'probe-results-recovery-01.json'
SINA_URL = ('https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_V21052021_4_12='
    '/InnerFuturesNewService.getDailyKLine')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'dependency-inventory.json',
    'diagnostic-failure-binding.json',
    'existing-apis.json', 'diagnostics.json', 'probe-results.json', 'probe-failure-binding.json',
    PROBE_RESULTS, 'probe-audit.json', 'exchange-parser-recovery.json',
    'component-values.parquet', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch68 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Binding changed')
    return result


def parsed_source(text, number):
    # Blank the non-Python article header while retaining every original line number.
    lines = text.splitlines(); start = CODE_START[number]
    require(lines[6] == '原文策略源码如下：' and lines[start-1].startswith('#'), 'Article prefix changed')
    lines[:start-1] = ['']*(start-1)
    return ast.parse('\n'.join(lines))


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return parsed_source(read_source(Path(row['source_copy']))[0], number)


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch69-functions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    rows = []
    for name in sorted({q[0] for q in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'function': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest()})
    helper = inspect.getsource(ak.get_shfe_daily.__globals__['requests_link'])
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'talib': talib.__version__, 'ta_core': talib.__ta_version__.decode(),
        'compatibility': talib.get_compatibility(),
        'unstable': {name: talib.get_unstable_period(name) for name in ('KAMA', 'ATR')},
        'apis': rows, 'queries': {key: list(value) for key, value in QUERIES.items()}, 'sina_url': SINA_URL,
        'shfe_url': ak.get_shfe_daily.__globals__['cons'].SHFE_DAILY_URL_20250630 % '20200102',
        'shfe_query_in_calendar': '20200102' in ak.get_shfe_daily.__globals__['calendar'],
        'original_retry_helper_source': helper,
        'network_policy': 'One GET per case, timeout8/10, no environment proxies/redirects; parent worker timeout45s',
        'limits': ['Continuous RB0/CU0 are supplier samples, not proved original 8888 index or historical dominant mapping',
            'One exchange daily table cannot prove long history, availability, multipliers, margin or settlement']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'required_inputs': ['Original 10/26-product 8888-index 50/15-day OHLC and historical dominant mapping',
            'Actual historical contracts/prior closes/open executions/expiry and decision availability',
            'Historical multiplier/tick/fee/margin/daily settlement and rollover/partial-fill semantics'],
        'gates': ['No registered commodity-futures history, mapping or settlement tables',
            'Unique Book has no accepted futures short/margin/multiplier/settlement semantics; no second ledger',
            'No approval to change global reentry, counter reset, loop return, fractional lots or rollover rules',
            'Strict financial versions/shares/PCF/names and existing pending decisions remain separately gated']}


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch69 adaptive-MA/momentum futures protection baseline started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    folder = directory / 'source-reviews'; folder.mkdir(); frozen = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; require(file_sha(path) == sha, 'Source SHA changed')
        shutil.copyfile(path, folder / f'{strategy_id(name)}.source'); frozen.append({'path': name, 'sha256': sha})
    save(directory / 'sources-frozen.json', frozen)
    return {'status': 'ok'}


def start(root, directory):
    binding(root, directory)
    failure = Path('data/staging/strategies-batch69/20261008-linear-fixture-failure')
    save(directory / 'diagnostic-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)}
        for p in sorted(failure.iterdir()) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = []; refs = []
    for i, name in enumerate(SOURCES):
        original = Path('repo/量化策略源代码') / name
        copied = directory / 'source-reviews' / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and
            file_sha(original) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        text, encoding = read_source(copied); tree = parsed_source(text, i)
        rows.append({'source_path': str(original), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(tree),
            'code_start_line': CODE_START[i], 'original_line_numbers_retained': True,
            'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only KAMA/ATR and futures quantity/API comparison, not platform equivalence or replacement ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch69 two complete futures sources/parameters reviewed; commodity trades remain input-gated')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            row['source_sha256'] == file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(parsed_source(text, i)) and
            row['code_start_line'] == CODE_START[i] and row['original_line_numbers_retained'], 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and
            file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    failure = read(directory / 'diagnostic-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '2 failed, 6 passed' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red evidence binding changed')


def diagnostics(directory):
    validate_sources(directory); rows = []
    context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 2, 9, 0),
        portfolio=SimpleNamespace(starting_cash=1000000., long_positions={}, short_positions={}))
    def namespace():
        orders = []; declarations = []; schedules = []
        ns = {'g': SimpleNamespace(), 'np': np, 'talib': talib, 'isnan': math.isnan,
            'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None),
            'order_target': lambda *a, **kw: orders.append({'args': list(a), 'kwargs': kw}),
            'get_dominant_future': lambda ins: ins+'_OLD', 'send_message': lambda *a: None,
            'set_benchmark': lambda x: declarations.append(['benchmark', x]),
            'set_option': lambda *a: declarations.append(['option', *a]),
            'set_subportfolios': lambda x: declarations.append(['subportfolios', x]),
            'set_order_cost': lambda x, **kw: declarations.append(['cost', x, kw]),
            'set_slippage': lambda x, **kw: declarations.append(['slippage', x, kw]),
            'SubPortfolioConfig': lambda **kw: kw, 'OrderCost': lambda **kw: kw,
            'PriceRelatedSlippage': lambda x: x,
            'run_daily': lambda fn, **kw: schedules.append(['daily', fn.__name__, kw]),
            'run_weekly': lambda fn, day, **kw: schedules.append(['weekly', fn.__name__, day, kw])}
        return ns, orders, declarations, schedules
    def error(case, callback, expected, orders):
        try: callback()
        except expected as exc: rows.append({'case': case, 'error': type(exc).__name__, 'orders': orders.copy()})
        else: raise ValueError('Expected original failure: '+case)
    for number, prefix in enumerate(('adaptive', 'momentum')):
        ns, orders, declarations, schedules = namespace()
        names = [n.name for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef)]
        selected(directory, number, names, ns); ns['initialize'](context); g = ns['g']
        require(len(g.instruments) == (10 if number == 0 else 26), 'Instrument count differs')
        require(g.Window == 15 and g.FilterTimes == .3 and g.NATRstop == 4 if number == 0 else
            g.BackWindow == 10 and g.HoldingWeek == 1, 'Parameters differ')
        rows.append({'case': prefix+'_defaults_schedule_cost', 'instruments': g.instruments.copy(),
            'declarations': declarations.copy(), 'schedules': schedules.copy(), 'fees_calculated': False})
        ns['attribute_history'] = lambda *a: pd.DataFrame({'open': [100.]})
        cash = 1000000./len(g.instruments); initial = ns['get_lots'](cash, 'CU')
        g.ATR['CU8888.XSGE'] = 3.; sized = ns['get_lots'](cash, 'CU')
        require(initial == cash*.0001/5 and sized == cash*.1/15 and not float(sized).is_integer(), 'ATR quantity differs')
        rows.append({'case': prefix+'_initial_and_prior_ATR_fractional_lots', 'initial_lots': initial,
            'prior_ATR_lots': sized, 'integer_rounding': False, 'cash_basis': 'starting_cash/product_count'})
        ns['attribute_history'] = lambda *a: pd.DataFrame({'open': []})
        require(ns['get_lots'](cash, 'CU') is None, 'Empty quantity window differs')
        rows.append({'case': prefix+'_empty_quantity_window_None'})
        ns['attribute_history'] = lambda *a: pd.DataFrame({'open': [100.]}); g.ATR['CU8888.XSGE'] = np.float64(0)
        with np.errstate(divide='ignore'): lots = ns['get_lots'](cash, 'CU')
        require(np.isinf(lots), 'Zero numpy ATR should yield infinite intention')
        rows.append({'case': prefix+'_zero_numpy_ATR_infinite_lot_intent', 'nonfinite_lot': True})
        g.MappingReal = {'CU': 'CU_OLD'}; context.portfolio.long_positions = {}
        context.portfolio.short_positions = {'CU_OLD': SimpleNamespace(total_amount=2)}; orders.clear()
        error(prefix+'_short_roll_reads_new_missing_position',
            lambda: ns['replace_old_futures'](context, 'CU', 'CU_NEW'), KeyError, orders)
        context.portfolio.long_positions = {'CU_OLD': SimpleNamespace(total_amount=3)}; orders.clear()
        error(prefix+'_mixed_roll_stops_after_long_intents',
            lambda: ns['replace_old_futures'](context, 'CU', 'CU_NEW'), KeyError, orders)
        require(len(orders) == 2 and g.MappingReal['CU'] == 'CU_OLD', 'Partial roll differs')
        context.portfolio.long_positions = {}; context.portfolio.short_positions['CU_NEW'] = SimpleNamespace(total_amount=0)
        orders.clear(); ns['replace_old_futures'](context, 'CU', 'CU_NEW')
        require(orders[1]['args'] == ['CU_NEW', 0] and g.MappingReal['CU'] == 'CU_NEW', 'Zero new short roll differs')
        rows.append({'case': prefix+'_roll_None_orders_still_changes_mapping', 'orders': orders.copy(),
            'mapping': g.MappingReal.copy(), 'old_contract_short_quantity': 2})
        g.instruments = ['CU', 'TA']; g.MappingReal = {'CU': 'CU_NEW', 'TA': 'TA_NEW'}
        g.MappingIndex = {'CU': 'CU8888.XSGE', 'TA': 'TA8888.XZCE'}; g.Times = {'CU_NEW': 0, 'TA_NEW': 0}
        requested = []; captured = []
        def history(code, count, *a):
            requested.append([code, count]); values = np.arange(count, dtype=float)+100.
            return pd.DataFrame({'close': values, 'high': values+2, 'low': values-1, 'open': values}, index=range(-count, 0))
        ns['get_CCFX_end_date'] = lambda *a: context.current_dt.date(); ns['attribute_history'] = history
        ns['market_open'](context); require(not requested, 'Expiry return should skip all history')
        rows.append({'case': prefix+'_expiry_returns_entire_callback', 'history_requests': requested.copy()})
        ns['get_CCFX_end_date'] = lambda *a: dt.date(2099, 1, 1)
        ns['attribute_history'] = lambda *a: pd.DataFrame({'close': [100.]}, index=pd.date_range('2020-01-01', periods=1))
        error(prefix+'_datetime_negative_label_KeyError', lambda: ns['market_open'](context), KeyError, orders)
        requested.clear(); ns['attribute_history'] = history
        if number == 0:
            ns['Trade'] = lambda ctx, real, index: captured.append({'contract': real, 'signal': g.Signal, 'cross': g.Cross})
            ns['Re_entry'] = lambda *a: None; g.Reentry_long = True
            def shock_history(code, count, *a):
                frame = history(code, count, *a)
                if count == 50:
                    frame.iloc[-1] = frame.iloc[-1]+20
                return frame
            ns['attribute_history'] = shock_history
            ns['market_open'](context)
            require([r['signal'] for r in captured] == [0, 0] and [r['cross'] for r in captured] == [1, 1], 'Global reentry differs')
            rows.append({'case': 'adaptive_shared_reentry_blocks_two_products', 'captured_intents': captured.copy()})
            ns['attribute_history'] = history; captured.clear(); ns['market_open'](context)
            require([r['cross'] for r in captured] == [0, 0], 'Linear trend must stay below the original STDDEV filter')
            kama = g.AMA['CU8888.XSGE']; expected = []; value = 114.
            for close in np.arange(115., 150.):
                value += (close-value)*(2./3.)**2; expected.append(value)
            require(np.allclose(kama[15:], expected, rtol=1e-14, atol=1e-12), 'KAMA linear independent seed differs')
            mean = math.fsum(expected[-15:])/15
            sd = math.sqrt(math.fsum((x-mean)**2 for x in expected[-15:])/15)
            require(math.isclose(g.Filter['CU8888.XSGE'], sd, rel_tol=1e-10) and
                math.isclose(g.ATR['CU8888.XSGE'], 3., rel_tol=1e-14), 'STDDEV/ATR independent numeric fixture differs')
            rows.append({'case': 'adaptive_50row_KAMA15_ATR15_last15_STDDEV_independent',
                'kama_last': float(kama[-1]), 'filter': float(g.Filter['CU8888.XSGE']), 'ATR': 3.,
                'last_KAMA_difference': float(kama[-1]-kama[-2]), 'filter_multiplier': .3,
                'arithmetic': 'Linear efficiency1, seed close14, alpha(2/3)^2; independent fsum population std'})
            captured.clear(); requested.clear()
            ns['attribute_history'] = lambda code, count, *a: history(code, min(count, 10))
            ns['market_open'](context)
            require(not captured and all(code != 'TA_NEW' for code, _ in requested), 'Short window should skip second product')
            rows.append({'case': 'adaptive_short50_window_returns_before_next_product', 'history_requests': requested.copy()})
            ns['attribute_history'] = lambda code, count, *a: history(code, count) if count == 1 else pd.DataFrame({'close': []})
            error('adaptive_empty50_accesses_last_before_length_guard', lambda: ns['market_open'](context), KeyError, orders)
            g.LastRealPrice = {'CU_NEW': 87.}; g.HighPrice = {'CU_NEW': 100.}; g.LowPrice = {'CU_NEW': False}
            g.Times = {'CU_NEW': 100}; g.ATR = {'CU8888.XSGE': 3.}; g.Reentry_long = False; orders.clear()
            context.portfolio.long_positions = {'CU_NEW': SimpleNamespace(total_amount=1)}; context.portfolio.short_positions = {}
            ns['Trailing_Stop'](context, 'CU_NEW', 'CU8888.XSGE')
            require(g.Reentry_long and g.Times['CU_NEW'] == 100 and len(orders) == 1, 'Stop should preserve counter')
            g.ClosePrice = pd.Series(np.full(50, 100.)); g.CurrentPrice = 87.
            selected(directory, number, ['Re_entry'], ns)['Re_entry'](context, 'CU_NEW')
            require(not g.Reentry_long, 'Accumulated counter should immediately unblock')
            rows.append({'case': 'adaptive_stop_counter100_immediately_unblocks', 'counter': g.Times['CU_NEW'],
                'reentry_long': g.Reentry_long, 'orders': orders.copy()})
            g.HighPrice['CU_NEW'] = 100.; g.LastRealPrice['CU_NEW'] = 88.; orders.clear()
            ns['Trailing_Stop'](context, 'CU_NEW', 'CU8888.XSGE'); require(not orders, 'Strict stop boundary changed')
            rows.append({'case': 'adaptive_stop_equal_4ATR_no_order', 'orders': orders.copy()})
        else:
            ns['market_open'](context)
            require(g.CurrentPrice['CU'] == 114. and g.Price_DaysAgo['CU'] == 105. and g.Signal['CU'] == 1, 'Momentum lag differs')
            rows.append({'case': 'momentum_15rows_10th_from_end_is_9_intervals', 'current_close': 114.,
                'comparison_close': 105., 'elapsed_intervals': 9, 'ATR': float(g.ATR['CU8888.XSGE']),
                'signals': g.Signal.copy()})
            requested.clear(); ns['attribute_history'] = lambda code, count, *a: history(code, min(count, 5))
            error('momentum_len5_passes_len1_guard_then_KeyError', lambda: ns['market_open'](context), KeyError, orders)
            ns['attribute_history'] = history; ns['market_open'](context)
            g.Signal['CU'] = 0; g.Signal['TA'] = 0; orders.clear(); ns['Trade'](context)
            require(not orders, 'Neutral momentum should hold positions')
            rows.append({'case': 'momentum_zero_signal_keeps_position_no_stop', 'orders': orders.copy(),
                'active_trailing_stop_definition': any(n.name == 'Trailing_Stop' for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef))})
            g.Signal = {'CU': 1}; g.TradeLots = {'CU_NEW': 2.5}; g.LastRealPrice = {'CU_NEW': 100.}
            context.portfolio.long_positions = {'CU_NEW': SimpleNamespace(total_amount=0)}
            context.portfolio.short_positions = {'CU_NEW': SimpleNamespace(total_amount=2)}; orders.clear()
            ns['Trade'](context)
            require(orders[0]['kwargs']['side'] == 'short' and orders[1]['args'] == ['CU_NEW', 2.5] and
                g.HighPrice['CU_NEW'] == 100., 'Fractional target/state after None differs')
            rows.append({'case': 'momentum_reverse_None_orders_records_high_price', 'orders': orders.copy(),
                'high_price': g.HighPrice['CU_NEW']})
    require(len(rows) == 28, 'Diagnostic count differs')
    return {'cases': rows, 'not_a_backtest': True, 'synthetic_fixture': True,
        'real_historical_operand_windows': 0, 'platform_equivalent': False,
        'limits': ['Explicit np and negative-label fixtures allow original functions to be diagnosed; no silent production fixes',
            'Orders are recorded intentions returning None, not fills, NAV or three-cost experiments',
            'TA-Lib fixture numbers and fee declarations do not prove original historical input/platform equivalence',
            'Backtrader KAMA uses SMA seed; original TA-Lib seed must not be substituted']}


def study(root, directory):
    binding(root, directory)
    with redirect_stdout(StringIO()): result = diagnostics(directory)
    save(directory / 'diagnostics.json', result)
    return archive(root, 'Batch69 original KAMA/ATR/momentum/rollover/reentry synthetic diagnostics archived; no trades')


def invoke_api(case):
    import akshare as ak
    name, params = QUERIES[case]
    if name == 'get_shfe_daily':
        def once(url, encoding='utf-8', method='get', data=None, headers=None):
            require(method == 'get' and data is None, 'Unexpected SHFE method')
            response = requests.get(url, headers=headers); response.encoding = encoding; return response
        with patch.dict(ak.get_shfe_daily.__globals__, {'requests_link': once}): return ak.get_shfe_daily(**params)
    return getattr(ak, name)(**params)


def worker(directory, case):
    folder = directory / PROBE_FOLDER / case; folder.mkdir(parents=True, exist_ok=False)
    name, params = QUERIES[case]
    row = {'case': case, 'function': name, 'parameters': params, 'status': 'failed',
        'attempts': [], 'responses': [], 'files': [], 'published': False, 'strict_usable': False,
        'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(not row['attempts'], 'Single request limit exceeded')
        expected_url = read(directory / 'existing-apis.json')['shfe_url'] if case == 'shfe' else SINA_URL
        require(method.upper() == 'GET' and url == expected_url, 'Unexpected request')
        row['attempts'].append({'method': method, 'url': url, 'parameters': kwargs.get('params')})
        session.trust_env = False; kwargs['timeout'] = (8, 10); kwargs['allow_redirects'] = False
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as out: out.write(response.content)
        row['responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code,
            'url': response.url, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API/settings changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke_api(case)
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    finally: save(folder / 'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); rows = []
    failure = Path('data/staging/strategies-batch69/20261008-API-shape-failure')
    files = [*sorted(p for p in failure.iterdir() if p.is_file()), directory / 'probe-results.json',
        *sorted((directory / 'probes').glob('*/result.json')), *sorted(directory.glob('worker-*.json'))]
    save(directory / 'probe-failure-binding.json', {str(p): file_sha(p) for p in files})
    for case in QUERIES:
        require(not (directory / PROBE_FOLDER / case).exists(), 'Case already attempted')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch69', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-recovery-01-{case}.json', {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before saving result')
        except subprocess.TimeoutExpired as exc:
            save(directory / f'worker-recovery-01-{case}.json', {'command': command, 'status': 'timeout',
                'stdout': str(exc.stdout), 'stderr': str(exc.stderr), 'timeout_seconds': 45})
            folder = directory / PROBE_FOLDER / case
            if not folder.exists(): folder.mkdir(parents=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'function': QUERIES[case][0], 'parameters': QUERIES[case][1],
                    'status': 'timeout', 'error': 'Parent subprocess deadline45s; request completion unproved',
                    'attempts': [], 'responses': [], 'files': [], 'published': False, 'strict_usable': False,
                    'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / PROBE_FOLDER / case / 'result.json'))
    save(directory / PROBE_RESULTS, {'results': rows, 'published': False, 'not_a_backtest': True})
    save(directory / 'probe-audit.json', probe_audit(directory))
    return archive(root, 'Batch69 three bounded commodity data probes archived; samples not admitted trading inputs')


def probe_audit(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'API changed')
    failure = read(directory / 'probe-failure-binding.json')
    for name, sha in failure.items(): require(file_sha(name) == sha, 'Probe failure evidence changed')
    first = read(directory / 'probe-results.json')['results']
    require([r['case'] for r in first] == list(QUERIES) and all(r['status'] == 'failed' and
        r['attempts'] == r['responses'] == r['files'] == [] and r['error'] == 'ValueError: Installed API/settings changed'
        for r in first), 'First pre-request failure evidence changed')
    red = read('data/staging/strategies-batch69/20261008-API-shape-failure/red-regression.json')
    require(red['returncode'] == 1 and '1 failed, 8 passed' in red['stdout'] and
        red['driver_sha256'] == failure['data/staging/strategies-batch69/20261008-API-shape-failure/driver.original.py'] and
        red['test_sha256'] == failure['data/staging/strategies-batch69/20261008-API-shape-failure/regression.original.py'], 'API red evidence changed')
    rows = read(directory / PROBE_RESULTS)['results']; require([r['case'] for r in rows] == list(QUERIES), 'Probe scope changed')
    audit = []
    for row in rows:
        case = row['case']; require(row == read(directory / PROBE_FOLDER / case / 'result.json') and
            row['function'] == QUERIES[case][0] and row['parameters'] == QUERIES[case][1] and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and len(row['attempts']) <= 1 and
            not row['published'] and not row['strict_usable'], 'Probe binding changed')
        for item in row['responses']+row['files']: require(file_sha(item['file']) == item['sha256'], 'Probe bytes changed')
        entry = {'case': case, 'status': row['status'], 'response_count': len(row['responses']), 'parsed_rows': 0,
            'original_8888_equivalence': False, 'historical_dominant_mapping_proved': False}
        if row['files']:
            require(len(row['files']) == 1 and row['status'] in ('success', 'empty'), 'Parsed status inconsistent')
            frame = pd.read_parquet(row['files'][0]['file']); dates = pd.to_datetime(frame['date'], errors='coerce') if len(frame) else pd.Series([], dtype='datetime64[ns]')
            require(len(frame) == row['files'][0]['rows'] and list(frame) == row['files'][0]['columns'], 'Parsed profile changed')
            wire = row['responses']; require(len(wire) == 1, 'Parsed data lacks wire evidence')
            calls = []
            def replay(session, method, url, **kw):
                require(not calls and method.upper() == 'GET' and url == row['attempts'][0]['url'] and
                    kw.get('params') == row['attempts'][0]['parameters'], 'Unexpected replay request')
                calls.append(url); response = requests.Response(); response.status_code = wire[0]['status_code']
                response._content = Path(wire[0]['file']).read_bytes(); response.encoding = wire[0]['encoding']
                response.url = wire[0]['url']; return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): again = invoke_api(case)
            pd.testing.assert_frame_equal(frame, again, check_exact=True)
            entry.update({'parsed_rows': len(frame), 'date_min': str(dates.min()), 'date_max': str(dates.max()),
                'invalid_dates': int(dates.isna().sum()), 'wire_parser_match': True,
                'rows_after_research_end': int((dates > pd.Timestamp('2026-09-29')).sum())})
        else:
            require(row['status'] in ('failed', 'timeout') and bool(row.get('error')), 'Missing failure evidence')
            entry['error'] = row['error']
        audit.append(entry)
    return {'samples': audit, 'not_a_backtest': True, 'published': False, 'new_trading_backtests': 0,
        'decision': 'Preserve genuine responses/failures; original contracts/index/mapping and futures execution remain blocked'}


def exchange_recovery(directory):
    row = read(directory / PROBE_FOLDER / 'shfe/result.json'); wire = row['responses']
    require(row['status'] == 'failed' and 'ArrowInvalid' in row['error'] and len(wire) == 1 and
        wire[0]['status_code'] == 200 and file_sha(wire[0]['file']) == wire[0]['sha256'], 'Expected saved SHFE serialization failure changed')
    calls = []
    def replay(session, method, url, **kw):
        require(not calls and method.upper() == 'GET' and url == row['attempts'][0]['url'] and not kw.get('params'), 'Unexpected SHFE replay')
        calls.append(url); response = requests.Response(); response.status_code = wire[0]['status_code']
        response._content = Path(wire[0]['file']).read_bytes(); response.encoding = wire[0]['encoding']; return response
    with patch.object(requests.sessions.Session, 'request', replay): frame = invoke_api('shfe')
    split = frame.to_dict(orient='split')
    split['data'] = [[None if pd.isna(value) else value for value in values] for values in split['data']]
    return {'status': 'parser_recovered_from_saved_wire', 'wire_sha256': wire[0]['sha256'],
        'original_failure_preserved': True, 'new_http_requests': 0, 'rows': len(frame),
        'original_dtypes': {k: str(v) for k, v in frame.dtypes.items()}, 'parsed_native_split': split,
        'empty_string_cells': sum(value == '' for values in split['data'] for value in values if isinstance(value, str)),
        'null_cells': sum(value is None for values in split['data'] for value in values),
        'not_a_backtest': True, 'published': False, 'strict_usable': False,
        'limits': ['Preserves numeric/string/blank parser values and native dtypes; no trading normalization or historical mapping inferred']}


def independent_atr(high, low, close, period):
    tr = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])) for i in range(1, len(close))]
    value = math.fsum(tr[:period])/period
    for item in tr[period:]: value = (value*(period-1)+item)/period
    return value


def numeric_components(frame, case):
    frame = frame.copy(); dates = pd.to_datetime(frame['date'], errors='raise')
    require(not dates.duplicated().any() and dates.is_monotonic_increasing, 'Supplier date order/uniqueness invalid')
    frame['date'] = dates
    beyond = int((dates > pd.Timestamp('2026-09-29')).sum()); frame = frame[frame.date.le('2026-09-29')].reset_index(drop=True)
    data = frame[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    valid = (np.isfinite(data).all(axis=1) & (data > 0).all(axis=1) &
        (data[:, 1] >= np.maximum(data[:, 0], data[:, 3])) & (data[:, 2] <= np.minimum(data[:, 0], data[:, 3])))
    rows = []; comparisons = 0; invalid = 0
    for model, count, period in (('adaptive50', 50, 15), ('momentum15', 15, 10)):
        for end in range(count-1, len(frame)):
            start = end-count+1
            if not valid[start:end+1].all(): invalid += 1; continue
            _, high, low, close = data[start:end+1].T
            atr = float(talib.ATR(high, low, close, period)[-1]); expected = independent_atr(high, low, close, period)
            require(math.isclose(atr, expected, rel_tol=1e-12, abs_tol=1e-10), 'Independent Wilder ATR differs')
            row = {'case': case, 'model': model, 'date': str(frame.date.iloc[end].date()), 'window': count,
                'period': period, 'atr': atr, 'close': float(close[-1]), 'kama_last': math.nan,
                'kama_delta': math.nan, 'population_std': math.nan, 'comparison_close': math.nan}
            if model == 'adaptive50':
                kama = talib.KAMA(close, period); delta = float(kama[-1]-kama[-2])
                std = float(talib.STDDEV(kama[-period:], period)[-1])
                mean = math.fsum(kama[-period:])/period
                expected_std = math.sqrt(math.fsum((value-mean)**2 for value in kama[-period:])/period)
                require(math.isclose(std, expected_std, rel_tol=1e-7, abs_tol=1e-6), 'Independent population STDDEV differs')
                row.update({'kama_last': float(kama[-1]), 'kama_delta': delta, 'population_std': std,
                    'raw_direction': 1 if delta > std*.3 else -1 if -delta > std*.3 else 0})
                comparisons += 1
            else:
                row.update({'comparison_close': float(close[-10]),
                    'raw_direction': 1 if close[-1] > close[-10] else -1 if close[-1] < close[-10] else 0})
            rows.append(row); comparisons += 1
    return pd.DataFrame(rows), {'case': case, 'supplier_rows': len(dates), 'retained_rows': len(frame),
        'rows_after_cutoff_excluded': beyond, 'invalid_OHLC_rows': int((~valid).sum()),
        'invalid_windows_blocked': invalid, 'component_rows': len(rows), 'independent_numeric_comparisons': comparisons,
        'date_min': str(frame.date.min()), 'date_max': str(frame.date.max()),
        'directions': {str(k): int(v) for k, v in pd.Series([r['raw_direction'] for r in rows], dtype=int).value_counts().items()}}


def components(directory):
    validate_sources(directory); require(api_evidence() == read(directory / 'existing-apis.json'), 'Indicator/API settings changed')
    frames = []; coverage = []
    for row in read(directory / PROBE_RESULTS)['results']:
        if row['case'] == 'shfe': continue
        require(row['status'] == 'success' and len(row['files']) == 1, 'Continuous sample missing')
        item = row['files'][0]; require(file_sha(item['file']) == item['sha256'], 'Supplier sample changed')
        values, profile = numeric_components(pd.read_parquet(item['file']), row['case'])
        frames.append(values); coverage.append(profile)
    return pd.concat(frames, ignore_index=True), {'coverage': coverage,
        'component_rows': sum(row['component_rows'] for row in coverage),
        'independent_numeric_comparisons': sum(row['independent_numeric_comparisons'] for row in coverage),
        'not_a_backtest': True, 'original_signal_reproduced': False, 'trading_variant_approved': False,
        'limits': ['Supplier continuous-contract numeric sensitivity study only; RB0/CU0 not equivalent to original 8888 indices',
            'Every 50/15-row window is independently restarted, not a continuously seeded indicator',
            'Raw direction is not original multi-product/global reentry state, weekly execution or an order',
            'No costs/NAV/positions or new trading experiments; invalid OHLC windows blocked, no interpolation']}


def prepare(root, directory):
    binding(root, directory); require(not (directory / 'component-values.parquet').exists(), 'Component already archived')
    save(directory / 'exchange-parser-recovery.json', exchange_recovery(directory))
    values, summary = components(directory); values.to_parquet(directory / 'component-values.parquet', index=False)
    save(directory / 'component-research.json', summary)
    return archive(root, 'Batch69 continuous-contract arithmetic study and saved-wire SHFE parser recovery archived; not original signals/trades')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch69 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); cat = offline_catalog(root, directory); dep = inventory(root); audit = probe_audit(directory)
        values, summary = components(directory); recovered = exchange_recovery(directory)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and audit == read(directory / 'probe-audit.json') and
        summary == read(directory / 'component-research.json') and recovered == read(directory / 'exchange-parser-recovery.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'evidence_sha256': {name: file_sha(directory / name) for name in
            ('diagnostics.json', 'probe-audit.json', 'component-values.parquet', 'component-research.json',
             'exchange-parser-recovery.json')}, 'not_a_backtest': True})
    return archive(root, 'Batch69 forbidden-network original diagnostics, wire parsers and three catalog outputs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch69.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch69_research.py',
            'tests/unit/test_batch60_research.py', 'tests/unit/test_catalog_futures.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as output: result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Implementation copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'], 'Offline proof changed')
    with redirect_stdout(StringIO()): diag = diagnostics(directory); audit = probe_audit(directory)
    require(diag == read(directory / 'diagnostics.json') and audit == read(directory / 'probe-audit.json') and
        inventory(root) == read(directory / 'dependency-inventory.json'), 'Recomputed evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory)
    progress = archive(root, 'Batch69 adaptive-MA/momentum futures source research accepted; original trading remains blocked')
    receipt = Path('docs/handoff/2026-10-08-batch69-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': diag, 'probe_audit': audit,
        'component': read(directory / 'component-research.json'),
        'exchange_parser_recovery': read(directory / 'exchange-parser-recovery.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'prepare', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
