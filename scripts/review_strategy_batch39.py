"""Freeze original ETF grids and verify available ATR/RSRS arithmetic only."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import hashlib
import importlib.util
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
from scipy.stats import linregress
import talib

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch29 import reference_atr
from scripts.review_strategy_batch31 import reference_ols, reference_z
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts import review_strategy_batch36 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch36/20261006-multihorizon-lr-breadth')
RECEIPT = Path('docs/handoff/2026-10-06-batch36-verification.json')
LATEST_RECEIPT = Path('docs/handoff/2026-10-06-batch38-verification.json')
SOURCES = ('2021年度精选策略/84.ETF网格交易策略.txt',
    '2024年度精选策略2/44.！！！最易上手的网格策略v2.0，设置几个参数就可以直接使用.txt',
    '2022年度精选策略/46.指数择时与仓位管理策略初探.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/backtrader', 'backtrader/indicators/smma.py'), ('repo/backtrader', 'backtrader/indicators/ols.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch39.py', 'tests/unit/test_batch39_research.py',
    'scripts/review_strategy_batch29.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'index-input.parquet', 'calendar-input.parquet', 'dependency-inventory.json', 'component-research.json', 'diagnostics.json',
    'probe-results.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'fund512900': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh512900'},
        'limit': 'Daily price sample cannot supply original 14:45 fund prices/events/status/rules'},
    'minute512900': {'function': 'fund_etf_hist_min_em', 'parameters': {'symbol': '512900', 'period': '1', 'adjust': '',
        'start_date': '2021-01-04 14:40:00', 'end_date': '2021-01-04 14:45:00'},
        'limit': 'Installed period1 endpoint returns latest5days; date cropping cannot obtain2021 prices'},
    'minute510500': {'function': 'fund_etf_hist_min_em', 'parameters': {'symbol': '510500', 'period': '1', 'adjust': '',
        'start_date': '2021-01-04 09:30:00', 'end_date': '2021-01-04 15:00:00'},
        'limit': 'Latest5day sample cannot reconstruct original minute grid or fund event/fill semantics'}}


def implementation(): return {name: file_sha(name) for name in FILES}


def indicator():
    result = {'version': talib.__version__, 'core': talib.__ta_version__.decode(),
        'compatibility': talib.get_compatibility(), 'unstable': talib.get_unstable_period('ATR')}
    require(result['version'] == '0.8.1' and result['core'].startswith('0.8.1 ') and
        result['compatibility'] == result['unstable'] == 0, 'ATR backend changed')
    return result


def binding(root, directory=None):
    receipt = read(RECEIPT); latest = read(LATEST_RECEIPT)
    require(receipt['status'] == latest['status'] == 'ok' and receipt['snapshot'] == latest['snapshot'] == SNAPSHOT, 'Accepted receipts differ')
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name; require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'latest_receipt_file': str(LATEST_RECEIPT), 'latest_receipt_sha256': file_sha(LATEST_RECEIPT),
        'upstream': previous.binding(root, ACCEPTED), 'files': rows, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'queries': QUERIES, 'apis': rows, 'indicator': indicator()}


def start(root, directory):
    bound = binding(root); apis = api_evidence()
    checkpoint(root, directory, 'Batch39 ETF grid/ATR position source review started')
    save(directory / 'input-binding.json', bound); save(directory / 'existing-apis.json', apis)
    old_path = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path), 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only ATR/OLS and ETF interface reference; no new ledger or platform runtime'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'statsmodels')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch39 three full sources frozen; minute-fund dependencies and untradeable index retained')


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


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); funds = ['512900.SH', '510500.SH']
    master = store.load_state(state, 'instruments', filters=[('instrument', 'in', funds)])
    minute = store.load_state(state, 'minute_universe', filters=[('instrument', 'in', funds)])
    rows = {}
    for name in ('bars_1d', 'bars_5m', 'corp_actions'):
        frame = store.load_state(state, name, filters=[('instrument', 'in', funds)])
        rows[name] = {fund: int(frame.instrument.eq(fund).sum()) for fund in funds}
    search = subprocess.run(['rg', '--files', '--hidden', str(root), '-g', '*512900*', '-g', '*510500*'], capture_output=True, text=True)
    require(search.returncode in (0, 1), search.stderr)
    return {'snapshot': SNAPSHOT, 'funds': funds, 'master_rows': len(master), 'minute_universe_rows': len(minute),
        'table_rows': rows, 'partitions': {name: len(state['tables'].get(name, {})) for name in ('bars_1d', 'bars_5m', 'minute_universe')},
        'filename_search': search.stdout.splitlines(), 'not_a_backtest': True,
        'limits': ['Published stock5m is not original fund1m', 'No minute aggregation is a replacement for missing fund prices/events/rules']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['Index20-row ATR and contiguous1100-slope algebra only, not grid fund data or a tradeable security',
            'SciPy OLS samples do not prove statsmodels/platform initialization, seed endpoints or original daily signals']})
    save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch39 actual fund input gaps and accepted index/calendar bytes frozen')


def inputs(directory):
    frame, sessions = previous.inputs(directory)
    require(np.isfinite(frame[['high', 'low', 'close']]).all().all() and frame.low.gt(0).all() and
        frame.high.ge(frame.low).all() and frame.close.between(frame.low, frame.high).all(), 'Invalid ATR/OLS OHLC operands')
    require(frame.date.tolist() == [day for day in sessions if frame.date.min() <= day <= frame.date.max()], 'Component calendar gap')
    return frame, sessions


def score_kernel(directory):
    fn = next(n for n in source_tree(directory, 2).body if isinstance(n, ast.FunctionDef) and n.name == 'market_open')
    names = ('section', 'mu', 'sigma', 'zscore', 'zscore_rightdev')
    nodes = [n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    require(len(nodes) == 5, 'Original RSRS algebra structure changed')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, '<original-rsrs-algebra-only>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def order_kernel(directory):
    fn = next(n for n in source_tree(directory, 2).body if isinstance(n, ast.FunctionDef) and n.name == 'market_open')
    node = next(n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp) and
        any(isinstance(x, ast.Name) and x.id == 'zscore_rightdev' for x in ast.walk(n.test)))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    return compile(module, '<original-rsrs-order-state-only>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def compute(directory):
    validate_sources(directory); frame, _ = inputs(directory); backend = indicator()
    require(len(frame) >= 1117, 'Incomplete ATR/RSRS component window')
    require(backend == read(directory / 'existing-apis.json')['indicator'], 'Frozen ATR backend differs')
    high, low, close = [frame[name].to_numpy(dtype=float) for name in ('high', 'low', 'close')]
    ns = {'tb': talib}; sha = selected(directory, 2, ('ATR',), ns); code, score_sha = score_kernel(directory)
    amounts = []; atr_errors = np.zeros(3); atr_boundaries = []; slopes = []; ref_slopes = []; regressions = []; regression_errors = np.zeros(3)
    capital = 1000000.
    for k in range(17, len(frame)):
        actual = linregress(low[k-17:k+1], high[k-17:k+1]); reference = reference_ols(low[k-17:k+1], high[k-17:k+1])
        values = [float(actual.intercept), float(actual.slope), float(actual.rvalue**2)]
        regression_errors = np.maximum(regression_errors, np.abs(np.asarray(values)-np.asarray(reference)))
        slopes.append(values[1]); ref_slopes.append(reference[1]); regressions.append((values[1], values[2], reference[1], reference[2]))
        if k < 19: continue
        actual_atr = list(map(float, ns['ATR'](high[k-19:k+1], low[k-19:k+1], close[k-19:k+1], capital)))
        value = reference_atr(high[k-19:k+1], low[k-19:k+1], close[k-19:k+1]); unit = .01*capital/value
        expected = [value, unit, unit*float(close[k])]
        require(all(math.isfinite(x) and x > 0 for x in (*actual_atr, *expected)), 'Unknown ATR/sizing operand')
        atr_errors = np.maximum(atr_errors, np.abs(np.asarray(actual_atr)-np.asarray(expected)))
        if int(actual_atr[1]) != int(expected[1]): atr_boundaries.append({'date': frame.date.iloc[k], 'actual': actual_atr, 'reference': expected})
        amounts.append(actual_atr)
    scores = []; score_error = 0.; boundaries = []
    for k in range(1099, len(slopes)):
        beta, r2, ref_beta, ref_r2 = regressions[k]
        ns = {'g': SimpleNamespace(ans=slopes[k-1099:k+1], M=1100), 'np': np, 'beta': beta, 'r2': r2}
        exec(code, ns); value = float(ns['zscore_rightdev']); expected = reference_z(ref_slopes[k-1099:k+1])*ref_beta*ref_r2
        require(math.isfinite(value) and math.isfinite(expected), 'Nonfinite RSRS algebra')
        score_error = max(score_error, abs(value-expected)); flags = [value > .7, value < -.7]; ref_flags = [expected > .7, expected < -.7]
        if flags != ref_flags: boundaries.append({'date': frame.date.iloc[k+17], 'actual': value, 'reference': expected,
            'flags': [bool(x) for x in flags], 'reference_flags': [bool(x) for x in ref_flags]})
        scores.append(value)
    return {'not_a_backtest': True, 'platform_equivalent': False, 'backend': backend, 'rows': len(frame),
        'ast_sha256': {'ATR': sha, 'rsrs_algebra': score_sha}, 'atr_windows': len(amounts), 'atr_first_end_date': frame.date.iloc[19],
        'last_end_date': frame.date.iloc[-1], 'atr_sizing_capital_sample': capital, 'atr_max_differences': atr_errors.tolist(),
        'atr_integer_unit_boundaries': atr_boundaries, 'atr_sha256': hashlib.sha256(np.asarray(amounts, dtype='<f8').tobytes()).hexdigest(),
        'amount_above_sample_capital': sum(v[2] > capital for v in amounts),
        'regression_windows': len(slopes), 'regression_max_differences': regression_errors.tolist(),
        'rsrs_algebra_windows': len(scores), 'rsrs_first_end_date': frame.date.iloc[1116], 'rsrs_max_difference': score_error,
        'rsrs_boundaries': boundaries, 'rsrs_sha256': hashlib.sha256(np.asarray(scores, dtype='<f8').tobytes()).hexdigest(),
        'limits': ['Sizing capital is an arithmetic operand, not a simulated portfolio', 'Contiguous1100 SciPy OLS samples, not original statsmodels seed/state',
            'No ETF minute prices, orders, costs, fills or NAV reconstructed']}


def grid_namespace(directory, number, cash=100000.):
    calls = []; orders = []; holder = {'price': 1., 'reply': None}; g = SimpleNamespace()
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 14, 45), portfolio=SimpleNamespace(cash=cash, available_cash=cash, positions={}))
    def order(kind, *args): orders.append([kind, *args]); return holder['reply']
    ns = {'g': g, 'log': SimpleNamespace(info=lambda *a: None),
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'set_order_cost': lambda *a, **kw: None,
        'OrderCost': lambda **kw: kw, 'set_slippage': lambda *a: None, 'PriceRelatedSlippage': lambda x: x,
        'run_daily': lambda fn, *a, **kw: calls.append([fn.__name__, list(a), kw]),
        'get_price': lambda *a, **kw: pd.DataFrame({'close': [holder['price']]}, index=[-1] if number == 0 else [0]),
        'get_bars': lambda *a, **kw: {'close': np.ones(60)},
        'order_value': lambda *a: order('value', *a), 'order_target': lambda *a: order('target', *a), 'order': lambda *a: order('quantity', *a)}
    names = ('initialize', 'market_open', 'after_market') if number == 0 else ('initialize', 'handle_data', 'cal_avg', 'run_check', 'run_adj')
    sha = selected(directory, number, names, ns)
    with redirect_stdout(io.StringIO()): ns['initialize'](ctx)
    return ns, ctx, holder, calls, orders, sha


def diagnostics(directory):
    cases = []
    ns, ctx, holder, calls, orders, grid84_sha = grid_namespace(directory, 0, 30000.)
    cases.append({'case': 'grid84_configuration', 'cash': ns['g'].cash, 'buy': ns['g'].initial_buy,
        'sell': ns['g'].initial_sell, 'step': ns['g'].unitprice, 'schedule': calls})
    for price in (.9, 1., .89, 1.01):
        orders.clear(); holder['price'] = price; ns['market_open'](ctx)
        cases.append({'case': 'grid84_strict_price', 'price': price, 'orders': orders.copy()})
    holder['price'] = .95; orders.clear(); ns['market_open'](ctx); ctx.portfolio.available_cash -= 1.
    with redirect_stdout(io.StringIO()): ns['after_market'](ctx)
    cases.append({'case': 'grid84_unrelated_cash_changes_grid', 'orders': orders.copy(), 'buy': ns['g'].initial_buy, 'sell': ns['g'].initial_sell})
    ns['get_price'] = lambda *a, **kw: pd.DataFrame({'close': [.89]}, index=pd.date_range('2021-01-04', periods=1))
    try: ns['market_open'](ctx)
    except KeyError as exc: cases.append({'case': 'grid84_date_integer_fault', 'error': str(exc)})
    else: raise AssertionError('Original grid84 integer label fault disappeared')

    ns, ctx, holder, calls, orders, grid44_sha = grid_namespace(directory, 1)
    cases.append({'case': 'grid44_configuration', 'max_net': ns['g'].max_net, 'schedule': calls})
    try: ns['handle_data'](ctx, None)
    except AttributeError as exc: cases.append({'case': 'grid44_none_initial_order_fault', 'error': str(exc), 'orders': orders.copy()})
    else: raise AssertionError('Original None order fault disappeared')
    holder['reply'] = SimpleNamespace(status='held', filled=100., price=1.); orders.clear(); ns['handle_data'](ctx, None)
    cases.append({'case': 'grid44_partial_held_adds_layer', 'net': ns['g'].net.copy(), 'amounts': list(ns['g'].buy_amount['510500.XSHG']), 'orders': orders.copy()})
    ctx.portfolio.positions = {'510500.XSHG': SimpleNamespace(security='510500.XSHG', avg_cost=1.)}
    holder['price'] = 1.1; holder['reply'] = SimpleNamespace(status='held', filled=1., price=1.1); orders.clear(); ns['handle_data'](ctx, None)
    cases.append({'case': 'grid44_partial_held_deletes_layer', 'net': ns['g'].net.copy(), 'position_keys': list(ctx.portfolio.positions), 'orders': orders.copy()})
    ns['g'].net = {'510500.XSHG': 2}; ns['g'].buy_amount = {'510500.XSHG': [20000., 10000.]}; holder['price'] = 1.01; orders.clear()
    ns['handle_data'](ctx, None)
    cases.append({'case': 'grid44_sell_uses_first_amount_nonlot_quantity', 'orders': orders.copy(), 'net': ns['g'].net.copy(),
        'amounts': list(ns['g'].buy_amount['510500.XSHG'])})
    ns['g'].net = {'510500.XSHG': 1}; holder['price'] = .94; holder['reply'] = None; ctx.portfolio.available_cash = 0.; orders.clear(); ns['handle_data'](ctx, None)
    cases.append({'case': 'grid44_zero_cash_still_requests_fixed_buy', 'orders': orders.copy(), 'max_net': ns['g'].max_net})
    try: ns['run_check'](ctx)
    except AttributeError as exc: cases.append({'case': 'grid44_missing_prior_cost_fault', 'error': str(exc)})
    else: raise AssertionError('Original prior-cost fault disappeared')
    ns['cal_avg'](ctx); ctx.portfolio.positions['510500.XSHG'].avg_cost = .5; before = list(ns['g'].buy_amount['510500.XSHG'])
    ns['run_check'](ctx); ns['run_adj'](ctx)
    cases.append({'case': 'grid44_cost_ratio_adjusts_base_only', 'base': ns['g'].base_price['510500.XSHG'],
        'amounts_before': before, 'amounts_after': list(ns['g'].buy_amount['510500.XSHG'])})
    ns['get_price'] = lambda *a, **kw: pd.DataFrame({'close': [1.]}, index=pd.date_range('2021-01-04', periods=1))
    try: ns['handle_data'](ctx, None)
    except KeyError as exc: cases.append({'case': 'grid44_date_integer_fault', 'error': str(exc)})
    else: raise AssertionError('Original grid44 integer label fault disappeared')

    seed_calls = []; requests_seen = []; seed_frame = pd.DataFrame({'high': np.arange(22.)+2., 'low': np.arange(22.)+1.})
    def model(y, x): seed_calls.append({'first': int(y.index[0]), 'last': int(y.index[-1]), 'rows': len(y)}); return SimpleNamespace(fit=lambda: SimpleNamespace(params=[0., 1.], rsquared=.5))
    ns = {'g': SimpleNamespace(), 'datetime': SimpleNamespace(timedelta=timedelta),
        'get_price': lambda *a: requests_seen.append([str(x) for x in a]) or seed_frame,
        'sm': SimpleNamespace(add_constant=lambda x: x, OLS=model)}
    seed_sha = selected(directory, 2, ('set_params',), ns); ns['set_params'](ctx)
    cases.append({'case': 'rsrs_original_seed_request_loop_only', 'requests': requests_seen, 'windows': seed_calls,
        'init': ns['g'].init, 'limit': 'Synthetic regression return stub only to trace original requests/row slicing'})
    try: pd.Series([0., 1.], index=['const', 'low'])[1]
    except KeyError as exc: cases.append({'case': 'rsrs_named_params_integer_fault', 'error': str(exc)})
    else: raise AssertionError('Original params integer-label fault disappeared')
    code, state_sha = order_kernel(directory)
    def state_case(name, score, sys_count=0, level=0, price=100., unit=3.8, positions=None, repeats=1):
        orders = []; g = SimpleNamespace(sys=sys_count, position_level=level, break_price=100., buy=.7, sell=-.7, ceof=.5, unit_limit=4)
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=positions or {}))
        ns = {'g': g, 'context': ctx, 'zscore_rightdev': score, 'security': '000300.XSHG', 'current_price': price,
            'atr': 10., 'vol': unit, 'amount': unit*price, 'log': SimpleNamespace(info=lambda *a: None),
            'order_value': lambda *a: orders.append(['value', *a]), 'order_target': lambda *a: orders.append(['target', *a])}
        for _ in range(repeats): exec(code, ns)
        cases.append({'case': name, 'orders': orders, 'sys': g.sys, 'level': g.position_level, 'break': g.break_price})
    state_case('rsrs_rejected_index_buy_updates_intent', .8)
    state_case('rsrs_add_inclusive_boundary', .8, 10, 3, 100.5)
    state_case('rsrs_reduce_inclusive_boundary', .8, 10, 4, 95.)
    state_case('rsrs_strict_buy_threshold', .7)
    state_case('rsrs_strict_sell_threshold', -.7, 10, 2, positions={'other': None})
    state_case('rsrs_clear_always_targets_index', -.8, 10, 2, positions={'a': None, 'b': None})
    state_case('rsrs_empty_positions_leave_stale_intent', -.8, 10, 2)
    state_case('rsrs_fractional_unit_sys_stays_zero', .8, unit=.3, repeats=6)
    ns = {'tb': talib}; selected(directory, 2, ('ATR',), ns)
    with np.errstate(divide='ignore', invalid='ignore'):
        zero = ns['ATR'](np.ones(20), np.ones(20), np.ones(20), 1000000.)
    try: int(zero[1])
    except OverflowError as exc: cases.append({'case': 'rsrs_zero_atr_nonfinite_unit', 'atr': float(zero[0]),
        'unit_is_finite': bool(np.isfinite(zero[1])), 'conversion_error': str(exc)})
    else: raise AssertionError('Original zero ATR conversion fault disappeared')
    return {'not_a_backtest': True, 'ast_sha256': {'grid84': grid84_sha, 'grid44': grid44_sha, 'seed_trace': seed_sha, 'order_state': state_sha},
        'cases': cases, 'limits': ['Original requests/order returns on synthetic data only; no fills, cash ledger, costs or NAV',
            'Range integer labels only isolate old positional-price semantics; date-label incompatibility separately preserved',
            'RSRS order subtree receives synthetic scores/ATR, not a reconstructed missing statsmodels runtime']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch39 original ATR/algebra components and grid/index-state diagnoses frozen')


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
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch39_dependency_probe', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch39', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch39 fund daily/minute supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch39 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch39 forbidden-network component/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch39.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch39_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch29_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / name).is_file() for name in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
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
    binding(root, directory); validate_catalog(root, directory); validate_sources(directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch39 original grid/ATR algebra accepted; fund dependencies and untradeable original index remain blocked')
    save(Path('docs/handoff/2026-10-06-batch39-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch39/20261006-grid-atr-position')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
