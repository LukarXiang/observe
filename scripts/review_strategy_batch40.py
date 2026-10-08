"""Freeze futures/grid/ETF rules; arithmetic and synthetic state evidence only."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
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
import talib

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch29 import reference_atr
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts import review_strategy_batch39 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch39/20261006-grid-atr-position')
RECEIPT = Path('docs/handoff/2026-10-06-batch39-verification.json')
SOURCES = ('2020年度精选策略/20 分钟K线数据重构 ATR自适应通道 请高手来迭代.txt',
    '2024年度精选策略2/53.超跌网格交易大法V1.2：稳健跑赢大盘-年化13%回撤7%.txt',
    '2024年度精选策略2/64.ETF轮动策略升级-增加盘中止损.txt')
REFERENCES = (('repo/vnpy', 'vnpy/trader/utility.py'), ('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/akshare', 'akshare/futures/futures_zh_sina.py'), ('repo/akshare', 'akshare/fund/fund_etf_em.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch40.py', 'tests/unit/test_batch40_research.py'}))
CORE = previous.CORE.copy()
QUERIES = {
    'futures_daily': {'function': 'futures_zh_daily_sina', 'parameters': {'symbol': 'RB0'},
        'limit': 'Provider continuous daily proxy cannot prove six original8888 indices, historical dominant contracts or intraday settlement'},
    'futures_minute': {'function': 'futures_zh_minute_sina', 'parameters': {'symbol': 'RB2610', 'period': '15'},
        'limit': 'Explicit contract sample is not a verified historical dominant contract or six-future night-session15m history'},
    'fund_minute': {'function': 'fund_etf_hist_min_em', 'parameters': {'symbol': '510050', 'period': '1', 'adjust': '',
        'start_date': '2021-01-04 14:45:00', 'end_date': '2021-01-04 14:50:00'},
        'limit': 'Installed1m endpoint returns latest5days; cannot reconstruct original2021 mixed-grid14:50 or ETF60m inputs'}}


def implementation(): return {name: file_sha(name) for name in FILES}


def indicator():
    result = previous.indicator(); result['ma_type'] = 0
    return result


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch39 differs')
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name; require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
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
    checkpoint(root, directory, 'Batch40 futures/mixed grid/ETF stop source review started')
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
            'use': 'Read-only array/TA-Lib/futures/fund reference; no platform core or ledger replacement'})
    command = ['rg', '--files', '--hidden', 'repo', 'data', '-g', '*jqlib*', '-g', '*technical_analysis*']
    result = subprocess.run(command, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search': {'command': command, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqlib', 'vnpy', 'prettytable')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch40 three full sources frozen; futures/mixed funds/KDJ dependencies retained')


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


def compile_nodes(nodes, namespace, label):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, label, 'exec'), namespace)
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()


def declared(directory):
    trees = [source_tree(directory, n) for n in range(3)]
    row = next(n for n in trees[1].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'origin_param_list_0' for t in n.targets))
    pool = ast.literal_eval(row.value)
    ns = {'g': SimpleNamespace(), 'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, start_date='2000-01-01'),
        'log': SimpleNamespace(info=lambda *a: None)}
    selected(directory, 2, ('set_params',), ns); ns['set_params']()
    return {'mixed_entries': len(pool), 'mixed_unique': len(set(r[0] for r in pool)), 'mixed_pool': [r[0] for r in pool],
        'etf_targets': ns['g'].ETF_targets, 'local_stocks': ns['g'].local_stocks,
        'futures': ['JD8888.XDCE', 'RU8888.XSGE', 'RB8888.XSGE', 'I8888.XDCE', 'J8888.XDCE', 'TA8888.XZCE']}


def inventory(root, directory):
    pools = declared(directory); requested = sorted(set(pools['mixed_pool']) | set(pools['etf_targets'].values()) | set(pools['futures']))
    instruments = [s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for s in requested]
    store = Store(root); state = store.state(SNAPSHOT); rows = {}
    for name in ('instruments', 'minute_universe', 'bars_1d', 'bars_5m', 'corp_actions'):
        frame = store.load_state(state, name, filters=[('instrument', 'in', instruments)])
        rows[name] = {s: int(frame.instrument.eq(s).sum()) for s in instruments}
    return {'snapshot': SNAPSHOT, 'declared': pools, 'requested_instruments': instruments, 'table_rows': rows,
        'futures_tables': {name: value for name, value in state['tables'].items() if 'future' in name},
        'not_a_backtest': True, 'limits': ['Published stock5m cannot replace fund1m/60m or futures15m/night-calendar',
            'Existing stock histories are not original platform KDJ or complete mixed-fund executions']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['Daily-index14-row arithmetic samples only; not original fund ranking/60m stopping/futures bars',
            'No KDJ seed/runtime reconstructed; no orders, fills, costs or NAV simulated']})
    save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch40 actual mixed/fund/future inventory and accepted arithmetic operands frozen')


def arithmetic_kernel(directory):
    fn = next(n for n in source_tree(directory, 2).body if isinstance(n, ast.FunctionDef) and n.name == 'get_signal')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    names = ('now_close', 'previous_close', 'ma_filter', 'ma_status', 'moment', 'amount')
    nodes = [n for n in loop.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    require(len(nodes) == 6, 'Original ETF arithmetic structure changed')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, '<original-etf-arithmetic-only>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def compute(directory):
    validate_sources(directory); frame, _ = previous.inputs(directory); backend = indicator()
    require(backend == read(directory / 'existing-apis.json')['indicator'], 'Frozen indicator backend differs')
    require(len(frame) >= 14, 'Incomplete ETF arithmetic window')
    code, sha = arithmetic_kernel(directory); close = frame.close.to_numpy(dtype=float)
    errors = np.zeros(3); samples = []; boundaries = []
    for k in range(13, len(frame)):
        window = close[k-13:k+1]
        # Negative labels isolate the legacy positional expression; actual date-label failure is diagnosed separately.
        ns = {'price_data': pd.DataFrame({'close': window}, index=range(-14, 0)), 'ta': talib,
            'g': SimpleNamespace(moment_period=13, ma_period=10, type_num=1), 'total_value': 1000000.}
        exec(code, ns)
        actual = [float(ns['moment']), float(ns['ma_filter']), float(ns['ma_status'])]
        ma = math.fsum(map(float, window[-10:])) / 10
        expected = [(float(window[-1])-float(window[1]))/float(window[1])*100, ma, float(window[-1])-ma]
        errors = np.maximum(errors, np.abs(np.asarray(actual)-np.asarray(expected)))
        flags = [actual[0] > 0, actual[2] > 0]; refs = [expected[0] > 0, expected[2] > 0]
        target = int(1000000./float(window[-1])/100)*100
        require(ns['amount'] == target and all(math.isfinite(x) for x in (*actual, *expected)), 'Nonfinite/changed ETF arithmetic')
        if flags != refs: boundaries.append({'date': frame.date.iloc[k], 'actual': actual, 'reference': expected, 'flags': flags, 'reference_flags': refs})
        samples.append([*actual, int(ns['amount'])])
    return {'not_a_backtest': True, 'platform_equivalent': False, 'backend': backend, 'ast_sha256': sha,
        'rows': len(frame), 'windows': len(samples), 'first_end_date': frame.date.iloc[13], 'last_end_date': frame.date.iloc[-1],
        'max_differences': errors.tolist(), 'strict_positive_boundaries': boundaries,
        'sample_capital': 1000000., 'sample_sha256': hashlib.sha256(np.asarray(samples, dtype='<f8').tobytes()).hexdigest(),
        'limits': ['Daily index operands only; not24-fund returns/ranking,13:30 60m MA20 stops or tradeable signals',
            'Original minus13 label isolated as positional12-row gap; source not repaired', 'No KDJ, irregular-future bars, fills or NAV reconstructed']}


def futures_namespace(directory):
    g = SimpleNamespace(); ns = {'g': g, 'np': np, 'talib': talib, 'log': SimpleNamespace(info=lambda *a: None),
        'get_dominant_future': lambda ins: ins+'TEST', 'get_security_info': lambda *a: None}
    cls = next(n for n in source_tree(directory, 0).body if isinstance(n, ast.ClassDef) and n.name == 'ArrayManager')
    sha = compile_nodes([cls], ns, '<original-futures-array-only>')
    names = ('set_parameter', 'set_future_list', 'DataPrepare', 'market_open', 'Trade', 'TrailingStop', 'Dont_Re_entry', 'replace_old_futures', 'get_future_code', 'get_lots')
    nodes = [n for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names), 'Original active futures functions changed')
    fn_sha = compile_nodes(nodes, ns, '<original-active-futures-functions>'); ns['set_parameter'](SimpleNamespace())
    return ns, sha, fn_sha


def grid_namespace(directory):
    g = SimpleNamespace(); ns = {'g': g, 'np': np, 'pd': pd, 'copy': __import__('copy'), 'datetime': __import__('datetime'),
        'log': SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, set_level=lambda *a: None),
        'set_option': lambda *a: None, 'set_benchmark': lambda *a: None, 'set_slippage': lambda *a, **k: None,
        'PriceRelatedSlippage': lambda x: x, 'set_order_cost': lambda *a, **k: None, 'OrderCost': lambda **k: k,
        'run_monthly': lambda *a, **kw: None, 'run_daily': lambda *a, **kw: None}
    nodes = [n for n in source_tree(directory, 1).body if isinstance(n, ast.Assign)]
    globals_sha = compile_nodes(nodes, ns, '<original-grid-literal-globals>')
    sha = selected(directory, 1, ('initialize', 'choose_stocks', 'reset', 'buy', 'cover', 'sell', 'check_buy_point'), ns)
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 14, 50), portfolio=SimpleNamespace(positions={}, positions_value=0., available_cash=1000000.))
    ns['choose_stocks'](ctx)
    data = {s: SimpleNamespace(last_price=20., high_limit=100., low_limit=0., paused=False, is_st=False) for s in g.origin_param_list_all}
    orders = []; ns.update(get_current_data=lambda: data, check_buy_point=lambda *a: 1.,
        get_bars=lambda *a, **kw: {'close': np.ones(90)*100.}, order_value=lambda *a: orders.append(list(a)),
        order_target_value=lambda *a: orders.append(list(a)))
    return ns, ctx, data, orders, globals_sha, sha


def etf_namespace(directory):
    ns = {'g': SimpleNamespace(), 'pd': pd, 'ta': talib, 'datetime': __import__('datetime'),
        'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, start_date=datetime(2000, 1, 1).date()),
        'log': SimpleNamespace(info=lambda *a: None)}
    sha = selected(directory, 2, ('set_params', 'get_before_after_trade_days', 'before_market_open', 'get_signal', 'ETF_trade', 'hold_check'), ns)
    ns['set_params'](); return ns, sha


def diagnostics(directory):
    cases = []; ns, array_sha, future_sha = futures_namespace(directory); g = ns['g']; AM = ns['ArrayManager']
    am = AM(3); am.updateBar(10., 11., 9., 10.); am.updateBar(12., 13., 8., 12.); am.updateBarArray(); am.clear()
    cases.append({'case': 'future_array_accumulates_then_shifts', 'arrays': {k: v.tolist() for k, v in am.VarsArrays.items()}, 'vars': am.Vars.copy()})
    g.instruments = ['RB', 'I']; g.TodayBar = {ns['get_future_code'](s): 2 for s in g.instruments}; events = []
    for k, s in enumerate(g.instruments):
        code = ns['get_future_code'](s); g.AM[code] = AM(); g.AM[code].updateBar(100.*(k+1), 100.*(k+1), 100.*(k+1), 100.*(k+1))
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 11, 30))
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': [999.], 'open': [999.], 'high': [999.], 'low': [999.]}, index=[ctx.current_dt])
    ns['market_open'] = lambda *a: events.append({k: v.exportArray('close')[-1].item() for k, v in g.AM.items()})
    ns['DataPrepare'](ctx); cases.append({'case': 'future_cut_omits_current_bar_and_dispatches_all', 'events': deepcopy(events),
        'vars': {k: v.Vars.copy() for k, v in g.AM.items()}})
    ctx.current_dt = datetime(2021, 1, 4, 9, 15); events.clear(); ns['DataPrepare'](ctx)
    cases.append({'case': 'future_no_night0915_ignores_current', 'events': events.copy(), 'vars': {k: v.Vars.copy() for k, v in g.AM.items()}})
    ctx.current_dt = datetime(2021, 1, 4, 9, 30)
    try: ns['DataPrepare'](ctx)
    except KeyError as exc: cases.append({'case': 'future_date_integer_fault', 'error': str(exc)})
    else: raise AssertionError('Future date-index fault disappeared')
    ns, _, _ = futures_namespace(directory); g = ns['g']; orders = []; ns['order_target'] = lambda *a, **kw: orders.append([list(a), kw])
    code = ns['get_future_code']('RB'); g.close = np.array([90.]); g.ATR[code] = 1.
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 11, 30),
        portfolio=SimpleNamespace(long_positions={'RBTEST': SimpleNamespace(total_amount=1)}, short_positions={}))
    try: ns['TrailingStop'](ctx, 'RBTEST', code)
    except KeyError as exc: cases.append({'case': 'future_missing_high_low_fault', 'error': str(exc)})
    else: raise AssertionError('Future uninitialized extrema fault disappeared')
    g.HighPrice[code] = 100.; g.LowPrice[code] = False; g.TodayBar[code] = 2; g.Times[code] = 40
    g.AM[code] = AM(); g.AM[code].VarsArrays['close'][:] = 100.; ns['TrailingStop'](ctx, 'RBTEST', code)
    stopped = g.Reentry_long; ns['Dont_Re_entry'](ctx, code, 'RB')
    cases.append({'case': 'future_stop_counter_not_reset_immediate_reentry_clear', 'orders': deepcopy(orders), 'stopped': stopped,
        'after': g.Reentry_long, 'times': g.Times[code], 'low': g.LowPrice[code]})
    g.HighPrice[code] = False; g.LowPrice[code] = False; g.close = np.array([200.]); orders.clear()
    ctx.portfolio.long_positions = {}; ctx.portfolio.short_positions = {'RBTEST': SimpleNamespace(total_amount=1)}
    ns['TrailingStop'](ctx, 'RBTEST', code)
    cases.append({'case': 'future_false_short_low_stays_zero', 'low': g.LowPrice[code], 'orders': orders.copy()})
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'open': [100.]}); g.ATR[code] = 3.
    lots = ns['get_lots'](50000., 'RB'); cases.append({'case': 'future_fractional_lots', 'lots': float(lots)})
    ctx.portfolio.long_positions = {'RBTEST': SimpleNamespace(total_amount=2)}; ctx.portfolio.short_positions = {'RBTEST': SimpleNamespace(total_amount=3)}
    orders.clear(); ns['replace_old_futures'](ctx, 'RB', 'RBNEW')
    cases.append({'case': 'future_roll_intent_both_sides', 'orders': deepcopy(orders)})
    channel = []
    for daily in (2, 3):
        ns, _, _ = futures_namespace(directory); g = ns['g']; g.MappingReal = {'RB': 'RBTEST'}; g.TodayBar[code] = daily
        g.AM[code] = AM(); values = np.ones(daily*15+1)*100.; values[-1] = 110.
        for key, data in (('close', values), ('high', values+1.), ('low', values-1.)): g.AM[code].VarsArrays[key][-len(values):] = data
        ns['Trade'] = lambda *a: None; ns['Dont_Re_entry'] = lambda *a: None
        with redirect_stdout(io.StringIO()): ns['market_open'](ctx)
        expected = reference_atr(values+1., values-1., values, daily*15)
        channel.append({'daily_bars': daily, 'atr': float(g.ATR[code]), 'reference': expected, 'mid': float(g.MidLine), 'signal': g.Signal})
    cases.append({'case': 'future_original_channel_synthetic_only', 'samples': channel})

    ns, ctx, data, orders, grid_globals, grid_sha = grid_namespace(directory); g = ns['g']
    cases.append({'case': 'grid_declared_duplicate_pool', 'entries': len(ns['origin_param_list_0']), 'unique': len(g.origin_param_list_all)})
    with redirect_stdout(io.StringIO()): ns['buy'](ctx)
    cases.append({'case': 'grid_rejected_buys_record_eleven_intents', 'orders': deepcopy(orders), 'recorded': len(g.buy_dict)})
    stock = next(iter(g.buy_dict)); value = next(iter(next(iter(g.buy_dict[stock].values())).values()))[0]
    cases.append({'case': 'grid_initial_record_nonlot_plus_one', 'price': data[stock].last_price, 'value': float(value), 'implied_quantity': value/data[stock].last_price})
    ns, ctx, data, orders, _, _ = grid_namespace(directory); g = ns['g']; a, b = list(g.origin_param_list_all)[:2]
    g.origin_param_list_all = {a: g.origin_param_list_all[a], b: g.origin_param_list_all[b]}; ns['origin_param_list_0'] = [ns['origin_param_list_0'][0]]
    g.last_opt_price = {a: {20.: ('2021-1-3', 100.)}}; g.buy_dict = {a: {20.: {'2021-1-3': (100., 30.)}}}
    with redirect_stdout(io.StringIO()): ns['reset'](ctx)
    cases.append({'case': 'grid_reset_uses_residual_stock_key', 'requested_reset': a, 'other_key': b, 'mapped_symbol': g.origin_param_list_all[b][0]})
    ns, ctx, data, orders, _, _ = grid_namespace(directory); g = ns['g']; stock = next(iter(g.origin_param_list_all))
    ctx.portfolio.positions = {stock: SimpleNamespace(total_amount=200, avg_cost=100.)}; data[stock].last_price = 120.
    g.last_opt_price = {stock: {100.: ('2021-9-30', 20000.)}}; g.buy_dict = {stock: {100.: {'2021-9-30': (20000., 110.)}}}
    ctx.current_dt = datetime(2021, 10, 1, 14, 50); ns['sell'](ctx)
    cases.append({'case': 'grid_unpadded_date_blocks_next_month_sell', 'orders': orders.copy(), 'lexical_later': '2021-10-1' > '2021-9-30'})
    data[stock].last_price = 200.; g.last_opt_price = {stock: {100.: ('2021-1-1', 20000.)}}; g.buy_dict = {stock: {100.: {'2021-1-1': (20000., 110.)}}}
    ctx.current_dt = datetime(2021, 1, 2, 14, 50); orders.clear(); ns['sell'](ctx); first = deepcopy(g.buy_dict[stock])
    ctx.current_dt = datetime(2021, 1, 3, 14, 50); ns['sell'](ctx)
    cases.append({'case': 'grid_half_sell_residual_revalued_current_price', 'first': first, 'second': deepcopy(g.buy_dict[stock]), 'orders': deepcopy(orders)})

    ns, etf_sha = etf_namespace(directory); g = ns['g']; fund_codes = [s for s in g.ETF_targets.values() if s != '399006.XSHE']
    ns['get_all_securities'] = lambda **kw: pd.DataFrame({'start_date': [datetime(2000, 1, 1).date()]*len(fund_codes)}, index=fund_codes)
    ns['get_all_trade_days'] = lambda: pd.bdate_range('2020-01-01', '2021-01-04').date
    ctx = SimpleNamespace(previous_date=datetime(2021, 1, 1).date(), current_dt=datetime(2021, 1, 4, 21),
        portfolio=SimpleNamespace(total_value=1000000., positions={}))
    ns['before_market_open'](ctx)
    cases.append({'case': 'etf_fund_list_removes_index_mapping', 'declared': len(g.ETF_targets), 'fund_filtered': len(g.ETFList),
        'index_retained': '399006.XSHE' in g.ETFList.values(), 'chip_local_eligible': '512760.XSHG' in g.local_stocks})
    g.ETFList = {'000300.XSHG': '510300.XSHG'}; ns['get_current_data'] = lambda: {}; ns['get_price'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(14.)+100.}, index=range(-14, 0))
    with redirect_stdout(io.StringIO()):
        try: ns['get_signal'](ctx)
        except AttributeError as exc: cases.append({'case': 'etf_original_append_fault', 'error': str(exc)})
        else: raise AssertionError('ETF original append fault disappeared')
    ns['get_price'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(14.)+100.}, index=pd.date_range('2021-01-01', periods=14))
    with redirect_stdout(io.StringIO()):
        try: ns['get_signal'](ctx)
        except KeyError as exc: cases.append({'case': 'etf_original_date_integer_fault', 'error': str(exc)})
        else: raise AssertionError('ETF original date fault disappeared')
    orders = []; messages = []; g.sells = ['old']; g.purchases = ['new']; g.df_etf = pd.DataFrame({'基金代码': ['new'], '股数': [100]})
    ns['order_target'] = lambda stock, quantity: orders.append({'stock': stock, 'quantity': quantity.tolist() if isinstance(quantity, np.ndarray) else quantity, 'type': type(quantity).__name__})
    ns['ETF_trade'](ctx); cases.append({'case': 'etf_target_is_array_and_sell_first', 'orders': deepcopy(orders)})
    ctx.portfolio.positions = {'fund': SimpleNamespace(total_amount=100, closeable_amount=0)}
    ns['send_message'] = lambda text: messages.append(text); ns['order_target_value'] = lambda *a: orders.append(list(a))
    ns['attribute_history'] = lambda *a: pd.DataFrame({'close': [100.]*21+[50.]}, index=range(-22, 0)); orders.clear(); ns['hold_check'](ctx)
    cases.append({'case': 'etf_stop_message_without_closeable_order', 'orders': orders.copy(), 'messages': messages.copy()})
    ctx.portfolio.positions['fund'].closeable_amount = 100; messages.clear(); ns['hold_check'](ctx)
    cases.append({'case': 'etf_stop_sell_only_when_closeable', 'orders': deepcopy(orders), 'messages': messages.copy()})
    ns['attribute_history'] = lambda *a: pd.DataFrame({'close': [100.]*21+[50.]}, index=pd.date_range('2021-01-01', periods=22))
    try: ns['hold_check'](ctx)
    except KeyError as exc: cases.append({'case': 'etf_stop_date_integer_fault', 'error': str(exc)})
    else: raise AssertionError('ETF stop date fault disappeared')
    result = {'not_a_backtest': True, 'ast_sha256': {'array': array_sha, 'futures': future_sha, 'grid_globals': grid_globals, 'grid': grid_sha, 'etf': etf_sha},
        'cases': cases, 'limits': ['Synthetic request/state/intent records only; no fills, costs, settlement, messages or NAV',
            'KDJ stub only isolates order bookkeeping; platform indicator/seed not reconstructed', 'No original dates, mapping or economic rules repaired']}
    # Price-keyed original dictionaries are normalized for JSON, without changing numeric state.
    return json.loads(json.dumps(result, allow_nan=False))


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory)
    with redirect_stdout(io.StringIO()): diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch40 original ETF arithmetic and future/grid/stop state diagnoses frozen')


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
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch40_dependency_probe', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch40', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch40 existing futures/fund supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch40 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch40 forbidden-network arithmetic/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch40.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch40_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch39_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


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
    binding(root, directory); validate_catalog(root, directory); validate_sources(directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    with redirect_stdout(io.StringIO()): require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch40 future/grid/ETF arithmetic accepted; original trade dependencies remain missing')
    save(Path('docs/handoff/2026-10-06-batch40-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch40/20261006-futures-mixed-grid')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
