"""Freeze trend/Alpha022/balanced-fund rules and verify available arithmetic."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
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

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch37 as previous
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch37/20261006-rps-lof-chase')
RECEIPT = Path('docs/handoff/2026-10-06-batch37-verification.json')
SOURCES = ('2023年度精选策略/33.趋势交易5.0 无择时-2年5倍不是梦.txt',
    '2023年度精选策略/61.分享一个最近两年非常有效的因子.txt', '2021年度精选策略/7.股债波动平衡.txt')
REFERENCES = (('repo/vnpy', 'vnpy/alpha/dataset/datasets/alpha_101.py'),
    ('repo/vnpy', 'vnpy/alpha/dataset/ts_function.py'), ('repo/vnpy', 'vnpy/alpha/dataset/cs_function.py'),
    ('repo/vnpy', 'tests/test_alpha101.py'), ('repo/baostock', 'baostock/demo/demo_hs300_stocks.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch38.py', 'tests/unit/test_batch38_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'price-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'offline-catalog.json',
    'component-research.corrected.json', 'component-clarification.json'}
QUERIES = {
    'hs3002021': {'provider': 'baostock', 'function': 'query_hs300_stocks', 'parameters': {'date': '2021-01-04'},
        'limit': 'Single dated weekly vendor pool cannot prove all original platform decision-date pools'},
    'lof161005': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz161005'},
        'limit': 'Single LOF daily sample cannot supply eight-fund events/status/execution'},
    'bond511010': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh511010'},
        'limit': 'Bond ETF price history is not complete eight-fund adjusted prices or verified trading rules'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch37 differs')
    rows = {}
    for name in ('price-input.parquet', 'input-analysis.json', 'input-binding.json'):
        path = ACCEPTED / name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted price evidence changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': previous.binding(root, ACCEPTED), 'files': rows, 'not_a_backtest': True}
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
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    bound = binding(root); apis = api_evidence()
    checkpoint(root, directory, 'Batch38 trend/Alpha022/balanced-fund source research started')
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
            'use': 'Read-only Alpha022 expression/operator schema and dated pool API; not jqlib replacement'})
    searches = []
    for command in (['rg', '--files', '--hidden', 'data', '-g', '*constituent*', '-g', '*hs300*', '-g', '*zz500*'],
        ['rg', '-n', '^def alpha_022|^def alpha022', 'repo', '-g', '*.py', '-g', '*.txt']):
        result = subprocess.run(command, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
        searches.append({'command': command, 'returncode': result.returncode, 'matches': result.stdout.splitlines()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path), 'dependency_searches': searches,
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqlib', 'polars', 'vnpy')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch38 three complete sources frozen; original pool/core/fund dependencies remain missing')


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


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory)
    path = directory / 'price-input.parquet'; require(not path.exists(), 'Input copy exists')
    shutil.copyfile(bound['files'][path.name]['file'], path); frame = pd.read_parquet(path)
    profile = read(ACCEPTED / 'input-analysis.json'); entries = Store(root).state(SNAPSHOT)['tables'].get('index_constituents', {})
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(path), 'sha256': file_sha(path),
        'rows': len(frame), 'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max(), 'pool': profile['pool'],
        'index_constituent_partitions': entries, 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Ten verified stocks only arithmetic samples, not original HS300 pools or fund data',
            'Complete traded-row windows only; no cross-stock fill or default history alignment substituted',
            'No jqlib Alpha022 core reconstructed, no stock/fund ranking or trading NAV computed']})
    return archive(root, 'Batch38 accepted ten-stock input bytes and actual missing pool inventory frozen')


def inputs(directory):
    frame, excluded = previous.inputs(directory)
    require(np.isfinite(frame[['high_adj', 'volume']]).all().all() and frame.high_adj.gt(0).all() and frame.volume.ge(0).all(), 'Unknown high/volume operands')
    return frame, excluded


def trend_kernel(directory):
    fn = next(n for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name == 'market_open')
    nodes = []
    for node in fn.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == 'df' for t in node.targets) and isinstance(node.value, ast.Attribute): nodes.append(node)
            elif any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value in ('ma100', 'ma60', 'ma30') for t in node.targets): nodes.append(node)
    for name in ('x', 'lis'):
        nodes.append(next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    require(len(nodes) == 6, 'Original trend arithmetic structure changed')
    ratios = {}
    for name in ('s_fall', 's_vol_ratio'):
        node = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        ratios[name] = compile(ast.Expression(node.value), f'<original-{name}>', 'eval')
    return compile(module, '<original-complete-window-trend>', 'exec'), ratios, hashlib.sha256(ast.dump(module).encode()).hexdigest()


def original_trend(directory, close, high, volume, kernel=None):
    close = np.asarray(close, dtype=float); high = np.asarray(high, dtype=float); volume = np.asarray(volume, dtype=float)
    require(len(close) == 100 and len(high) == 30 and len(volume) == 180 and
        np.isfinite(close).all() and np.isfinite(high).all() and np.isfinite(volume).all() and
        min(close) > 0 and min(high) > 0 and min(volume) >= 0, 'Invalid complete trend operands')
    code, ratios, _ = kernel or trend_kernel(directory)
    ns = {'df0': pd.DataFrame({'operator_sample': close}), 'np': np, 'linregress': linregress}
    exec(code, ns); row = ns['df'].iloc[0]; k, intercept, r = map(float, ns['lis'][0])
    fall = float(eval(ratios['s_fall'], {'high_max_30': float(max(high)), 'close_1': float(close[-1])}))
    vol = float(eval(ratios['s_vol_ratio'], {'df_vol': pd.DataFrame({'operator_sample': volume})}).iloc[0])
    slope_ratio = float(np.divide(k, intercept))
    flags = [close[-1] > row.ma100, row.ma30 > row.ma60, row.ma60 > row.ma100, r > .5, slope_ratio > .005, fall <= 1.1, vol <= 1.5]
    return [float(row.ma100), float(row.ma60), float(row.ma30), k, intercept, r, fall, vol], [bool(x) for x in flags]


def reference_trend(close, high, volume):
    close = list(map(float, close)); xmean = 49.5; ymean = math.fsum(close)/100
    sx = math.fsum((k-xmean)**2 for k in range(100)); sy = math.fsum((v-ymean)**2 for v in close)
    cov = math.fsum((k-xmean)*(v-ymean) for k, v in enumerate(close)); k = cov/sx; intercept = ymean-k*xmean
    r = cov/math.sqrt(sx*sy) if sy else math.nan
    ma60 = math.fsum(close[-60:])/60; ma30 = math.fsum(close[-30:])/30; fall = max(high)/close[-1]
    total = math.fsum(volume); vol = math.fsum(volume[-7:])/7/(total/180) if total else math.nan
    ratio = float(np.divide(k, intercept))
    flags = [close[-1] > ymean, ma30 > ma60, ma60 > ymean, r > .5, ratio > .005, fall <= 1.1, vol <= 1.5]
    return [ymean, ma60, ma30, k, intercept, r, fall, vol], [bool(flag) for flag in flags]


def normalised_components(document):
    corrected = deepcopy(document); changes = 0
    for stock in corrected['stocks']:
        for boundary in stock['boundaries']:
            flags = boundary['reference_flags']
            for k, flag in enumerate(flags):
                if type(flag) is bool: continue
                require(type(flag) is str and flag in ('True', 'False'), 'Unknown boundary flag encoding')
                flags[k] = flag == 'True'; changes += 1
    return corrected, changes


def component_file(directory):
    original = directory / 'component-research.json'; corrected = directory / 'component-research.corrected.json'
    proof = read(directory / 'component-clarification.json')
    require(proof['original_sha256'] == file_sha(original) and proof['corrected_sha256'] == file_sha(corrected), 'Component correction binding changed')
    expected, changes = normalised_components(read(original))
    require(changes > 0 and proof['changed_boolean_flags'] == changes and read(corrected) == expected, 'Component correction scope changed')
    for name, sha in proof['before_fix_sha256'].items(): require(file_sha(directory / name) == sha, 'Before-fix evidence changed')
    require(proof['not_a_backtest'] is True and proof['arithmetic_changed'] is False, 'Component correction fidelity changed')
    return corrected


def clarify(root, directory):
    binding(root, directory); validate_sources(directory)
    corrected, changes = normalised_components(read(directory / 'component-research.json'))
    require(changes > 0, 'No boolean encoding correction needed')
    save(directory / 'component-research.corrected.json', corrected)
    save(directory / 'component-clarification.json', {
        'original_sha256': file_sha(directory / 'component-research.json'),
        'corrected_sha256': file_sha(directory / 'component-research.corrected.json'),
        'changed_boolean_flags': changes, 'before_fix_sha256': {name: file_sha(directory / name) for name in
            ('review_strategy_batch38.before-bool-fix.py', 'test_batch38_research.before-bool-fix.py')},
        'not_a_backtest': True, 'arithmetic_changed': False,
        'regression': 'test_reference_flags_are_native_json_booleans failed before the fix: 1 failed / 22 deselected',
        'reason': 'NumPy bool in reference_flags was serialized as a string by default=str; normalize diagnostic type only'})
    component_file(directory)
    return archive(root, 'Batch38 boolean-only component archive correction frozen; original arithmetic and artifact preserved')


def reference_volatility(close, down=False):
    values = [math.log(float(b)/float(a)) for a, b in zip(close[:-1], close[1:], strict=True)]
    if down: values = [min(v, 0.) for v in values]
    mean = math.fsum(values)/len(values); sigma = math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1))
    clipped = [max(mean-3*sigma, min(mean+3*sigma, v)) for v in values]
    center = math.fsum(clipped)/len(clipped)
    return math.sqrt(math.fsum((v-center)**2 for v in clipped)/(len(clipped)-1))*math.sqrt(250)*100


def compute(directory):
    validate_sources(directory); frame, excluded = inputs(directory); kernel = trend_kernel(directory)
    ns = {'np': np, 'math': math}; volatility_sha = selected(directory, 2, ('get_volatility',), ns)
    result = {'not_a_backtest': True, 'platform_equivalent': False, 'excluded_known_paused_rows': excluded,
        'source_ast_sha256': {'trend': kernel[2], 'volatility': volatility_sha}, 'stocks': []}
    for instrument, group in frame.groupby('instrument', sort=True):
        group = group.reset_index(drop=True); close = group.close_adj.to_numpy(); high = group.high_adj.to_numpy(); factors = group.back_factor.to_numpy(); volume = group.volume.to_numpy()
        trend = []; wave = []; boundaries = []; max_diff = np.zeros(8); max_wave_diff = 0.; eligible = 0
        for k in range(39, len(group)):
            actual_wave = float(ns['get_volatility'](pd.DataFrame({'close': close[k-39:k+1]}), False))
            expected_wave = reference_volatility(close[k-39:k+1]); max_wave_diff = max(max_wave_diff, abs(actual_wave-expected_wave)); wave.append(actual_wave)
            if k >= 179:
                c = close[k-99:k+1]/factors[k]; h = high[k-29:k+1]/factors[k]; v = volume[k-179:k+1]
                actual, flags = original_trend(directory, c, h, v, kernel); expected, ref_flags = reference_trend(c, h, v)
                require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), 'Trend finite masks differ')
                finite = np.isfinite(actual); max_diff[finite] = np.maximum(max_diff[finite], np.abs(np.array(actual)[finite]-np.array(expected)[finite]))
                if flags != ref_flags:
                    boundaries.append({'date': group.date.iloc[k], 'actual': [x if math.isfinite(x) else None for x in actual],
                        'reference': [x if math.isfinite(x) else None for x in expected], 'flags': flags, 'reference_flags': ref_flags})
                trend.append(actual); eligible += all(flags)
        result['stocks'].append({'instrument': instrument, 'traded_rows': len(group), 'trend_windows': len(trend),
            'trend_first_end_date': group.date.iloc[179], 'wave_windows': len(wave), 'wave_first_end_date': group.date.iloc[39],
            'last_end_date': group.date.iloc[-1], 'trend_columns': ['ma100', 'ma60', 'ma30', 'k', 'intercept', 'r', 'high_close_ratio', 'volume_ratio'],
            'trend_max_differences': max_diff.tolist(), 'wave_max_difference': max_wave_diff,
            'trend_sha256': hashlib.sha256(np.asarray(trend, dtype='<f8').tobytes()).hexdigest(),
            'wave_sha256': hashlib.sha256(np.asarray(wave, dtype='<f8').tobytes()).hexdigest(), 'all_filters_true': eligible, 'boundaries': boundaries})
    return result


def diagnostics(directory):
    from datetime import datetime
    cases = []; log = SimpleNamespace(info=lambda *a: None)
    ns = {'g': SimpleNamespace(), 'log': log, 'get_current_data': lambda: {}, 'get_index_stocks': lambda *a: ['synthetic'],
        'history': lambda *a, **kw: pd.DataFrame({'synthetic': np.arange(1., 101.)})}
    selected(directory, 0, ('market_open', 'orderStocks'), ns)
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 9, 30), portfolio=SimpleNamespace(positions={}, total_value=1000.))
    try: ns['market_open'](ctx)
    except TypeError as exc: cases.append({'case': 'trend_removed_fillna_method', 'error': str(exc)})
    else: raise AssertionError('Original fillna method fault disappeared')
    frame = pd.DataFrame({'a': [1., 2.], 'b': [np.nan, 4.]}, index=pd.date_range('2021-01-01', periods=2)).T
    try: frame[99]
    except KeyError as exc: cases.append({'case': 'trend_datetime_column99_fault', 'error': str(exc)})
    else: raise AssertionError('Date column compatibility fault disappeared')
    cases.append({'case': 'trend_cross_stock_ffill_direction', 'before_b': [None, 4.], 'after_b': frame.ffill().loc['b'].tolist()})
    orders = []; ns.update(order_target_value=lambda *a: orders.append(list(a)), g=SimpleNamespace(buy_list=['keep', 'new']))
    ctx.portfolio.positions = {'old': None, 'keep': None}; ns['orderStocks'](ctx)
    cases.append({'case': 'trend_sell_adjust_buy_equal_total', 'orders': orders.copy()})
    ns['g'].buy_list = []; orders.clear(); ns['orderStocks'](ctx)
    cases.append({'case': 'trend_empty_targets_clear', 'orders': orders.copy()})
    c = np.arange(100., 200.)
    base, _ = original_trend(directory, c, c[-30:], np.ones(180)); scaled, _ = original_trend(directory, c*2, c[-30:]*2, np.ones(180))
    cases.append({'case': 'trend_absolute_slope_price_scale', 'k': base[3], 'scaled_k': scaled[3], 'k_i': base[3]/base[4], 'scaled_k_i': scaled[3]/scaled[4]})
    calls = []; ns = {'g': SimpleNamespace(), 'log': log, 'alpha_022': lambda *a: calls.append([str(x) for x in a]) or pd.Series([1.], index=['a'])}
    selected(directory, 1, ('before_market_open', 'market_open', 'paused_filter'), ns)
    ctx.previous_date = datetime(2021, 1, 1).date()
    try: ns['before_market_open'](ctx)
    except AttributeError as exc: cases.append({'case': 'alpha022_actual_call_removed_order', 'calls': calls, 'error': str(exc)})
    else: raise AssertionError('Original Series.order fault disappeared')
    orders = []; ns.update(order_target=lambda *a: orders.append(['target', *a]), order_value=lambda *a: orders.append(['value', *a]))
    ns['g'].stocks_to_sell = ['old']; ns['g'].stocks_to_buy = ['a', 'b']; ctx.portfolio.available_cash = 800.
    ns['market_open'](ctx); cases.append({'case': 'alpha_sell_then_new_cash_only', 'orders': orders.copy(), 'cash_per_new': ns['g'].cash})
    ns['g'].stocks_to_sell = []; ns['g'].stocks_to_buy = []; orders.clear(); ns['market_open'](ctx)
    cases.append({'case': 'alpha_no_new_targets_cash_zero', 'orders': orders.copy(), 'cash': ns['g'].cash})
    ns['get_current_data'] = lambda: {'a': SimpleNamespace(paused=False), 'b': SimpleNamespace(paused=True)}
    cases.append({'case': 'alpha_current_pause_filter', 'selected': ns['paused_filter'](['a', 'b'])})

    ns = {'np': np, 'pd': pd, 'math': math, 'g': SimpleNamespace(), 'log': log}
    selected(directory, 2, ('get_volatility', 'need_balance', 'rebalance', 'market_open', 'after_market_close'), ns)
    try: ns['get_volatility'](pd.DataFrame({'close': []}))
    except AttributeError as exc: cases.append({'case': 'wave_empty_np_NaN_removed', 'error': str(exc)})
    else: raise AssertionError('Original np.NaN fault disappeared')
    rising = np.exp(np.arange(40)*.01); flat = np.ones(40); outlier = np.exp(np.r_[np.zeros(39), 1.])
    for name, values, down in [('flat', flat, False), ('down_only_rising', rising, True), ('outlier_clipped', outlier, False)]:
        actual = float(ns['get_volatility'](pd.DataFrame({'close': values}), down)); expected = reference_volatility(values, down)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), 'Synthetic wave differs')
        cases.append({'case': f'wave_{name}', 'actual': actual, 'reference': expected})
    ns['g'].position = pd.DataFrame({'position': [.5, .5]}, index=['a', 'b'])
    ctx.portfolio.positions = {'a': SimpleNamespace(value=500.)}; ctx.portfolio.total_value = 1000.
    cases.append({'case': 'balance_missing_target_ignored', 'result': bool(ns['need_balance'](ctx))})
    orders = []; ns['order_target_value'] = lambda *a: orders.append(list(a)); ctx.portfolio.positions = {'a': SimpleNamespace(value=700.)}
    ns['rebalance'](ctx); cases.append({'case': 'balance_sell_overweight_first', 'orders': orders})
    histories = []; ns['history'] = lambda *a, **kw: histories.append({'args': list(a), 'kwargs': kw}) or pd.DataFrame({s: 100.+np.sin(np.arange(40)/3+i) for i, s in enumerate(kw['security_list'])})
    ctx.portfolio.positions = {'a': SimpleNamespace(value=1000.)}; ctx.portfolio.available_cash = 0.
    with redirect_stdout(io.StringIO()):
        ns['market_open'](ctx)
        cases.append({'case': 'balance_monday_with_positions_skips', 'history_calls': len(histories)})
        ctx.current_dt = datetime(2021, 1, 8, 9, 30); ns['market_open'](ctx)
    cases.append({'case': 'balance_friday_original_pool_weights', 'requests': histories, 'weights': ns['g'].position.weight.tolist(),
        'position_sum': float(ns['g'].position.position.sum())})
    ns['history'] = lambda *a, **kw: pd.DataFrame({s: np.ones(40) for s in kw['security_list']})
    with redirect_stdout(io.StringIO()): ns['market_open'](ctx)
    cases.append({'case': 'balance_zero_wave_invalid_weights', 'zero_waves': int(ns['g'].position.wave.eq(0).sum()),
        'nan_weights': int(ns['g'].position.position.isna().sum()), 'need_balance': bool(ns['need_balance'](ctx))})
    ns['after_market_close'](None); cases.append({'case': 'balance_after_close_returns_immediately', 'result': None})
    return {'not_a_backtest': True, 'cases': cases, 'limits': ['Original requests/orders on synthetic inputs only; no fills, fees, NAV or IRR loops',
        'Complete trend windows avoid ffill; original missing-data direction and compatibility faults are diagnosed separately',
        'Alpha022 reference code is not executed as missing platform core']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch38 original trend/volume/winsor-volatility arithmetic and source diagnostics frozen')


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
                    while result.next(): values.append(result.get_row_data()); require(len(values) <= 10000, 'Provider row cap exceeded')
                    frame = pd.DataFrame(values, columns=result.fields)
            else: frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch38_dependency_probe', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch38', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch38 dated HS300/LOF/bond price supplement attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); accepted = component_file(directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch38 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(accepted), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch38 forbidden-network operator/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch38.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch38_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch37_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); component_file(directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
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
    accepted = component_file(directory); checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(accepted) == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(accepted) and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch38 trend/winsor-wave arithmetic accepted; original pool/Alpha022/fund dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch38-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'clarify', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch38/20261006-trend-alpha-balanced')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
