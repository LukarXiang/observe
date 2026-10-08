"""Freeze framework/RSRS/MESA/northbound components without a second trading ledger."""
import argparse
import ast
import cmath
import datetime
import hashlib
import importlib.util
import inspect
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
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch27 as prior
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = prior.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch27/20261006-value-ma-turtle')
RECEIPT = Path('docs/handoff/2026-10-06-batch27-verification.json')
SOURCES = ('2021年度精选策略/44.借助JqData搭建简易回测框架.txt',
    '2021年度精选策略/59.【复现】RSRS择时改进.txt',
    '2021年度精选策略/83.识别趋势震荡之神器 MESA最大熵谱分析（一）：滤波器建立.txt',
    '2023年度精选策略/74.勾股神器2代码40行，年化一百点.txt')
REFERENCES = (('repo/hikyuu', 'hikyuu_cpp/hikyuu/indicator/imp/IRSRSBeta.cpp'),
    ('repo/akshare', 'akshare/stock_feature/stock_hsgt_em.py'), ('repo/akshare', 'akshare/index/index_zh_em.py'))
FILES = tuple(sorted(set(prior.FILES) | {'scripts/review_strategy_batch28.py', 'tests/unit/test_batch28_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'input-analysis.json',
    'price-input.parquet', 'index-input.parquet', 'component-research.json', 'diagnostics.json',
    'existing-apis.json', 'probe-results.json', 'component-offline.json', 'offline-verification.json'}
QUERIES = {'northbound': {'function': 'stock_hsgt_individual_detail_em',
    'parameters': {'symbol': '600519', 'start_date': '20200601', 'end_date': '20200702'},
    'limit': 'Institution-level A-share percentage is not proven equivalent to aggregate platform share_ratio'},
    'minute_index': {'function': 'index_zh_a_hist_min_em', 'parameters': {'symbol': '000300', 'period': '1',
        'start_date': '2020-06-01 09:30:00', 'end_date': '2020-07-02 15:00:00'},
        'limit': 'One-minute endpoint only requests latest five days; historical date arguments are local filters'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    upstream = read(RECEIPT); old = read(ACCEPTED / 'baseline.json'); store = Store(root); state = store.state(SNAPSHOT)
    require(upstream['status'] == 'ok' and upstream['snapshot'] == SNAPSHOT and
        file_sha(ACCEPTED / 'baseline.json') == upstream['checks']['evidence_sha256']['baseline.json'], 'Accepted snapshot receipt differs')
    entry = state['tables']['index_1d']['all']; path = store.root / entry['file']; frame = pd.read_parquet(path)
    require(state['tables']['index_1d'] == old['published']['tables']['index_1d'] and
        fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'], 'Accepted index reference/content differs')
    price = ACCEPTED / 'price-input.parquet'
    require(file_sha(price) == upstream['checks']['evidence_sha256']['price-input.parquet'], 'Accepted prices changed')
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'prices': {'file': str(price), 'sha256': file_sha(price)}, 'stock_binding': prior.binding(root),
        'index': {'file': str(path), 'sha256': file_sha(path), 'entry': entry}, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Batch28 input binding changed')
    return result


def api_evidence():
    import akshare as ak
    from akshare.index.index_zh_em import index_code_id_map_em
    rows = []
    for fn in (ak.stock_hsgt_individual_detail_em, ak.index_zh_a_hist_min_em, index_code_id_map_em):
        code = inspect.getsource(fn)
        rows.append({'function': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'version': ak.__version__, 'apis': rows, 'queries': QUERIES,
        'missing_modules': [n for n in ('statsmodels', 'arch', 'jqdatasdk', 'tushare') if importlib.util.find_spec(n) is None]}


def start(root, directory):
    checkpoint(root, directory, 'Batch28 framework/RSRS/MESA/northbound source research started')
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old_path = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        p = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(p) == old[name]['bytes_sha256'], 'Source reviewed/changed')
        shutil.copyfile(p, copied); text, encoding = read_source(p)
        rows.append({'source_path': str(p), 'source_sha256': file_sha(p), 'source_copy': str(copied),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); references = []
    for repo, name in REFERENCES:
        p = Path(repo) / name; copied = folder / p.name; shutil.copyfile(p, copied)
        references.append({'file': str(p), 'copy': str(copied), 'sha256': file_sha(p),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'RSRS slope reference only, not WLS replacement; provider ratio schema and recent-only minute restriction'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': references, 'catalog': catalog,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path), 'snapshot': SNAPSHOT,
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch28 four complete reviews frozen; original trading dependencies retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'References missing')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def methods(directory, number, class_name, names, namespace):
    classes = [n for n in tree(directory, number).body if isinstance(n, ast.ClassDef) and n.name == class_name]
    require(len(classes) == 1, 'Missing/ambiguous source class')
    nodes = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names), 'Missing/ambiguous source methods')
    require(all(all(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in n.decorator_list) for n in nodes), 'Unsupported source decorator')
    # Exclude class assignments such as Context() and all module authentication/import side effects.
    cls = ast.ClassDef(name=class_name, bases=[], keywords=[], body=nodes, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, '<frozen-original-methods>', 'exec'), namespace)
    return namespace[class_name], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def expr(directory, number, function, target, class_name=None):
    scope = tree(directory, number)
    if class_name is not None: scope = next(n for n in scope.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == function)
    nodes = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
    require(len(nodes) == 1, 'Missing/ambiguous source expression')
    return compile(ast.Expression(nodes[0]), '<frozen-original-expression>', 'eval'), hashlib.sha256(ast.dump(nodes[0]).encode()).hexdigest()


def prepare(root, directory):
    bound = binding(root, directory)
    require(not any((directory / n).exists() for n in ('price-input.parquet', 'index-input.parquet', 'input-analysis.json')), 'Prepared archive exists')
    frame, ret_sha = index_sample(directory, pd.read_parquet(bound['index']['file']))
    shutil.copyfile(bound['prices']['file'], directory / 'price-input.parquet')
    frame.to_parquet(directory / 'index-input.parquet', index=False)
    rows = {}
    for name in ('price', 'index'):
        p = directory / f'{name}-input.parquet'; frame = pd.read_parquet(p)
        rows[name] = {'file': str(p), 'sha256': file_sha(p), 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'inputs': rows, 'pool': list(prior.INSTRUMENTS),
        'ret_source_ast_sha256': ret_sha, 'ret_policy': 'Original close/pre_close-1; pre_close mapped from vendor preclose, not adjacent close',
        'not_a_backtest': True, 'limits': ['Ten stocks do not replace original fixed HS300 pool',
            'No northbound ratios or original minute data; back-adjusted verified trading rows only for RSI arithmetic']})
    return archive(root, 'Batch28 accepted ten-stock and HS300 daily component inputs frozen')


def index_sample(directory, source):
    require({'date', 'index', 'close', 'preclose', 'high', 'low', 'volume'} <= set(source), 'Index schema missing')
    frame = source.rename(columns={'index': 'instrument', 'preclose': 'pre_close'}).sort_values('date').reset_index(drop=True)
    require(set(frame.instrument) == {'000300.SH'} and not frame.date.duplicated().any(), 'Index scope/duplicates differ')
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    require(np.isfinite(frame[['close', 'pre_close', 'high', 'low', 'volume']]).all().all() and
        frame[['close', 'pre_close', 'high', 'low', 'volume']].gt(0).all().all(), 'Incomplete index input')
    code, sha = return_expression(directory)
    frame['ret'] = eval(code, {}, {'price_df': frame})
    return frame, sha


def return_expression(directory):
    cls = next(n for n in tree(directory, 1).body if isinstance(n, ast.ClassDef) and n.name == 'RSRS')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'query_data')
    nodes = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and
        isinstance(t.value, ast.Name) and t.value.id == 'price_df' and isinstance(t.slice, ast.Constant) and t.slice.value == 'ret' for t in n.targets)]
    require(len(nodes) == 1, 'Missing/ambiguous source return expression')
    return compile(ast.Expression(nodes[0]), '<original-return>', 'eval'), hashlib.sha256(ast.dump(nodes[0]).encode()).hexdigest()


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); frames = {}
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and doc['pool'] == list(prior.INSTRUMENTS), 'Input scope differs')
    require(set(doc['inputs']) == {'price', 'index'}, 'Input set differs')
    for name, row in doc['inputs'].items():
        p = directory / f'{name}-input.parquet'
        require(str(p) == row['file'] and file_sha(p) == row['sha256'], 'Component input changed')
        frame = pd.read_parquet(p)
        require(len(frame) == row['rows'] and frame.date.min() == row['first'] and frame.date.max() == row['last'] and
            not frame.duplicated(['date', 'instrument']).any(), 'Input profile differs'); frames[name] = frame
    return frames


def wilder(values, period=14):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) > period and np.isfinite(x).all(), 'Invalid RSI window')
    changes = np.diff(x); gain = math.fsum(max(float(d), 0.) for d in changes[:period]) / period
    loss = math.fsum(max(-float(d), 0.) for d in changes[:period]) / period
    for d in changes[period:]:
        gain = (gain * (period - 1) + max(float(d), 0.)) / period
        loss = (loss * (period - 1) + max(-float(d), 0.)) / period
    return 100 * gain / (gain + loss) if gain + loss > 1e-14 else 0.


def compute(directory):
    validate_sources(directory); data = inputs(directory); index = data['index'].set_index('date'); ns = {'pd': pd, 'np': np}
    cls, methods_sha = methods(directory, 1, 'RSRS', ('_get_vol_weights', '_mark_flag'), ns)
    vol_expr, vol_sha = expr(directory, 1, '_cal_ret_quantile', 'ret_std', 'RSRS')
    rank_expr, rank_sha = expr(directory, 1, '_cal_ret_quantile', 'ret_quantile', 'RSRS')
    rank_node = next(n for n in ast.walk(tree(directory, 1)) if isinstance(n, ast.FunctionDef) and n.name == '_cal_ret_quantile')
    rank_lambdas = [n for n in ast.walk(rank_node) if isinstance(n, ast.Lambda)]
    require(len(rank_lambdas) == 1, 'Rank lambda differs')
    rank_fn = eval(compile(ast.Expression(rank_lambdas[0]), '<original-rank>', 'eval'))
    # Actual global windows for the original three-signal HS300 case are 18/700.
    holder = SimpleNamespace(price_df=index, N=18, M=700); std = eval(vol_expr, {'np': np}, {'self': holder})
    returns = index.ret.to_numpy(); volume = index.volume.to_numpy(); weight_max = std_max = rank_max = 0.; weights = []
    for k in range(len(index)):
        window = returns[max(0, k - 17):k + 1]; finite = window[np.isfinite(window)]
        if len(finite):
            mean = math.fsum(map(float, finite)) / len(finite)
            expected = math.sqrt(math.fsum((float(v) - mean) ** 2 for v in finite) / len(finite))
            std_max = max(std_max, abs(float(std.iloc[k]) - expected))
        if k >= 17:
            v = volume[k - 17:k + 1]; actual = np.array(cls._get_vol_weights(pd.Series(v)))
            expected = v / math.fsum(map(float, v)); weight_max = max(weight_max, float(np.max(np.abs(actual - expected))))
            weights.append(actual)
    ranks = []
    for k in range(699, len(index)):
        values = std.iloc[k - 699:k + 1].to_numpy(); require(np.isfinite(values).all(), 'Incomplete rank window')
        # Preserve the original lambda; label -1 explicitly denotes the last operand for arithmetic only.
        actual = float(rank_fn(pd.Series(values, index=[*range(699), -1])))
        expected = (np.count_nonzero(values < values[-1]) + (np.count_nonzero(values == values[-1]) + 1) / 2) / 700
        rank_max = max(rank_max, abs(actual - expected)); ranks.append(actual)
    require(max(weight_max, std_max, rank_max) < 1e-10, 'RSRS numeric component differs')
    rsi_expr, rsi_sha = expr(directory, 3, 'bx_strategy', 'rsi'); groups = []; all_rsi = []; paused = 0
    for instrument, frame in data['price'].groupby('instrument', sort=True):
        frame = frame.sort_values('date'); paused += int((~frame.is_trading).sum()); trading = frame[frame.is_trading]
        require(trading.adjustment_status.eq('usable').all() and np.isfinite(trading[['close_adj', 'back_factor']]).all().all() and
            trading.back_factor.gt(0).all(), 'Unknown trading adjustment cannot be skipped')
        x = trading.close_adj.to_numpy(); factors = trading.back_factor.to_numpy(); values = []; max_diff = 0.; decision_diff = 0
        for k in range(99, len(x)):
            a = x[k - 99:k + 1] / factors[k]
            actual = float(eval(rsi_expr, {'talib': talib, 'numpy': np}, {'a': a})); expected = wilder(a)
            max_diff = max(max_diff, abs(actual - expected)); decision_diff += ((40 <= actual <= 80) != (40 <= expected <= 80)); values.append(actual)
        require(max_diff < 1e-9 and decision_diff == 0, 'RSI numeric/boundary component differs')
        groups.append({'instrument': instrument, 'windows': len(values), 'max_abs_difference': max_diff, 'range_gate_differences': decision_diff,
            'sha256': hashlib.sha256(np.array(values, dtype='<f8').tobytes()).hexdigest()}); all_rsi.extend(values)
    spectrum_expr, spectrum_sha = expr(directory, 2, 'Filter', 'spectrum')
    alpha1, alpha2 = .05, -.1; sigma = np.std([alpha1, alpha2]); grid = np.linspace(0, .5, 1000)
    original = eval(spectrum_expr, {'np': np, 'pi': math.pi, 'sigma': sigma, 'alpha1': alpha1, 'alpha2': alpha2})
    actual = np.array([original(float(f)) for f in grid]); expected = np.array([2 * (abs(alpha1 - alpha2) / 2) /
        abs(1 - alpha1 * cmath.exp(-2j * math.pi * f) - alpha2 * cmath.exp(-4j * math.pi * f)) ** 2 for f in grid])
    spectrum_max = float(np.max(np.abs(actual - expected))); require(spectrum_max < 1e-12, 'Toy MESA spectrum differs')
    peak = int(np.argmax(actual))
    return {'not_a_backtest': True, 'original_strategy_complete': False, 'snapshot': SNAPSHOT,
        'input_sha256': {n: file_sha(directory / f'{n}-input.parquet') for n in ('price', 'index')},
        'source_ast_sha256': {'rsrs_methods': methods_sha, 'ret_std': vol_sha, 'ret_quantile': rank_sha, 'rsi': rsi_sha, 'spectrum': spectrum_sha},
        'rsrs': {'instrument': '000300.SH', 'N': 18, 'M': 700, 'rows': len(index), 'weight_windows': len(weights),
            'weight_max_abs_difference': weight_max, 'std_max_abs_difference': std_max, 'rank_windows': len(ranks), 'rank_max_abs_difference': rank_max,
            'weight_sha256': hashlib.sha256(np.asarray(weights, dtype='<f8').tobytes()).hexdigest(),
            'std_sha256': hashlib.sha256(std.to_numpy(dtype='<f8').tobytes()).hexdigest(),
            'rank_sha256': hashlib.sha256(np.asarray(ranks, dtype='<f8').tobytes()).hexdigest(),
            'scope': 'Original volume and ret std plus rank lambda on explicit last-label operands; original date-index chain still fails; no OLS/WLS/full signal'},
        'rsi': {'talib': talib.__version__, 'period': 14, 'history': 100, 'known_paused_rows_excluded': paused, 'groups': groups,
            'windows': len(all_rsi), 'scope': 'Verified trading-row arithmetic; not original pool/northbound strategy or platform equivalence'},
        'mesa': {'toy_coefficients': [alpha1, alpha2], 'bins': len(grid), 'max_abs_difference': spectrum_max,
            'peak_index': peak, 'source_peak_frequency': peak / 1000 * .5, 'actual_grid_peak_frequency': float(grid[peak]),
            'sha256': hashlib.sha256(actual.astype('<f8').tobytes()).hexdigest(), 'real_model_fitted': False}}


def northbound_case(directory, previous, current, holds=(), rsi=60.):
    orders = []; context = SimpleNamespace(bx_cash=dict(previous), pool=list(set(previous) | set(current)),
        portfolio=SimpleNamespace(positions=dict.fromkeys(holds), total_value=100000.))
    ns = {'bx_cashflow': lambda *a: dict(current), 'get_bars': lambda *a, **kw: {'close': np.arange(100.)},
        'talib': SimpleNamespace(RSI=lambda a: np.array([rsi])), 'numpy': np,
        'order_value': lambda security, value: orders.append(['buy', security, value]),
        'order_target': lambda security, value: orders.append(['sell', security, value])}
    sha = selected(directory, 3, ('bx_strategy',), ns); ns['bx_strategy'](context)
    return {'orders': orders, 'next_filtered_baseline': context.bx_cash, 'source_ast_sha256': sha, 'synthetic': True, 'not_a_backtest': True}


def diagnostics(directory):
    ns = {'np': np, 'pd': pd}; cls, sha = methods(directory, 1, 'RSRS',
        ('_get_pretreatment_day', '_get_vol_weights', '_mark_flag', '_cal_ret_quantile', '_check_threshold_key_name'), ns)
    obj = cls(); obj.freq = {'钝化RSRS': (18, 700), '右偏修正标准分RSRS': (18, 600), '成交额加权钝化RSRS': (19, 500)}
    obj._get_pretreatment_day(); actual_windows = [obj.N, obj.M]
    obj.freq = dict.fromkeys(('钝化RSRS', '右偏修正标准分RSRS', '成交额加权钝化RSRS'), (18, 700))
    obj._get_pretreatment_day(); grouping = obj.freq[(18, 700)]
    states = cls._mark_flag(pd.DataFrame({'signal': [.8, .9, 1.1, -.5, -.6, 1.2, np.nan]}), 'signal', (1., -.5))
    cases = [{'case': 'global_windows_override', 'windows': actual_windows}, {'case': 'third_group_nested', 'grouping': grouping},
        {'case': 'first_literal_and_nan_exit', 'flag': states[0], 'hold': states[1], 'buy': states[2], 'sell': states[3]}]
    obj.N = 2; obj.M = 2; obj.price_df = pd.DataFrame({'ret': [0., .1, -.2, .3]}, index=pd.date_range('2020-01-01', periods=4))
    try: obj._cal_ret_quantile()
    except KeyError as exc: cases.append({'case': 'date_rank_index', 'error': type(exc).__name__})
    else: raise AssertionError('Original date-index defect not reproduced')
    order, order_sha = methods(directory, 0, 'Order', ('buy', 'sell'), {})
    holder = SimpleNamespace(cash=1000., position={}, total_value=1000., current_dt='synthetic', trade_history=[])
    obj = order(); obj.context = holder; obj.buy('a', 10., -5)
    cases.append({'case': 'negative_buy_quantity', 'cash': holder.cash, 'count': holder.position['a']['count']})
    holder = SimpleNamespace(cash=1000., position={}, total_value=1000., current_dt='synthetic', trade_history=[])
    obj.context = holder; obj.buy('a', 10., 10); obj.buy('b', 10., 10)
    cases.append({'case': 'multiasset_buy_total', 'source_total': holder.total_value, 'synthetic_expected': 1000.})
    obj.sell('b', 10., 10)
    cases.append({'case': 'multiasset_sell_total', 'source_total': holder.total_value, 'synthetic_expected': 1000.})
    ns = {'datetime': datetime, 'pd': pd, 'np': np, 'math': math,
        'get_price': lambda *a, **kw: pd.DataFrame({'close': [10., 11., 12.]})}
    filter_sha = selected(directory, 2, ('Filter',), ns)
    try: ns['Filter'](datetime.datetime(2020, 7, 2))
    except AttributeError as exc:
        require('append' in str(exc), 'Unexpected Filter diagnostic'); cases.append({'case': 'mesa_append', 'error': type(exc).__name__})
    else: raise AssertionError('Original append defect not reproduced')
    for name, previous, current, holds, rsi in (
        ('northbound_first_baseline', {}, {'a': 4.}, (), 60.),
        ('northbound_below_filter_disappears', {'a': 4.}, {}, ('a',), 60.),
        ('northbound_reappears', {}, {'a': 4.5}, ('a',), 60.),
        ('northbound_stale_hold_cap', {'a': 4., 'b': 4.}, {'a': 4.5, 'b': 4.5}, tuple(f'h{k}' for k in range(9)), 60.),
        ('northbound_rsi40_inclusive', {'a': 4.}, {'a': 4.5}, (), 40.),
        ('northbound_rsi80_inclusive', {'a': 4.}, {'a': 4.5}, (), 80.),
        ('northbound_low_rsi_block', {'a': 4.}, {'a': 4.5}, (), 39.),
        ('northbound_binary_delta', {'a': 4.}, {'a': 4.3}, (), 60.)):
        cases.append({'case': name, **northbound_case(directory, previous, current, holds, rsi)})
    return {'synthetic_only': True, 'not_a_backtest': True, 'rsrs_ast_sha256': sha, 'order_ast_sha256': order_sha,
        'filter_ast_sha256': filter_sha, 'cases': cases}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study implementation changed')
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch28 real daily RSI/RSRS arithmetic and toy MESA spectrum verified; defects archived')


def worker(root, directory, endpoint):
    import akshare as ak
    api = read(directory / 'existing-apis.json'); require(api == api_evidence(), 'Probe API changed')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'),
        'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire']) < 10, 'Probe response limit reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status_code': response.status_code})
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, query['function'])(**query['parameters'])
        path = raw.save(root, 'batch28_dependency_probe', endpoint, directory.name, frame)
        row.update(status='success' if len(frame) else 'empty', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}])
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(read(directory / 'existing-apis.json') == api_evidence(), 'API evidence changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['published'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['status'] in ('failed', 'timeout', 'empty', 'success'), 'Probe result binding differs')
        for item in row['wire']: require(file_sha(item['file']) == item['sha256'], 'Probe wire changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw missing'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and
                bool(len(frame)) == (row['status'] == 'success'), 'Probe raw changed')
        else: require(not row['files'] and row.get('error'), 'Failed probe accepted data/no error')
        rows.append(row)
    return rows


def probe(root, directory):
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch28', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists():
                    save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query, 'status': 'timeout',
                        'files': [], 'wire': [], 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json'),
                        'error': 'Parent deadline45s; partial unaccepted wire files may remain'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch28 bounded northbound ratio/minute-index supplementation archived; no substitution/publication')


def offline(root, directory):
    before = implementation(); binding(root, directory)
    def denied(*a, **kw): raise AssertionError('Batch28 offline attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline implementation/diagnostics differ')
    save(directory / 'component-offline.json', result)
    require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline bytes differ')
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'sha256': file_sha(directory / 'component-offline.json'), 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch28 forbidden-network components and diagnostics match byte-for-byte')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch28.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch28_research.py',
            'tests/unit/test_batch27_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_batch23_research.py', 'tests/unit/test_batch21_research.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Frozen code exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']) and
        CORE <= checked['evidence_sha256'].keys(), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    off = read(directory / 'offline-verification.json')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and
        off['socket_network_disabled'] is True and off['sha256'] == file_sha(directory / 'component-research.json') ==
        file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed component differs')
    protection = protect(root, directory); progress = archive(root, 'Batch28 four source reviews/components/probes/offline accepted; no original trade reconstruction')
    save(Path('docs/handoff/2026-10-06-batch28-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'protection': protection,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'prepare', 'study', 'probe', 'worker', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch28/20261006-rsrs-mesa-northbound')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
