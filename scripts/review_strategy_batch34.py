"""Freeze ETF momentum/RSRS/bias rules and independently check longest verified components."""
import argparse
import ast
from contextlib import redirect_stdout
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
import warnings

import numpy as np
import pandas as pd
import requests

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch33 as previous
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch31 import reference_ols, reference_z
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch33/20261006-bank-etf-rotation')
RECEIPT = Path('docs/handoff/2026-10-06-batch33-verification.json')
SOURCES = ('2023年度精选策略/4.8年10倍，回撤小，有滑点！ETF动量简单轮动策略！.txt',
    '2023年度精选策略/16.8年13倍的ETF动量轮动策略，有滑点，无未来函数，回撤小！.txt',
    '2024年度精选策略2/66.ETF动量轮动MA乖离择时.txt',
    '2023年度精选策略/15.动量ETF轮动RSRS择时-升级.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/ols.py'), ('repo/backtrader', 'backtrader/indicators/sma.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch34.py', 'tests/unit/test_batch34_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'index-input.parquet', 'calendar-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'index': {'provider': 'baostock', 'function': 'query_history_k_data_plus', 'parameters': {'code': 'sz.399006',
        'fields': 'date,code,open,high,low,close,volume,amount', 'start_date': '2005-01-01', 'end_date': '2026-09-29', 'frequency': 'd', 'adjustflag': '3'},
        'limit': 'One benchmark sample does not supply fund pool/events or original intraday execution'},
    'fund500': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh510500'},
        'limit': 'One fund price sample does not prove original adjustment/events/states/rules or11:30 execution'},
    'fund50': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz159949'},
        'limit': 'One fund price sample does not prove events/states/rules or14:50 execution'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch33 differs')
    require(file_sha(ACCEPTED / 'input-binding.json') == receipt['checks']['evidence_sha256']['input-binding.json'], 'Accepted batch33 binding changed')
    upstream = previous.binding(root, ACCEPTED)
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted index/calendar bytes differ')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': upstream, 'files': rows, 'not_a_backtest': True}
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


def start(root, directory, checkpointed=False):
    if checkpointed:
        require(Store(root).published() == read(directory / 'baseline.json')['published'] and
            read(directory / 'progress-start.json')['sources'] == 695, 'Startup baseline differs')
        require(not (directory / 'input-binding.json').exists() and not (directory / 'existing-apis.json').exists() and
            not (directory / 'source-reviews').exists(), 'Cannot resume completed start')
        correction = read(directory / 'startup-correction.json')
        require(correction['returncode'] == 1 and file_sha(correction['original_code_file']) == correction['original_code_sha256'], 'Startup evidence changed')
        require(file_sha(directory / 'implementation-evidence.before.json') ==
            file_sha(Path(root) / 'catalog/strategies/implementation-evidence.json'), 'Startup implementation registry changed')
    else: checkpoint(root, directory, 'Batch34 ETF momentum/RSRS/bias source audit started')
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
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
            'use': 'Read-only SMA formula and finance provider comparison; not a platform replacement'})
    search = ['rg', '--files', 'repo', '-g', '*rsrs*', '-g', '*momentum*']
    result = subprocess.run(search, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search': {'command': search, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqlib', 'jqfactor')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch34 four full source reviews frozen; distinct pools/options and duplicate source retained')


def resume(root, directory): return start(root, directory, checkpointed=True)


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path)
        profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns),
            'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['HS300 OHLC supports original benchmark timing; close alone is only a fund-score operator sample',
            'No stock/fund universe substitution; no trading results, NAV or fees calculated']})
    return archive(root, 'Batch34 accepted HS300/calendar bytes frozen for explicitly limited operator research')


def inputs(directory):
    doc = read(directory / 'input-analysis.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and doc['platform_equivalent'] is False and
        doc['component_instrument'] == '000300.SH' and set(doc['profiles']) == {'index-input.parquet', 'calendar-input.parquet'}, 'Input scope differs')
    frames = {}
    for name, row in doc['profiles'].items():
        path = directory / name; require(row['file'] == str(path) and file_sha(path) == row['sha256'], 'Input binding changed')
        frame = pd.read_parquet(path); keys = ['date', 'index'] if name.startswith('index') else ['date']
        require(len(frame) == row['rows'] and list(frame.columns) == row['columns'] and frame.date.min() == row['first'] and
            frame.date.max() == row['last'] and not frame[keys].isna().any().any() and not frame.duplicated(keys).any(), 'Input profile differs')
        frames[name] = frame
    frame = frames['index-input.parquet'].sort_values('date').reset_index(drop=True)
    calendar = frames['calendar-input.parquet']; sessions = set(calendar[calendar.is_open].date)
    require(set(frame['index']) == {'000300.SH'} and set(frame.date) <= sessions and
        np.isfinite(frame[['close', 'volume']]).all().all() and frame.close.gt(0).all() and frame.volume.ge(0).all(), 'Invalid component input')
    return frame, sorted(sessions)


def tree(directory, number): return previous.tree(directory, number)


def configuration(directory, number):
    calls = []; g = SimpleNamespace()
    ns = {'g': g, 'set_benchmark': lambda value: calls.append(['benchmark', value]),
        'set_option': lambda *args: calls.append(['option', *args]), 'set_slippage': lambda value: calls.append(['slippage', value]),
        'FixedSlippage': lambda value: value, 'OrderCost': lambda **kw: kw,
        'set_order_cost': lambda value, **kw: calls.append(['cost', value, kw]),
        'log': SimpleNamespace(set_level=lambda *a: None), 'run_daily': lambda fn, **kw: calls.append(['schedule', fn, kw]),
        'initial_slope_series': lambda: list(range(600)), 'my_trade': 'my_trade', 'check_lose': 'check_lose', 'print_trade_info': 'print_trade_info'}
    selected(directory, number, ('initialize',), ns); ns['initialize'](None)
    return g, calls


def kernels(directory):
    hashes = {}
    for name in ('get_rank', 'get_ols', 'initial_slope_series', 'get_zscore', 'get_timing_signal', 'my_trade', 'adjust_position', 'check_lose'):
        nodes = [next(n for n in tree(directory, k).body if isinstance(n, ast.FunctionDef) and n.name == name) for k in (0, 1)]
        dumps = [ast.dump(n) for n in nodes]; require(dumps[0] == dumps[1], 'Shared daily RSRS function differs')
        hashes[name] = hashlib.sha256(dumps[0].encode()).hexdigest()
    doc = read(directory / 'source-reviews/review.json')
    require(doc['sources'][1]['source_sha256'] == doc['sources'][3]['source_sha256'] and
        ast.dump(tree(directory, 1)) == ast.dump(tree(directory, 3)), '15/16 duplicate differs')
    fn = next(n for n in tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name == 'get_rank')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    require(isinstance(loop.body[0], ast.Assign) and isinstance(loop.body[-1], ast.Expr), 'Momentum loop differs')
    code = compile(ast.Module(body=loop.body[1:-1], type_ignores=[]), '<original-momentum-expression>', 'exec')
    return code, hashes


def momentum_case(code, values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (29,) and np.isfinite(values).all() and (values > 0).all(), 'Invalid momentum window')
    ns = {'np': np, 'math': math, 'data': pd.DataFrame({'close': values})}; exec(code, ns)
    return [float(ns[n]) for n in ('slope', 'intercept', 'r_squared', 'annualized_returns', 'score')]


def component_namespace(directory, number, frame):
    g, _ = configuration(directory, number); holder = {'end': len(frame)-1}; scores = []
    def history(stock, count, unit, fields):
        require(stock == '000300.XSHG' and unit == '1d' and count in (18, 20, 23, 29, 200, 618), 'Original component request differs')
        require(holder['end'] >= count-1, 'Insufficient component window')
        result = frame.iloc[holder['end']-count+1:holder['end']+1][fields].reset_index(drop=True).copy()
        if number == 2 and count == 200: result.index = range(-200, 0)
        return result
    ns = {'g': g, 'np': np, 'math': math, 'attribute_history': history}
    names = ('get_rank', 'get_zscore', 'get_timing_signal') if number == 2 else ('get_ols', 'get_zscore', 'initial_slope_series', 'get_timing_signal')
    sha = selected(directory, number, names, ns)
    original = ns['get_zscore']
    def zscore(values):
        value = original(values); scores.append(float(value)); return value
    ns['get_zscore'] = zscore
    return ns, holder, scores, sha


def reference_bias(values):
    x = list(map(float, values)); require(len(x) == 200 and np.isfinite(x).all() and min(x) > 0, 'Invalid bias window')
    ratio = [x[k]/(math.fsum(x[k-59:k+1])/60) for k in range(59, 200)]
    means = [math.fsum(ratio[k-9:k+1])/10 for k in range(9, len(ratio))]
    return reference_z(means[-30:])


def compute(directory):
    validate_sources(directory); frame, sessions = inputs(directory); code, hashes = kernels(directory)
    dates = [day for day in sessions if frame.date.min() <= day <= frame.date.max()]
    require(frame.date.tolist() == dates, 'Benchmark has missing verified session')
    ns, holder, zscores, sha = component_namespace(directory, 0, frame); holder['end'] = 617
    ns['g'].slope_series = ns['initial_slope_series']()[:-1]
    initial = frame.iloc[:618]
    reference = [reference_ols(initial.low.iloc[k:k+18], initial.high.iloc[k:k+18])[1] for k in range(600)][:-1]
    require(len(reference) == len(ns['g'].slope_series) == 599, 'Original seed differs')
    seed_error = max(abs(float(a)-b) for a,b in zip(ns['g'].slope_series, reference, strict=True))
    max_rsrs = 0.; boundaries = []; counts = {}; digest = hashlib.sha256()
    with redirect_stdout(io.StringIO()):
        for k in range(617, len(frame)):
            holder['end'] = k; signal = ns['get_timing_signal']('000300.XSHG'); window = frame.iloc[k-17:k+1]
            _, slope, r2 = reference_ols(window.low, window.high); reference.append(slope)
            expected_score = reference_z(reference[-600:])*r2
            actual_ols = ns['get_ols'](window.low, window.high); actual_score = zscores[-1]*float(actual_ols[2])
            today = math.fsum(map(float, frame.close.iloc[k-19:k+1]))/20
            before = math.fsum(map(float, frame.close.iloc[k-22:k-2]))/20
            expected = 'BUY' if expected_score > .7 and today > before else 'SELL' if expected_score < -.7 and today < before else 'KEEP'
            max_rsrs = max(max_rsrs, abs(actual_score-expected_score))
            if signal != expected: boundaries.append({'end_date': frame.date.iloc[k], 'actual': signal, 'expected': expected})
            counts[signal] = counts.get(signal, 0)+1
            digest.update(json.dumps({'date': frame.date.iloc[k], 'score': actual_score, 'signal': signal}, sort_keys=True).encode())
    bias_ns, bias_holder, bias_scores, bias_sha = component_namespace(directory, 2, frame)
    bias_error = 0.; bias_boundaries = []; bias_counts = {}; bdigest = hashlib.sha256()
    for k in range(199, len(frame)):
        bias_holder['end'] = k; signal = bias_ns['get_timing_signal']('000300.XSHG'); score = bias_scores[-1]
        expected_score = reference_bias(frame.close.iloc[k-199:k+1]); expected = 'BUY' if expected_score > 4 else 'SELL' if expected_score < -4 else 'KEEP'
        bias_error = max(bias_error, abs(score-expected_score))
        if signal != expected: bias_boundaries.append({'end_date': frame.date.iloc[k], 'actual': signal, 'expected': expected})
        bias_counts[signal] = bias_counts.get(signal, 0)+1
        bdigest.update(json.dumps({'date': frame.date.iloc[k], 'score': score, 'signal': signal}, sort_keys=True).encode())
    momentum_error = 0.; mcount = 0; mdigest = hashlib.sha256(); normalized_error = 0.
    for k in range(28, len(frame)):
        values = frame.close.iloc[k-28:k+1].to_numpy(dtype=float); actual = momentum_case(code, values)
        intercept, slope, r2 = reference_ols(range(29), np.log(values)); score = math.expm1(slope*250)*r2
        momentum_error = max(momentum_error, abs(actual[-1]-score)); mcount += 1
        mdigest.update(json.dumps({'date': frame.date.iloc[k], 'score': actual[-1]}, sort_keys=True).encode())
    bias_ns['g'].stock_pool = ['000300.XSHG']
    with redirect_stdout(io.StringIO()):
        for k in range(19, len(frame)):
            bias_holder['end'] = k; row = bias_ns['get_rank'](['ignored'])
            x = frame.close.iloc[k-19:k+1].to_numpy(dtype=float); slope = reference_ols(range(20), x/x[0])[1]
            normalized_error = max(normalized_error, abs(float(row[1])-slope))
    require(max(seed_error, max_rsrs, bias_error, momentum_error, normalized_error) < 1e-9, 'Independent component mismatch')
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'original_strategy_complete': False, 'strategy_results': [],
        'input_sha256': {n: file_sha(directory/n) for n in ('index-input.parquet', 'calendar-input.parquet')},
        'backend': {'numpy': np.__version__, 'pandas': pd.__version__}, 'shared_ast_sha256': hashes,
        'rsrs': {'source_ast_sha256': sha, 'windows': len(frame)-617, 'seed_slopes': 599, 'seed_last_end_index': 615,
            'first_append_end_index': 617, 'omitted_seed_end_index': 616, 'max_seed_difference': seed_error,
            'max_score_difference': max_rsrs, 'boundaries': boundaries, 'signals': counts, 'sha256': digest.hexdigest()},
        'bias': {'source_ast_sha256': bias_sha, 'windows': len(frame)-199, 'max_score_difference': bias_error,
            'boundaries': bias_boundaries, 'signals': bias_counts, 'sha256': bdigest.hexdigest()},
        'momentum29': {'windows': mcount, 'max_score_difference': momentum_error, 'sha256': mdigest.hexdigest()},
        'normalized20': {'windows': len(frame)-19, 'max_slope_difference': normalized_error},
        'limits': ['HS300 is original timing reference but not original fund ranking/trading pool',
            'No ETF orders/fills/NAV/costs; original seed gap retained; all completed-end samples, not intraday execution',
            'Legacy negative integer labels explicitly adapt bias zscore; original dated-Series failure retained']}


def diagnostics(directory):
    code, hashes = kernels(directory); cases = []; log = SimpleNamespace(info=lambda *a: None, debug=lambda *a: None)
    for k in (0, 1, 2, 3):
        g, calls = configuration(directory, k)
        cases.append({'case': f'configuration_{k}', 'pool': g.stock_pool, 'calls': calls})
    cases.append({'case': 'duplicate15_16', 'byte_equal': True, 'shared_functions': hashes})
    ns = {'g': SimpleNamespace(stock_pool=['actual'], momentum_day=29, stock_num=1), 'np': np,
        'attribute_history': lambda *a: pd.DataFrame({'close': np.linspace(1., 2., 29)})}
    selected(directory, 0, ('get_rank',), ns)
    try: ns['get_rank'](['argument'])
    except NameError as exc: cases.append({'case': 'missing_math_injection', 'error': str(exc)})
    else: raise AssertionError('Original undocumented math injection supplied')
    ns['math'] = math
    with redirect_stdout(io.StringIO()): rank = ns['get_rank'](['argument'])
    cases.append({'case': 'rank_ignores_argument', 'result': rank})
    for value, expected in [(.7, 'KEEP'), (-.7, 'KEEP'), (.700001, 'BUY'), (-.700001, 'SELL')]:
        g = SimpleNamespace(ref_stock='000300.XSHG', mean_day=20, mean_diff_day=3, N=18, M=600, slope_series=[], score_threshold=.7)
        rising = value >= 0
        def history(stock, count, unit, fields):
            return pd.DataFrame({'close': np.arange(23.)+1 if rising else np.arange(23.,0.,-1)}) if count == 23 else None
        local = {'g': g, 'attribute_history': history, 'get_ols': lambda *a: (0., 1., 1.), 'get_zscore': lambda *a: value}
        local['attribute_history'] = lambda stock, count, unit, fields: history(stock,count,unit,fields) if count == 23 else SimpleNamespace(low=None,high=None)
        selected(directory, 0, ('get_timing_signal',), local)
        with redirect_stdout(io.StringIO()): signal = local['get_timing_signal'](None)
        require(signal == expected, 'Timing boundary differs'); cases.append({'case': f'rsrs_threshold_{value}', 'signal': signal})
    trade = {'g': SimpleNamespace(stock_pool=['candidate']), 'log': log, 'get_rank': lambda *a: ['candidate'],
        'get_timing_signal': lambda *a: 'KEEP', 'adjust_position': lambda *a: None}
    trade['g'].ref_stock = '000300.XSHG'; calls = []
    for name in ('filter_st_stock', 'filter_limitup_stock', 'filter_limitdown_stock', 'filter_paused_stock'):
        trade[name] = lambda *a: a[-1]
    trade['adjust_position'] = lambda context, rows: calls.append(rows)
    selected(directory, 0, ('my_trade',), trade)
    with redirect_stdout(io.StringIO()): trade['my_trade'](SimpleNamespace(portfolio=SimpleNamespace(positions={})))
    cases.append({'case': 'keep_actually_rebalances', 'targets': calls})
    for score in (4., -4., 4.000001, -4.000001):
        frame = pd.DataFrame({'close': np.linspace(1.,2.,200)}); local = {'g': SimpleNamespace(ref_stock='000300.XSHG'),
            'attribute_history': lambda *a: frame.copy(), 'get_zscore': lambda *a: score}
        selected(directory, 2, ('get_timing_signal',), local)
        cases.append({'case': f'bias_threshold_{score}', 'signal': local['get_timing_signal'](None)})
    local = {'np': np}; selected(directory, 2, ('get_zscore',), local)
    try: local['get_zscore'](pd.Series(np.arange(30.), index=pd.date_range('2020-01-01', periods=30)))
    except KeyError as exc: cases.append({'case': 'dated_negative_index_error', 'error': str(exc)})
    else: raise AssertionError('Original legacy zscore index failure disappeared')
    orders = []; local = {'g': SimpleNamespace(stock_num=1), 'log': log,
        'open_position': lambda *a: orders.append(list(a)) or False, 'close_position': lambda *a: False}
    selected(directory, 2, ('adjust_position',), local)
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={},cash=10000.)); local['adjust_position'](ctx, ['fund', .001])
    cases.append({'case': 'bias_score_enters_order_loop', 'orders': orders})
    local = {'g': SimpleNamespace(stock_num=1), 'log': log, 'close_position': lambda *a: False, 'open_position': lambda *a: False}
    selected(directory, 0, ('adjust_position',), local)
    try: local['adjust_position'](ctx, ['fund'])
    except KeyError as exc: cases.append({'case': 'missing_platform_zero_position', 'error': str(exc)})
    else: raise AssertionError('Original missing zero-position mapping not exposed')
    for number, ratio in [(0,.2),(0,.20001),(2,.9),(2,.89999)]:
        pos = SimpleNamespace(security='fund',avg_cost=100.,price=100.*ratio,value=1000.,total_amount=100)
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'fund':pos})); orders=[]
        local = {'log':log,'order_target_value':lambda *a: orders.append(list(a))}; selected(directory, number, ('check_lose',), local)
        with redirect_stdout(io.StringIO()): local['check_lose'](ctx)
        cases.append({'case': f'stop_{number}_{ratio}', 'orders':orders})
    local = {'OrderStatus': SimpleNamespace(held='held'), 'order_target_value_':lambda *a: SimpleNamespace(filled=50,amount=100,status='held')}
    selected(directory, 0, ('open_position','close_position'), local)
    cases.append({'case': 'partial_fill_open_true_close_false', 'open':local['open_position']('fund',1000.),
        'close':local['close_position'](SimpleNamespace(security='fund'))})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); constant = momentum_case(code, np.ones(29))
    cases.append({'case':'constant_momentum_not_repaired','finite_score':math.isfinite(constant[-1])})
    return {'not_a_backtest': True, 'cases': cases,
        'limits': ['Original fragments and synthetic order intentions only; no trading engine or fees/NAV',
            'Wrong mixed-type target retained for diagnosis; corrected variant requires explicit confirmation']}

def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch34 long momentum/RSRS/bias components and original source defects/state diagnostics frozen')


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
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch34_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}],
            limit=query['limit'], strict_usable=False)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch34', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch34 original benchmark/ETF supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch34 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch34 forbidden-network operator/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch34.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch34_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch33_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
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
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch34 momentum/RSRS/bias components accepted; missing original trading dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch34-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'resume', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch34/20261006-momentum-rsrs-bias')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
