"""Freeze original RSRS/BIAS sources and verify available long reference history."""
import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import inspect
from io import StringIO
import json
import math
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
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts import review_strategy_batch78 as previous
from scripts import review_strategy_batch76 as batch76
from scripts import review_strategy_batch75 as batch75
from scripts import probe_strategy_batch74_fallback as native74
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch59 import index_operand
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch79/20261009-RSRS-bias-momentum')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略1/31.动量因子加RSRS择时和ETF轮动每日调仓.txt',
    '2024年度精选策略1/42.大盘ETF动量轮动RSRS择时策略-无纳指（5年半861%）.txt')
SOURCE_SHA = ('8ee830377a447239b901212ec77b4b0ca9331d9d59a406ad57f053da751cc38f',
    '03236e7dbf922054cd262bde0c1a496835e1f174dfc27101114c5d19a51f5174')
POOLS = (previous.POOL, ('510300.XSHG', '510050.XSHG', '159949.XSHE'))
REFERENCES = (('repo/backtrader', 'backtrader/indicators/ols.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'), ('repo/rqalpha', 'rqalpha/portfolio/account.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch79.py', 'tests/unit/test_batch79_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'reused-inputs.json', 'index-input.parquet', 'index-input-binding.json', 'dependency-inventory.json',
    'existing-apis.json', 'source-reviews/review.json', 'probe-results.json', 'candidate-audit.json',
    'diagnostics.json', 'source-extraction.json', 'index-components.parquet', 'momentum-components.parquet',
    'rank-components.parquet', 'component-research.json', 'offline-verification.json', 'offline-catalog.json',
    'development-failure-binding.json', 'development-slope-binding.json'}
CASE = 'fund-159949-60m'
QUERY = {'symbol': '159949', 'start_date': '2016-01-01 00:00:00',
    'end_date': '2026-09-29 23:59:59', 'period': '60', 'adjust': ''}
ENDPOINT = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
PARAMETERS = {'fields1': 'f1,f2,f3,f4,f5,f6', 'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    'klt': '60', 'fqt': '0', 'secid': '0.159949', 'beg': '0', 'end': '20500000'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch78-verification.json')
    review = Path('docs/handoff/2026-10-09-batch78-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted78 changed')
    for name in ('scripts/review_strategy_batch78.py', 'tests/unit/test_batch78_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted78 entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def reused_inputs():
    result = {s.removeprefix('fund-')+'.'+('XSHE' if s.startswith('fund-159') else 'XSHG'): r
        for s, r in previous.reused_inputs()['inputs'].items()}
    for folder, name, case, key, instrument in (
        (native74.DIRECTORY, 'results.json', 'fund-510050', 'sina-fallback/results.json', '510050.XSHG'),
        (batch76.DIRECTORY, 'probe-results.json', 'fund-159949', 'probe-results.json', '159949.XSHE'),
        (previous.DIRECTORY, 'probe-results.json', previous.CASE, 'probe-results.json', '513100.XSHG')):
        parent = folder.parent if folder == native74.DIRECTORY else folder
        require(file_sha(folder/name) == read(parent/'checked-state.json')['evidence_sha256'][key], 'Candidate registry changed')
        rows = [r for r in read(folder/name)['results'] if r['case'] == case]
        require(len(rows) == 1, 'Candidate scope changed'); result[instrument] = rows[0]
    old = read(batch75.DIRECTORY/'reused-ETF.json')
    require(file_sha(batch75.DIRECTORY/'reused-ETF.json') == read(batch75.DIRECTORY/'checked-state.json')['evidence_sha256']['reused-ETF.json'] and
        file_sha(old['receipt']) == old['receipt_sha256'] and file_sha(old['probe']) == old['probe_sha256'], '510300 reuse changed')
    result['510300.XSHG'] = {'status': 'success', **old['payload']}
    require(set(result) == set(POOLS[0]) | set(POOLS[1]), 'Fund scope changed')
    for row in result.values():
        require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'] and
            len(pd.read_parquet(row['file'])) == row['rows'], 'Fund payload changed')
    return {'inputs': result, 'strict_usable': False, 'new_daily_requests': 0, 'not_a_backtest': True}


def api_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_min_em; source = inspect.getsource(fn)
    helper = inspect.getsource(fn.__globals__['get_market_id'])
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'source': re.sub(r'"ut": "[0-9a-f]+"', '"ut": "<vendor-marker-redacted>"', source),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'helper': helper,
        'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(), 'query': QUERY,
        'policy': 'One60m GET;8/10s,parent45s,no proxy/retry/redirect; vendor marker omitted from request records',
        'limits': ['API uses beg0/end20500000 then client date slicing; query dates do not prove requested historical coverage',
            'jqlib WR implementation absent; synthetic WR only isolates branch order',
            'Native daily and minute candidates lack historical identity/events/adjustment/fees/platform execution proof']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    funds = [s[:6]+('.SH' if s.endswith('XSHG') else '.SZ') for s in (*POOLS[0], *POOLS[1])]
    return {'snapshot': SNAPSHOT, 'registered_fund_rows': {t: len(store.load_state(state, t, filters=[('instrument', 'in', funds)]))
        for t in ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')},
        'benchmark399006_rows': len(store.load_state(state, 'index_1d', filters=[('index', '=', '399006.SZ')])),
        'original_input_rows_admitted': 0, 'not_a_backtest': True,
        'gates': ['ETF history identity/events/adjustment/state/fees and9:45 or7:00/9:30/9:35 execution unproved',
            '42 WR21/14 platform history unavailable;60m22-row/11:27 last_price/closeable_amount incomplete',
            '42 original rank returns [instrument,score,change] to security-list execution; numeric security operands possible',
            '42 Series position compatibility and undefinedvalue stoploss retained; no approved corrected trading variant']}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    frame, operand = index_operand(root); reused = reused_inputs()
    checkpoint(root, directory, 'Batch79 RSRS/BIAS launch baseline; seven prior native daily inputs preserved')
    save(directory/'input-binding.json', bound); save(directory/'reused-inputs.json', reused)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    path = directory/'index-input.parquet'
    with path.open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory/'index-input-binding.json', {**operand, 'frozen_sha256': file_sha(path)})
    save(directory/'dependency-inventory.json', inventory(root)); save(directory/'existing-apis.json', api_evidence())
    return {'status': 'ok', 'daily_inputs_reused': 7, 'index_rows': len(frame)}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(); rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    (directory/'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch79 two complete RSRS/BIAS sources frozen; original seeds/KEEP/triple-list defects retained')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/SOURCES[number]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['review'] == REVIEWS[SOURCES[number]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    for binding_name in ('development-failure-binding.json', 'development-slope-binding.json'):
        for name, sha in read(directory/binding_name).items():
            require(file_sha(name) == sha, 'Development failure evidence changed')


def tree(directory, number):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(source)[0])


def selected(directory, number, ns):
    nodes = [n for n in tree(directory, number).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (19 if number else 9) and all(safe_function(n) for n in nodes), 'Unsafe/missing definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<batch79-frozen-definitions>', 'exec'), ns)
    return ns


def namespace(directory, number):
    ns = {'g': SimpleNamespace(), 'np': np, 'math': math,
        'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None),
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'set_slippage': lambda *a: None,
        'FixedSlippage': lambda v: v, 'OrderCost': lambda **kw: kw, 'set_order_cost': lambda *a, **kw: None,
        'run_daily': lambda *a, **kw: None, 'record': lambda **kw: None, 'send_message': lambda *a: None,
        'OrderStatus': SimpleNamespace(held='held')}
    selected(directory, number, ns)
    ns['initial_slope_series'] = (lambda: ([1.]*608, [0.]*8)) if number else (lambda: [1.]*600)
    if number: ns['initial_stock_motion'] = lambda pool: {s: [0.] for s in pool}
    ns['initialize'](None); return ns


def extracted(directory, kind, ns):
    number = int(kind != 'log'); source = tree(directory, number)
    if kind == 'log':
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'get_socre')
        body = original.body; args = original.args
    elif kind == 'bias':
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'get_rank')
        loop = next(n for n in original.body if isinstance(n, ast.For))
        body = [next(n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'score'),
            ast.Return(value=ast.Name(id='score', ctx=ast.Load()))]
        args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='bias')], kwonlyargs=[], kw_defaults=[], defaults=[])
    else:
        require(kind == 'prefix', 'Unknown extraction')
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'get_timing_signal')
        body = []
        for node in original.body:
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'WR' for n in ast.walk(node)): break
            body.append(node)
        require(len(body) == 10, 'WR prefix changed')
        body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load())
            for n in ('slope', 'r2', 'rsrs_score', 'rsrs_slope', 'idex_slope')], ctx=ast.Load())))
        args = original.args
    node = ast.FunctionDef(name='reference_'+kind, args=args, body=body, decorator_list=[], type_params=[])
    ast.fix_missing_locations(node); require(safe_function(node), 'Unsafe extraction')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch79-extracted-arithmetic>', 'exec'), ns)
    return ns[node.name], ast.dump(node, include_attributes=True)


def independent_z(values):
    mean = math.fsum(values)/len(values)
    std = math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
    require(std > 0, 'Constant slope variance'); return (values[-1]-mean)/std


def independent_slope(x, y):
    require(len(x) == len(y) and len(x) > 1, 'Invalid slope operands')
    mx = math.fsum(x)/len(x); my = math.fsum(y)/len(y)
    denominator = math.fsum((v-mx)**2 for v in x)
    require(denominator > 0, 'Constant independent variable')
    return math.fsum((a-mx)*(b-my) for a, b in zip(x, y, strict=True))/denominator


def close_enough(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected) and
        math.isclose(actual, expected, rel_tol=3e-9, abs_tol=3e-10), 'Independent '+name+' differs')
    return abs(actual-expected)


def diagnostics(directory):
    validate_sources(directory); rows = []
    for number in range(2):
        ns = namespace(directory, number); costs = []; slips = []; schedules = []
        ns['set_order_cost'] = lambda value, **kw: costs.append([value, kw])
        ns['set_slippage'] = slips.append
        ns['run_daily'] = lambda fn, **kw: schedules.append([fn.__name__, kw]); ns['initialize'](None)
        rows.append({'case': f'{number}_defaults', 'costs': costs, 'slippage': slips, 'schedules': schedules,
            'pool': ns['g'].stock_pool if number else ns['g'].index_pool})
        x = pd.Series(np.arange(18, dtype=float)+10); y = x*1.25+3
        rows.append({'case': f'{number}_ols', 'values': [float(v) for v in ns['get_ols'](x, y)]})
        rows.append({'case': f'{number}_zscore', 'value': float(ns['get_zscore']([1., 2., 3.]))})
    ns = namespace(directory, 0); g = ns['g']; orders = []
    ns['get_stock_pool'] = lambda: POOLS[0]; ns['order_target_value'] = lambda s, v: orders.append([s, v])
    for signal in ('BUY', 'KEEP', 'SELL'):
        ns['get_signal'] = lambda value=signal: value
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=defaultdict(lambda: SimpleNamespace(total_amount=0)), available_cash=100.))
        orders.clear()
        with redirect_stdout(StringIO()): ns['trade'](ctx)
        rows.append({'case': '31_empty_'+signal, 'orders': orders.copy(), 'slots': g.stock_number})
    for signal in ('KEEP', 'SELL'):
        ns['get_signal'] = lambda value=signal: value
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={POOLS[0][0]: SimpleNamespace(total_amount=100)}, available_cash=100.))
        orders.clear()
        with redirect_stdout(StringIO()): ns['trade'](ctx)
        rows.append({'case': '31_held_'+signal, 'orders': orders.copy()})
    ns['get_signal'] = lambda: 'KEEP'; ns['get_stock_pool'] = lambda: ()
    orders.clear()
    with redirect_stdout(StringIO()): ns['trade'](ctx)
    rows.append({'case': '31_empty_pool_KEEP', 'orders': orders.copy()})
    ns = namespace(directory, 1); selected(directory, 1, ns); g = ns['g']; ns['attribute_history'] = lambda *a, **kw: pd.DataFrame(
        {'close': np.exp(np.arange(111)*.001)}, index=pd.date_range('2020-01-01', periods=111))
    for name in ('initial_stock_motion', 'get_rank'):
        try: ns[name](*( [None, POOLS[1]] if name == 'get_rank' else [POOLS[1]] ))
        except KeyError as exc: rows.append({'case': '42_'+name+'_position_error', 'exception': type(exc).__name__})
        else: raise AssertionError('Expected Series positional compatibility failure')
    orders = []; g.check_out_list = [POOLS[1][0], 10., 2.]; g.timing_signal = 'BUY'
    ns['order_target_value'] = lambda s, v: orders.append([s, v])
    ctx = SimpleNamespace(current_dt=SimpleNamespace(hour=9, minute=35), portfolio=SimpleNamespace(
        positions=defaultdict(lambda: SimpleNamespace(total_amount=0)), cash=100.))
    with redirect_stdout(StringIO()): ns['my_sell2buy'](ctx)
    rows.append({'case': '42_triple_list_numeric_orders', 'orders': orders.copy()})
    orders.clear(); ctx.portfolio.positions = {'X': SimpleNamespace(security='X', avg_cost=100., price=10.)}
    try:
        with redirect_stdout(StringIO()): ns['check_lose'](ctx)
    except NameError as exc: rows.append({'case': '42_stoploss_undefined_value', 'exception': type(exc).__name__, 'orders': orders.copy()})
    else: raise AssertionError('Expected undefinedvalue after stoploss intent')
    # Synthetic WR isolates branch precedence, never supplies real historical WR.
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'low': np.arange(18)+10., 'high': np.arange(18)*1.2+20., 'close': np.arange(18)+50.})
    ns['get_ols'] = lambda *a: (0., 1., 1.)
    for name, score, z_slope, price_slope, wr in (
        ('WR_priority', 1., -1., -1., 97.), ('falling_positive', 1., -1., 0., 96.),
        ('falling_index', -.5, 1., -1., 96.), ('strong_index', -1., 1., 13., 96.),
        ('score_BUY', -.67, 0., 0., 96.), ('score_equal_SELL', -.68, 0., 0., 96.)):
        g.slope_series = [1.]*608; g.rsrs_score_history = [0.]*8
        ns['get_zscore'] = lambda *a, v=score: v; ns['get_zscore_slope'] = lambda *a, v=z_slope: v
        ns['np'] = SimpleNamespace(arange=np.arange, polyfit=lambda *a, v=price_slope: np.array([v, 0.]))
        ns['WR'] = lambda *a, v=wr, **kw: ({g.ref_stock: v}, {g.ref_stock: v})
        signal = ns['get_timing_signal'](SimpleNamespace(previous_date='synthetic'), g.ref_stock)
        rows.append({'case': '42_'+name, 'signal': signal, 'slopes': len(g.slope_series), 'zscores': len(g.rsrs_score_history), 'synthetic_WR': True})
    return {'cases': rows, 'not_a_backtest': True, 'external_messages': 0,
        'platform_assumptions': 'Explicit math/default-position/WR/order stubs; Series failures and numeric securities are original defects, no trading correction'}


def index_components(directory):
    bound = read(directory/'index-input-binding.json'); path = directory/'index-input.parquet'
    require(file_sha(path) == bound['frozen_sha256'], 'Index changed')
    frame = pd.read_parquet(path).sort_values('date').reset_index(drop=True)
    require(len(frame) == 5280 and frame.date.max() <= '2026-09-29' and
        np.isfinite(frame[['low', 'high', 'close']].to_numpy(float)).all() and (frame.low > 0).all() and (frame.high > frame.low).all(), 'Invalid index operands')
    records = []; summaries = []; extraction = {}; max_error = {n: 0. for n in ('slope', 'r2', 'score', 'MA', 'rsrs_slope', 'index_slope')}
    for number in range(2):
        ns = namespace(directory, number); selected(directory, number, ns); g = ns['g']; end = [625 if number else 617]
        def history(stock, count, *a, **kw):
            require(stock == '000300.XSHG', 'Index identifier changed')
            return frame.iloc[end[0]+1-count:end[0]+1].reset_index(drop=True).copy()
        ns['attribute_history'] = history
        independent = [batch75.independent_ols(frame.low.iloc[i:i+18], frame.high.iloc[i:i+18]) for i in range(608 if number else 600)]
        slopes = [v[1] for v in independent]
        if number:
            g.slope_series, g.rsrs_score_history = ns['initial_slope_series']()
            zscores = [independent_z(slopes[i+1:i+601])*independent[i+600][2] for i in range(8)]
            scorer, extraction['index42_prefix'] = extracted(directory, 'prefix', ns)
            require(len(g.slope_series) == len(slopes) == 608 and len(g.rsrs_score_history) == len(zscores) == 8, '42 seed lengths differ')
        else:
            g.slope_series = ns['initial_slope_series']()[:-1]; slopes = slopes[:-1]
            require(len(g.slope_series) == len(slopes) == 599, '31 seed length differs')
        for a, b in zip(g.slope_series, slopes, strict=True): close_enough(a, b, 'seed slope')
        if number:
            for a, b in zip(g.rsrs_score_history, zscores, strict=True): close_enough(a, b, 'seed zscore')
        signals = []
        with redirect_stdout(StringIO()):
            for k in range(end[0], len(frame)):
                end[0] = k; window = frame.iloc[k-17:k+1]
                _, slope, r2 = batch75.independent_ols(window.low, window.high); slopes.append(slope)
                score = independent_z(slopes[-600:])*r2
                if number:
                    zscores.append(score); rsrs_slope = independent_slope(range(8), zscores[-8:])
                    index_slope = independent_slope(range(8), window.close.iloc[-8:])
                    actual = tuple(float(v) for v in scorer(None, g.ref_stock)); expected = (slope, r2, score, rsrs_slope, index_slope)
                    for name, a, b in zip(('slope', 'r2', 'score', 'rsrs_slope', 'index_slope'), actual, expected, strict=True):
                        max_error[name] = max(max_error[name], close_enough(a, b, name))
                    slopes.pop(0); zscores.pop(0)
                    require(len(slopes) == len(g.slope_series) == 608 and len(zscores) == len(g.rsrs_score_history) == 8, '42 state length changed')
                    records.append({'source': '42', 'history_end': str(frame.date.iloc[k]), 'slope': actual[0], 'r2': actual[1],
                        'score': actual[2], 'independent_score': score, 'rsrs_slope': actual[3], 'index_slope': actual[4],
                        'signal': 'blocked_WR', 'independent_signal': 'blocked_WR', 'complete_signal': False})
                else:
                    signal = ns['get_signal'](); _, a_slope, a_r2 = ns['get_ols'](window.low, window.high)
                    a_score = ns['get_zscore'](g.slope_series[-600:])*a_r2
                    closes = frame.close.iloc[k-22:k+1].to_numpy(float)
                    today = math.fsum(closes[3:])/20; before = math.fsum(closes[:-3])/20
                    expected = 'BUY' if score > .7 and today > before else 'SELL' if score < -.7 and today < before else 'KEEP'
                    require(signal == expected, 'Independent31 signal differs'); signals.append(signal)
                    for name, a, b in (('slope', a_slope, slope), ('r2', a_r2, r2), ('score', a_score, score),
                        ('MA', float(np.mean(closes[3:])), today), ('MA', float(np.mean(closes[:-3])), before)):
                        max_error[name] = max(max_error[name], close_enough(a, b, name))
                    records.append({'source': '31', 'history_end': str(frame.date.iloc[k]), 'slope': float(a_slope), 'r2': float(a_r2),
                        'score': float(a_score), 'independent_score': score, 'today_MA': float(np.mean(closes[3:])),
                        'before_MA': float(np.mean(closes[:-3])), 'signal': signal, 'independent_signal': expected, 'complete_signal': True})
        summaries.append({'source': '42' if number else '31', 'windows': len(frame)-(625 if number else 617),
            'first_history_end': str(frame.date.iloc[625 if number else 617]), 'last_history_end': str(frame.date.iloc[-1]),
            'complete_signal': not bool(number), 'signal_counts': pd.Series(signals, dtype=str).value_counts().to_dict(),
            'seed_policy': '626 rows->608 slopes/8 shifted zscore*R2 pairs; append latest, then pop one' if number else '618 rows->600 slopes,discard599;append latest start600;skipped start599 retained'})
    return pd.DataFrame(records), {'sources': summaries, 'max_abs_difference': max_error, 'signal_differences': 0,
        '42_complete_signal_windows': 0, 'not_a_backtest': True}, extraction


def momentum_components(directory):
    operands = read(directory/'reused-inputs.json')['inputs']; spaces = [namespace(directory, n) for n in range(2)]
    log_score, log_ast = extracted(directory, 'log', spaces[0]); bias_score, bias_ast = extracted(directory, 'bias', spaces[1])
    records = []; valid = [{}, {}]; blocked = []; maxima = {'log_score': 0., 'bias_score': 0.}
    for number, pool in enumerate(POOLS):
        for stock in pool:
            row = operands[stock]; require(file_sha(row['file']) == row['sha256'], 'Fund history changed')
            frame = pd.read_parquet(row['file']).copy(); frame['date'] = pd.to_datetime(frame.date)
            require(frame.date.notna().all() and not frame.date.duplicated().any(), 'Fund dates invalid')
            frame = frame.loc[frame.date <= pd.Timestamp('2026-09-29')].sort_values('date').set_index('date')
            frame['close'] = pd.to_numeric(frame.close, errors='raise'); width = 110 if number else 25; valid[number][stock] = {}
            for k in range(width-1, len(frame)):
                window = frame.iloc[k+1-width:k+1][['close']].copy(); prices = window.close.to_numpy(float); date = window.index[-1]
                if not np.isfinite(prices).all() or not (prices > 0).all() or np.ptp(prices) == 0:
                    blocked.append([number, stock, str(date.date()), 'invalid_or_constant_close']); continue
                if number:
                    bias = (window.close/window.close.rolling(90).mean()).iloc[-20:].to_numpy(float)
                    actual = float(bias_score(bias))
                    other_bias = [prices[i]/(math.fsum(prices[i-89:i+1])/90) for i in range(90, 110)]
                    independent = independent_slope(range(20), [v/other_bias[0] for v in other_bias])*10000
                else:
                    spaces[0]['attribute_history'] = lambda *a, f=window, **kw: f.copy(); actual = float(log_score(stock))
                    independent = previous.independent(prices, False)[0]
                name = 'bias_score' if number else 'log_score'; maxima[name] = max(maxima[name], close_enough(actual, independent, name))
                valid[number][stock][date] = (tuple(window.index), actual, independent, window)
                records.append({'source': '42' if number else '31', 'instrument': stock, 'history_end': str(date.date()), 'score': actual,
                    'independent_score': independent, 'operand_policy': 'ndarray_bias_positional_reference_only' if number else 'original_get_socre'})
    ranks = []; mismatches = 0; leaders = 0
    for number, pool in enumerate(POOLS):
        common = sorted(set.intersection(*(set(valid[number].get(s, {})) for s in pool)))
        for date in common:
            items = [valid[number][s][date] for s in pool]
            if any(v[0] != items[0][0] for v in items[1:]):
                blocked.append([number, 'ALL', str(date.date()), 'unequal_native_dates']); continue
            holdings = ('null', *pool) if number else ('null',)
            for holding in holdings:
                scores = [v[1]*(1.04 if number and s == holding else 1.) for s, v in zip(pool, items, strict=True)]
                others = [v[2]*(1.04 if number and s == holding else 1.) for s, v in zip(pool, items, strict=True)]
                order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
                independent_order = sorted(range(len(pool)), key=lambda i: others[i], reverse=True)
                rank = [pool[i] for i in order]; independent_rank = [pool[i] for i in independent_order]
                if not number:
                    ns = spaces[0]; ns['attribute_history'] = lambda s, *a, **kw: valid[0][s][date][3].copy()
                    original = list(ns['get_stock_pool']()); require(original == rank, 'Original31 rank differs')
                mismatches += rank != independent_rank; leaders += rank[0] != independent_rank[0]
                ranks.append({'source': '42' if number else '31', 'history_end': str(date.date()), 'controlled_holding': holding,
                    'rank': '|'.join(rank), 'independent_rank': '|'.join(independent_rank), 'leader': rank[0],
                    'original_full_rank_executed': not bool(number), 'not_a_portfolio_state': True})
    scores_frame = pd.DataFrame(records); ranks_frame = pd.DataFrame(ranks); coverage = {}
    for source in ('31', '42'):
        scores_source = scores_frame.loc[scores_frame.source == source]; ranks_source = ranks_frame.loc[ranks_frame.source == source]
        coverage[source] = {'single_fund_windows': len(scores_source), 'rank_rows': len(ranks_source),
            'rank_dates': int(ranks_source.history_end.nunique()),
            'first_history_end': ranks_source.history_end.min() if len(ranks_source) else None,
            'last_history_end': ranks_source.history_end.max() if len(ranks_source) else None,
            'original_full_rank_calls': int(ranks_source.original_full_rank_executed.sum()),
            'years': {str(year): {'rank_dates': int(group.history_end.nunique()), 'rank_rows': len(group)}
                for year, group in ranks_source.groupby(ranks_source.history_end.str[:4])}}
    return (scores_frame, ranks_frame, {'windows': len(records), 'rank_rows': len(ranks), 'coverage': coverage,
        'independent_rank_differences': mismatches, 'independent_leader_differences': leaders, 'blocked': blocked,
        'max_abs_difference': maxima, 'not_a_backtest': True,
        'limits': ['42 score executes only original assignment with ndarray bias; full original Series get_rank still errors',
            '42 four controlled holding cases are arithmetic counterfactuals, not a realised portfolio path',
            'Native raw histories have known discontinuities and unproved event/adjustment vintage; no trading return computed']},
        {'log_score': log_ast, 'bias_score_ndarray_operand': bias_ast})


def expected_request(method, url, kwargs, number):
    params = {k: v for k, v in (kwargs.get('params') or {}).items() if k != 'ut'}
    require(number == 0 and method.upper() == 'GET' and url == ENDPOINT and params == PARAMETERS, 'Unexpected/repeated request')
    return {'method': 'GET', 'url': url, 'parameters': params, 'vendor_marker_recorded': False}


def invoke():
    import akshare as ak
    return ak.fund_etf_hist_min_em(**QUERY)


def worker(directory):
    folder = directory/'probes'/CASE; folder.mkdir(exist_ok=False)
    row = {'case': CASE, 'query': QUERY, 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'request_count_complete': True, 'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        row['attempts'].append(expected_request(method, url, kwargs, len(row['attempts'])))
        save(folder/'request-intent.json', row['attempts'][-1])
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder/'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'Native API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(StringIO()): frame = invoke()
        path = folder/'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = type(exc).__name__
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); (directory/'probes').mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'scripts.review_strategy_batch79', 'worker', '--directory', str(directory)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        save(directory/'worker.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        require(result.returncode == 0, 'Worker failed before archive')
    except subprocess.TimeoutExpired:
        save(directory/'worker.json', {'command': command, 'status': 'timeout', 'seconds': 45})
        folder = directory/'probes'/CASE; folder.mkdir(exist_ok=True)
        if not (folder/'result.json').exists():
            intent = folder/'request-intent.json'
            save(folder/'result.json', {'case': CASE, 'query': QUERY, 'status': 'timeout',
                'attempts': [read(intent)] if intent.exists() else [], 'request_count_complete': False, 'wire_responses': [],
                'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s deadline; completion unproved',
                'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False})
    save(directory/'probe-results.json', {'results': [read(directory/'probes'/CASE/'result.json')], 'original_input_rows_admitted': 0})
    return archive(root, 'Batch79 one bounded60m native attempt archived; no trading input published')


def candidate_audit(directory):
    validate_sources(directory); require(api_evidence() == read(directory/'existing-apis.json') and
        reused_inputs() == read(directory/'reused-inputs.json'), 'API/reused histories changed')
    row = read(directory/'probe-results.json')['results'][0]
    require(row == read(directory/'probes'/CASE/'result.json') and row['case'] == CASE and row['query'] == QUERY and
        row['api_sha256'] == file_sha(directory/'existing-apis.json'), 'Minute binding changed')
    for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Minute response changed')
    for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial response changed')
    coverage = {}
    if row['status'] in ('success', 'empty'):
        require(len(row['attempts']) == len(row['wire_responses']) == 1 and file_sha(row['file']) == row['sha256'], 'Minute capture changed')
        calls = []
        def replay(session, method, url, **kwargs):
            require(expected_request(method, url, kwargs, len(calls)) == row['attempts'][0], 'Minute replay request differs'); calls.append(True)
            wire = row['wire_responses'][0]; response = requests.Response(); response.status_code = wire['status_code']
            response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
            response.raise_for_status(); return response
        with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke()
        pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
        require(len(calls) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Minute parse scope differs')
        times = pd.to_datetime(frame['时间'])
        coverage = {'rows': len(frame), 'first': str(times.min()) if len(frame) else None,
            'last': str(times.max()) if len(frame) else None, 'historical_full_pool_complete': False}
    else: require(row.get('error'), 'Minute failure lacks cause')
    profiles = []
    for stock, operand in read(directory/'reused-inputs.json')['inputs'].items():
        frame = pd.read_parquet(operand['file']); dates = pd.to_datetime(frame.date)
        profiles.append({'instrument': stock, 'rows': len(frame), 'first': str(dates.min().date()), 'last': str(dates.max().date()),
            'post_cutoff_rows': int((dates > pd.Timestamp('2026-09-29')).sum()), 'strict_usable': False})
    return {'daily_profiles': profiles, 'minute_status': row['status'], 'minute_coverage': coverage,
        'minute_native_parser_match': row['status'] in ('success', 'empty'),
        'new_requests': len(row['attempts']) if row['request_count_complete'] else None,
        'recorded_request_intents': len(row['attempts']), 'request_count_complete': row['request_count_complete'],
        'http_statuses': [w['status_code'] for w in row['wire_responses']], 'original_input_rows_admitted': 0, 'not_a_backtest': True}


def audit(root, directory):
    binding(root, directory); save(directory/'candidate-audit.json', candidate_audit(directory))
    return archive(root, 'Batch79 seven reused daily candidates and bounded60m coverage audited; trade gates retained')


def study(root, directory):
    outputs = ('diagnostics.json', 'source-extraction.json', 'index-components.parquet', 'momentum-components.parquet', 'rank-components.parquet', 'component-research.json')
    require(all(not (directory/n).exists() for n in outputs), 'Study archive exists'); binding(root, directory)
    validate_sources(directory); before = implementation(); diag = diagnostics(directory)
    index, index_summary, index_ast = index_components(directory)
    momentum, ranks, momentum_summary, momentum_ast = momentum_components(directory)
    require(before == implementation(), 'Implementation changed during study')
    for name, table in (('index-components.parquet', index), ('momentum-components.parquet', momentum), ('rank-components.parquet', ranks)):
        with (directory/name).open('xb') as stream: table.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diag); save(directory/'source-extraction.json', {**index_ast, **momentum_ast})
    save(directory/'component-research.json', {'index': index_summary, 'momentum': momentum_summary, 'implementation_sha256': before,
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_input_rows_admitted': 0})
    return archive(root, 'Batch79 longest original31 RSRS/log and42 preWR/positional BIAS references archived; no trading returns')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch79 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); index, index_summary, index_ast = index_components(directory)
        momentum, ranks, momentum_summary, momentum_ast = momentum_components(directory)
        candidates = candidate_audit(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    for name, table in (('index-components.parquet', index), ('momentum-components.parquet', momentum), ('rank-components.parquet', ranks)):
        pd.testing.assert_frame_equal(table, pd.read_parquet(directory/name), check_exact=True)
    require(before == implementation() and read(directory/'component-research.json') == {'index': index_summary, 'momentum': momentum_summary,
        'implementation_sha256': before, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_input_rows_admitted': 0} and
        diag == read(directory/'diagnostics.json') and {**index_ast, **momentum_ast} == read(directory/'source-extraction.json') and
        candidates == read(directory/'candidate-audit.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    frame, operand = index_operand(root)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory/'index-input.parquet'), check_exact=True)
    require({**operand, 'frozen_sha256': file_sha(directory/'index-input.parquet')} == read(directory/'index-input-binding.json'), 'Index snapshot binding changed')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch79 forbidden-network diagnostics/index/momentum/ranks/minute parser/inventory/three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch79.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch79_research.py', 'tests/unit/test_batch78_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def sanity(root, directory):
    binding(root, directory); path = directory/'specialized-tests.log'
    with path.open('x') as stream: result = subprocess.run([sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch79_research.py'], stdout=stream, stderr=subprocess.STDOUT)
    save(directory/'specialized-tests.json', {'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path), 'implementation_sha256': implementation()})
    require(result.returncode == 0, 'Specialized tests failed: '+str(path)); return {'status': 'passed'}


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
    summary = read(directory/'component-research.json')
    require(summary['momentum']['independent_leader_differences'] == summary['momentum']['independent_rank_differences'] ==
        summary['index']['signal_differences'] == summary['index']['42_complete_signal_windows'] == 0, 'Independent ranking/signal gate differs')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch79 RSRS/BIAS source and longest reference research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch79-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'diagnostics': read(directory/'diagnostics.json'), 'components': summary, 'inventory': read(directory/'dependency-inventory.json'),
        'candidates': read(directory/'candidate-audit.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'sanity', 'worker', 'probe', 'audit', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); result = worker(args.directory) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
