"""Freeze ETF information/rotation sources and long reference components."""
import argparse
import ast
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

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch59 import index_operand
from scripts import review_strategy_batch74 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch75/20261008-ETF-info-rotation')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2021年度精选策略/43.ETF资源收集整合（附代码与excel表格）.txt',
    '2023年度精选策略/54.ETF轮动策略-入门1.0.txt')
SOURCE_SHA = ('204be735c75cec3b10a9ebc74d26bc5bb028b40a7942bb948049cc2853e0ddb5',
    'd8722005b2ba5d3f1b028bfcb49ca35bd8157deefa9b9a4e1ab5de9e500996c3')
POOL = ('159915.XSHE', '510300.XSHG', '510500.XSHG')
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/backtrader', 'backtrader/indicators/ols.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch75.py', 'tests/unit/test_batch75_research.py',
    'scripts/review_strategy_batch59.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'missing-inputs.json', 'index-input.parquet', 'index-input-binding.json', 'reused-ETF.json',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json', 'diagnostics.json',
    'index-components.parquet', 'ETF-components.parquet', 'component-research.json',
    'probe-results.json', 'candidate-audit.json', 'ranking-components.parquet', 'ranking-research.json',
    'offline-catalog.json', 'offline-verification.json'}
QUERIES = {'159915': 'sz159915', '510500': 'sh510500'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-08-batch74-verification.json')
    proof = read(previous.DIRECTORY / 'final-binding-verification.json')
    require(read(receipt)['status'] == proof['status'] == 'ok' and file_sha(receipt) == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch74.py') == proof['final_script_sha256'], 'Accepted batch74 changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'proof_sha256': file_sha(previous.DIRECTORY / 'final-binding-verification.json'), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def missing_inputs():
    command = ['rg', '--files', '--hidden', '--no-ignore', 'repo', 'data']
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    names = ('ETFS.xlsx', 'ETFS_NEW.xlsx')
    found = {name: sorted(p for p in process.stdout.splitlines() if Path(p).name == name) for name in names}
    return {'command': command, 'matches': found, 'missing': [n for n in names if not found[n]],
        'scope': 'Local repo/data only; external machines are not searched', 'replacement_allowed': False,
        'redacted_site': 'http://.../', 'site_not_inferred': True, 'not_a_backtest': True}


def preflight(root, directory):
    bound = binding(root); frame, operand = index_operand(root); missing = missing_inputs()
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch75 ETF information/rotation source research started')
    save(directory / 'input-binding.json', bound); save(directory / 'missing-inputs.json', missing)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    path = directory / 'index-input.parquet'
    with path.open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory / 'index-input-binding.json', {**operand, 'frozen_sha256': file_sha(path),
        'use': 'Daily reference OHLC only; original platform vintage and callback equality unproved'})
    receipt20 = Path('docs/handoff/2026-10-05-batch20-verification.json')
    probe = Path('data/staging/strategies-batch20/20261005-macd-patterns/probes/etf510300/result.json')
    row = read(probe)['files'][0]
    accepted20 = read(receipt20)
    require(accepted20['status'] == 'ok' and
        file_sha(probe) == accepted20['checks']['evidence_sha256']['probes/etf510300/result.json'] and file_sha(row['file']) == row['sha256'] and
        len(pd.read_parquet(row['file'])) == row['rows'], 'Prior ETF input changed')
    save(directory / 'reused-ETF.json', {'receipt': str(receipt20), 'receipt_sha256': file_sha(receipt20),
        'probe': str(probe), 'probe_sha256': file_sha(probe), 'payload': row, 'strict_usable': False})
    return {'status': 'ok', 'index_rows': len(frame), 'missing_original_excel': missing['missing']}


def api_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_sina; module = inspect.getmodule(fn); code = inspect.getsource(fn)
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'decoder': module.hk_js_decode, 'decoder_sha256': hashlib.sha256(module.hk_js_decode.encode()).hexdigest(),
        'queries': QUERIES, 'policy': 'One native Sina request per missing ETF; no retry/proxy/redirect,8/10s,parent45s',
        'original_site_redacted': True}


def inventory(root, directory):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'original_excel': read(directory / 'missing-inputs.json'), 'pool': list(POOL),
        'index_rows': read(directory / 'index-input-binding.json')['rows'],
        'reused_ETF_rows': read(directory / 'reused-ETF.json')['payload']['rows'],
        'original_trades_admitted': 0, 'not_a_backtest': True,
        'gates': ['Original ETF identity/events/adjustment/daily state/fund fees incomplete',
            'Original1-minute last price/limit states and9:30 versus open callback order unproved',
            'Redacted ETF information site and original Excel unavailable; no invented replacement',
            'Fixed ex-post ETF pool and historical platform data vintage remain limitations']}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码') / name; copied = folder / (strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    refs = []; (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository) / name; copied = directory / 'references' / source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Native ETF history parser and regression reference; no second trading ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder / 'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch75 two complete ETF information/rotation sources frozen; original trading gates retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(SOURCES) and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, names, namespace):
    copied = directory / 'source-reviews' / (strategy_id(SOURCES[number])+'.source')
    require(file_sha(copied) == SOURCE_SHA[number], 'Frozen source changed')
    nodes = [n for n in ast.parse(read_source(copied)[0]).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names) and
        all(previous.previous.safe_function(n) for n in nodes), 'Unsafe/missing selected definition')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch75-definitions>', 'exec'), namespace)
    return namespace


def diagnostics(directory):
    validate_sources(directory); rows = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, error, fn):
        try: fn()
        except error as exc: record(case, exception=type(exc).__name__, detail=str(exc)[:200])
        else: raise AssertionError('Expected original error: '+case)
    html = ['<div class="infoOfFund"><td>name:ETF|kind:index</td></div>']
    info = selected(directory, 0, ['get_content'], {'BeautifulSoup': BeautifulSoup, 're': re, 'get_html': lambda code: html[0]})
    record('HTML_two_pairs', values=info['get_content']('515950'))
    for content, label, error in [('<div></div>', 'HTML_missing_node', AttributeError),
        ('<div class="infoOfFund"><td>name</td></div>', 'HTML_no_separator', IndexError),
        ('<div class="infoOfFund"><td>name:ETF|kind</td></div>', 'HTML_odd_parts', IndexError)]:
        html[0] = content; failure(label, error, lambda: info['get_content']('515950'))
    calls = []
    web = selected(directory, 0, ['get_html'], {'requests': SimpleNamespace(get=lambda url, **kw:
        calls.append(url) or SimpleNamespace(text='synthetic'))})
    web['get_html']('515950'); record('redacted_URL_never_requested', local_stub_calls=calls)
    log = SimpleNamespace(info=lambda *a: None, debug=lambda *a: None, set_level=lambda *a: None)
    ns = selected(directory, 1, ['initialize', 'get_rank', 'get_ols', 'initial_slope_series', 'get_zscore',
        'get_timing_signal', 'filter_paused_stock', 'filter_st_stock', 'filter_limitup_stock', 'filter_limitdown_stock',
        'order_target_value_', 'open_position', 'close_position', 'adjust_position', 'my_trade', 'check_lose', 'print_trade_info'],
        {'g': SimpleNamespace(), 'np': np, 'log': log, 'math': math})
    g = ns['g']; schedules = []; costs = []; slips = []
    ns.update(set_benchmark=lambda *a: None, set_option=lambda *a: None,
        set_slippage=lambda s: slips.append(s), FixedSlippage=lambda s: s,
        set_order_cost=lambda c, **kw: costs.append([c, kw]), OrderCost=lambda **kw: kw,
        run_daily=lambda fn, **kw: schedules.append([fn.__name__, kw]), initial_slope_series=lambda: list(range(600)))
    ns['initialize'](SimpleNamespace()); record('actual_pool_parameters_seed_and_schedules', pool=g.stock_pool,
        N=g.N, M=g.M, seed_len=len(g.slope_series), seed_last=g.slope_series[-1], schedules=schedules, costs=costs, slips=slips)
    prices = {POOL[0]: np.exp(np.arange(29)*.002), POOL[1]: np.exp(np.arange(29)*.001), POOL[2]: np.exp(np.arange(29)*-.001)}
    ns['attribute_history'] = lambda stock, *a, **kw: pd.DataFrame({'close': prices[stock]})
    record('ranking_uses_global_pool', ignored_argument=['OTHER'], rank=ns['get_rank'](['OTHER']))
    del ns['math']; failure('implicit_math_missing', NameError, lambda: ns['get_rank'](list(POOL))); ns['math'] = math
    with np.errstate(all='ignore'):
        record('zero_variance_zscore', finite=bool(np.isfinite(ns['get_zscore']([1.]*600))))
        record('constant_regression_R2', finite=bool(np.isfinite(ns['get_ols'](np.arange(18), np.ones(18))[2])))
    for score, rising in ((.7, True), (.7001, True), (-.7, False), (-.7001, False)):
        ns['get_ols'] = lambda *a: (0., 1., 1.); ns['get_zscore'] = lambda *a, s=score: s
        ns['attribute_history'] = lambda stock, count, *a, r=rising, **kw: pd.DataFrame(
            {'close': np.arange(count)*(1 if r else -1)+100, 'low': np.arange(count)+1., 'high': np.arange(count)+2.})
        record('timing_boundary_'+str(score), signal=ns['get_timing_signal']('IGNORED'))
    current = {p: SimpleNamespace(paused=False, is_st=False, name='ETF', high_limit=11., low_limit=9.) for p in POOL}
    ns['get_current_data'] = lambda: current; orders = []
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={}, cash=10000.))
    ns['history'] = lambda *a, **kw: {p: np.array([11.]) for p in POOL}
    record('new_limit_up_excluded', retained=ns['filter_limitup_stock'](ctx, list(POOL)))
    ctx.portfolio.positions = {POOL[0]: SimpleNamespace(security=POOL[0], total_amount=100)}
    record('held_limit_up_retained', retained=ns['filter_limitup_stock'](ctx, list(POOL)))
    ns['history'] = lambda *a, **kw: {p: np.array([9.]) for p in POOL}
    record('held_limit_down_retained', retained=ns['filter_limitdown_stock'](ctx, list(POOL)))
    for field, value in [('paused', True), ('is_st', True), ('name', '*ETF')]:
        setattr(current[POOL[0]], field, value)
        fn = ns['filter_paused_stock'] if field == 'paused' else ns['filter_st_stock']
        record('filter_'+field, retained=fn([POOL[0]])); setattr(current[POOL[0]], field, False if field != 'name' else 'ETF')
    ns['get_rank'] = lambda *a: [POOL[0]]; ns['get_timing_signal'] = lambda *a: 'KEEP'
    ns['history'] = lambda *a, **kw: {p: np.array([10.]) for p in POOL}
    ns['adjust_position'] = lambda c, pool: orders.append(['adjust', pool.copy()])
    ns['my_trade'](ctx); record('KEEP_still_adjusts', calls=orders.copy()); orders.clear()
    ns['get_rank'] = lambda *a: [POOL[1]]; ns['get_timing_signal'] = lambda *a: 'BUY'
    ns['close_position'] = lambda p: orders.append(['close', p.security])
    ns['my_trade'](ctx); record('510300_forces_SELL', calls=orders.copy()); orders.clear()
    ns['get_rank'] = lambda *a: [POOL[0]]; current[POOL[0]].paused = True
    ns['get_timing_signal'] = lambda *a: 'KEEP'; ns['my_trade'](ctx)
    record('filtered_top_not_replaced', calls=orders.copy()); orders.clear(); current[POOL[0]].paused = False
    original_adjust = selected(directory, 1, ['adjust_position'], ns)['adjust_position']
    ctx.portfolio.positions = {}; failure('missing_position_key', KeyError, lambda: original_adjust(ctx, [POOL[0]]))
    ctx.portfolio.positions = {p: SimpleNamespace(security=p, total_amount=100) for p in POOL[:2]}
    ns['close_position'] = lambda p: ctx.portfolio.positions.pop(p.security)
    failure('immediate_position_delete_during_adjust', RuntimeError, lambda: original_adjust(ctx, []))
    ns['get_rank'] = lambda *a: []; ns['get_timing_signal'] = lambda *a: 'SELL'
    ctx.portfolio.positions = {p: SimpleNamespace(security=p, total_amount=100) for p in POOL[:2]}
    failure('immediate_position_delete_during_SELL', RuntimeError, lambda: ns['my_trade'](ctx))
    selected(directory, 1, ['close_position'], ns); ns['OrderStatus'] = SimpleNamespace(held='held')
    for filled, status, amount in [(0, 'cancelled', 100), (50, 'cancelled', 100), (100, 'held', 100)]:
        ns['order_target_value'] = lambda *a, f=filled, s=status, am=amount: SimpleNamespace(filled=f, status=s, amount=am)
        record('order_state_'+str(filled), open=ns['open_position']('A', 1000), close=ns['close_position'](SimpleNamespace(security='A')))
    ns['order_target_value'] = lambda *a: orders.append(list(a))
    for price in (80., 79.99, 80.01):
        orders.clear(); ctx.portfolio.positions = {'A': SimpleNamespace(security='A', avg_cost=100., price=price, value=price*100, total_amount=100)}
        ns['check_lose'](ctx); record('stoploss_'+str(price), orders=orders.copy())
    ctx.portfolio.positions['A'].avg_cost = 0
    failure('zero_cost_stoploss', ZeroDivisionError, lambda: ns['check_lose'](ctx))
    return {'cases': rows, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'scope': 'Original SHA-bound function definitions only; platform dependencies are explicit synthetic local stubs'}


def independent_ols(x, y):
    x = list(map(float, x)); y = list(map(float, y)); n = len(x)
    mx = math.fsum(x)/n; my = math.fsum(y)/n
    dx = [v-mx for v in x]; dy = [v-my for v in y]
    slope = math.fsum(a*b for a, b in zip(dx, dy, strict=True))/math.fsum(a*a for a in dx)
    intercept = my-slope*mx
    r2 = 1-math.fsum((b-(slope*a+intercept))**2 for a, b in zip(x, y, strict=True))/math.fsum(b*b for b in dy)
    return intercept, slope, r2


def index_components(directory):
    bound = read(directory / 'index-input-binding.json'); path = directory / 'index-input.parquet'
    require(file_sha(path) == bound['frozen_sha256'], 'Index input changed')
    frame = pd.read_parquet(path).sort_values('date').reset_index(drop=True)
    require(len(frame) >= 618 and not frame.date.duplicated().any() and frame.date.max() <= '2026-09-29', 'Index coverage invalid')
    require(np.isfinite(frame[['low', 'high', 'close']].to_numpy(float)).all() and
        (frame.low > 0).all() and (frame.high > frame.low).all() and (frame.close > 0).all(), 'Invalid index reference')
    end = [617]; calls = []
    def history(stock, count, *a, **kw):
        require(stock == '000300.XSHG', 'Original reference changed'); calls.append(count)
        return frame.iloc[end[0]+1-count:end[0]+1].reset_index(drop=True).copy()
    g = SimpleNamespace(ref_stock='000300.XSHG', N=18, M=600, mean_day=20, mean_diff_day=3, score_threshold=.7)
    ns = selected(directory, 1, ['get_ols', 'initial_slope_series', 'get_zscore', 'get_timing_signal'], {'g': g, 'np': np, 'attribute_history': history})
    g.slope_series = ns['initial_slope_series']()[:-1]
    independent = [independent_ols(frame.low.iloc[i:i+18], frame.high.iloc[i:i+18])[1] for i in range(599)]
    rows = []; maximum = {'slope': 0., 'r2': 0., 'score': 0., 'MA': 0.}; differences = 0
    with redirect_stdout(StringIO()):
        for k in range(617, len(frame)):
            end[0] = k; signal = ns['get_timing_signal'](g.ref_stock)
            block = frame.iloc[k-17:k+1]; _, slope, r2 = independent_ols(block.low, block.high)
            independent.append(slope); tail = independent[-600:]; mean = math.fsum(tail)/len(tail)
            std = math.sqrt(math.fsum((v-mean)**2 for v in tail)/len(tail)); score = (tail[-1]-mean)/std*r2
            close = frame.close.iloc[k-22:k+1].to_numpy(float); today = math.fsum(close[3:])/20; before = math.fsum(close[:-3])/20
            expected = 'BUY' if score > .7 and today > before else 'SELL' if score < -.7 and today < before else 'KEEP'
            _, actual_slope, actual_r2 = ns['get_ols'](block.low, block.high)
            actual_score = ns['get_zscore'](g.slope_series[-600:])*actual_r2
            maximum['slope'] = max(maximum['slope'], abs(slope-actual_slope)); maximum['r2'] = max(maximum['r2'], abs(r2-actual_r2))
            maximum['score'] = max(maximum['score'], abs(score-actual_score)); maximum['MA'] = max(maximum['MA'],
                abs(today-np.mean(close[3:])), abs(before-np.mean(close[:-3])))
            differences += signal != expected
            rows.append({'history_end': str(frame.date.iloc[k]), 'slope': actual_slope, 'R2': actual_r2,
                'score': actual_score, 'today_MA': float(np.mean(close[3:])), 'before_MA': float(np.mean(close[:-3])),
                'signal': signal, 'independent_signal': expected, 'independent_score': score})
    require(differences == 0, 'Independent index signal differs')
    return pd.DataFrame(rows), {'rows': len(frame), 'windows': len(rows), 'first_history_end': rows[0]['history_end'],
        'last_history_end': rows[-1]['history_end'], 'signal_differences': differences, 'max_abs_difference': maximum,
        'seed_policy': '600 initialization windows starts0..599 then discard599; append latest window start600; skipped start599 preserved',
        'signal_counts': pd.Series([r['signal'] for r in rows]).value_counts().to_dict(), 'not_a_backtest': True,
        'limits': ['Daily history-end components only; no claim of next9:30/platform calendar/callback equivalence',
            'No ETF ranking/filter/stoploss or portfolio returns in index-only study']}


def ETF_components(directory):
    reused = read(directory / 'reused-ETF.json'); row = reused['payload']
    require(file_sha(reused['probe']) == reused['probe_sha256'] and file_sha(row['file']) == row['sha256'], 'Reused ETF changed')
    frame = pd.read_parquet(row['file']); dates = pd.to_datetime(frame.date); excluded = int((dates > pd.Timestamp('2026-09-29')).sum())
    frame = frame.loc[dates <= pd.Timestamp('2026-09-29')].copy(); frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    frame = frame.sort_values('date').reset_index(drop=True)
    require(not frame.date.duplicated().any(), 'Duplicate ETF dates')
    x = np.arange(29); values = []; blocked = []; largest = 0.
    for k in range(28, len(frame)):
        close = frame.close.iloc[k-28:k+1].to_numpy(float)
        if not np.isfinite(close).all() or (close <= 0).any(): blocked.append(str(frame.date.iloc[k])); continue
        y = np.log(close); slope, intercept = np.polyfit(x, y, 1)
        denom = 28*np.var(y, ddof=1)
        if denom == 0: blocked.append(str(frame.date.iloc[k])); continue
        r2 = 1-sum((y-(slope*x+intercept))**2)/denom; score = math.pow(math.exp(slope), 250)-1; score *= r2
        _, alternative, other_r2 = independent_ols(x, y); other = math.expm1(alternative*250)*other_r2
        largest = max(largest, abs(score-other))
        values.append({'history_end': str(frame.date.iloc[k]), 'instrument': '510300.XSHG', 'score': score, 'independent_score': other})
    return pd.DataFrame(values), {'rows': len(frame), 'windows': len(values), 'blocked_windows': len(blocked),
        'blocked_dates': blocked, 'after_cutoff': excluded, 'max_score_abs_difference': largest, 'not_a_backtest': True,
        'limits': ['Unadjusted supplier candidate; source original adjusted-price and event equivalence unproved',
            'One ETF arithmetic is not three-ETF ranking or trading; original29 rows with missing observations not filled']}


def study(root, directory):
    targets = ['diagnostics.json', 'index-components.parquet', 'ETF-components.parquet', 'component-research.json']
    require(all(not (directory / name).exists() for name in targets), 'Study outputs already exist')
    binding(root, directory); before = implementation(); diag = diagnostics(directory)
    index, summary = index_components(directory); ETF, other = ETF_components(directory)
    require(before == implementation(), 'Implementation changed during study')
    for name, frame in [('index-components.parquet', index), ('ETF-components.parquet', ETF)]:
        with (directory / name).open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory / 'diagnostics.json', diag); save(directory / 'component-research.json',
        {'index': summary, 'ETF': other, 'not_a_backtest': True, 'new_trading_backtests': 0, 'implementation_sha256': before})
    return archive(root, 'Batch75 original function boundaries and full daily index/ETF reference arithmetic archived')


def expected_request(case):
    return f'https://finance.sina.com.cn/realstock/company/{QUERIES[case]}/hisdata_klc2/klc_kl.js'


def worker(directory, case):
    import akshare as ak
    folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=False)
    result = {'case': case, 'symbol': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'api_sha256': file_sha(directory / 'existing-apis.json'), 'published': False, 'strict_usable': False}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(not result['attempts'] and method.upper() == 'GET' and url == expected_request(case) and kwargs.get('params') is None, 'Unexpected/repeated request')
        result['attempts'].append({'method': 'GET', 'url': url}); session.trust_env = False
        kwargs.update(timeout=(8, 10), allow_redirects=False); response = original(session, method, url, **kwargs)
        path = folder / 'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = ak.fund_etf_hist_sina(symbol=QUERIES[case])
        path = folder / 'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        result.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    binding(root, directory); require(not (directory / 'probes').exists() and not (directory / 'probe-results.json').exists(), 'Probes already attempted')
    results = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch75', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Worker failed before result archive')
        except subprocess.TimeoutExpired:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'seconds': 45})
            folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'result.json').exists(): save(folder / 'result.json', {'case': case, 'symbol': QUERIES[case],
                'status': 'timeout', 'attempts': [], 'wire_responses': [], 'error': '45s deadline; completion unproved',
                'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                'api_sha256': file_sha(directory / 'existing-apis.json'), 'published': False, 'strict_usable': False})
        results.append(read(directory / 'probes' / case / 'result.json'))
        print(case+': '+results[-1]['status'], file=sys.stderr, flush=True)
    save(directory / 'probe-results.json', {'results': results, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0})
    return archive(root, 'Batch75 bounded missing ETF history attempts retained as separate candidates')


def candidate_audit(directory):
    import akshare as ak
    require(api_evidence() == read(directory / 'existing-apis.json'), 'API changed')
    results = read(directory / 'probe-results.json')['results']
    require(len(results) == len(QUERIES) and {r['case'] for r in results} == set(QUERIES), 'Probe scope changed')
    profiles = []
    for row in results:
        case = row['case']; require(row['symbol'] == QUERIES[case] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe binding changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Raw response changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Parsed binding changed')
            calls = []
            def replay(session, method, url, **kwargs):
                require(not calls and method.upper() == 'GET' and url == expected_request(case) and kwargs.get('params') is None, 'Unexpected native replay')
                calls.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = ak.fund_etf_hist_sina(symbol=QUERIES[case])
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(calls) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Native parse differs')
            if len(frame):
                dates = pd.to_datetime(frame.date); profile.update(rows=len(frame), first=str(dates.min().date()), last=str(dates.max().date()),
                    after_cutoff=int((dates > pd.Timestamp('2026-09-29')).sum()), unique_dates=dates.nunique())
            profile['native_parser_match'] = True
        else: require(row.get('error'), 'Failed probe lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'not_a_backtest': True,
        'limits': ['Native close candidates lack original adjusted-price/events/state/1-minute and fee proof',
            'Dates after2026-09-29 are retained in raw archive and excluded from all components']}


def audit(root, directory):
    binding(root, directory); save(directory / 'candidate-audit.json', candidate_audit(directory))
    return archive(root, 'Batch75 native ETF parser replay and historical coverage audited; original trades blocked')


def ranking_components(directory):
    reused = read(directory / 'reused-ETF.json')['payload']; rows = read(directory / 'probe-results.json')['results']
    payloads = {'510300': reused}
    for row in rows:
        if row['status'] == 'success': payloads[row['case']] = row
    columns = ['history_end', 'rank', 'independent_rank', 'forced_sell_if_unfiltered', 'score_159915', 'score_510300', 'score_510500']
    missing = sorted(set(QUERIES)-set(payloads))
    if missing: return pd.DataFrame(columns=columns), {'windows': 0, 'missing': missing, 'not_a_backtest': True,
        'limits': ['Full original fixed pool candidate histories unavailable; no replacement or synthetic ranking']}
    frames = {}; inputs = []
    for instrument in POOL:
        row = payloads[instrument[:6]]; require(file_sha(row['file']) == row['sha256'], 'Ranking input changed')
        frame = pd.read_parquet(row['file']); dates = pd.to_datetime(frame.date)
        inputs.append({'instrument': instrument, 'file': row['file'], 'sha256': row['sha256'], 'rows': len(frame),
            'after_cutoff': int((dates > pd.Timestamp('2026-09-29')).sum())})
        frame = frame.loc[dates <= pd.Timestamp('2026-09-29')].copy(); frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        frame = frame.sort_values('date').reset_index(drop=True); require(not frame.date.duplicated().any(), 'Ranking duplicate dates')
        frames[instrument] = frame
    dates = sorted(set.intersection(*(set(f.date) for f in frames.values()))); positions = {}
    for instrument, frame in frames.items(): positions[instrument] = {d:i for i, d in enumerate(frame.date)}
    windows = {}; g = SimpleNamespace(stock_pool=list(POOL), stock_num=1, momentum_day=29)
    ns = selected(directory, 1, ['get_rank'], {'g': g, 'np': np, 'math': math,
        'attribute_history': lambda stock, count, *a, **kw: windows[stock].copy()})
    values = []; blocked = []; maximum = 0.; differences = 0
    with redirect_stdout(StringIO()):
        for k in range(28, len(dates)):
            expected_dates = dates[k-28:k+1]; invalid = False
            for instrument, frame in frames.items():
                end = positions[instrument][dates[k]]; block = frame.iloc[end-28:end+1].reset_index(drop=True)
                close = block.close.to_numpy(float)
                if block.date.tolist() != expected_dates or len(block) != 29 or not np.isfinite(close).all() or (close <= 0).any() or np.var(np.log(close)) == 0:
                    invalid = True; break
                windows[instrument] = block
            if invalid: blocked.append(dates[k]); continue
            rank = ns['get_rank'](list(POOL))[0]; scores = {}; independent = {}
            for instrument in POOL:
                x = np.arange(29); y = np.log(windows[instrument].close.to_numpy(float)); slope, intercept = np.polyfit(x, y, 1)
                r2 = 1-sum((y-(slope*x+intercept))**2)/(28*np.var(y, ddof=1))
                scores[instrument] = (math.pow(math.exp(slope), 250)-1)*r2
                _, other, other_r2 = independent_ols(x, y); independent[instrument] = math.expm1(other*250)*other_r2
                maximum = max(maximum, abs(scores[instrument]-independent[instrument]))
            other_rank = sorted(POOL, key=lambda p: independent[p], reverse=True)[0]; differences += rank != other_rank
            values.append({'history_end': dates[k], 'rank': rank, 'independent_rank': other_rank,
                'forced_sell_if_unfiltered': rank == '510300.XSHG', **{'score_'+p[:6]: scores[p] for p in POOL}})
    require(differences == 0, 'Independent ETF ranking differs')
    return pd.DataFrame(values, columns=columns), {'inputs': inputs, 'windows': len(values), 'blocked_windows': len(blocked),
        'blocked_dates': blocked, 'first_history_end': values[0]['history_end'] if values else None,
        'last_history_end': values[-1]['history_end'] if values else None, 'rank_differences': differences,
        'max_score_abs_difference': maximum, 'not_a_backtest': True,
        'limits': ['Raw supplier29-row full-pool arithmetic; original adjustment and historical state/events unproved',
            'Native per-instrument29 dates must equal common29 dates; missing dates are not filled or inferred pauses',
            'Ranking510300 forces SELL only if retained after original filters; no1-minute filters/orders/returns here']}


def candidate_study(root, directory):
    require(not (directory / 'ranking-components.parquet').exists() and not (directory / 'ranking-research.json').exists(), 'Ranking archive exists')
    binding(root, directory); frame, summary = ranking_components(directory)
    with (directory / 'ranking-components.parquet').open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory / 'ranking-research.json', summary)
    return archive(root, 'Batch75 full-pool ETF candidate ranking arithmetic archived; original filters/trades still blocked')


def offline(root, directory):
    require(not (directory / 'offline-verification.json').exists(), 'Offline archive exists')
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch75 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); index, summary = index_components(directory); ETF, other = ETF_components(directory)
        dep = inventory(root, directory); candidates = candidate_audit(directory); ranking, ranked = ranking_components(directory); cat = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(index, pd.read_parquet(directory / 'index-components.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(ETF, pd.read_parquet(directory / 'ETF-components.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(ranking, pd.read_parquet(directory / 'ranking-components.parquet'), check_exact=True)
    component = read(directory / 'component-research.json')
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and summary == component['index'] and
        other == component['ETF'] and dep == read(directory / 'dependency-inventory.json') and
        ranked == read(directory / 'ranking-research.json') and candidates == read(directory / 'candidate-audit.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch75 forbidden-network diagnostics/index/ETF references/native parser and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch75.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch75_research.py',
            'tests/unit/test_batch74_research.py', 'tests/unit/test_external_cash_flows.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(not (directory / 'checked-state.json').exists(), 'Checked archive exists')
    folder = directory / 'checks'; folder.mkdir(exist_ok=False)
    for k, command in enumerate(commands()):
        path = folder / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
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
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == current == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory / 'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch75 ETF information/rotation source and long reference study accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch75-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory / 'diagnostics.json'),
        'components': read(directory / 'component-research.json'), 'ranking': read(directory / 'ranking-research.json'),
        'inventory': read(directory / 'dependency-inventory.json'),
        'candidates': read(directory / 'candidate-audit.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'audit', 'candidate_study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
