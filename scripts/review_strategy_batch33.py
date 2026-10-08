"""Freeze bank wizard and ETF rotation rules; validate components without trading."""
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
import talib

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch32 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch32/20261006-bluechip-trix-boll')
RECEIPT = Path('docs/handoff/2026-10-06-batch32-verification.json')
SOURCES = ('2020年度精选策略/64 12年年化34%，Sharpe1.2，银行股轮动.txt',
    '2021年度精选策略/18.6年17倍的etf+分级b的轮动趋势策略跟随感悟.txt',
    '2021年度精选策略/21.策略coding吐槽帖——以“网红”ETF轮动为例.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/sma.py'),
    ('repo/baostock', 'baostock/demo/demo_profit_data.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch33.py', 'tests/unit/test_batch33_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'index-input.parquet', 'calendar-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'index': {'provider': 'baostock', 'function': 'query_history_k_data_plus', 'parameters': {'code': 'sz.399001',
        'fields': 'date,code,open,high,low,close,volume,amount', 'start_date': '2020-01-02', 'end_date': '2020-01-10', 'frequency': 'd', 'adjustflag': '3'},
        'limit': 'One index sample does not supply the complete dynamic weakest-index pool or fund prices'},
    'industry': {'provider': 'baostock', 'function': 'query_stock_industry', 'parameters': {'date': '2020-01-02'},
        'limit': 'Provider industry/date sample is not proven original historical SW28 classifications or financial vintage'},
    'fund': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz150019'},
        'limit': 'Leveraged classified-fund prices do not supply historical conversion/delisting/states/default fees or intraday fills'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch32 differs')
    require(file_sha(ACCEPTED / 'input-binding.json') == receipt['checks']['evidence_sha256']['input-binding.json'], 'Accepted batch32 binding changed')
    upstream = previous.binding(root, ACCEPTED)
    upstream_receipt = previous.ACCEPTED
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = upstream_receipt / name
        accepted31 = read(previous.RECEIPT)
        require(file_sha(path) == accepted31['checks']['evidence_sha256'][name], 'Accepted index/calendar bytes differ')
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


def start(root, directory):
    checkpoint(root, directory, 'Batch33 bank and two distinct ETF rotation source audit started')
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
    search = ['rg', '-n', r'^def (financial_data_filter_qujian|get_sort_dataframe|order_style|judge_security_max_proportion|max_buy_value_or_amount|sell_by_amount_or_percent_or_none)', 'repo', '-g', '*.py', '-g', '*.txt']
    result = subprocess.run(search, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search': {'command': search, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('kuanke', 'jqdata', 'jqlib', 'jqfactor')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch33 three full source reviews frozen; original wizard/pool/fund/intraday gaps retained')


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
        'limits': ['HS300 close/volume are operator samples, not ETF prices/live11:30 or weakest-index original pool',
            'No stock/fund universe substitution; no trading results, NAV or fees calculated']})
    return archive(root, 'Batch33 accepted HS300/calendar bytes frozen for explicitly limited operator research')


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


def tree(directory, number): return previous.source_tree(directory, number)


def kernels(directory):
    functions = ('get_signal', 'EmotionMonitor', 'ETFtrade', 'before_market_open', 'get_before_after_trade_days')
    hashes = {}
    for name in functions:
        nodes = [next(n for n in tree(directory, k).body if isinstance(n, ast.FunctionDef) and n.name == name) for k in (1, 2)]
        dumps = [ast.dump(n, include_attributes=False) for n in nodes]; require(dumps[0] == dumps[1], 'Shared ETF source kernel differs')
        hashes[name] = hashlib.sha256(dumps[0].encode()).hexdigest()
    fn = next(n for n in tree(directory, 1).body if isinstance(n, ast.FunctionDef) and n.name == 'get_signal')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    body = [n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and
        n.targets[0].id in ('cp_increase', 'ma_n1', 'pre_price')]
    require(len(body) == 3, 'Score expressions differ')
    score = compile(ast.Module(body=body, type_ignores=[]), '<original-etf-score>', 'exec')
    # Diagnostic suffix receives a supplied frame; original append failure is checked separately.
    suffix = ast.FunctionDef(name='diagnostic_suffix', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='context'), ast.arg(arg='df_etf')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=fn.body[3:], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[suffix], type_ignores=[]))
    return score, compile(module, '<original-etf-suffix-diagnostic>', 'exec'), hashes


def params(directory, number=1):
    g = SimpleNamespace(); ns = {'g': g, 'log': SimpleNamespace(info=lambda *a: None),
        'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, start_date='diagnostic')}
    selected(directory, number, ('set_params',), ns); ns['set_params'](); return g


def score_case(code, values, current_price):
    x = np.asarray(values, dtype=float)
    require(x.shape == (13,) and np.isfinite(x).all() and (x > 0).all() and math.isfinite(current_price) and current_price > 0, 'Invalid score sample')
    ns = {'g': SimpleNamespace(lag1=13, lag2=13), 'close_data': {'close': x}, 'current_price': float(current_price)}
    exec(code, ns); return [float(ns[n]) for n in ('cp_increase', 'ma_n1', 'pre_price')]


def emotion_case(directory, values, dated=False):
    x = np.asarray(values, dtype=float)
    require(x.shape == (13,) and np.isfinite(x).all() and (x >= 0).all(), 'Invalid emotion sample')
    volume = pd.Series(x, index=pd.date_range('2020-01-01', periods=13)) if dated else x
    g = SimpleNamespace(lag=6, lag0=7, target_market='component'); calls = []
    def history(**kw): calls.append(kw); return {'volume': volume}
    ns = {'g': g, 'talib': talib, 'attribute_history': history}
    selected(directory, 1, ('EmotionMonitor',), ns)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); signal = ns['EmotionMonitor']()
    return int(signal), float(g.emotion_rate), calls


def reference_emotion(values):
    x = list(map(float, values)); means = [math.fsum(x[k-6:k+1]) / 7 for k in range(6, 13)]
    diff = [v - mean for v, mean in zip(x[6:], means, strict=True)]
    signal = (1 if all(v >= 0 for v in diff[-3:]) else 0) if diff[-1] >= 0 else (-1 if all(v < 0 for v in diff[-6:]) else 0)
    rate = round((x[-1] / means[-1] - 1) * 100, 2) if means[-1] > 0 else math.nan
    return signal, rate, means


def compute(directory):
    validate_sources(directory); frame, sessions = inputs(directory); score, _, hashes = kernels(directory)
    require(talib.get_compatibility() == 0, 'TA-Lib settings differ')
    session_positions = {day: k for k, day in enumerate(sessions)}; digest = hashlib.sha256(); boundaries = []; errors = np.zeros(3)
    counts = {}; count = skipped = 0; mean_error = 0.
    for k in range(13, len(frame)):
        rows = frame.iloc[k-13:k+1]
        if session_positions[rows.date.iloc[-1]] - session_positions[rows.date.iloc[0]] != 13: skipped += 1; continue
        prices = rows.close.iloc[:-1].to_numpy(); current = float(rows.close.iloc[-1]); actual = score_case(score, prices, current)
        mean = math.fsum(prices) / 13; expected = [(current/prices[0]-1)*100, mean, (current/mean-1)*100]
        errors = np.maximum(errors, np.abs(np.asarray(actual)-expected)/np.maximum(np.abs(expected), 1.))
        volumes = rows.volume.iloc[:-1].to_numpy(dtype=float); signal, rate, _ = emotion_case(directory, volumes)
        expected_signal, expected_rate, means = reference_emotion(volumes); ma = talib.MA(volumes, 7)
        mean_error = max(mean_error, float(np.max(np.abs(ma[6:]-means)/np.maximum(np.abs(means), 1.))))
        comparisons = (actual[0] >= .1 and actual[2] >= 0, expected[0] >= .1 and expected[2] >= 0)
        if signal != expected_signal or rate != expected_rate or comparisons[0] != comparisons[1]:
            boundaries.append({'date': rows.date.iloc[-1], 'signal': signal, 'reference_signal': expected_signal,
                'rate': rate, 'reference_rate': expected_rate, 'score_accepted': comparisons[0], 'reference_score_accepted': comparisons[1]})
        digest.update(json.dumps({'date': rows.date.iloc[-1], 'scores': actual, 'signal': signal, 'rate': rate}, sort_keys=True).encode())
        counts[str(signal)] = counts.get(str(signal), 0) + 1; count += 1
    require(errors.max() < 1e-10 and mean_error < 1e-10, 'Operator numeric mismatch')
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'original_strategy_complete': False, 'strategy_results': [],
        'input_sha256': {n: file_sha(directory / n) for n in ('index-input.parquet', 'calendar-input.parquet')},
        'backend': {'talib': talib.__version__, 'ta_version': talib.__ta_version__.decode(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'compatibility': talib.get_compatibility(), 'MA_type': 'default SMA/0', 'SMA_unstable': 'not applicable'},
        'shared_ast_sha256': hashes, 'windows': count, 'calendar_gap_windows_rejected': skipped,
        'first_decision_sample': frame.date.iloc[13], 'last_decision_sample': frame.date.iloc[-1],
        'emotion_counts': counts, 'output_sha256': digest.hexdigest(), 'boundaries': boundaries,
        'max_score_relative_differences': errors.tolist(), 'max_ma7_relative_difference': mean_error,
        'limits': ['HS300 score uses next completed close as numeric sample; not original ETF/live11:30 signal',
            'HS300 volume is not original dynamic weakest-index selection; ndarray explicitly adapts original position indices',
            'No orders/fills/NAV/costs computed; source compatibility failures remain unrepaired']}


def suffix_case(directory, rows, emotion=0, stale=True):
    _, suffix, _ = kernels(directory); g = params(directory); g.buy = ['stale'] if stale else []
    ns = {'g': g, 'log': SimpleNamespace(info=lambda *a: None), 'EmotionMonitor': lambda: emotion}
    exec(suffix, ns); frame = pd.DataFrame(rows, columns=['基金代码', '对应指数', '周期涨幅', '均线差值'])
    ns['diagnostic_suffix'](None, frame)
    return {'signal': g.signal, 'buy': g.buy, 'target_market': g.target_market}


def trade_case(directory, signal, buys, positions, total=10000., accept_sell=False):
    g = params(directory); g.signal = signal; g.buy = buys; calls = []
    # A hypothetical snapshot-iterating platform mapping permits immediate accepted-sale diagnostics.
    class SnapshotPositions(dict):
        def __iter__(self): return iter(tuple(self.keys()))
    cls = SnapshotPositions if accept_sell else dict
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=cls({s: SimpleNamespace(value=v) for s, v in positions.items()}),
        total_value=total, available_cash=1000., returns=0.))
    def sell(s, v):
        calls.append(['target', s, v])
        if accept_sell: ctx.portfolio.positions.pop(s, None)
    ns = {'g': g, 'log': SimpleNamespace(info=lambda *a: None), 'order_target': sell,
        'order_target_value': lambda *a: calls.append(['value', *a])}
    selected(directory, 1, ('ETFtrade',), ns)
    ns['ETFtrade'](ctx); return calls


def diagnostics(directory):
    cases = []; maps = [params(directory, k).ETF_targets for k in (1, 2)]
    cases.append({'case': 'distinct_active_mappings', 'maps': maps})
    rows = [('strong', 'strong_index', .1, 0.), ('weak', 'weak_index', -.2, -1.)]
    for name, values, emotion in [('exact_thresholds', rows, 0), ('weakest_emotion_clear', rows, -1), ('empty_stale', [], 0),
        ('below_return', [('a', 'idx', .099, 1.)], 0), ('below_mean', [('a', 'idx', 1., -.001)], 0)]:
        cases.append({'case': name, **suffix_case(directory, values, emotion)})
    g = params(directory); g.ETFList = {'index': 'fund'}
    ns = {'g': g, 'pd': pd, 'log': SimpleNamespace(info=lambda *a: None),
        'get_current_data': lambda: {'fund': SimpleNamespace(last_price=1.)},
        'attribute_history': lambda *a, **kw: {'close': np.ones(13)}}
    selected(directory, 1, ('get_signal',), ns)
    try: ns['get_signal'](None)
    except AttributeError as exc: cases.append({'case': 'original_append_error', 'error': str(exc)})
    else: raise AssertionError('Original append failure disappeared')
    try: emotion_case(directory, np.ones(13), dated=True)
    except KeyError as exc: cases.append({'case': 'original_dated_negative_index_error', 'error': str(exc)})
    else: raise AssertionError('Original negative-index failure disappeared')
    date_ns = {'pd': pd, 'get_all_trade_days': lambda: list(pd.date_range('2020-01-01', periods=20).date)}
    selected(directory, 1, ('get_before_after_trade_days',), date_ns)
    try: date_ns['get_before_after_trade_days']('2020-01-20', 13)
    except NameError as exc: cases.append({'case': 'original_datetime_injection_missing', 'error': str(exc)})
    else: raise AssertionError('Undocumented platform datetime injection supplied')
    try: trade_case(directory, 'BUY', [], {})
    except ZeroDivisionError as exc: cases.append({'case': 'initial_buy_empty_targets_error', 'error': str(exc)})
    else: raise AssertionError('Original division by zero disappeared')
    for name, signal, buys, positions, accept in [('clear_rejected', 'CLEAR', [], {'old': 9000.}, False),
        ('clear_accepted', 'CLEAR', [], {'old': 9000.}, True), ('rebalance_equal5000', 'BUY', ['a'], {'a': 5000.}, False),
        ('rebalance_over5000', 'BUY', ['a'], {'a': 4999.}, False)]:
        cases.append({'case': name, 'orders': trade_case(directory, signal, buys, positions, accept_sell=accept)})
    for name, values in [('constant_volume', np.ones(13)), ('decreasing_volume', np.arange(13., 0., -1)), ('zero_volume', np.zeros(13))]:
        signal, rate, _ = emotion_case(directory, values); cases.append({'case': name, 'signal': signal, 'rate': rate if math.isfinite(rate) else None})
    bank = {'g': SimpleNamespace(), 'pd': pd}
    names = ('check_stocks_initialize', 'buy_initialize', 'check_stocks_sort_initialize', 'check_stocks', 'check_stocks_sort',
        'get_check_stocks_sort_input_dict', 'holded_filter', 'industry_filter', 'sell_every_day', 'financial_statements_filter')
    selected(directory, 0, names, bank)
    for name in ('check_stocks_initialize', 'buy_initialize', 'check_stocks_sort_initialize'): bank[name]()
    g = bank['g']; g.check_stocks_days = 0; g.check_stocks_refresh_rate = 300; calls = []
    bank.update(get_security_universe=lambda *a: calls.append(g.check_stocks_days) or ['candidate'],
        get_check_stocks_sort_input_dict=lambda: {}, check_stocks_sort=lambda ctx, rows, *a: rows)
    for name in ('industry_filter', 'concept_filter', 'st_filter', 'delisted_filter', 'financial_statements_filter', 'situation_filter',
        'technical_indicators_filter', 'pattern_recognition_filter', 'other_func_filter'): bank[name] = lambda ctx, rows, *a: rows
    refresh = []
    for k in range(601):
        before = len(calls); bank['check_stocks'](None)
        if len(calls) != before: refresh.append(k)
    cases.append({'case': 'bank_300_callback_clock', 'refresh_indices': refresh, 'industry_list': g.industry_list,
        'allocation': [g.order_style_str, g.order_style_value]})
    selected(directory, 0, ('industry_filter', 'financial_statements_filter', 'get_check_stocks_sort_input_dict', 'check_stocks_sort'), bank)
    bank['get_industry_stocks'] = lambda code: ['bank'] if code == '801780' else ['nonbank']
    cases.append({'case': 'bank_industry_union', 'stocks': bank['industry_filter'](None, ['bank', 'nonbank', 'outside'], g.industry_list)})
    bank.update(valuation=SimpleNamespace(pe_ratio='PE', market_cap='cap'), indicator=SimpleNamespace(roe='ROE'))
    cases.append({'case': 'bank_sort_configuration', 'fields': {name: list(value) for name, value in bank['get_check_stocks_sort_input_dict']().items()}})
    try: bank['financial_statements_filter'](None, ['a'])
    except NameError as exc: cases.append({'case': 'bank_missing_finance_wizard', 'error': str(exc)})
    else: raise AssertionError('Unknown wizard financial helper supplied')
    bank['get_sort_dataframe'] = lambda rows, field, config: pd.DataFrame({field: [1.]}, index=rows)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        try: result = bank['check_stocks_sort'](None, ['a'], {'PE': ('asc', .1)})
        except TypeError as exc: row = {'case': 'bank_sum_compatibility', 'error': str(exc)}
        else: row = {'case': 'bank_sum_compatibility', 'result': result}
    row['warnings'] = [{'category': type(w.message).__name__, 'message': str(w.message)} for w in observed]; cases.append(row)
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'held': object()})); g.open_sell_securities = ['gone', 'held', 'held']; orders = []
    bank['order_target_value'] = lambda *a: orders.append(list(a)); bank['sell_every_day'](ctx)
    cases.append({'case': 'bank_rejected_retry', 'orders': orders, 'pending': g.open_sell_securities})
    bank['order_target_value'] = lambda *a: ctx.portfolio.positions.pop(a[0], None); bank['sell_every_day'](ctx)
    cases.append({'case': 'bank_accepted_retry', 'pending': g.open_sell_securities})
    return {'not_a_backtest': True, 'cases': cases,
        'limits': ['Original fragments and synthetic stub order intentions only; no fill/cash/NAV implementation',
            'Accepted CLEAR sales assume hypothetical snapshot-iterating positions; platform mapping equivalence unproved',
            'Suffix after append is explicitly diagnostic, not successful original get_signal execution']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch33 long score/volume operator samples and original source defects/state diagnostics frozen')


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
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch33_dependency_probe', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch33', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch33 original index/industry/classified-fund supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch33 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = previous.offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    previous.validate_catalog(root, directory); return archive(root, 'Batch33 forbidden-network operator/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch33.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch33_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch32_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); previous.validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
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
    binding(root, directory); previous.validate_catalog(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch33 bank and distinct ETF components accepted; missing original trading dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch33-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch33/20261006-bank-etf-rotation')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
