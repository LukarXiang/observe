"""Study exact ATR and northbound arithmetic; missing wizard/minute inputs stay blocked."""
import argparse
import ast
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
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch28 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch28/20261006-rsrs-mesa-northbound')
RECEIPT = Path('docs/handoff/2026-10-06-batch28-verification.json')
SOURCES = ('2020年度精选策略/37 个股止损.txt', '2020年度精选策略/38 一个简单的跟踪聪明钱策略.txt',
    '2020年度精选策略/73 跟着港资（北向资金）买A股.txt', '2021年度精选策略/29.北上资金（北向资金港资外资）因子分析与策略分享.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_hsgt_em.py'), ('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch29.py', 'tests/unit/test_batch29_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'input-analysis.json', 'price-input.parquet',
    'component-research.json', 'diagnostics.json', 'existing-apis.json', 'probe-results.json', 'component-offline.json', 'offline-verification.json',
    'clip-order-diagnostics.json', 'source-review-supplement.json'}
CLIP_GAP = '逐列独立3σ缩尾可使原有效OHLC倒置；十个真实窗口操作数已证实，原ATR照算不等于仓位逻辑有效'
QUERIES = {'holdings': {'function': 'stock_hsgt_stock_statistics_em',
    'parameters': {'symbol': '北向持股', 'start_date': '20200601', 'end_date': '20200601'},
    'limit': 'One-date final vendor holdings sample; not STK_EL_TOP_ACTIVATE buy/sell or proven platform share_number vintage'},
    'minute_stock': {'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600519', 'period': '1',
        'start_date': '2020-06-01 09:30:00', 'end_date': '2020-06-01 15:00:00', 'adjust': ''},
        'limit': 'One-minute endpoint requests latest five days then filters dates, not original 230-row history'}}
SEARCH = ['rg', '-n', r'^def security_stopprofit|^def order_style|^def judge_security_max_proportion|^def max_buy_value_or_amount',
    'repo', '-g', '*.py', '-g', '*.txt']


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted receipt differs')
    path = ACCEPTED / 'price-input.parquet'
    require(file_sha(path) == receipt['checks']['evidence_sha256']['price-input.parquet'], 'Accepted prices changed')
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'price_file': str(path), 'price_sha256': file_sha(path), 'upstream_binding': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Batch29 binding changed')
    return result


def api_evidence():
    import akshare as ak
    rows = []
    for name in sorted({q['function'] for q in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'function': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'version': ak.__version__, 'apis': rows, 'queries': QUERIES,
        'missing_modules': [n for n in ('kuanke', 'jqdatasdk', 'jqdata') if importlib.util.find_spec(n) is None],
        'not_top_active_api': True}


def start(root, directory):
    checkpoint(root, directory, 'Batch29 stop/smart-money/northbound complete source research started')
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
            'use': 'Wilder true-range/smoothing comparison; vendor holding schema and recent-only minute restriction, no strategy substitution'})
    search = subprocess.run(SEARCH, capture_output=True, text=True); require(search.returncode in (0, 1), search.stderr)
    tick = Path('repo/量化策略源代码/2022年度精选策略/9.高频Tick频率策略分享.txt'); copied = folder / 'tick9-not-wizard.source'
    shutil.copyfile(tick, copied)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': references, 'catalog': catalog,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path), 'snapshot': SNAPSHOT,
        'dependency_search': {'command': SEARCH, 'returncode': search.returncode, 'matches': search.stdout.splitlines()},
        'non_equivalent_tick_function': {'file': str(tick), 'copy': str(copied), 'sha256': file_sha(tick),
            'finding': 'Two-argument security_stopprofit and ten-times futures profit; not three-argument wizard implementation'},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch29 four complete source reviews frozen; core wizard/top-active/minute dependencies blocked')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    supplement = validate_supplement(directory) if (directory / 'source-review-supplement.json').exists() else None
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        allowed = [REVIEWS[name]]
        if name == SOURCES[1]:
            before = json.loads(json.dumps(REVIEWS[name])); before['gaps'].remove(CLIP_GAP)
            allowed = [supplement['before']] if supplement else [REVIEWS[name], before]
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] in allowed and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'References missing')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')
    row = doc['non_equivalent_tick_function']
    require(file_sha(row['file']) == row['sha256'] == file_sha(row['copy']) and doc['dependency_search']['command'] == SEARCH, 'Dependency evidence changed')


def validate_supplement(directory):
    doc = read(directory / 'source-review-supplement.json'); review = read(directory / 'source-reviews/review.json')['sources'][1]
    expected = json.loads(json.dumps(doc['before']))
    if CLIP_GAP not in expected['gaps']: expected['gaps'].append(CLIP_GAP)
    require(doc['source_sha256'] == review['source_sha256'] and doc['before'] == review['review'] and
        doc['after'] == expected == REVIEWS[SOURCES[1]] and
        doc['review_sha256'] == file_sha(directory / 'source-reviews/review.json') and
        doc['component_sha256'] == file_sha(directory / 'component-research.json') and
        doc['clip_sha256'] == file_sha(directory / 'clip-order-diagnostics.json'), 'Review supplement differs')
    return doc


def clarify(root, directory):
    review = read(directory / 'source-reviews/review.json')['sources'][1]
    after = json.loads(json.dumps(review['review']))
    if CLIP_GAP not in after['gaps']: after['gaps'].append(CLIP_GAP)
    require(after == REVIEWS[SOURCES[1]], 'Unexpected review amendment')
    save(directory / 'source-review-supplement.json', {'source_sha256': review['source_sha256'], 'before': review['review'], 'after': after,
        'review_sha256': file_sha(directory / 'source-reviews/review.json'), 'component_sha256': file_sha(directory / 'component-research.json'),
        'clip_sha256': file_sha(directory / 'clip-order-diagnostics.json')})
    validate_supplement(directory); catalog = catalog_strategies(root, 'repo/量化策略源代码')
    return {'catalog': catalog, 'progress': archive(root, 'Batch29 exact-source independent clipping limitation added with immutable review supplement')}


def atomic_expression(directory, number, prefix):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    lines = [line.strip() for line in read_source(Path(row['source_copy']))[0].splitlines() if line.strip().startswith(prefix)]
    require(len(lines) == 1, 'Missing/ambiguous original assignment')
    module = ast.parse(lines[0]); require(len(module.body) == 1 and isinstance(module.body[0], ast.Assign), 'Unexpected original statement')
    node = module.body[0].value
    return compile(ast.Expression(node), '<frozen-original-assignment>', 'eval'), hashlib.sha256(ast.dump(node).encode()).hexdigest()


def prepare(root, directory):
    bound = binding(root, directory)
    require(not any((directory / n).exists() for n in ('price-input.parquet', 'input-analysis.json')), 'Prepared archive exists')
    path = directory / 'price-input.parquet'; shutil.copyfile(bound['price_file'], path); frame = pd.read_parquet(path)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
        'first': frame.date.min(), 'last': frame.date.max(), 'pool': list(previous.prior.INSTRUMENTS), 'not_a_backtest': True,
        'price_policy': 'Accepted close/high/low back-adjusted windows anchored to verified current raw price, known pauses only excluded',
        'limits': ['Ten stocks are not original index pool', 'No one-minute/holding/top-active samples accepted as strategy inputs']})
    return archive(root, 'Batch29 accepted ten-stock long price input frozen for exact ATR component')


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); path = directory / 'price-input.parquet'
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and doc['pool'] == list(previous.prior.INSTRUMENTS), 'Input scope differs')
    require(str(path) == doc['file'] and file_sha(path) == doc['sha256'], 'Component input changed'); frame = pd.read_parquet(path)
    require(len(frame) == doc['rows'] and frame.date.min() == doc['first'] and frame.date.max() == doc['last'] and
        not frame.duplicated(['date', 'instrument']).any(), 'Input profile differs'); return frame


def reference_clip(values):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) > 1 and np.isfinite(x).all(), 'Invalid clip input')
    mean = math.fsum(map(float, x)) / len(x); sd = math.sqrt(math.fsum((float(v) - mean) ** 2 for v in x) / len(x))
    return np.array([min(max(float(v), mean - 3 * sd), mean + 3 * sd) for v in x])


def reference_atr(high, low, close, period=14):
    h, l, c = [np.asarray(x, dtype=float) for x in (high, low, close)]
    require(h.ndim == l.ndim == c.ndim == 1 and len(h) == len(l) == len(c) and len(c) > period and
        all(np.isfinite(x).all() for x in (h, l, c)), 'Invalid ATR input')
    tr = [max(float(h[k] - l[k]), abs(float(h[k] - c[k - 1])), abs(float(l[k] - c[k - 1]))) for k in range(1, len(c))]
    value = math.fsum(tr[:period]) / period
    for v in tr[period:]: value = (value * (period - 1) + v) / period
    return value


def atr_kernel(directory):
    holder = {}; ns = {'np': np, 'talib': talib, 'log': SimpleNamespace(info=lambda *a: None),
        'attribute_history': lambda *a, **kw: {n: v.copy() for n, v in holder['arrays'].items()}}
    sha = selected(directory, 1, ('fun_normalizeData', 'fun_getATR'), ns)
    return holder, ns, sha


def compute(directory):
    validate_sources(directory); frame = inputs(directory); holder, ns, sha = atr_kernel(directory)
    require(talib.get_compatibility() == 0 and talib.get_unstable_period('ATR') == 0, 'ATR engine settings differ')
    groups = []; paused = 0
    for instrument, prices in frame.groupby('instrument', sort=True):
        prices = prices.sort_values('date'); paused += int((~prices.is_trading).sum()); trading = prices[prices.is_trading]
        require(trading.adjustment_status.eq('usable').all() and np.isfinite(trading[['close_adj', 'high_adj', 'low_adj', 'back_factor']]).all().all() and
            trading.back_factor.gt(0).all(), 'Unknown trading input cannot be skipped')
        raw_values = trading[['close_adj', 'high_adj', 'low_adj']].to_numpy(); factors = trading.back_factor.to_numpy()
        actual_values = []; clip_sha = hashlib.sha256(); max_clip = max_atr = 0.; clipped = irregular = 0
        for k in range(23, len(trading)):
            array = raw_values[k - 23:k + 1] / factors[k]
            holder['arrays'] = {n: array[:, j].copy() for j, n in enumerate(('close', 'high', 'low'))}
            actual = float(ns['fun_getATR'](instrument))
            require(math.isfinite(actual) and actual >= 0, 'Invalid original ATR output')
            original_clip = np.column_stack([ns['fun_normalizeData'](array[:, j].copy()) for j in range(3)])
            expected_clip = np.column_stack([reference_clip(array[:, j]) for j in range(3)])
            max_clip = max(max_clip, float(np.max(np.abs(original_clip - expected_clip) / np.maximum(1., np.abs(expected_clip)))))
            expected = reference_atr(expected_clip[:, 1], expected_clip[:, 2], expected_clip[:, 0])
            max_atr = max(max_atr, abs(actual - expected) / max(1., abs(expected)))
            clipped += int(np.count_nonzero(original_clip != array))
            irregular += int(np.count_nonzero((original_clip[:, 1] < original_clip[:, 0]) | (original_clip[:, 0] < original_clip[:, 2])))
            clip_sha.update(original_clip.astype('<f8').tobytes()); actual_values.append(actual)
        require(max_clip < 1e-10 and max_atr < 1e-10, 'Original ATR/clip arithmetic differs')
        groups.append({'instrument': instrument, 'windows': len(actual_values), 'clip_max_relative_difference': max_clip,
            'atr_max_relative_difference': max_atr, 'clipped_operands': clipped, 'ohlc_inversions_after_independent_clipping': irregular,
            'clip_sha256': clip_sha.hexdigest(), 'atr_sha256': hashlib.sha256(np.array(actual_values, dtype='<f8').tobytes()).hexdigest()})
    return {'not_a_backtest': True, 'original_strategy_complete': False, 'snapshot': SNAPSHOT, 'input_sha256': file_sha(directory / 'price-input.parquet'),
        'source_ast_sha256': sha, 'talib': {'python': talib.__version__, 'c_core': talib.__ta_version__.decode(), 'compatibility': 0, 'ATR_unstable': 0},
        'history': 24, 'atr_period': 14, 'known_paused_rows_excluded': paused, 'windows': sum(r['windows'] for r in groups), 'groups': groups,
        'scope': 'Exact source daily normalization/TA-Lib ATR arithmetic on verified histories; no smart-money minute ranking or portfolio/NAV'}


class QueryField:
    def in_(self, values): return values


def holding_kernel(directory):
    holder = {}; field = QueryField(); table = SimpleNamespace(code=field, share_number=field, day=field)
    ns = {'pd': pd, 'finance': SimpleNamespace(STK_HK_HOLD_INFO=table, run_query=lambda *a: holder['shares'].copy()),
        'query': lambda *a: SimpleNamespace(filter=lambda *a: None), 'get_price': lambda *a, **kw: {'close': holder['close'].copy()}}
    sha = selected(directory, 3, ('get_factor_data',), ns)
    return holder, ns['get_factor_data'], sha


def diagnostics(directory):
    cases = []; g = SimpleNamespace(filter_holded=False, selled_security_list={}, daily_risk_management=False); calls = []
    ns = {'g': g, 'get_index_stocks': lambda name: {'a': ['c', 'a'], 'b': ['b', 'a']}[name],
        'security_stopprofit': lambda *args: calls.append(args[1:])}
    pool_sha = selected(directory, 0, ('get_security_universe', 'check_stocks_sort', 'holded_filter', 'risk_management', 'selled_security_list_count'), ns)
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={'b': object()}))
    pool = ns['get_security_universe'](context, ['a', 'b'], [])
    cases.append({'case': 'empty_sort_preserves_lexical_pool', 'pool': ns['check_stocks_sort'](context, pool, {})})
    cases.append({'case': 'user_code_expands_characters', 'pool': ns['get_security_universe'](context, [], ['600519.XSHG']), 'inactive_default': True})
    cases.append({'case': 'false_hold_flag_filters_existing', 'pool': ns['holded_filter'](context, pool)})
    g.open_sell_securities = []; ns['risk_management'](context)
    cases.append({'case': 'unresolved_wizard_call', 'threshold': calls[0][0], 'third_argument': calls[0][1], 'callee_not_executed': True})
    g.selled_security_list = {'a': 0}; ns['selled_security_list_count'](context)
    cases.append({'case': 'daily_reset', 'daily_risk_management': g.daily_risk_management, 'sell_days': g.selled_security_list})
    ns = {'datetime': SimpleNamespace(date=SimpleNamespace(today=lambda: datetime.date(2026, 10, 6))),
        'timedelta': datetime.timedelta, 'get_security_info': lambda *a: SimpleNamespace(start_date=datetime.date(2026, 1, 1))}
    date_sha = selected(directory, 1, ('filter_new_and_sub_new',), ns)
    cases.append({'case': 'wall_clock_and_ignored_days', 'at_2020_backtest_but_2026_clock': ns['filter_new_and_sub_new'](['a'], days=9999)})
    ns = {'pd': pd, 'np': np, 'math': math, 'nan': np.nan, 'attribute_history': lambda *a, **kw: pd.DataFrame({'open': [10.], 'close': [11.], 'volume': [100.]})}
    smart_sha = selected(directory, 1, ('get_smart_money_factor',), ns)
    try: ns['get_smart_money_factor']('a')
    except AttributeError as exc:
        require('sort' in str(exc), 'Unexpected smart diagnostic'); cases.append({'case': 'smart_legacy_sort', 'error': type(exc).__name__})
    else: raise AssertionError('Legacy smart sort defect not reproduced')
    ns = {'g': SimpleNamespace(risk_ratio=.1)}; position_sha = selected(directory, 1, ('calPosition',), ns)
    try: ns['calPosition'](SimpleNamespace(portfolio=SimpleNamespace(total_value=100000.)), [])
    except ZeroDivisionError: cases.append({'case': 'empty_risk_pool', 'error': 'ZeroDivisionError'})
    else: raise AssertionError('Empty risk division not reproduced')
    ns = {'attribute_history': lambda *a, **kw: pd.DataFrame({'close': [10.]}, index=pd.date_range('2020-01-01', periods=1))}
    selected(directory, 1, ('get_close_price',), ns)
    try: ns['get_close_price']('a', 1)
    except KeyError: cases.append({'case': 'date_close_zero_index', 'error': 'KeyError'})
    else: raise AssertionError('Original date close index defect not reproduced')
    ns = {'get_close_price': lambda *a: 10.}; selected(directory, 1, ('get_growth_rate',), ns)
    try: ns['get_growth_rate']('a')
    except NameError as exc:
        require('isnan' in str(exc), 'Unexpected growth diagnostic'); cases.append({'case': 'unprovided_isnan', 'error': 'NameError'})
    else: raise AssertionError('Undefined isnan dependency not reproduced')
    net_code, net_sha = atomic_expression(directory, 2, "df['net'] =")
    filter_code, filter_sha = atomic_expression(directory, 2, 'df = df[(df.link_id')
    df = pd.DataFrame({'code': ['south', 'a', 'b'], 'buy': [1000., 10., 20.], 'sell': [0., 30., 50.], 'link_id': [310003, 310001, 310002]})
    df['net'] = eval(net_code, {}, {'df': df}); df = df.sort_values('net', ascending=False)
    df = eval(filter_code, {}, {'df': df})
    cases.append({'case': 'negative_northbound_top_is_still_selected', 'selection': df.code.iloc[:1].tolist(), 'net': float(df.net.iloc[0]), 'arithmetic_only': True})
    sort_code, _ = atomic_expression(directory, 2, 'df = df.sort(columns')
    try: eval(sort_code, {}, {'df': df})
    except AttributeError: cases.append({'case': 'top_active_legacy_sort', 'error': 'AttributeError'})
    else: raise AssertionError('Legacy top-active sort defect not reproduced')
    days = [datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)]; ns = {'get_all_trade_days': lambda: days}
    selected(directory, 2, ('shifttradingday',), ns)
    cases.append({'case': 'first_calendar_negative_wrap', 'result': ns['shifttradingday'](days[0], -1).isoformat()})
    holder, fn, holding_sha = holding_kernel(directory); day = datetime.date(2020, 6, 1)
    holder['shares'] = pd.DataFrame({'code': ['a', 'b'], 'share_number': [10., 20.]})
    holder['close'] = pd.DataFrame({'a': [5.], 'b': [2.]}, index=[day]); result = fn(['a', 'b'], day)
    cases.append({'case': 'holding_number_times_price', 'values': {n: float(result[n].iloc[0]) for n in result}})
    holder['shares'] = pd.DataFrame({'code': [], 'share_number': []}); result = fn(['a'], day)
    cases.append({'case': 'empty_holding_query', 'empty': result.empty})
    holder['shares'] = pd.DataFrame({'code': ['a'], 'share_number': [10.]}); holder['close'] = pd.DataFrame({'b': [2.]}, index=[day])
    result = fn(['a'], day); cases.append({'case': 'mismatched_columns_nonempty_nan', 'empty': result.empty, 'all_missing': bool(result.isna().all().all())})
    holder['shares'] = pd.DataFrame({'code': ['a', 'a'], 'share_number': [10., 20.]}); holder['close'] = pd.DataFrame({'a': [5.]}, index=[day])
    try:
        result = fn(['a'], day); cases.append({'case': 'duplicate_holding_codes', 'duplicate_columns': bool(result.columns.duplicated().any()), 'values': result.values.tolist()})
    except ValueError as exc: cases.append({'case': 'duplicate_holding_codes', 'error': type(exc).__name__})
    rank_code, _ = atomic_expression(directory, 3, 'stock_list = factor_data.T.sort_index')
    try: eval(rank_code, {}, {'factor_data': pd.DataFrame({'a': [50.]}, index=[day]), 'context': SimpleNamespace(previous_date=day), 'g': SimpleNamespace(buy_stock_count=10)})
    except TypeError: cases.append({'case': 'holding_rank_legacy_by', 'error': 'TypeError'})
    else: raise AssertionError('Legacy holding sort_index defect not reproduced')
    return {'synthetic_only': True, 'not_a_backtest': True, 'source_ast_sha256': {'pool': pool_sha, 'clock': date_sha, 'smart': smart_sha,
        'risk_budget': position_sha, 'net': net_sha, 'market_filter': filter_sha, 'holding_factor': holding_sha}, 'cases': cases}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study implementation changed')
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch29 exact long ATR arithmetic and synthetic source defects archived without trading')


def clip_order_diagnostics(directory):
    validate_sources(directory); frame = inputs(directory); component = read(directory / 'component-research.json')
    holder, ns, sha = atr_kernel(directory); cases = []
    groups = [r for r in component['groups'] if r['ohlc_inversions_after_independent_clipping']]
    for group in groups:
        instrument = group['instrument']; trading = frame[frame.instrument.eq(instrument) & frame.is_trading].sort_values('date')
        x = trading[['close_adj', 'high_adj', 'low_adj']].to_numpy(); factors = trading.back_factor.to_numpy(); dates = trading.date.to_numpy()
        for k in range(23, len(trading)):
            array = x[k - 23:k + 1] / factors[k]
            clipped = np.column_stack([ns['fun_normalizeData'](array[:, j].copy()) for j in range(3)])
            mask = (clipped[:, 1] < clipped[:, 0]) | (clipped[:, 0] < clipped[:, 2])
            if not mask.any(): continue
            expected = np.column_stack([reference_clip(array[:, j]) for j in range(3)])
            reference_mask = (expected[:, 1] < expected[:, 0]) | (expected[:, 0] < expected[:, 2])
            require(np.array_equal(mask, reference_mask) and np.allclose(clipped, expected, rtol=1e-12, atol=1e-12), 'Clip inversion reference differs')
            require(((array[:, 1] >= array[:, 0]) & (array[:, 0] >= array[:, 2])).all(), 'Raw OHLC inversion is not clipping evidence')
            holder['arrays'] = {n: array[:, j].copy() for j, n in enumerate(('close', 'high', 'low'))}
            original_atr = float(ns['fun_getATR'](instrument)); raw_atr = float(talib.ATR(array[:, 1], array[:, 2], array[:, 0], timeperiod=14)[-1])
            for pos in np.flatnonzero(mask):
                cases.append({'instrument': instrument, 'window_end': str(dates[k]), 'operand_date': str(dates[k - 23 + pos]),
                    'columns': ['close', 'high', 'low'], 'raw': array[pos].tolist(), 'original_clipped': clipped[pos].tolist(),
                    'reference_clipped': expected[pos].tolist(), 'raw_ohlc_valid': True, 'source_atr': original_atr, 'unclipped_atr': raw_atr,
                    'window_dates': dates[k - 23:k + 1].tolist(), 'raw_window': array.tolist(), 'clipped_window': clipped.tolist()})
    require(len(cases) == sum(r['ohlc_inversions_after_independent_clipping'] for r in component['groups']), 'Clip case count differs')
    return {'not_a_backtest': True, 'fix_applied': False, 'source_ast_sha256': sha,
        'input_sha256': file_sha(directory / 'price-input.parquet'), 'component_sha256': file_sha(directory / 'component-research.json'),
        'cause': 'Independent per-field mean +/-3 population std clipping does not preserve cross-field OHLC order',
        'count': len(cases), 'cases': cases, 'limits': ['Unclipped ATR is a numeric diagnostic, not a replacement strategy',
            'Cases count window operands, not distinct trading signals; original source behavior is retained']}


def boundaries(root, directory):
    binding(root, directory); before = implementation(); result = clip_order_diagnostics(directory)
    require(before == implementation(), 'Clip diagnostic implementation drift')
    save(directory / 'clip-order-diagnostics.json', result)
    return archive(root, 'Batch29 source independent clipping creates ten OHLC inversions; full window evidence retained without repair')


def worker(root, directory, endpoint):
    import akshare as ak
    require(read(directory / 'existing-apis.json') == api_evidence(), 'Probe API changed')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'),
        'status': 'failed', 'published': False, 'files': [], 'wire': []}; original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire']) < 10, 'Probe response cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status_code': response.status_code})
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, query['function'])(**query['parameters'])
        path = raw.save(root, 'batch29_dependency_probe', endpoint, directory.name, frame)
        row.update(status='success' if len(frame) else 'empty', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}])
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(read(directory / 'existing-apis.json') == api_evidence(), 'API evidence changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['published'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['status'] in ('failed', 'timeout', 'empty', 'success'), 'Probe binding differs')
        for item in row['wire']: require(file_sha(item['file']) == item['sha256'], 'Probe wire changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw missing'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and
                bool(len(frame)) == (row['status'] == 'success'), 'Probe raw changed')
        else: require(not row['files'] and row.get('error'), 'Failed probe accepted raw/no error')
        rows.append(row)
    return rows


def probe(root, directory):
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch29', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists():
                    save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query, 'status': 'timeout', 'files': [], 'wire': [],
                        'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'error': 'Parent deadline45s; partial unaccepted wire files may remain'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch29 bounded share-number/minute supplementation archived with original semantic limits')


def offline(root, directory):
    before = implementation(); binding(root, directory)
    def denied(*a, **kw): raise AssertionError('Batch29 offline attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); clip_diagnostic = clip_order_diagnostics(directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json') and
        clip_diagnostic == read(directory / 'clip-order-diagnostics.json'), 'Offline implementation/diagnostics differ')
    save(directory / 'component-offline.json', result)
    require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline bytes differ')
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'sha256': file_sha(directory / 'component-offline.json'), 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch29 forbidden-network exact ATR/diagnostics match byte-for-byte')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch29.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch29_research.py',
            'tests/unit/test_batch28_research.py', 'tests/unit/test_batch27_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_signal_slots.py'], ['git', 'diff', '--check']]


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
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json') and
        clip_order_diagnostics(directory) == read(directory / 'clip-order-diagnostics.json'), 'Recomputed component differs')
    protection = protect(root, directory); progress = archive(root, 'Batch29 four complete reviews/long ATR/probes/offline accepted; original trading gaps retained')
    save(Path('docs/handoff/2026-10-06-batch29-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'source_review_supplement': validate_supplement(directory),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'protection': protection,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'prepare', 'study', 'boundaries', 'clarify', 'probe', 'worker', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch29/20261006-wizard-smart-northbound')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
