"""Freeze gold-stock source components and missing recommendation dependencies."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
import hashlib
import inspect
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from scipy import stats

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch57 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch57/20261007-futures-dependency')
RECEIPT = Path('docs/handoff/2026-10-07-batch57-verification.json')
PRICE = Path('data/staging/strategies-batch47/20261007-candidate-momentum/price-input.parquet')
SOURCES = ('2023年度精选策略/55.【分享】券商金股组合增强.txt',
    '2024年度精选策略2/100.【复现】金股组合增强—分析师推荐概率.txt')
SOURCE_SHA = ('6c7edf0ef4bcee88cd335ade773e2edaf5aaf8617300d6b6c22804066a9aeacd',
    '55b4a191313db44f97129d5a0d72f2c38063e32d095115c2df76a37c73866c16')
REFERENCES = (('repo/akshare', 'akshare/stock_fundamental/stock_recommend.py'),
    ('repo/akshare', 'akshare/stock/stock_zh_a_sina.py'),
    ('repo/skfolio', 'src/skfolio/optimization/convex/_mean_risk.py'))
QUERIES = {'recommend': ('stock_institute_recommend', {'symbol': '最新投资评级'}),
    'rating_history': ('stock_institute_recommend_detail', {'symbol': '002230'}),
    'minute30': ('stock_zh_a_minute', {'symbol': 'sz002230', 'period': '30', 'adjust': ''})}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch58.py', 'tests/unit/test_batch58_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'missing-inputs.json', 'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'price-input.parquet', 'component-values.parquet', 'component-research.json', 'diagnostics.json',
    'probe-results.json', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch57 changed')
    price_sha = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']['price-input.parquet']
    require(file_sha(PRICE) == price_sha, 'Accepted daily input changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'price_file': str(PRICE), 'price_sha256': price_sha, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def missing_inputs():
    command = ['rg', '--files', 'repo', 'data']; result = subprocess.run(command, capture_output=True, text=True, check=True)
    paths = sorted(result.stdout.splitlines()); targets = ('gold_stock_20210609.csv', 'trade.csv')
    matches = {name: [p for p in paths if Path(p).name == name] for name in targets}
    return {'command': command, 'candidates': matches, 'missing': [n for n in targets if not matches[n]],
        'scope': 'Visible rg files only; not proof of absence from external machines or hidden/ignored folders',
        'replacement_allowed': False, 'not_a_backtest': True}


def preflight(root, directory):
    bound = binding(root); missing = missing_inputs()
    require(len(missing['missing']) == 2, 'Original CSV candidate found; inspect before proceeding')
    checkpoint(root, directory, 'Batch58 gold-stock source review started; missing lists cannot be replaced')
    save(directory / 'input-binding.json', bound); save(directory / 'missing-inputs.json', missing)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    shutil.copyfile(PRICE, directory / 'price-input.parquet')
    return {'status': 'ok', 'missing': missing['missing']}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, (name, params) in QUERIES.items():
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': name, 'parameters': params, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
        'queries': QUERIES, 'apis': rows,
        'limits': ['Latest ratings and one-stock rating history do not recreate monthly gold-stock membership or probability-model output',
            'Minute endpoint has a1970row cap and no historical start/end; it also calls daily qfq internally even for raw output',
            'No historical recommendation availability, model training, complete 2019+30m or jqlib optimizer equivalence supplied']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'missing_contracts': ['original_monthly_recommendation_csv_and_availability', 'probability_model_and_original_trade_csv',
            'historical_gold_stock_30m_and_optimizer_semantics'], 'ledger_sha256': file_sha('src/observe/ledger/book.py'),
        'not_a_backtest': True, 'limits': ['Ten stock daily histories are arithmetic operands, not original monthly candidates',
            'No orders, net values or fee implementation added; future execution must reuse the unique Book']}


def definitions(tree):
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = ast.parse(text)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True, 'definition_ast': definitions(tree)})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only interface/objective/budget reference; no alternate ledger or inferred jqlib equivalence'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch58 two full gold-stock sources frozen; original CSVs/model/minutes missing')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy']))
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Definition evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(file_sha(directory / 'price-input.parquet') == file_sha(PRICE), 'Daily operand changed')


def selected(directory, number, names, ns):
    import __future__
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Missing selected definition')
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            require(node.name in ('AF_factor', 'RetN_momentum', 'Q_factor', 'FactorWeight') and
                all(isinstance(n, (ast.FunctionDef, ast.Expr)) for n in node.body), 'Unsafe class initialization')
        else: require(not node.decorator_list, 'Decorated standalone function is not selected')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-gold-stock-components>', 'exec',
        flags=__future__.annotations.compiler_flag), ns)
    return ns


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def arithmetic(directory):
    validate_sources(directory); ns = selected(directory, 0, ['AF_factor', 'RetN_momentum'], {'pd': pd, 'np': np, 'Union': Union})
    data = pd.read_parquet(directory / 'price-input.parquet'); rows = []; profiles = []
    for instrument, frame in data.groupby('instrument', sort=True):
        frame = frame.sort_values('date').reset_index(drop=True)
        require(not frame.date.duplicated().any(), 'Duplicate daily dates')
        for length, component in ((22, 'AF20'), (121, 'RM120')):
            values = {k: np.lib.stride_tricks.sliding_window_view(frame[k+'_adj'].to_numpy(), length).T for k in ('close', 'high', 'low')}
            trading = np.lib.stride_tricks.sliding_window_view(frame.is_trading.to_numpy(), length).all(axis=1)
            good = trading.copy()
            for v in values.values(): good &= np.isfinite(v).all(axis=0) & (v > 0).all(axis=0)
            indices = np.flatnonzero(good); blocked = frame.date.iloc[length-1:].to_numpy()[~good].tolist()
            prices = {k: pd.DataFrame(v[:, good]) for k, v in values.items()}; prices['paused'] = pd.DataFrame(np.zeros_like(prices['close']))
            close, high, low = (prices[k].to_numpy() for k in ('close', 'high', 'low'))
            amplitude = high / low - 1
            if component == 'AF20':
                obj = ns['AF_factor']([], 'arithmetic-only', 20); obj.data = prices
                high_result = obj.calc(.5, 'high'); high_mask = obj._q_split().to_numpy()[-20:]
                low_result = obj.calc(.5, 'low'); low_mask = obj._q_split().to_numpy()[-20:]
                independent_high = stats.rankdata(-close, axis=0, method='average')[-20:] / length <= .5
                independent_low = stats.rankdata(close, axis=0, method='average')[-20:] / length <= .5
                prior_bad = ((close[1:] / close[:-1] - 1 < -.09) & (high[1:] == low[1:]))[-21:-1]
                selected_high = independent_high & ~prior_bad; selected_low = independent_low & ~prior_bad
                require(np.array_equal(high_mask, independent_high) and np.array_equal(low_mask, independent_low) and
                    np.array_equal(obj._get_paused().to_numpy()[-20:].astype(bool), ~prior_bad), 'AF condition differs')
                result = high_result.to_numpy(dtype=float) - low_result.to_numpy(dtype=float)
                reference = np.array([math.fsum(amplitude[-20:, j] * selected_high[:, j])/20 -
                    math.fsum(amplitude[-20:, j] * selected_low[:, j])/20 for j in range(len(indices))])
            else:
                obj = ns['RetN_momentum']([], 'arithmetic-only', 120); obj.data = prices
                result = obj.calc('lamb', .3).to_numpy()
                mask = stats.rankdata(amplitude[1:], axis=0, method='average') / 120 <= .3
                require(np.array_equal(obj._lamb_split(prices['high'].iloc[1:] / prices['low'].iloc[1:] - 1, .3).to_numpy(), mask), 'RM condition differs')
                returns = close[1:] / close[:-1] - 1
                reference = np.array([math.fsum(returns[:, j] * mask[:, j]) for j in range(len(indices))])
            require(np.isfinite(result).all() and np.allclose(result, reference, rtol=1e-12, atol=1e-14), 'Arithmetic value differs')
            dates = frame.date.iloc[length-1:].to_numpy()[good].tolist()
            for date, value, ref in zip(dates, result, reference, strict=True):
                rows.append({'instrument': instrument, 'date': date, 'component': component, 'value': float(value), 'independent_value': float(ref)})
            profiles.append({'instrument': instrument, 'component': component, 'length': length, 'candidate_windows': len(good),
                'complete_trading_windows': len(indices), 'blocked_windows': len(blocked), 'blocked_dates': blocked,
                'condition_differences': 0, 'max_absolute_error': float(np.max(np.abs(result-reference))) if len(indices) else 0.0})
    summary = {'profiles': profiles, 'calculated_windows': len(rows), 'records_sha256': digest(rows), 'not_a_backtest': True,
        'platform_equivalent': False, 'limits': ['Each column is one independent time window, not original monthly recommendation cross section',
            'Only complete finite positive actually-trading windows; missing or paused windows blocked without fill or dropping dates',
            'AF uses22row average ranks/next-day masks/20row zero-inclusive mean; RM uses120low-amplitude simple-return sums',
            'Post-adjusted supplier prices and floating arithmetic retained; condition validation is not platform pricing equivalence',
            'No smart-money30m, factor combination/optimizer, original assets, costs or portfolio performance claimed']}
    return summary, pd.DataFrame(rows)


def diagnostics(directory):
    validate_sources(directory); rows = []; g = SimpleNamespace(); orders = []
    ns = {'pd': pd, 'np': np, 'Union': Union, 'BytesIO': BytesIO, 'g': g, 'datetime': datetime,
        'order_target': lambda c, v: orders.append(['target', c, v]),
        'order_target_value': lambda c, v: orders.append(['value', c, v])}
    selected(directory, 0, ['AF_factor', 'Q_factor', 'RetN_momentum', 'FactorWeight', 'get_net_promoter_score',
        'read_gold_stock', 'composition_factors', 'opt_pos_weight', 'order2list', 'order2dict', 'before_trading_start',
        'get_target_securities', 'time2str'], ns)
    selected(directory, 1, ['TradeFunc', 'df2order', 'get_month_end_day'], ns)
    def failure(name, call, expected):
        try: call()
        except expected as exc: rows.append({'case': name, 'error': type(exc).__name__})
        else: raise ValueError('Expected original failure: '+name)
    context = SimpleNamespace(previous_date=pd.Timestamp('2021-06-09'),
        portfolio=SimpleNamespace(long_positions={'OLD': object()}, total_value=100000))
    ns['dt'] = datetime
    ns['normalize_code'] = lambda code: code.replace('.SZ', '.XSHE').replace('.SH', '.XSHG')
    csv = '所属日期,推荐机构,股票名称,所属行业,股票代码\n2021/6/1,A,a,i,002230.SZ\n2021/6/1,A,a,i,002230.SZ\n2021/6/1,B,b,i,000001.SZ\n2021/6/1,A,h,i,000001.HK\n2021/7/1,A,a,i,002230.SZ\n'
    ns['read_file'] = lambda name: csv.encode(); ns['read_gold_stock']()
    score = ns['get_net_promoter_score'](g.gold_stock_frame.loc['2021-06'])
    require(score['002230.XSHE'] == 1 and len(g.gold_stock_frame) == 4, 'CSV filter/count fixture differs')
    duplicated = pd.DataFrame({'股票代码': ['A']*3+['B'], '推荐机构': ['X']*3+['Y']})
    rate = ns['get_net_promoter_score'](duplicated)['A']; require(rate == 1.5, 'Original row counting differs')
    rows.append({'case': 'csv_hk_filter_months_and_duplicate_rate', 'recommendation_rate': rate, 'synthetic_csv': True})
    ns['get_target_securities'](context); require(isinstance(g.base_target, pd.DataFrame), 'Monthly selection differs')
    context.previous_date = pd.Timestamp('2021-07-09'); ns['get_target_securities'](context)
    require(isinstance(g.base_target, pd.Series), 'Single monthly row did not become Series')
    failure('single_month_row_unique_failure', lambda: g.base_target['股票代码'].unique(), AttributeError)
    context.previous_date = pd.Timestamp('2021-08-09')
    failure('missing_month_blocks', lambda: ns['get_target_securities'](context), ValueError)
    ns['get_all_trade_days'] = lambda: pd.date_range('2021-06-01', periods=10, freq='B')
    failure('month_end_M_current_pandas_failure', ns['get_month_end_day'], ValueError)
    failure('weekly_index_weekofyear_removed', lambda: pd.DatetimeIndex(['2021-06-01']).weekofyear, AttributeError)
    calls = []; ns['get_price'] = lambda *a, **kw: calls.append(kw)
    q = ns['Q_factor']([], 'synthetic', frequency='15m'); q.get_data()
    require(calls[-1]['count'] == 160.0 and calls[-1]['frequency'] == '30m', 'Q query declaration differs')
    rows.append({'case': 'q_count_float_hardcoded_30m', 'query': calls[-1], 'count_type': type(calls[-1]['count']).__name__})
    q.data = pd.DataFrame({'time': range(4), 'code': ['A']*4, 'open': [10.]*4,
        'close': [14., 11., 10., 100.], 'volume': [80., 10., 10., 0.]})
    actual = float(q.calc(.5).iloc[0]); expected = 14 / ((14*80+11*10+10*10)/100)
    require(math.isclose(actual, expected), 'Smart-money first-row fallback differs')
    rows.append({'case': 'q_zero_volume_removed_first_row_fallback', 'value': actual, 'independent_value': expected, 'synthetic_minutes': True})
    af = ns['AF_factor']([], 'synthetic', 20)
    c = np.full(22, 10.); c[5] = 9.; h = c+1; l = c-1; h[5] = l[5] = 9.
    af.data = {k: pd.DataFrame({'A': v}) for k, v in {'close': c, 'high': h, 'low': l, 'paused': np.zeros(22)}.items()}
    af.calc(.5, 'high'); mask = af._get_paused()['A']
    require(bool(mask.iloc[5]) and not bool(mask.iloc[6]), 'AF next-day mask differs')
    rows.append({'case': 'af_one_word_drop_masks_next_day', 'drop_day_usable': bool(mask.iloc[5]), 'next_day_usable': bool(mask.iloc[6])})
    rm = ns['RetN_momentum']([], 'synthetic', 120)
    rm.data = {k: pd.DataFrame({'A': np.linspace(10, 20, 121) * ratio}) for k, ratio in {'close': 1., 'high': 1.1, 'low': .9}.items()}
    failure('inactive_rm_q_undefined_lmb', lambda: rm.calc('q', .3), NameError)
    failure('inactive_rm_sort_missing_group', lambda: rm.calc('sort', .3, 'A'), AttributeError)
    fw = ns['FactorWeight'].__new__(ns['FactorWeight']); fw.last_factor = pd.DataFrame({'F1': [2.], 'F2': [3.]})
    fw.past_factor = fw.past_returns = None
    fw._get_factor_return = lambda *a: pd.DataFrame({'F1': [1., 3.], 'F2': [2., 4.]})
    raw_weight = fw.fac_ret_half(False); normalized = fw.fac_ret_half(True)
    require(raw_weight.iloc[0] == 13 and math.isclose(normalized.iloc[0], 2.6, rel_tol=1e-14), 'Factor-return behavior differs')
    rows.append({'case': 'factor_return_half_only_normalizes_mean', 'raw': float(raw_weight.iloc[0]), 'normalized': float(normalized.iloc[0]),
        'ic_half_weights': ns['FactorWeight']._build_halflife_wight(4, 2).tolist()})
    names = []
    class DummyWeight:
        def __init__(self, *a): pass
        def __getattr__(self, name):
            def method(*a): names.append(name); return pd.Series([len(names)], index=['A'])
            return method
    ns['FactorWeight'] = DummyWeight; ns['prepare_data'] = lambda *a: (None, None)
    combination = ns['composition_factors']([], 'synthetic', 5, 12)
    require(len(names) == 7 and combination.shape[1] == 6 and combination['最大化IC_IR加权法'].iloc[0] == 7, 'Eager duplicate method differs')
    rows.append({'case': 'eager_methods_duplicate_key_overwrite', 'calls': names.copy(), 'columns': combination.columns.tolist()})
    def explode(*a): raise RuntimeError('synthetic unused optimizer failure')
    DummyWeight.fac_maxicir_samp = explode
    failure('unused_combination_method_still_blocks', lambda: ns['composition_factors']([], 'synthetic', 5, 12), RuntimeError)
    declarations = []
    def declared(name):
        def call(**kw): declarations.append({'name': name, 'arguments': kw}); return {'name': name, **kw}
        return call
    for name in ('MaxSharpeRatio', 'MaxProfit', 'MinVariance', 'RiskParity', 'WeightConstraint', 'Bound', 'portfolio_optimizer'):
        ns[name] = declared(name)
    ns['opt_pos_weight'](['A'], '2021-06-09', 120, 'MaxProfit')
    require(len(declarations) == 7 and declarations[-1]['arguments']['target']['name'] == 'MaxProfit', 'Optimizer intent differs')
    rows.append({'case': 'optimizer_eager_objectives_constraints_only', 'calls': declarations, 'no_optimizer_solution': True})
    ns['order2list'](context, []); require(not orders, 'Empty list clears holdings')
    ns['order2list'](context, ['A', 'B']); require(orders == [['target', 'OLD', 0], ['value', 'A', 50000.], ['value', 'B', 50000.]], 'Order intents differ')
    rows.append({'case': 'empty_list_retains_holdings_equal_weight_sell_first', 'orders': orders.copy(), 'synthetic_orders': True}); orders.clear()
    ns['order2dict'](context, {}); require(orders == [['target', 'OLD', 0]], 'Empty dict intent differs')
    rows.append({'case': 'empty_dict_clears_old_holdings', 'orders': orders.copy()}); orders.clear()
    ns['record'] = lambda **kw: None; context.previous_date = pd.Timestamp('2021-06-09')
    g.target_df = pd.DataFrame({'asset': ['600000.XSHG']}, index=[context.previous_date]); ns['TradeFunc'](context)
    require(orders[-1] == ['value', '6', 100000.], 'Original singleton string slicing differs')
    rows.append({'case': 'probability_singleton_first_character_order', 'orders': orders.copy(), 'synthetic_orders': True}); orders.clear()
    context.previous_date = pd.Timestamp('2021-06-10'); ns['TradeFunc'](context); require(not orders, 'Missing recommendation submits orders')
    rows.append({'case': 'missing_recommendation_retains_holdings', 'orders': []})
    costs = []; ns['set_slippage'] = lambda v: None; ns['FixedSlippage'] = lambda v: v
    ns['set_commission'] = lambda v: costs.append(v); ns['PerTrade'] = lambda **kw: kw
    for date in ('2009-01-01', '2009-01-02', '2011-01-01', '2011-01-02', '2013-01-01', '2013-01-02'):
        context.current_dt = datetime.datetime.fromisoformat(date); ns['before_trading_start'](context)
    require([r['buy_cost'] for r in costs] == [.003, .002, .002, .001, .001, .0003], 'Strict cost boundaries differ')
    rows.append({'case': 'commission_strict_date_boundaries_declarations', 'costs': costs, 'fees_not_calculated': True})
    return {'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False, 'synthetic_fixture': True,
        'limits': ['CSV/minutes/factor cross section/orders/optimizer are explicit diagnostic fixtures, not original historical inputs or fills',
            'Original compatibility and inactive failures preserved; no economic-rule repair authorized by these diagnostics']}


def study(root, directory):
    binding(root, directory); summary, values = arithmetic(directory)
    path = directory / 'component-values.parquet'; require(not path.exists(), 'Component values exist'); values.to_parquet(path, index=False)
    save(directory / 'component-research.json', summary)
    with redirect_stdout(StringIO()): diag = diagnostics(directory)
    save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch58 AF/RM daily arithmetic and synthetic source diagnostics archived; no trades')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); name, params = QUERIES[endpoint]
    row = {'endpoint': endpoint, 'function': name, 'parameters': params, 'status': 'failed', 'files': [], 'wire_responses': [],
        'published': False, 'strict_usable': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}; original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire_responses']) < 8, 'Request cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(row["wire_responses"]):03d}.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API changed')
        import akshare as ak
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, name)(**params)
        path = Path(root) / 'raw/gold_stock_probe_batch58' / endpoint / f'{directory.name}.parquet'; require(not path.exists(), 'Raw exists')
        path = raw.save(root, 'gold_stock_probe_batch58', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API evidence changed')
    rows = read(directory / 'probe-results.json')['results']; require([r['endpoint'] for r in rows] == list(QUERIES), 'Probe scope changed')
    for row in rows:
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['published'] is False and row['strict_usable'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['function'] == QUERIES[row['endpoint']][0] and
            row['parameters'] == QUERIES[row['endpoint']][1] and row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Probe binding changed')
        for item in row['wire_responses']+row['files']: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status']=='success'), 'Probe raw profile differs')
        else: require(not row['files'] and row.get('error'), 'Failed probe raw/error differs')
    return rows


def probe(root, directory):
    rows = []; logs = directory / 'logs'; logs.mkdir()
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch58', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline45s', 'published': False, 'strict_usable': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch58 existing rating/recommendation/minute API supplements archived')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch58 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        component, values = arithmetic(directory); diag = diagnostics(directory); dependency = inventory(root); catalog = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    save(directory / 'component-offline.json', component)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and dependency == read(directory / 'dependency-inventory.json'), 'Offline outputs changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'values_sha256': file_sha(directory / 'component-values.parquet'), 'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch58 forbidden-network arithmetic/diagnostics/catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch58.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch58_research.py', 'tests/unit/test_batch56_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); validate_probes(directory)
    before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') and off['values_sha256'] == file_sha(directory / 'component-values.parquet') and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline proof changed')
    validate_probes(directory)
    with redirect_stdout(StringIO()): component, values = arithmetic(directory); diag = diagnostics(directory)
    require(component == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        inventory(root) == read(directory / 'dependency-inventory.json'), 'Recomputed research changed')
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch58 two gold-stock reviews accepted; original data/optimizer dependencies remain')
    receipt = Path('docs/handoff/2026-10-07-batch58-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch58/20261007-gold-stock')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
