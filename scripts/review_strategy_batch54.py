"""Freeze industry breadth, forecast tools and convertible-bond research limitations."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
from fractions import Fraction
import hashlib
import inspect
from io import StringIO
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

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch47 import native
from scripts import review_strategy_batch53 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch53/20261007-fund-intraday')
RECEIPT = Path('docs/handoff/2026-10-07-batch53-verification.json')
PRICE = Path('data/staging/strategies-batch47/20261007-candidate-momentum/price-input.parquet')
SOURCES = ('2023年度精选策略/96.市场宽度20210622.txt',
    '2024年度精选策略1/68.业绩预告小工具--已更新.txt', '2024年度精选策略2/86.可转债双低策略.txt')
SOURCE_SHA = ('355249a376ce4e95e28d954b416146270e9507806d2cbb6729f45545f993a99a',
    'a665307b7afe38d1bb4eb90e170f4bb874eb5ddb639472f627c40b7e53e438dc',
    '4d9c32ace0f6a2540d52c76676dbd88c94349a86e462ae27178737cf183c925b')
REFERENCES = (('repo/akshare', 'akshare/stock/stock_board_industry_em.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_yjyg_em.py'), ('repo/akshare', 'akshare/bond/bond_zh_cov.py'))
QUERIES = {'industry': ('stock_board_industry_name_em', {}), 'forecast': ('stock_yjyg_em', {'date': '20230630'}),
    'convertible': ('bond_zh_hs_cov_daily', {'symbol': 'sh113053'})}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch54.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json', 'price-input.parquet',
    'component-research.json', 'diagnostics.json', 'probe-results.json', 'offline-verification.json', 'offline-catalog.json'}
MOCK_TIME = datetime.datetime(2026, 10, 7, 12)


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch53 changed')
    price_sha = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']['price-input.parquet']
    require(file_sha(PRICE) == price_sha, 'Accepted stock operand changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'price_file': str(PRICE), 'price_sha256': price_sha, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch54 breadth/forecast/convertible source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    shutil.copyfile(PRICE, directory / 'price-input.parquet')
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, (name, params) in QUERIES.items():
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': name, 'parameters': params, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'queries': QUERIES, 'apis': rows,
        'limits': ['Current Eastmoney industry boards are not historical SW classifications',
            'A report-date forecast query does not prove complete first-publication/revision vintages',
            'One convertible daily price series cannot supply conversion-price/lifecycle/availability events']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'missing_contracts': ['historical_industries', 'historical_instrument_status', 'earnings_forecast_vintages',
            'convertible_basic_and_daily', 'convertible_conversion_price_events', 'convertible_lifecycle_and_trading_rules'],
        'not_a_backtest': True, 'limits': ['Final merged annual/quarterly financials are not earnings forecasts or publication vintages',
            'Missing contracts describe semantic dependencies, not assumed registered table names']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = ast.parse(text)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        magic = [ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Attribute) and n.func.attr in ('run_cell_magic', 'run_line_magic')]
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True, 'magic_ast': magic,
            'platform_tables': sorted({n.value.id+'.'+n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
                isinstance(n.value, ast.Name) and n.value.id in ('bond', 'finance') and n.attr.isupper()}),
            'function_ast': {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Existing industry/forecast/convertible interfaces only; no alternate ledger or historical-vintage inference'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch54 three full breadth/forecast/convertible sources frozen')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Selected source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json'); require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy'])); tree = source_tree(directory, i)
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and row['function_ast'] ==
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)} and row['magic_ast'] ==
            [ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and n.func.attr in ('run_cell_magic', 'run_line_magic')], 'AST evidence changed')
        require(row['platform_tables'] == sorted({n.value.id+'.'+n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
            isinstance(n.value, ast.Name) and n.value.id in ('bond', 'finance') and n.attr.isupper()}), 'Platform table evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(file_sha(directory / 'price-input.parquet') == read(directory / 'input-binding.json')['price_sha256'], 'Copied price operand changed')


def selected(directory, number, names, ns):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and not any(n.decorator_list for n in nodes), 'Selected functions missing/decorated')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-original-function>', 'exec'), ns)


def compute(directory):
    validate_sources(directory); frame = pd.read_parquet(directory / 'price-input.parquet'); rows = []; boundaries = []
    require(frame.adjustment_status.eq('usable').all() and not frame.duplicated(['date', 'instrument']).any(), 'Invalid stock operands')
    kernel = next(n.value for n in ast.walk(source_tree(directory, 0)) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'df_bias' for t in n.targets))
    code = compile(ast.Expression(body=kernel), '<original-strict-close-ma20>', 'eval')
    for instrument, group in frame.groupby('instrument', sort=True):
        group = group.sort_values('date'); values = group.close_adj.to_numpy(dtype=float); dates = group.date.astype(str).tolist()
        require(np.all(np.isfinite(values) | np.isnan(values)) and (values[np.isfinite(values)] > 0).all(), 'Invalid adjusted prices')
        means = pd.Series(values).rolling(20).mean().to_numpy(); finite = np.array([np.isfinite(w).all() for w in np.lib.stride_tricks.sliding_window_view(values, 20)])
        actual = eval(code, {'df_close': pd.DataFrame([values]), 'df_ma20': pd.DataFrame([means]), 'p_count': len(values)}).iloc[0].to_numpy()[19:]
        expected_means = np.array([math.fsum(w)/20 if ok else math.nan for w, ok in zip(np.lib.stride_tricks.sliding_window_view(values, 20), finite, strict=True)])
        expected = values[19:] > expected_means; changes = np.flatnonzero(finite & (actual != expected))
        for k in changes:
            window = values[k:k+20]; binary = sum(map(Fraction.from_float, map(float, window)), Fraction())/20
            decimal = sum((Fraction(str(float(v))) for v in window), Fraction())/20
            boundaries.append({'instrument': instrument, 'date': dates[k+19], 'window': window.tolist(),
                'window_float_hex': [float(v).hex() for v in window], 'close': float(window[-1]), 'pandas_ma20': float(means[k+19]),
                'fsum_ma20': float(expected_means[k]), 'pandas_flag': bool(actual[k]), 'reference_flag': bool(expected[k]),
                'exact_binary_mean': str(binary), 'exact_decimal_mean': str(decimal),
                'exact_binary_gt': Fraction.from_float(float(window[-1])) > binary, 'exact_decimal_gt': Fraction(str(float(window[-1]))) > decimal})
        eligible220 = [np.isfinite(w).all() for w in np.lib.stride_tricks.sliding_window_view(values, 220)]
        rows.append({'instrument': instrument, 'rows': len(values), 'first': dates[0], 'last': dates[-1],
            'finite_ma20_windows': int(finite.sum()), 'missing_ma20_windows': int((~finite).sum()),
            'max_abs_difference': float(np.max(np.abs(means[19:][finite]-expected_means[finite]))), 'strict_flag_boundaries': len(changes),
            'eligible_220row_cohort_windows': int(sum(eligible220)), 'blocked_220row_cohort_windows': int(len(eligible220)-sum(eligible220))})
    return native({'rows': rows, 'boundaries': boundaries, 'not_a_backtest': True, 'original_kernel_retained': True,
        'arithmetic_port': 'Removed rolling(axis=1) evaluated as the same per-stock pandas rolling mean; original strict expression retained',
        'limits': ['Ten frozen stocks and extended2005-2026 arithmetic are not the original2021 all-A historical pool/industry statistics',
            'NaNs remain missing; original220row complete-cohort gate separately counted, no forward fill or suspension skipping',
            'No market breadth series, trade, portfolio or alternative net-value implementation constructed']})


class Field:
    def __init__(self, name): self.name = name
    def __getattr__(self, name): return Field(self.name + '.' + name)
    def label(self, name): return self
    def asc(self): return self.name
    def __ge__(self, value): return ('>=', self.name, str(value))
    def __gt__(self, value): return ('>', self.name, str(value))
    def __le__(self, value): return ('<=', self.name, str(value))
    def __lt__(self, value): return ('<', self.name, str(value))
    def __eq__(self, value): return ('==', self.name, str(value))


class Query:
    def __init__(self, *fields): self.fields = [f.name for f in fields]; self.conditions = []; self.limit_value = None
    def filter(self, *conditions): self.conditions.extend(conditions); return self
    def order_by(self, *fields): return self
    def limit(self, value): self.limit_value = value; return self
    def evidence(self): return {'fields': self.fields, 'conditions': self.conditions, 'limit': self.limit_value}


def diagnostics(directory):
    validate_sources(directory); rows = []
    width = {'pd': pd, 'datetime': datetime, 'get_industry': lambda *a, **kw: {'A': {'sw_l1': {'industry_code': 'I'}}, 'B': {}}}
    selected(directory, 0, ['getStockIndustry', 'get_industry_width'], width)
    require(width['getStockIndustry'](['A', 'B'], 'sw_l1', datetime.date(2021, 6, 18)).to_dict() == {'A': 'I'}, 'Original absent industry handling differs')
    rows.append({'case': 'missing_industry_mapping_omitted', 'original_mapping': {'A': 'I'}})
    times = pd.bdate_range('2020-01-01', periods=220); calls = []
    width.update(get_industries=lambda **kw: pd.DataFrame({'name': ['industry']}, index=['I']),
        get_trade_days=lambda **kw: times.date, get_all_securities=lambda **kw: pd.DataFrame(index=['A']),
        get_price=lambda *a, **kw: pd.DataFrame({'time': times, 'code': ['A']*220, 'close': np.arange(220)+100.}))
    def industry(*a, **kw): calls.append(str(kw['p_day'])); return pd.Series({'A': 'I'})
    width['getStockIndustry'] = industry
    try: width['get_industry_width'](datetime.date(2021, 6, 18), 200, 'sw_l1')
    except TypeError as exc: rows.append({'case': 'original_rolling_axis_removed', 'error': type(exc).__name__, 'message': str(exc), 'industry_query_dates': calls})
    else: raise ValueError('Expected removed rolling-axis failure')
    flags = pd.DataFrame({'day': [True, False, True, True]}); flags['industry_code'] = ['I', 'I', 'J', None]
    grouped = ((flags.groupby('industry_code').sum()*100)/flags.groupby('industry_code').count()).round()
    require(grouped.day.sum() == 150 and flags.day.mean()*100 == 75, 'Original industry-sum versus market breadth differs')
    rows.append({'case': 'industry_percentage_sum_is_not_market_percentage', 'industry_sum': 150., 'market_ratio': 75., 'unclassified_stock_excluded_from_industries': True})
    frozen_datetime = SimpleNamespace(today=lambda: MOCK_TIME, now=lambda: MOCK_TIME)
    finance = Field('finance'); forecasts = []; annual_calls = []
    forecast = {'pd': pd, 'datetime': SimpleNamespace(datetime=frozen_datetime), 'finance': SimpleNamespace(STK_FIN_FORCAST=finance.STK_FIN_FORCAST),
        'query': Query, 'balance': Field('balance'), 'income': Field('income'), 'indicator': Field('indicator'), 'valuation': Field('valuation'),
        'get_security_info': lambda code: SimpleNamespace(display_name='current_' + code)}
    selected(directory, 1, ['forcast', 'get_profit'], forecast)
    require(forecast['forcast'].__defaults__ == forecast['get_profit'].__defaults__ == (MOCK_TIME,), 'Default time not frozen')
    def run_forecast(q):
        forecasts.append(q.evidence())
        return pd.DataFrame({'股票代码': ['A', 'B'], '发布日期': ['2023-07-01', '2026-01-01'], '报告截止': ['2023-06-30']*2,
            '报告期': [2, 2], '预告类型': [1, 1], '同比Min%': [1., 2.], '同比Max%': [3., 4.], '概述': ['mock']*2,
            '净利润Min': [10., 20.], '净利润Max': [30., 40.], '去年同期': [5., 6.]})
    forecast['finance'].run_query = run_forecast; output = forecast['forcast']('2023-06-30')
    require(output['发布日期'].tolist() == ['2026-01-01', '2023-07-01'] and forecasts[0]['conditions'] == [('>=', 'finance.STK_FIN_FORCAST.end_date', '2023-06-30')], 'Original forecast cutoff/sort differs')
    rows.append({'case': 'report_end_filter_does_not_limit_publication_date', 'query': forecasts[0], 'mock_result': output.to_dict('records')})
    def fundamentals(q, statDate):
        annual_calls.append(statDate)
        return pd.DataFrame({'期末': [statDate+'-12-31'], '净利润': [math.nan if statDate == '2025' else 100.], '同比%': [1.]})
    forecast['get_fundamentals'] = fundamentals
    forecast['finance'].run_query = lambda q: pd.DataFrame({'期末': ['2023-12-31'], '净利润': [10.], '同比%': [2.]})
    combined = forecast['get_profit']('A', '2023-06-30')
    require(annual_calls == [str(2026-i) for i in range(10)] and combined.loc['2025-12-31', '净利润'] == 0. and combined.index.duplicated().sum() == 1, 'Original annual date/fill/merge differs')
    rows.append({'case': 'system_year_not_input_year_missing_zero_and_duplicate_forecast_period', 'mock_time': str(MOCK_TIME),
        'input_date': '2023-06-30', 'annual_stat_dates': annual_calls, 'missing_profit_filled': 0., 'duplicate_period_rows': 1})
    bond_fields = Field('bond'); queries = []; today = '2022-04-15'
    tables = {'BOND_BASIC_INFO': pd.DataFrame({'short_name': ['mock'], 'company_code': ['A'], 'maturity_date': [datetime.date(2025, 1, 1)],
        'list_date': [datetime.date(2020, 1, 1)], 'interest_begin_date': [datetime.date(2019, 1, 1)]}),
        'CONBOND_BASIC_INFO': pd.DataFrame({'convert_start_date': [datetime.date(2020, 6, 1)]}),
        'CONBOND_DAILY_PRICE': pd.DataFrame({'close': [120.]}),
        'CONBOND_DAILY_CONVERT': pd.DataFrame({'date': [datetime.date(2025, 1, 1)]}),
        'CONBOND_CONVERT_PRICE_ADJUST': pd.DataFrame({'adjust_date': [datetime.date(2020, 1, 1)], 'new_convert_price': [10.]})}
    def run_bond(q): queries.append(q.evidence()); return tables[q.fields[0].split('.')[-1]].copy()
    bond = {'pd': pd, 'np': np, 'datetime': datetime, 'query': Query, 'print': lambda *a: None,
        'bond': SimpleNamespace(**{name: getattr(bond_fields, name) for name in tables}, run_query=run_bond),
        'get_price': lambda **kw: pd.DataFrame({'close': [20.]}, index=pd.to_datetime([today])), 'get_bond_extra_info': lambda code: (999., 'AAA')}
    selected(directory, 2, ['get_bond_detail', 'get_target_conv_bond_by_double_low', 'get_benchmark_conv_bond', 'sell_bond', 'update_bond_price'], bond)
    detail = bond['get_bond_detail']('mock', today)
    require(detail == (0, 120., 20., 10., 0, 0, 80., 1, 0, 1), 'Original double-low details differ')
    future = next(q for q in queries if q['fields'] == ['bond.CONBOND_DAILY_CONVERT'])
    require(('>=', 'bond.CONBOND_DAILY_CONVERT.date', today) in future['conditions'], 'Original future convert query differs')
    rows.append({'case': 'future_conversion_record_used_for_past_tradeability', 'query': future, 'mock_future_record': '2025-01-01', 'original_detail': detail})
    bond['get_bond_extra_info'] = lambda code: (1., 'D')
    require(bond['get_bond_detail']('mock', today) == detail, 'Unused current redemption/rating unexpectedly used')
    rows.append({'case': 'current_redemption_and_rating_fetched_but_unused', 'original_detail_unchanged': True})
    tables['BOND_BASIC_INFO'].loc[0, 'list_date'] = datetime.date(2022, 4, 15)
    try: bond['get_bond_detail']('mock', today)
    except AttributeError as exc: rows.append({'case': 'listing_day_invalid_branch_removed_numpy_NaN', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected removed np.NaN failure')
    tables['BOND_BASIC_INFO'].loc[0, 'list_date'] = datetime.date(2020, 1, 1)
    tables['CONBOND_DAILY_CONVERT'] = tables['CONBOND_DAILY_CONVERT'].iloc[:0]
    try: bond['get_bond_detail']('mock', today)
    except AttributeError as exc: rows.append({'case': 'no_future_conversion_hits_removed_numpy_NaN', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected no-future-record np.NaN failure')
    candidate = pd.DataFrame({'code': ['A', 'B'], 'tradeable': [1, 1], 'reach_first_trade_day': [1, 1], 'is_delist': [0, 0],
        'bond_price': [100., 60.], 'double_low': [110., 90.]})
    bond['get_bond_info'] = lambda date: candidate.copy(); targets = bond['get_target_conv_bond_by_double_low'](today, 10, 1000.)
    require([r[0] for r in targets] == ['B', 'A'] and np.isclose(sum(p*q for _, p, q in targets), 200.), 'Original candidate denominator differs')
    rows.append({'case': 'fewer_candidates_keep_original_denominator_and_fractional_quantity', 'mock_targets': targets, 'allocated_arithmetic_amount': 200., 'unallocated_input_amount': 800.})
    delisted = candidate.iloc[:1].copy(); delisted['is_delist'] = 1; delisted['bond_price'] = 120.
    bond['get_bond_info'] = lambda date: delisted.copy(); amount, remaining = bond['sell_bond'](today, [('A', 100., 10.)])
    require(amount == 1000. and remaining == [], 'Original delist stored-price liquidation differs')
    rows.append({'case': 'delist_arithmetic_uses_stored_price_not_redemption', 'stored_price': 100., 'mock_current_price': 120., 'returned_amount': amount})
    delisted['is_delist'] = 0; delisted['tradeable'] = 0
    require(bond['update_bond_price'](today, [('A', 100., 10.)]) == [('A', 100., 10.)], 'Original nontradeable stale mark differs')
    rows.append({'case': 'nontradeable_bond_keeps_stored_mark', 'mock_current_price': 120., 'stored_mark': 100.})
    bond['get_bond_info'] = lambda date: candidate.iloc[:0].copy()
    try: bond['get_benchmark_conv_bond'](today, 10, 1000.)
    except ZeroDivisionError as exc: rows.append({'case': 'empty_bond_benchmark_division_by_zero', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected empty-benchmark failure')
    return native({'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False, 'mock_time': str(MOCK_TIME),
        'limits': ['Query/field mocks only capture original conditions and provide explicit synthetic frames; they do not implement a platform/database',
            'Dynamic defaults and now are frozen by the synthetic namespace, without editing original AST',
            'No original convertible backtest, pickle cache, scraper, plotting/widget callbacks or alternate cash/net-value loop executed']})


def study(root, directory):
    binding(root, directory); save(directory / 'component-research.json', compute(directory)); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch54 MA20/window arithmetic and original query/lifecycle defects archived; no trades')


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
        path = Path(root) / 'raw/breadth_forecast_probe_batch54' / endpoint / f'{directory.name}.parquet'; require(not path.exists(), 'Raw exists')
        path = raw.save(root, 'breadth_forecast_probe_batch54', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch54', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline45s', 'published': False, 'strict_usable': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch54 existing industry/forecast/convertible supplements archived')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch54 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        component = compute(directory); diag = diagnostics(directory); dependency = inventory(root); catalog = offline_catalog(root, directory)
    save(directory / 'component-offline.json', component)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and dependency == read(directory / 'dependency-inventory.json'), 'Offline outputs changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch54 forbidden-network original diagnostics/MA20 arithmetic/catalog match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch54.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_research_queries.py',
         'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_catalog_macro.py',
         'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


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
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') and off['component_sha256'] ==
        file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline proof changed')
    validate_probes(directory); require(inventory(root) == read(directory / 'dependency-inventory.json') and diagnostics(directory) == read(directory / 'diagnostics.json') and
        compute(directory) == read(directory / 'component-research.json'), 'Recomputed research changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch54 three source reviews accepted; historical industry/forecast/bond originals remain blocked')
    save(Path('docs/handoff/2026-10-07-batch54-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 3})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch54/20261007-breadth-forecast-bond')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
