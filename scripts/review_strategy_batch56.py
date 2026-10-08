"""Freeze convertible-factor and intraday futures sources; no trading backtest."""
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
from scipy import optimize, stats

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch47 import native
from scripts.review_strategy_batch54 import Query
from scripts import review_strategy_batch55 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch55/20261007-bond-dependency')
RECEIPT = Path('docs/handoff/2026-10-07-batch55-verification.json')
PRICE = Path('data/staging/strategies-batch47/20261007-candidate-momentum/price-input.parquet')
SOURCES = ('2024年度精选策略2/82.可转债知识总结.txt',
    '2021年度精选策略/72.【股指期货】收盘折溢价策略.txt', '2022年度精选策略/45.股指期货跨合约套利.txt')
SOURCE_SHA = ('d5b780dc601a73ef85d230087d661a32b6c4adc3030a7ba147cebdc0719b3e20',
    'b2051102a114a49d6179cc5c451485eb0ef0f08406ad22f36a6c48c768bdd60c',
    'ceb3104eef975f3b2c465de33f7c691114c3b10175832d5a4fb6751887d6fb76')
REFERENCES = (('repo/backtrader', 'backtrader/comminfo.py'), ('repo/vnpy', 'vnpy/trader/object.py'),
    ('repo/akshare', 'akshare/bond/bond_zh_cov.py'), ('repo/akshare', 'akshare/futures/futures_zh_sina.py'))
QUERIES = {'bond_list': ('bond_zh_cov', {}), 'if1906_minute': ('futures_zh_minute_sina', {'symbol': 'IF1906', 'period': '1'}),
    'if1909_minute': ('futures_zh_minute_sina', {'symbol': 'IF1909', 'period': '1'})}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch56.py', 'tests/unit/test_batch56_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json', 'price-input.parquet',
    'candidate-screen.json', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'offline-verification.json', 'offline-catalog.json'}
SCREEN_EXCLUDED_APIS = {'get_fundamentals', 'get_fundamentals_continuously', 'get_history_fundamentals', 'get_valuation',
    'finance.run_query', 'macro.run_query', 'bond.run_query', 'get_all_securities', 'get_index_stocks', 'get_industry_stocks',
    'get_factor_values', 'get_ticks', 'get_call_auction'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch55 changed')
    price_sha = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']['price-input.parquet']
    require(file_sha(PRICE) == price_sha, 'Accepted stock operand changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'price_file': str(PRICE), 'price_sha256': price_sha, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def candidate_screen(directory):
    path = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; rows = []
    for item in read(path):
        if (item['review_status'] == '人工审查完成' or item['duplicate_of'] is not None or '股票候选' not in item['asset_scope'] or
            'ETF/基金候选' in item['asset_scope'] or SCREEN_EXCLUDED_APIS.intersection(item['apis']) or
            not any(a in item['apis'] for a in ('order', 'order_value', 'order_target', 'order_target_value'))): continue
        source = Path('repo/量化策略源代码') / item['path']; require(file_sha(source) == item['bytes_sha256'], 'Screen source changed')
        rows.append({k: item[k] for k in ('path', 'bytes_sha256', 'asset_scope', 'frequencies', 'scheduled_times', 'apis', 'gaps')})
    require(len(rows) == 15, 'Screen candidate count differs')
    return {'catalog_file': str(path), 'catalog_sha256': file_sha(path), 'excluded_apis': sorted(SCREEN_EXCLUDED_APIS), 'sources': rows,
        'not_a_backtest': True, 'new_manual_reviews_from_screen': 0,
        'limits': ['Static screening is neither exhaustive tradability proof nor manual review; false asset candidates and unused calls require reading',
            'No ranking by observed performance; screening does not authorize changing original assets/frequencies/financial dependencies']}


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch56 convertible/futures source research started; no trading run')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    shutil.copyfile(PRICE, directory / 'price-input.parquet'); save(directory / 'candidate-screen.json', candidate_screen(directory))
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, (name, params) in QUERIES.items():
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': name, 'parameters': params, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
        'queries': QUERIES, 'apis': rows,
        'limits': ['Current convertible list/quotes are not historical lifecycle/conversion/financial/industry vintages',
            'Sina FewMinLine has only symbol/period, not historical start/end; an expired-contract sample cannot prove2019minute completeness',
            'No historical dominant IF timeline, contract multiplier/tick/settlement/margin/fees supplied by these calls']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'missing_contracts': ['convertible_daily_conversion_and_lifecycle', 'historical_financial_and_industry_vintages',
            'historical_if_dominant_and_contract_minutes', 'futures_settlement_and_trading_rules'],
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'limits': ['Stock daily/index/5m tables are not convertible/futures histories; no placeholder rows or aliases inserted',
            'Original commission/margin declarations do not prove exchange history or a supported futures ledger',
            'Future execution must deepen the existing Book after actual contract/settlement data and rules are verified']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = ast.parse(text)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True,
            'platform_tables': sorted({n.value.id+'.'+n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
                isinstance(n.value, ast.Name) and n.value.id == 'bond' and n.attr.isupper()}),
            'function_ast': {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only commission/margin/contract/interface reference; no alternate ledger or historical-vintage inference'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch56 three full convertible/futures sources frozen; originals await dependencies')


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
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}, 'AST evidence changed')
        require(row['platform_tables'] == sorted({n.value.id+'.'+n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
            isinstance(n.value, ast.Name) and n.value.id == 'bond' and n.attr.isupper()}), 'Table evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(file_sha(directory / 'price-input.parquet') == read(directory / 'input-binding.json')['price_sha256'], 'Copied price operand changed')
    require(candidate_screen(directory) == read(directory / 'candidate-screen.json'), 'Candidate screen changed')


def selected(directory, number, names, ns):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and not any(n.decorator_list for n in nodes), 'Selected functions missing/decorated')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-original-function>', 'exec'), ns)


def compute(directory):
    validate_sources(directory); frame = pd.read_parquet(directory / 'price-input.parquet'); rows = []; boundaries = []
    require(frame.adjustment_status.eq('usable').all() and not frame.duplicated(['date', 'instrument']).any(), 'Invalid stock operands')
    function = next(n for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name == 'momentom_factoring')
    kernel = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'returns' for t in n.targets))
    code = compile(ast.Expression(body=kernel), '<original126row-endpoint-ratio>', 'eval')
    for instrument, group in frame.groupby('instrument', sort=True):
        group = group.sort_values('date'); values = group.close_adj.to_numpy(dtype=float); dates = group.date.astype(str).tolist()
        require(np.all(np.isfinite(values) | np.isnan(values)) and (values[np.isfinite(values)] > 0).all(), 'Invalid adjusted prices')
        valid = missing = interior = 0; max_diff = 0.
        for k, window in enumerate(np.lib.stride_tricks.sliding_window_view(values, 126)):
            actual = float(eval(code, {'price': pd.DataFrame({'close': window})}))
            if not np.isfinite(window[[0, -1]]).all():
                require(math.isnan(actual), 'Missing endpoint was silently filled'); missing += 1; continue
            exact = Fraction.from_float(float(window[-1])) / Fraction.from_float(float(window[0])) - 1
            max_diff = max(max_diff, abs(actual-float(exact))); valid += 1
            interior += int(not np.isfinite(window).all())
            if (actual > 0) != (exact > 0): boundaries.append({'instrument': instrument, 'date': dates[k+125],
                'first': float(window[0]), 'last': float(window[-1]), 'first_hex': float(window[0]).hex(), 'last_hex': float(window[-1]).hex(),
                'original_return': actual, 'exact_binary_return': str(exact)})
        require(max_diff < 1e-12, 'Independent endpoint ratio differs materially')
        rows.append({'instrument': instrument, 'rows': len(values), 'first': dates[0], 'last': dates[-1], 'window_rows': 126,
            'finite_endpoint_windows': valid, 'missing_endpoint_windows': missing, 'interior_missing_with_finite_endpoints': interior, 'max_abs_difference': max_diff})
    ns = {}; selected(directory, 0, ['season_selection'], ns); quarters = []
    for year in range(2005, 2027):
        for month in range(1, 13):
            day = datetime.date(year, month, 1); actual, sample = ns['season_selection'](day.isoformat())
            index = year*4+(month-1)//3-1
            fmt = lambda i: f'{i//4}q{i%4+1}'
            require(actual == fmt(index) and sample == [fmt(index-i) for i in range(1, 14)], 'Original prior-quarter selection differs')
            quarters.append({'date': day.isoformat(), 'latest_quarter': actual, 'prior13': sample, 'no_fallback_sue_sample12': sample[-12:]})
    return native({'rows': rows, 'boundaries': boundaries, 'quarter_cases': quarters, 'original_kernel_retained': True, 'not_a_backtest': True,
        'limits': ['Ten frozen stock endpoints and264quarter dates are arithmetic, not a historical convertible universe/financial version query',
            'Interior NaNs retained; original endpoint-only expression can return a finite value despite missing middle rows',
            'No skipped suspension, zero fill, futures proxy, portfolio, trade, cost or alternative net-value implementation']})


class Field:
    def __init__(self, name): self.name = name
    def __getattr__(self, name): return Field(self.name+'.'+name)
    def __eq__(self, value): return ('==', self.name, str(value))
    def __ne__(self, value): return ('!=', self.name, str(value))
    def __ge__(self, value): return ('>=', self.name, str(value))
    def __gt__(self, value): return ('>', self.name, str(value))
    def __le__(self, value): return ('<=', self.name, str(value))
    def __lt__(self, value): return ('<', self.name, str(value))
    def in_(self, value): return ('in', self.name, [str(v) for v in value])


def failed_case(rows, name, call, error):
    try: call()
    except error as exc: rows.append({'case': name, 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected original failure: '+name)


def diagnostics(directory):
    validate_sources(directory); rows = []; bond_fields = Field('bond'); calls = []
    bond = {'pd': pd, 'np': np, 'datetime': datetime, 'stats': stats, 'curve_fit': optimize.curve_fit, 'query': Query,
        'income': Field('income'), 'valuation': Field('valuation'), 'print': lambda *a: None, 'get_industry': lambda *a, **kw: {'A': {'sw_l1': {'industry_code': '801010'}}}}
    tables = {'CONBOND_DAILY_CONVERT': pd.DataFrame({'code': ['B'], 'convert_premium_rate': [1.]}),
        'CONBOND_BASIC_INFO': pd.DataFrame({'code': ['B'], 'company_code': ['A']}),
        'CONBOND_DAILY_PRICE': pd.DataFrame({'code': ['B'], 'close': [100.]})}
    def run_query(q): calls.append(q.evidence()); return tables[q.fields[0].split('.')[1]].copy()
    bond['bond'] = SimpleNamespace(**{name: getattr(bond_fields, name) for name in tables}, run_query=run_query)
    names = ['season_selection', 'SUE_factoring', 'epbp_factoring', 'premium_factoring', 'volume_factoring', 'skew_factoring',
        'convertible_selection', 'basic_selection', 'daily_prevention', 'correlation_calculation']
    selected(directory, 0, names, bond)
    for name, args in [('convertible_selection', ('2022-01-01', '2022-02-10')), ('basic_selection', ('2022-01-01', '2022-02-10')),
        ('daily_prevention', (['B.XSHG'], [], '2022-02-10'))]:
        failed_case(rows, 'empty_date_slice_'+name, lambda n=name, a=args: bond[n](*a), ValueError)
    for fallback in (False, True):
        dates = []
        def fundamentals(q, statDate):
            dates.append(statDate)
            if fallback and statDate == '2021q4': return pd.DataFrame(columns=['code', 'statDate', 'operating_revenue'])
            return pd.DataFrame({'code': ['A'], 'statDate': [statDate], 'operating_revenue': [100.]})
        bond['get_fundamentals'] = fundamentals
        failed_case(rows, 'sue_mixed_statdate_aggregation_'+str(fallback), lambda: bond['SUE_factoring'](['A'], '2022-02-01'), TypeError)
        _, sample = bond['season_selection']('2022-02-01')
        expected = ['2021q4', '2021q3']+sample[1:] if fallback else ['2021q4']+sample[-12:]
        require(dates == expected, 'Original SUE window/fallback differs'); rows[-1]['stat_dates'] = dates
    valuation_calls = []
    def valuations(q, statDate):
        valuation_calls.append(statDate)
        return pd.DataFrame({'code': ['A'], 'pubDate': ['synthetic'], 'pe_ratio': [-10.], 'pb_ratio': [-2.]})
    bond['get_fundamentals'] = valuations; inverse = bond['epbp_factoring'](['A'], '2022-02-01')
    require(valuation_calls == ['2021q4'] and np.isclose(inverse.epbp_factor.iloc[0], -.6), 'Original inverse factor differs')
    rows.append({'case': 'epbp_retains_negative_inverses', 'stat_dates': valuation_calls, 'synthetic_factor': inverse.epbp_factor.iloc[0]})
    premium_calls = []; values = {'A': 100., 'B': 80., 'C': 60.}
    def premium_query(q):
        premium_calls.append(q.evidence()); symbol = q.conditions[0][2]; value = values[symbol]; rate = 20/value+.1
        if 'CONBOND_DAILY_CONVERT' in q.fields[0]:
            return pd.DataFrame({'date': ['2022-01-31', '2022-01-03'], 'convert_premium_rate': [rate+.2, rate]})
        return pd.DataFrame({'close': [value*(1+rate)]})
    bond['bond'].run_query = premium_query
    output = bond['premium_factoring'](list(values), '2022-01-01', '2022-02-01')
    require(np.allclose(output.fixed_premium, [20/v+.1 for v in values.values()], rtol=1e-8, atol=1e-8) and
        all(q['conditions'][1][2] == '2022-01-03' for q in premium_calls if 'CONBOND_DAILY_PRICE' in q['fields'][0]), 'Original unsorted premium/endpoints differ')
    rows.append({'case': 'premium_unsorted_last_row_and_inverse_fit', 'queries': premium_calls, 'synthetic_output': output.to_dict('records'), 'rates_are_explicit_synthetic_ratios': True})
    bond['bond'].run_query = lambda q: pd.DataFrame({'date': ['2022-01-03'], 'convert_premium_rate': [0.]}) if 'CONBOND_DAILY_CONVERT' in q.fields[0] else pd.DataFrame({'close': [100.]})
    failed_case(rows, 'premium_zero_truthiness_excludes_all_fit_inputs', lambda: bond['premium_factoring'](['A'], '2022-01-01', '2022-02-01'), ValueError)
    volume_calls = []
    def volumes(q):
        volume_calls.append(q.evidence()); return pd.DataFrame({'volume': [30.] if q.conditions[1][2].startswith('2022-01-24') else [100.]})
    bond['bond'].run_query = volumes; volume = bond['volume_factoring'](['A'], '2022-02-07', 2022, 2, 7)
    require(volume.iloc[0, 0] == .3, 'Original volume ratio differs')
    rows.append({'case': 'volume_natural14_and90_days', 'queries': volume_calls, 'synthetic_ratio': volume.iloc[0, 0]})
    bond['bond'].run_query = lambda q: pd.DataFrame({'volume': [0.]})
    with np.errstate(divide='ignore', invalid='ignore'): volume = bond['volume_factoring'](['A'], '2022-02-07', 2022, 2, 7)
    require(math.isnan(volume.iloc[0, 0]), 'Original zero volume denominator differs')
    rows.append({'case': 'volume_zero_denominator_nan', 'synthetic_ratio': 'nan'})
    skew_prices = np.array([10., 12., 11., 13., 9., 14.]); skew_volumes = np.array([1., 3., 5., 2., 6., 4.])
    bond['bond'].run_query = lambda q: pd.DataFrame({'volume': skew_volumes, 'close': skew_prices, 'exchange_code': ['XSHG']*6})
    skew = bond['skew_factoring'](['A'], '2022-02-07', 2022, 2, 7).cv_close_skew_63D.iloc[0]
    expected_skew = stats.skew(skew_prices*skew_volumes, bias=False)
    require(abs(skew-expected_skew) < 1e-12 and abs(skew-stats.skew(skew_prices, bias=False)) > .01, 'Original product skew differs')
    rows.append({'case': 'skew_is_volume_price_product_not_weighted_price', 'synthetic_skew': skew, 'independent_scipy_skew': expected_skew})
    correlation_calls = []; dates = pd.date_range('2017-12-10', periods=10).date; prices = np.array([100., 110., 105., 120., 90., 115., 108., 130., 112., 140.])
    def correlation(q):
        correlation_calls.append(q.evidence())
        if 'CONBOND_DAILY_CONVERT' in q.fields[0]: return pd.DataFrame({'date': dates, 'code': ['A']*10, 'convert_premium_rate': np.arange(60., 70.)})
        return pd.DataFrame({'date': dates, 'close': prices})
    bond['bond'].run_query = correlation
    failed_case(rows, 'correlation_object_date_aggregation_failure', lambda: bond['correlation_calculation'](2018), TypeError)
    rows[-1]['queries'] = correlation_calls.copy()
    dates = pd.date_range('2017-12-10', periods=10); correlation_calls.clear()
    result = bond['correlation_calculation'](2018)
    expected_corr = stats.pearsonr(np.arange(60., 65.), (prices[5:]/prices[:5]-1)*100)
    require(len(correlation_calls) == 24 and all(q['conditions'][-2][2] == '2017-12-02 00:00:00' for q in correlation_calls) and
        np.allclose(result['correlation'], expected_corr.statistic) and np.allclose(result['p_value'], expected_corr.pvalue), 'Original correlation label/start differs')
    rows.append({'case': 'correlation_fixed_december_start_and_five_row_label', 'queries': correlation_calls, 'synthetic_result': result,
        'label_pairs': 5, 'synthetic_date_dtype': 'datetime64', 'platform_equivalent': False})
    diagnostic_futures(directory, rows)
    return native({'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Explicit synthetic query/price/portfolio/order frames capture original conditions, not a platform/database or fills',
            'Numeric-only initial conversion table intentionally reaches date parser; full mixed statDate frame separately exposes aggregation failure',
            'Object-date correlation aggregation failure is preserved separately; datetime64 synthetic dates only reach the unchanged five-row label core',
            'Synthetic minute negative integer labels are explicit test fixtures, not proof of platform history/index behavior',
            'No original top-level queries, plot/cache/backtest or alternative commission/margin/settlement/net-value loop executed']})


def diagnostic_futures(directory, rows):
    g = SimpleNamespace(main='IFMOCK', thresh=0, flag=''); orders = []; options = []; costs = []; schedules = []
    ns = {'pd': pd, 'np': np, 'dt': datetime, 'g': g, 'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None),
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: options.append(list(a)), 'set_subportfolios': lambda *a: None,
        'SubPortfolioConfig': lambda **kw: kw, 'OrderCost': lambda **kw: kw,
        'set_order_cost': lambda cost, **kw: costs.append({'declaration': cost, 'type': kw['type']}),
        'StepRelatedSlippage': lambda value: value, 'set_slippage': lambda value: None,
        'run_daily': lambda fn, **kw: schedules.append({'function': fn.__name__, **kw}),
        'order': lambda *a, **kw: orders.append([list(a), kw]), 'order_target': lambda *a, **kw: orders.append([list(a), kw]),
        'get_dominant_future': lambda underlying: 'IFMOCK', 'get_trades': lambda: {}}
    selected(directory, 1, ['initialize', 'before_market_open', 'close_position', 'open_position', 'market_open', 'after_market_close'], ns)
    context = SimpleNamespace(portfolio=SimpleNamespace(starting_cash=1000000, available_cash=800000, positions={'IFA': 1, 'IFB': 1}))
    ns['initialize'](context); ns['before_market_open'](context)
    require([r['time'] for r in schedules] == ['09:00', '10:00', '14:59', '15:30'] and ['futures_margin_rate', .15] in options and
        costs == [{'declaration': {'open_commission': .000023, 'close_commission': .000023, 'close_today_commission': .0023}, 'type': 'index_futures'}], 'Original IF declarations differ')
    rows.append({'case': 'close_premium_schedules_and_cost_declarations', 'schedules': schedules.copy(), 'cost_declarations': costs.copy(), 'options': options.copy()})
    frame = pd.DataFrame({'close': [100.]*58+[90.], 'volume': [1.]*59}, index=range(-59, 0)); history_calls = []
    def history(*a, **kw): history_calls.append([list(a), kw]); return frame.copy()
    ns['attribute_history'] = history; ns['open_position'](context)
    require(orders == [[['IFMOCK', 2], {}]] and g.flag == 'long', 'Original strict weighted price/400000 budget differs')
    rows.append({'case': 'close_premium_weighted_price_and_400000_budget', 'orders': orders.copy(), 'history_calls': history_calls.copy(), 'failed_stub_return_updates_flag': True})
    orders.clear(); context.portfolio.available_cash = 399999; ns['open_position'](context)
    require(orders == [[['IFMOCK', 0], {}]] and g.flag == 'long', 'Original zero quantity order differs')
    rows.append({'case': 'close_premium_zero_quantity_still_updates_flag', 'orders': orders.copy()})
    orders.clear(); frame['volume'] = 0.; ns['open_position'](context); require(not orders, 'Original all-zero volume branch differs')
    rows.append({'case': 'close_premium_zero_volume_silent_no_order', 'orders': []})
    frame['volume'] = 1.; frame['close'] = 100.; ns['open_position'](context); require(not orders, 'Original strict equality opens position')
    rows.append({'case': 'close_premium_equal_close_no_open', 'orders': []})
    g.flag = 'long'; ns['close_position'](context)
    require(orders == [[['IFA', 0, None, 'long'], {}], [['IFB', 0, None, ''], {}]] and g.flag == '', 'Original per-position flag reset differs')
    rows.append({'case': 'close_premium_shared_flag_reset_after_first_position', 'orders': orders.copy()})
    orders.clear(); ns['market_open'](context); require(not orders, 'Original unreachable market_open logic ran')
    rows.append({'case': 'close_premium_dead_market_open_returns', 'orders': []})
    frame.index = pd.date_range('2022-01-01', periods=59)
    failed_case(rows, 'close_premium_datetime_index_minus1_failure', lambda: ns['open_position'](context), KeyError)
    schedules.clear(); options.clear(); costs.clear(); orders.clear(); g = SimpleNamespace(); ns['g'] = g
    selected(directory, 2, ['initialize', 'before_market_open', 'set_slip_fee', 'market_open'], ns)
    context.current_dt = datetime.datetime(2019, 4, 22); context.portfolio.positions_value = 0; ns['initialize'](context)
    declarations = []
    for day, expected_today, expected_margin in [('2017-09-17', .00092, .2), ('2017-09-18', .00069, .15),
        ('2018-12-03', .00046, .1), ('2019-04-22', .000345, .1)]:
        context.current_dt = datetime.datetime.fromisoformat(day); costs.clear(); options.clear(); ns['before_market_open'](context)
        require(costs[0]['declaration']['close_today_commission'] == expected_today and g.futures_margin_rate == expected_margin,
            'Original historical commission/margin declaration differs')
        declarations.append({'date': day, 'cost': costs[0], 'options': options.copy()})
    rows.append({'case': 'fixed_pair_every_bar_four_original_fee_margin_declarations', 'schedules': schedules.copy(), 'declarations': declarations})
    def pair_case(values, position_value):
        orders.clear(); traces = []; context.portfolio.positions_value = position_value
        def pair_history(symbol, count, **kw):
            traces.append({'symbol': symbol, 'count': count, **kw})
            return pd.DataFrame({'close': 1000+values if symbol == 'IF1906.CCFX' else np.full(len(values), 1000.)}, index=range(-len(values), 0))
        ns['attribute_history'] = pair_history; ns['market_open'](context)
        return {'history_calls': traces, 'orders': orders.copy(), 'synthetic_position_value': position_value, 'actual_rows': len(values)}
    opened = pair_case(np.arange(300.), 0)
    require(opened['orders'] == [[['IF1906.CCFX', 1], {'side': 'short', 'close_today': False}],
        [['IF1909.CCFX', 1], {'side': 'long', 'close_today': False}]], 'Original fixed contracts/pair opening differs')
    rows.append({'case': 'fixed_pair_upper_quantile_two_order_intents', **opened})
    closed = pair_case(np.arange(300.)[::-1], 1)
    require([r[0][1] for r in closed['orders']] == [0, 0], 'Original lower quantile pair exit differs')
    rows.append({'case': 'fixed_pair_lower_quantile_two_exit_intents', **closed})
    equal = pair_case(np.full(300, 5.), 0); require(not equal['orders'], 'Original pair equality opens position')
    rows.append({'case': 'fixed_pair_equal_quantile_no_open', **equal})
    short = pair_case(np.array([3., 7.]), 0); require(len(short['orders']) == 2, 'Original pair short window handling differs')
    rows.append({'case': 'fixed_pair_two_rows_still_uses_quantiles', **short})
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(300.)}, index=pd.date_range('2019-01-01', periods=300))
    failed_case(rows, 'fixed_pair_datetime_index_minus1_failure', lambda: ns['market_open'](context), KeyError)


def study(root, directory):
    binding(root, directory); save(directory / 'component-research.json', compute(directory)); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch56 original126row/quarter arithmetic and convertible/futures diagnostics archived; no trades')


def diagnose(root, directory):
    binding(root, directory)
    require(read(directory / 'study-failure-01.json')['diagnostics_saved'] is False and
        not (directory / 'diagnostics.json').exists(), 'Diagnostics recovery is not applicable')
    require(compute(directory) == read(directory / 'component-research.json'), 'Accepted arithmetic changed during recovery')
    save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch56 failed diagnostic fixture repaired; original arithmetic retained and diagnostics newly archived')


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
        path = Path(root) / 'raw/convertible_futures_probe_batch56' / endpoint / f'{directory.name}.parquet'; require(not path.exists(), 'Raw exists')
        path = raw.save(root, 'convertible_futures_probe_batch56', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch56', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline45s', 'published': False, 'strict_usable': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch56 existing convertible list/expired IF minutes supplements archived')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch56 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        component = compute(directory); diag = diagnostics(directory); dependency = inventory(root); catalog = offline_catalog(root, directory)
    save(directory / 'component-offline.json', component)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and dependency == read(directory / 'dependency-inventory.json'), 'Offline outputs changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch56 forbidden-network original diagnostics/endpoint arithmetic/catalog match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch56.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch56_research.py', 'tests/unit/test_catalog_bond.py', 'tests/unit/test_catalog_dependencies.py',
         'tests/unit/test_catalog_schedules.py', 'tests/unit/test_catalog_macro.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


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
    protection = protect(root, directory); progress = archive(root, 'Batch56 three source reviews accepted; original convertible/futures work remains blocked by dependencies')
    save(Path('docs/handoff/2026-10-07-batch56-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 3})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'diagnose', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch56/20261007-convertible-futures')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
