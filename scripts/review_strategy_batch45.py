"""Freeze original momentum/EPO and daily/monthly trend-correlation research."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
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
from scipy.linalg import solve

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts import review_strategy_batch44 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch44/20261007-epo-correlation')
RECEIPT = Path('docs/handoff/2026-10-07-batch44-verification.json')
SOURCES = ('聚宽2025年精选/17多品种ETF动量轮动+EPO优化.txt',
    '聚宽2025年精选/74趋势筛选后相关性最小etf轮动.txt',
    '聚宽2025年精选/97趋势筛选后相关性最小etf轮动-加速10倍版.txt')
EXPECTED_SOURCE_SHA = ('ac55a9e0dfcb9b9244c68935b08f1a0dea18a6e9e54ac01baf73346425ed4f98',
    'f3b337a8b235fed948f174590c46c4327dc1000cbc62456f1a09274e14cc19bf',
    '0a2d130d41420c5f78b13b4f6edbe8ed2b30ed9a8e65a16f3738de17f25737a5')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/sma.py'),
    ('repo/backtrader', 'backtrader/indicators/basicops.py'),
    ('repo/skfolio', 'src/skfolio/moments/covariance/_shrunk_covariance.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_sina.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch45.py', 'tests/unit/test_batch45_research.py'}))
CORE = set(previous.CORE)
QUERIES = {
    'innovation_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '159992', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'One innovation ETF sample does not prove thirteen/thirty pools, events or1200/3500-row availability'},
    'hk_daily': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz159740'},
        'limit': 'One overseas ETF daily sample does not prove historical QDII calendars, events or9:30/10:00 execution'},
    'gold518800_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '518800', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': '518800 is not518880; one daily sample is not complete original funds or fund execution rules'}}
SAMPLE3 = ('600036.SH', '600085.SH', '600519.SH')


def implementation(): return {name: file_sha(name) for name in FILES}


def preflight(root, directory):
    checkpoint(root, directory, 'Batch45 momentum/EPO and daily/monthly trend research started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok', 'not_a_backtest': True}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch44 differs')
    files = {}
    for name in ('price-input.parquet', 'index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name; require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted operand changed')
        files[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'final_binding_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'files': files, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    apis = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); code = inspect.getsource(fn)
        apis.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    numerical = []
    for fn in (solve, pd.Series.__getitem__, pd.core.window.rolling.Rolling.mean):
        code = inspect.getsource(fn)
        numerical.append({'function': f'{fn.__module__}.{fn.__qualname__}', 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': previous.scipy.__version__, 'queries': QUERIES, 'apis': apis, 'numerical': numerical}


def start(root, directory):
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old_path = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name, sha in zip(SOURCES, EXPECTED_SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repo, name in REFERENCES:
        path = Path(repo) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only SMA/fsum, covariance and ETF interface references; original estimator/rolling algorithm retained'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch45 three complete momentum/EPO and trend source reviews frozen')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 3, 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog differs')
    for name, sha, row in zip(SOURCES, EXPECTED_SOURCE_SHA, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['source_sha256'] == sha and
            row['review'] == REVIEWS[name] and file_sha(row['source_path']) == file_sha(row['source_copy']) == sha, 'Reviewed source differs')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference differs')


def pools(directory):
    result = []
    for number in range(3):
        values = [ast.literal_eval(n.value) for n in ast.walk(source_tree(directory, number)) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'g' and t.attr == 'etf_pool' for t in n.targets)]
        require(len(values) == 1, 'Original pool ambiguous'); result.append(values[0])
    require(list(map(len, result)) == [13, 30, 30] and result[1] == result[2], 'Original pools differ')
    return result


def inventory(root, directory):
    originals = pools(directory); assets = sorted({s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for p in originals for s in p})
    store = Store(root); state = store.state(SNAPSHOT); tables = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage', 'instrument_status'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', assets)])
        tables[table] = {s: int(frame.instrument.eq(s).sum()) if 'instrument' in frame else 0 for s in assets}
    return {'snapshot': SNAPSHOT, 'original_pools': originals, 'requested_assets': assets, 'tables': tables,
        'not_a_backtest': True, 'limits': ['Stock operands do not replace original funds,3500-row availability, calendars or intraday execution']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    price = pd.read_parquet(directory / 'price-input.parquet')
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, **profiles['price-input.parquet'], 'profiles': profiles,
        'pool': sorted(set(price.instrument)), 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Ten verified stocks are formula operands; complete traded-row windows do not prove original ETF histories',
            'RangeIndex operands expose original integer-label arithmetic; platform DateTimeIndex compatibility is diagnosed separately']})
    save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch45 accepted arithmetic inputs and original fund inventory frozen')


def inputs(directory): return previous.inputs(directory)


def function_ast(directory, number, name):
    return next(n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name == name)


def namespaces(directory):
    result = []
    for number in range(3):
        ns = {'np': np, 'pd': pd, 'solve': solve, 'math': math,
            'g': SimpleNamespace(m_days=34 if number == 0 else 25, stock_num=3, _lambda=10, w=.2)}
        names = ('get_rank', 'epo', 'run_optimization', 'trade') if number == 0 else (
            'calculate_ma', 'count_days_above', 'get_trend_length', 'min_corr', 'get_rank', 'trade', 'mingcheng', 'initialize')
        if number == 2: names += ('up_strength',)
        selected(directory, number, names, ns)
        if number:
            fn = deepcopy(function_ast(directory, number, 'min_corr'))
            require(isinstance(fn.body[-1], ast.Return) and isinstance(fn.body[-1].value, ast.Name) and fn.body[-1].value.id == 'etf_pool', 'Original correlation return differs')
            fn.name = 'min_corr_details'
            fn.body[-1].value = ast.Dict(keys=[ast.Constant('selected'), ast.Constant('scores')],
                values=[ast.Name(id='etf_pool', ctx=ast.Load()), ast.Name(id='corr_mean', ctx=ast.Load())])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<original-correlation-details>', 'exec'), ns)
        result.append(ns)
    return result


def shared_kernel_evidence(directory):
    rows = {}
    for name in ('calculate_ma', 'count_days_above', 'get_trend_length', 'min_corr', 'get_rank'):
        text = [ast.dump(function_ast(directory, number, name)) for number in (1, 2)]
        require(text[0] == text[1], 'Daily/monthly numerical kernels differ')
        rows[name] = hashlib.sha256(text[0].encode()).hexdigest()
    return rows


def rank_kernel(directory, number):
    loop = next(n for n in function_ast(directory, number, 'get_rank').body if isinstance(n, ast.For))
    nodes = [n for n in loop.body if isinstance(n, ast.Assign)]; require(len(nodes) == 7, 'Original score assignments differ')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, '<original-trend-rank>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def original_score(values, code, count):
    require(len(values) == count and np.isfinite(values).all() and min(values) > 0, 'Incomplete score operands')
    ns = {'np': np, 'math': math, 'g': SimpleNamespace(m_days=count), 'etf': 'sample',
        'attribute_history': lambda *a, **kw: pd.DataFrame({'close': values})}
    exec(code, ns); return [float(ns[k]) for k in ('annualized_returns', 'r_squared', 'score')]


def reference_score(values):
    y = list(map(math.log, values)); n = len(y); mid = (n - 1) / 2; mean = math.fsum(y) / n
    slope = math.fsum((k - mid) * (v - mean) for k, v in enumerate(y)) / math.fsum((k - mid) ** 2 for k in range(n))
    intercept = mean - slope * mid; annual = math.expm1(slope * 250); total = math.fsum((v - mean) ** 2 for v in y)
    r2 = 1 - math.fsum((v - (slope * k + intercept)) ** 2 for k, v in enumerate(y)) / total if total else math.nan
    return [annual, r2, annual * r2]


def reference_epo(prices):
    values = prices.to_numpy(); cov, means = previous.covariance(values[1:] / values[:-1] - 1)
    diagonal = np.diag(cov); anchor = (1 / diagonal) / np.sum(1 / diagonal); shrunk = .8 * cov + .2 * np.diag(diagonal)
    signal_portfolio = np.linalg.solve(shrunk, means)
    gamma = math.sqrt(anchor @ shrunk @ anchor) / math.sqrt(signal_portfolio @ shrunk @ signal_portfolio)
    weights = .8 * gamma * signal_portfolio + .2 * np.linalg.solve(shrunk, diagonal * anchor)
    weights = np.maximum(weights, 0); require(np.isfinite(weights).all() and weights.sum() > 0, 'Degenerate EPO reference')
    return weights / weights.sum()


def reference_correlation(prices):
    cov, _ = previous.covariance(np.diff(np.log(prices.to_numpy()), axis=0)); std = np.sqrt(np.diag(cov))
    require(np.isfinite(cov).all() and np.all(std > 0), 'Degenerate correlation arithmetic')
    scores = np.abs(cov / std[:, None] / std[None, :]).mean(axis=0); names = list(prices.columns)
    return sorted(names, key=lambda s: scores[names.index(s)])[:4], scores


def stable_ma(values, count):
    return np.array([math.nan] * (count - 1) + [math.fsum(map(float, values[k - count + 1:k + 1])) / count for k in range(count - 1, len(values))])


def cumulative_trend(flags):
    require(len(flags) > 0, 'Empty trend comparison')
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(flags, dtype=bool), False]))
    lengths = edges[1::2] - edges[::2]
    return sum(int(n) * (int(n) + 1) // 2 for n in lengths) / len(flags)


def compute(directory):
    validate_sources(directory); price, excluded = inputs(directory); ns = namespaces(directory); shared = shared_kernel_evidence(directory)
    wide = price.pivot(index='date', columns='instrument', values='close_adj').sort_index()
    panel = wide.loc[:, list(SAMPLE3)].dropna(); epo_maximum = 0.; zero_boundaries = []
    require(len(panel) >= 1200 and len(wide.dropna()) >= 243, 'Insufficient complete arithmetic panels')
    for k in range(1199, len(panel)):
        sample = panel.iloc[k - 1199:k + 1]; actual = previous.original_epo(sample, ns[0], 1200); expected = reference_epo(sample)
        require(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=1e-9, atol=1e-11), 'Anchored EPO differs')
        epo_maximum = max(epo_maximum, float(np.max(np.abs(actual - expected))))
        if list(actual > 0) != list(expected > 0): zero_boundaries.append(str(panel.index[k]))
    common = wide.dropna(); selection_boundaries = []; max_corr = 0.
    for k in range(242, len(common)):
        sample = common.iloc[k - 242:k + 1]; ns[1]['history'] = lambda *a, **kw: sample.copy()
        actual = ns[1]['min_corr_details'](list(sample)); expected, scores = reference_correlation(sample)
        for s, score in zip(sample.columns, scores, strict=True): max_corr = max(max_corr, abs(actual['scores'][s] - score))
        if actual['selected'] != expected: selection_boundaries.append({'date': str(common.index[k]), 'original': actual['selected'], 'reference': expected})
    ranks = []; trends = []; comparisons = []; trend_boundaries = []; ineligible = []
    for number, count in ((0, 34), (1, 25)):
        code, sha = rank_kernel(directory, number); maximums = [0.] * 3; windows = 0; boundaries = []; nonfinite = []
        for instrument, group in price.groupby('instrument', sort=True):
            group = group.sort_values('date').reset_index(drop=True); values = group.close_adj.to_numpy()
            for k in range(count - 1, len(group)):
                sample = values[k - count + 1:k + 1]; actual = original_score(sample, code, count); expected = reference_score(sample)
                if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                    nonfinite.append({'instrument': instrument, 'date': str(group.date.iloc[k])}); continue
                for j, (a, b) in enumerate(zip(actual, expected, strict=True)):
                    require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), 'Momentum score differs'); maximums[j] = max(maximums[j], abs(a - b))
                if ((actual[2] > 0) if number == 0 else (-.5 < actual[2] < 4.5)) != ((expected[2] > 0) if number == 0 else (-.5 < expected[2] < 4.5)):
                    boundaries.append({'instrument': instrument, 'date': str(group.date.iloc[k])})
                windows += 1
        ranks.append({'price_rows': count, 'windows': windows, 'max_differences': maximums, 'condition_boundaries': boundaries, 'nonfinite_windows': nonfinite, 'ast_sha256': sha})
    for instrument, group in price.groupby('instrument', sort=True):
        group = group.sort_values('date').reset_index(drop=True); values = group.close_adj.to_numpy()
        if len(group) < 3500:
            ineligible.append({'instrument': instrument, 'verified_traded_rows': len(group), 'required': 3500}); continue
        ma10 = stable_ma(values, 10); ma30 = stable_ma(values, 30); windows = 0; maximum = 0.
        for k in range(3499, len(group)):
            sample = pd.Series(values[k - 3499:k + 1]); actual = ns[1]['count_days_above'](sample, 10, 30)
            short = ns[1]['calculate_ma'](sample, 10).to_numpy()[30:]; long = ns[1]['calculate_ma'](sample, 30).to_numpy()[30:]
            reference_flags = ma10[k - 3469:k + 1] > ma30[k - 3469:k + 1]
            changed = np.flatnonzero((short > long) != reference_flags)
            if len(changed): comparisons.append({'instrument': instrument, 'date': str(group.date.iloc[k]), 'window_positions': (changed + 30).tolist()})
            expected = cumulative_trend(reference_flags); maximum = max(maximum, abs(actual - expected))
            require(actual == cumulative_trend(short > long), 'Original cumulative state differs')
            if (actual > 3) != (expected > 3): trend_boundaries.append({'instrument': instrument, 'date': str(group.date.iloc[k]), 'original': actual, 'reference': expected})
            windows += 1
        trends.append({'instrument': instrument, 'windows': windows, 'first': str(group.date.iloc[3499]), 'last': str(group.date.iloc[-1]), 'max_ratio_difference': maximum})
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False, 'source_sha256': list(EXPECTED_SOURCE_SHA),
        'shared_daily_monthly_kernel_sha256': shared, 'epo1200': {'windows': len(panel) - 1199, 'sample_instruments': list(panel),
            'first': str(panel.index[1199]), 'last': str(panel.index[-1]), 'max_weight_difference': epo_maximum,
            'zero_weight_boundaries': zero_boundaries},
        'correlation243': {'windows': len(common) - 242, 'first': str(common.index[242]), 'last': str(common.index[-1]),
            'max_score_difference': max_corr, 'selection_boundaries': selection_boundaries},
        'ranks': ranks, 'trend3500': {'ranges': trends, 'ineligible': ineligible, 'mean_condition_boundaries': comparisons, 'filter_boundaries': trend_boundaries},
        'excluded_known_suspended_daily_rows': excluded,
        'limits': ['All original numerical statements retained; RangeIndex arithmetic is not platform history index compatibility',
            'Stock samples are formula operands only; no original ETF selection, fills, costs or NAV produced',
            'Shared kernel AST identity proves numerical code identity only; daily recompute and monthly cache state differ']}


def diagnostics(directory):
    ns = namespaces(directory); rows = []
    def add(case, **kw): rows.append(previous.json_value({'case': case, **deepcopy(kw)}))
    t = np.arange(1200, dtype=float); prices = pd.DataFrame({f'S{k}': 100 * np.exp(.0001 * (k + 1) * t + .01 * np.sin(t / (k + 2))) for k in range(3)})
    calls = []
    def requested(*a, **kw): calls.append({'args': list(a), 'kwargs': kw}); return {'close': prices.copy()}
    ns[0]['get_price'] = requested; weights = ns[0]['run_optimization'](list(prices), '2021-01-04')
    add('epo17_original1200_request_and_w02',requests=calls,weights=weights,reference=reference_epo(prices))
    returns = prices.pct_change().dropna(); diagonal = np.diag(returns.cov()); anchor = (1 / diagonal) / sum(1 / diagonal)
    add('epo17_endogenous_anchored_ignores_lambda',weights10=ns[0]['epo'](returns, returns.mean(), 10, 'anchored', .2, anchor),
        weights40=ns[0]['epo'](returns, returns.mean(), 40, 'anchored', .2, anchor))
    zero = pd.DataFrame(np.tile([-1., 1.], (3, 1)).T.repeat(5, axis=0), columns=['A', 'B', 'C'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); result = ns[0]['epo'](zero, np.zeros(3), 10, 'anchored', .2, np.ones(3) / 3)
    add('epo17_zero_signal_gamma_not_regularized',weights=result,warnings=[str(w.message) for w in caught])
    ns[0]['attribute_history'] = lambda s, *a, **kw: pd.DataFrame({'close': 100 * np.exp((.01 if s == 'strong' else -.001) * np.arange(34))})
    add('rank17_positive_only_no_upper_score_cap',rank=ns[0]['get_rank'](['negative', 'strong']))
    orders = []; current = SimpleNamespace(previous_date='2021-01-04', portfolio=SimpleNamespace(total_value=10000., available_cash=10000., positions={'old': SimpleNamespace(total_amount=100)}))
    ns[0]['g'].etf_pool = ['A', 'B', 'C', 'D']; ns[0]['get_rank'] = lambda *a: ['B', 'C', 'D', 'A']
    ns[0]['run_optimization'] = lambda stocks, end: np.array([.2, .3, .5]); ns[0]['order_target_value'] = lambda *a: orders.append(list(a)); ns[0]['trade'](current)
    add('epo17_sell_then_three_nav_targets_without_fill_check',orders=orders)
    orders.clear(); ns[0]['get_rank'] = lambda *a: []; requested_empty = []
    def empty(stocks, end): requested_empty.append(stocks); raise ValueError('Synthetic provider rejects empty selection')
    ns[0]['run_optimization'] = empty
    try: ns[0]['trade'](current)
    except ValueError as exc: add('epo17_empty_selection_still_optimizes_after_sell',orders=orders,requested=requested_empty,error=str(exc))
    for number in (1, 2):
        fresh = namespaces(directory)[number]
        rising = pd.Series(np.arange(3500, dtype=float) + 100)
        add(f'trend{number}_3500_triangular_not_average_run',ratio=fresh['count_days_above'](rising, 10, 30),reference=3471 / 2)
        dated = rising.copy(); dated.index = pd.date_range('2005-01-01', periods=3500)
        try: fresh['count_days_above'](dated, 10, 30)
        except KeyError as exc: add(f'trend{number}_datetime_integer_label_incompatible',error=f'KeyError: {exc}')
        try: fresh['count_days_above'](rising.iloc[:30], 10, 30)
        except ZeroDivisionError as exc: add(f'trend{number}_30_rows_zero_denominator',error=f'ZeroDivisionError: {exc}')
    calls = []; fresh = namespaces(directory)[1]
    fresh['history'] = lambda *a, **kw: (calls.append({'args': list(a), 'kwargs': kw}) or pd.DataFrame({'equal': [3.], 'above': [3.01]}))
    fresh['count_days_above'] = lambda data, *a: data.iloc[0]
    add('trend_history3500_pre_skip_paused_and_strict_gt3',selected=fresh['get_trend_length'](['equal', 'above'], 3),requests=calls)
    t = np.arange(243, dtype=float); p = pd.DataFrame({'A': 100 * np.exp(.01 * np.sin(t)), 'B': 100 * np.exp(.012 * np.sin(t + .5))})
    fresh = namespaces(directory)[1]; fresh['history'] = lambda *a, **kw: p.copy()
    add('corr243_absolute_self_tie_stable_order',selected=fresh['min_corr'](list(p)))
    missing = p.copy(); missing.loc[3, 'A'] = np.nan; fresh['history'] = lambda *a, **kw: missing.copy()
    add('corr243_one_missing_price_drops_column',selected=fresh['min_corr'](list(missing)))
    fresh.pop('math'); fresh['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(25) + 100.})
    try: fresh['get_rank'](['A'])
    except NameError as exc: add('rank_platform_math_injection',error=f'NameError: {exc}')
    calls = []; candidate_seen = []
    for number in (1, 2):
        fresh = namespaces(directory)[number]; fresh['g'].etf_pool = ['A', 'B']; fresh['g'].trend_length_ranking = ['cached']
        fresh['get_trend_length'] = lambda *a: (calls.append(number) or ['current'])
        fresh['min_corr'] = lambda names: (candidate_seen.append([number, list(names)]) or [])
        fresh['get_rank'] = lambda *a: []; fresh['order_target_value'] = lambda *a: None; fresh['trade'](current)
    add('daily74_recomputes_monthly97_reads_cache',trend_calls=calls,correlation_candidates=candidate_seen)
    fresh = namespaces(directory)[2]; fresh['g'].etf_pool = ['A', 'B']; fresh['get_trend_length'] = lambda *a: ['updated']
    fresh['up_strength'](current); add('monthly97_up_strength_replaces_cache',cache=fresh['g'].trend_length_ranking)
    scheduled = []; fresh = namespaces(directory)[2]
    for name in ('set_slippage', 'set_order_cost', 'set_benchmark', 'set_option'):
        fresh[name] = lambda *a, **kw: None
    fresh['FixedSlippage'] = lambda *a, **kw: object(); fresh['OrderCost'] = lambda **kw: object(); fresh['log'] = SimpleNamespace(set_level=lambda *a: None)
    fresh['get_trend_length'] = lambda *a: ['initialized']
    for name in ('run_daily', 'run_monthly'):
        fresh[name] = lambda fn, *a, _name=name, **kw: scheduled.append([_name, fn.__name__, list(a), kw])
    fresh['initialize'](current); add('monthly97_initialize_then_monthfirst9_and_daily10',cache=fresh['g'].trend_length_ranking,schedules=scheduled)
    fresh = namespaces(directory)[1]; fresh['g'].etf_pool = ['A', 'B']; fresh['get_trend_length'] = lambda *a: ['new']; fresh['min_corr'] = lambda *a: ['new']; fresh['get_rank'] = lambda *a: ['new']
    orders = []; fresh['order_target_value'] = lambda *a: orders.append(list(a)); fresh['trade'](current)
    add('trend_rejected_sell_blocks_new_buy',orders=orders)
    current.portfolio.positions = {}; orders.clear()
    try: fresh['trade'](current)
    except KeyError as exc: add('trend_empty_positions_requires_zero_proxy',error=f'KeyError: {exc}',orders=orders)
    return {'not_a_backtest': True, 'platform_equivalent': False, 'cases': rows,
        'limits': ['Synthetic prices/positions/provider responses and recorded intents only; no fills/costs/NAV',
            'Unchanged original functions; RangeIndex is arithmetic input shape only, not a fix of platform history']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory)
    with redirect_stdout(io.StringIO()): diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch45 original long EPO/trend/rank/correlation arithmetic and state diagnoses frozen')


def worker(root, directory, endpoint):
    import akshare as ak
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
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()): frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive sample'); path = raw.save(root, 'batch45_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}], strict_usable=False, limit=query['limit'])
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch45', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint,
                    'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False,
                    'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch45 three original ETF supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch45 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline component differs')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch45 forbidden-network arithmetic/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch45.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch45_research.py',
            'tests/unit/test_batch44_research.py', 'tests/unit/test_catalog_margin_weekly.py', 'tests/unit/test_catalog_bars.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def checks(root, directory):
    before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
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
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked state differs')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory); validate_sources(directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(read(directory / 'dependency-inventory.json') == inventory(root, directory), 'Original fund inventory differs')
    with redirect_stdout(io.StringIO()): require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed component differs')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch45 original momentum/EPO and trend research accepted; ETF trade inputs remain missing')
    save(Path('docs/handoff/2026-10-07-batch45-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch45/20261007-momentum-epo-trend')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
