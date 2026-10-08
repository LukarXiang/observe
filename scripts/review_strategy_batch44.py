"""Archive original EPO/correlation rules and bounded dependency research."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import datetime as dt
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
import scipy
from scipy.linalg import solve

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts import review_strategy_batch38 as stocks
from scripts import review_strategy_batch43 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch43/20261007-candle-weekly-margin')
RECEIPT = Path('docs/handoff/2026-10-07-batch43-verification.json')
SOURCES = ('2024年度精选策略1/52.增强型投资组合优化（EPO）方法试用.txt',
    '聚宽2025年精选/32EPO优化低相关etf组合.txt',
    '聚宽2025年精选/26波动率过滤后相关性最小etf轮动.txt')
EXPECTED_SOURCE_SHA = ('0baad6159f87bc1cef0160f4eef1c69cda21ff018dc91b1d57a0837440148160',
    '64074440cb797c6f2dd94bb4063730c632c89b2e9eb59386ded1adaf39a0746e',
    'bb629ecfcfe853369ac5d3192c1377688c5626a9be944758801ad945e4ecbf1b')
REFERENCES = (('repo/skfolio', 'src/skfolio/moments/covariance/_shrunk_covariance.py'),
    ('repo/skfolio', 'src/skfolio/moments/covariance/_empirical_covariance.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_sina.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch44.py', 'tests/unit/test_batch44_research.py'}))
CORE = {'baseline.json', 'catalog-before.json', 'strategy_catalog.before.py', 'input-binding.json',
    'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json', 'dependency-inventory.json',
    'price-input.parquet', 'index-input.parquet', 'calendar-input.parquet', 'component-research.json',
    'diagnostics.json', 'probe-results.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'gold_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '518880', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'One gold ETF sample cannot prove four/ten/thirty original funds, events or10:00 execution'},
    'dividend_sina': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh510880'},
        'limit': 'One dividend ETF daily sample is not original histories, price adjustment or fund execution rules'},
    'nasdaq_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '513100', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'Overseas ETF bars cannot prove historical QDII calendars, events,10:00 fills or complete pools'}}
SAMPLE4 = ('600036.SH', '600085.SH', '600196.SH', '600519.SH')


def implementation(): return {name: file_sha(name) for name in FILES}


def preflight(root, directory):
    checkpoint(root, directory, 'Batch44 original EPO/correlation source research started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok', 'not_a_backtest': True}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch43 differs')
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
    code = inspect.getsource(solve)
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
        'queries': QUERIES, 'apis': apis, 'solver': {'function': 'scipy.linalg.solve',
            'signature': str(inspect.signature(solve)), 'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()}}


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
            'use': 'Read-only covariance/interface reference; estimator differs from original EPO and is not substituted'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch44 three complete EPO/correlation source reviews frozen')


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
        tree = source_tree(directory, number)
        values = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'g' and t.attr == 'etf_pool' for t in n.targets)]
        require(len(values) == 1, 'Original pool ambiguous'); result.append(values[0])
    require(list(map(len, result)) == [4, 10, 30], 'Original pool sizes differ')
    return result


def inventory(root, directory):
    originals = pools(directory); assets = sorted({s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for p in originals for s in p})
    store = Store(root); state = store.state(SNAPSHOT); tables = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage', 'instrument_status'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', assets)])
        tables[table] = {s: int(frame.instrument.eq(s).sum()) if 'instrument' in frame else 0 for s in assets}
    return {'snapshot': SNAPSHOT, 'original_pools': originals, 'requested_assets': assets, 'tables': tables,
        'not_a_backtest': True, 'limits': ['No sample stock pool replaces original ETFs or their events/calendars/10:00 execution']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    price = pd.read_parquet(directory / 'price-input.parquet')
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, **profiles['price-input.parquet'], 'profiles': profiles,
        'pool': sorted(set(price.instrument)), 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Ten verified stocks are arithmetic operands only, not an ETF economic variant',
            'Multiasset samples use common verified traded rows; this does not prove original platform calendar or pause semantics']})
    save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch44 accepted arithmetic inputs and actual37-fund inventory frozen')


def inputs(directory):
    price, excluded = stocks.inputs(directory); doc = read(directory / 'input-analysis.json')
    for name, row in doc['profiles'].items():
        path = directory / name; require(row['file'] == str(path) and file_sha(path) == row['sha256'], 'Operand differs')
        frame = pd.read_parquet(path)
        require(len(frame) == row['rows'] and list(frame.columns) == row['columns'] and frame.date.min() == row['first'] and frame.date.max() == row['last'], 'Operand profile differs')
    require(np.isfinite(price.close_adj).all() and price.close_adj.gt(0).all(), 'Invalid close operands')
    return price, excluded


def namespaces(directory):
    result = []
    for number in range(3):
        ns = {'np': np, 'pd': pd, 'solve': solve, 'math': math, 'datetime': dt, 'g': SimpleNamespace(m_days=25)}
        names = ('epo', 'run_optimization', 'trade') if number < 2 else ('min_corr', 'get_rank', 'trade')
        if number == 1: names += ('filter_new_stock',)
        selected(directory, number, names, ns)
        if number == 2:
            fn = deepcopy(next(n for n in source_tree(directory, 2).body if isinstance(n, ast.FunctionDef) and n.name == 'min_corr'))
            require(isinstance(fn.body[-1], ast.Return) and isinstance(fn.body[-1].value, ast.Name) and fn.body[-1].value.id == 'etf_pool', 'Original correlation return differs')
            fn.name = 'min_corr_details'
            # Expose original locals without changing any filtering or sorting statement.
            fn.body[-1].value = ast.Dict(keys=[ast.Constant(k) for k in ('selected', 'volatility', 'scores')],
                values=[ast.Name(id=k, ctx=ast.Load()) for k in ('etf_pool', 'v', 'corr_mean')])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<original-correlation-details>', 'exec'), ns)
        result.append(ns)
    return result


def covariance(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and len(values) >= 2 and np.isfinite(values).all(), 'Incomplete covariance operands')
    means = np.array([math.fsum(map(float, column)) / len(values) for column in values.T])
    centered = values - means
    return centered.T @ centered / (len(values) - 1), means


def original_epo(prices, ns, count):
    require(len(prices) == count and np.isfinite(prices).all().all() and prices.gt(0).all().all(), 'Incomplete EPO price operands')
    ns['get_price'] = lambda *a, **kw: {'close': prices.copy()}
    weights = ns['run_optimization'](list(prices.columns), None)
    return np.asarray(list(weights.values()) if isinstance(weights, dict) else weights, dtype=float)


def reference_epo(prices, anchored):
    p = prices.to_numpy(dtype=float); returns = p[1:] / p[:-1] - 1
    cov, means = covariance(returns); diagonal = np.diag(cov); anchor = (1 / diagonal) / np.sum(1 / diagonal)
    if anchored:
        require(np.isfinite(means).all() and means @ np.diag(1 / diagonal) @ means > 0, 'Degenerate anchored signal')
        return anchor
    shrunk = .4 * cov + .6 * np.diag(diagonal)
    weights = np.linalg.solve(shrunk, anchor) / 10
    weights = np.maximum(weights, 0)
    require(weights.sum() > 0, 'Degenerate simple weights')
    return weights / weights.sum()


def original_correlation(prices, ns, details=False):
    require(len(prices) == 729 and np.isfinite(prices).all().all() and prices.gt(0).all().all(), 'Incomplete729-row operands')
    ns['history'] = lambda *a, **kw: prices.copy()
    return ns['min_corr_details' if details else 'min_corr'](list(prices.columns))


def reference_correlation(prices):
    returns = np.diff(np.log(prices.to_numpy()), axis=0); cov, _ = covariance(returns)
    volatility = np.sqrt(np.diag(cov)) * math.sqrt(243); selected_cols = (volatility > .05) & (volatility < .33)
    names = list(prices.columns[selected_cols]); cov = cov[np.ix_(selected_cols, selected_cols)]
    if not names: return [], volatility.tolist(), []
    std = np.sqrt(np.diag(cov)); corr = cov / std[:, None] / std[None, :]
    scores = np.abs(corr).mean(axis=0)
    return sorted(names, key=lambda s: scores[names.index(s)])[:4], volatility.tolist(), scores.tolist()


def rank_kernel(directory):
    fn = next(n for n in source_tree(directory, 2).body if isinstance(n, ast.FunctionDef) and n.name == 'get_rank')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    nodes = [n for n in loop.body if isinstance(n, ast.Assign)]
    require(len(nodes) == 7, 'Original score assignment structure differs')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, '<original-correlation-rank>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def original_score(values, code):
    require(len(values) == 25 and np.isfinite(values).all() and min(values) > 0, 'Incomplete25-row score operands')
    ns = {'np': np, 'math': math, 'g': SimpleNamespace(m_days=25), 'etf': 'sample',
        'attribute_history': lambda *a, **kw: pd.DataFrame({'close': values})}
    exec(code, ns)
    return [float(ns[k]) for k in ('annualized_returns', 'r_squared', 'score')]


def reference_score(values):
    y = list(map(math.log, values)); mean = math.fsum(y) / 25
    sx = math.fsum((k - 12) ** 2 for k in range(25))
    slope = math.fsum((k - 12) * (v - mean) for k, v in enumerate(y)) / sx
    intercept = mean - slope * 12; annual = math.expm1(slope * 250)
    total = math.fsum((v - mean) ** 2 for v in y)
    r2 = 1 - math.fsum((v - (slope * k + intercept)) ** 2 for k, v in enumerate(y)) / total if total else math.nan
    return [annual, r2, annual * r2]


def compute(directory):
    validate_sources(directory); price, excluded = inputs(directory); ns = namespaces(directory)
    wide = price.pivot(index='date', columns='instrument', values='close_adj').sort_index()
    panel4 = wide.loc[:, list(SAMPLE4)].dropna(); panel10 = wide.dropna()
    require(len(panel4) >= 1200 and panel10.shape[1] == 10 and len(panel10) >= 729, 'Insufficient complete arithmetic panels')
    epo_results = []
    for number, panel, count in ((0, panel4, 1200), (1, panel10, 250)):
        differences = 0.; windows = 0; zero_boundaries = []
        for k in range(count - 1, len(panel)):
            sample = panel.iloc[k - count + 1:k + 1]
            actual = original_epo(sample, ns[number], count); expected = reference_epo(sample, number == 0)
            require(actual.shape == expected.shape and np.isfinite(actual).all() and
                np.allclose(actual, expected, rtol=1e-10, atol=1e-12) and abs(actual.sum() - 1) < 1e-12, 'EPO weights differ')
            differences = max(differences, float(np.max(np.abs(actual - expected))))
            if list(actual > 0) != list(expected > 0): zero_boundaries.append(str(panel.index[k]))
            windows += 1
        epo_results.append({'source': SOURCES[number], 'price_rows': count, 'windows': windows,
            'sample_instruments': list(panel.columns), 'first': str(panel.index[count - 1]), 'last': str(panel.index[-1]),
            'max_weight_difference': differences, 'zero_weight_boundaries': zero_boundaries})
    selections = []; corr_windows = 0; volatility_boundaries = []; max_corr = 0.; max_volatility = 0.
    for k in range(728, len(panel10)):
        sample = panel10.iloc[k - 728:k + 1]
        actual = original_correlation(sample, ns[2], details=True); expected, volatility, scores = reference_correlation(sample)
        admitted = [s for s, v in zip(sample.columns, volatility, strict=True) if .05 < v < .33]
        original_admitted = list(actual['volatility'].index)
        if original_admitted != admitted:
            volatility_boundaries.append({'date': str(panel10.index[k]), 'original': original_admitted, 'reference': admitted})
        else:
            for s in admitted:
                max_volatility = max(max_volatility, abs(float(actual['volatility'][s]) - volatility[list(sample.columns).index(s)]))
            for s, score in zip(admitted, scores, strict=True): max_corr = max(max_corr, abs(float(actual['scores'][s]) - score))
        if actual['selected'] != expected: selections.append({'date': str(panel10.index[k]), 'original': actual['selected'], 'reference': expected})
        corr_windows += 1
    code, sha = rank_kernel(directory); max_score = [0.] * 3; boundaries = []; score_windows = 0; nonfinite = []
    ranges = []
    for instrument, group in price.groupby('instrument', sort=True):
        group = group.sort_values('date').reset_index(drop=True); values = group.close_adj.to_numpy(); local = 0
        for k in range(24, len(group)):
            sample = values[k - 24:k + 1]; actual = original_score(sample, code); expected = reference_score(sample)
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                nonfinite.append({'instrument': instrument, 'date': str(group.date.iloc[k]), 'reason': 'original_or_reference_nonfinite_score'}); continue
            for j, (a, b) in enumerate(zip(actual, expected, strict=True)):
                require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), 'Momentum score differs'); max_score[j] = max(max_score[j], abs(a - b))
            if (-.5 < actual[2] < 4.5) != (-.5 < expected[2] < 4.5): boundaries.append({'instrument': instrument, 'date': str(group.date.iloc[k])})
            score_windows += 1; local += 1
        ranges.append({'instrument': instrument, 'first': str(group.date.iloc[24]), 'last': str(group.date.iloc[-1]), 'windows': local})
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False, 'source_sha256': list(EXPECTED_SOURCE_SHA),
        'epo': epo_results, 'correlation729': {'windows': corr_windows, 'first': str(panel10.index[728]),
            'last': str(panel10.index[-1]), 'selection_boundaries': selections, 'volatility_boundaries': volatility_boundaries,
            'max_admitted_volatility_difference': max_volatility, 'max_correlation_score_difference': max_corr,
            'sample_instruments': list(panel10.columns)},
        'momentum25': {'windows': score_windows, 'ranges': ranges, 'max_differences': max_score, 'condition_boundaries': boundaries,
            'nonfinite_windows': nonfinite, 'ast_sha256': sha}, 'excluded_known_suspended_daily_rows': excluded,
        'limits': ['All stock panels are formula operands only, not original ETF pools or calendar-equivalent histories',
            'Common traded row intersections deliberately do not fill pauses; no trade signals/orders/NAV/cost performance produced']}


def json_value(value):
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [json_value(v) for v in value]
    if isinstance(value, np.generic): return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


def diagnostics(directory):
    ns = namespaces(directory); rows = []
    def add(case, **kw): rows.append(json_value({'case': case, **deepcopy(kw)}))
    t = np.arange(1200, dtype=float)
    prices = pd.DataFrame({f'S{k}': 100 * np.exp(.0001 * (k + 1) * t + .01 * np.sin(t / (k + 2))) for k in range(4)})
    calls = []
    def requested(*a, **kw): calls.append({'args': list(a), 'kwargs': kw}); return {'close': prices.copy()}
    ns[0]['get_price'] = requested; result = ns[0]['run_optimization'](list(prices), '2021-01-04')
    add('epo52_original1200_request_and_inverse_variance_anchor',requests=calls,weights=result,reference=reference_epo(prices, True))
    zero = pd.DataFrame(np.tile([-1., 1.], (10, 1)).T.repeat(5, axis=0), columns=[f'S{k}' for k in range(10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); result = ns[0]['epo'](zero, np.zeros(10), 10, 'anchored', 1, np.ones(10) / 10)
    add('epo52_w1_still_evaluates_zero_signal_gamma',weights=result,warnings=[str(w.message) for w in caught])
    constant = pd.DataFrame({'A': np.zeros(10), 'B': np.arange(10, dtype=float)})
    try: ns[0]['epo'](constant, np.ones(2), 10, 'simple', .6)
    except Exception as exc: add('epo_zero_variance_not_regularized',error=f'{type(exc).__name__}: {exc}')
    r = pd.DataFrame({'A': np.sin(np.arange(50)) * .01, 'B': np.sin(np.arange(50) + .4) * .02})
    matrices = []
    def solver(a, b): matrices.append(a.copy()); return solve(a, b)
    ns[1]['solve'] = solver; result = ns[1]['epo'](r, np.array([.7, .3]), 10, 'simple', .6)
    cov = r.cov().to_numpy(); expected = .4 * cov + .6 * np.diag(np.diag(cov)); unused = .4 * expected + .6 * np.diag(np.diag(cov))
    add('epo32_uses_cov_tilde_not_computed_shrunk_cov',weights=result,solve_matrix=matrices[0],reference=expected,
        unused_matrix=unused,diff_from_unused=float(np.max(np.abs(matrices[0] - unused))))
    ns[1]['solve'] = solve
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); result = ns[1]['epo'](r, np.array([-1., -1.]), 10, 'simple', .6)
    add('epo_all_negative_normalization_can_be_nonfinite',weights=result,warnings=[str(w.message) for w in caught])
    orders = []; current = SimpleNamespace(previous_date=dt.date(2021, 1, 4), portfolio=SimpleNamespace(total_value=10000.))
    ns[0]['g'] = SimpleNamespace(etf_pool=['A', 'B']); ns[0]['run_optimization'] = lambda *a: np.array([.4, .6])
    ns[0]['order_target_value'] = lambda *a: orders.append(list(a)); ns[0]['trade'](current)
    add('epo52_original_pool_order_without_sell_first',orders=orders)
    orders.clear(); ns[1]['g'] = SimpleNamespace(etf_pool=['A', 'B']); ns[1]['filter_new_stock'] = lambda *a: ['B']
    ns[1]['run_optimization'] = lambda *a: {'B': 1.}; ns[1]['order_target_value'] = lambda *a: orders.append(list(a))
    ns[1]['trade'](current); add('epo32_filtered_old_holding_not_explicitly_cleared',orders=orders)
    fresh = {'datetime': dt, 'get_security_info': lambda s: SimpleNamespace(start_date=current.previous_date - dt.timedelta(days=30 if s == 'equal' else 29))}
    selected(directory, 1, ('filter_new_stock',), fresh)
    add('epo32_listing30_inclusive',accepted=fresh['filter_new_stock'](current, ['equal', 'new'], 30))
    fresh.pop('datetime')
    try: fresh['filter_new_stock'](current, ['equal'], 30)
    except NameError as exc: add('epo32_datetime_platform_injection',error=f'NameError: {exc}')
    t = np.arange(729, dtype=float); a = 100 * np.exp(.01 * np.sin(t)); p = pd.DataFrame({'A': a, 'B': a.copy(), 'flat': 100.})
    ns[2]['history'] = lambda *a, **kw: p.copy(); result = ns[2]['min_corr'](list(p))
    add('corr729_abs_self_correlation_ties_keep_input_order',selected=result,reference=reference_correlation(p)[0])
    missing = p.copy(); missing.loc[3, 'A'] = np.nan; ns[2]['history'] = lambda *a, **kw: missing.copy()
    add('corr729_missing_one_price_drops_whole_column',selected=ns[2]['min_corr'](list(missing)))
    ns[2].pop('math')
    try: ns[2]['min_corr'](list(p))
    except NameError as exc: add('corr_math_platform_injection',error=f'NameError: {exc}')
    ns[2]['math'] = math; ns[2]['g'] = SimpleNamespace(m_days=25, etf_pool=['new'])
    ns[2]['min_corr'] = lambda *a: ['new']; ns[2]['get_rank'] = lambda *a: ['new']
    orders = []; ns[2]['order_target_value'] = lambda *a: orders.append(list(a))
    current.portfolio = SimpleNamespace(available_cash=10000., positions={'old': SimpleNamespace(total_amount=100)})
    ns[2]['trade'](current); add('corr_rejected_sell_blocks_new_buy',orders=orders)
    current.portfolio.positions = {}; orders.clear()
    try: ns[2]['trade'](current)
    except KeyError as exc: add('corr_plain_empty_positions_requires_platform_zero_proxy',error=f'KeyError: {exc}',orders=orders)
    code, _ = rank_kernel(directory)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); result = original_score(np.ones(25) * 100, code)
    add('rank25_constant_log_can_be_nonfinite',values=result,warnings=[str(w.message) for w in caught])
    return {'not_a_backtest': True, 'platform_equivalent': False, 'cases': rows,
        'limits': ['Explicitly synthetic prices/positions/responses and recorded intents only; no fills, costs or NAV',
            'Platform datetime/math, missing-holding proxy and price return shape equivalence unproved']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory)
    with redirect_stdout(io.StringIO()): diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch44 EPO/correlation/score long arithmetic and original diagnoses frozen')


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
        require(0 < len(frame) <= 100000, 'Empty/excessive sample'); path = raw.save(root, 'batch44_dependency_probe', endpoint, directory.name, frame)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch44', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint,
                    'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False,
                    'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch44 three original ETF supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch44 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline component differs')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch44 forbidden-network arithmetic/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch44.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch44_research.py',
            'tests/unit/test_batch43_research.py', 'tests/unit/test_catalog_margin_weekly.py', 'tests/unit/test_catalog_bars.py',
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
    protection = protect(root, directory); progress = archive(root, 'Batch44 EPO/correlation research accepted; original ETF trade inputs remain missing')
    save(Path('docs/handoff/2026-10-07-batch44-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch44/20261007-epo-correlation')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
