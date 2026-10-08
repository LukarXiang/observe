"""Freeze ETF candidate-pool and momentum-label research without inventing trades."""
import argparse
import ast
from contextlib import redirect_stdout
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

import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch45 as operands
from scripts import review_strategy_batch46 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = operands.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch46/20261007-full-module-prefix')
RECEIPT = Path('docs/handoff/2026-10-07-batch46-verification.json')
SOURCES = ('聚宽2025年精选/66手把手教你构建ETF策略候选池.py',
    '聚宽2025年精选/86手把手教你构建ETF策略候选池优化版.txt', '2024年度精选策略2/43.ETF动量因子评估.txt')
EXPECTED_SOURCE_SHA = ('68cdb63a4a93fe4d849fd7f8faf8680c76600dbdc0bb01e7d27764c4be9d02a5',
    'af583b792607a041d913a6b3f1cf9084eea8090f7bfb6ea53533f33ac0537fc6',
    '803d0afa67862c0ab4060ed88de841463120c7d8f26989d4a68c0c5b09d2e057')
REFERENCES = (('repo/skfolio', 'examples/clustering/plot_4_nco.py'),
    ('repo/alphalens', 'alphalens/performance.py'), ('repo/akshare', 'akshare/fund/fund_etf_em.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_sina.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch47.py', 'tests/unit/test_batch47_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'price-input.parquet', 'index-input.parquet', 'calendar-input.parquet', 'dependency-inventory.json',
    'component-research.json', 'diagnostics.json', 'probe-results.json', 'component-offline.json',
    'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'pool_history': {'function': 'fund_etf_category_ths', 'parameters': {'symbol': 'ETF', 'date': '20211201'},
        'limit': 'A requested-date list is not proof of historical versions, complete dynamic pools or original runtime dates'},
    'consumer_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '159928', 'period': 'daily',
        'start_date': '20130920', 'end_date': '20211201', 'adjust': ''},
        'limit': 'One raw fund daily sample cannot prove four-fund adjusted inputs, default fields or company actions'},
    'sse50_daily': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh510050'},
        'limit': 'One daily sample does not prove complete pools, original money fields, historical names or calendars'}}
RANGES = (
    {'dates': (36, 38), 'liquidity': (45, 54), 'returns': (71, 77), 'cluster': (88, 95),
     'representatives': (101, 113), 'correlation': (131, 158), 'union': (133, 147), 'age': (177, 179)},
    {'dates': (28, 31), 'liquidity': (34, 43), 'returns': (47, 54), 'cluster': (58, 63),
     'representatives': (65, 72), 'correlation': (76, 99), 'union': (78, 92)},
    {'flat_slopes': (66, 71), 'flat_labels': (90, 95), 'ic': (121, 121), 'dated_slopes': (142, 144),
     'dated_labels': (146, 148), 'concat': (154, 155), 'cross': (158, 165), 'leader': (200, 206)})
SLOPES = (3, 5, 10, 20, 30, 60)
LABELS = (1, 5, 10, 20, 30, 60, 90)
SAMPLE4 = ('000651.SZ', '600036.SH', '600085.SH', '600519.SH')
ANCHOR = dt.date(2026, 9, 29)


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'],
        'Accepted batch46 changed')
    source_receipt = read(previous.RECEIPT); files = {}
    for name in ('price-input.parquet', 'index-input.parquet', 'calendar-input.parquet'):
        path = previous.ACCEPTED / name; require(file_sha(path) == source_receipt['checks']['evidence_sha256'][name], 'Accepted operand changed')
        files[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'final_binding_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'files': files, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    binding(root); checkpoint(root, directory, 'Batch47 original candidate-pool and momentum research started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok', 'not_a_backtest': True}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    numerical = []
    for fn in (KMeans.__init__, KMeans.fit_predict, silhouette_score, np.polyfit, pd.DataFrame.corr, pd.Series.pct_change):
        code = inspect.getsource(fn); numerical.append({'function': f'{fn.__module__}.{fn.__qualname__}',
            'signature': str(inspect.signature(fn)), 'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
        'queries': QUERIES, 'apis': rows, 'numerical': numerical,
        'kmeans_defaults': KMeans(n_clusters=30, random_state=42).get_params(), 'original_block_threads': 1}


def start(root, directory):
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old_path = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name, sha in zip(SOURCES, EXPECTED_SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only KMeans/IC/fund interface reference; original defaults and Pearson IC retained, no NCO or Spearman substitution'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old), 'Source set changed')
    require({r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch47 three complete nontrading source reviews frozen')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 3, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, sha, row in zip(SOURCES, EXPECTED_SOURCE_SHA, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['source_sha256'] == sha and
            row['review'] == REVIEWS[name] and file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source review changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
        commit = subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        require(commit == row['commit'], 'Reference commit changed')


def tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'] == EXPECTED_SOURCE_SHA[number], 'Selected source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def block(directory, number, name):
    lo, hi = RANGES[number][name]; nodes = [n for n in tree(directory, number).body if lo <= n.lineno <= n.end_lineno <= hi]
    if name == 'representatives': nodes = [n for n in nodes if isinstance(n, ast.Assign)]
    require(nodes and min(n.lineno for n in nodes) == lo and max(n.end_lineno for n in nodes) == hi, 'Original AST block range changed')
    module = ast.Module(body=nodes, type_ignores=[])
    return compile(module, f'<original-{number}-{name}>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def run_block(directory, number, name, namespace):
    code, _ = block(directory, number, name)
    with redirect_stdout(io.StringIO()), threadpool_limits(limits=1): exec(code, namespace)


def original_pool(directory):
    return next(ast.literal_eval(n.value) for n in tree(directory, 2).body if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'stkList' for t in n.targets))


def inventory(root, directory):
    assets = [s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for s in original_pool(directory)]
    store = Store(root); state = store.state(SNAPSHOT); tables = {}; master = store.load_state(state, 'instruments')
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage', 'instrument_status'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', assets)])
        tables[table] = {s: int(frame.instrument.eq(s).sum()) if 'instrument' in frame else 0 for s in assets}
    candidates = master[master.instrument.str.match(r'^(?:51|15)\d{4}\.(?:SH|SZ)$')]
    return {'snapshot': SNAPSHOT, 'original_fixed_pool': assets, 'tables': tables, 'registered_tables': sorted(state['tables']),
        'master_kind_counts': master.kind.value_counts().to_dict(), 'master_etf_code_candidates': sorted(candidates.instrument.unique()),
        'not_a_backtest': True, 'limits': ['Code prefixes are candidates only; not historical ETF pool/name/start/end-date versions',
            'Frozen stock operands lack money/start_date/end_date and cannot produce original ETF candidate pools']}


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Operand exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    price = pd.read_parquet(directory / 'price-input.parquet')
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, **profiles['price-input.parquet'], 'profiles': profiles,
        'pool': sorted(set(price.instrument)), 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Ten verified stocks are numeric operands only; no original ETF IC/performance or candidate pool inferred',
            'Original sample2013-09-20/2021-12-01 remains separate from extended stock arithmetic window']})
    save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch47 accepted operands and original fixed/dynamic fund dependencies frozen')


def native(value):
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [native(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else str(float(value))
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def slope_kernel(directory):
    nodes = [n for n in ast.walk(tree(directory, 2)) if isinstance(n, ast.Lambda)]
    require(len(nodes) == 2 and ast.dump(nodes[0]) == ast.dump(nodes[1]), 'Original two slope kernels differ')
    return eval(compile(ast.Expression(body=nodes[0]), '<original-normalized-price-slope>', 'eval'), {'np': np})


def reference_slope(values):
    values = list(map(float, values)); n = len(values); require(n >= 2 and min(values) > 0 and all(map(math.isfinite, values)), 'Invalid slope operands')
    mid = (n - 1) / 2
    return 100 * math.fsum((k - mid) * (v / values[0]) for k, v in enumerate(values)) / math.fsum((k - mid) ** 2 for k in range(n))


def reference_correlation(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y, strict=True) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2: return math.nan
    mx = math.fsum(a for a, _ in pairs) / len(pairs); my = math.fsum(b for _, b in pairs) / len(pairs)
    xx = math.fsum((a - mx) ** 2 for a, _ in pairs); yy = math.fsum((b - my) ** 2 for _, b in pairs)
    return math.fsum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(xx * yy) if xx and yy else math.nan


def dated_matrices(directory, panel):
    require(panel.shape[1] == 4 and panel.index.is_unique and panel.index.is_monotonic_increasing, 'Original cross-section requires four ordered assets')
    ns = {'np': np, 'pd': pd, 'cdata': panel.copy()}
    for name in ('dated_slopes', 'dated_labels'): run_block(directory, 2, name, ns)
    slopes, labels = ns['slps_date'], ns['rets_date']
    run_block(directory, 2, 'concat', ns); run_block(directory, 2, 'cross', ns); run_block(directory, 2, 'leader', ns)
    return slopes, labels, ns


def compute(directory):
    price, excluded = operands.inputs(directory); fn = slope_kernel(directory); slope_rows = []; label_rows = []
    for instrument, group in price.groupby('instrument', sort=True):
        values = group.close_adj.to_numpy(); dates = group.date.astype(str).tolist()
        for count in SLOPES:
            actual = pd.Series(values).rolling(count).apply(fn, raw=True).to_numpy()[count - 1:]
            expected = np.array([reference_slope(w) for w in np.lib.stride_tricks.sliding_window_view(values, count)])
            differences = np.flatnonzero((actual > 0) != (expected > 0))
            slope_rows.append({'instrument': instrument, 'rows': count, 'windows': len(actual), 'first': dates[count - 1], 'last': dates[-1],
                'max_abs_difference': float(np.max(np.abs(actual - expected))), 'strict_positive_differences': len(differences),
                'boundaries': [{'date': dates[int(k) + count - 1], 'original': float(actual[k]), 'reference': float(expected[k])} for k in differences]})
        for count in LABELS:
            actual = (pd.Series(values).shift(-count) / pd.Series(values)).to_numpy()
            expected = np.array([float(values[k + count]) / float(values[k]) for k in range(len(values) - count)])
            require(np.isnan(actual[-count:]).all(), 'Future-label tail was filled')
            label_rows.append({'instrument': instrument, 'horizon': count, 'windows': len(expected), 'first': dates[0],
                'last_decision': dates[-count - 1], 'last_label_observation': dates[-1], 'tail_nan_rows': count,
                'max_abs_difference': float(np.max(np.abs(actual[:-count] - expected)))})
    common = price.pivot(index='date', columns='instrument', values='close_adj').dropna(); pairs = []
    for count in (240, 450):
        sample = common.iloc[-count:]; returns = sample.pct_change().iloc[1:]; corr = returns.corr(); expected = np.array([
            [reference_correlation(returns[a], returns[b]) for b in returns] for a in returns])
        pairs.append({'price_rows': count, 'return_rows': len(returns), 'assets': list(sample), 'first': str(sample.index[0]),
            'last': str(sample.index[-1]), 'max_abs_difference': float(np.max(np.abs(corr.to_numpy() - expected))),
            'strict_gt085_differences': int(np.sum((corr.to_numpy() > .85) != (expected > .85))),
            'original_clusters': 30 if count == 240 else 24, 'observed_numeric_assets': len(sample.columns),
            'original_clustering_possible': False, 'limit': 'Ten stock operands are fewer than original clusters; no reduced cluster count used'})
    panel = price[price.instrument.isin(SAMPLE4)].pivot(index='date', columns='instrument', values='close_adj').reindex(columns=SAMPLE4).dropna()
    slopes, labels, ns = dated_matrices(directory, panel)
    reference_slopes = [panel.apply(lambda col, n=n: col.rolling(n).apply(reference_slope, raw=True)) for n in SLOPES]
    flat_s = pd.DataFrame({f'slp_{n}': frame.to_numpy().reshape(-1) for n, frame in zip(SLOPES, slopes, strict=True)})
    flat_r = pd.DataFrame({f'ret_{n}': frame.to_numpy().reshape(-1) for n, frame in zip(LABELS, labels, strict=True)})
    ic_ns = {'pd': pd, 'slps': flat_s, 'rets': flat_r}; run_block(directory, 2, 'ic', ic_ns)
    reference_ic = np.array([[reference_correlation(a.to_numpy().reshape(-1), b.to_numpy().reshape(-1)) for b in labels] for a in reference_slopes])
    cross_differences = []; valid_cross = 0
    for k, original in enumerate(ns['cors']):
        for i, a in enumerate(reference_slopes):
            for j, b in enumerate(labels):
                ref = reference_correlation(a.iloc[k], b.iloc[k]); actual = original.iloc[i, j]
                require(math.isfinite(ref) == math.isfinite(actual), 'Cross-sectional IC validity differs')
                if math.isfinite(ref): valid_cross += 1; cross_differences.append(abs(actual - ref))
    leader_differences = []
    for k in range(19, len(panel)):
        actual = slopes[3].iloc[k].sort_values(ascending=False).index[0]
        expected = reference_slopes[3].iloc[k].sort_values(ascending=False).index[0]
        if actual != expected: leader_differences.append({'date': str(panel.index[k]), 'original': actual, 'reference': expected})
    result = {'not_a_backtest': True, 'platform_equivalent': False, 'snapshot': SNAPSHOT,
        'source_sha256': list(EXPECTED_SOURCE_SHA), 'kernel_sha256': {f'{i}/{name}': block(directory, i, name)[1] for i in range(3) for name in RANGES[i]},
        'slope_windows': slope_rows, 'future_label_windows': label_rows, 'endpoint_correlation': pairs, 'suspended_rows_excluded': excluded,
        'four_stock_ic': {'assets': list(panel), 'rows': len(panel), 'first': str(panel.index[0]), 'last': str(panel.index[-1]),
            'flatten_order': 'date-major, then original supplied four-column order', 'ic': ic_ns['IC'].to_numpy(),
            'reference_ic': reference_ic, 'max_abs_difference': float(np.max(np.abs(ic_ns['IC'].to_numpy() - reference_ic))),
            'first100_cross_sections': len(ns['cors']), 'cross_finite_cells': valid_cross, 'cross_max_abs_difference': max(cross_differences),
            'leader20_valid_rows': len(panel) - 19, 'leader20_startup_nan_rows_kept': 19, 'leader20_rank_differences': leader_differences,
            'leader_ic': pd.DataFrame(ns['test']).corr().iloc[:1, 1:].to_numpy(),
            'limit': 'Intermediate expression arithmetic only; original flat assignments remain broken and original ETF conclusions unproved'},
        'limits': ['No money/listing histories in frozen operands; no full original candidate pool, clustering, IC or200% trading return',
            'Future ratios remain research labels; no fill, NAV, fee or return-to-strategy implementation']}
    return native(result)


def diagnostic_namespace():
    class FixedDateTime(dt.datetime):
        @classmethod
        def today(cls): return cls(ANCHOR.year, ANCHOR.month, ANCHOR.day)
    return {'np': np, 'pd': pd, 'datetime': SimpleNamespace(date=dt.date, datetime=FixedDateTime),
        'KMeans': KMeans, 'silhouette_score': silhouette_score, 'today': str(ANCHOR), 'end_date': str(ANCHOR)}


def diagnostics(directory):
    rows = []
    def add(case, **data): rows.append({'case': case, **native(data)})
    for number in (0, 1):
        ns = diagnostic_namespace(); master = pd.DataFrame({'start_date': [dt.date(2019, 1, 1), dt.date(2020, 1, 1),
            dt.date(2020, 6, 1), dt.date(2021, 1, 1), dt.date(2023, 1, 1)], 'end_date': [dt.date(2099, 1, 1),
            ANCHOR, ANCHOR - dt.timedelta(days=1), dt.date(2099, 1, 1), dt.date(2099, 1, 1)]}, index=list('ABCDE'))
        requests_seen = []; ns['get_all_securities'] = lambda *a: (requests_seen.append(list(a)) or master.copy())
        run_block(directory, number, 'dates', ns)
        add(f'pool{number}_original_start_end_filters',codes=ns['df'].code.tolist(),requests=requests_seen)
        threshold = 5e7 if number == 0 else 1e7; n = 1000 if number == 0 else 300; calls = []
        ns['df'] = pd.DataFrame({'code': ['equal', 'above', 'below'], 'display_name': ['equal', 'above', 'below']})
        def price(code, **kw):
            calls.append({'code': code, 'kwargs': {k: str(v) if isinstance(v, dt.date) else v for k, v in kw.items()}})
            money = threshold + {'equal': 0, 'above': 100, 'below': -100}[code]
            return pd.DataFrame({'close': np.arange(n, dtype=float) + 100, 'money': money})
        ns['get_price'] = price; run_block(directory, number, 'liquidity', ns)
        add(f'pool{number}_strict_liquidity_and_count',selected=ns['df'].code.tolist(),requests=calls,threshold=threshold)
        ns = diagnostic_namespace(); ns['df'] = pd.DataFrame({'code': ['A', 'B']})
        ns['get_price'] = lambda code, **kw: pd.DataFrame({'close': np.arange(240 if code == 'A' else 239, dtype=float) + 100})
        try: run_block(directory, number, 'returns', ns)
        except ValueError as exc: add(f'pool{number}_ragged_history_not_aligned',error=f'{type(exc).__name__}: {exc}')
        ns = diagnostic_namespace(); ns['prices'] = pd.DataFrame(np.arange(100).reshape(10, 10), columns=[f'S{k}' for k in range(10)])
        try: run_block(directory, number, 'cluster', ns)
        except ValueError as exc: add(f'pool{number}_ten_assets_cannot_fit_original_clusters',error=f'{type(exc).__name__}: {exc}')
        ns = diagnostic_namespace(); ns['prices'] = pd.DataFrame(np.random.default_rng(47).normal(size=(60, 40)), columns=[f'S{k}' for k in range(40)])
        seen = []
        def score(frame, values):
            value = silhouette_score(frame, values); seen.append({'columns': len(frame.columns), 'contains_cluster_id': 'cluster_id' in frame, 'score': value}); return value
        ns['silhouette_score'] = score; run_block(directory, number, 'cluster', ns)
        with threadpool_limits(limits=1): unlabelled_score = silhouette_score(ns['prices'].T, ns['y_pred'])
        add(f'pool{number}_installed_kmeans_and_silhouette_features',params=ns['cluster'].get_params(),labels=ns['y_pred'].tolist(),
            calls=seen,return_features=60,unlabelled_score=unlabelled_score)
        codes = list('ABCD'); angles = np.deg2rad([60, 0, 20, 40]); corr = pd.DataFrame(np.cos(angles[:, None] - angles), index=codes, columns=codes)
        ns = diagnostic_namespace(); ns.update(df=pd.DataFrame({'code': codes}), corr=corr); run_block(directory, number, 'union', ns)
        add(f'pool{number}_overlapping_sets_not_merged',groups=[sorted(s) for s in ns['union']],matrix=corr.to_numpy())
        ns = diagnostic_namespace(); ns.update(df=pd.DataFrame({'code': ['A', 'B']}), corr=pd.DataFrame([[1, .85], [.85, 1]], index=['A', 'B'], columns=['A', 'B']))
        run_block(directory, number, 'union', ns); add(f'pool{number}_strict_positive085',groups=[sorted(s) for s in ns['union']])
        ns['corr'].iloc[0, 1] = ns['corr'].iloc[1, 0] = -.99; run_block(directory, number, 'union', ns)
        add(f'pool{number}_negative_correlation_not_absolute',groups=[sorted(s) for s in ns['union']])
    ns = diagnostic_namespace(); ns['df'] = pd.DataFrame({'code': list('ABC'), 'start_date': [dt.date(2019, 1, 1),dt.date(2020, 1, 1),dt.date(2020, 1, 2)]})
    run_block(directory, 0, 'age', ns); add('pool66_post2020_filter_only',selected=ns['df'].code.tolist(),pool86_has_executable_age_block='age' in RANGES[1])
    panel = pd.DataFrame(np.arange(480, dtype=float).reshape(120,4)+100, columns=list('ABCD'))
    for name in ('flat_slopes', 'flat_labels'):
        ns = {'np': np, 'pd': pd, 'cdata': panel.copy()}
        try: run_block(directory, 2, name, ns)
        except (TypeError, ValueError) as exc: add(f'factor43_{name}_2d_to_column_incompatible',error_type=type(exc).__name__,error=f'{type(exc).__name__}: {exc}')
    ns = diagnostic_namespace(); ns.pop('np'); ns.update(df=pd.DataFrame({'code':['A']}), get_price=lambda *a,**kw:pd.DataFrame({'close':[100.,101.,102.]}))
    try: run_block(directory, 0, 'returns', ns)
    except NameError as exc: add('pool_np_platform_injection',error=f'NameError: {exc}')
    expression = next(n for n in tree(directory, 2).body if n.lineno == 227)
    try: eval(compile(ast.Expression(body=expression.value),'<original-plot-acf>','eval'),{'slp_20':panel,'ax':object()})
    except NameError as exc: add('factor43_plot_acf_not_imported',error=f'NameError: {exc}')
    slopes, labels, ns = dated_matrices(directory, panel)
    add('factor43_labels_are_ratios_and_tail_missing',first_ratio=labels[1].iloc[0,0],reference=panel.iloc[5,0]/panel.iloc[0,0],tail_nan_rows=int(labels[-1].isna().all(axis=1).sum()))
    add('factor43_leader20_startup_nan_kept',first20_scores=[r[0] for r in ns['test'][:20]],rows=len(ns['test']))
    return {'not_a_backtest': True, 'platform_equivalent': False, 'cases': rows,
        'limits': ['Synthetic provider/master data, prices or matrices; no original fund pool/model/performance',
            'Original statements unchanged; only reviewed blocks run, without plot/display/global warnings setup']}


def study(root, directory):
    binding(root,directory); validate_sources(directory); before=implementation(); result=compute(directory); diagnostic=diagnostics(directory)
    require(before==implementation(),'Study implementation changed'); save(directory/'component-research.json',result); save(directory/'diagnostics.json',diagnostic)
    return archive(root,'Batch47 normalized-price slope, future-label and original candidate state evidence frozen')


def worker(root, directory, endpoint):
    import akshare as ak
    require(api_evidence()==read(directory/'existing-apis.json'),'Probe API changed')
    query=QUERIES[endpoint]; folder=directory/'probe'/endpoint; folder.mkdir(parents=True,exist_ok=False); socket.setdefaulttimeout(8)
    row={'endpoint':endpoint,'query':query,'api_sha256':file_sha(directory/'existing-apis.json'),'status':'failed','published':False,'files':[],'wire':[]}
    original=requests.sessions.Session.request
    def request(session,method,url,**kw):
        require(len(row['wire'])<10,'Response cap exceeded'); session.trust_env=False; kw['timeout']=(8,10)
        response=original(session,method,url,**kw); path=folder/f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file':str(path),'sha256':file_sha(path),'url':response.url,'status':response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session,'request',request),redirect_stdout(io.StringIO()): frame=getattr(ak,query['function'])(**query['parameters'])
        require(0<len(frame)<=100000,'Empty/excessive sample'); path=raw.save(root,'batch47_dependency_probe',endpoint,directory.name,frame)
        row.update(status='sample',files=[{'file':str(path),'sha256':file_sha(path),'rows':len(frame),'columns':list(frame.columns)}],strict_usable=False,limit=query['limit'])
    except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
    save(directory/f'probe-{endpoint}.json',row); return row


def validate_probes(directory):
    require(api_evidence()==read(directory/'existing-apis.json'),'Probe API changed'); rows=[]
    for endpoint,query in QUERIES.items():
        row=read(directory/f'probe-{endpoint}.json')
        require(row['endpoint']==endpoint and row['query']==query and row['api_sha256']==file_sha(directory/'existing-apis.json') and row['published'] is False,'Probe binding changed')
        for item in [*row['files'],*row['wire']]: require(file_sha(item['file'])==item['sha256'],'Probe file changed')
        if row['status']=='sample':
            require(row['strict_usable'] is False and row['limit']==query['limit'] and len(row['files'])==1,'Unproven sample admitted')
            item=row['files'][0]; frame=pd.read_parquet(item['file'])
            require(0<len(frame)==item['rows']<=100000 and list(frame.columns)==item['columns'],'Sample profile changed')
        else: require(row['status'] in ('failed','timeout') and not row['files'] and row.get('error'),'Failed probe admitted data')
        rows.append(row)
    return rows


def probe(root,directory):
    binding(root,directory)
    for endpoint,query in QUERIES.items():
        command=[sys.executable,'-m','scripts.review_strategy_batch47','worker','--root',root,'--directory',str(directory),'--endpoint',endpoint]
        with (directory/f'probe-{endpoint}.stdout').open('x') as out,(directory/f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command,stdout=out,stderr=err,timeout=45,check=True)
            except subprocess.TimeoutExpired:
                if not (directory/f'probe-{endpoint}.json').exists(): save(directory/f'probe-{endpoint}.json',{'endpoint':endpoint,'query':query,
                    'api_sha256':file_sha(directory/'existing-apis.json'),'status':'timeout','published':False,'files':[],'wire':[],'error':'Child exceeded45s'})
    save(directory/'probe-results.json',{'results':validate_probes(directory),'published':False})
    return archive(root,'Batch47 original pool/two fund supplementation attempts frozen; no publication')


def offline(root,directory):
    binding(root,directory); before=implementation()
    def denied(*a,**kw): raise AssertionError('Batch47 forbidden-network recheck attempted network')
    with patch.object(socket,'socket',denied),patch.object(socket,'create_connection',denied),redirect_stdout(io.StringIO()):
        result=compute(directory); diagnostic=diagnostics(directory); catalog=offline_catalog(root,directory)
    require(before==implementation() and diagnostic==read(directory/'diagnostics.json'),'Offline implementation/diagnoses differ')
    save(directory/'component-offline.json',result); require(file_sha(directory/'component-offline.json')==file_sha(directory/'component-research.json'),'Offline component differs')
    save(directory/'offline-catalog.json',catalog); save(directory/'offline-verification.json',{'result':'match','differences':0,
        'socket_network_disabled':True,'implementation_sha256':before,'sha256':file_sha(directory/'component-offline.json'),'not_a_backtest':True})
    validate_catalog(root,directory); return archive(root,'Batch47 forbidden-network component/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests','scripts/review_strategy_batch47.py',
        'scripts/prepare_handoff.py','scripts/build_strategy_catalog.py','scripts/verify_financial_import.py'],
        [sys.executable,'-m','scripts.verify_offline_tests','-q','tests/unit/test_batch47_research.py','tests/unit/test_batch45_research.py',
         'tests/unit/test_catalog_prefix.py','tests/unit/test_catalog_schedules.py','tests/integration/test_strategy_merge.py'],['git','diff','--check']]


def checks(root,directory):
    before=implementation(); rows=[]
    for k,command in enumerate(commands()):
        path=directory/f'check-{k}.log'
        with path.open('x') as out: result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
        rows.append({'command':command,'returncode':result.returncode,'log':str(path),'sha256':file_sha(path)}); require(result.returncode==0,f'Check failed: {path}')
    require(before==implementation(),'Checked implementation changed')
    for name in FILES:
        copied=directory/'implementation-checked'/name; copied.parent.mkdir(parents=True,exist_ok=True); require(not copied.exists(),'Checked copy exists'); shutil.copyfile(name,copied)
    save(directory/'checked-state.json',{'status':'passed','commands':rows,'implementation_sha256':before,
        'evidence_sha256':{p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status':'passed','commands':rows}


def finish(root,directory):
    checked=read(directory/'checked-state.json'); off=read(directory/'offline-verification.json')
    require(checked['status']=='passed' and checked['implementation_sha256']==implementation() and CORE<=checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']]==commands() and all(r['returncode']==0 for r in checked['commands']),'Checked state changed')
    for name,sha in checked['evidence_sha256'].items(): require(file_sha(directory/name)==sha,'Checked evidence changed')
    for name,sha in checked['implementation_sha256'].items(): require(file_sha(directory/'implementation-checked'/name)==sha,'Checked copy changed')
    require(off['implementation_sha256']==implementation() and off['result']=='match' and off['differences']==0 and off['socket_network_disabled'] is True and
        off['sha256']==file_sha(directory/'component-research.json')==file_sha(directory/'component-offline.json'),'Offline evidence changed')
    binding(root,directory); validate_catalog(root,directory); validate_sources(directory)
    require(read(directory/'probe-results.json')=={'results':validate_probes(directory),'published':False},'Probe summary changed')
    require(read(directory/'dependency-inventory.json')==inventory(root,directory),'Fund inventory changed')
    require(compute(directory)==read(directory/'component-research.json') and diagnostics(directory)==read(directory/'diagnostics.json'),'Recomputed component differs')
    require(Store(root).published()==read(directory/'baseline.json')['published'],'Publication changed')
    protection=protect(root,directory); progress=archive(root,'Batch47 original ETF candidate/momentum research accepted; original data remain missing')
    save(Path('docs/handoff/2026-10-07-batch47-verification.json'),{'status':'ok','reviews':read(directory/'source-reviews/review.json'),
        'progress':progress,'snapshot':SNAPSHOT,'checks':checked,'offline':off,'offline_catalog':read(directory/'offline-catalog.json'),
        'protection':protection,'probes':read(directory/'probe-results.json'),'not_a_backtest':True,'original_strategy_complete':False})
    return {'status':'ok','checkpoint':progress['checkpoint'],'manually_reviewed':progress['manually_reviewed']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','start','prepare','study','worker','probe','offline','checks','finish'])
    parser.add_argument('--root',default='data'); parser.add_argument('--directory',default='data/staging/strategies-batch47/20261007-candidate-momentum')
    parser.add_argument('--endpoint',choices=list(QUERIES)); args=parser.parse_args()
    result=worker(args.root,Path(args.directory),args.endpoint) if args.action=='worker' else globals()[args.action](args.root,Path(args.directory))
    print(json.dumps(result,ensure_ascii=False,default=str))
