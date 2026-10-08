"""Freeze RSRS/reversal/coin sources and reproduce original indicator components."""
import argparse
import ast
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
import warnings

import numpy as np
import pandas as pd
import requests
import scipy
from scipy import stats

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store, fingerprint
from observe.runs import environment, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch22 import validate_offline
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-211958-89b5'
SOURCES = ('2022年度精选策略/61.最简单的动量模型，能否应对A股暴涨？(二).txt',
    '2023年度精选策略/66.基于动量和反转效应的沪深300成分股策略.txt',
    '2024年度精选策略1/75.【复现】个股动量效应的识别及球队硬币因子.txt',
    '2022年度精选策略/78.RSRS系列一：完整复现版(自带快速回测框架).txt')
REFERENCES = ('hikyuu_cpp/hikyuu/indicator/imp/IRSRSBeta.cpp', 'hikyuu_cpp/hikyuu/indicator/imp/IRSRSBull.cpp')
FILES = {'scripts/review_strategy_batch23.py', 'tests/unit/test_batch23_research.py', 'src/observe/strategy_catalog.py',
    'src/observe/data/sources/baostock.py', 'scripts/review_strategy_batch21.py', 'scripts/review_strategy_batch22.py',
    'scripts/review_strategy_batch18.py', 'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py', 'scripts/archive_strategy_progress.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-api.json', 'probe/result.json', 'input-analysis.json',
    'research-inputs.parquet', 'component-research.json', 'component-offline.json', 'offline-verification.json', 'diagnostics.json',
    'input-failure.json', 'source-review-supplement.json', 'start-failure.json', 'dependency-supplement.json'}
UPSTREAM = Path('docs/handoff/2026-10-06-batch22-verification.json')
UPSTREAM_BASELINE = Path('data/staging/strategies-batch22/20261006-industry-premium-models/baseline.json')
QUERY = {'index': '000300.SH', 'date': '2021-01-04'}
OLD_POOL_GAP = '现有沪深300周频成分仅2024Q2；get_index_stocks不传date的开盘时点等价未证明'
NEW_POOL_GAP = '本机冻结快照无沪深300成分表；远程2024Q2周频成分尚未迁移；get_index_stocks开盘时点等价未证明'


def clarified_reviews(directory):
    row = read(directory / 'source-review-supplement.json'); initial = read(directory / 'source-reviews/review.json')['sources'][1]
    before = json.loads(json.dumps(REVIEWS[SOURCES[1]])); before['gaps'][0] = OLD_POOL_GAP
    require(row['review_sha256'] == file_sha(directory / 'source-reviews/review.json') and
        row['source_sha256'] == initial['source_sha256'] and row['before'] == initial['review'] == before and
        row['after'] == REVIEWS[SOURCES[1]], 'Review supplement differs')
    return row


def supplement(root, directory):
    review = read(directory / 'source-reviews/review.json'); before = review['sources'][1]['review']
    save(directory / 'input-failure.json', {'status': 'failed', 'action': 'analyze', 'error': 'ValueError: Missing coverage table',
        'cause': 'Local frozen snapshot has calendar but no index_constituents; remote weekly table was not migrated',
        'snapshot': SNAPSHOT, 'input_written': False, 'recovery': 'Explicit zero local constituent coverage; RSRS index-only research may proceed'})
    save(directory / 'source-review-supplement.json', {'review_sha256': file_sha(directory / 'source-reviews/review.json'),
        'source_sha256': review['sources'][1]['source_sha256'], 'before': before, 'after': REVIEWS[SOURCES[1]]})
    clarified_reviews(directory)
    reference = Path('repo/qlib/examples/benchmarks_dynamic/baseline/rolling_benchmark.py')
    copied = directory / 'references/rolling_benchmark.py'; shutil.copyfile(reference, copied)
    import baostock.util.socketutil as socketutil
    socket_code = inspect.getsource(socketutil.SocketUtil.connect)
    save(directory / 'dependency-supplement.json', {'qlib_reference': {'file': str(reference), 'copy': str(copied), 'sha256': file_sha(reference),
        'commit': subprocess.run(['git', '-C', 'repo/qlib', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
        'finding': 'Same-name example takes conf_path/horizon and kwargs; not original src.rolling implementation or factor kernel, not substituted/executed'},
        'probe_transport': {'file': inspect.getfile(socketutil), 'sha256': file_sha(inspect.getfile(socketutil)),
            'connect_source': socket_code, 'finding': 'connect catches socket creation/connection Exception but accesses mySockect after catch; observed UnboundLocalError masks original exception',
            'original_network_exception': 'not captured by third-party code; no DNS-specific claim', 'not_patched_or_retried': True}})
    catalog_strategies(root, 'repo/量化策略源代码')
    return archive(root, 'Batch23 local constituent absence clarified; source supplement and initial input failure retained')


def index_binding(root):
    store = Store(root); state = store.state(SNAPSHOT); receipt = read(UPSTREAM)
    require(file_sha(UPSTREAM_BASELINE) == receipt['checks']['evidence_sha256']['baseline.json'], 'Accepted upstream baseline differs')
    require(receipt['status'] == 'ok' and receipt['reviews']['snapshot'] == SNAPSHOT and
        read(UPSTREAM_BASELINE)['published']['tables'] == state['tables'], 'Accepted snapshot differs')
    entry = state['tables']['index_1d']['all']; path = store.root / entry['file']
    frame = pd.read_parquet(path)
    require(fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'], 'Frozen index partition differs')
    return {'receipt_file': str(UPSTREAM), 'receipt_sha256': file_sha(UPSTREAM), 'snapshot': SNAPSHOT,
        'partition_file': str(path), 'partition_sha256': file_sha(path), 'partition_fingerprint': entry['sha'], 'rows': entry['rows']}


def bound_index(root, directory):
    expected = read(directory / 'source-reviews/review.json')['index_binding']
    require(expected == index_binding(root), 'Accepted index binding changed')
    return expected


def start(root, directory):
    checkpoint(root, directory, 'Batch23 original RSRS/reversal/coin dependency research started')
    return complete_start(root, directory)


def resume_start(root, directory):
    require((directory / 'baseline.json').exists() and not (directory / 'source-reviews/review.json').exists(), 'Unexpected start recovery state')
    save(directory / 'start-failure.json', {'status': 'failed', 'action': 'start', 'error': 'ValueError: Frozen index partition differs',
        'cause': 'Store entry sha is a DataFrame content fingerprint, not Parquet file byte SHA256',
        'partition_file': 'data/std/index_1d/all__2843c31b.parquet',
        'content_fingerprint': '2843c31b8faaa9401b2f527b4671750bb7d551e1a3c53b2efc54cfd6c268e8a6',
        'byte_sha256': '28a1ccbb565895091aa8f1cb865254f5626d3514a7ba035073113d95d8e6c406',
        'recovery': 'Retain original baseline/copies/API/references/catalog; compare both fingerprint and newly frozen byte SHA', 'not_a_data_change': True})
    return complete_start(root, directory)


def complete_start(root, directory):
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Publication differs')
    progress = read(directory / 'progress-start.json')
    previous_file = Path(root) / 'catalog/strategies' / progress['catalog_id'] / 'catalog.json'
    require(file_sha(previous_file) == progress['catalog_sha256'], 'Starting catalog changed')
    previous = {r['path']: r for r in read(previous_file)}; sources = []; folder = directory / 'source-reviews'; folder.mkdir(exist_ok=True)
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(previous[name]['review_status'] != '人工审查完成' and file_sha(source) == previous[name]['bytes_sha256'], 'Already reviewed/source differs')
        if copied.exists(): require(file_sha(copied) == file_sha(source), 'Existing source copy changed')
        else: shutil.copyfile(source, copied)
        text, encoding = read_source(source)
        sources.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    searches = []
    for command in (['rg', '--files', '--hidden', 'repo', 'data', '-g', '*FactorZoo*', '-g', 'build_factor.py', '-g', 'qlib_workflow.py',
            '-g', 'rolling.py', '-g', 'all_data.pkl', '-g', 'volatility_momentum.pkl', '-g', 'data_dhpr.pkl'],
        ['rg', '-l', r'^class SportBettingsFactor|^class VolatilityMomentum|^class QlibFlow|^class RollingBenchmark|^def load2qlib', 'repo', '-g', '*.py', '-g', '*.txt']):
        process = subprocess.run(command, capture_output=True, text=True); require(process.returncode in (0, 1), process.stderr)
        searches.append({'command': command, 'returncode': process.returncode, 'matches': process.stdout.splitlines()})
    ref_folder = directory / 'references'; ref_folder.mkdir(exist_ok=True); references = []
    commit = subprocess.run(['git', '-C', 'repo/hikyuu', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    for name in REFERENCES:
        source = Path('repo/hikyuu') / name; copied = ref_folder / source.name
        if copied.exists(): require(file_sha(copied) == file_sha(source), 'Existing reference copy changed')
        else: shutil.copyfile(source, copied)
        references.append({'source': str(source), 'copy': str(copied), 'sha256': file_sha(source), 'commit': commit,
            'use': 'Read-only covariance/R2 design comparison; defaults20/60, sample std and start differ, not substituted or executed'})
    import baostock as bs
    code = inspect.getsource(bs.query_hs300_stocks)
    if (directory / 'existing-api.json').exists(): validate_api(directory)
    else: save(directory / 'existing-api.json', {'version': bs.__version__, 'name': 'query_hs300_stocks', 'source': code,
        'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'signature': str(inspect.signature(bs.query_hs300_stocks)), 'query': QUERY})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(folder / 'review.json', {'snapshot': SNAPSHOT, 'sources': sources, 'previous_catalog_file': str(previous_file),
        'previous_catalog_sha256': file_sha(previous_file), 'catalog': catalog, 'index_binding': index_binding(root), 'references': references,
        'dependency_searches': searches, 'installed_modules': {n: importlib.util.find_spec(n) is not None for n in ('statsmodels', 'qlib', 'empyrical')},
        'strategy_results': [], 'limits': ['Only original SciPy get_slope, no original quick ledger/NAV/fee loops',
            '2003 requested beginning not available; existing 2005+ provider data is explicitly a component research input']})
    return archive(root, 'Batch23 four complete source reviews and C++ reference differences archived')


def worker(root, directory):
    folder = directory / 'probe'; folder.mkdir(exist_ok=False); socket.setdefaulttimeout(10)
    result = {'query': QUERY, 'status': 'failed', 'files': [], 'published': False, 'api_sha256': file_sha(directory / 'existing-api.json')}
    try:
        validate_api(directory)
        with BaoStock(root).session() as source: frame = source.index_constituents(QUERY['index'], QUERY['date'])
        path = raw.save(root, 'rsrs_reversal_probe_batch23', 'hs300_20210104', directory.name, frame)
        result['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        result['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', result)
    return result


def validate_api(directory):
    import baostock as bs
    row = read(directory / 'existing-api.json'); fn = bs.query_hs300_stocks
    require(row['version'] == bs.__version__ and row['name'] == 'query_hs300_stocks' and row['query'] == QUERY and
        row['signature'] == str(inspect.signature(fn)) and hashlib.sha256(row['source'].encode()).hexdigest() ==
        row['source_sha256'] == hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(), 'API identity differs')
    return row


def validate_probe(directory):
    validate_api(directory); row = read(directory / 'probe/result.json')
    require(row['query'] == QUERY and row['api_sha256'] == file_sha(directory / 'existing-api.json') and row['published'] is False, 'Probe binding differs')
    require(row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Unknown probe status')
    if row['status'] in ('success', 'empty'):
        require(len(row['files']) == 1, 'Probe input count differs'); item = row['files'][0]
        require(file_sha(item['file']) == item['sha256'], 'Probe data changed'); frame = pd.read_parquet(item['file'])
        require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status'] == 'success'), 'Probe rows/schema differs')
    else: require(not row['files'] and row.get('error'), 'Failed probe accepted input/no error')
    return row


def probe(root, directory):
    require(not (directory / 'probe').exists(), 'Probe archive exists')
    command = [sys.executable, '-m', 'scripts.review_strategy_batch23', 'worker', '--root', str(root), '--directory', str(directory)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=45); require(process.returncode == 0, process.stderr)
    except subprocess.TimeoutExpired as exc:
        output = directory / 'probe/result.json'
        if not output.exists(): save(output, {'query': QUERY, 'status': 'timeout', 'files': [], 'published': False,
            'api_sha256': file_sha(directory / 'existing-api.json'), 'error': str(exc), 'limits': ['Interrupted partial output not accepted']})
    result = validate_probe(directory)
    archive(root, f'Batch23 historic CSI300 constituent probe {result["status"]}; no publication')
    return result


def index_sample(frame):
    require({'date', 'index', 'open', 'high', 'low', 'close'} <= set(frame) and frame['index'].eq('000300.SH').all(), 'Unexpected index/schema')
    data = frame[['date', 'open', 'high', 'low', 'close']].copy(); data.date = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
    data = data.sort_values('date').reset_index(drop=True)
    require(len(data) > 800 and not data.date.duplicated().any() and data.date.iloc[0] == '2005-01-05' and data.date.iloc[-1] == '2026-09-29', 'Index scope differs')
    require(np.isfinite(data.iloc[:, 1:].to_numpy(dtype=float)).all() and data.iloc[:, 1:].gt(0).all().all() and
        data.high.ge(data[['open', 'low', 'close']].max(axis=1)).all() and data.low.le(data[['open', 'close']].min(axis=1)).all(), 'Invalid index OHLC')
    return data


def coverage(root, data):
    store = Store(root); state = store.state(SNAPSHOT)
    require(state['tables'].get('calendar'), 'Missing calendar table')
    for table in ('calendar', 'index_constituents'):
        for entry in state['tables'].get(table, {}).values():
            require(fingerprint(pd.read_parquet(store.root / entry['file'])) == entry['sha'], 'Coverage partition changed')
    calendar = store.load_state(state, 'calendar'); dates = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    expected = dates[calendar.is_open & dates.between('2005-01-05', '2026-09-29')].sort_values().tolist()
    require(data.date.tolist() == expected, 'Index calendar incomplete')
    constituents = store.load_state(state, 'index_constituents', filters=[('index', '=', '000300.SH')])
    return {'rows': len(constituents), 'first': str(constituents.date.min()) if len(constituents) else None,
        'last': str(constituents.date.max()) if len(constituents) else None,
        'strict_usable': False, 'not_original_daily_pool_proof': True}


def analyze(root, directory):
    bound = bound_index(root, directory); data = index_sample(pd.read_parquet(bound['partition_file']))
    constituents = coverage(root, data); probe_result = validate_probe(directory)
    output = directory / 'research-inputs.parquet'; require(not output.exists(), 'Input archive exists'); data.to_parquet(output, index=False)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'input_file': str(output), 'input_sha256': file_sha(output),
        'rows': len(data), 'first': data.date.iloc[0], 'last': data.date.iloc[-1], 'calendar_missing': 0,
        'constituents': constituents, 'probe_status': probe_result['status'], 'not_a_backtest': True,
        'limits': ['Original requested2003 period unavailable; research uses verified2005+ explicitly',
            'Original get_price/provider equivalence unproved; no ETF events/status/execution rules supplemented']})
    return archive(root, 'Batch23 complete 5280-bar frozen index input and actual constituent coverage archived')


def reference_regression(low, high):
    x, y = [float(v) for v in low], [float(v) for v in high]
    require(len(x) == len(y) and len(x) >= 2 and all(math.isfinite(v) for v in [*x, *y]), 'Invalid regression inputs')
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xx = math.fsum((v - mx) ** 2 for v in x); yy = math.fsum((v - my) ** 2 for v in y)
    require(xx > 0 and yy > 0, 'Degenerate regression window')
    xy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return xy / xx, xy ** 2 / (xx * yy)


def reference_indicators(frame, n, m):
    require(n >= 2 and m >= 2 and len(frame) > n + m, 'Insufficient indicator inputs')
    count = len(frame); beta = [math.nan] * count; r2 = [math.nan] * count
    result = [[math.nan] * count for _ in range(3)]
    for k in range(n - 1, count): beta[k], r2[k] = reference_regression(frame.low.iloc[k - n + 1:k + 1], frame.high.iloc[k - n + 1:k + 1])
    for k in range(n + m - 2, count):
        window = beta[k - m + 1:k + 1]; mean = math.fsum(window) / m
        sigma = math.sqrt(math.fsum((v - mean) ** 2 for v in window) / m); require(sigma > 0, 'Zero beta dispersion')
        z = (beta[k] - mean) / sigma; result[0][k] = z
        if k < count - 10 and k >= m: result[1][k] = z * r2[k]; result[2][k] = z * r2[k] * beta[k]
    return [np.asarray(values) for values in result]


def get_kernel(directory, data):
    frame = data.copy(); frame.index = pd.to_datetime(frame.date); frame = frame.drop(columns='date')
    ns = {'np': np, 'pd': pd, 'stats': stats, 'data_panel': frame}
    fingerprint = selected(directory, 3, ('get_slope',), ns)
    return ns, fingerprint


def run_kernel(ns, n, m):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always'); outputs = ns['get_slope'](n, m)
    return outputs, {'count': len(seen), 'messages': sorted({str(w.message) for w in seen})}


def compute(root, directory):
    bound_index(root, directory); info = read(directory / 'input-analysis.json')
    require(info['snapshot'] == SNAPSHOT and file_sha(info['input_file']) == info['input_sha256'], 'Research input changed')
    sample = pd.read_parquet(info['input_file']); require(index_sample(sample.assign(index='000300.SH')).equals(sample), 'Sample contract differs')
    results = []; fingerprint = None
    for label, frame in [('original_available_period', sample[sample.date.le('2019-07-04')]), ('extended_frozen_period', sample)]:
        for n, m in ((18, 600), (16, 300)):
            ns, fingerprint = get_kernel(directory, frame); outputs, warned = run_kernel(ns, n, m)
            reference = reference_indicators(frame, n, m); comparisons = []; rows = []
            for name, actual, expected in zip(('z', 'adjusted', 'right'), outputs, reference, strict=True):
                require(np.array_equal(np.isnan(actual), np.isnan(expected)) and not np.isinf(actual).any(), 'Indicator validity differs')
                mask = np.isfinite(actual); delta = np.abs(actual[mask] - expected[mask])
                require(mask.any() and delta.max() < 1e-8, 'Indicator values differ')
                require(np.array_equal(actual > .7, expected > .7) and np.array_equal(actual < -.7, expected < -.7), 'Strict threshold classification differs')
                comparisons.append({'name': name, 'finite': int(mask.sum()), 'max_abs_difference': float(delta.max())})
            for k, day in enumerate(frame.date):
                rows.append({'date': day, **{name: float(a[k]) if math.isfinite(a[k]) else None for name, a in zip(('z', 'adjusted', 'right'), outputs, strict=True)}})
            results.append({'period': label, 'n': n, 'm': m, 'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
                'comparisons': comparisons, 'warnings': warned, 'values': rows})
    return {'snapshot': SNAPSHOT, 'input_sha256': info['input_sha256'], 'source_sha256': read(directory / 'source-reviews/review.json')['sources'][3]['source_sha256'],
        'selected_ast_sha256': fingerprint, 'backend': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'linregress_source_sha256': hashlib.sha256(inspect.getsource(stats.linregress).encode()).hexdigest()},
        'results': results, 'not_a_backtest': True, 'limits': ['Original scalar indicator function only; no quick trade/fees/NAV',
            'Adjusted/right tail availability depends on future row presence; not accepted as causal trading features',
            'Hikyuu sample std/default windows not substituted; full original 2003 beginning unavailable']}


def diagnostics(root, directory):
    rows = []; orders = []; history_calls = []
    g = SimpleNamespace(init=True, ans=list(np.linspace(.8, 1.2, 1200)), M=1200, buy=.7, sell=-.7, security='510300.XSHG')
    def history(*args, **kwargs): history_calls.append(True); raise AssertionError('First callback must not query')
    ns = {'np': np, 'g': g, 'attribute_history': history, 'log': SimpleNamespace(info=lambda *a: None),
        'order_value': lambda *a: orders.append(a), 'order_target': lambda *a: orders.append(a)}
    selected(directory, 0, ('market_open',), ns); ns['market_open'](SimpleNamespace(portfolio=SimpleNamespace(available_cash=100000)))
    require(g.init is False and not orders and not history_calls and len(g.ans) == 1200, 'First callback differs')
    rows.append({'case': 'rsrs61_initial_callback', 'orders': [], 'history_calls': 0, 'beta_pool_unchanged': True, 'no_real_orders': True})
    try: pd.Series([1., 2.], index=['const', 'low'])[1]
    except KeyError: rows.append({'case': 'named_OLS_params_integer_index', 'error': 'KeyError', 'limit': 'Pandas parameter-shape diagnostic only; statsmodels not installed'})
    ns = {'g': SimpleNamespace(), 'set_benchmark': lambda *a: None, 'set_option': lambda *a: None,
        'log': SimpleNamespace(info=lambda *a: None), 'OrderCost': lambda **a: a, 'set_order_cost': lambda *a, **kw: None,
        'run_daily': lambda *a, **kw: None, 'handle': lambda *a: None}
    selected(directory, 1, ('initialize', 'cal_date_fzxy'), ns); ns['initialize'](None); triggers = []
    for day in range(1, 361):
        ns['cal_date_fzxy'](None)
        if ns['g'].flag_fzxy: triggers.append(day); ns['g'].flag_fzxy = False
    require(triggers == [1, 180, 360], 'Counter differs')
    rows.append({'case': 'reversal_counter', 'callbacks': 360, 'triggers': triggers})
    calls = []; ns = {'handle_fzxy': lambda *a: calls.append('reversal'), 'handle_dlxy': lambda *a: calls.append('momentum')}
    selected(directory, 1, ('handle',), ns); ns['handle'](None); require(calls == ['reversal'], 'Dispatch differs')
    rows.append({'case': 'reversal_dispatch', 'calls': calls})
    synthetic = pd.DataFrame({'A': np.arange(100., 130.)}, index=pd.date_range('2020-01-01', periods=30))
    ns = {'g': SimpleNamespace(flag_fzxy=True), 'get_index_stocks': lambda *a: ['A'], 'history': lambda **kw: synthetic.copy(),
        'NaN': np.nan, 'order_value': lambda *a: orders.append(a), 'order_target': lambda *a: orders.append(a)}
    selected(directory, 1, ('handle_fzxy',), ns)
    try: ns['handle_fzxy'](SimpleNamespace(portfolio=SimpleNamespace(total_value=100000)))
    except KeyError: rows.append({'case': 'reversal_date_series_index', 'error': 'KeyError', 'flag_consumed_before_failure': ns['g'].flag_fzxy is False})
    require(not orders, 'Unexpected diagnostic orders')
    # Execute only the exact reviewed nested assignment, with all other trading statements excluded.
    source = read(directory / 'source-reviews/review.json')['sources'][1]; text, _ = read_source(Path(source['source_copy']))
    require(file_sha(source['source_copy']) == source['source_sha256'], 'Chained assignment source changed')
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'handle_fzxy')
    assignment = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "stock_data['ret'][stock]" for t in n.targets))
    table = pd.DataFrame({'ret': [np.nan]}, index=['A'])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always'); exec(compile(ast.Module(body=[assignment], type_ignores=[]), source['source_copy'], 'exec'), {'stock_data': table, 'stock': 'A', 'ret': .3})
    require(table.ret.isna().all(), 'Expected current pandas chain behavior differs')
    rows.append({'case': 'original_chained_ret_assignment', 'stored_ret': None, 'warnings': sorted({str(w.message) for w in seen}),
        'selected_ast_sha256': hashlib.sha256(ast.dump(assignment).encode()).hexdigest()})
    data = pd.read_parquet(read(directory / 'input-analysis.json')['input_file']); ns, fingerprint = get_kernel(directory, data.iloc[:800]); before, _ = run_kernel(ns, 16, 300)
    ns, _ = get_kernel(directory, data.iloc[:801]); after, _ = run_kernel(ns, 16, 300); k = 790
    require(before[0][k] == after[0][k] and math.isnan(before[2][k]) and math.isfinite(after[2][k]), 'Future-presence diagnosis differs')
    rows.append({'case': 'future_row_changes_past_indicator_availability', 'past_date': data.date.iloc[k], 'future_added_date': data.date.iloc[800],
        'prefix_rows': [800, 801], 'parameters': [16, 300], 'plain_before': float(before[0][k]), 'plain_after': float(after[0][k]),
        'right_before': None, 'right_after': float(after[2][k]), 'selected_ast_sha256': fingerprint, 'not_fixed': True})
    require(len(rows) == 7, 'Expected diagnostics missing')
    return {'cases': rows, 'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'not_a_backtest': True, 'limits': ['Only reviewed functions/statements or explicit synthetic index shapes; no source top-level/PKL/model fit/real orders']}


def study(root, directory):
    save(directory / 'component-research.json', compute(root, directory)); save(directory / 'diagnostics.json', diagnostics(root, directory))
    return archive(root, 'Batch23 original SciPy RSRS two fixed-parameter/two-period components and seven diagnostics archived')


def offline(root, directory):
    def denied(*args, **kwargs): raise AssertionError('Network forbidden')
    before = file_sha(directory / 'component-research.json')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), patch.object(requests.sessions.Session, 'request', denied):
        repeated = compute(root, directory); diagnosed = diagnostics(root, directory)
    save(directory / 'component-offline.json', repeated)
    require(repeated == read(directory / 'component-research.json') and diagnosed == read(directory / 'diagnostics.json') and
        before == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline original/component differs')
    save(directory / 'offline-verification.json', {'status': 'match', 'differences': 0, 'original_unchanged': True, 'original_sha256': before,
        'recomputed_sha256': file_sha(directory / 'component-offline.json'), 'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True})
    return archive(root, 'Batch23 original indicators byte-identical and all diagnostics matched offline, not a trading reproduction')


def checks(root, directory):
    validate_probe(directory); validate_offline(directory)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
        'scripts/verify_financial_import.py', 'scripts/review_strategy_batch23.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch23_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True); results.append({'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        print(results[-1], flush=True); require(process.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    checked = read(directory / 'checks.json')
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']) and
        FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checks incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    reviews = read(directory / 'source-reviews/review.json'); clarification = clarified_reviews(directory)
    bound_index(root, directory); validate_probe(directory); repeated = validate_offline(directory)
    require(reviews['snapshot'] == SNAPSHOT and file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'] and
        [r['source_path'] for r in reviews['sources']] == [str(Path('repo/量化策略源代码') / n) for n in SOURCES], 'Review scope differs')
    for name, row in zip(SOURCES, reviews['sources'], strict=True):
        final_review = clarification['after'] if name == SOURCES[1] else row['review']
        require(file_sha(row['source_path']) == row['source_sha256'] == row['source_copy_sha256'] == file_sha(row['source_copy']) and REVIEWS[name] == final_review, 'Source/rules changed')
    for row in reviews['references']: require(file_sha(row['source']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')
    dependencies = read(directory / 'dependency-supplement.json'); ref = dependencies['qlib_reference']; transport = dependencies['probe_transport']
    require(file_sha(ref['file']) == ref['sha256'] == file_sha(ref['copy']) and file_sha(transport['file']) == transport['sha256'], 'Supplement dependencies changed')
    analysis = read(directory / 'input-analysis.json')
    sample = index_sample(pd.read_parquet(reviews['index_binding']['partition_file']))
    pd.testing.assert_frame_equal(pd.read_parquet(analysis['input_file']), sample)
    require(analysis['constituents'] == coverage(root, sample) and analysis['calendar_missing'] == 0 and
        analysis['rows'] == len(sample) and analysis['first'] == sample.date.iloc[0] and analysis['last'] == sample.date.iloc[-1] and
        analysis['probe_status'] == validate_probe(directory)['status'] and analysis['not_a_backtest'] is True, 'Coverage analysis differs')
    require(compute(root, directory) == read(directory / 'component-research.json') == read(directory / 'component-offline.json') and
        diagnostics(root, directory) == read(directory / 'diagnostics.json'), 'Reconstructed research differs')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'clarification': clarification, 'dependencies': dependencies, 'analysis': analysis,
        'research': read(directory / 'component-research.json'), 'diagnostics': read(directory / 'diagnostics.json'), 'offline': repeated,
        'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch23 original RSRS indicators and dependency audit accepted; entire library continues')}
    target = Path('docs/handoff/2026-10-06-batch23-verification.json'); save(target, result)
    return {'status': 'ok', 'output': str(target), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'resume_start', 'supplement', 'worker', 'probe', 'analyze', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch23/20261006-rsrs-reversal-coins'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str), flush=True)
