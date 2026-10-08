"""Archive MACD/pattern source reviews, bounded probes and non-trading studies."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import date
import hashlib
import importlib.util
import inspect
import json
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

from observe.data import raw, standardize as std
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import environment, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import FIELDS, equal_cells
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.run_strategy_batch19 import reference_ema
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-211958-89b5'
SOURCES = (
    '2022年度精选策略/53.【指标研究】MACD日、周、月以及分别级别研究（包含低位二次金叉算法）.txt',
    '2024年度精选策略2/55.【复现】技术指标形态识别.txt',
    '2022年度精选策略/84.MACD新研究，配合ETF十年稳稳的幸福.txt',
    '2023年度精选策略/73.MACD多周期共振，精准打击！.txt',
    '2022年度精选策略/89.MACD+波动率过滤+追踪止损 期货择时汇总.txt',
)
QUERIES = {
    'daily': ('query_history_k_data_plus', {'code': 'sz.000001', 'fields': FIELDS,
        'start_date': '2005-01-05', 'end_date': '2026-09-29', 'frequency': 'd', 'adjustflag': '3'}),
    'weekly': ('query_history_k_data_plus', {'code': 'sz.000001', 'fields': 'date,code,open,high,low,close,volume,amount',
        'start_date': '2005-01-05', 'end_date': '2026-09-29', 'frequency': 'w', 'adjustflag': '3'}),
    'monthly': ('query_history_k_data_plus', {'code': 'sz.000001', 'fields': 'date,code,open,high,low,close,volume,amount',
        'start_date': '2005-01-05', 'end_date': '2026-09-29', 'frequency': 'm', 'adjustflag': '3'}),
    'hourly': ('query_history_k_data_plus', {'code': 'sz.000001', 'fields': 'date,time,code,open,high,low,close,volume,amount',
        'start_date': '2026-06-01', 'end_date': '2026-09-29', 'frequency': '60', 'adjustflag': '3'}),
    'etf510300': ('fund_etf_hist_sina', {'symbol': 'sh510300'}),
    'sw801030': ('index_hist_sw', {'symbol': '801030', 'period': 'day'}),
}
FILES = {'src/observe/strategy_catalog.py', 'scripts/review_strategy_batch20.py', 'scripts/run_strategy_batch19.py', 'tests/unit/test_batch20_research.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'source-review-supplement.json', 'existing-apis.json', 'probe-results.json',
        'input-analysis.json', 'diagnostics.json', 'study-inputs.parquet', 'macd-research.json', 'research-offline-verification.json'}


def start(root, directory):
    checkpoint(root, directory, 'Batch20 MACD/pattern checkpoint; source and dependency research')
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Publication differs')
    latest = read(Path(root) / 'catalog/strategies/latest.json')
    prior_file = Path(latest['directory']) / 'catalog.json'; prior = {r['path']: r for r in read(prior_file)}
    target = directory / 'source-reviews'; target.mkdir(); records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = target / f'{strategy_id(name)}.source'
        require(prior[name]['review_status'] != '人工审查完成' and file_sha(source) == prior[name]['bytes_sha256'], 'Source already reviewed/changed')
        shutil.copyfile(source, copied); text, encoding = read_source(source)
        records.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    search = subprocess.run(['rg', '--files', '--hidden', 'repo', 'data', '-g', 'technical_analysis_patterns.py',
        '-g', 'sw_lv1.csv', '-g', 'res_json.json'], capture_output=True, text=True)
    require(search.returncode in (0, 1), search.stderr)
    symbols = subprocess.run(['rg', '-l', 'rolling_patterns2pool|def plot_patterns_chart', 'repo', '-g', '*.py', '-g', '*.txt'], capture_output=True, text=True)
    require(symbols.returncode in (0, 1), symbols.stderr)
    import akshare as ak
    import baostock as bs
    apis = []
    for name in sorted({name for name, _ in QUERIES.values()}):
        fn = getattr(bs if name.startswith('query_') else ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    save(directory / 'existing-apis.json', {'baostock_version': bs.__version__, 'akshare_version': ak.__version__, 'apis': apis, 'queries': QUERIES})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(target / 'review.json', {'snapshot': SNAPSHOT, 'sources': records, 'catalog': catalog,
        'previous_catalog_file': str(prior_file), 'previous_catalog_sha256': file_sha(prior_file),
        'dependency_searches': [{'argv': p.args, 'returncode': p.returncode, 'matches': p.stdout.splitlines()} for p in (search, symbols)],
        'installed_modules': {n: importlib.util.find_spec(n) is not None for n in ('talib', 'scipy', 'statsmodels', 'technical_analysis_patterns')},
        'strategy_results': [], 'limits': ['Two sources are research only', 'No economic variant or substitute detector implemented']})
    return archive(root, 'Batch20 five full source reviews archived; MACD/pattern/futures gaps recorded')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    socket.setdefaulttimeout(15); evidence = read(directory / 'existing-apis.json')
    name, params = QUERIES[endpoint]
    require(evidence['queries'][endpoint] == [name, params], 'Query changed')
    result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'wire_responses': [], 'published': False}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'Response cap reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15)
        response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        if name.startswith('query_'):
            import baostock as module
            require(module.__version__ == evidence['baostock_version'], 'BaoStock changed')
        else:
            import akshare as module
            require(module.__version__ == evidence['akshare_version'], 'AKShare changed')
        fn = getattr(module, name); bound = next(r for r in evidence['apis'] if r['name'] == name)
        require(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == bound['source_sha256'], 'API code changed')
        if name.startswith('query_'):
            with BaoStock(root).session() as source: frame = source._rows(name, params, lambda: fn(**params))
        else:
            with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = fn(**params)
        path = raw.save(root, 'macd_patterns_probe_batch20', endpoint, directory.name, frame)
        result['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame), 'parameters': params})
        result['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    require(not (directory / 'probes').exists() and not (directory / 'probe-results.json').exists(), 'Probes already exist')
    results = []
    for endpoint in QUERIES:
        argv = [sys.executable, '-m', 'scripts.review_strategy_batch20', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=90)
            row = {'endpoint': endpoint, 'returncode': process.returncode, 'stdout': process.stdout[-1200:], 'stderr': process.stderr[-1200:]}
        except subprocess.TimeoutExpired:
            row = {'endpoint': endpoint, 'status': 'timeout', 'timeout_seconds': 90}
        terminal = directory / 'probes' / endpoint / 'result.json'
        row['result'] = read(terminal) if terminal.exists() else {'status': row.get('status', 'failed'), 'files': [], 'wire_responses': []}
        results.append(row); print(json.dumps({'endpoint': endpoint, 'result': row['result']}, ensure_ascii=False), flush=True)
        archive(root, f'Batch20 {endpoint} probe {row["result"]["status"]}; standalone unpublished inputs')
    save(directory / 'probe-results.json', {'results': results, 'snapshot': SNAPSHOT, 'published': False,
        'api_sha256': file_sha(directory / 'existing-apis.json')})
    return {'status': 'archived', 'endpoints': len(results)}


def frames(directory):
    result = read(directory / 'probe-results.json')
    require(result['snapshot'] == SNAPSHOT and result['published'] is False, 'Probe scope differs')
    require(result['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'API evidence changed')
    require(len(result['results']) == len(QUERIES) and {r['endpoint'] for r in result['results']} == set(QUERIES), 'Probe endpoint set incomplete')
    output = {}
    for row in result['results']:
        terminal = directory / 'probes' / row['endpoint'] / 'result.json'
        if terminal.exists(): require(row['result'] == read(terminal), 'Terminal probe response differs')
        else: require(row.get('status') == 'timeout' or row.get('returncode', 0) != 0, 'Terminal probe response missing')
        require(row['result'].get('endpoint', row['endpoint']) == row['endpoint'], 'Endpoint identity differs')
        require(len(row['result']['files']) == (1 if row['result']['status'] in ('success', 'empty') else 0), 'Probe files incomplete')
        for item in row['result']['wire_responses']: require(file_sha(item['file']) == item['sha256'], 'Wire response changed')
        for item in row['result']['files']:
            require(item['parameters'] == QUERIES[row['endpoint']][1], 'Probe parameters changed')
            require(file_sha(item['file']) == item['sha256'], 'Probe data changed')
            frame = pd.read_parquet(item['file']); require(len(frame) == item['rows'] and list(frame) == item['columns'], 'Probe shape changed')
            if row['result']['status'] == 'success': output[row['endpoint']] = frame
    return output


def sample(frame, unit):
    data = frame.copy(); data['date'] = pd.to_datetime(data.date, errors='raise').dt.date
    require(data.date.le(date(2026, 9, 29)).all(), 'Beyond frozen end')
    if unit == 'weekly': data = data[data.date.lt(date(2026, 9, 28))]
    if unit == 'monthly': data = data[data.date.lt(date(2026, 9, 1))]
    if unit == 'daily':
        require(data.tradestatus.isin(['0', '1']).all(), 'Unknown raw trading status')
        data = data[data.tradestatus.eq('1')]
    key = 'time' if unit == 'hourly' else 'date'
    data = data.sort_values(key).tail(200)
    require(len(data) == 200 and not data[key].duplicated().any(), 'Insufficient/duplicate sample')
    require(data.code.eq('sz.000001').all(), 'Wrong security')
    for column in ('open', 'high', 'low', 'close'):
        data[column] = pd.to_numeric(data[column], errors='raise')
        require(np.isfinite(data[column]).all() and data[column].gt(0).all(), 'Invalid price')
    require((data.high >= data[['open', 'close', 'low']].max(axis=1)).all() and
        (data.low <= data[['open', 'close', 'high']].min(axis=1)).all(), 'Invalid OHLC')
    return pd.DataFrame({'unit': unit, 'timestamp': data[key].astype(str).tolist(), 'close': data.close.to_numpy()})


def analyze(root, directory):
    downloaded = frames(directory); store = Store(root); state = store.state(SNAPSHOT); inputs = []; profiles = []
    if 'daily' in downloaded:
        incoming = std.daily(downloaded['daily']); incoming['date'] = pd.to_datetime(incoming.date).dt.date
        old = store.load_state(state, 'bars_1d', filters=[('instrument', '=', '000001.SZ')]); old['date'] = pd.to_datetime(old.date).dt.date
        common = old.merge(incoming, on=['date', 'instrument'], suffixes=('_old', '_new'), validate='one_to_one')
        columns = list(old.columns.difference(['date', 'instrument']))
        differences = equal_cells(common[[c + '_old' for c in columns]].set_axis(columns, axis=1),
            common[[c + '_new' for c in columns]].set_axis(columns, axis=1), columns)
        require(len(common) == len(old) and not any(differences.values()), 'Frozen daily overlap differs')
        overlap = {'old_rows': len(old), 'overlap_rows': len(common), 'differences': differences}
    else: overlap = {'status': 'missing'}
    for unit in ('daily', 'weekly', 'monthly', 'hourly'):
        if unit not in downloaded:
            profiles.append({'unit': unit, 'status': 'missing'}); continue
        try:
            data = sample(downloaded[unit], unit); inputs.append(data)
            profiles.append({'unit': unit, 'status': 'sample_ready', 'provider_rows': len(downloaded[unit]), 'sample_rows': len(data),
                'first': data.timestamp.iloc[0], 'last': data.timestamp.iloc[-1], 'price_semantics': 'BaoStock adjustflag=3; original get_bars equality unproved'})
        except Exception as exc: profiles.append({'unit': unit, 'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}'})
    index = store.load_state(state, 'index_1d', filters=[('index', '=', '000300.SH')]).sort_values('date').tail(60)
    require(len(index) == 60 and str(index.date.iloc[-1]) == '2026-09-29', 'Index frozen window missing')
    require(np.isfinite(index.close).all() and index.close.gt(0).all(), 'Invalid index close')
    inputs.append(pd.DataFrame({'unit': 'csi300_daily', 'timestamp': index.date.astype(str).tolist(), 'close': index.close.to_numpy()}))
    target = directory / 'study-inputs.parquet'; require(not target.exists(), 'Inputs exist')
    pd.concat(inputs, ignore_index=True).to_parquet(target, index=False)
    extras = []
    for endpoint in ('etf510300', 'sw801030'):
        if endpoint in downloaded:
            frame = downloaded[endpoint]; column = 'date' if 'date' in frame else '日期'
            dates = pd.to_datetime(frame[column], errors='coerce')
            extras.append({'endpoint': endpoint, 'status': 'sample_only', 'rows': len(frame), 'first': str(dates.min()), 'last': str(dates.max()),
                'invalid_dates': int(dates.isna().sum()), 'duplicate_dates': int(dates.duplicated().sum()), 'rows_through_frozen_end': int(dates.dt.date.le(date(2026, 9, 29)).sum())})
        else: extras.append({'endpoint': endpoint, 'status': 'unavailable'})
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'daily_overlap': overlap, 'samples': profiles, 'extras': extras,
        'input_file': str(target), 'input_sha256': file_sha(target), 'published': False,
        'limits': ['Weekly/monthly current incomplete periods excluded', 'Native provider periods not proved equal to JoinQuant',
                   'Chart/index history does not supply ETF events/status/rules, historic SW universe or missing pattern algorithm']})
    return archive(root, 'Batch20 provider samples and exact frozen overlap checks archived; no data publication')


def supplement(root, directory):
    prior = read(directory / 'source-reviews/review.json'); row = prior['sources'][3]
    require(file_sha(row['source_path']) == row['source_sha256'], 'Correction source changed')
    save(directory / 'source-review-supplement.json', {'prior_review_sha256': file_sha(directory / 'source-reviews/review.json'),
        'source_sha256': row['source_sha256'], 'source_index': 3, 'prior_review': row['review'], 'review': REVIEWS[SOURCES[3]],
        'reason': 'Original buy_stock uses callback available_cash/N; every_bar exits on ret>=0.1. Initial archive retained.'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    return archive(root, f'Batch20 source73 callback cash and inclusive profit threshold clarified; catalog {catalog["catalog_id"]}')


def selected(directory, number, names, namespace):
    record = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(record['source_copy']) == record['source_sha256'], 'Selected source changed')
    text, _ = read_source(Path(record['source_copy'])); tree = ast.parse(text)
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    require(set(definitions) == set(names), 'Source functions missing')
    module = ast.Module(body=[definitions[n] for n in names], type_ignores=[])
    exec(compile(module, record['source_copy'], 'exec'), namespace)
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()


def reference_macd(prices, fast=12, slow=26, signal=9, scale=2):
    values = list(map(float, prices)); require(1 < fast < slow and signal > 1 and np.isfinite(values).all(), 'Invalid MACD inputs')
    lookback = slow + signal - 2
    if len(values) <= lookback: return tuple(np.full(len(values), np.nan) for _ in range(3))
    slow_values = reference_ema(values, slow)
    fast_values = [np.nan] * (slow - fast) + reference_ema(values[slow - fast:], fast)
    dif = [f - s for f, s in zip(fast_values, slow_values, strict=True)]
    dea = [np.nan] * (slow - 1) + reference_ema(dif[slow - 1:], signal)
    dif[:lookback] = [np.nan] * lookback
    return np.array(dif), np.array(dea), scale * (np.array(dif) - np.array(dea))


def cross_case(extra):
    signs = ([-1] * 4 + [1, 1, -1, -1, 1, 1, -1, -1, 1]) if extra else ([1] * 4 + [-1, -1, 1, 1, -1, -1, 1])
    dea = np.array([2.] * (8 if extra else 6) + [-3.] * 5)
    dif = dea + signs
    return dif, dea, 2 * (dif - dea)


def backend():
    lib = Path(talib._ta_lib.__file__)
    return {'wrapper': talib.__version__, 'c_version': talib.__ta_version__.decode(), 'compatibility': talib.get_compatibility(),
        'ema_unstable': talib.get_unstable_period('EMA'), 'binary': str(lib), 'binary_sha256': file_sha(lib)}


def compute(directory):
    analysis = read(directory / 'input-analysis.json'); require(file_sha(analysis['input_file']) == analysis['input_sha256'], 'Study input changed')
    settings = backend(); require(settings['wrapper'] == '0.8.1' and settings['compatibility'] == 0 and settings['ema_unstable'] == 0, 'Indicator backend differs')
    ns = {'tl': talib, 'np': np, 'pd': pd}; sha = selected(directory, 0, ('MACD', 'is_second_low_gold_cross'), ns)
    rows = []; values = pd.read_parquet(analysis['input_file'])
    for unit, frame in values.groupby('unit', sort=True):
        if unit == 'csi300_daily': continue
        prices = frame.close.to_numpy(dtype=float); actual = ns['MACD'](prices, 12, 26, 9); oracle = reference_macd(prices)
        errors = []
        for a, b in zip(actual, oracle, strict=True):
            require(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=0, atol=1e-10, equal_nan=True), 'MACDEXT independent arithmetic differs')
            errors.append(float(np.nanmax(np.abs(a - b))))
        for direction in (1, -1):
            flags = lambda v: (direction * v[0][:-1] < direction * v[1][:-1]) & (direction * v[0][1:] > direction * v[1][1:])
            require(np.array_equal(flags(actual), flags(oracle)), 'Strict crossover differs from independent arithmetic')
        require(bool(ns['is_second_low_gold_cross'](actual)) == bool(ns['is_second_low_gold_cross'](oracle)), 'Second crossover predicate differs')
        rows.append({'unit': unit, 'rows': len(prices), 'finite_values': sum(int(np.isfinite(a).sum()) for a in actual),
            'max_absolute_differences': errors, 'last5': [a[-5:].tolist() for a in actual],
            'second_low_gold_cross': bool(ns['is_second_low_gold_cross'](actual))})
    boundary = [{'events': n, 'actual': bool(ns['is_second_low_gold_cross'](cross_case(extra))), 'expected': extra} for n, extra in ((4, False), (5, True))]
    require(all(r['actual'] == r['expected'] for r in boundary), 'Literal event-count boundary differs')
    idx = values[values.unit.eq('csi300_daily')].close.to_numpy(dtype=float)
    etf_ns = {'np': np, 'talib': talib, 'array': np.array, 'kData': lambda context, security, count: {'close': idx}}
    etf_sha = selected(directory, 2, ('macdSignal',), etf_ns)
    actual_hist = talib.MACD(idx, fastperiod=5, slowperiod=15, signalperiod=7)[2]
    oracle_hist = reference_macd(idx, 5, 15, 7, 1)[2]
    require(np.allclose(actual_hist, oracle_hist, atol=1e-10, rtol=0, equal_nan=True), 'MACD84 arithmetic differs')
    expected = 0 if sum(oracle_hist[-20:-5]) > 0 and oracle_hist[-1] > 0 else 1 if sum(oracle_hist[-20:-5]) > 0 and sum(oracle_hist[-5:]) > 0 else 2
    state = etf_ns['macdSignal'](None); require(state == expected, 'Original ETF component signal differs')
    return {'status': 'research_recomputed_with_limits', 'snapshot': SNAPSHOT, 'input_sha256': analysis['input_sha256'], 'backend': settings,
        'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'selected_ast_sha256': [sha, etf_sha], 'macd53': rows, 'event_boundary': boundary,
        'macd84_component': {'rows': 60, 'state': int(state), 'max_absolute_difference': float(np.nanmax(np.abs(actual_hist - oracle_hist))),
            'earlier15_sum': float(sum(actual_hist[-20:-5])), 'latest5_sum': float(sum(actual_hist[-5:])), 'last_hist': float(actual_hist[-1])},
        'not_a_backtest': True, 'not_a_strategy_reproduction': True,
        'limits': ['Provider price/period and original platform equality unproved', 'Fixed 2026-09-29 research endpoint replaces unfrozen dynamic date explicitly',
                   'MACD53 contains no trading policy; MACD84 signal component is not ETF execution', 'No PnL or costs computed']}


def study(root, directory):
    result = compute(directory); save(directory / 'macd-research.json', result)
    archive(root, 'Batch20 original MACD functions independently recomputed; component research only')
    return result


def diagnose(root, directory):
    ns = {'pd': pd, 'np': np}; pattern_sha = selected(directory, 1, ('pretreatment_events', 'get_win_rate', 'get_pl'), ns)
    dates = pd.date_range('2021-01-01', periods=5)
    factors = pd.DataFrame({'event': [1]}, index=pd.MultiIndex.from_tuples([(dates[2], 'A')], names=['date', 'asset']))
    prices = pd.DataFrame({'A': [10., 11., 12., 13., 14.]}, index=dates)
    try: ns['pretreatment_events'](factors, prices, 1, 1)
    except TypeError as exc: pattern_error = str(exc)
    else: raise AssertionError('Expected original set index incompatibility')
    stats = pd.DataFrame([[.1, -.2, 0., np.nan]])
    win, pl = float(ns['get_win_rate'](stats).iloc[0]), float(ns['get_pl'](stats).iloc[0])
    require(win == 1 / 3 and pl == -.5, 'Literal statistics changed')
    g = SimpleNamespace(stocks=['D'], s=['A', 'B', 'C'], Now_date='2026-09-29')
    pool_ns = {'g': g, 'get_macd': lambda *args: (np.array([-1., 1.]), np.zeros(2), np.array([-2., 2.])), 'is_second_low_gold_cross': lambda *args: False}
    pool_sha = selected(directory, 3, ('get_D',), pool_ns)
    with redirect_stdout(sys.stderr): pool_ns['get_D']()
    require(g.s == ['B'], 'Original iterated removal behavior differs'); remaining = g.s.copy()
    orders = []; g = SimpleNamespace(instruments=['RB'], Signal={'RB': 1}, dontbuy={'RB': 0}, dontsellshort={'RB': 0},
        filter_var=[], fliter_var=['RB'], TradeLots={}, MappingReal={'RB': 'RB2610'}, LastRealPrice={'RB2610': 1.})
    context = SimpleNamespace(portfolio=SimpleNamespace(starting_cash=100000., long_positions={'RB2610': SimpleNamespace(total_amount=0)}, short_positions={}))
    trade_ns = {'g': g, 'get_dominant_future': lambda ins: 'RB2610', 'get_lots': lambda cash, ins: 1,
        'order_target': lambda contract, amount, **kw: orders.append({'contract': contract, 'amount': amount, **kw}), 'log': SimpleNamespace(info=lambda *args: None)}
    trade_sha = selected(directory, 4, ('Trade',), trade_ns)
    with redirect_stdout(sys.stderr): trade_ns['Trade'](context)
    require(g.BuyList == ['RB'] and orders == [{'contract': 'RB2610', 'amount': 0, 'side': 'short'}, {'contract': 'RB2610', 'amount': 1, 'side': 'long'}], 'Typo diagnostic differs')
    result = {'status': 'diagnosed_not_corrected', 'selected_ast_sha256': [pattern_sha, pool_sha, trade_sha],
        'pattern_set_index_error': pattern_error, 'synthetic_win_rate': win, 'synthetic_signed_profit_loss': pl,
        'iterative_pool_removal': {'initial': ['A', 'B', 'C'], 'remaining': remaining},
        'futures_filter_typo': {'filter_var': g.filter_var, 'fliter_var': g.fliter_var, 'buylist': g.BuyList, 'stub_order_intents': orders},
        'external_orders_messages': 0, 'not_a_backtest': True}
    save(directory / 'diagnostics.json', result); archive(root, 'Batch20 pattern compatibility, pool removal and futures filter defects archived')
    return result


def offline(root, directory):
    original = directory / 'macd-research.json'; before = file_sha(original)
    def forbidden(*args, **kwargs): raise AssertionError('Research network disabled')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        repeated = compute(directory)
    require(repeated == read(original) and file_sha(original) == before, 'Research recomputation differs/original changed')
    target = directory / 'macd-research-offline.json'; save(target, repeated)
    result = {'status': 'match', 'differences': 0, 'original_unchanged': True, 'original_file': str(original), 'original_sha256': before,
        'recomputed_file': str(target), 'recomputed_sha256': file_sha(target), 'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True}
    save(directory / 'research-offline-verification.json', result)
    archive(root, 'Batch20 frozen MACD component research offline repeat match/0; no trading reproduction claim')
    return result


def checks(root, directory):
    require(not (directory / 'checks.json').exists(), 'Checks already exist')
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py',
        'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py', 'scripts/review_strategy_batch20.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch20_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'],
        ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        print(results[-1], flush=True); require(process.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    checked = read(directory / 'checks.json')
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']), 'Checks failed')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked sets incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    reviews = read(directory / 'source-reviews/review.json')
    require(reviews['snapshot'] == SNAPSHOT and [r['source_path'] for r in reviews['sources']] ==
        [str(Path('repo/量化策略源代码') / name) for name in SOURCES], 'Reviewed source set differs')
    require(file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'], 'Prior catalog changed')
    correction = read(directory / 'source-review-supplement.json')
    require(correction['prior_review_sha256'] == file_sha(directory / 'source-reviews/review.json') and correction['source_index'] == 3 and
        correction['source_sha256'] == reviews['sources'][3]['source_sha256'] and correction['prior_review'] == reviews['sources'][3]['review'], 'Correction unbound')
    for number, row in enumerate(reviews['sources']):
        name = Path(row['source_path']).relative_to('repo/量化策略源代码').as_posix()
        review = correction['review'] if number == 3 else row['review']
        require(file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']) and review == REVIEWS[name], 'Source/review changed')
    downloaded = frames(directory); require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    research = read(directory / 'macd-research.json'); repeated = read(directory / 'macd-research-offline.json'); repeat = read(directory / 'research-offline-verification.json')
    require(research == repeated and research['backend'] == backend() and research['not_a_backtest'] and research['snapshot'] == SNAPSHOT and
        research['source_sha256'] == [r['source_sha256'] for r in reviews['sources']], 'Research/backend/source changed')
    require(repeat['status'] == 'match' and repeat['differences'] == 0 and repeat['original_unchanged'] and
        file_sha(repeat['original_file']) == repeat['original_sha256'] and file_sha(repeat['recomputed_file']) == repeat['recomputed_sha256'], 'Repeat evidence differs')
    analysis = read(directory / 'input-analysis.json')
    require(analysis['snapshot'] == SNAPSHOT and file_sha(analysis['input_file']) == analysis['input_sha256'] == research['input_sha256'], 'Input fingerprint differs')
    inputs = pd.read_parquet(analysis['input_file'])
    for item in analysis['samples']:
        if item['status'] != 'sample_ready': continue
        unit = item['unit']; expected = sample(downloaded[unit], unit)
        pd.testing.assert_frame_equal(inputs[inputs.unit.eq(unit)].reset_index(drop=True), expected)
    store = Store(root)
    index = store.load_state(store.state(SNAPSHOT), 'index_1d', filters=[('index', '=', '000300.SH')]).sort_values('date').tail(60)
    expected = pd.DataFrame({'unit': 'csi300_daily', 'timestamp': index.date.astype(str).tolist(), 'close': index.close.to_numpy()})
    pd.testing.assert_frame_equal(inputs[inputs.unit.eq('csi300_daily')].reset_index(drop=True), expected)
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        require(file_sha(target) == sha, 'Frozen implementation differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'analysis': read(directory / 'input-analysis.json'),
        'review_supplement': correction, 'research': research, 'research_offline': repeat, 'diagnostics': read(directory / 'diagnostics.json'), 'checks': checked,
        'protection': protection, 'strategy_results': [], 'progress': archive(root, 'Batch20 five reviews, MACD component research and dependencies verified; full strategy work continues')}
    target = Path('docs/handoff/2026-10-05-batch20-verification.json'); save(target, result)
    return {'status': 'ok', 'output': str(target), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'worker', 'probe', 'analyze', 'supplement', 'study', 'diagnose', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch20/20261005-macd-patterns'))
    parser.add_argument('--endpoint', choices=list(QUERIES))
    args = parser.parse_args()
    if args.action == 'worker':
        if args.endpoint is None: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
