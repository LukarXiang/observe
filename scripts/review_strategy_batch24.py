"""Audit volatility/value sources and frozen long-period research components."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
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
import pyarrow.parquet as pq
import requests
from sklearn import linear_model

from observe.data import raw, standardize
from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store, fingerprint
from observe.runs import environment, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import INSTRUMENTS
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch22 import validate_offline
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-211958-89b5'
POOL = list(INSTRUMENTS)
SOURCES = ('2022年度精选策略/85.年化20%+的极简指数增强，受RSRS和LLT启发.txt',
    '2022年度精选策略/90.相信波动率还是相信基本面？波动与估值因子A股驱动力测试.txt',
    '2020年度精选策略/77 RSRS大盘择时优化.txt', '2020年度精选策略/93 RSRS——大盘择时.txt',
    '2020年度精选策略/27 RSRS指标择时（30分钟线）.txt')
UPSTREAM = Path('docs/handoff/2026-10-06-batch23-verification.json')
BASELINE = Path('data/staging/strategies-batch23/20261006-rsrs-reversal-coins/baseline.json')
INDEX_RECEIPT = Path('docs/handoff/2026-10-05-batch12-verification.json')
INDEX_TERMINAL = Path('data/staging/strategies-batch12/20261005-daily-candidates/probes/bs_000001/result.json')
REFERENCES = (('repo/baostock', 'baostock/demo/demo_stock_Valuation_indicator_data.py'),
    ('repo/qlib', 'qlib/data/dataset/processor.py'), ('repo/qlib', 'qlib/utils/data.py'))
FIELDS = {'peTTM': 'pe_ttm', 'pbMRQ': 'pb_mrq', 'psTTM': 'ps_ttm', 'pcfNcfTTM': 'pcf_ncf_ttm'}
QUERIES = {'ps_history': {'code': 'sh.600519', 'fields': 'date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM',
    'start_date': '2014-02-11', 'end_date': '2026-09-29', 'frequency': 'd', 'adjustflag': '3'},
    'lof_160706': {'symbol': 'sz160706'}}
FILES = {'scripts/review_strategy_batch24.py', 'tests/unit/test_batch24_research.py', 'src/observe/strategy_catalog.py',
    'src/observe/data/prices.py', 'src/observe/data/standardize.py', 'src/observe/data/sources/baostock.py',
    'scripts/probe_strategy_batch14.py', 'scripts/review_strategy_batch18.py', 'scripts/review_strategy_batch21.py',
    'scripts/review_strategy_batch22.py', 'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py', 'scripts/archive_strategy_progress.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-apis.json', 'input-binding.json', 'input-analysis.json',
    'index-input.parquet', 'price-input.parquet', 'value-input.parquet', 'raw-value-audit.json',
    'probe-results.json', 'component-research.json', 'component-offline.json', 'offline-verification.json', 'diagnostics.json',
    'stock-binding.json', 'price-preflight-failure.json'}


def binding(root):
    receipt = read(UPSTREAM); state = Store(root).state(SNAPSHOT)
    require(receipt['status'] == 'ok' and receipt['reviews']['snapshot'] == SNAPSHOT and
        file_sha(BASELINE) == receipt['checks']['evidence_sha256']['baseline.json'] and
        state['tables'] == read(BASELINE)['published']['tables'], 'Accepted snapshot differs')
    upstream = read(INDEX_RECEIPT); terminal = read(INDEX_TERMINAL)
    entry = next(r for r in upstream['probes']['results'] if r['endpoint'] == 'bs_000001')
    accepted = next(r for r in entry['evidence_files'] if Path(r['file']) == INDEX_TERMINAL)
    require(upstream['status'] == 'ok_with_deferred_strategies' and file_sha(INDEX_TERMINAL) == accepted['sha256'] and
        terminal == entry['result'] and terminal['status'] == 'success' and terminal['published'] is False and
        terminal['parameters'] == {'code': 'sh.000001', 'start': '2005-01-05', 'end': '2026-09-29', 'adjustflag': '3'}, 'Accepted index probe differs')
    require(file_sha(terminal['raw_file']) == terminal['raw_sha256'], 'Accepted index raw differs')
    return {'snapshot': SNAPSHOT, 'receipt_file': str(UPSTREAM), 'receipt_sha256': file_sha(UPSTREAM),
        'baseline_file': str(BASELINE), 'baseline_sha256': file_sha(BASELINE), 'index_receipt_file': str(INDEX_RECEIPT),
        'index_receipt_sha256': file_sha(INDEX_RECEIPT), 'index_terminal_file': str(INDEX_TERMINAL),
        'index_terminal_sha256': file_sha(INDEX_TERMINAL), 'index_raw_file': terminal['raw_file'], 'index_raw_sha256': terminal['raw_sha256']}


def calendar_dates(root):
    store = Store(root); state = store.state(SNAPSHOT)
    for entry in state['tables']['calendar'].values():
        require(fingerprint(pd.read_parquet(store.root / entry['file'])) == entry['sha'], 'Calendar partition differs')
    frame = store.load_state(state, 'calendar'); dates = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    return dates[frame.is_open & dates.between('2005-01-05', '2026-09-29')].sort_values().tolist()


def start(root, directory):
    checkpoint(root, directory, 'Batch24 complete value/volatility/RSRS source research started')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    bound = binding(root); save(directory / 'input-binding.json', bound)
    prior = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory / 'source-reviews'; folder.mkdir(); sources = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'], 'Source reviewed/changed')
        shutil.copyfile(source, copied); text, encoding = read_source(source)
        sources.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    references = []; ref_dir = directory / 'references'; ref_dir.mkdir()
    for repository, relative in REFERENCES:
        source = Path(repository) / relative; copied = ref_dir / source.name; shutil.copyfile(source, copied)
        references.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only daily valuation fields and CSZScoreNorm/sample std; no source top-level, training/global zscore substitution'})
    import akshare as ak
    import baostock as bs
    apis = []
    for fn in (bs.query_history_k_data_plus, ak.fund_etf_hist_sina):
        code = inspect.getsource(fn); apis.append({'name': fn.__name__, 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'signature': str(inspect.signature(fn))})
    save(directory / 'existing-apis.json', {'baostock_version': bs.__version__, 'akshare_version': ak.__version__, 'apis': apis, 'queries': QUERIES})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(folder / 'review.json', {'snapshot': SNAPSHOT, 'sources': sources, 'previous_catalog_file': str(prior),
        'previous_catalog_sha256': file_sha(prior), 'catalog': catalog, 'references': references, 'research_pool': POOL,
        'pool_policy': 'Predeclared ten existing verified-history stocks for component research, not original index pools or trading universe',
        'strategy_results': [], 'limits': ['No original message hooks, top-level, legacy OLS, PKL/model fit or trading/NAV loops executed']})
    return archive(root, 'Batch24 five full source reviews and PS raw recovery plan archived')


def index_sample(frame, dates):
    require({'date', 'code', 'close'} <= set(frame) and frame.code.eq('sh.000001').all(), 'Index identity/schema differs')
    result = frame[['date', 'close']].copy(); result.date = pd.to_datetime(result.date).dt.strftime('%Y-%m-%d')
    result.close = pd.to_numeric(result.close, errors='raise'); result = result.sort_values('date').reset_index(drop=True)
    require(result.date.tolist() == dates and not result.date.duplicated().any() and
        np.isfinite(result.close).all() and result.close.gt(0).all(), 'Index calendar/price differs')
    return result


def stock_binding(root):
    store = Store(root); state = store.state(SNAPSHOT); rows = []
    for table in ('bars_1d', 'adj_factors', 'adj_coverage'):
        require(state['tables'].get(table), 'Missing stock input table')
        for part, entry in sorted(state['tables'][table].items()):
            path = store.root / entry['file']; frame = pd.read_parquet(path)
            require(fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'], 'Accepted stock partition differs')
            rows.append({'table': table, 'part': part, 'file': str(path), 'fingerprint': entry['sha'], 'rows': entry['rows'], 'sha256': file_sha(path)})
    return rows


def verify_stock_binding(root, rows):
    store = Store(root); state = store.state(SNAPSHOT)
    require([(r['table'], r['part']) for r in rows] == [(table, part) for table in ('bars_1d', 'adj_factors', 'adj_coverage')
        for part in sorted(state['tables'][table])], 'Stock partition scope differs')
    for row in rows:
        entry = state['tables'][row['table']][row['part']]
        require(row['file'] == str(store.root / entry['file']) and row['fingerprint'] == entry['sha'] and row['rows'] == entry['rows'] and
            file_sha(row['file']) == row['sha256'], 'Stock partition bytes/binding differs')


def validate_prices(data):
    require(set(data.instrument) == set(POOL) and not data.duplicated(['date', 'instrument']).any() and
        data.adjustment_status.eq('usable').all() and np.isfinite(data.back_factor).all() and data.back_factor.gt(0).all(), 'Stock price/adjustment contract differs')
    trading = data.loc[data.is_trading, 'close_adj']
    require(np.isfinite(trading).all() and trading.gt(0).all() and not np.isinf(data.close_adj).any(), 'Invalid traded price')
    return data


def price_sample(root, partition_rows=None):
    verify_stock_binding(root, partition_rows if partition_rows is not None else stock_binding(root))
    store = Store(root); state = store.state(SNAPSHOT); filters = [('instrument', 'in', POOL)]
    bars = store.load_state(state, 'bars_1d', columns=['date', 'instrument', 'close', 'is_trading'], filters=filters)
    view = with_adjusted(bars, store.load_state(state, 'adj_factors', filters=filters), store.load_state(state, 'adj_coverage', filters=filters))
    data = view[['date', 'instrument', 'close', 'close_adj', 'back_factor', 'is_trading', 'adjustment_status']].copy()
    data.date = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d'); data = data.sort_values(['date', 'instrument']).reset_index(drop=True)
    return validate_prices(data)


def valuation_sample(frame, day):
    require({'date', 'code', *FIELDS} <= set(frame), 'Raw valuation schema differs')
    require(frame.date.eq(day).all() and not frame.code.duplicated().any(), 'Raw valuation date/key differs')
    codes = {standardize.to_baostock(i): i for i in POOL}; result = frame[frame.code.isin(codes)].copy()
    data = pd.DataFrame({'date': result.date, 'instrument': result.code.map(codes)})
    for source, target in FIELDS.items():
        data[target] = pd.to_numeric(result[source].replace('', np.nan), errors='raise')
        require(not np.isinf(data[target]).any(), 'Infinite valuation input')
    return data.sort_values(['date', 'instrument']).reset_index(drop=True)


def raw_values(root, dates):
    folder = Path(root) / 'raw/baostock/daily_market'; records = []; parts = []; missing = []
    for day in dates:
        path = folder / f'{day}.parquet'
        if not path.exists(): missing.append(day); continue
        frame = pd.read_parquet(path, columns=['date', 'code', *FIELDS]); data = valuation_sample(frame, day)
        records.append({'file': str(path), 'sha256': file_sha(path), 'date': day, 'raw_rows': len(frame), 'selected_rows': len(data),
            'ps_nonblank_raw_rows': int(frame.psTTM.notna().sum() - frame.psTTM.eq('').sum()), 'columns': pq.read_schema(path).names})
        parts.append(data)
    require(parts, 'No old valuation input')
    return pd.concat(parts, ignore_index=True), {'source_files': records, 'missing_dates': missing, 'pool': POOL,
        'fields': FIELDS, 'not_published': True, 'strict_usable': False,
        'limits': ['Final vendor PS/PE/PB definitions and revision vintages not original-platform evidence',
            'pcfNcfTTM retained independently, never used as original PCF or PS substitute; pre2021 raw not inferred']}


def prepare(root, directory):
    require(binding(root) == read(directory / 'input-binding.json'), 'Input binding changed'); dates = calendar_dates(root)
    index = index_sample(pd.read_parquet(read(directory / 'input-binding.json')['index_raw_file']), dates)
    partitions = stock_binding(root); save(directory / 'stock-binding.json', partitions)
    prices = price_sample(root, partitions)
    save(directory / 'price-preflight-failure.json', {'action': 'price_sample preflight', 'status': 'failed',
        'error': 'ValueError: Stock price/adjustment contract differs', 'observed_known_pause_price_null_rows': 718,
        'cause': 'Initial sample-wide finite-price assertion rejected legitimate known-pause NaNs before window eligibility',
        'recovery': 'Retain raw/hfq nulls; require complete adjustment/traded prices, and reject every64-row window with a pause or missing price',
        'actual_known_pause_null_rows': int((~prices.is_trading & prices.close_adj.isna()).sum()), 'source_prices_unchanged': True})
    values, audit = raw_values(root, [d for d in dates if d >= '2021-01-04'])
    paths = {}
    for name, frame in (('index', index), ('price', prices), ('value', values)):
        path = directory / f'{name}-input.parquet'; require(not path.exists(), 'Input archive exists'); frame.to_parquet(path, index=False)
        paths[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'raw-value-audit.json', audit)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'inputs': paths, 'calendar': dates, 'research_pool': POOL,
        'local_constituent_tables': sorted(Store(root).state(SNAPSHOT)['tables'].get('index_constituents', {})),
        'raw_value_days': len(audit['source_files']), 'missing_raw_dates': audit['missing_dates'], 'not_a_backtest': True})
    return archive(root, 'Batch24 existing PS raw fields and ten-stock adjusted long-history component inputs frozen')


def validate_api(directory):
    import akshare as ak
    import baostock as bs
    row = read(directory / 'existing-apis.json')
    require(row['queries'] == QUERIES and row['akshare_version'] == ak.__version__ and row['baostock_version'] == bs.__version__, 'API query/version differs')
    for fn in (bs.query_history_k_data_plus, ak.fund_etf_hist_sina):
        item = next(r for r in row['apis'] if r['name'] == fn.__name__)
        require(item['signature'] == str(inspect.signature(fn)) and hashlib.sha256(item['source'].encode()).hexdigest() ==
            item['source_sha256'] == hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(), 'API source differs')
    return row


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(10)
    row = {'endpoint': endpoint, 'parameters': QUERIES[endpoint], 'status': 'failed', 'files': [], 'wire_responses': [],
        'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire_responses']) < 8, 'Probe request bound reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(row["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status_code': response.status_code})
        return response
    try:
        validate_api(directory)
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request):
            if endpoint == 'ps_history':
                with BaoStock(root).session() as source:
                    frame = source._rows('query_history_k_data_plus', QUERIES[endpoint], lambda: source.bs.query_history_k_data_plus(**QUERIES[endpoint]))
            else:
                import akshare as ak
                frame = ak.fund_etf_hist_sina(**QUERIES[endpoint])
        path = raw.save(root, 'volatility_value_probe_batch24', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    validate_api(directory); rows = read(directory / 'probe-results.json')['results']
    require(len(rows) == len(QUERIES) and {r['endpoint'] for r in rows} == set(QUERIES), 'Probe scope differs')
    for row in rows:
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['published'] is False and
            row['parameters'] == QUERIES[row['endpoint']] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe terminal binding differs')
        require(row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Unknown probe status')
        for item in row['wire_responses']: require(file_sha(item['file']) == item['sha256'], 'Probe response changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe file count differs'); item = row['files'][0]
            require(file_sha(item['file']) == item['sha256'], 'Probe raw changed'); frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status'] == 'success'), 'Probe rows/schema differs')
        else: require(not row['files'] and row.get('error'), 'Failed probe accepted raw/no error')
    return rows


def probe(root, directory):
    results = []
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe archive exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch24', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'{endpoint}-worker-log.json', {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, process.stderr)
        except subprocess.TimeoutExpired as exc:
            if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'parameters': QUERIES[endpoint],
                'status': 'timeout', 'files': [], 'wire_responses': [], 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'error': str(exc)})
        results.append(read(folder / 'result.json')); archive(root, f'Batch24 {endpoint} probe {results[-1]["status"]}, not published')
    save(directory / 'probe-results.json', {'results': results, 'published': False}); validate_probes(directory)
    return {'status': 'archived', 'results': results}


def reference_z(values):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and len(x) >= 2 and np.isfinite(x).all(), 'Invalid cross-section inputs')
    output = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        column = x[:, j].tolist(); mean = math.fsum(column) / len(column)
        sigma = math.sqrt(math.fsum((v - mean) ** 2 for v in column) / (len(column) - 1))
        if sigma: output[:, j] = [(v - mean) / sigma for v in column]
    return output


def reference_wave(prices):
    data = np.asarray(prices, dtype=float)
    require(data.ndim == 2 and data.shape[0] == 64 and data.shape[1] >= 2 and
        np.isfinite(data).all() and (data > 0).all(), 'Invalid wave inputs')
    factors = []
    for j in range(data.shape[1]):
        returns = [data[k, j] / data[k - 1, j] - 1 for k in range(1, 64)]; mean = math.fsum(returns) / 63
        factors.append([math.sqrt(math.fsum((v - mean) ** 2 for v in returns) / 62),
            math.sqrt(math.fsum((v - mean) ** 2 for v in returns if v < mean) / 62)])
    z = reference_z(factors); return z[:, 0] * .5 + z[:, 1] * .5


def wave_window_usable(prices, trading):
    data = np.asarray(prices, dtype=float); flags = np.asarray(trading, dtype=bool)
    require(data.ndim == 2 and data.shape == flags.shape and data.shape[0] == 64, 'Invalid window shape')
    return bool(np.isfinite(data).all() and (data > 0).all() and flags.all())


def reference_gate(prices):
    data = list(map(float, prices)); require(len(data) == 20 and all(math.isfinite(v) and v > 0 for v in data), 'Invalid gate inputs')
    ranges = [int(max(data[k:k + 10]) / min(data[k:k + 10]) * 100) / 100. for k in range(10)]
    return not (min(data[15:19]) == min(data[10:20]) and max(ranges[5:]) != max(ranges))


def compare(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and not np.isinf(a).any(), 'Component validity differs')
    finite = np.isfinite(a); delta = np.abs(a[finite] - b[finite]); maximum = float(delta.max()) if finite.any() else 0.
    require(maximum < 1e-8, 'Component values differ'); return maximum


def validate_inputs(root, directory):
    info = read(directory / 'input-analysis.json')
    require(info['snapshot'] == SNAPSHOT and info['research_pool'] == POOL and info['not_a_backtest'] is True and
        set(info['inputs']) == {'index', 'price', 'value'} and info['calendar'] == calendar_dates(root) and
        info['local_constituent_tables'] == sorted(Store(root).state(SNAPSHOT)['tables'].get('index_constituents', {})), 'Research scope differs')
    frames = {}
    for name, item in info['inputs'].items():
        path = directory / f'{name}-input.parquet'
        require(item['file'] == str(path) and file_sha(path) == item['sha256'], 'Research input changed')
        frame = pd.read_parquet(path)
        require(len(frame) == item['rows'] and frame.date.min() == item['first'] and frame.date.max() == item['last'], 'Input profile differs')
        frames[name] = frame
    require(frames['index'].date.tolist() == info['calendar'] and set(frames['price'].instrument) == set(POOL) and
        set(frames['value'].instrument) <= set(POOL), 'Component universe differs')
    return info, frames


def compute(root, directory):
    require(binding(root) == read(directory / 'input-binding.json'), 'Research binding changed'); info, frames = validate_inputs(root, directory)
    verify_stock_binding(root, read(directory / 'stock-binding.json'))
    index, prices, values = [frames[n] for n in ('index', 'price', 'value')]
    require(index.date.tolist() == info['calendar'], 'Research calendar differs')
    holder = {}; ns_gate = {'floor': math.floor, 'kData': lambda context, stock, count: {'close': holder['close']}, 'log': SimpleNamespace(info=lambda *a: None)}
    gate_ast = selected(directory, 0, ('indexwarn',), ns_gate); gates = []
    for k in range(19, len(index)):
        holder['close'] = index.close.iloc[k - 19:k + 1].to_numpy(); actual = ns_gate['indexwarn'](None)
        require(actual == reference_gate(holder['close']), 'Gate differs')
        gates.append({'last_input_date': index.date.iloc[k], 'idxwarn': bool(actual), 'ready': not bool(actual)})
    ns_wave = {'pd': pd, 'g': SimpleNamespace(yb=63), 'get_price': lambda *a, **kw: {'close': holder['prices']}}
    wave_ast = selected(directory, 1, ('get_df_wave', 'std_ud'), ns_wave)
    matrix = prices.pivot(index='date', columns='instrument', values='close_adj').reindex(index=info['calendar'], columns=POOL)
    trading = prices.pivot(index='date', columns='instrument', values='is_trading').reindex(index=info['calendar'], columns=POOL).fillna(False)
    waves = []; excluded = []; wave_max = 0.
    for k in range(63, len(matrix)):
        window = matrix.iloc[k - 63:k + 1]
        if not wave_window_usable(window.to_numpy(), trading.iloc[k - 63:k + 1].to_numpy()):
            excluded.append(str(matrix.index[k])); continue
        holder['prices'] = window; actual = ns_wave['get_df_wave'](POOL, SimpleNamespace(previous_date=matrix.index[k]))
        reference = reference_wave(window.to_numpy()); delta = compare(actual.reindex(POOL).wave, reference); wave_max = max(wave_max, delta)
        waves.append({'last_input_date': str(matrix.index[k]), 'values': {i: float(actual.wave[i]) if pd.notna(actual.wave[i]) else None for i in POOL}})
    require(waves, 'No complete ten-stock wave windows')
    ns_value = {'pd': pd, 'valuation': SimpleNamespace(pb_ratio='pb_ratio', pe_ratio='pe_ratio', ps_ratio='ps_ratio', code=SimpleNamespace(in_=lambda stocks: stocks)),
        'query': lambda *args: SimpleNamespace(filter=lambda codes: codes), 'get_fundamentals': lambda q, date: holder['values'].copy()}
    value_ast = selected(directory, 1, ('get_df_value',), ns_value); value_rows = []; incomplete = []; value_max = 0.
    for day, frame in values.groupby('date', sort=True):
        packet = frame.set_index('instrument').reindex(POOL)
        if not np.isfinite(packet[['pb_mrq', 'pe_ttm', 'ps_ttm']].to_numpy()).all(): incomplete.append(day); continue
        holder['values'] = packet[['pb_mrq', 'pe_ttm', 'ps_ttm']].rename(columns={'pb_mrq': 'pb_ratio', 'pe_ttm': 'pe_ratio', 'ps_ttm': 'ps_ratio'}).assign(code=POOL)
        actual = ns_value['get_df_value'](POOL, SimpleNamespace(previous_date=day))
        reference = reference_z(packet[['pb_mrq', 'pe_ttm', 'ps_ttm']].to_numpy()).sum(axis=1) / 3
        delta = compare(actual.reindex(POOL).value, reference); value_max = max(value_max, delta)
        value_rows.append({'last_input_date': day, 'values': {i: float(actual.value[i]) if pd.notna(actual.value[i]) else None for i in POOL}})
    require(value_rows, 'No complete raw value cross-sections')
    return {'snapshot': SNAPSHOT, 'inputs': info['inputs'], 'research_pool': POOL, 'backend': {'numpy': np.__version__, 'pandas': pd.__version__},
        'selected_ast_sha256': {'gate': gate_ast, 'wave': wave_ast, 'value': value_ast}, 'gates': gates,
        'wave': {'window_price_rows': 64, 'rows': waves, 'excluded_last_dates': excluded, 'max_abs_difference': wave_max},
        'value': {'rows': value_rows, 'incomplete_dates': incomplete, 'max_abs_difference': value_max},
        'not_a_backtest': True, 'limits': ['Explicit math.floor injection is research runtime, original platform import unproved',
            'Ten-stock component cross-sections are not original000009/000906 pool or selection',
            'Wave rejects every window with any known pause/missing price; no unknown get_price fill substitution',
            'Vendor PS/PE/PB and hfq ratios not original-platform equivalence; PCF not used; no trades/cost/NAV']}


def diagnostics(root, directory):
    orders = []; quiet = SimpleNamespace(info=lambda *a: None); rows = []
    ns = {'kData': lambda *a: {'close': np.arange(100., 120.)}, 'log': quiet}; selected(directory, 0, ('indexwarn',), ns)
    try: ns['indexwarn'](None)
    except NameError as exc: rows.append({'case': 'gate_floor_missing_without_injection', 'error': str(exc)})
    ns = {'pd': pd, 'g': SimpleNamespace(feasible_stocks=['A'], factor='value', factor_sort={'value': True}, percent=.1), 'get_df_wave': lambda *a: pd.DataFrame({'wave': [1.]}, index=['A']),
        'get_df_value': lambda *a: pd.DataFrame({'value': [1.]}, index=['A'])}; selected(directory, 1, ('get_holding_list',), ns)
    try: ns['get_holding_list'](None)
    except AttributeError as exc: rows.append({'case': 'legacy_dataframe_sort_missing', 'error': str(exc)})
    ns = {'get_price': lambda *a, **kw: {'paused': pd.DataFrame({'A': [0.]})}, 'get_current_data': lambda: {},
        'attribute_history': lambda *a, **kw: pd.DataFrame({'paused': np.zeros(63)})}; selected(directory, 1, ('set_feasible_stocks',), ns)
    try: ns['set_feasible_stocks'](['A'], 63, SimpleNamespace(current_dt='2024-03-29'))
    except TypeError as exc: rows.append({'case': 'paused_dataframe_sum_column_names', 'error': str(exc)})
    ns = {'order_target_value': lambda *a: orders.append(a)}; selected(directory, 1, ('rebalance',), ns)
    try: ns['rebalance'](SimpleNamespace(portfolio=SimpleNamespace(portfolio_value=100000., positions={})), [])
    except ZeroDivisionError: rows.append({'case': 'empty_selection_division', 'error': 'ZeroDivisionError', 'orders': []})
    ns = {'np': np, 'g': SimpleNamespace(init=True, security='160706.XSHE', N=18, M=1100, ans=list(np.linspace(.8, 1.2, 1100)), buy=.7, sell=-.7),
        'log': quiet, 'attribute_history': lambda *a, **kw: pd.DataFrame({'volume': np.ones(18)}),
        'order_value': lambda *a: orders.append(a), 'order_target': lambda *a: orders.append(a)}
    selected(directory, 2, ('market_open',), ns); output = io.StringIO()
    with redirect_stdout(output): ns['market_open'](SimpleNamespace(current_dt=datetime(2024, 3, 29, 9, 30), portfolio=SimpleNamespace(available_cash=100000.)))
    require(not orders and ns['g'].init is False and len(ns['g'].ans) == 1100, 'LOF initial callback differs')
    rows.append({'case': 'lof_initial_zero_signal', 'stdout': output.getvalue(), 'orders': [], 'beta_pool_rows': 1100, 'volume_rows': 18})
    rows.append({'case': 'old_pandas_regression_rolling_missing', 'pandas_stats_directory_exists': (Path(pd.__file__).parent / 'stats').exists(),
        'rolling_mean_exists': hasattr(pd, 'rolling_mean'), 'rolling_std_exists': hasattr(pd, 'rolling_std'), 'not_replaced': True})
    x = np.arange(400.) + 100; high = x * (1.01 + .002 * np.sin(x / 10)) + 2; history_calls = []
    def history(stock, count, unit, field):
        history_calls.append({'count': count, 'unit': unit, 'field': field})
        return pd.DataFrame({field: (high if field == 'high' else x)[-count:]})
    ns = {'np': np, 'linear_model': linear_model, 'mean': np.mean, 'std': np.std, 'attribute_history': history,
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'record': lambda **kw: None, 'log': quiet,
        'order_value': lambda *a: orders.append(a), 'order_target_value': lambda *a: orders.append(a)}
    selected(directory, 4, ('initialize', 'before_trading_start', 'handle_data'), ns); context = SimpleNamespace(portfolio=SimpleNamespace(available_cash=100000.))
    ns['initialize'](context); ns['before_trading_start'](context); first = len(context.beat_list); ns['before_trading_start'](context)
    second = len(context.beat_list); require((first, second) == (382, 764), 'Thirty-minute warmup pool differs')
    callbacks = []
    for k in range(1, 91):
        before = len(context.z_beta_r2_list); ns['handle_data'](context, None)
        if len(context.z_beta_r2_list) != before: callbacks.append(k)
    require(callbacks == [30, 60, 90] and len(context.beat_list) == 764 and not orders, 'Thirty-minute callbacks/pool differs')
    rows.append({'case': 'thirty_minute_accumulating_pool', 'warmup_sizes': [first, second], 'signal_callbacks': callbacks,
        'beta_pool_after_callbacks': len(context.beat_list), 'history_calls': history_calls, 'orders': [], 'synthetic_only': True})
    ns = {}; selected(directory, 1, ('std_ud',), ns)
    actual = ns['std_ud'](pd.DataFrame({'A': [.01, .02, .03]})).iloc[0]
    require(actual > 0 and abs(actual - math.sqrt(.0001 / 2)) < 1e-15, 'Below-mean downside differs')
    rows.append({'case': 'positive_returns_still_have_below_mean_downside', 'returns': [.01, .02, .03], 'value': float(actual)})
    require(len(rows) == 8, 'Expected diagnostics missing')
    return {'cases': rows, 'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'not_a_backtest': True, 'limits': ['Only exact reviewed functions with explicit local synthetic history and order stubs; no messages/legacy OLS/top-level']}


def study(root, directory):
    result = compute(root, directory); diagnosed = diagnostics(root, directory)
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnosed)
    return archive(root, 'Batch24 long index gate, ten-stock wave/value components and eight diagnostics archived')


def offline(root, directory):
    def denied(*args, **kwargs): raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), patch.object(requests.sessions.Session, 'request', denied):
        result = compute(root, directory); diagnosed = diagnostics(root, directory)
    save(directory / 'component-offline.json', result); before = file_sha(directory / 'component-research.json')
    require(result == read(directory / 'component-research.json') and diagnosed == read(directory / 'diagnostics.json') and
        before == file_sha(directory / 'component-offline.json'), 'Offline component differs')
    save(directory / 'offline-verification.json', {'status': 'match', 'differences': 0, 'original_unchanged': True,
        'original_sha256': before, 'recomputed_sha256': file_sha(directory / 'component-offline.json'),
        'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True})
    return archive(root, 'Batch24 original research components byte-identical offline; not original trading reproduction')


def checks(root, directory):
    validate_probes(directory); validate_offline(directory)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
        'scripts/verify_financial_import.py', 'scripts/review_strategy_batch24.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch24_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True); row = {'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        results.append(row); print(row, flush=True); require(process.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    checked = read(directory / 'checks.json'); require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']) and
        FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checks incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()) and
        all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Checked code/evidence changed')
    reviews = read(directory / 'source-reviews/review.json'); require(binding(root) == read(directory / 'input-binding.json'), 'Binding changed')
    require([r['source_path'] for r in reviews['sources']] == [str(Path('repo/量化策略源代码') / n) for n in SOURCES] and
        reviews['research_pool'] == POOL and file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'], 'Review scope differs')
    for name, row in zip(SOURCES, reviews['sources'], strict=True):
        require(file_sha(row['source_path']) == row['source_sha256'] == row['source_copy_sha256'] == file_sha(row['source_copy']) and REVIEWS[name] == row['review'], 'Source/rules changed')
    for row in reviews['references']: require(file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')
    info = read(directory / 'input-analysis.json'); dates = calendar_dates(root); require(info['calendar'] == dates, 'Calendar changed')
    audit = read(directory / 'raw-value-audit.json'); actual, audit_again = raw_values(root, [d for d in dates if d >= '2021-01-04'])
    require(audit == audit_again and info['raw_value_days'] == len(audit['source_files']) and info['missing_raw_dates'] == audit['missing_dates'], 'Raw value audit differs')
    pd.testing.assert_frame_equal(actual, pd.read_parquet(info['inputs']['value']['file']))
    pd.testing.assert_frame_equal(price_sample(root, read(directory / 'stock-binding.json')), pd.read_parquet(info['inputs']['price']['file']))
    pd.testing.assert_frame_equal(index_sample(pd.read_parquet(read(directory / 'input-binding.json')['index_raw_file']), dates), pd.read_parquet(info['inputs']['index']['file']))
    require(compute(root, directory) == read(directory / 'component-research.json') == read(directory / 'component-offline.json') and
        diagnostics(root, directory) == read(directory / 'diagnostics.json'), 'Reconstructed research differs')
    probes = validate_probes(directory); offline_row = validate_offline(directory)
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed'); protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'input_analysis': info, 'raw_value_audit': audit,
        'research': read(directory / 'component-research.json'), 'diagnostics': read(directory / 'diagnostics.json'),
        'probes': probes, 'offline': offline_row, 'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch24 PS raw recovery and original long-period research accepted; whole library continues')}
    output = Path('docs/handoff/2026-10-06-batch24-verification.json'); save(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'worker', 'probe', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch24/20261006-volatility-value-raw'))
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    if args.action == 'worker':
        require(args.endpoint is not None, 'Worker endpoint required'); result = worker(args.root, args.directory, args.endpoint)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
