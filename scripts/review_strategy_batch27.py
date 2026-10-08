"""Archive three exact-source component studies; missing trading inputs stay blocked."""
import argparse
import ast
import hashlib
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

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import INSTRUMENTS
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261006-145049-e06c'
UPSTREAM = Path('docs/handoff/2026-10-06-batch26-verification.json')
ACCEPTED = Path('data/staging/strategies-batch24/20261006-volatility-value-raw')
PRICE_RECEIPT = Path('docs/handoff/2026-10-06-batch24-verification.json')
SOURCES = ('2022年度精选策略/39.个股择时有效？因子选股+双均线择时效果初步尝试.txt',
    '2023年度精选策略/44.海龟突破系统，19-21年化150，无未来函数和过拟合.txt',
    '2023年度精选策略/65.PB-POE+双均线--加入jq以来第一个成功的交易策略 一.txt')
REFERENCES = (('repo/rqalpha', 'rqalpha/examples/turtle.py'),
    ('repo/baostock', 'baostock/demo/demo_profit_data.py'), ('repo/qlib', 'qlib/data/dataset/processor.py'))
FILES = ('scripts/review_strategy_batch27.py', 'tests/unit/test_batch27_research.py', 'src/observe/strategy_catalog.py',
    'scripts/review_strategy_batch21.py', 'scripts/review_strategy_batch18.py', 'scripts/run_strategy_batch4.py',
    'scripts/verify_strategy_batch3.py', 'scripts/archive_strategy_progress.py', 'src/observe/data/store.py',
    'src/observe/data/sources/baostock.py', 'src/observe/data/raw.py', 'src/observe/runs.py', 'scripts/verify_offline_tests.py')
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'input-analysis.json', 'existing-apis.json',
    'price-input.parquet', 'value-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'supplement-api.json', 'supplement-result.json'}
QUERIES = {'pool_2020': {'kind': 'pool', 'index': '000300.SH', 'date': '2020-01-02'},
    'roe_2020': {'kind': 'profit', 'code': 'sh.600519', 'year': 2020, 'quarter': 1},
    'minute_2020': {'kind': 'minute', 'code': 'sh.600519', 'fields': 'date,time,code,open,high,low,close,volume,amount,adjustflag',
        'start_date': '2020-01-02', 'end_date': '2020-01-03', 'frequency': '5', 'adjustflag': '3'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(UPSTREAM); accepted = read(PRICE_RECEIPT); state = Store(root).state(SNAPSHOT)
    require(receipt['status'] == accepted['status'] == 'ok' and receipt['recovery']['snapshot'] == SNAPSHOT and
        state['batch_id'] == receipt['recovery']['batch_id'], 'Accepted recovery differs')
    paths = {'upstream': UPSTREAM, 'price_receipt': PRICE_RECEIPT}
    for name in ('price-input.parquet', 'value-input.parquet', 'stock-binding.json'):
        path = ACCEPTED / name
        require(file_sha(path) == accepted['checks']['evidence_sha256'][name], 'Accepted component input changed')
        paths[name] = path
    rows = read(ACCEPTED / 'stock-binding.json')
    require({(r['table'], r['part']) for r in rows} == {(t, p) for t in ('bars_1d', 'adj_factors', 'adj_coverage') for p in state['tables'][t]}, 'Accepted stock partition scope differs')
    for row in rows:
        entry = state['tables'][row['table']][row['part']]; actual = Store(root).root / entry['file']
        require(entry['sha'] == row['fingerprint'] and entry['rows'] == row['rows'] and
            actual.resolve() == Path(row['file']).resolve() and file_sha(actual) == row['sha256'], 'Accepted stock bytes/reference changed')
    result = {'snapshot': SNAPSHOT, 'files': {n: {'file': str(p), 'sha256': file_sha(p)} for n, p in paths.items()},
        'stock_partitions': rows, 'research_pool': list(INSTRUMENTS), 'not_a_backtest': True,
        'limits': ['Ten-stock research cross-section is not any original index pool', 'Final vendor valuations lack historical revisions']}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Batch27 binding changed')
    return result


def api_evidence():
    import baostock as bs
    apis = []
    for fn in (bs.query_hs300_stocks, bs.query_profit_data, bs.query_history_k_data_plus):
        code = inspect.getsource(fn)
        apis.append({'name': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'version': bs.__version__, 'queries': QUERIES, 'apis': apis}


def start(root, directory):
    checkpoint(root, directory, 'Batch27 factor/MA and actual turtle source audit started')
    save(directory / 'input-binding.json', binding(root))
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
            'use': 'Original turtle differs from RQAlpha ATR/55/20; BaoStock quarterly ROE is not platform vintage; sample CS norm context'})
    save(directory / 'existing-apis.json', api_evidence())
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': references, 'catalog': catalog,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path), 'snapshot': SNAPSHOT,
        'strategy_results': [], 'not_a_backtest': True})
    return archive(root, 'Batch27 three complete source reviews frozen; pool/ROE/intraday gaps retained')


def prepare(root, directory):
    binding(root, directory); store = Store(root); state = store.state(SNAPSHOT)
    require(not any((directory / n).exists() for n in ('price-input.parquet', 'value-input.parquet', 'input-analysis.json')), 'Prepared input archive exists')
    price = pd.read_parquet(ACCEPTED / 'price-input.parquet')
    bars = store.load_state(state, 'bars_1d', columns=['date', 'instrument', 'close', 'high', 'low', 'is_trading'],
        filters=[('instrument', 'in', list(INSTRUMENTS))])
    bars.date = pd.to_datetime(bars.date).dt.strftime('%Y-%m-%d')
    out = price.merge(bars, on=['date', 'instrument'], validate='one_to_one', suffixes=('', '_bar'))
    require(len(out) == len(price) and out.is_trading.eq(out.is_trading_bar).all(), 'Raw/adjusted price binding differs')
    pd.testing.assert_series_equal(out.close, out.close_bar, check_names=False)
    out = out.drop(columns=['close_bar', 'is_trading_bar']); out['high_adj'] = out.high * out.back_factor; out['low_adj'] = out.low * out.back_factor
    out.to_parquet(directory / 'price-input.parquet', index=False)
    shutil.copyfile(ACCEPTED / 'value-input.parquet', directory / 'value-input.parquet')
    inputs = {}
    for name in ('price', 'value'):
        p = directory / f'{name}-input.parquet'; frame = pd.read_parquet(p)
        inputs[name] = {'file': str(p), 'sha256': file_sha(p), 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'inputs': inputs, 'not_a_backtest': True,
        'pool': list(INSTRUMENTS), 'local_constituent_tables': sorted(state['tables'].get('index_constituents', {})),
        'price_policy': 'Verified back-adjusted historical windows; only threshold/sign relations, no intraday price/strategy targets'})
    return archive(root, 'Batch27 accepted ten-stock long history and PB/PE/PS component inputs frozen')


def inputs(directory):
    info = read(directory / 'input-analysis.json'); frames = {}
    require(info['snapshot'] == SNAPSHOT and info['pool'] == list(INSTRUMENTS) and info['not_a_backtest'], 'Research scope changed')
    for name, row in info['inputs'].items():
        p = directory / f'{name}-input.parquet'; require(str(p) == row['file'] and file_sha(p) == row['sha256'], 'Component input changed')
        frame = pd.read_parquet(p); require(len(frame) == row['rows'] and frame.date.min() == row['first'] and frame.date.max() == row['last'], 'Input profile differs')
        require(not frame.duplicated(['date', 'instrument']).any(), 'Duplicate research rows'); frames[name] = frame
    require(set(frames) == {'price', 'value'}, 'Component inputs missing'); return frames


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(SOURCES) and doc['not_a_backtest'], 'Source review scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior source catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference bytes changed')


def expression(directory, number, function, target):
    fn = next(n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name == function)
    nodes = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
    require(len(nodes) == 1, 'Ambiguous/missing original expression')
    node = nodes[0].value
    return compile(ast.Expression(node), '<frozen-original-expression>', 'eval'), hashlib.sha256(ast.dump(node).encode()).hexdigest()


def reference_z(values):
    x = np.asarray(values, dtype=float); require(x.ndim == 2 and len(x) > 1 and not np.isinf(x).any(), 'Invalid value cross-section')
    out = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        valid = np.isfinite(x[:, j]); finite = x[valid, j]; n = len(finite)
        if n < 2: continue
        mean = math.fsum(map(float, finite)) / n
        sd = math.sqrt(math.fsum((float(v) - mean) ** 2 for v in finite) / (n - 1))
        if sd > 0: out[valid, j] = (finite - mean) / sd
    return out


class QueryField:
    def in_(self, names): return names


def value_function(directory):
    holder = {}; field = QueryField()
    ns = {'pd': pd, 'query': lambda *args: SimpleNamespace(filter=lambda *args: None),
        'valuation': SimpleNamespace(**{n: field for n in ('code', 'pb_ratio', 'pe_ratio', 'ps_ratio')}),
        'get_fundamentals': lambda *args, **kw: holder['frame'].copy()}
    sha = selected(directory, 0, ('get_df_value',), ns)
    return holder, ns['get_df_value'], sha


def compute(directory):
    validate_sources(directory); frames = inputs(directory); holder, original, value_sha = value_function(directory); values = []
    for day, frame in frames['value'].groupby('date', sort=True):
        frame = frame.sort_values('instrument'); holder['frame'] = frame[['instrument', 'pb_mrq', 'pe_ttm', 'ps_ttm']].rename(
            columns={'instrument': 'code', 'pb_mrq': 'pb_ratio', 'pe_ttm': 'pe_ratio', 'ps_ttm': 'ps_ratio'})
        actual = original(frame.instrument.tolist(), SimpleNamespace(previous_date=day)).value.to_numpy()
        z = reference_z(frame[['pb_mrq', 'pe_ttm', 'ps_ttm']]); expected = np.sum(z, axis=1) / 3
        require(np.array_equal(np.isnan(actual), np.isnan(expected)), 'Value mask differs')
        delta = np.abs(actual[np.isfinite(actual)] - expected[np.isfinite(expected)])
        require(not len(delta) or delta.max() < 1e-10, 'Value component differs')
        values.append({'date': str(day), 'rows': len(frame), 'max_abs_difference': float(delta.max()) if len(delta) else 0.,
            'value_sha256': hashlib.sha256(actual.astype('<f8').tobytes()).hexdigest(), 'not_original_pool': True})
    specs = ((2, 'check_holding', 'MAs'), (2, 'check_holding', 'MAl'), (1, 'choice_stock', 'p_max'),
        (1, 'choice_stock', 'zdf'), (1, 'chicang', 'bf_p'))
    exprs = {name: expression(directory, n, fn, name) for n, fn, name in specs}; rows = []; blocked = 0; boundaries = []
    for stock, all_rows in frames['price'].groupby('instrument', sort=True):
        all_rows = all_rows.sort_values('date')
        require(all_rows.loc[all_rows.is_trading, 'adjustment_status'].eq('usable').all(), 'Unverified trading history cannot be skipped')
        valid = all_rows.is_trading
        frame = all_rows[valid].reset_index(drop=True)
        require(np.isfinite(frame[['close_adj', 'high_adj', 'low_adj']]).all().all() and frame.low_adj.gt(0).all(), 'Invalid verified price window')
        blocked += int((~valid).sum())
        for k in range(80, len(frame)):
            close = frame.close_adj.iloc[k - 59:k + 1]; high = frame.high_adj.iloc[k - 80:k + 1]; low = frame.low_adj.iloc[k - 2:k + 1]
            ns = {'close_data_MAs': pd.DataFrame({'close': close.iloc[-10:]}), 'close_data_MAl': pd.DataFrame({'close': close}),
                'df55': pd.DataFrame({'high': high}), 'df1': {'close': close.iloc[-9:].to_numpy()},
                'df5': pd.DataFrame({'bf': (frame.high_adj.iloc[k - 2:k + 1].to_numpy() - low.to_numpy()) / low.to_numpy()})}
            actual = {name: float(eval(code, {}, ns)) for name, (code, _) in exprs.items()}
            expected = {'MAs': math.fsum(map(float, close.iloc[-10:])) / 10, 'MAl': math.fsum(map(float, close)) / 60,
                'p_max': max(map(float, high)), 'zdf': (float(close.iloc[-1]) - float(close.iloc[-9])) / float(close.iloc[-9]),
                'bf_p': max((float(h) - float(l)) / float(l) for h, l in zip(frame.high_adj.iloc[k - 2:k + 1], low, strict=True))}
            error = max(abs(actual[n] - expected[n]) / max(1., abs(expected[n])) for n in actual)
            require(error < 1e-12, 'Price component differs')
            current = float(close.iloc[-1]); flags = lambda v: [current > v['MAl'] and v['MAs'] > v['MAl'], current < v['MAs'] and v['MAs'] < v['MAl']]
            if flags(actual) != flags(expected): boundaries.append({'instrument': stock, 'date': frame.date.iloc[k], 'pandas': actual, 'fsum': expected})
            rows.append({'instrument': stock, 'date': frame.date.iloc[k], **actual, 'relative_difference': error,
                'ma_buy': flags(actual)[0], 'ma_sell': flags(actual)[1], 'not_an_intraday_signal': True})
    return {'snapshot': SNAPSHOT, 'source_value_ast_sha256': value_sha, 'source_expressions': {n: sha for n, (_, sha) in exprs.items()},
        'value_cross_sections': values, 'price_windows': rows, 'excluded_paused_or_unverified_rows': blocked,
        'strict_ma_condition_boundaries': boundaries, 'not_a_backtest': True, 'original_strategy_complete': False,
        'limits': ['Ten-stock CS normalization is not original index ranking', 'No current intraday p, cost, trades, NAV or turtle targets',
            'Verified trading rows skip known pauses for attribute_history approximation; platform adjustment/windows unproved',
            'Original 9-close arithmetic evaluated with positional array operands only; current pandas integer-index defect retained separately']}


def diagnostics(directory):
    cases = []
    ns = {}; selected(directory, 0, ('rebalance',), ns)
    try: ns['rebalance'](SimpleNamespace(portfolio=SimpleNamespace(portfolio_value=1000)), [])
    except ZeroDivisionError: cases.append({'case': 'empty_factor_list', 'error': 'ZeroDivisionError'})
    ns = {'get_price': lambda *a, **k: {'paused': pd.DataFrame([[0]], columns=['x'])}, 'get_current_data': lambda: {},
        'attribute_history': lambda *a, **k: pd.DataFrame({'paused': [0]})}
    selected(directory, 0, ('set_feasible_stocks',), ns)
    try: ns['set_feasible_stocks'](['x'], 63, SimpleNamespace(current_dt='2024-01-02'))
    except TypeError: cases.append({'case': 'sum_paused_dataframe', 'error': 'TypeError'})
    ns = {'g': SimpleNamespace(feasible_stocks=['x'], factor='value', factor_sort={'value': True}, percent=.1),
        'get_df_marketcap': lambda *a: pd.DataFrame(), 'get_df_beta': lambda *a: pd.DataFrame(),
        'get_df_value': lambda *a: pd.DataFrame({'value': [1.]}, index=['x'])}
    selected(directory, 0, ('get_holding_list',), ns)
    try: ns['get_holding_list'](None)
    except AttributeError: cases.append({'case': 'legacy_dataframe_sort', 'error': 'AttributeError'})
    calls = []; g = SimpleNamespace(days=0, refresh_rate=30)
    ns = {'g': g, 'check_stocks': lambda c: calls.append(c.callback) or [], 'check_holding': lambda c: []}
    selected(directory, 2, ('trade',), ns)
    for k in range(91): ns['trade'](SimpleNamespace(callback=k, portfolio=SimpleNamespace(positions={}, cash=1000)))
    require(calls == [0, 30, 60, 90], 'Original schedule differs')
    cases.append({'case': 'actual_30_callback_refresh', 'selection_callbacks': calls})
    code, _ = expression(directory, 1, 'choice_stock', 'zdf')
    try: eval(code, {}, {'df1': pd.DataFrame({'close': [9., 10.]}, index=pd.date_range('2024-01-02', periods=2))})
    except KeyError: cases.append({'case': 'legacy_series_integer_index', 'error': 'KeyError'})
    require(len(cases) == 5, 'Original diagnostics did not reach expected defects')
    return {'cases': cases, 'synthetic_stub_diagnostics_only': True, 'not_a_backtest': True}


def study(root, directory):
    binding(root, directory); save(directory / 'component-research.json', compute(directory)); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch27 exact PB/PE/PS and long 81/9/3 plus 10/60 price components independently checked; no trades')


def worker(root, directory, endpoint):
    if endpoint == 'ak_roe': return supplement_worker(root, directory)
    cfg = QUERIES[endpoint]; result = {'endpoint': endpoint, 'parameters': cfg, 'status': 'failed', 'published': False, 'files': [],
        'api_sha256': file_sha(directory / 'existing-apis.json')}; folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'API changed'); socket.setdefaulttimeout(10)
        with BaoStock(root).session() as src:
            if cfg['kind'] == 'pool': frame = src.index_constituents(cfg['index'], cfg['date'])
            elif cfg['kind'] == 'profit':
                params = {k: v for k, v in cfg.items() if k != 'kind'}
                frame = src._rows('query_profit_data', params, lambda: src.bs.query_profit_data(**params))
            else:
                params = {k: v for k, v in cfg.items() if k != 'kind'}
                frame = src._rows('query_history_k_data_plus', params, lambda: src.bs.query_history_k_data_plus(**params))
        p = raw.save(root, 'batch27_dependency_probe', endpoint, directory.name, frame)
        result.update(status='success' if len(frame) else 'empty', files=[{'file': str(p), 'sha256': file_sha(p), 'rows': len(frame), 'columns': list(frame)}])
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    save(folder / 'result.json', result); return result


def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API drift')
    rows = read(directory / 'probe-results.json')['results']; require(len(rows) == len(QUERIES) and {r['endpoint'] for r in rows} == set(QUERIES), 'Probe scope differs')
    for r in rows:
        require(r == read(directory / 'probes' / r['endpoint'] / 'result.json') and r['parameters'] == QUERIES[r['endpoint']] and
            r['published'] is False and r['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe terminal binding differs')
        require(r['status'] in ('success', 'empty', 'failed', 'timeout'), 'Unknown probe status')
        if r['status'] in ('success', 'empty'):
            require(len(r['files']) == 1, 'Probe raw missing'); item = r['files'][0]; frame = pd.read_parquet(item['file'])
            require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (r['status'] == 'success'), 'Probe raw changed')
        else: require(not r['files'] and r.get('error'), 'Failed probe accepted raw/no error')
    return rows


def probe(root, directory):
    rows = []
    for endpoint in QUERIES:
        logs = directory / 'logs'; logs.mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'scripts.review_strategy_batch27', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                p = directory / 'probes' / endpoint / 'result.json'
                if not p.exists(): save(p, {'endpoint': endpoint, 'parameters': QUERIES[endpoint], 'status': 'timeout', 'published': False,
                    'files': [], 'api_sha256': file_sha(directory / 'existing-apis.json'), 'error': 'Parent deadline45s'})
        rows.append(read(directory / 'probes' / endpoint / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows}); validate_probes(directory)
    return archive(root, 'Batch27 historical pool/ROE/5minute existing API probes archived, no publication')


def supplement_api():
    import akshare as ak
    fn = ak.stock_financial_analysis_indicator; code = inspect.getsource(fn)
    return {'version': ak.__version__, 'function': fn.__name__, 'parameters': {'symbol': '600519', 'start_year': '2020'},
        'signature': str(inspect.signature(fn)), 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'limit': 'One-stock quarterly final indicators do not prove original ROE vintages/full-market pool; original API dropna retained'}


def supplement_worker(root, directory):
    evidence = supplement_api(); require(evidence == read(directory / 'supplement-api.json'), 'Supplement API drift')
    folder = directory / 'supplement-wire'; folder.mkdir(exist_ok=False)
    row = {'function': evidence['function'], 'parameters': evidence['parameters'], 'status': 'failed', 'files': [], 'wire': [],
        'published': False, 'api_sha256': file_sha(directory / 'supplement-api.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire']) < 10, 'Supplement request cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status_code': response.status_code})
        return response
    try:
        import akshare as ak
        with patch.object(requests.sessions.Session, 'request', request): frame = ak.stock_financial_analysis_indicator(**evidence['parameters'])
        path = raw.save(root, 'batch27_dependency_probe', 'ak_roe', directory.name, frame)
        row.update(status='success' if len(frame) else 'empty', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}])
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    save(directory / 'supplement-result.json', row); return row


def validate_supplement(directory):
    api = read(directory / 'supplement-api.json'); require(api == supplement_api(), 'Supplement API changed')
    row = read(directory / 'supplement-result.json')
    require(row['function'] == api['function'] and row['parameters'] == api['parameters'] and row['published'] is False and
        row['api_sha256'] == file_sha(directory / 'supplement-api.json') and row['status'] in ('failed', 'timeout', 'empty', 'success'), 'Supplement result binding differs')
    for item in row['wire']: require(file_sha(item['file']) == item['sha256'], 'Supplement response changed')
    if row['status'] in ('success', 'empty'):
        require(len(row['files']) == 1, 'Supplement raw missing'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
        require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status'] == 'success'), 'Supplement raw changed')
    else: require(not row['files'] and row.get('error'), 'Failed supplement accepted raw/no error')
    return row


def supplement(root, directory):
    save(directory / 'supplement-api.json', supplement_api())
    command = [sys.executable, '-m', 'scripts.review_strategy_batch27', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', 'ak_roe']
    with (directory / 'supplement.stdout').open('x') as out, (directory / 'supplement.stderr').open('x') as err:
        try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
        except subprocess.TimeoutExpired:
            if not (directory / 'supplement-result.json').exists():
                api = read(directory / 'supplement-api.json')
                save(directory / 'supplement-result.json', {'function': api['function'], 'parameters': api['parameters'], 'status': 'timeout',
                    'files': [], 'wire': [], 'published': False, 'api_sha256': file_sha(directory / 'supplement-api.json'), 'error': 'Parent deadline45s; partial unaccepted wire files may remain'})
    validate_supplement(directory)
    return archive(root, 'Batch27 independent Sina historical financial-indicator fallback archived without publication')


def offline(root, directory):
    before = implementation(); binding(root, directory)
    def denied(*a, **k): raise AssertionError('Batch27 offline attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        repeated = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Offline implementation drift')
    require(diagnostic == read(directory / 'diagnostics.json'), 'Offline diagnostics differ')
    save(directory / 'component-offline.json', repeated)
    require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline bytes differ')
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'sha256': file_sha(directory / 'component-offline.json'), 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch27 forbidden-network component/diagnostic recomputation byte-identical')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch27.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch27_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch26_archive.py',
            'tests/unit/test_valuations.py', 'tests/integration/test_valuation_recovery.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_probes(directory); validate_supplement(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Frozen implementation exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json'); require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation(), 'Checked implementation differs')
    require([r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']) and CORE <= checked['evidence_sha256'].keys(), 'Checks/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence differs')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Frozen code differs')
    for row in checked['commands']: require(file_sha(row['log']) == row['sha256'], 'Checked log differs')
    off = read(directory / 'offline-verification.json')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence differs')
    binding(root, directory); validate_probes(directory); validate_supplement(directory)
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Component/diagnostic drift')
    protection = protect(root, directory); progress = archive(root, 'Batch27 three reviews/components/offline/probes accepted; no original trading reconstruction')
    result = {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'), 'progress': progress, 'snapshot': SNAPSHOT,
        'probes': read(directory / 'probe-results.json'), 'supplement': read(directory / 'supplement-result.json'), 'offline': off, 'protection': protection, 'checks': checked,
        'not_a_backtest': True, 'original_strategy_complete': False}
    save(Path('docs/handoff/2026-10-06-batch27-verification.json'), result)
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'prepare', 'study', 'probe', 'supplement', 'worker', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch27/20261006-value-ma-turtle')
    parser.add_argument('--endpoint', choices=[*QUERIES, 'ak_roe']); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
