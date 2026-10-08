"""Archive macro rules and numeric operands without claiming historical trades."""
import argparse
import ast
from contextlib import redirect_stdout
from decimal import Decimal
from fractions import Fraction
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
from scipy.optimize import curve_fit

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch49 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch49/20261007-query-classification')
RECEIPT = Path('docs/handoff/2026-10-07-batch49-verification.json')
SOURCES = ('2022年度精选策略/6.择时，还是宏观数据靠谱（宏观择时集合）.txt',
    '2022年度精选策略/57.宏观指标择时检测.txt',
    '2024年度精选策略1/21.一种宏观数据的中长线策略，年化15%，最大回撤9%.txt')
SOURCE_SHA = ('da7304f60faa5d11a2ddc02f9796b3dcd943886f088b6944f2ba917a46fc6d98',
    '4165bfdabda3863180f3c2bc354dd7fe986f8ea4724eeaf2fb62311edf745aad',
    '5d55e30ef811ceda9ef64f13f874d38ae84204dcbf2fb248274188563d08c88b')
QUERIES = {'fixed_investment': ('macro_china_gdzctz', {}), 'cpi': ('macro_china_cpi', {}), 'ppi': ('macro_china_ppi', {})}
MACRO_INPUTS = {'pmi': ('macro_pmi', '6121f0fb8f1dba5672b75c24d5d56346a401e0a1b330d9935a5d788430b7d895'),
    'money': ('macro_money', '3ea84b196fbb4959a4d038db1c44750b58f35c61d1f56c37edae988adf2193c4')}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch50.py', 'tests/unit/test_batch50_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'source-reviews/review.json',
    'existing-apis.json', 'input-analysis.json', 'pmi-input.parquet', 'money-input.parquet', 'index-input.parquet',
    'component-research.json', 'probe-results.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
CORE |= {'initial-study-failure.json', 'review_strategy_batch50.initial.py', 'rolling-boundary-diagnosis.json',
    'initial-test-failure/result.json', 'initial-test-failure/correction.json'}
CONTINUOUS = 'get_position_from_continus_increase'
ROLLING = 'get_rolling_positon'


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch49 changed')
    probe_path = Path('data/staging/strategies-batch12/20261005-daily-candidates/probe-results.json')
    probes = {r['endpoint']: r['result'] for r in read(probe_path)['results']}
    files = {}
    for name, (endpoint, sha) in MACRO_INPUTS.items():
        row = probes[endpoint]; path = Path(row['raw_file'])
        require(row['status'] == 'success' and row['strict_usable'] is False and row['published'] is False and
            row['raw_sha256'] == file_sha(path) == sha, 'Old macro operand changed')
        files[f'{name}-input.parquet'] = {'file': str(path), 'sha256': sha, 'rows': row['rows'], 'strict_usable': False}
    operand = Path('data/staging/strategies-batch47/20261007-candidate-momentum/index-input.parquet')
    old = read('docs/handoff/2026-10-07-batch47-verification.json')
    require(file_sha(operand) == old['checks']['evidence_sha256']['index-input.parquet'], 'Accepted index changed')
    files['index-input.parquet'] = {'file': str(operand), 'sha256': file_sha(operand), 'rows': 5280, 'intraday_usable': False}
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'old_macro_probe_sha256': file_sha(probe_path), 'files': files, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch50 macro source review started; no trade experiment')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, (name, params) in QUERIES.items():
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': name, 'parameters': params, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        source = Path('repo/量化策略源代码') / name; text, encoding = read_source(source)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(source, copied)
        tree = ast.parse(text)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True,
            'function_ast': {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}})
    ref = Path('repo/akshare/akshare/economic/macro_china.py'); folder = directory / 'references'; folder.mkdir()
    copied = folder / ref.name; shutil.copyfile(ref, copied)
    refs = [{'file': str(ref), 'copy': str(copied), 'sha256': file_sha(ref),
        'commit': subprocess.run(['git', '-C', 'repo/akshare', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
        'use': 'Existing PMI aggregates/investment/CPI/PPI APIs; no proof of original subfields or release versions'}]
    save(directory / 'existing-apis.json', api_evidence())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs,
        'catalog': catalog, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch50 three full macro sources and exact parameter AST frozen')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True, 'Source scope changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True):
        require(row['review'] == REVIEWS[name] and row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy'])); tree = ast.parse(text)
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and row['function_ast'] ==
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}, 'Full function evidence changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, names, namespace):
    validate_sources(directory); row = read(directory / 'source-reviews/review.json')['sources'][number]
    tree = ast.parse(read_source(Path(row['source_copy']))[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Selected functions missing')
    require(all(ast.dump(n, include_attributes=False) == row['function_ast'][n.name] for n in nodes), 'Function AST changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-macro-component>', 'exec'), namespace)


def parse_monthly(frame, columns):
    dates = pd.to_datetime(frame['月份'], format='%Y年%m月份', errors='raise').dt.to_period('M').astype(str)
    require(not dates.duplicated().any(), 'Duplicate macro month')
    values = frame[columns].apply(pd.to_numeric, errors='raise'); values.index = dates
    values = values.sort_index()
    require(np.isfinite(values.to_numpy()).all() and values.index.tolist() == pd.period_range(values.index[0], values.index[-1], freq='M').astype(str).tolist(), 'Macro values/gaps invalid')
    return values


def prepare(root, directory):
    bound = binding(root, directory); profiles = {}
    for name, row in bound['files'].items():
        copied = directory / name; require(not copied.exists(), 'Operand exists'); shutil.copyfile(row['file'], copied)
        frame = pd.read_parquet(copied); require(len(frame) == row['rows'], 'Operand row count differs')
        profiles[name] = {'sha256': file_sha(copied), 'rows': len(frame), 'columns': list(frame)}
    pmi = parse_monthly(pd.read_parquet(directory / 'pmi-input.parquet'), ['制造业-指数'])
    money = parse_monthly(pd.read_parquet(directory / 'money-input.parquet'), ['货币(M1)-同比增长', '货币和准货币(M2)-同比增长'])
    state = Store(root).state(SNAPSHOT); counts = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'fund_actions', 'fund_coverage', 'adj_factors', 'adj_coverage'):
        if table not in state['tables']: counts[table] = 0; continue
        frame = Store(root).load_state(state, table, filters=[('instrument', 'in', ['510300.SH', '511010.SH'])])
        counts[table] = len(frame)
    save(directory / 'input-analysis.json', {'profiles': profiles, 'pmi_months': [pmi.index[0], pmi.index[-1]],
        'money_months': [money.index[0], money.index[-1]], 'two_fund_rows': counts, 'not_a_backtest': True, 'strict_usable': False,
        'limits': ['Final provider monthly values have no historical release/version proof; aggregate PMI lacks production/order/inventory/import subindices',
            'No complete macro schema, original Tushare CPI/PPI versions or original intraday ETF execution; index cannot trade',
            'Index EOD month-end arithmetic is not original include_now 14:55 monthly/daily bars']})
    return archive(root, 'Batch50 old 225/224 macro operands frozen; original release and fund inputs missing')


def continuous_reference(values, n, delay, how):
    require(n >= 1 and delay >= 0 and how in ('up', 'down', 'dowm'), 'Invalid signal parameters')
    data = list(np.asarray(values, dtype=np.float32)); out = []
    for k in range(n, len(data) - delay):
        good = all(data[j] < data[j+1] if how == 'up' else data[j] > data[j+1] for j in range(k-n, k))
        out.append((k + delay, int(good)))
    return out


def compute(directory):
    validate_sources(directory)
    for name, row in read(directory / 'input-binding.json')['files'].items(): require(file_sha(directory / name) == row['sha256'], 'Operand changed')
    pmi = parse_monthly(pd.read_parquet(directory / 'pmi-input.parquet'), ['制造业-指数']).iloc[:, 0]
    money = parse_monthly(pd.read_parquet(directory / 'money-input.parquet'), ['货币(M1)-同比增长', '货币和准货币(M2)-同比增长'])
    spread = money.iloc[:, 0] - money.iloc[:, 1]; ns = {'np': np, 'pd': pd}
    selected(directory, 0, [CONTINUOUS], ns); rows = []
    for number, width, series in ((0, 18, pmi), (1, 24, pmi), (1, 24, spread)):
        selected(directory, number, [CONTINUOUS], ns)
        for how, delay in (('up', 0), ('dowm', 1)):
            for k in range(width - 1, len(series)):
                part = series.iloc[k-width+1:k+1]; actual = ns[CONTINUOUS](part, 2, delay=delay, how=how)
                expected = continuous_reference(part, 2, delay, how)
                require(actual.index.tolist() == [part.index[i] for i, _ in expected] and actual.iloc[:, 0].tolist() == [v for _, v in expected], 'Continuous formula differs')
                rows.append({'source': number, 'operand': 'pmi' if series is pmi else 'm1_m2_yoy', 'month': series.index[k],
                    'window': width, 'how': how, 'delay': delay, 'value': int(actual.iloc[-1, 0])})
    selected(directory, 1, [ROLLING], ns); rolling = []; boundaries = []
    for series in (pmi, spread):
        for width in (1, 2, 3):
            for how, delay in (('up', 0), ('dowm', 1)):
                actual = ns[ROLLING](series, width, delay=delay, how=how)
                expected = []
                for k in range(len(series) - delay):
                    a = math.fsum(series.iloc[k-width+1:k+1]) / width if k >= width-1 else math.nan
                    b = math.fsum(series.iloc[k-width:k]) / width if k >= width else math.nan
                    expected.append(float(a > b if how == 'up' else a < b))
                values = actual.iloc[:, 0].tolist(); require(len(values) == len(expected), 'Rolling shape differs')
                for k, (value, stable) in enumerate(zip(values, expected, strict=True)):
                    if value == stable: continue
                    require(k >= width, 'Invalid warmup difference')
                    left = series.iloc[k-width+1:k+1].tolist(); right = series.iloc[k-width:k].tolist()
                    a = math.fsum(left)/width; b = math.fsum(right)/width; means = series.rolling(width).mean()
                    require(abs(a-b) < 1e-12 and abs(float(means.iloc[k]-means.iloc[k-1])) < 1e-12, 'Non-boundary formula difference')
                    binary = (sum(map(Fraction.from_float, left), Fraction()) - sum(map(Fraction.from_float, right), Fraction()))/width
                    decimal = (sum((Fraction(str(v)) for v in left), Fraction()) - sum((Fraction(str(v)) for v in right), Fraction()))/width
                    boundaries.append({'operand': 'pmi' if series is pmi else 'm1_m2_yoy', 'n': width, 'how': how, 'delay': delay,
                        'input_month': series.index[k], 'output_month': series.index[k+delay], 'left': left, 'right': right, 'original': value, 'stable': stable,
                        'pandas_means': [float(means.iloc[k]), float(means.iloc[k-1])], 'stable_means': [a, b],
                        'exact_binary_difference': str(binary), 'decimal_repr_difference': str(decimal), 'not_a_backtest': True})
                rolling.append({'operand': 'pmi' if series is pmi else 'm1_m2_yoy', 'n': width, 'delay': delay, 'how': how,
                    'values': values, 'independent_stable_values': expected})
    require(boundaries == read(directory / 'rolling-boundary-diagnosis.json')['differences'], 'Diagnosed boundaries changed')
    exact = {'np': np, 'pd': pd, 'Decimal': Decimal}
    selected(directory, 2, ['func', 'percentile', 'change_to_yeak_k', 'year_move_average'], exact)
    curves = []; x = np.arange(13); u = [k-6 for k in range(13)]; u2 = math.fsum(t*t for t in u)/13
    for k in range(12, len(pmi)):
        values = pmi.iloc[k-12:k+1].to_numpy(); params = curve_fit(exact['func'], x, values)[0]
        actual = float(24*params[0]+params[1])
        a = math.fsum((t*t-u2)*v for t, v in zip(u, values, strict=True))/math.fsum((t*t-u2)**2 for t in u)
        b = math.fsum(t*v for t, v in zip(u, values, strict=True))/math.fsum(t*t for t in u)
        reference = 12*a+b
        require(abs(actual-reference) < 1e-5, 'Quadratic slope differs')
        rank = exact['percentile'](values, values[-1]); ref_rank = sum(v < values[-1] for v in values)*100/12
        require(rank == ref_rank, 'Strict percentile differs')
        curves.append({'month': pmi.index[k], 'slope': actual, 'reference': reference, 'positive_boundary_difference': bool((actual>0) != (reference>0)),
            'pmi_gate': bool(actual>0 and values[-1]>=50 and values[0]<=values[-1]), 'strict_percentile': rank})
    index = pd.read_parquet(directory / 'index-input.parquet'); index['date'] = pd.to_datetime(index.date)
    monthly = index.groupby(index.date.dt.to_period('M'), sort=True).tail(1)[['date', 'close']].reset_index(drop=True)
    annual = []
    for k in range(108, len(monthly)):
        part = monthly.iloc[k-108:k+1].copy(); exact['get_bars'] = lambda *a, _part=part, **kw: _part.copy()
        points = exact['change_to_yeak_k'](); ref = {str(y): float(g.iloc[-1]['close']) for y, g in part.groupby(part.date.dt.year, sort=True)}
        require(points == ref, 'Annual conversion differs')
        actual = exact['year_move_average'](points); expected = float(sum((Decimal(str(v)) for v in ref.values()), Decimal('0'))/Decimal('10'))
        require(actual == expected, 'Annual fixed denominator differs')
        annual.append({'month': str(part.iloc[-1].date.to_period('M')), 'years': len(points), 'fixed_divisor': 10, 'value': actual})
    diagnostics = []
    selected(directory, 0, [ROLLING, 'get_float'], ns)
    for fn, args, error in ((ROLLING, (pmi, 3), AttributeError), ('get_float', (pmi,), NameError)):
        try: ns[fn](*args)
        except error as exc: diagnostics.append({'source': 0, 'function': fn, 'error': type(exc).__name__, 'message': str(exc)})
        else: raise AssertionError('Expected original compatibility failure missing')
    selected(directory, 1, ['get_float'], ns)
    try: ns['get_float'](pmi)
    except NameError as exc: diagnostics.append({'source': 1, 'function': 'get_float', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise AssertionError('Expected missing float32 injection')
    try: float(np.array(pd.DataFrame({'close': [100.]}))[0])
    except TypeError as exc: diagnostics.append({'source': 2, 'function': 'array_to_float', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise AssertionError('Expected NumPy array conversion failure')
    intent = []
    for number, states in ((0, (1, 0, -1)), (1, (1, .5, 0))):
        calls = []; observed = []
        env = {'datetime': __import__('datetime'), 'g': SimpleNamespace(stocks='000300.XSHG'),
            'log': SimpleNamespace(info=lambda *a: None), 'print': lambda *a: None,
            'order_value': lambda *a: calls.append(['order_value', *a]), 'order_target': lambda *a: calls.append(['order_target', *a])}
        def price(*a, **kw):
            observed.append(kw)
            return pd.DataFrame({'close': [100.] * 20 + [50.]})
        env['get_price'] = price; selected(directory, number, ['market_open'], env)
        context = SimpleNamespace(previous_date=__import__('datetime').date(2020, 1, 1),
            portfolio=SimpleNamespace(available_cash=100., total_value=1000.))
        for state in states:
            calls.clear(); observed.clear(); env['g'].position = state; env['market_open'](context)
            expected = ['order_value', '000300.XSHG', 100.] if state == 1 else (
                ['order_value', '000300.XSHG', 500.] if state == states[1] else ['order_target', '000300.XSHG', 0])
            require(calls == [expected] and len(observed) == 1 and observed[0]['count'] == 21, 'Original order intent differs')
            intent.append({'source': number, 'state': state, 'mock_orders': list(calls), 'unapplied_20d_drop': -.5})
    calls = []; exact['g'] = SimpleNamespace(times=0, order_amount=.04, stock_security='510300.XSHG')
    exact['order_value'] = lambda *a: calls.append(['order_value', *a]); selected(directory, 2, ['order_func'], exact)
    for times, available, expected in ((0, 1000., 40.), (4, 500., 500.), (8, 50., None)):
        calls.clear(); exact['g'].times = times; exact['order_func'](1000., available, 1.)
        require(calls == ([] if expected is None else [['order_value', '510300.XSHG', expected]]), 'Original exponential allocation differs')
        intent.append({'source': 2, 'times': times, 'available_cash': available, 'mock_orders': list(calls)})
    return {'not_a_backtest': True, 'strict_usable': False, 'platform_equivalent': False, 'source_sha256': SOURCE_SHA,
        'continuous': rows, 'rolling': rolling, 'rolling_boundaries': boundaries, 'quadratic': curves, 'annual': annual,
        'diagnostics': diagnostics, 'order_intents': intent,
        'limits': ['All macro numeric values are provider-final operands, not historically available signals',
            'Percentile on aggregate PMI verifies arithmetic only; missing original subindices not substituted',
            'Synthetic get_bars supplies verified EOD monthly operands solely for annual arithmetic']}


def study(root, directory):
    binding(root, directory); result = compute(directory); save(directory / 'component-research.json', result)
    return archive(root, 'Batch50 exact macro component arithmetic and compatibility failures archived; no trades')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    name, params = QUERIES[endpoint]; row = {'endpoint': endpoint, 'function': name, 'parameters': params,
        'status': 'failed', 'files': [], 'wire_responses': [], 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire_responses']) < 8, 'Request cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(row["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API changed')
        import akshare as ak
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, name)(**params)
        target = Path(root) / 'raw/macro_probe_batch50' / endpoint / f'{directory.name}.parquet'; require(not target.exists(), 'Raw exists')
        path = raw.save(root, 'macro_probe_batch50', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API evidence changed')
    rows = read(directory / 'probe-results.json')['results']; require([r['endpoint'] for r in rows] == list(QUERIES), 'Probe scope changed')
    for row in rows:
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['published'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['function'] == QUERIES[row['endpoint']][0] and
            row['parameters'] == QUERIES[row['endpoint']][1], 'Probe binding changed')
        for item in row['wire_responses'] + row['files']: require(file_sha(item['file']) == item['sha256'], 'Probe input changed')
        require(row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Unknown probe state')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status'] == 'success'), 'Probe raw changed')
        else: require(not row['files'] and row.get('error'), 'Failed probe raw/error differs')
    return rows


def probe(root, directory):
    rows = []; logs = directory / 'logs'; logs.mkdir()
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch50', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline 45s', 'published': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch50 existing fixed-investment/CPI/PPI supplement attempts frozen')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch50 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); catalog = offline_catalog(root, directory)
    save(directory / 'component-offline.json', result)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline component changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    return archive(root, 'Batch50 forbidden-network macro component/catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch50.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch50_research.py',
         'tests/unit/test_batch25_research.py', 'tests/unit/test_catalog_macro.py', 'tests/unit/test_catalog_research_queries.py',
         'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline proof changed')
    validate_probes(directory); require(json.loads(json.dumps(compute(directory))) == read(directory / 'component-research.json'), 'Component changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch50 macro source research accepted; original strategies blocked by missing historical inputs')
    save(Path('docs/handoff/2026-10-07-batch50-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection,
        'progress': progress, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch50/20261007-macro-rules')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
