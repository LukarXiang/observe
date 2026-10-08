"""Freeze RSI/momentum/chip sources and study original components without trades."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
import hashlib
from heapq import nlargest
import importlib.util
import inspect
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import tokenize
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
import talib

from observe.data import raw
from observe.data.store import Store
from observe.runs import environment, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch20 import backend
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.run_strategy_batch19 import reference_ema
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-211958-89b5'
SOURCES = ('2022年度精选策略/87.RSI衍生指标择时轮动A股ETF.txt', '2024年度精选策略1/49.年化62%的动量策略.txt',
    '2024年度精选策略1/77.【复现】筹码分布因子.txt', '2024年度精选策略1/90.【复现】凸显度因子.txt',
    '2022年度精选策略/33.ETF单均线跟踪轮动.txt')
QUERIES = {'etf2016': ('fund_etf_category_ths', {'symbol': 'ETF', 'date': '20160608'}),
    'etf2026': ('fund_etf_category_ths', {'symbol': 'ETF', 'date': '20260929'}),
    'chip000001': ('stock_cyq_em', {'symbol': '000001', 'adjust': ''})}
REUSED = Path('data/staging/strategies-batch20/20261005-macd-patterns/probes/etf510300/result.json')
PRIOR_RECEIPT = Path('docs/handoff/2026-10-05-batch20-verification.json')
REFERENCE = ('polars_ta/tdx/_chip.py', 'polars_ta/tdx/pattern.py', 'tests/tdx/chip_test.py')
FILES = {'scripts/review_strategy_batch21.py', 'scripts/review_strategy_batch20.py', 'scripts/run_strategy_batch19.py',
    'src/observe/strategy_catalog.py', 'tests/unit/test_batch21_research.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'upstream-binding.json', 'existing-apis.json', 'probe-results.json', 'input-analysis.json',
    'research-inputs.parquet', 'component-research.json', 'stop-override-verification.json', 'diagnostics.json', 'offline-verification.json', 'component-offline.json'}


def start(root, directory):
    checkpoint(root, directory, 'Batch21 source/component/dependency research checkpoint')
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Published snapshot differs')
    previous_file = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    previous = {r['path']: r for r in read(previous_file)}; target = directory / 'source-reviews'; target.mkdir(); records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = target / f'{strategy_id(name)}.source'
        require(previous[name]['review_status'] != '人工审查完成' and file_sha(source) == previous[name]['bytes_sha256'], 'Source changed/already reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(source)
        records.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    searches = []
    for argv in (['rg', '--files', '--hidden', 'repo', 'data', '-g', 'turnover_coefficient_ops.py', '-g', 'cyq_ops.py',
            '-g', 'qlib_workflow.py', '-g', 'factor_analyze.py', '-g', '*chip*dataset.pkl', '-g', 'turnovercoeff_dataset.pkl', '-g', 'trained_model.pkl', '-g', 'pred.pkl'],
        ['rg', '-l', r'^def calc_sigma|^def calc_weight|^class CYQK_C|^class TurnCoeffChips|^class Chips|^class ARC\(', 'repo', '-g', '*.py', '-g', '*.txt']):
        process = subprocess.run(argv, capture_output=True, text=True); require(process.returncode in (0, 1), process.stderr)
        searches.append({'argv': argv, 'returncode': process.returncode, 'matches': process.stdout.splitlines()})
    reference_dir = directory / 'reference'; reference_dir.mkdir(); references = []
    commit = subprocess.run(['git', '-C', 'repo/polars_ta', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    for name in REFERENCE:
        source = Path('repo/polars_ta') / name; copied = reference_dir / Path(name).name; shutil.copyfile(source, copied)
        references.append({'source': str(source), 'copy': str(copied), 'sha256': file_sha(source), 'commit': commit,
            'use': 'Read-only design comparison; not original scr operator and not executed'})
    upstream_receipt(PRIOR_RECEIPT, REUSED)
    reused = read(REUSED); require(reused['status'] == 'success' and reused['endpoint'] == 'etf510300' and len(reused['files']) == 1, 'Reused ETF evidence incomplete')
    for item in [*reused['files'], *reused['wire_responses']]: require(file_sha(item['file']) == item['sha256'], 'Reused ETF evidence changed')
    import akshare as ak
    apis = []
    for name in sorted({n for n, _ in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'signature': str(inspect.signature(fn))})
    save(directory / 'existing-apis.json', {'version': ak.__version__, 'apis': apis, 'queries': QUERIES})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(target / 'review.json', {'snapshot': SNAPSHOT, 'sources': records, 'catalog': catalog,
        'previous_catalog_file': str(previous_file), 'previous_catalog_sha256': file_sha(previous_file), 'dependency_searches': searches,
        'installed_modules': {n: importlib.util.find_spec(n) is not None for n in ('qlib', 'empyrical', 'torch', 'talib')}, 'references': references,
        'reused_etf_evidence': {'file': str(REUSED), 'sha256': file_sha(REUSED), 'result': reused},
        'strategy_results': [], 'limitations': ['Missing scr modules prevent faithful Qlib workflow recreation', 'Existing chip implementations are different algorithms']})
    return archive(root, 'Batch21 five full reviews and missing Qlib artifacts archived; literal reversal/ETF defects retained')


def upstream_receipt(receipt_file, terminal_file):
    receipt = read(receipt_file)
    require(receipt['status'] == 'ok' and receipt['reviews']['snapshot'] == SNAPSHOT, 'Upstream receipt scope differs')
    require(file_sha(terminal_file) == receipt['checks']['evidence_sha256']['probes/etf510300/result.json'], 'Upstream terminal differs from accepted receipt')
    return {'receipt_file': str(receipt_file), 'receipt_sha256': file_sha(receipt_file), 'terminal_file': str(terminal_file), 'terminal_sha256': file_sha(terminal_file)}


def bind(root, directory):
    row = upstream_receipt(PRIOR_RECEIPT, REUSED)
    reused = read(directory / 'source-reviews/review.json')['reused_etf_evidence']
    require(row['terminal_file'] == reused['file'] and row['terminal_sha256'] == reused['sha256'], 'Initial reused evidence differs')
    save(directory / 'upstream-binding.json', {**row, 'review_sha256': file_sha(directory / 'source-reviews/review.json'),
        'note': 'Added accepted upstream receipt binding; initial review remains immutable'})
    return archive(root, 'Batch21 reused ETF evidence bound to accepted batch20 receipt; initial archive retained')


def bound_upstream(directory):
    row = read(directory / 'upstream-binding.json')
    require(row['receipt_file'] == str(PRIOR_RECEIPT) and row['terminal_file'] == str(REUSED) and
        row['receipt_sha256'] == file_sha(PRIOR_RECEIPT) and row['review_sha256'] == file_sha(directory / 'source-reviews/review.json'), 'Upstream binding changed')
    actual = upstream_receipt(PRIOR_RECEIPT, REUSED)
    require(all(actual[key] == row[key] for key in actual), 'Upstream receipt association differs')
    return row


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(15)
    result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'wire_responses': [], 'published': False}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'HTTP response cap reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15); response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        import akshare as ak
        evidence = read(directory / 'existing-apis.json'); name, parameters = QUERIES[endpoint]; fn = getattr(ak, name)
        bound = next(r for r in evidence['apis'] if r['name'] == name)
        require(ak.__version__ == evidence['version'] and evidence['queries'][endpoint] == [name, parameters] and
            hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == bound['source_sha256'], 'API evidence changed')
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = fn(**parameters)
        path = raw.save(root, 'fund_chip_probe_batch21', endpoint, directory.name, frame)
        result['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame), 'parameters': parameters})
        result['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    require(not (directory / 'probes').exists() and not (directory / 'probe-results.json').exists(), 'Probe archive exists')
    results = []
    for endpoint in QUERIES:
        argv = [sys.executable, '-m', 'scripts.review_strategy_batch21', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=75)
            row = {'endpoint': endpoint, 'returncode': process.returncode, 'stdout': process.stdout[-1200:], 'stderr': process.stderr[-1200:]}
        except subprocess.TimeoutExpired: row = {'endpoint': endpoint, 'status': 'timeout', 'timeout_seconds': 75}
        terminal = directory / 'probes' / endpoint / 'result.json'
        row['result'] = read(terminal) if terminal.exists() else {'endpoint': endpoint, 'status': row.get('status', 'failed'), 'files': [], 'wire_responses': []}
        results.append(row); print(json.dumps(row['result'], ensure_ascii=False), flush=True)
        archive(root, f'Batch21 {endpoint} probe {row["result"]["status"]}; no historical pool promoted')
    save(directory / 'probe-results.json', {'snapshot': SNAPSHOT, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'results': results})
    return {'status': 'archived'}


def frames(directory):
    result = read(directory / 'probe-results.json')
    require(result['snapshot'] == SNAPSHOT and result['published'] is False and result['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe scope/API differs')
    require(len(result['results']) == len(QUERIES) and {r['endpoint'] for r in result['results']} == set(QUERIES), 'Probe endpoint set incomplete')
    output = {}
    for row in result['results']:
        terminal = directory / 'probes' / row['endpoint'] / 'result.json'
        if terminal.exists(): require(read(terminal) == row['result'], 'Terminal probe differs')
        else: require(row.get('status') == 'timeout' or row.get('returncode', 0) != 0, 'Terminal response missing')
        require(row['result']['endpoint'] == row['endpoint'] and len(row['result']['files']) == (1 if row['result']['status'] in ('success', 'empty') else 0), 'Probe component differs')
        for item in row['result']['wire_responses']: require(file_sha(item['file']) == item['sha256'], 'Wire data changed')
        for item in row['result']['files']:
            require(item['parameters'] == QUERIES[row['endpoint']][1] and file_sha(item['file']) == item['sha256'], 'Query/file changed')
            data = pd.read_parquet(item['file']); require(len(data) == item['rows'] and list(data) == item['columns'], 'Shape changed')
            if row['result']['status'] == 'success': output[row['endpoint']] = data
    return output


def etf_sample(frame):
    data = frame.copy(); data['date'] = pd.to_datetime(data.date, errors='raise').dt.date
    data = data[data.date.le(date(2026, 9, 29))].sort_values('date').tail(100)
    require(len(data) == 100 and not data.date.duplicated().any() and data.date.iloc[-1] == date(2026, 9, 29), 'ETF sample unavailable')
    for name in ('open', 'high', 'low', 'close'):
        data[name] = pd.to_numeric(data[name], errors='raise')
        require(np.isfinite(data[name]).all() and data[name].gt(0).all(), 'Invalid ETF price')
    require((data.high >= data[['open', 'close', 'low']].max(axis=1)).all() and (data.low <= data[['open', 'close', 'high']].min(axis=1)).all(), 'Invalid OHLC')
    return data[['date', 'open', 'high', 'low', 'close']].reset_index(drop=True)


def analyze(root, directory):
    bound_upstream(directory)
    downloaded = frames(directory); reused = read(directory / 'source-reviews/review.json')['reused_etf_evidence']
    require(file_sha(reused['file']) == reused['sha256'], 'Reused manifest changed')
    item = reused['result']['files'][0]; require(file_sha(item['file']) == item['sha256'], 'Reused prices changed')
    data = etf_sample(pd.read_parquet(item['file']))
    calendar = Store(root).load_state(Store(root).state(SNAPSHOT), 'calendar'); days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.date
    expected = sorted(d for d in days if d <= date(2026, 9, 29))[-100:]
    require(data.date.tolist() == expected, 'ETF sample missing calendar observation')
    path = directory / 'research-inputs.parquet'; require(not path.exists(), 'Input archive exists'); data.to_parquet(path, index=False)
    profiles = []
    for endpoint, frame in downloaded.items():
        profile = {'endpoint': endpoint, 'rows': len(frame), 'strict_usable': False, 'status': 'sample_only'}
        if endpoint.startswith('etf'):
            requested = date.fromisoformat('2016-06-08' if endpoint == 'etf2016' else '2026-09-29')
            latest = pd.to_datetime(frame['最新-交易日'], errors='coerce').dt.date
            profile.update(requested_date=str(requested), latest_newer_than_query=int((latest > requested).sum()),
                query_date_note='API itself constructs 查询日期 from input; this is not evidence of historical membership/names')
        else:
            dates = pd.to_datetime(frame['日期'], errors='coerce')
            profile.update(first=str(dates.min()), last=str(dates.max()), rows_through_frozen_end=int(dates.dt.date.le(date(2026, 9, 29)).sum()),
                note='API returns only last90; algorithm differs from original scr operators')
        profiles.append(profile)
    comparison = None
    if {'etf2016', 'etf2026'} <= set(downloaded):
        a, b = downloaded['etf2016'], downloaded['etf2026']
        require(not a['基金代码'].duplicated().any() and not b['基金代码'].duplicated().any(), 'Duplicate fund code')
        comparison = {'same_code_set': set(a['基金代码']) == set(b['基金代码']),
            'same_names_by_code': a.set_index('基金代码')['基金名称'].sort_index().equals(b.set_index('基金代码')['基金名称'].sort_index()),
            'historical_net_available_rows': int(pd.to_numeric(a['当前-单位净值'], errors='coerce').notna().sum()),
            'examples': a[a['基金代码'].isin(['510300', '159995', '588000'])].astype(object).where(a.notna(), None).to_dict('records'),
            'strict_usable': False, 'reason': 'Date-specific NAV columns coexist with the same full list/current names and latest fields; historical membership not proved'}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'published': False, 'input_file': str(path), 'input_sha256': file_sha(path),
        'etf': {'rows': 100, 'first': str(data.date.iloc[0]), 'last': str(data.date.iloc[-1]), 'calendar_missing': 0,
            'semantics': 'Reused raw Sina chart OHLC; original dynamic adjustment/events not proved equivalent'},
        'probe_profiles': profiles, 'fund_universe_comparison': comparison, 'missing': sorted(set(QUERIES) - set(downloaded)), 'strategy_results': []})
    return archive(root, 'Batch21 100 frozen ETF bars prepared for signal component; dependency gaps remain explicit')


def selected(directory, number, names, namespace):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    text, _ = read_source(Path(row['source_copy'])); lines = text.splitlines(keepends=True); starts = {}
    for i, line in enumerate(lines):
        if line.startswith('def '): starts[line.split('(', 1)[0][4:].strip()] = i
    require(set(names) <= set(starts), 'Selected functions missing'); functions = []
    # Compile exact reviewed function slices; Python2 printing elsewhere is never translated or executed.
    for name in names:
        first = starts[name]; fragment = ''.join(lines[first:]); depth = 0; entered = False; end = len(lines) - first
        for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
            if token.type == tokenize.INDENT: depth += 1; entered = True
            elif token.type == tokenize.DEDENT:
                depth -= 1
                if entered and depth == 0: end = token.start[0] - 1; break
        node = ast.parse(''.join(lines[first:first + end]))
        require(len(node.body) == 1 and isinstance(node.body[0], ast.FunctionDef) and node.body[0].name == name, 'Unexpected statements in selected slice')
        functions.append(node.body[0])
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, row['source_copy'], 'exec'), namespace)
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()


def reference_rsi(prices, period=6):
    data = list(map(float, prices)); require(period > 1 and np.isfinite(data).all(), 'Invalid RSI inputs')
    output = [np.nan] * len(data)
    if len(data) <= period: return np.array(output)
    changes = [data[i] - data[i - 1] for i in range(1, len(data))]
    gain = sum(max(v, 0.) for v in changes[:period]) / period; loss = sum(max(-v, 0.) for v in changes[:period]) / period
    for i in range(period, len(data)):
        if i > period:
            gain = (gain * (period - 1) + max(changes[i - 1], 0.)) / period
            loss = (loss * (period - 1) + max(-changes[i - 1], 0.)) / period
        output[i] = 100 * gain / (gain + loss) if gain + loss else 0.
    return np.array(output)


def compute(directory):
    bound_upstream(directory)
    analysis = read(directory / 'input-analysis.json'); require(file_sha(analysis['input_file']) == analysis['input_sha256'], 'Research input changed')
    data = pd.read_parquet(analysis['input_file']); settings = backend()
    require(settings['wrapper'] == '0.8.1' and settings['compatibility'] == 0 and settings['ema_unstable'] == 0 and talib.get_unstable_period('RSI') == 0, 'Backend differs')
    params = {'nRSI': 6, 'ma': 12, 'buyThreshold': 40, 'sellThreshold': 55, 'bian1': .8, 'bian2': .6}
    namespace = {'np': np, 'talib': talib, 'g': SimpleNamespace(**params), 'attribute_history': lambda *args, **kwargs: {c: data[c].to_numpy() for c in ('open', 'high', 'low', 'close')}}
    sha = selected(directory, 0, ('buyOrSellCheck',), namespace)
    jump = np.log(data.close.to_numpy() / data.high.to_numpy()); rsi = talib.RSI(jump, 6); oracle_rsi = reference_rsi(jump)
    ema = talib.EMA(data.close.to_numpy(), 12); oracle_ema = np.array(reference_ema(data.close.tolist(), 12))
    require(np.allclose(rsi, oracle_rsi, atol=1e-10, rtol=0, equal_nan=True) and np.allclose(ema, oracle_ema, atol=1e-10, rtol=0, equal_nan=True), 'Independent indicator differs')
    width = float(data.high.iloc[-1]) - float(data.low.iloc[-1]); require(width > 0, 'Source body/range undefined on flat last bar')
    a = abs((float(data.open.iloc[-1]) - float(data.close.iloc[-1])) / width)
    expected = 1 if oracle_rsi[-1] <= 40 and (data.close.iloc[-1] > oracle_ema[-1] or a <= .8) else -1 if oracle_rsi[-1] >= 55 and (data.close.iloc[-1] <= oracle_ema[-1] or a <= .6) else 0
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={}))
    actual = namespace['buyOrSellCheck']('510300.XSHG', context); require(actual == expected, 'Source timing signal differs')
    stops = []
    for ratio in (.89, 1.11):
        context.portfolio.positions = {'510300.XSHG': SimpleNamespace(avg_cost=float(data.close.iloc[-1]) / ratio)}
        state = namespace['buyOrSellCheck']('510300.XSHG', context); require(state == -1, 'Source strict stop override differs')
        stops.append({'close_cost_ratio': ratio, 'state': state})
    return {'status': 'component_recomputed_with_limits', 'snapshot': SNAPSHOT, 'backend': {**settings, 'rsi_unstable': talib.get_unstable_period('RSI')},
        'source_sha256': read(directory / 'source-reviews/review.json')['sources'][0]['source_sha256'], 'selected_ast_sha256': sha, 'parameters': params,
        'input_sha256': analysis['input_sha256'], 'rows': len(data), 'finite_rsi': int(np.isfinite(rsi).sum()), 'finite_ema': int(np.isfinite(ema).sum()),
        'max_rsi_absolute_difference': float(np.nanmax(np.abs(rsi - oracle_rsi))), 'max_ema_absolute_difference': float(np.nanmax(np.abs(ema - oracle_ema))),
        'last_rsi': float(rsi[-1]), 'last_ema': float(ema[-1]), 'body_range_ratio': a, 'flat_position_state': actual, 'stop_overrides': stops,
        'not_a_backtest': True, 'not_a_strategy_reproduction': True,
        'limits': ['Single raw ETF sample is not historical universe or original adjusted prices', 'No fund selection, fills, NAV or cost scenario computed', 'Source economic defects retained']}


def study(root, directory):
    result = compute(directory); save(directory / 'component-research.json', result)
    archive(root, 'Batch21 original LN(close/high) RSI6/EMA12 component checked; not ETF rotation backtest')
    return result


def stops(root, directory):
    research = read(directory / 'component-research.json')
    inputs = {'open': 10.2, 'close': 10., 'high': 10.4, 'low': 9.8}
    namespace = {'talib': talib, 'g': SimpleNamespace(**research['parameters']),
        'attribute_history': lambda *args, **kwargs: {k: np.full(100, v) for k, v in inputs.items()}}
    sha = selected(directory, 0, ('buyOrSellCheck',), namespace)
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={}))
    def forbidden(*args, **kwargs): raise AssertionError('Synthetic diagnostic network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        base = namespace['buyOrSellCheck']('510300.XSHG', context); require(base == 1, 'Synthetic baseline is not a buy')
        results = []
        for ratio in (.89, 1.11):
            context.portfolio.positions = {'510300.XSHG': SimpleNamespace(avg_cost=10. / ratio)}
            state = namespace['buyOrSellCheck']('510300.XSHG', context); require(state == -1, 'Cost exit failed to override buy')
            results.append({'close_cost_ratio': ratio, 'base_state': base, 'state': state})
    require(sha == research['selected_ast_sha256'], 'Synthetic diagnostic function differs')
    result = {'status': 'match', 'source_sha256': research['source_sha256'], 'selected_ast_sha256': sha,
        'research_sha256': file_sha(directory / 'component-research.json'), 'input_kind': 'synthetic_constant_100_rows', 'input_prices': inputs,
        'parameters': research['parameters'], 'cases': results, 'network': 'socket connect/connect_ex and requests forbidden', 'not_a_backtest': True}
    save(directory / 'stop-override-verification.json', result)
    archive(root, 'Batch21 source cost exits verified to override synthetic buy; initial real-sample study retained')
    return result


def diagnose(root, directory):
    namespace = {'nlargest': nlargest, 'g': SimpleNamespace(security1=[], security2=[], day_count=91, stock_price=5),
        'datetime': SimpleNamespace(timedelta=timedelta), 'log': SimpleNamespace(info=lambda *args: None),
        'order_target': lambda *args: None, 'order_value': lambda *args: None,
        'get_price': lambda *args, **kwargs: pd.DataFrame({'close': np.linspace(10, 12, 91)})}
    sha = selected(directory, 1, ('min_dict', 'market_open', 'get_momentum'), namespace)
    picked = namespace['min_dict']({'up': .1, 'flat': 0., 'down': -.1}, 2); require(picked == ['down', 'flat'], 'Original low-return ranking differs')
    context = SimpleNamespace(current_dt=datetime(2026, 9, 29, 9, 30), portfolio=SimpleNamespace(available_cash=100.))
    errors = []
    for fn, args, expected_error in (('market_open', (context,), ZeroDivisionError), ('get_momentum', (context, ['000001.XSHE']), KeyError)):
        try: namespace[fn](*args)
        except expected_error as exc: errors.append({'function': fn, 'error': f'{type(exc).__name__}: {exc}'})
        else: raise AssertionError('Original compatibility/empty-newlist error absent')
    stv = {}; stv_sha = selected(directory, 3, ('get_stv_feature',), stv); expression = stv['get_stv_feature']()
    require(expression == 'If(Abs($close/Ref($close,1)-1)>=0.1,Abs($close/Ref($close,1)-1)*100,$turnover_rate)', 'Literal STV constants differ')
    calls = []; g = SimpleNamespace(hs='300', zz='gem', sz='50', lag=13, ETF300='300', ETF500='gem', ETF50='50', ETFrili='cash', hour=14, minute=25)
    signal_ns = {'g': g, 'getStockPrice': lambda stock, count: {'300': (100., 102., 100.), 'gem': (100., 105., 100.), '50': (100., 110., 120.)}[stock],
        'sell_the_stocks': lambda *args: calls.append('sell'), 'buy_the_stocks': lambda *args: calls.append('buy')}
    signal_sha = selected(directory, 4, ('get_signal', 'handle_data'), signal_ns)
    context = SimpleNamespace(current_dt=datetime(2026, 9, 29, 14, 25), portfolio=SimpleNamespace(positions={k: SimpleNamespace(total_amount=0) for k in ('300', 'gem', '50')}))
    signal = signal_ns['get_signal'](context); require(signal == 'ETF50', 'Source wrong-market check differs')
    signal_ns['handle_data'](context, None); require(calls == [], 'Missing ETF50 dispatch diagnostic differs')
    result = {'status': 'original_defects_diagnosed', 'selected_ast_sha256': [sha, stv_sha, signal_sha], 'low_return_ranking': picked, 'errors': errors,
        'stv_expression': expression, 'stv_comment_multiplier': 1000, 'stv_code_multiplier': 100,
        'etf33': {'signal': signal, 'index50_above_mean': False, 'gem_above_mean': True, 'dispatch_calls': calls},
        'external_messages_orders': 0, 'not_a_backtest': True}
    save(directory / 'diagnostics.json', result); archive(root, 'Batch21 low-return selection, zero-newlist, STV constants and ETF33 branch diagnostics archived')
    return result


def offline(root, directory):
    original = directory / 'component-research.json'; before = file_sha(original)
    def forbidden(*args, **kwargs): raise AssertionError('Research network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden), patch.object(requests.sessions.Session, 'request', forbidden): repeated = compute(directory)
    require(repeated == read(original) and file_sha(original) == before, 'Offline component differs/original changed')
    target = directory / 'component-offline.json'; save(target, repeated)
    result = {'status': 'match', 'differences': 0, 'original_unchanged': True, 'original_sha256': before, 'recomputed_sha256': file_sha(target),
        'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True}
    save(directory / 'offline-verification.json', result); archive(root, 'Batch21 frozen indicator component offline match/0 differences')
    return result


def checks(root, directory):
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
        'scripts/verify_financial_import.py', 'scripts/review_strategy_batch21.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch21_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'], ['git', 'diff', '--check']]
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
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked artifacts missing')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    reviews = read(directory / 'source-reviews/review.json')
    upstream = bound_upstream(directory)
    require(reviews['snapshot'] == SNAPSHOT and [r['source_path'] for r in reviews['sources']] == [str(Path('repo/量化策略源代码') / name) for name in SOURCES], 'Source set differs')
    require(file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, reviews['sources'], strict=True):
        require(file_sha(row['source_path']) == row['source_sha256'] == row['source_copy_sha256'] == file_sha(row['source_copy']) and REVIEWS[name] == row['review'], 'Source/rules changed')
    for row in reviews['references']: require(file_sha(row['source']) == row['sha256'] == file_sha(row['copy']), 'Reference source changed')
    reused = reviews['reused_etf_evidence']; require(file_sha(reused['file']) == reused['sha256'] and read(reused['file']) == reused['result'], 'Reused ETF manifest changed')
    for item in [*reused['result']['files'], *reused['result']['wire_responses']]: require(file_sha(item['file']) == item['sha256'], 'Reused ETF data changed')
    frames(directory); require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    analysis = read(directory / 'input-analysis.json'); research = read(directory / 'component-research.json'); repeated = read(directory / 'component-offline.json'); repeat = read(directory / 'offline-verification.json')
    require(analysis['snapshot'] == research['snapshot'] == SNAPSHOT and analysis['input_sha256'] == research['input_sha256'] == file_sha(analysis['input_file']), 'Research input differs')
    pd.testing.assert_frame_equal(pd.read_parquet(analysis['input_file']), etf_sample(pd.read_parquet(reused['result']['files'][0]['file'])))
    require(research == repeated and research['source_sha256'] == reviews['sources'][0]['source_sha256'] and research['backend'] == {**backend(), 'rsi_unstable': talib.get_unstable_period('RSI')}, 'Research/source/backend differs')
    require(repeat['status'] == 'match' and repeat['differences'] == 0 and repeat['original_unchanged'] and research['not_a_backtest'] and
        repeat['original_sha256'] == file_sha(directory / 'component-research.json') and repeat['recomputed_sha256'] == file_sha(directory / 'component-offline.json'), 'Repeat evidence differs')
    stop_check = read(directory / 'stop-override-verification.json')
    require(stop_check['status'] == 'match' and stop_check['research_sha256'] == file_sha(directory / 'component-research.json') and
        stop_check['source_sha256'] == research['source_sha256'] and stop_check['selected_ast_sha256'] == research['selected_ast_sha256'] and
        stop_check['cases'] == [{'close_cost_ratio': ratio, 'base_state': 1, 'state': -1} for ratio in (.89, 1.11)], 'Stop override evidence differs')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'upstream_binding': upstream, 'analysis': analysis, 'research': research, 'offline': repeat,
        'stop_override_check': stop_check, 'diagnostics': read(directory / 'diagnostics.json'), 'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch21 five reviews, original RSI component and dependency probes verified; strategy library continues')}
    target = Path('docs/handoff/2026-10-05-batch21-verification.json'); save(target, result)
    return {'status': 'ok', 'output': str(target), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'bind', 'worker', 'probe', 'analyze', 'study', 'stops', 'diagnose', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--endpoint', choices=list(QUERIES))
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch21/20261005-rsi-momentum-chips'))
    args = parser.parse_args()
    if args.action == 'worker':
        if args.endpoint is None: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
