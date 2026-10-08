"""Archive original industry factors and risk-premium/model dependency evidence."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import timedelta
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
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data import raw
from observe.data.store import Store
from observe.runs import environment, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-211958-89b5'
SOURCES = ('2021年度精选策略/33.【研报复现】基于风险溢价的沪深300择时-目前依然有效.txt',
    '2022年度精选策略/20.行业轮动的黄金律：日内动量与隔夜反转.txt',
    '2023年度精选策略/71.lightGBM.txt', '2024年度精选策略2/20.【复现】因子择时？？？.txt')
QUERIES = {'sw2016': ('index_analysis_daily_sw', {'symbol': '一级行业', 'start_date': '20160608', 'end_date': '20160608'}),
    'sw2026': ('index_analysis_daily_sw', {'symbol': '一级行业', 'start_date': '20260929', 'end_date': '20260929'}),
    'old801020': ('index_hist_sw', {'symbol': '801020', 'period': 'day'}),
    'gc182': ('bond_zh_hs_daily', {'symbol': 'sh204182'})}
UPSTREAM = Path('docs/handoff/2026-10-05-batch20-verification.json')
TERMINAL = Path('data/staging/strategies-batch20/20261005-macd-patterns/probes/sw801030/result.json')
FILES = {'scripts/review_strategy_batch22.py', 'tests/unit/test_batch22_research.py', 'src/observe/strategy_catalog.py',
    'scripts/review_strategy_batch21.py', 'scripts/review_strategy_batch18.py', 'scripts/run_strategy_batch4.py',
    'scripts/verify_strategy_batch3.py', 'scripts/archive_strategy_progress.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-apis.json', 'probe-results.json', 'input-analysis.json',
    'research-inputs.parquet', 'component-research.json', 'component-offline.json', 'offline-verification.json', 'diagnostics.json',
    'source-review-supplement.json', 'analysis-failure.json', 'calendar-failure.json', 'study-failure.json'}


def upstream():
    receipt = read(UPSTREAM)
    require(receipt['status'] == 'ok' and receipt['reviews']['snapshot'] == SNAPSHOT and
        file_sha(TERMINAL) == receipt['checks']['evidence_sha256']['probes/sw801030/result.json'], 'Accepted upstream terminal differs')
    terminal = read(TERMINAL)
    require(terminal['status'] == 'success' and terminal['endpoint'] == 'sw801030' and len(terminal['files']) == 1, 'Upstream endpoint differs')
    for row in [*terminal['files'], *terminal['wire_responses']]: require(file_sha(row['file']) == row['sha256'], 'Upstream input differs')
    return {'receipt': str(UPSTREAM), 'receipt_sha256': file_sha(UPSTREAM), 'terminal': str(TERMINAL),
        'terminal_sha256': file_sha(TERMINAL), 'result': terminal}


def bound_upstream(directory):
    row = read(directory / 'source-reviews/review.json')['upstream']
    require(row == upstream(), 'Upstream binding changed')
    return row


def clarified_reviews(directory):
    initial = read(directory / 'source-reviews/review.json'); supplement = read(directory / 'source-review-supplement.json')
    row = initial['sources'][2]; after = REVIEWS[SOURCES[2]]
    require(supplement['review_sha256'] == file_sha(directory / 'source-reviews/review.json') and
        supplement['source_sha256'] == row['source_sha256'] and supplement['before'] == row['review'] and supplement['after'] == after, 'Rule supplement binding differs')
    expected = json.loads(json.dumps(row['review']))
    require(expected['rules']['metrics'] in (after['rules']['metrics'], 'LGBMClassifier.predict实际硬标签，被当概率求AUC/多阈值'), 'Unexpected earlier metrics rule')
    expected['rules']['metrics'] = after['rules']['metrics']
    require(expected == after, 'Supplement changed unrelated rules')
    return supplement


def clarify(root, directory):
    row = read(directory / 'source-reviews/review.json')['sources'][2]
    save(directory / 'source-review-supplement.json', {'review_sha256': file_sha(directory / 'source-reviews/review.json'),
        'source_sha256': row['source_sha256'], 'before': row['review'], 'after': REVIEWS[SOURCES[2]],
        'reason': 'Classifier hard-label AUC and later Booster probability threshold stages must be distinguished; initial evidence retained'})
    clarified_reviews(directory)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    return {**archive(root, 'Batch22 LightGBM Classifier/Booster distinction corrected in a SHA-bound supplement'), 'catalog': catalog['catalog_id']}


def start(root, directory):
    checkpoint(root, directory, 'Batch22 original risk-premium/industry/model research started')
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Publication/snapshot differs')
    previous_file = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    previous = {r['path']: r for r in read(previous_file)}
    target = directory / 'source-reviews'; target.mkdir(); sources = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = target / f'{strategy_id(name)}.source'
        require(previous[name]['review_status'] != '人工审查完成' and file_sha(source) == previous[name]['bytes_sha256'], 'Already reviewed/source differs')
        shutil.copyfile(source, copied); text, encoding = read_source(source)
        sources.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    command = ['rg', '--files', '--hidden', 'repo', 'data', '-g', 'trainData_20211101_all.txt', '-g', 'gbm_model_v1*.pkl',
        '-g', 'gbm_train_model_v2_all.pkl', '-g', 'Ret_mat.csv', '-g', 'datas.csv', '-g', 'df_m_shifted_.csv', '-g', 'weight_in.csv']
    search = subprocess.run(command, capture_output=True, text=True); require(search.returncode in (0, 1), search.stderr)
    import akshare as ak
    apis = []
    for name in sorted({name for name, _ in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'signature': str(inspect.signature(fn)), 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    save(directory / 'existing-apis.json', {'version': ak.__version__, 'apis': apis, 'queries': QUERIES})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(target / 'review.json', {'snapshot': SNAPSHOT, 'sources': sources, 'previous_catalog_file': str(previous_file),
        'previous_catalog_sha256': file_sha(previous_file), 'catalog': catalog, 'upstream': upstream(),
        'dependency_search': {'command': command, 'returncode': search.returncode, 'matches': search.stdout.splitlines()},
        'strategy_results': [], 'limits': ['No model training, unknown pickle loading or index trade simulation',
            'Historical 28-industry and macro availability semantics remain unproved']})
    return archive(root, 'Batch22 four full source reviews archived; original inputs/model gaps retained')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(15)
    result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'wire_responses': [], 'published': False,
        'api_evidence_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'HTTP response cap reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path),
            'sha256': file_sha(path), 'tls_verified': kwargs.get('verify', session.verify)})
        return response
    try:
        import akshare as ak
        evidence = read(directory / 'existing-apis.json'); name, parameters = QUERIES[endpoint]; fn = getattr(ak, name)
        row = next(r for r in evidence['apis'] if r['name'] == name)
        require(ak.__version__ == evidence['version'] and evidence['queries'][endpoint] == [name, parameters] and
            hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == row['source_sha256'], 'API changed')
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = fn(**parameters)
        path = raw.save(root, 'industry_macro_probe_batch22', endpoint, directory.name, frame)
        result['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame), 'parameters': parameters})
        result['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    require(not (directory / 'probes').exists(), 'Probe archive exists'); results = []
    for endpoint in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch22', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=50)
            require(process.returncode == 0, process.stderr)
        except subprocess.TimeoutExpired as exc:
            path = directory / 'probes' / endpoint / 'result.json'
            if not path.exists(): save(path, {'endpoint': endpoint, 'status': 'timeout', 'files': [], 'wire_responses': [],
                'published': False, 'api_evidence_sha256': file_sha(directory / 'existing-apis.json'),
                'error': str(exc), 'limits': ['Worker interrupted; partial raw files are not accepted inputs']})
        results.append(read(directory / 'probes' / endpoint / 'result.json'))
        archive(root, f'Batch22 dependency {endpoint}: {results[-1]["status"]}; archive only')
        print(endpoint, results[-1]['status'], flush=True)
    save(directory / 'probe-results.json', {'results': results, 'published': False})
    return {'status': 'ok', 'results': results}


def industry_sample(frame):
    require({'代码', '日期', '开盘', '收盘'} <= set(frame), 'Industry schema differs')
    require(frame['代码'].astype(str).eq('801030').all(), 'Unexpected industry code')
    data = frame.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close'})[['date', 'open', 'close']].copy()
    data.date = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
    data = data[data.date.between('2005-12-01', '2017-12-31')].sort_values('date').reset_index(drop=True)
    for column in ('open', 'close'): data[column] = pd.to_numeric(data[column], errors='raise').astype(float)
    require(len(data) > 15 and not data.date.duplicated().any(), 'Insufficient/duplicate industry sample')
    require(np.isfinite(data[['open', 'close']].to_numpy(dtype=float)).all() and data[['open', 'close']].gt(0).all().all(), 'Invalid industry price')
    return data


def validate_probes(directory):
    import akshare as ak
    evidence = read(directory / 'existing-apis.json')
    require(evidence['version'] == ak.__version__ and evidence['queries'] == {k: [n, p] for k, (n, p) in QUERIES.items()}, 'API scope differs')
    require({r['name'] for r in evidence['apis']} == {n for n, _ in QUERIES.values()}, 'API set differs')
    for item in evidence['apis']:
        fn = getattr(ak, item['name'])
        require(hashlib.sha256(item['source'].encode()).hexdigest() == item['source_sha256'] ==
            hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() and item['signature'] == str(inspect.signature(fn)), 'API source/signature differs')
    probes = read(directory / 'probe-results.json'); profiles = []
    require(probes['published'] is False and [r['endpoint'] for r in probes['results']] == list(QUERIES), 'Probe scope differs')
    for terminal in probes['results']:
        require(terminal == read(directory / 'probes' / terminal['endpoint'] / 'result.json') and terminal['published'] is False and
            terminal['api_evidence_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe summary/API binding differs')
        require(terminal['status'] in ('success', 'empty', 'failed', 'timeout'), 'Unknown probe status')
        for item in [*terminal['files'], *terminal['wire_responses']]: require(file_sha(item['file']) == item['sha256'], 'Probe artifact changed')
        if terminal['status'] in ('success', 'empty'):
            require(len(terminal['files']) == 1, 'Probe input set differs')
            item = terminal['files'][0]; data = pd.read_parquet(item['file'])
            require(item['parameters'] == QUERIES[terminal['endpoint']][1] and list(data) == item['columns'] and len(data) == item['rows'] and
                (bool(len(data)) == (terminal['status'] == 'success')), 'Probe parameters/schema/rows differ')
        else: require(not terminal['files'] and terminal.get('error'), 'Failed probe contains accepted input/no error')
        profiles.append({'endpoint': terminal['endpoint'], 'status': terminal['status'], 'rows': [r['rows'] for r in terminal['files']],
            'strict_usable': False, 'limit': 'Sample/failed interface only; original vendor/history/availability equivalence unproved'})
    return profiles


def analyze(root, directory):
    row = bound_upstream(directory); data = industry_sample(pd.read_parquet(row['result']['files'][0]['file']))
    calendar = Store(root).load_state(Store(root).state(SNAPSHOT), 'calendar')
    dates = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    expected = dates[calendar.is_open & dates.between('2005-12-01', '2017-12-31')].sort_values().tolist()
    require(not set(data.date) - set(expected), 'Unexpected industry trading date')
    missing = sorted(set(expected) - set(data.date))
    profiles = validate_probes(directory)
    path = directory / 'research-inputs.parquet'; require(not path.exists(), 'Input archive exists'); data.to_parquet(path, index=False)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'input_file': str(path), 'input_sha256': file_sha(path),
        'industry': '801030 only', 'rows': len(data), 'first': data.date.iloc[0], 'last': data.date.iloc[-1],
        'status': 'limited_component_inputs' if missing else 'calendar_complete_component_inputs',
        'calendar_missing': len(missing), 'missing_sessions': missing, 'calendar_sessions': expected,
        'numeric_conversion': 'Strict pd.to_numeric errors=raise; original raw string fields unchanged',
        'probe_profiles': profiles, 'not_a_backtest': True, 'limits': ['Retrospective provider index sample, not original Juyuan universe',
            'Single-industry ranks are trivial and not evidence for original 28-industry ranking',
            'Cross-gap available-row values are literal diagnostics only; calendar-incomplete windows unavailable, never filled']})
    return archive(root, 'Batch22 verified original-date single-industry input and dependency attempts archived')


def reference_factors(open_values, close_values):
    opens, closes = np.asarray(open_values, float), np.asarray(close_values, float)
    require(opens.ndim == closes.ndim == 1 and len(opens) == len(closes) and len(opens) > 15 and
        np.isfinite(opens).all() and np.isfinite(closes).all() and (opens > 0).all() and (closes > 0).all(), 'Invalid factor inputs')
    r0 = [c / o - 1 for o, c in zip(opens, closes, strict=True)]
    r1 = [opens[k] / closes[k - 1] - 1 for k in range(1, len(opens))]
    return {'M': [closes[k] / closes[k - 15] - 1 for k in range(15, len(opens))],
        'M0': [math.fsum(r0[k - 14:k + 1]) for k in range(14, len(opens))],
        'M1': [math.fsum(r1[k - 15:k]) for k in range(15, len(opens))]}


def average_ranks(values, descending=False):
    return [1 + sum(other > value if descending else other < value for other in values) +
        (sum(other == value for other in values) - 1) / 2 for value in values]


def calendar_windows(dates, sessions, width):
    require(len(dates) == len(set(dates)) and dates == sorted(dates) and len(sessions) == len(set(sessions)) and
        sessions == sorted(sessions) and set(dates) <= set(sessions) and width > 0, 'Invalid calendar window inputs')
    positions = {day: k for k, day in enumerate(sessions)}
    return {day: k + 1 >= width and positions[day] + 1 >= width and
        dates[k - width + 1:k + 1] == sessions[positions[day] - width + 1:positions[day] + 1] for k, day in enumerate(dates)}


def compute(directory):
    bound_upstream(directory); row = read(directory / 'input-analysis.json')
    require(file_sha(row['input_file']) == row['input_sha256'], 'Research input changed')
    sample = pd.read_parquet(row['input_file']); require(industry_sample(pd.DataFrame({'代码': '801030', '日期': sample.date, '开盘': sample.open, '收盘': sample.close})).equals(sample), 'Research sample differs')
    namespace = {'pd': pd, 'np': np}; fingerprint = selected(directory, 1, ('get_alpha',), namespace)
    opens = pd.DataFrame({'801030': sample.open.to_numpy()}, index=pd.to_datetime(sample.date))
    closes = pd.DataFrame({'801030': sample.close.to_numpy()}, index=opens.index)
    outputs = namespace['get_alpha'](opens, closes, '2006', '2017'); reference = reference_factors(sample.open, sample.close)
    aligned = {'M': opens.index[15:], 'M0': opens.index[14:], 'M1': opens.index[15:]}; components = {}; differences = {}; availability = {}
    require(row['missing_sessions'] == sorted(set(row['calendar_sessions']) - set(sample.date)) and
        row['calendar_missing'] == len(row['missing_sessions']), 'Calendar gap declaration differs')
    for name, actual in zip(('M', 'M0', 'M1'), outputs[:3], strict=True):
        expected = pd.Series(reference[name], index=aligned[name]).loc['2006':'2017']
        require(actual.index.equals(expected.index), 'Original factor dates differ')
        delta = np.abs(actual.iloc[:, 0].to_numpy() - expected.to_numpy()); require(delta.max() < 1e-12, f'{name} values differ')
        differences[name] = {'values': len(expected), 'max_abs_difference': float(delta.max())}
        complete = calendar_windows(sample.date.tolist(), row['calendar_sessions'], 15 if name == 'M0' else 16)
        components[name] = [{'date': str(day.date()), 'value': float(value), 'calendar_complete': complete[str(day.date())]} for day, value in actual.iloc[:, 0].items()]
        valid = sum(r['calendar_complete'] for r in components[name])
        availability[name] = {'calendar_complete_values': valid, 'calendar_incomplete_values': len(components[name]) - valid}
    require(outputs[3].eq(2).all().all(), 'Single-industry rank differs')
    # Equal pairs exercise average ties with all 28 columns; this is a synthetic ranking diagnostic.
    synthetic_open = pd.DataFrame(np.ones((40, 28)) * 100, index=pd.date_range('2017-01-01', periods=40))
    synthetic_close = synthetic_open * (1 + np.tile(np.arange(14), 2) / 100)
    synth = namespace['get_alpha'](synthetic_open, synthetic_close, '2017', '2017')
    a, b = synth[1].iloc[-1].tolist(), synth[2].iloc[-1].tolist()
    ranks = [x + y for x, y in zip(average_ranks(a), average_ranks(b, True), strict=True)]
    require(synth[3].iloc[-1].tolist() == ranks, '28-column rank diagnostic differs')
    return {'snapshot': SNAPSHOT, 'input_sha256': row['input_sha256'], 'source_sha256': read(directory / 'source-reviews/review.json')['sources'][1]['source_sha256'],
        'selected_ast_sha256': fingerprint, 'backend': {'numpy': np.__version__, 'pandas': pd.__version__},
        'industry': '801030', 'differences': differences, 'components': components, 'availability': availability,
        'synthetic_28_column_ranks': ranks, 'not_a_backtest': True,
        'limits': ['Only original factor function, no groups/long-short/fees/NAV', 'Synthetic ranks do not replace 28 real industry histories',
            'Original 15 available-row computation retained; all calendar-incomplete values are unavailable diagnostics, not trade signals']}


def diagnostics(directory):
    import lightgbm as lgb
    rows = []; ns = {'pd': pd, 'np': np}; selected(directory, 0, ('get_position',), ns)
    dates = pd.date_range('2020-01-01', periods=3)
    ns['final_df'] = pd.DataFrame({'pre1_date': pd.Series(dates, index=dates).shift(1),
        'pre2_date': pd.Series(dates, index=dates).shift(2), 'singal1': [1., 2., 3.]}, index=dates)
    try: ns['get_position'](dates[0])
    except KeyError: rows.append({'case': 'risk_premium_initial_NaT', 'error': 'KeyError', 'not_fixed': True})
    require(rows and ns['get_position'](dates[2]) == 1, 'Position diagnosis differs')
    rows.append({'case': 'risk_premium_valid_prior_dates', 'position': 1})
    text, _ = read_source(Path(read(directory / 'source-reviews/review.json')['sources'][2]['source_copy']))
    try: ast.parse(text)
    except SyntaxError as exc: rows.append({'case': 'lightgbm_complete_source', 'error': 'SyntaxError', 'line': exc.lineno, 'message': exc.msg})
    require(any(r['case'] == 'lightgbm_complete_source' for r in rows), 'Expected incomplete source differs')
    ns = {}; selected(directory, 2, ('get_acc_for_T_F',), ns)
    with redirect_stdout(io.StringIO()):
        try: ns['get_acc_for_T_F']([1, 1], [0, 1])
        except ZeroDivisionError: rows.append({'case': 'all_positive_predictions', 'error': 'ZeroDivisionError', 'not_fixed': True})
    ns = {'pd': pd, 'np': np}; selected(directory, 3, ('weight_timing_threshold',), ns)
    times = pd.date_range('2020-01-01', periods=2); ret = pd.DataFrame({'factor': [1., 1.]}, index=times)
    pred = pd.DataFrame({'factor': [-1, -1]}, index=times); weight = pd.DataFrame({'factor': [1., 1.]}, index=times)
    high = pd.DataFrame({'factor': [1., 1.]}, index=times); low = high.copy(); low.iloc[1, 0] = 0
    try: ns['weight_timing_threshold'](ret, pred.copy(), weight, high, .5)
    except TypeError as exc:
        require('dtype' in str(exc) and '0.1' in str(exc), 'Unexpected fractional-weight failure')
        rows.append({'case': 'fractional_z_in_original_integer_table', 'z': .1, 'error': 'TypeError', 'not_fixed': True})
    first = ns['weight_timing_threshold'](ret, pred.copy(), weight, high, .5, z=0)
    changed = ns['weight_timing_threshold'](ret, pred.copy(), weight, low, .5, z=0)
    require(first.factor.tolist() == [0, 0] and changed.factor.tolist() == [1., 1.], 'Whole-column override diagnosis differs')
    rows.append({'case': 'future_low_r2_rewrites_past_weight', 'synthetic_z': 0, 'original': first.factor.tolist(),
        'future_r2_changed': changed.factor.tolist(), 'not_fixed': True,
        'limit': 'Integer diagnostic parameter avoids fractional-assignment failure; no default-z successful execution claimed'})
    ns = {'pd': pd, 'np': np, 'timedelta': timedelta}; selected(directory, 1, ('get_month', 'get_group'), ns)
    ns['M0'] = pd.DataFrame({'x': [1., 2.]}, index=times)
    try: ns['get_month'](ns['M0'])
    except ValueError: rows.append({'case': 'industry_month_alias', 'error': 'ValueError', 'not_fixed': True})
    require(len(rows) == 7, 'Expected diagnostics missing')
    return {'cases': rows, 'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'lightgbm': {'version': lgb.__version__, 'train_signature': str(inspect.signature(lgb.train)),
            'early_stopping_rounds_supported': 'early_stopping_rounds' in inspect.signature(lgb.train).parameters,
            'model_fitting': False},
        'not_a_backtest': True, 'limits': ['Only selected reviewed functions; no top-level, model fitting, pickle loading or orders']}


def study(root, directory):
    save(directory / 'component-research.json', compute(directory)); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch22 original industry factors independently verified; compatibility and causal defects diagnosed')


def diagnose(root, directory):
    require(compute(directory) == read(directory / 'component-research.json') and
        read(directory / 'study-failure.json')['component_research_sha256'] == file_sha(directory / 'component-research.json'), 'Partial original component changed')
    save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch22 seven untouched original-function diagnostics archived; default fractional failure retained')


def offline(root, directory):
    def denied(*args, **kwargs): raise AssertionError('Network forbidden')
    before = file_sha(directory / 'component-research.json')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), patch.object(requests.sessions.Session, 'request', denied):
        repeated = compute(directory); diagnosed = diagnostics(directory)
    save(directory / 'component-offline.json', repeated)
    require(repeated == read(directory / 'component-research.json') and diagnosed == read(directory / 'diagnostics.json'), 'Offline components differ')
    require(before == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline bytes differ')
    save(directory / 'offline-verification.json', {'status': 'match', 'differences': 0, 'original_sha256': before,
        'recomputed_sha256': file_sha(directory / 'component-offline.json'), 'original_unchanged': True,
        'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True})
    return archive(root, 'Batch22 original-date factor component and seven diagnostics repeated offline, byte-identical research')


def validate_offline(directory):
    row = read(directory / 'offline-verification.json')
    require(row['status'] == 'match' and row['differences'] == 0 and row['original_unchanged'] is True and
        row['not_a_strategy_reproduction'] is True and row['network'] == 'socket connect/connect_ex and requests forbidden' and
        row['original_sha256'] == file_sha(directory / 'component-research.json') ==
        row['recomputed_sha256'] == file_sha(directory / 'component-offline.json'), 'Offline receipt status/binding differs')
    return row


def checks(root, directory):
    validate_probes(directory); validate_offline(directory)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py',
        'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py', 'scripts/review_strategy_batch22.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch22_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'], ['git', 'diff', '--check']]
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
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']) and
        FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checks incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    reviews = read(directory / 'source-reviews/review.json'); bound_upstream(directory); supplement = clarified_reviews(directory)
    require(reviews['snapshot'] == SNAPSHOT and file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'], 'Prior catalog/snapshot changed')
    require([r['source_path'] for r in reviews['sources']] == [str(Path('repo/量化策略源代码') / n) for n in SOURCES], 'Source set differs')
    for name, row in zip(SOURCES, reviews['sources'], strict=True):
        expected = supplement['after'] if name == SOURCES[2] else row['review']
        require(file_sha(row['source_path']) == row['source_sha256'] == row['source_copy_sha256'] == file_sha(row['source_copy']) and REVIEWS[name] == expected, 'Source/rules changed')
    analysis = read(directory / 'input-analysis.json')
    require(analysis['probe_profiles'] == validate_probes(directory) and analysis['snapshot'] == SNAPSHOT and analysis['not_a_backtest'] is True, 'Input analysis scope differs')
    calendar = Store(root).load_state(Store(root).state(SNAPSHOT), 'calendar'); dates = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    require(analysis['calendar_sessions'] == dates[calendar.is_open & dates.between('2005-12-01', '2017-12-31')].sort_values().tolist(), 'Frozen calendar differs')
    repeat = validate_offline(directory)
    pd.testing.assert_frame_equal(pd.read_parquet(analysis['input_file']), industry_sample(pd.read_parquet(reviews['upstream']['result']['files'][0]['file'])))
    require(compute(directory) == read(directory / 'component-research.json') == read(directory / 'component-offline.json') and
        diagnostics(directory) == read(directory / 'diagnostics.json'), 'Reconstructed research differs')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'review_supplement': supplement, 'analysis': analysis,
        'research': read(directory / 'component-research.json'), 'diagnostics': read(directory / 'diagnostics.json'),
        'offline': repeat, 'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch22 four reviews and original industry factor research accepted; entire strategy library continues')}
    target = Path('docs/handoff/2026-10-06-batch22-verification.json'); save(target, result)
    return {'status': 'ok', 'output': str(target), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'clarify', 'worker', 'probe', 'analyze', 'study', 'diagnose', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--endpoint', choices=list(QUERIES))
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch22/20261006-industry-premium-models'))
    args = parser.parse_args()
    if args.action == 'worker':
        if args.endpoint is None: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
