"""Freeze macro and open-fund dependencies; audit exact functions without trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
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

from observe.data import raw
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
SOURCES = ('2022年度精选策略/15.A股-宏观动量-v5-Clone1.txt',
    '2022年度精选策略/96.宏观择时多空组合——十年收益12倍，最大回撤6.7%.txt',
    '2024年度精选策略2/84.多大规模的基金收益最好？偏股基金规模研究.txt')
UPSTREAM = Path('docs/handoff/2026-10-06-batch24-verification.json')
BASELINE = Path('data/staging/strategies-batch24/20261006-volatility-value-raw/baseline.json')
QUERIES = {'pmi': ('macro_china_pmi', {}), 'shibor': ('macro_china_shibor_all', {}),
    'money_supply': ('macro_china_money_supply', {}), 'fund_scale': ('fund_scale_open_sina', {'symbol': '混合型基金'})}
REFERENCES = ('akshare/economic/macro_china.py', 'akshare/fund/fund_scale_sina.py')
FILES = {'scripts/review_strategy_batch25.py', 'tests/unit/test_batch25_research.py', 'src/observe/strategy_catalog.py',
    'src/observe/data/store.py', 'src/observe/runs.py',
    'scripts/review_strategy_batch21.py', 'scripts/review_strategy_batch22.py', 'scripts/review_strategy_batch18.py',
    'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py', 'scripts/archive_strategy_progress.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-apis.json', 'input-binding.json', 'input-analysis.json',
    'index-input.parquet', 'component-research.json', 'component-offline.json', 'diagnostics.json', 'offline-verification.json', 'probe-results.json'}
SIGNALS = ('get_rolling_positon', 'get_position_from_continus_increase', 'get_position_from_long_short_monving_average')


def upstream():
    receipt = read(UPSTREAM)
    require(receipt['status'] == 'ok' and receipt['reviews']['snapshot'] == SNAPSHOT and
        file_sha(BASELINE) == receipt['checks']['evidence_sha256']['baseline.json'], 'Upstream receipt differs')
    return {'file': str(UPSTREAM), 'sha256': file_sha(UPSTREAM), 'baseline_file': str(BASELINE), 'baseline_sha256': file_sha(BASELINE)}


def index_binding(root):
    store = Store(root); state = store.state(SNAPSHOT); entries = []
    require(state['tables'] == read(BASELINE)['published']['tables'], 'Accepted snapshot differs')
    for table in ('index_1d', 'calendar'):
        require(state['tables'].get(table), 'Missing index/calendar table')
        for part, entry in sorted(state['tables'][table].items()):
            path = store.root / entry['file']; frame = pd.read_parquet(path)
            require(fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'], 'Index partition differs')
            entries.append({'table': table, 'part': part, 'file': str(path), 'rows': entry['rows'], 'fingerprint': entry['sha'], 'sha256': file_sha(path)})
    return {'snapshot': SNAPSHOT, 'upstream': upstream(), 'partitions': entries}


def index_sample(root):
    store = Store(root); state = store.state(SNAPSHOT)
    frame = store.load_state(state, 'index_1d', filters=[('index', '==', '000300.SH')])[['date', 'close']].copy()
    frame.date = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d'); frame = frame.sort_values('date').reset_index(drop=True)
    cal = store.load_state(state, 'calendar'); dates = pd.to_datetime(cal.date).dt.strftime('%Y-%m-%d')
    dates = dates[cal.is_open & dates.between('2005-01-05', '2026-09-29')].sort_values().tolist()
    require(frame.date.tolist() == dates and not frame.date.duplicated().any() and np.isfinite(frame.close).all() and frame.close.gt(0).all(), 'Index sample/calendar differs')
    return frame


def api_evidence():
    import akshare as ak
    apis = []
    for name in sorted({n for n, _ in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'signature': str(inspect.signature(fn)), 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'version': ak.__version__, 'queries': QUERIES, 'apis': apis}


def start(root, directory):
    checkpoint(root, directory, 'Batch25 original macro/open-fund source research started')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    save(directory / 'input-binding.json', index_binding(root)); prior = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; target = directory / 'source-reviews'; target.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = target / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'], 'Source changed/already reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_sha256': file_sha(path), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    command = ['rg', '--files', '--hidden', 'repo', 'data', '-g', 'PMI组合.xls', '-g', 'SHIBOR数据.xls', '-g', '国债到期收益率.xls',
        '-g', '企业债到期收益率(AAA).xls', '-g', '信贷.xls', '-g', '存款准备金率-大型存款类机构.xls', '-g', '离岸汇率数据.xls', '-g', 'CPI与PPI.xls']
    search = subprocess.run(command, capture_output=True, text=True); require(search.returncode in (0, 1), search.stderr)
    refdir = directory / 'references'; refdir.mkdir(); refs = []
    commit = subprocess.run(['git', '-C', 'repo/akshare', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    for name in REFERENCES:
        path = Path('repo/akshare') / name; copied = refdir / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path), 'commit': commit,
            'use': 'Read-only API coverage: PMI aggregate lacks subindices; fund current total scale is not quarterly stock_value'})
    save(directory / 'existing-apis.json', api_evidence())
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(target / 'review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'catalog': catalog, 'previous_catalog_file': str(prior),
        'previous_catalog_sha256': file_sha(prior), 'references': refs, 'dependency_search': {'command': command, 'returncode': search.returncode, 'matches': search.stdout.splitlines()},
        'strategy_results': [], 'limits': ['No top-level, unknown PKL, simplified NAV, RF fit or trading/message hooks executed']})
    return archive(root, 'Batch25 three full macro/open-fund source reviews and unavailable original inputs archived')


def prepare(root, directory):
    require(index_binding(root) == read(directory / 'input-binding.json'), 'Input binding changed'); frame = index_sample(root)
    path = directory / 'index-input.parquet'; require(not path.exists(), 'Input archive exists'); frame.to_parquet(path, index=False)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
        'first': frame.date.min(), 'last': frame.date.max(), 'local_macro_tables': [], 'local_open_fund_tables': [],
        'actual_snapshot_tables': sorted(Store(root).state(SNAPSHOT)['tables']), 'not_a_backtest': True,
        'purpose': 'Verified real index price context for old-resample failure only, never substitute macro/fund inputs'})
    return archive(root, 'Batch25 frozen index diagnostic context retained; macro/open-fund dependencies absent')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(10)
    name, params = QUERIES[endpoint]; row = {'endpoint': endpoint, 'function': name, 'parameters': params, 'status': 'failed',
        'files': [], 'wire_responses': [], 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
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
        path = raw.save(root, 'macro_fund_probe_batch25', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API changed')
    rows = read(directory / 'probe-results.json')['results']
    require(len(rows) == len(QUERIES) and {r['endpoint'] for r in rows} == set(QUERIES), 'Probe scope differs')
    for row in rows:
        name, params = QUERIES[row['endpoint']]
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['function'] == name and row['parameters'] == params and
            row['published'] is False and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe binding differs')
        require(row['status'] in ('failed', 'timeout', 'success', 'empty'), 'Unknown probe status')
        for item in row['wire_responses']: require(file_sha(item['file']) == item['sha256'], 'Wire response changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and
                bool(len(frame)) == (row['status'] == 'success'), 'Raw profile differs')
        else: require(not row['files'] and row.get('error'), 'Failed probe has accepted raw/no error')
    return rows


def probe(root, directory):
    results = []
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe archive exists'); logs = directory / 'logs'; logs.mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'scripts.review_strategy_batch25', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline 45s', 'published': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        results.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': results, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch25 four existing macro/current-fund API attempts archived without publication')


def signal_reference(values, kind, n, delay, how, short_n=2):
    data = list(map(float, values)); require(n >= 1 and delay >= 0 and how in ('up', 'down') and kind in ('rolling', 'continuous', 'ma') and not np.isinf(data).any(), 'Invalid signal inputs')
    require(kind != 'ma' or short_n >= 1, 'Invalid short window')
    output = []
    def mean(end, window):
        part = data[max(0, end - window + 1):end + 1]
        return math.fsum(part) / window if len(part) == window and all(math.isfinite(v) for v in part) else math.nan
    start = n if kind == 'continuous' else 0
    for k in range(start, len(data)):
        if kind == 'rolling':
            a, b = mean(k, n), mean(k - 1, n); value = float(a > b if how == 'up' else a < b)
        elif kind == 'continuous':
            value = float(all(data[j] < data[j + 1] if how == 'up' else data[j] > data[j + 1] for j in range(k - n, k)))
        else:
            diff = mean(k, short_n) - mean(k, n)
            value = float(diff > 0 if how == 'up' else diff < 0) if math.isfinite(diff) else math.nan
        output.append((k, value))
    return [(k + delay, value) for k, value in output if k + delay < len(data) and math.isfinite(value)]


def synthetic_series():
    dates = pd.period_range('2009-01', periods=36, freq='M').astype(str)
    values = [10. + k % 6 for k in range(36)]; values[17] = math.nan
    return pd.Series(values, index=dates, name='synthetic_macro_not_observed')


def compute(root, directory):
    require(index_binding(root) == read(directory / 'input-binding.json'), 'Input binding changed'); ns = {'pd': pd}
    code = selected(directory, 1, SIGNALS, ns); series = synthetic_series(); cases = []
    for kind, fn in zip(('rolling', 'continuous', 'ma'), SIGNALS, strict=True):
        specs = [(n, None) for n in (1, 2, 3)] if kind == 'rolling' else [(2, None)] if kind == 'continuous' else [(8, 4), (10, 2), (10, 3), (10, 4), (14, 4)]
        for how in ('up', 'down'):
            for delay in (1, 2, 3):
                for n, short in specs:
                    actual = ns[fn](series, n, delay=delay, how=how) if kind != 'ma' else ns[fn](series, long_n=n, short_n=short, delay=delay, how=how)
                    expected = signal_reference(series, kind, n, delay, how, short_n=short); require(actual.index.tolist() == [series.index[k] for k, _ in expected] and
                        actual.position.tolist() == [v for _, v in expected], 'Original synthetic signal differs')
                    cases.append({'kind': kind, 'window': n, 'short_window': short, 'delay_rows': delay, 'direction': how,
                        'values': {str(i): float(v) for i, v in actual.position.items()}, 'differences': 0})
    return {'snapshot': SNAPSHOT, 'source_ast_sha256': code, 'synthetic_input': {i: float(v) if pd.notna(v) else None for i, v in series.items()},
        'cases': cases, 'not_a_backtest': True, 'synthetic_only': True, 'backend': {'pandas': pd.__version__, 'numpy': np.__version__},
        'limits': ['No real macro factor/availability/ranking or long-short NAV claim; shifts count observed rows, never invented calendar months']}


class Field:
    def __init__(self, name): self.name = name
    def in_(self, values): return ('in', self.name, values)
    def __lt__(self, value): return ('lt', self.name, value)
    def __gt__(self, value): return Predicate(('gt', self.name, value))
    def __eq__(self, value): return ('eq', self.name, value)
    def __ge__(self, value): return ('ge', self.name, value)
    def __le__(self, value): return ('le', self.name, value)


class Predicate:
    def __init__(self, value): self.value = value
    def __or__(self, other): return ('or', self.value, other)


class Query:
    def __init__(self, fields): self.fields = [f.name for f in fields]; self.filters = []
    def filter(self, *criteria): self.filters.extend(criteria); return self


def fund_quarter_diagnostic(directory):
    queries = []; main = SimpleNamespace(**{n: Field(n) for n in ('main_code', 'operate_mode_id', 'underlying_asset_type_id', 'invest_style_id', 'start_date', 'end_date')})
    portfolio = SimpleNamespace(**{n: Field(n) for n in ('code', 'stock_rate', 'period_end')})
    def run(q):
        queries.append({'fields': q.fields, 'filters': q.filters.copy()})
        return pd.DataFrame({q.fields[0]: ['000001']})
    ns = {'datetime': datetime, 'timedelta': datetime.timedelta, 'finance': SimpleNamespace(FUND_MAIN_INFO=main, FUND_PORTFOLIO=portfolio, run_query=run), 'query': lambda *fields: Query(fields)}
    selected(directory, 2, ('get_report_date', 'get_fund_code'), ns); rows = []
    for day in ('2024-01-31', '2024-04-30', '2024-07-31', '2024-10-31'):
        quarter = ns['get_report_date'](day); before = len(queries); codes = ns['get_fund_code'](quarter, stock_rate_min=70)
        calls = queries[before:]; selected_quarter = next(c[2] for c in calls[1]['filters'] if c[0] == 'eq' and c[1] == 'period_end')
        rows.append({'decision_date': day, 'stock_value_quarter': quarter, 'stock_rate_quarter': selected_quarter, 'queries': calls, 'codes': codes})
    require(rows[1]['stock_value_quarter'] == '2024-03-31' and rows[1]['stock_rate_quarter'] == '2023-12-31', 'Fund double-quarter lag differs')
    return rows


def diagnostics(root, directory):
    rows = []; ns = {}; selected(directory, 0, ('good_cpi',), ns)
    inputs = [-1., 0., 4.999999, 5., math.nan]; values = [ns['good_cpi'](v) for v in inputs]
    require(values == [0., 1., 1., 0., 0.], 'CPI boundary differs'); rows.append({'case': 'cpi_bounds_nan', 'inputs': [-1., 0., 4.999999, 5., None], 'values': values})
    text, _ = read_source(Path(read(directory / 'source-reviews/review.json')['sources'][0]['source_copy']))
    try: ast.parse(text)
    except SyntaxError as exc: rows.append({'case': 'python2_source_parse_failure', 'line': exc.lineno, 'message': exc.msg})
    require(len(rows) == 2, 'Python2 defect missing')
    ns = {'pd': pd}; selected(directory, 1, ('get_month_list',), ns); months = ns['get_month_list']('2024-03-01', '2024-05-31')
    require(months == [f'2024-{k:02d}' for k in range(3, 13)], 'Same-year month bug differs')
    rows.append({'case': 'same_year_months_exceed_end', 'start': '2024-03-01', 'end': '2024-05-31', 'months': months})
    info = read(directory / 'input-analysis.json'); require(info['file'] == str(directory / 'index-input.parquet') and file_sha(info['file']) == info['sha256'], 'Index context changed')
    frame = pd.read_parquet(info['file']); pd.testing.assert_frame_equal(frame, index_sample(root))
    prices = frame.assign(date=pd.to_datetime(frame.date)).set_index('date').close
    ns = {'pd': pd, 'get_price': lambda *a, **kw: {'close': prices}}; selected(directory, 1, ('get_month_list', 'get_profit_monthly'), ns)
    try: ns['get_profit_monthly']('2009-01-01', '2019-03-29')
    except (TypeError, ValueError) as exc: rows.append({'case': 'legacy_month_resample_failure', 'error': f'{type(exc).__name__}: {exc}', 'index_context_rows': len(frame)})
    ns = {'pd': pd}; selected(directory, 1, SIGNALS, ns); source, _ = read_source(Path(read(directory / 'source-reviews/review.json')['sources'][1]['source_copy']))
    tree = ast.parse(source); assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inventory_idx' for t in node.targets)]
    require(len(assignments) == 4, 'Inventory assignment scope differs'); series = synthetic_series().ffill()
    ns['mac_monmanufacturing_sell'] = pd.DataFrame({'inventory_idx': series})
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<reviewed_inventory_assignments>', 'exec'), ns)
    reference = signal_reference(series, 'ma', 10, 2, 'up', short_n=4)
    require(ns['inventory_idx'].position.tolist() == [v for _, v in reference], 'Inventory overwrite differs')
    concatenation = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_position_for_sell' for t in node.targets))
    names = [n.id for n in concatenation.value.args[0].elts]; require(names.count('inventory_idx') == 2, 'Duplicate inventory weight differs')
    rows.append({'case': 'short_inventory_overwritten_and_duplicated', 'original_assignment_ast_sha256': hashlib.sha256(ast.dump(ast.Module(body=assignments, type_ignores=[])).encode()).hexdigest(),
        'final_long_short_delay': [10, 4, 2], 'final_concat_inputs': names, 'inventory_columns': 2, 'total_columns': len(names), 'synthetic_only': True})
    rows.append({'case': 'fund_double_quarter_and_age_cutoff', 'rows': fund_quarter_diagnostic(directory), 'platform_datetime_injection_explicit': True})
    events = []; state = SimpleNamespace(retry_num=0, finish_trade=False, fund_targetpositions={'B': 1.})
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'A': object()}, positions_value=1., total_value=100.))
    ns = {'g': state, 'sell': lambda *a: events.append('sell'), 'buy': lambda *a: events.append('buy'), 'record': lambda **kw: None}
    selected(directory, 2, ('trade',), ns)
    for _ in range(10): ns['trade'](ctx)
    require(state.retry_num == 0 and events == ['sell', 'buy'] * 10, 'Fund retry behavior differs')
    rows.append({'case': 'fund_retry_not_incremented', 'callbacks': 10, 'retry_after': state.retry_num, 'stub_sell_buy_calls': len(events)})
    events.clear(); ctx.portfolio.positions = {'B': object()}; ns['trade'](ctx)
    require(state.finish_trade and not events, 'Fund code-only finish differs')
    rows.append({'case': 'fund_finishes_with_codes_only', 'position_weight': .01, 'target_weight': 1., 'finish_trade': state.finish_trade, 'stub_sell_buy_calls': 0})
    state = SimpleNamespace(last_month=0, min_asset=150000000., funds_num=10); ctx = SimpleNamespace(current_dt=datetime.datetime(2024, 4, 30, 9, 30))
    ns = {'g': state, 'log': SimpleNamespace(info=lambda *a: None), 'get_report_date': lambda *a: '2024-03-31', 'get_fund_code': lambda *a, **kw: [],
        'get_stock_asset': lambda *a, **kw: pd.DataFrame(columns=['stock_value']), 'trade': lambda *a: None}
    selected(directory, 2, ('check_out',), ns); output = io.StringIO()
    with redirect_stdout(output): ns['check_out'](ctx)
    require(state.fund_targetpositions == {}, 'Empty fund target differs')
    rows.append({'case': 'empty_fund_dict_no_division', 'targets': state.fund_targetpositions, 'stdout': output.getvalue(), 'downstream_trading_not_evaluated': True})
    require(len(rows) == 9, 'Expected diagnostics missing')
    return json.loads(json.dumps({'cases': rows, 'not_a_backtest': True, 'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'limits': ['Only exact reviewed definitions/assignments with explicit synthetic inputs, query/order stubs; no real macro/OTC fund execution']}))


def study(root, directory):
    result = compute(root, directory); diag = diagnostics(root, directory)
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch25 exact macro kernels and nine source defects independently diagnosed; synthetic only')


def offline(root, directory):
    def denied(*args, **kwargs): raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), patch.object(requests.sessions.Session, 'request', denied):
        result = compute(root, directory); diag = diagnostics(root, directory)
    save(directory / 'component-offline.json', result); before = file_sha(directory / 'component-research.json')
    require(result == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and before == file_sha(directory / 'component-offline.json'), 'Offline differs')
    save(directory / 'offline-verification.json', {'status': 'match', 'differences': 0, 'original_unchanged': True, 'original_sha256': before,
        'recomputed_sha256': file_sha(directory / 'component-offline.json'), 'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True})
    return archive(root, 'Batch25 synthetic source diagnostics byte-identical offline; not trading reproduction')


def checks(root, directory):
    validate_probes(directory); validate_offline(directory)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
        'scripts/verify_financial_import.py', 'scripts/review_strategy_batch25.py'], [sys.executable, '-m', 'pytest', '-q',
        'tests/unit/test_batch25_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True); row = {'command': command, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
        results.append(row); print(row, flush=True); require(proc.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    checked = read(directory / 'checks.json'); require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']) and
        FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checks incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()) and
        all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Checked code/evidence changed')
    reviews = read(directory / 'source-reviews/review.json'); require(reviews['snapshot'] == SNAPSHOT and
        [r['source_path'] for r in reviews['sources']] == [str(Path('repo/量化策略源代码') / n) for n in SOURCES] and
        file_sha(reviews['previous_catalog_file']) == reviews['previous_catalog_sha256'], 'Review scope changed')
    for name, row in zip(SOURCES, reviews['sources'], strict=True):
        require(file_sha(row['source_path']) == row['source_sha256'] == row['source_copy_sha256'] == file_sha(row['source_copy']) and row['review'] == REVIEWS[name], 'Source/rules changed')
    for row in reviews['references']: require(file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')
    require(index_binding(root) == read(directory / 'input-binding.json'), 'Index binding changed'); info = read(directory / 'input-analysis.json')
    frame = index_sample(root); require(info['rows'] == len(frame) and info['first'] == frame.date.min() and info['last'] == frame.date.max() and
        info['snapshot'] == SNAPSHOT and info['not_a_backtest'] is True and info['actual_snapshot_tables'] == sorted(Store(root).state(SNAPSHOT)['tables']), 'Index profile changed')
    require(compute(root, directory) == read(directory / 'component-research.json') == read(directory / 'component-offline.json') and
        diagnostics(root, directory) == read(directory / 'diagnostics.json'), 'Reconstructed research differs')
    probes = validate_probes(directory); offline_row = validate_offline(directory)
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed'); protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen implementation differs')
    result = {'status': 'ok', 'environment': environment(), 'reviews': reviews, 'input_analysis': info, 'input_binding': read(directory / 'input-binding.json'),
        'research': read(directory / 'component-research.json'), 'diagnostics': read(directory / 'diagnostics.json'), 'probes': probes,
        'offline': offline_row, 'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch25 macro/open-fund original defects and missing inputs accepted; whole library continues')}
    output = Path('docs/handoff/2026-10-06-batch25-verification.json'); save(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'worker', 'probe', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch25/20261006-macro-open-fund'))
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    if args.action == 'worker':
        require(args.endpoint is not None, 'Endpoint required'); result = worker(args.root, args.directory, args.endpoint)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
