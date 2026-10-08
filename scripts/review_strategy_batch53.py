"""Freeze dynamic fund, half-hour research and minute limit-touch source evidence."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
import hashlib
import inspect
from io import StringIO
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

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts import review_strategy_batch52 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch52/20261007-fund-universe')
RECEIPT = Path('docs/handoff/2026-10-07-batch52-verification.json')
SOURCES = ('2021年度精选策略/25.基本不耍六毛的ETF轮动策略.txt',
    '2021年度精选策略/66.【复现】A股日内动量效应（一）半小时 涨跌幅间的规律.txt',
    '2021年度精选策略/75.最近2年年化100%以上回撤12%的打板策略-验证策略.txt')
SOURCE_SHA = ('d8741c3c839fa0686462abad50b35041bca561b3f3dcae1d77cdb3b1892f9f26',
    'bfbad3a743a1c7d94b55866b613da674f564563f3a94b4ab48d076bb46b125c4',
    'e5490946e348c1cad9745537b6007601470c4e4ff07a9d8dff5a09310b033f95')
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/akshare', 'akshare/index/index_zh_em.py'), ('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'))
QUERIES = {'fund_pool': ('fund_etf_category_sina', {'symbol': 'ETF基金'}),
    'index_halfhour': ('index_zh_a_hist_min_em', {'symbol': '000300', 'period': '30',
        'start_date': '2014-01-01 09:30:00', 'end_date': '2020-04-24 15:00:00'}),
    'stock_minute': ('stock_zh_a_hist_min_em', {'symbol': '000001', 'period': '1', 'adjust': '',
        'start_date': '2024-03-29 09:30:00', 'end_date': '2024-03-29 11:30:00'})}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch53.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'diagnostics.json', 'probe-results.json', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch52 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'), 'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch53 dynamic fund/halfhour/minute source research started')
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
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
        'talib': {'version': talib.__version__, 'compatibility': talib.get_compatibility()}, 'queries': QUERIES, 'apis': rows,
        'limits': ['Sina category is a current list, not a historical universe',
            'Eastmoney minute interfaces have limited coverage; requested historical range is not evidence of returned coverage',
            'No result is published or silently substituted into the frozen snapshot']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); counts = {}
    indices = ['000300.SH', '000905.SH', '000001.SH', '399001.SZ']
    for table in ('index_1d', 'bars_5m'):
        column = 'index' if table == 'index_1d' else 'instrument'
        frame = store.load_state(state, table, filters=[(column, 'in', indices)])
        counts[table] = {s: int(frame[column].eq(s).sum()) if column in frame else 0 for s in indices}
    instruments = store.load_state(state, 'instruments')
    counts['instrument_kind'] = instruments.kind.value_counts().to_dict()
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']), 'rows': counts,
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'not_a_backtest': True, 'limits': ['Daily index rows do not establish half-hour data',
            'Stock five-minute bars do not establish one-minute touch ordering or fill queues',
            'Historical ETF/LOF/mmf membership, money/adjustment/status and execution coverage remain unproved']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = ast.parse(text)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True,
            'function_ast': {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Existing current fund list and limited historical minute interfaces; no substitute universe or ledger'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch53 three full dynamic fund/halfhour/minute sources frozen')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json'); require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy'])); tree = ast.parse(text)
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and row['function_ast'] ==
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}, 'AST evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, names, ns):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Selected source changed')
    tree = ast.parse(read_source(Path(row['source_copy']))[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Selected functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-original-function>', 'exec'), ns)


def diagnostics(directory):
    validate_sources(directory); rows = []; log = SimpleNamespace(info=lambda *a: None)
    ns = {'pd': pd, 'talib': talib, 'g': SimpleNamespace(), 'log': log,
        'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, type='synthetic', start_date='2026-09-28')}
    selected(directory, 0, ['set_params', 'choice_etfs', 'choice_mmf', 'get_signal', 'ETFtrade', 'EmotionMonitor'], ns)
    ns['set_params'](); context = SimpleNamespace(previous_date=datetime.date(2026, 9, 29))
    codes = list('ABCDEFGH'); volumes = pd.DataFrame({s: [float(k+1)]*15 for k, s in enumerate(codes)})
    volumes.loc[0, 'H'] = np.nan
    ns['get_all_securities'] = lambda *a, **kw: pd.DataFrame(index=codes)
    ns['history'] = lambda *a, **kw: volumes.copy(); ns['choice_etfs'](context)
    require(ns['g'].ETFList == list('GFEDCB'), 'Original fund volume selection differs')
    rows.append({'case': 'fund_top6_drop_missing_no_separate_listing_age_filter', 'selected': ns['g'].ETFList,
        'synthetic_info_start_date': '2026-09-28', 'mock_history_rows': 15})
    ns['get_all_securities'] = lambda *a, **kw: pd.DataFrame({'type': ['mmf', 'mmf', 'mmf', 'etf']}, index=list('ABCD'))
    money = pd.DataFrame({'A': [1e7]*15, 'B': [1e7+1]*15, 'C': [2e7]*15})
    closes = pd.DataFrame({'B': np.linspace(100, 102, 15), 'C': np.linspace(100, 101, 15)})
    ns['history'] = lambda *a, **kw: money.copy() if kw['field'] == 'money' else closes.copy()
    ns['choice_mmf'](context); require(ns['g'].empty_keep_stock == 'B', 'Original money-fund threshold/ranking differs')
    rows.append({'case': 'mmf_money_strict_threshold_and_return_winner', 'selected': 'B', 'equal_10m_excluded': 'A'})
    original_emotion = ns['EmotionMonitor']
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'volume': np.linspace(100., 80., 13)}, index=pd.date_range('2026-09-01', periods=13))
    try: original_emotion()
    except KeyError as exc: rows.append({'case': 'emotion_negative_label_index_in_current_pandas', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected original negative-label index failure')
    ns['attribute_history'] = lambda *a, **kw: {'volume': np.linspace(100., 80., 13)}
    require(original_emotion() == -1, 'Original six-negative MA7 arithmetic differs')
    rows.append({'case': 'emotion_array_arithmetic_six_negative_ma7', 'result': -1, 'synthetic_shape_is_not_platform_equivalence': True})
    ns['g'].ETFList = ['A']; ns['g'].empty_keep_stock = ''; ns['EmotionMonitor'] = lambda: 0
    ns['attribute_history'] = lambda *a, **kw: {'close': np.ones(13)*100}
    ns['get_current_data'] = lambda: {'A': SimpleNamespace(last_price=101.)}
    try: ns['get_signal'](context)
    except AttributeError as exc: rows.append({'case': 'original_dataframe_append_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected original append compatibility failure')
    for immediate in (False, True):
        orders = []; portfolio = SimpleNamespace(positions={}, available_cash=1000., returns=0., total_value=1000.)
        ns['g'].signal = 'BUY'; ns['g'].buy = ['A', 'B']
        def buy(s, value):
            orders.append([s, value])
            if immediate: portfolio.available_cash = 0.
        ns['order_value'] = buy; ns['order_target'] = lambda *a: None
        ns['ETFtrade'](SimpleNamespace(portfolio=portfolio))
        require(len(orders) == (1 if immediate else 2), 'Original sequential cash visibility differs')
        rows.append({'case': 'fund_sequential_cash_visibility', 'mock_cash_update': immediate, 'mock_orders': orders})
    half = {'pd': pd, 'np': np, 'get_price': lambda *a, **kw: pd.DataFrame({'close': [100.]}, index=pd.to_datetime(['2020-04-24']))}
    selected(directory, 1, ['preprocessing'], half)
    try: half['preprocessing']('000300.XSHG', '2014-01-01', '2020-04-24')
    except AttributeError as exc: rows.append({'case': 'halfhour_original_append_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected half-hour append compatibility failure')
    drawn = []; plot = SimpleNamespace(figure=lambda *a, **kw: None, title=lambda *a: None,
        plot=lambda values, **kw: drawn.append(np.asarray(values).tolist()), show=lambda: None)
    half.update(plt=plot, get_security_info=lambda *a: SimpleNamespace(display_name='synthetic'))
    selected(directory, 1, ['plot_cum'], half)
    times = pd.to_datetime(['2020-04-24 09:30', '2020-04-24 10:00', '2020-04-24 11:30', '2020-04-24 13:30', '2020-04-24 15:00'])
    daily = pd.Series(150./np.array([100., 105., 110., 130., 150.])-1, index=times)
    ret = pd.DataFrame({'ret': [.05, 130/110-1], 'frequency': ['signal', 'target']}, index=times[[1, 3]])
    half['plot_cum'](ret, daily, '000300.XSHG', ('signal', 'target'))
    require(len(drawn) == 2 and np.allclose(drawn[0], [150/110]), 'Original end-of-day return alignment differs')
    rows.append({'case': 'halfhour_target_uses_previous_slot_to_day_end_return', 'original_arithmetic_curve': drawn[0],
        'half_hour_end_to_start_ratio': 130/110, 'day_end_to_start_ratio': 150/110, 'synthetic_only': True})
    tree = ast.parse(read_source(Path(read(directory / 'source-reviews/review.json')['sources'][1]['source_copy']))[0])
    annual = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_CalAnnReturns')
    exec(compile(ast.Module(body=[annual], type_ignores=[]), '<original-annual-helper>', 'exec'), half)
    value = float(half['_CalAnnReturns'](np.array([.1, .2])))
    require(np.isclose(value, 1.2**125-1), 'Original annual first-return exclusion differs')
    rows.append({'case': 'halfhour_annual_divides_by_first_cumulative_value', 'synthetic_returns': [.1, .2], 'original_value': value})
    touch = {'np': np, 'g': SimpleNamespace(), 'log': log, 'datetime': datetime}
    selected(directory, 2, ['zt', 'before_selling', 'after_selling', 'filter_specials', 'buy', 'sell', 'Screening_stock_purchases'], touch)
    prices = {s: SimpleNamespace(high_limit=10., last_price=9., day_open=9., low_limit=8., is_st=False, paused=False, name=s) for s in 'ABCD'}
    highs = {s: np.ones(120)*9 for s in 'ABCD'}
    for s, k in zip('ABCD', (0, 58, 59, 119), strict=True): highs[s][k] = 10.
    calls = []; touch['get_current_data'] = lambda: prices
    def history(*a, **kw): calls.append({'args': list(a), 'kwargs': kw}); return highs
    touch['history'] = history; found = touch['zt'](None, list('ABCD'))
    before = touch['before_selling'](None, found, 60, touch['g'].limit_lists)
    after = touch['after_selling'](None, found, 60, touch['g'].limit_lists)
    require(before == [['A', 1], ['B', 59]] and after == [['C', 60], ['D', 120]] and calls[0]['kwargs']['unit'] == '1m', 'Original touch split differs')
    rows.append({'case': 'first_120_one_minute_high_touch_and_60_boundary', 'before': before, 'after': after, 'mock_request': calls[0]})
    highs['D'][0] = 10.01
    require(touch['zt'](None, ['D']) == [], 'Original strict maximum-equality differs')
    rows.append({'case': 'maximum_above_limit_excluded_despite_exact_touch', 'mock_high_max': 10.01})
    require(touch['filter_specials'](None, ['A']) == ['A'], 'Original active filter differs')
    rows.append({'case': 'active_filter_has_no_listing_date_query', 'selected': ['A'], 'listing_date_input': None})
    now = datetime.datetime(2026, 9, 29, 11, 30); portfolio = SimpleNamespace(total_value=10000., cash=10000.)
    context = SimpleNamespace(current_dt=now, portfolio=portfolio)
    intents = []; touch['LimitOrderStyle'] = lambda p: {'limit': p}
    touch['order_value'] = lambda *a: intents.append({'callback_time': str(context.current_dt), 'args': list(a)})
    touch['get_security_info'] = lambda s: SimpleNamespace(display_name=s)
    touch['get_all_securities'] = lambda: pd.DataFrame(index=list('ABC'))
    touch['filter_specials'] = lambda c, s: s
    touch['No_limit_for_three_days'] = lambda c, s: s; touch['attracting_funds'] = lambda c, s: s
    touch['g'] = SimpleNamespace(open_time=now.replace(hour=9, minute=30), minute=60, buy_stock_count=10, open_positions=0, number_of_positions=0)
    touch['Screening_stock_purchases'](context)
    require(touch['g'].number_of_positions == 3 and len(intents) == 3 and all(r['callback_time'].endswith('11:30:00') for r in intents), 'Original rejected-intent count/time differs')
    rows.append({'case': 'all_orders_submitted_at_callback_and_count_updates_without_fill', 'mock_order_return': None,
        'counter': touch['g'].number_of_positions, 'mock_orders': intents, 'historical_log_time_does_not_change_order_time': True})
    pos = SimpleNamespace(security='A', closeable_amount=100)
    portfolio.positions = {'A': SimpleNamespace(total_amount=100)}; portfolio.long_positions = {'A': pos}
    context.subportfolios = [SimpleNamespace(long_positions={'A': SimpleNamespace(acc_avg_cost=9.)})]
    sales = []; touch['order_target_value'] = lambda *a: sales.append(list(a)); touch['g'].number_of_positions = 0
    touch['sell'](context)
    require(touch['g'].number_of_positions == -1 and len(sales) == 1, 'Original sell counter without fill differs')
    rows.append({'case': 'rejected_sell_intent_can_decrement_counter_below_zero', 'counter': -1, 'mock_orders': sales})
    schedule = []; touch.update(unschedule_all=lambda: None, run_daily=lambda fn, time: schedule.append([fn.__name__, time]))
    selected(directory, 2, ['before_open', 'after_code_changed'], touch); touch['after_code_changed'](None)
    require([r[1] for r in schedule] == ['9:30', '11:30', '10:30'], 'Original callback registration differs')
    rows.append({'case': 'callbacks_registered_only_by_after_code_changed', 'schedules': schedule})
    return {'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Synthetic history/cash/price/plot functions diagnose original code, not orders or trades in the unique ledger',
            'Original compatibility failures retained; no economic rule or unavailable data is silently repaired']}


def study(root, directory):
    binding(root, directory); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch53 original selection/compatibility/callback/intended-order diagnoses archived; no backtest')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); name, params = QUERIES[endpoint]
    row = {'endpoint': endpoint, 'function': name, 'parameters': params, 'status': 'failed', 'files': [], 'wire_responses': [],
        'published': False, 'strict_usable': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}; original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire_responses']) < 8, 'Request cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(row["wire_responses"]):03d}.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API changed')
        import akshare as ak
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, name)(**params)
        path = Path(root) / 'raw/intraday_probe_batch53' / endpoint / f'{directory.name}.parquet'; require(not path.exists(), 'Raw exists')
        path = raw.save(root, 'intraday_probe_batch53', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API evidence changed')
    rows = read(directory / 'probe-results.json')['results']; require([r['endpoint'] for r in rows] == list(QUERIES), 'Probe scope changed')
    for row in rows:
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['published'] is False and row['strict_usable'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['function'] == QUERIES[row['endpoint']][0] and
            row['parameters'] == QUERIES[row['endpoint']][1] and row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Probe binding changed')
        for item in row['wire_responses']+row['files']: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status']=='success'), 'Probe raw profile differs')
        else: require(not row['files'] and row.get('error'), 'Failed probe raw/error differs')
    return rows


def probe(root, directory):
    rows = []; logs = directory / 'logs'; logs.mkdir()
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch53', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline45s', 'published': False, 'strict_usable': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch53 existing current-fund/halfhour-index/one-minute-stock supplement attempts archived')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch53 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); dependency = inventory(root); catalog = offline_catalog(root, directory)
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and dependency == read(directory / 'dependency-inventory.json'), 'Offline outputs changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch53 forbidden-network original diagnostics and corpus catalog match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch53.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_catalog_fund_universe.py',
         'tests/unit/test_catalog_schedules.py', 'tests/unit/test_catalog_research_queries.py', 'tests/unit/test_catalog_dependencies.py',
         'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); validate_probes(directory)
    before = implementation(); rows = []
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json'), 'Offline proof changed')
    validate_probes(directory); require(inventory(root) == read(directory / 'dependency-inventory.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed research changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch53 three source reviews accepted; fund and intraday originals remain blocked')
    save(Path('docs/handoff/2026-10-07-batch53-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 3})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch53/20261007-fund-intraday')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
