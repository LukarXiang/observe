"""Archive exact bluechip/TRIX/RSI/BOLL rules; missing trading inputs stay blocked."""
import argparse
import ast
from contextlib import redirect_stdout
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
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch31 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch31/20261006-capm-weekly-rsrs-finance')
RECEIPT = Path('docs/handoff/2026-10-06-batch31-verification.json')
SOURCES = ('2020年度精选策略/70 蓝筹&均线.txt', '2020年度精选策略/42 低估值+TRIX+RSI 低回撤策略.txt',
    '2024年度精选策略2/26.近几年一直有效的股票BOLL择时策略.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/bollinger.py'), ('repo/backtrader', 'backtrader/indicators/trix.py'),
    ('repo/baostock', 'baostock/demo/demo_profit_data.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch32.py', 'tests/unit/test_batch32_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'price-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json', 'component-offline.json',
    'offline-verification.json', 'offline-catalog.json'}
QUERIES = {'balance': {'provider': 'baostock', 'function': 'query_balance_data', 'parameters': {'code': 'sh.600519', 'year': 2020, 'quarter': 1},
        'limit': 'Quarterly solvency ratios are not original whole-pool balance fields or historical revisions'},
    'index': {'provider': 'baostock', 'function': 'query_history_k_data_plus', 'parameters': {'code': 'sz.399101',
        'fields': 'date,code,open,high,low,close,volume,amount', 'start_date': '2020-01-02', 'end_date': '2020-01-10', 'frequency': 'd', 'adjustflag': '3'},
        'limit': 'Index prices do not supply historical 399101 membership or circulating-cap ranks'},
    'minute': {'provider': 'akshare', 'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600519', 'period': '1',
        'start_date': '2020-01-02 09:30:00', 'end_date': '2020-01-02 15:00:00', 'adjust': ''},
        'limit': 'Latest-five-day endpoint cannot establish 2020 09:35/14:50 historical execution'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch31 differs')
    files = {}
    for name in ('price-input.parquet', 'baseline.json', 'input-binding.json'):
        path = ACCEPTED / name; require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
        files[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT), 'files': files,
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    import baostock as bs
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(bs if query['provider'] == 'baostock' else ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    checkpoint(root, directory, 'Batch32 bluechip/TRIX-RSI/mixed-window BOLL audit started')
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old_path = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path), 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'BOLL/TRIX formula comparison and quarterly finance API only; not platform substitutions'})
    search = ['rg', '-n', r'^def (MA_judge_duotou|financial_data_filter_dayu|order_style|judge_security_max_proportion|max_buy_value_or_amount|sell_by_amount_or_percent_or_none)', 'repo', '-g', '*.py', '-g', '*.txt']
    result = subprocess.run(search, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search': {'command': search, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch32 three complete original rules frozen; wizard/financial/pool/intraday gaps retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def prepare(root, directory):
    binding(root, directory); validate_sources(directory); path = directory / 'price-input.parquet'
    require(not path.exists() and not (directory / 'input-analysis.json').exists(), 'Input archive exists')
    shutil.copyfile(ACCEPTED / path.name, path); frame = pd.read_parquet(path)
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
        'first': frame.date.min(), 'last': frame.date.max(), 'pool': list(previous.stock_binding.INSTRUMENTS),
        'not_a_backtest': True, 'platform_equivalent': False,
        'policy': 'Only verified traded rows;25-row BOLL component uses decision-end dynamic price scale and explicit negative integer labels for original positional samples',
        'limits': ['Ten stocks are not original HS300/399101 pools', 'No original wizard, TRIX/RSI platform seed, circulating-cap rank or minute execution supplied']})
    return archive(root, 'Batch32 accepted ten-stock long prices frozen; no original universe replacement')


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); path = directory / 'price-input.parquet'
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and doc['platform_equivalent'] is False and
        doc['pool'] == list(previous.stock_binding.INSTRUMENTS) and doc['file'] == str(path) and file_sha(path) == doc['sha256'], 'Input binding changed')
    frame = pd.read_parquet(path)
    require(len(frame) == doc['rows'] and frame.date.min() == doc['first'] and frame.date.max() == doc['last'] and
        set(frame.instrument) == set(doc['pool']) and not frame[['date', 'instrument']].isna().any().any() and
        not frame.duplicated(['date', 'instrument']).any(), 'Input profile/universe differs')
    return frame


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    lines = read_source(Path(row['source_copy']))[0].splitlines(keepends=True)
    starts = [k for k, line in enumerate(lines) if line.startswith(('import ', 'from '))]
    require(starts, 'Original import/code start missing')
    return ast.parse(''.join(lines[starts[0]:]))


def boll_kernel(directory):
    cls = next(n for n in source_tree(directory, 2).body if isinstance(n, ast.ClassDef) and n.name == 'Boll399101Strat')
    period = [n.value for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'N' for t in n.targets)]
    require(len(period) == 1 and isinstance(period[0], ast.Constant) and period[0].value == 20, 'Original BOLL period differs')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_stock_list')
    loops = [n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stock']
    require(len(loops) == 1 and len(loops[0].body) == 11, 'Original BOLL loop differs')
    first = loops[0].body[0]
    require(isinstance(first, ast.Assign) and first.targets[0].id == 'data' and isinstance(first.value, ast.Attribute) and
        isinstance(first.value.value, ast.Call) and first.value.value.func.id == 'attribute_history', 'Original BOLL request differs')
    module = ast.Module(body=loops[0].body[1:], type_ignores=[])
    return compile(module, '<original-boll-loop>', 'exec'), hashlib.sha256(ast.dump(loops[0]).encode()).hexdigest()


def boll_case(code, values, held=False, loss=None, positional=True):
    series = pd.Series(np.asarray(values, dtype=float), index=range(-len(values), 0) if positional else None)
    ns = {'data': series, 'self': SimpleNamespace(N=20), 'stock': 'component', 'context': object(), 'get_pos_amount': lambda *a: int(held),
        'g': SimpleNamespace(loss_price={} if loss is None else {'component': loss}), 'final_list': []}
    exec(code, ns)
    return ns


def reference_boll(values):
    x = list(map(float, values)); require(len(x) == 25 and np.isfinite(x).all() and min(x) > 0, 'Invalid BOLL window')
    mean = math.fsum(x) / 25; last = x[-20:]; sample_mean = math.fsum(last) / 20
    std = math.sqrt(math.fsum((v - sample_mean) ** 2 for v in last) / 19); low = min(x[-10:-2])
    return [mean, std, mean + 2 * std, mean - 2 * std, low], x[-2] < mean - 2 * std and x[-1] > x[-2] and x[-1] > low


def compute(directory):
    validate_sources(directory); frame = inputs(directory); code, sha = boll_kernel(directory); groups = []; boundaries = []; error = np.zeros(5)
    for stock, history in frame.groupby('instrument', sort=True):
        history = history.sort_values('date'); paused = int((~history.is_trading).sum()); history = history[history.is_trading].reset_index(drop=True)
        require(history.adjustment_status.eq('usable').all() and np.isfinite(history[['close_adj', 'back_factor']]).all().all() and
            history[['close_adj', 'back_factor']].gt(0).all().all(), 'Unknown adjusted price')
        digest = hashlib.sha256(); count = entries = 0
        for k in range(24, len(history)):
            values = history.close_adj.iloc[k - 24:k + 1].to_numpy() / history.back_factor.iloc[k]
            ns = boll_case(code, values); actual = [float(ns[n]) for n in ('meandelta', 'stddev', 'upperbound', 'lowerbound', 'prev_low')]
            expected, entry = reference_boll(values); error = np.maximum(error, np.abs(np.asarray(actual) - expected) / np.maximum(np.abs(expected), 1.))
            selected_entry = bool(ns['final_list']); entries += int(selected_entry); count += 1
            if selected_entry != entry: boundaries.append({'stock': stock, 'date': history.date.iloc[k], 'original': selected_entry, 'reference': entry})
            digest.update(json.dumps({'date': history.date.iloc[k], 'values': actual, 'entry': selected_entry}, sort_keys=True).encode())
        groups.append({'stock': stock, 'windows': count, 'entry_conditions': entries, 'known_paused_rows_excluded': paused, 'sha256': digest.hexdigest()})
    require(error.max() < 1e-10, 'BOLL numeric mismatch')
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'original_strategy_complete': False, 'strategy_results': [],
        'input_sha256': file_sha(directory / 'price-input.parquet'), 'backend': {'numpy': np.__version__, 'pandas': pd.__version__},
        'boll': {'source_ast_sha256': sha, 'groups': groups, 'comparison_boundaries': boundaries,
            'max_relative_differences': dict(zip(('mean25', 'sample_std20', 'upper', 'lower', 'low8'), error.tolist()))},
        'limits': ['Entry conditions only; no full original universe/cash/order/stop-map continuity/NAV',
            'Negative integer labels explicitly adapt legacy positional samples; dated-Series compatibility remains blocked',
            'Unknown wizard MA and platform TRIX/RSI seeds not replaced by TA-Lib']}


def trix_namespace(directory, trix=2., matrix=1., rsi=60.):
    calls = []; ns = {'g': SimpleNamespace(date='2020-01-02')}
    names = {'MAX_OWN_NUM', 'CASH_SP_COUNT', 'CASH_MAX_USE', 'CASH_MAX_USE_PERSTOCK'}
    constants = {n.targets[0].id: n.value.value for n in source_tree(directory, 1).body if isinstance(n, ast.Assign) and
        isinstance(n.targets[0], ast.Name) and n.targets[0].id in names and isinstance(n.value, ast.Constant)}
    require(set(constants) == names, 'Original cash/slot constants missing'); ns.update(constants)
    def indicator(name, values, stock, day, **kw):
        calls.append({'indicator': name, 'date': day, 'parameters': kw})
        return tuple({stock: v} for v in values) if len(values) > 1 else {stock: values[0]}
    ns.update(TRIX=lambda stock, day, **kw: indicator('TRIX', [trix, matrix], stock, day, **kw),
        RSI=lambda stock, day, **kw: indicator('RSI', [rsi], stock, day, **kw))
    selected(directory, 1, ('bei_li', 'RS', 'btj1', 'btj2', 'stj1', 'Check_Stocks', 'buy_pd', 'tdbuy', 'sell_pd', 'tdsell', 'ji_lu'), ns)
    return ns, calls


def diagnostics(directory):
    cases = []; ns = {'pd': pd, 'g': SimpleNamespace()}
    selected(directory, 0, ('check_stocks', 'check_stocks_sort', 'get_security_universe', 'holded_filter', 'buy', 'sell', 'trade', 'technical_indicators_filter',
        'financial_statements_filter', 'buy_initialize', 'sell_initialize', 'check_stocks_initialize', 'check_stocks_sort_initialize', 'risk_management_initialize'), ns)
    for fn in ('buy_initialize', 'sell_initialize', 'check_stocks_initialize', 'check_stocks_sort_initialize', 'risk_management_initialize'): ns[fn]()
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'held': object()}, available_cash=10000.))
    cases.append({'case': 'bluechip_false_filter_holded_excludes_existing', 'stocks': ns['holded_filter'](ctx, ['held', 'new'])})
    cases.append({'case': 'bluechip_empty_sort_preserves_order', 'stocks': ns['check_stocks_sort'](ctx, ['z', 'a'], {}, 'desc')})
    ns['get_index_stocks'] = lambda *a: ['b', 'a', 'b']; cases.append({'case': 'bluechip_pool_sorted_deduped', 'stocks': ns['get_security_universe'](ctx, ['000300.XSHG'], [])})
    try: ns['technical_indicators_filter'](ctx, ['candidate'])
    except NameError as exc: cases.append({'case': 'bluechip_missing_wizard_ma', 'error': str(exc)})
    else: raise AssertionError('Unknown wizard helper unexpectedly supplied')
    ns['g'].max_hold_stocknum = 5; calls = []
    ns.update(order_style=lambda *a: calls.append(list(a[1])) or {s: 100. for s in a[1]}, judge_security_max_proportion=lambda *a: a[2],
        max_buy_value_or_amount=lambda *a: a[1], order=lambda *a: calls.append('order'), MarketOrderStyle=lambda: None)
    ctx.portfolio.positions = {f'h{k}': object() for k in range(6)}; ns['buy'](ctx, ['a', 'b', 'c'])
    cases.append({'case': 'bluechip_negative_slots_slice_without_orders', 'allocation_lists': calls})
    stock_list = ['a', 'b', 'c']; table = pd.DataFrame({'code': stock_list, 'total_assets': [100., 100., 100.], 'total_liability': [10., 20., 30.]})
    fn = ast.parse(previous.fragment(directory, 1, 'Check_Stocks')).body[0]
    tail = ast.Module(body=fn.body[2:-1], type_ignores=[]); local = {'Stocks': table.copy()}; exec(compile(tail, '<original-debt-filter>', 'exec'), local)
    cases.append({'case': 'trix_financial_selects_above_median_debt', 'stocks': list(local['Code']), 'median': float(local['me'])})
    for value in (14.999, 15., 20., 20.001, 49.999, 50., 55., 55.001, 79.999, 80., 85., 85.001, np.nan):
        local, _ = trix_namespace(directory, rsi=value)
        cases.append({'case': 'rsi_strict_threshold', 'rsi': float(value) if np.isfinite(value) else 'nan', 'signal': local['RS']('component')})
    local, requests_made = trix_namespace(directory); cases.append({'case': 'trix_two_buy_tests_repeat_both_indicators',
        'buy': bool(local['btj1']('component') and local['btj2']('component')), 'calls': requests_made})
    local, _ = trix_namespace(directory, trix=1., matrix=1., rsi=60.); cases.append({'case': 'trix_equal_is_neutral', 'signal': local['bei_li']('component')})
    local, _ = trix_namespace(directory); local['g'].tdbuy = ['a']; orders = []
    local['order_value'] = lambda *a: orders.append(list(a)); ctx.portfolio.available_cash = 100000.; local['tdbuy'](ctx)
    cases.append({'case': 'trix_single_candidate_uses_half_budget', 'orders': orders})
    local['g'].tdsell = ['a']; local['get_orders'] = lambda: {'partial': SimpleNamespace(action='close', filled=1, security='a', price=10.)}
    local['g'].HighAfterEntry = {'a': 12.}; local['log'] = SimpleNamespace(info=lambda *a: None); local['ji_lu'](ctx)
    cases.append({'case': 'trix_partial_sell_deletes_high_state', 'remaining': local['g'].HighAfterEntry})
    code, _ = boll_kernel(directory); values = np.arange(25.) + 100
    try: boll_case(code, values, positional=False)
    except KeyError as exc: cases.append({'case': 'boll_legacy_rangeindex_fails', 'error': str(exc)})
    else: raise AssertionError('Legacy negative RangeIndex unexpectedly accepted')
    try: boll_case(code, values, held=True)
    except KeyError as exc: cases.append({'case': 'boll_held_without_loss_state_fails', 'error': str(exc)})
    else: raise AssertionError('Missing stored loss state unexpectedly accepted')
    for name, loss in (('equal_stop_keeps', values[-1]), ('strict_stop_exits', values[-1] + .01)):
        local = boll_case(code, values, held=True, loss=loss); cases.append({'case': f'boll_{name}', 'kept': bool(local['final_list'])})
    profit = np.full(25, 100.); profit[-3:] = [140., 130., 125.]
    local = boll_case(code, profit, held=True, loss=90.)
    cases.append({'case': 'boll_current_upper_used_for_past_close_profit_exit', 'kept': bool(local['final_list']), 'upper': float(local['upperbound'])})
    ns2 = {'g': SimpleNamespace(max_cash=200000, limit_price=[1, 100]), 'log': SimpleNamespace(info=lambda *a: None, debug=lambda *a: None)}
    selected(directory, 2, ('adjust_position', 'get_pos_amount', 'close_position', 'open_position', 'order_target_value_', 'filter_price_stocks'), ns2)
    prices = {'held': 101., 'new_high': 101., 'too_low': .99, 'one': 1., 'hundred': 100.}
    ns2['get_current_data'] = lambda: {s: SimpleNamespace(last_price=p) for s, p in prices.items()}
    ctx.portfolio.positions = {'held': SimpleNamespace(total_amount=100, security='held')}
    cases.append({'case': 'boll_price_filter_asymmetric_held_exception', 'stocks': ns2['filter_price_stocks'](ctx, list(prices), 20)})
    candidate = np.full(25, 100.); candidate[17] = 90.; candidate[-2:] = [85., 90.5]
    local = boll_case(code, candidate); ctx.portfolio.positions = {}
    ns2['get_current_data'] = lambda: {'component': SimpleNamespace(last_price=101.)}
    accepted = ns2['filter_price_stocks'](ctx, local['final_list'], 20)
    cases.append({'case': 'boll_filtered_candidate_retains_pretrade_loss', 'selected': local['final_list'], 'accepted': accepted, 'loss_state': local['g'].loss_price})
    for name, cash, price, positions, targets in (('cash_rounding', 10099., 10., {}, ['a', 'b']),
        ('share110_equal_allowed', 1100., 10., {}, ['a']), ('share_below110_skips', 1099., 10., {}, ['a']),
        ('rejected_sell_occupies_slot', 10000., 10., {'old': SimpleNamespace(security='old', total_amount=100)}, ['a'])):
        orders = []; ctx.portfolio.positions = positions; ctx.portfolio.cash = cash
        ns2.update(order_target_value=lambda *a: orders.append(list(a)), attribute_history=lambda *a, **kw: pd.DataFrame({'close': [price]}))
        ns2['adjust_position'](ctx, targets); cases.append({'case': f'boll_{name}', 'orders': orders})
    return {'not_a_backtest': True, 'cases': cases,
        'limits': ['Stub order intentions and synthetic indicator values only; no fills/cash/NAV implementation',
            'Unknown MA/TRIX platform seeds and source economic rules not repaired']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch32 original mixed BOLL long windows and original financial/technical/order diagnostics frozen')


def worker(root, directory, endpoint):
    import akshare as ak
    import baostock as bs
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API differs')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(8)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kw):
        require(len(row['wire']) < 10, 'Response cap exceeded'); session.trust_env = False; kw['timeout'] = (8, 10)
        response = original(session, method, url, **kw); path = folder / f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status': response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()):
            if query['provider'] == 'baostock':
                with BaoStock(root).session():
                    result = getattr(bs, query['function'])(**query['parameters']); values = []
                    require(result.error_code == '0', f'Provider error: {result.error_code}/{result.error_msg}')
                    while result.next(): values.append(result.get_row_data()); require(len(values) <= 100, 'Provider row cap exceeded')
                    frame = pd.DataFrame(values, columns=result.fields)
            else: frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch32_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}],
            limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['published'] is False, 'Probe binding differs')
        for item in [*row['files'], *row['wire']]: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] == 'sample':
            require(row['strict_usable'] is False and row['limit'] == query['limit'] and len(row['files']) == 1, 'Unproven sample admitted')
            item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(0 < len(frame) == item['rows'] <= 100000 and list(frame.columns) == item['columns'], 'Sample profile differs')
        else: require(row['status'] in ('failed', 'timeout') and not row['files'] and row.get('error'), 'Failed probe admitted data')
        rows.append(row)
    return rows


def probe(root, directory):
    binding(root, directory)
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch32', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch32 historical solvency/index/minute supplementation attempts archived; no publication')


def offline_catalog(root, directory):
    proof = Path(root) / 'catalog/strategies/implementation-evidence.json'
    require(file_sha(proof) == file_sha(directory / 'implementation-evidence.before.json'), 'Implementation registry changed')
    folder = directory / 'catalog-offline'; folder.mkdir(exist_ok=False); copied = folder / 'catalog/strategies/implementation-evidence.json'
    copied.parent.mkdir(parents=True); shutil.copyfile(proof, copied); result = catalog_strategies(folder, 'repo/量化策略源代码')
    original = read(directory / 'source-reviews/review.json')['catalog']; require(result['catalog_id'] == original['catalog_id'], 'Offline catalog identity differs')
    rows = []
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        a = Path(original['output']) / name; b = Path(result['output']) / name; require(file_sha(a) == file_sha(b), 'Offline catalog bytes differ')
        rows.append({'name': name, 'original_file': str(a), 'repeated_file': str(b), 'sha256': file_sha(a)})
    return {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'catalog_id': result['catalog_id'],
        'files': rows, 'implementation_evidence_sha256': file_sha(proof)}


def validate_catalog(root, directory):
    doc = read(directory / 'offline-catalog.json'); original = read(directory / 'source-reviews/review.json')['catalog']
    require(doc['result'] == 'match' and doc['differences'] == 0 and doc['socket_network_disabled'] is True and doc['catalog_id'] == original['catalog_id'] and
        len(doc['files']) == 3 and {r['name'] for r in doc['files']} == {'catalog.json', 'summary.json', 'catalog.parquet'}, 'Offline catalog scope differs')
    require(doc['implementation_evidence_sha256'] == file_sha(Path(root) / 'catalog/strategies/implementation-evidence.json') ==
        file_sha(directory / 'implementation-evidence.before.json'), 'Implementation registry differs')
    for row in doc['files']:
        a = Path(original['output']) / row['name']; b = directory / 'catalog-offline/catalog/strategies' / doc['catalog_id'] / row['name']
        require(row['original_file'] == str(a) and row['repeated_file'] == str(b) and file_sha(a) == row['sha256'] == file_sha(b), 'Offline catalog bytes/path differ')
    require(read(Path(root) / 'catalog/strategies/latest.json')['catalog_id'] == doc['catalog_id'], 'Latest catalog differs')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch32 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch32 forbidden-network components/diagnostics/catalog match byte-for-byte')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch32.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch32_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch17_archive.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked code changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True); require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory); require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch32 bluechip/TRIX-RSI/BOLL rules and component evidence accepted; missing original trading inputs remain blocked')
    save(Path('docs/handoff/2026-10-06-batch32-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch32/20261006-bluechip-trix-boll')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
