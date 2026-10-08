"""Archive diffusion/leader original functions and true-price arithmetic, not trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
from decimal import Decimal
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
import warnings

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch52 import dependency_state
from scripts.review_strategy_batch58 import definitions, digest
from scripts import review_strategy_batch60 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch60/20261007-commodity-turtle-MA')
RECEIPT = Path('docs/handoff/2026-10-07-batch60-verification.json')
SOURCES = ('2021年度精选策略/61.【复现】扩散指标择时.txt',
    '2021年度精选策略/56.大盘一路向上时，追总龙头2个月3倍.txt')
SOURCE_SHA = ('0b8bd5bc5814d4831f3fa86b534624cddd2c4552d082296eb0f38cc15b6c8ef9',
    'd0e0dff6d879a1a27cb5e5d85324b35b8d283bf4b414f13da0279ac6e3b1a93d')
OPERAND = Path('data/staging/strategies-batch47/20261007-candidate-momentum/price-input.parquet')
OPERAND_SHA = '3f564aaaebb2ef1a303c3286287965e239a7e2337608b5675418f1df100def5b'
CALENDAR = OPERAND.with_name('calendar-input.parquet')
CALENDAR_SHA = 'b29ce632824660c74ef93d32701ff010cf2626e2007718e2b921b3464cc767f5'
REFERENCES = (('repo/backtrader', 'backtrader/indicators/momentum.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'), ('repo/akshare', 'akshare/stock/stock_industry.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch61.py', 'tests/unit/test_batch61_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'diagnostic-failure-binding.json', 'start-failure-binding.json', 'component-failure-binding.json',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json', 'supplement-decision.json',
    'price-input.parquet', 'calendar-input.parquet', 'calendar-binding.json', 'component-research.json', 'diagnostics.json', 'component-offline.json',
    'diagnostics-offline.json', 'offline-catalog.json', 'offline-verification.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch60.py') == proof['final_script_sha256'], 'Batch60 changed')
    accepted = read('docs/handoff/2026-10-07-batch47-verification.json')
    require(file_sha(OPERAND) == OPERAND_SHA == accepted['checks']['evidence_sha256']['price-input.parquet'], 'Accepted price changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'), 'upstream': previous.binding(root, ACCEPTED),
        'price_file': str(OPERAND), 'price_sha256': OPERAND_SHA, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch61 diffusion/leader original-source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    import baostock as bs
    rows = []
    for fn in (bs.query_hs300_stocks, ak.fund_etf_hist_em, ak.stock_zh_index_daily_em,
            ak.stock_board_industry_cons_em, ak.stock_zh_a_hist_min_em):
        code = inspect.getsource(fn)
        rows.append({'function': fn.__name__, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'apis': rows,
        'limits': ['Vendor weekly dated members are not proved daily platform membership',
            'Current industry members do not supply historical HY001-HY011 pools/names',
            'ETF/index daily history cannot replace leader minute execution or historical circulating-cap versions']}


def supplements():
    records = []
    for batch, folder in ((42, 'weekly-trend-dqn'), (53, 'fund-intraday'), (54, 'breadth-forecast-bond')):
        path = Path(f'data/staging/strategies-batch{batch}/20261007-{folder}/probe-results.json')
        receipt = read(f'docs/handoff/2026-10-07-batch{batch}-verification.json')
        require(file_sha(path) == receipt['checks']['evidence_sha256']['probe-results.json'], 'Accepted supplement changed')
        rows = read(path)['results']
        require(all(r['status'] == 'failed' and not r['files'] and not r['published'] for r in rows), 'Prior failure changed')
        records.append({'file': str(path), 'sha256': file_sha(path), 'results': rows})
    return {'accepted_failed_probes': records, 'new_requests': 0, 'new_samples': 0, 'published': False, 'not_a_backtest': True,
        'decision': 'Reuse dated HS300 login failure and verified same-host Eastmoney DNS failures; no current industry substitution',
        'missing': ['Daily historical HS300 members and circulating-cap visible versions and all-member price endpoints',
            '510300 start/event/status/price/execution evidence', 'Historical HY001-HY011 pools/names/listing/status/high_limit',
            'Original Shanghai000001 three-day close and leader minute prices/fills/limit queues'],
        'limits': ['A prior failed request is evidence of that attempt, not proof data does not exist at the provider',
            'Changing symbols or parser on a DNS-failing host does not resolve name resolution']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); profiles = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'index_1d', 'corp_actions', 'adj_factors', 'instrument_status'):
        key = 'index' if table == 'index_1d' else 'instrument'
        frame = store.load_state(state, table, filters=[(key, 'in', ['510300.SH', '000001.SH'])])
        profiles[table] = {s: int(frame[key].eq(s).sum()) if key in frame else 0 for s in ('510300.SH', '000001.SH')}
    members = store.load_state(state, 'index_constituents', filters=[('index', '=', '000300.SH')])
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in rows.values()) for t, rows in state['tables'].items()},
        'original_asset_rows': profiles, 'HS300_members': {'rows': len(members),
            'first': str(members.date.min()) if len(members) else None, 'last': str(members.date.max()) if len(members) else None,
            'strict_usable_rows': int(members.strict_usable.sum()) if len(members) else 0,
            'table_registered': 'index_constituents' in state['tables'],
            'older_handoff_snapshot': '20261003-224421-9d2b',
            'older_handoff_snapshot_present': (Path(root) / 'snapshots/20261003-224421-9d2b.json').exists()},
        'historical_equity_dependencies': dependency_state(root), 'ledger_sha256': file_sha('src/observe/ledger/book.py'),
        'not_a_backtest': True, 'limits': ['Ten-stock numeric operands do not establish full historical HS300/HY universes',
            'ROC endpoints on frozen adjusted values do not prove platform query adjustment equivalence',
            'Historical shares/PCF/names remain required for micro400 and small-value100']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    failure = Path('data/staging/strategies-batch61/20261007-diffusion-leader-diagnostic-failure-01')
    save(directory / 'diagnostic-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    failed_start = Path('data/staging/strategies-batch61/20261007-diffusion-leader')
    save(directory / 'start-failure-binding.json', {p.relative_to(failed_start).as_posix(): {'file': str(p), 'sha256': file_sha(p)}
        for p in sorted(failed_start.rglob('*')) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True, 'definition_ast': definitions(ast.parse(text))})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only ROC period/formula and ETF/industry interface comparison; no alternate ledger or signal substitution'})
    shutil.copyfile(OPERAND, directory / 'price-input.parquet')
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch61 two complete diffusion/leader sources frozen; original trading inputs remain gated')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha and
            row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Source/rule/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
        commit = subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        require(commit == row['commit'], 'Reference commit changed')
    require(file_sha(directory / 'price-input.parquet') == OPERAND_SHA, 'Local operand changed')
    receipt47 = Path('docs/handoff/2026-10-07-batch47-verification.json')
    require(file_sha(CALENDAR) == file_sha(directory / 'calendar-input.parquet') == CALENDAR_SHA ==
        read(receipt47)['checks']['evidence_sha256']['calendar-input.parquet'] and
        read(directory / 'calendar-binding.json') == {'file': str(CALENDAR), 'sha256': CALENDAR_SHA,
            'accepted_receipt_sha256': file_sha(receipt47)}, 'Calendar evidence changed')
    failure = read(directory / 'component-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Component failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Component failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 5 passed' in red['stdout'] and
        "between instances of 'datetime.date' and 'str'" in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Component red regression changed')
    failure = read(directory / 'diagnostic-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure archive changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 3 passed' in red['stdout'] and
        'assert 0 == 2' in red['stdout'] and red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Failure regression changed')
    failure = read(directory / 'start-failure-binding.json')
    require({'start-failure.json', 'start-failed-driver.py', 'start-failed-regression.py', 'baseline.json'} <= failure.keys(), 'Failed-start evidence missing')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failed-start evidence changed')
    red = read(failure['start-failure.json']['file'])
    require(red['status'] == 'failed' and red['returncode'] == 1 and '1 failed' in red['stdout'] and
        'AttributeError' in red['stdout'] and red['driver_sha256'] == failure['start-failed-driver.py']['sha256'] and
        red['test_sha256'] == failure['start-failed-regression.py']['sha256'], 'Failed-start regression changed')


def selected(directory, number, names, ns):
    import __future__
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing selected function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-diffusion-leader-functions>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns


def diagnostics(directory):
    validate_sources(directory); rows = []; quiet = SimpleNamespace(info=lambda *a: None)
    ns = selected(directory, 0, ['GetROC', 'CalROCSingal', 'rolling_average', 'GetSingal', 'StockBuy', 'StockSell', 'before_trading_start'], {'np': np})
    dates = list(pd.bdate_range('2020-01-01', periods=100).date); requests = []
    ns['get_trade_days'] = lambda **kw: dates[-kw['count']:]
    def prices(stocks, **kw):
        requests.append(str(kw['end_date']))
        return pd.DataFrame({'code': stocks, 'close': [20., 0., np.nan, 30.] if kw['end_date'] == dates[0] else [40., 10., 50., 15.]})
    ns['get_price'] = prices
    value = ns['GetROC'](list('ABCD'), dates[-1], 100)
    require(value.to_dict() == {'A': 1., 'D': -.5} and requests == [str(dates[-1]), str(dates[0])], 'Original ROC endpoints differ')
    rows.append({'case': 'ROC100_means_99_calendar_intervals_and_excludes_zero_nan', 'mock_dates': requests, 'ROC': value.to_dict(),
        'intervals': 99, 'filter_ast': ast.dump(ast.parse("last[~last['close'].isna() & last['close'] != 0]"), include_attributes=False),
        'current_float_filter_matches_intended_mask': True})
    ns['GetROC'] = lambda *a: pd.Series({'A': 1., 'B': -.5, 'C': .25})
    ns['get_valuation'] = lambda *a, **kw: pd.DataFrame({'code': list('ABC'), 'circulating_market_cap': [100., 200., 100.]})
    weighted = ns['CalROCSingal'](list('ABC'), dates[-1], 100, 'mktcap'); count = ns['CalROCSingal'](list('ABC'), dates[-1], 100, 'avg')
    require(weighted == .3125 and count == 2, 'Original weighted positive ROC differs')
    rows.append({'case': 'mktcap_is_positive_ROC_magnitude_not_positive_cap_share', 'weighted': weighted, 'avg_count': count, 'positive_cap_share': .5})
    ns['get_valuation'] = lambda *a, **kw: pd.DataFrame({'code': ['A', 'B'], 'circulating_market_cap': [100., 100.]})
    missing = ns['CalROCSingal'](list('ABC'), dates[-1], 100, 'mktcap')
    require(missing == .5, 'Original alignment/skipna differs')
    rows.append({'case': 'missing_positive_valuation_silently_skipped', 'weighted': missing, 'missing_code': 'C'})
    arr = np.arange(120, dtype=float); fast = ns['rolling_average'](arr, 90); slow = ns['rolling_average'](fast, 30)
    require(len(fast) == 31 and len(slow) == 2 and fast[-1] == 74.5 and slow[-1] == 60., 'Original double smoothing differs')
    rows.append({'case': '120_values_yield_31_fast_and_2_slow', 'fast_last': float(fast[-1]), 'slow_last': float(slow[-1])})
    nan_arr = arr.copy(); nan_arr[0] = np.nan; poisoned = ns['rolling_average'](nan_arr, 90)
    require(np.isnan(poisoned).all(), 'Original cumulative NaN propagation differs')
    rows.append({'case': 'one_NaN_poisons_all_later_cumulative_means', 'nan_outputs': int(np.isnan(poisoned).sum())})
    ns.update(g=SimpleNamespace(begin_date=dates[0], singal=arr.copy(), N=100, N1=90, N2=30, index_symbol='index', weight_method='mktcap'),
        log=quiet, record=lambda **kw: None, get_index_stocks=lambda *a, **kw: ['A'], CalROCSingal=lambda *a: 119.)
    context = SimpleNamespace(current_dt=datetime.datetime.combine(dates[-1], datetime.time(9, 30)), previous_date=dates[-2])
    ns['GetSingal'](context)
    require(ns['g'].singal[-2:].tolist() == [119., 119.], 'Original initialization endpoint duplicate differs')
    rows.append({'case': 'initial_prepare_then_first_trade_repeats_previous_date_endpoint', 'last_values': ns['g'].singal[-2:].tolist(),
        'fixture': 'Prepared sequence already includes previous_date; next call is after ETF begin_date'})
    ns['g'].begin_date = context.current_dt.date(); old = ns['g'].singal.copy(); ns['GetSingal'](context)
    require(np.array_equal(old, ns['g'].singal), 'Begin-date bypass differs')
    rows.append({'case': 'ETF_begin_date_does_not_append', 'unchanged': True})
    ns['g'].singal = np.ones(120); equality = ns['GetSingal'](context)
    require(not equality, 'Original strict double-mean comparison differs')
    rows.append({'case': 'equal_fast_slow_is_false', 'signal': bool(equality)})
    fees = []; ns.update(datetime=datetime, FixedSlippage=lambda x: x, set_slippage=lambda *a: None,
        PerTrade=lambda **kw: kw, set_commission=lambda x: fees.append(x))
    for t in (datetime.datetime(2013, 1, 1), datetime.datetime(2013, 1, 1, 9, 30)):
        ns['before_trading_start'](SimpleNamespace(current_dt=t))
    require([r['buy_cost'] for r in fees] == [.001, .0003], 'Original strict fee-date boundary differs')
    rows.append({'case': 'fee_boundary_is_strict_datetime_comparison', 'times': ['2013-01-01T00:00', '2013-01-01T09:30'],
        'declarations_only': fees, 'not_cost_backtest': True})
    orders = []; ns.update(order_target_value=lambda *a: orders.append(list(a)), order_target=lambda *a: orders.append(list(a)))
    context.portfolio = SimpleNamespace(total_value=10000., long_positions={'A': object()})
    ns['StockBuy']('A', context); ns['StockBuy']('B', context); ns['StockSell']('A', context); ns['StockSell']('B', context)
    require(orders == [['B', 10000.], ['A', 0]], 'Original held/absent order guards differ')
    rows.append({'case': 'buy_only_when_absent_and_sell_only_when_held', 'mock_orders': orders})

    leader = selected(directory, 1, ['fit_linear', 'handle_data', 'before_trading_start', 'filter_special'], {'np': np, 'log': quiet})
    leader['history'] = lambda **kw: pd.DataFrame({'000001.XSHG': [10., 11., 12.]})
    try: leader['fit_linear'](None, 3)
    except TypeError as exc: rows.append({'case': 'leader_original_OLS_array_float_fails', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected NumPy array scalar compatibility failure')
    context = SimpleNamespace(current_dt=datetime.datetime(2020, 6, 1, 10), portfolio=SimpleNamespace(positions={}, available_cash=100000.))
    g = SimpleNamespace(daily_buy_count=1, today_bought_stocks=set(), max_zt_days=4, buy_list=['A', 'B', 'C'])
    current = {s: SimpleNamespace(high_limit=11., name=s) for s in 'ABC'}; data = {s: SimpleNamespace(close=10.) for s in 'ABC'}
    calls = []
    def order(s, cash):
        calls.append([s, cash]); context.portfolio.available_cash -= cash/2
        return object()
    leader.update(g=g, fit_linear=lambda *a: 1., get_current_data=lambda: current, order_value=order)
    leader['handle_data'](context, data)
    require(calls == [['A', 100000.], ['B', 50000.]] and sorted(g.today_bought_stocks) == ['A', 'B'], 'Original daily-count overshoot differs')
    rows.append({'case': 'daily_buy_one_can_submit_two_accepted_orders_in_same_loop', 'mock_orders': [r.copy() for r in calls],
        'today_bought': sorted(g.today_bought_stocks), 'mock_cash_policy': 'Each accepted intent consumes half requested value, not actual ledger fills'})
    sales = []; context.portfolio.positions = {'A': SimpleNamespace(closeable_amount=100)}
    context.current_dt = context.current_dt.replace(hour=14, minute=51); leader['order_target'] = lambda *a: sales.append(list(a))
    leader['handle_data'](context, data)
    require(sales == [['A', 0]], 'Selling before daily-count early return differs')
    rows.append({'case': '14_51_sell_still_runs_after_daily_buy_limit', 'mock_orders': sales})
    calls.clear(); context.portfolio.positions = {}; context.current_dt = context.current_dt.replace(hour=10, minute=0)
    context.portfolio.available_cash = 5000.; g.today_bought_stocks = set(); leader['handle_data'](context, data)
    require(not calls, 'Original cash greater-than500-share boundary differs')
    rows.append({'case': 'cash_equal_500_shares_does_not_buy', 'cash': 5000., 'price': 10., 'mock_orders': []})
    leader['g'] = SimpleNamespace(stocks_exsit={'A'})
    def history(count, unit, field, stocks):
        values = np.full(count, 11. if field != 'low' else 10.)
        return {'A': values}
    leader.update(history=history, get_price=lambda *a, **kw: pd.DataFrame({'high_limit': [11.]}))
    leader['before_trading_start'](context)
    require(leader['g'].max_zt_days == 11 and leader['g'].buy_list == ['A'], 'Original limit-up count differs')
    rows.append({'case': 'leader_counts_at_most_11_non_one_price_limit_days', 'max_days': 11, 'mock_history_shape': 'dict of arrays'})
    leader['history'] = lambda count, unit, field, stocks: pd.DataFrame({'A': np.full(count, 11. if field != 'low' else 10.)},
        index=pd.bdate_range('2020-01-01', periods=count))
    leader['get_price'] = lambda *a, **kw: pd.DataFrame({'high_limit': [12.]})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try: leader['before_trading_start'](context)
        except KeyError as exc: rows.append({'case': 'leader_negative_label_fails_with_datetime_index', 'error': type(exc).__name__, 'message': str(exc)})
        else: raise ValueError('Expected negative-label failure')
    require(any(issubclass(w.category, pd.errors.ChainedAssignmentError) for w in captured) and
        leader['g'].x1['A'].iloc[0] == 11., 'Original chained assignment no-op differs')
    rows.append({'case': 'today_limit_chained_assignment_does_not_update_x1', 'intended_limit': 12., 'retained_limit': 11.,
        'warnings': [w.category.__name__ for w in captured], 'x1_not_read_by_handle_data': True})
    statuses = {s: SimpleNamespace(is_st=False, paused=False, name=s) for s in ('A', 'B', '688001.XSHG')}
    leader.update(get_current_data=lambda: statuses, get_security_info=lambda s: SimpleNamespace(start_date=context.current_dt.date()-datetime.timedelta(days=150 if s == 'A' else 151)))
    require(leader['filter_special'](context, list(statuses)) == ['B'], 'Listing strict150 or STAR filter differs')
    rows.append({'case': 'listing_strictly_over_150_natural_days_and_exclude688', 'selected': ['B']})
    return {'cases': rows, 'synthetic_fixture': True, 'real_historical_operand_windows': 0,
        'not_a_backtest': True, 'platform_equivalent': False, 'limits': ['Mock queries/orders diagnose original functions only; no fills/costs/NAV',
            'Original defects retained; no revised trading variant or inferred historical pool']}


def component(directory):
    validate_sources(directory); price = pd.read_parquet(directory / 'price-input.parquet')
    price['date'] = pd.to_datetime(price.date).dt.date
    pivot = price.pivot(index='date', columns='instrument', values='close_adj').sort_index()
    calendar = pd.read_parquet(directory / 'calendar-input.parquet'); calendar['date'] = pd.to_datetime(calendar.date).dt.date
    dates = calendar.loc[calendar.is_open & calendar.date.between(pivot.index.min(), pivot.index.max()), 'date'].tolist()
    require(dates == list(pivot.index), 'Stock operand dates do not match frozen open calendar')
    stocks = list(pivot.columns); ns = selected(directory, 0, ['GetROC'], {})
    count = 0; blocked = 0; max_error = 0.; output = []; by_stock = {s: 0 for s in stocks}
    for i in range(99, len(dates)):
        ns['get_trade_days'] = lambda **kw: dates[i-kw['count']+1:i+1]
        ns['get_price'] = lambda requested, **kw: pd.DataFrame({'code': stocks, 'close': pivot.loc[kw['end_date']].to_numpy()})
        result = ns['GetROC'](stocks, dates[i], 100)
        expected_codes = set()
        last_row = pivot.iloc[i-99]; now_row = pivot.iloc[i]
        for s in stocks:
            last = last_row[s]; now = now_row[s]
            if not np.isfinite(last) or not np.isfinite(now) or last == 0:
                blocked += 1; continue
            expected_codes.add(s); expected = float(Decimal(str(now))/Decimal(str(last))-1)
            error = abs(float(result[s])-expected); max_error = max(max_error, error)
            require(error <= 1e-12*max(1., abs(expected)), 'Independent ROC arithmetic differs')
            output.append({'date': str(dates[i]), 'offset_date': str(dates[i-99]), 'instrument': s,
                'last_close_adj': float(last), 'now_close_adj': float(now), 'original_ROC': float(result[s]), 'independent_ROC': expected})
            count += 1; by_stock[s] += 1
        require(set(result.index) == expected_codes, 'Original ROC missing-endpoint behavior differs')
    return {'input_sha256': OPERAND_SHA, 'calendar_sha256': CALENDAR_SHA, 'rows': len(price), 'pool': stocks, 'first': str(dates[0]), 'last': str(dates[-1]),
        'calendar_rows': len(dates), 'endpoint_windows': count, 'unavailable_endpoint_windows': blocked,
        'maximum_absolute_error': max_error, 'result_fingerprint': digest(output), 'intervals': 99,
        'windows_by_stock': by_stock, 'first_last_samples': output[:2] + output[-2:],
        'not_a_backtest': True, 'platform_equivalent': False, 'limits': ['Ten frozen stocks are arithmetic operands, not original HS300 members or HY pool',
            'close_adj input basis is explicit; original query adjustment anchoring remains unproved',
            'No full-pool weighted diffusion, ETF/leader signal, orders, three-cost or strategy return inferred']}


def study(root, directory):
    binding(root, directory)
    accepted = read('docs/handoff/2026-10-07-batch47-verification.json')
    require(file_sha(CALENDAR) == CALENDAR_SHA == accepted['checks']['evidence_sha256']['calendar-input.parquet'], 'Accepted calendar changed')
    require(not (directory / 'calendar-input.parquet').exists(), 'Calendar copy exists')
    shutil.copyfile(CALENDAR, directory / 'calendar-input.parquet')
    save(directory / 'calendar-binding.json', {'file': str(CALENDAR), 'sha256': CALENDAR_SHA, 'accepted_receipt_sha256': file_sha('docs/handoff/2026-10-07-batch47-verification.json')})
    freeze_component_failure(directory)
    save(directory / 'diagnostics.json', diagnostics(directory)); save(directory / 'component-research.json', component(directory))
    return archive(root, 'Batch61 original defects and long-price ROC endpoint arithmetic archived; no trading backtest')


def freeze_component_failure(directory):
    failure = Path('data/staging/strategies-batch61/20261007-diffusion-leader-component-failure-01')
    save(directory / 'component-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})


def recover_component(root, directory):
    binding(root, directory); freeze_component_failure(directory); validate_sources(directory)
    require(diagnostics(directory) == read(directory / 'diagnostics.json'), 'Previously frozen diagnostics changed')
    save(directory / 'component-research.json', component(directory))
    return archive(root, 'Batch61 text/date boundary normalized; accepted calendar and original ROC component archived without overwriting prior stage')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch61 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); comp = component(directory); dep = inventory(root); decision = supplements(); catalog = offline_catalog(root, directory)
    save(directory / 'diagnostics-offline.json', diag); save(directory / 'component-offline.json', comp)
    require(before == implementation() and file_sha(directory / 'diagnostics-offline.json') == file_sha(directory / 'diagnostics.json') and
        file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        dep == read(directory / 'dependency-inventory.json') and decision == read(directory / 'supplement-decision.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'diagnostics_sha256': file_sha(directory / 'diagnostics.json'),
        'component_sha256': file_sha(directory / 'component-research.json'), 'not_a_backtest': True})
    return archive(root, 'Batch61 forbidden-network original diagnostics/component and three catalogs match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch61.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch61_research.py', 'tests/unit/test_batch60_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    require(read(directory / 'existing-apis.json') == api_evidence() and read(directory / 'supplement-decision.json') == supplements(), 'API/decision changed')
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') == file_sha(directory / 'diagnostics-offline.json') and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline proof changed')
    with redirect_stdout(StringIO()): diag = diagnostics(directory); comp = component(directory)
    require(diag == read(directory / 'diagnostics.json') and comp == read(directory / 'component-research.json') and
        inventory(root) == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json') and
        api_evidence() == read(directory / 'existing-apis.json'), 'Recomputed evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch61 diffusion/leader research accepted; original trading inputs remain gated')
    receipt = Path('docs/handoff/2026-10-07-batch61-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'component': comp, 'diagnostics': diag,
        'protection': protection, 'progress': progress, 'supplements': read(directory / 'supplement-decision.json'),
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'recover_component', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch61/20261007-diffusion-leader-02')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
