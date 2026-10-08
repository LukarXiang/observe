"""Freeze commodity turtle/MA rules and dependencies without invented trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
import hashlib
import inspect
from io import StringIO
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
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions, digest
from scripts import review_strategy_batch59 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch59/20261007-TD-IF')
RECEIPT = Path('docs/handoff/2026-10-07-batch59-verification.json')
SOURCES = ('2020年度精选策略/22 海龟交易法则升级版---（分钟级回测+合约变更移仓）.txt',
    '2020年度精选策略/09 商品期货 多品种日频双均线模型 作为给入门者的小礼物.txt')
SOURCE_SHA = ('9c4de2f241ff8bb293a96ebd9cf042c0f1dfca68b6895add41f4936aabbac1bf',
    'f9cd21141068d0de24a6c474e60249d354aed0f1ae3a0eb00a8637a77847936e')
CODE_START = (10, 12)
REFERENCES = (('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/vnpy', 'vnpy/trader/utility.py'), ('repo/rqalpha', 'rqalpha/examples/turtle.py'),
    ('repo/akshare', 'akshare/futures/futures_daily_bar.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch60.py',
    'tests/unit/test_batch60_research.py', 'src/observe/execution.py', 'src/observe/ledger/book.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'diagnostic-failure-binding.json',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'supplement-decision.json', 'diagnostics.json', 'probe-results.json',
    'diagnostics-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERY = {'function': 'get_shfe_daily', 'parameters': {'date': '20200102'}}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch59.py') == proof['final_script_sha256'], 'Batch59 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch60 commodity turtle/MA source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    rows = []
    for name in ('futures_zh_minute_sina', 'futures_zh_daily_sina', 'match_main_contract', 'get_shfe_daily'):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'function': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest()})
    fn = ak.get_shfe_daily; helper = fn.__globals__['requests_link']; code = inspect.getsource(helper)
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'apis': rows,
        'shfe_url_template': fn.__globals__['cons'].SHFE_DAILY_URL_20250630,
        'calendar_contains_query': QUERY['parameters']['date'] in fn.__globals__['calendar'],
        'query': QUERY, 'original_retry_helper': {'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()},
        'connection_policy': 'One direct request, no environment proxy, timeout8/10s; temporarily replace only retry helper, preserve installed parser',
        'limits': ['Single-day exchange contracts/settlement sample is not original long ordered main mapping or 8888/9999 synthetic history',
            'Current main matching and continuous CU0 cannot supply historical dominant contracts/minute rollover or historical margin/fees']}


def supplements():
    prior = previous.supplements()
    return {'prior_dns_evidence': prior, 'query': QUERY, 'new_probe': 'SHFE historical20200102 all-contract daily sample',
        'decision': 'Reuse same-host stock2.finance.sina.com.cn DNS failures for minute/daily; inspect independent exchange host once',
        'missing': ['CU_historical_dominant_mapping_actual_contract_21day_and_1m',
            '18commodity_original_8888_index_50day_and_dominant_open_prices',
            'historical_expiration_multiplier_tick_margin_settlement_fees_and_execution'],
        'published': False, 'not_a_backtest': True}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in rows.values()) for t, rows in state['tables'].items()},
        'historical_equity_dependencies': previous.historical_dependencies(root),
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'limits': ['No registered commodity futures contract/index/minute/settlement tables; equity/index prices are not substitute operands',
            'Unique Book has no proved futures margin/multiplier/settlement semantics; no alternate ledger introduced',
            'Historical shares/PCF/names still gate micro400 and small-value100']}


def parse_source(text, start):
    return ast.parse('\n'.join(text.splitlines()[start-1:]))


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    failure = Path('data/staging/strategies-batch60/20261007-commodity-turtle-MA-diagnostic-failure-01')
    save(directory / 'diagnostic-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha, start_line in zip(SOURCES, SOURCE_SHA, CODE_START, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path); tree = parse_source(text, start_line)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha and
            old[name]['code_start_line'] == start_line, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'code_start_line': start_line,
            'review': REVIEWS[name], 'newly_reviewed': True, 'definition_ast': definitions(tree)})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only standard ATR/Donchian/order and exchange-parser comparison, not original platform equivalence or alternate ledger'})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch60 two complete commodity strategy sources frozen; original inputs unavailable')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    return parse_source(read_source(Path(row['source_copy']))[0], CODE_START[number])


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha and
            row['code_start_line'] == CODE_START[i] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(source_tree(directory, i)), 'Source/rule/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    failure = read(directory / 'diagnostic-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure evidence scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'External diagnostic failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 2 passed' in red['stdout'] and "name 'after_market_close' is not defined" in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red regression binding differs')


def selected(directory, number, names, ns):
    import __future__
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing selected function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-commodity-functions>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns


def diagnostics(directory):
    validate_sources(directory); rows = []; declarations = []; schedules = []; orders = []; messages = []
    log = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    def ns():
        def order(kind):
            def intent(*args, **kwargs): orders.append({'kind': kind, 'args': list(args), 'kwargs': kwargs}); return None
            return intent
        return {'g': SimpleNamespace(), 'log': log, 'np': np, 'mean': np.mean,
            'order': order('order'), 'order_target': order('target'), 'send_message': lambda *a: messages.append(list(a)),
            'set_benchmark': lambda x: declarations.append(['benchmark', x]),
            'set_option': lambda *a: declarations.append(['option', *a]),
            'set_subportfolios': lambda x: declarations.append(['subportfolios', x]),
            'set_order_cost': lambda x, **kw: declarations.append(['cost', x, kw]),
            'set_slippage': lambda x, **kw: declarations.append(['slippage', x, kw]),
            'SubPortfolioConfig': lambda **kw: kw, 'OrderCost': lambda **kw: kw,
            'FixedSlippage': lambda x: ['fixed', x], 'PriceRelatedSlippage': lambda x: ['relative', x],
            'run_daily': lambda f, **kw: schedules.append([f.__name__, kw])}
    turtle = selected(directory, 0, ['initialize', 'before_market_open', 'while_open', 'check_stop', 'set_markprice',
        'add_or_close', 'check_break', 'calc_ATR', 'get_unit', 'reset', 'get_future_code'], ns())
    ma = selected(directory, 1, ['initialize', 'set_info', 'set_future_list', 'before_market_open', 'market_open', 'after_market_close', 'Trade',
        'Dont_Re_entry', 'TrailingStop', 'replace_old_futures', 'get_future_code', 'get_lots', 'get_CCFX_end_date'], ns())
    context = SimpleNamespace(portfolio=SimpleNamespace(starting_cash=1000000., total_value=1000000.,
        long_positions={}, short_positions={}), current_dt=datetime.datetime(2020, 1, 2, 9, 0))
    def failure(name, call, expected):
        try: call()
        except expected as exc: rows.append({'case': name, 'error': type(exc).__name__, 'orders': orders.copy()})
        else: raise ValueError('Expected original failure: '+name)
    turtle['initialize'](context); tg = turtle['g']
    require((tg.symbol, tg.N, tg.limit, tg.frequency, tg.shift) == ('CU', 20, 4, '1m', True), 'Turtle defaults differ')
    rows.append({'case': 'turtle_defaults_schedule_and_cost_declarations', 'declarations': declarations.copy(),
        'schedules': schedules.copy(), 'margin_explicitly_set': False, 'fees_not_calculated': True})
    require(not any(r[:2] == ['option', 'futures_margin_rate'] for r in declarations), 'Turtle margin unexpectedly set')
    codes = {n: turtle['get_future_code'](n) for n in ('CU', 'IF', 'IH', 'IC')}
    require(codes == {'CU': 'CU9999.XSGE', 'IF': '000300.XSHG', 'IH': '000016.XSHG', 'IC': '000905.XSHG'}, 'Duplicate keys differ')
    rows.append({'case': 'turtle_duplicate_index_keys_overwrite_future_codes', 'codes': codes})
    sample = pd.DataFrame({'high': [11., 13., 21.], 'low': [9., 11., 19.], 'close': [10., 12., 20.]})
    actual = float(turtle['calc_ATR'](sample)); wrapped = [11., 3., 9.]
    require(actual == math.fsum(wrapped)/3, 'Wrapped ATR differs')
    rows.append({'case': 'turtle_ATR_first_row_wraps_to_last_close_and_uses_all_rows', 'value': actual,
        'wrapped_true_ranges': wrapped, 'conventional_two_range_mean': 6.})
    missing = selected(directory, 0, ['calc_ATR'], {})
    failure('turtle_ATR_requires_unproved_mean_numpy_injection', lambda: missing['calc_ATR'](sample), NameError)
    failure('turtle_zero_ATR_divides_by_zero', lambda: turtle['get_unit'](1000., 0., 'CU'), ZeroDivisionError)
    require(turtle['get_unit'](1000., 10., 'IC') == 1. and turtle['get_unit'](1000., 10., 'IH') == 1., 'Original multiplier constants differ')
    rows.append({'case': 'turtle_IC_IH_original_multiplier_is_one', 'IC': 1, 'IH': 1, 'historical_contract_rule_proved': False})
    tg.upperbound = tg.lowerbound = 100.
    require(turtle['check_break'](100.) == 'long', 'Break equality priority differs')
    thresholds = [turtle['add_or_close'](price, 100., 10., pos) for price, pos in ((105., 1), (80., 1), (95., -1), (120., -1))]
    require(thresholds == ['longadd', 'longclose', 'shortadd', 'shortclose'], 'ATR threshold equality differs')
    rows.append({'case': 'turtle_inclusive_break_and_add_close_boundaries', 'equal_upper_lower': 'long', 'signals': thresholds})
    def turtle_state(**updates):
        tg.__dict__.update(symbol='CU', future='CU_NEW', lastfuture='CU_NEW', position=0, add=0, lastprice=0.,
            markprice=0., ATR=10., upperbound=105., lowerbound=95., shift=True, contract_change=False, limit=4, frequency='1m')
        tg.__dict__.update(updates); orders.clear()
        turtle['history'] = lambda *a: pd.DataFrame({'close': [110.]})
    turtle_state(); turtle['while_open'](context)
    require((tg.position, tg.add, tg.lastprice) == (1, 1, 110.) and len(orders) == 1, 'None entry state differs')
    rows.append({'case': 'turtle_None_entry_still_updates_internal_units', 'orders': orders.copy(), 'position': tg.position, 'add': tg.add})
    turtle_state(ATR=100.); context.portfolio.total_value = 100.; turtle['while_open'](context)
    require(orders[0]['args'][1] == 0 and tg.position == 1, 'Zero-lot state differs')
    rows.append({'case': 'turtle_zero_lot_intent_still_counts_position', 'orders': orders.copy(), 'position': tg.position})
    context.portfolio.total_value = 1000000.
    turtle_state(position=3, add=3, lastprice=100., markprice=110., upperbound=1000., lowerbound=1.)
    turtle['while_open'](context); first = orders.copy(); turtle['history'] = lambda *a: pd.DataFrame({'close': [120.]})
    turtle['while_open'](context)
    require(tg.add == tg.position == 4 and len(orders) == 1, 'Unit limit differs')
    rows.append({'case': 'turtle_limit_four_includes_initial_entry', 'last_add': first, 'orders_after_fifth_attempt': orders.copy(), 'position': tg.position})
    turtle_state(position=1, add=2, lastprice=100., markprice=100., upperbound=1000., lowerbound=1.,
        contract_change=True, lastfuture='CU_OLD', adj=2., ATR=100.)
    context.portfolio.long_positions = {'CU_OLD': SimpleNamespace(total_amount=3)}
    turtle['history'] = lambda *a: pd.DataFrame({'close': [200.]}); turtle['while_open'](context)
    require(len(orders) == 3 and orders[0]['args'] == ['CU_OLD', 0] and orders[1]['args'] == ['CU_NEW', 3] and
        tg.position == 2 and tg.add == 3, 'Stale lastprice rollover/add differs')
    rows.append({'case': 'turtle_roll_adjusts_mark_only_and_can_add_same_bar_after_None_orders', 'orders': orders.copy(),
        'position': tg.position, 'add': tg.add, 'markprice': tg.markprice, 'lastprice': tg.lastprice})
    turtle_state(position=1, add=2, lastprice=100., markprice=100., upperbound=1000., lowerbound=1.,
        contract_change=True, lastfuture='CU_OLD', adj=2., shift=False)
    turtle['history'] = lambda *a: pd.DataFrame({'close': [200.]}); turtle['while_open'](context)
    require(tg.position == tg.add == 0 and tg.lastprice == 0 and tg.markprice == 100., 'No-shift reset differs')
    rows.append({'case': 'turtle_no_shift_resets_units_but_keeps_markprice', 'orders': orders.copy(), 'markprice': tg.markprice})
    turtle_state(position=1, contract_change=True, lastfuture='MISSING', adj=1.)
    failure('turtle_roll_missing_old_position_raises', lambda: turtle['while_open'](context), KeyError)
    turtle_state(); turtle['get_dominant_future'] = lambda *a: 'CU_NEW'; turtle['attribute_history'] = lambda *a: sample.iloc[:2]
    turtle['before_market_open'](context)
    require(tg.upperbound == 13. and tg.lowerbound == 11., 'Partial channel differs')
    rows.append({'case': 'turtle_partial_two_row_history_not_blocked', 'upper': tg.upperbound, 'lower': tg.lowerbound, 'ATR': float(tg.ATR)})
    for position, mark, price in ((1, 100., 95.), (-1, 100., 105.)):
        tg.position = position; tg.markprice = mark; require(turtle['check_stop'](price) is True, 'Trailing equality differs')
    rows.append({'case': 'turtle_five_percent_stop_is_inclusive', 'long95': True, 'short105': True})
    declarations.clear(); schedules.clear(); orders.clear()
    ma['get_dominant_future'] = lambda symbol: symbol+'_OLD'; ma['initialize'](context); mg = ma['g']
    require(len(mg.instruments) == 18 and (mg.FastWindow, mg.SlowWindow, mg.stop, mg.margin_rate) == (5, 20, .05, .15), 'MA defaults differ')
    rows.append({'case': 'MA_eighteen_symbols_schedule_and_cost_declarations', 'instruments': mg.instruments.copy(),
        'declarations': declarations.copy(), 'schedules': schedules.copy(), 'fees_not_calculated': True})
    ma['attribute_history'] = lambda *a: pd.DataFrame({'open': [100.]})
    lots = ma['get_lots'](1000000./18, 'CU')
    require(isinstance(lots, pd.Series) and list(lots.index) == ['open'] and math.isclose(float(lots.iloc[0]), (1000000./18)/(100*.15*5)), 'Series lots differ')
    rows.append({'case': 'MA_get_lots_returns_unrounded_one_column_Series_with_leverage', 'values': lots.to_dict(), 'fractional': True})
    ma['attribute_history'] = lambda *a: pd.DataFrame({'open': []})
    require(ma['get_lots'](1000., 'CU') is None, 'Empty lots differs')
    rows.append({'case': 'MA_empty_lot_history_returns_None'})
    mg.MappingReal = {'CU': 'CU_OLD'}; orders.clear(); context.portfolio.long_positions = {}
    context.portfolio.short_positions = {'CU_OLD': SimpleNamespace(total_amount=2)}
    failure('MA_short_roll_reads_new_contract_position_and_raises', lambda: ma['replace_old_futures'](context, 'CU', 'CU_NEW'), KeyError)
    context.portfolio.long_positions = {'CU_OLD': SimpleNamespace(total_amount=3)}; orders.clear()
    failure('MA_mixed_roll_fails_after_long_intents_before_mapping_update', lambda: ma['replace_old_futures'](context, 'CU', 'CU_NEW'), KeyError)
    require(len(orders) == 2 and mg.MappingReal['CU'] == 'CU_OLD', 'Partial roll sequence differs')
    context.portfolio.long_positions = {}; context.portfolio.short_positions['CU_NEW'] = SimpleNamespace(total_amount=0); orders.clear()
    ma['replace_old_futures'](context, 'CU', 'CU_NEW')
    require(orders[1]['args'] == ['CU_NEW', 0] and mg.MappingReal['CU'] == 'CU_NEW', 'Wrong new short quantity differs')
    rows.append({'case': 'MA_short_roll_uses_zero_new_quantity_and_updates_mapping_after_None', 'orders': orders.copy(), 'mapping': mg.MappingReal.copy()})
    mg.instruments = ['CU']; mg.MappingReal = {'CU': 'CU_OLD'}; mg.future_list = ['CU_OLD']
    context.portfolio.short_positions = {}; orders.clear(); ma['get_dominant_future'] = lambda *a: 'CU_NEW'
    ma['attribute_history'] = lambda *a: pd.DataFrame({'open': [100.]}); ma['before_market_open'](context)
    require(mg.future_list == ['CU_OLD', 'CU_NEW'] and messages == [['开始交易']], 'Old list retention differs')
    rows.append({'case': 'MA_before_open_retains_old_contract_list_entry', 'future_list': mg.future_list.copy(),
        'send_message_calls': messages.copy(), 'external_messages_sent': False})
    context.portfolio.long_positions = {'CU_NEW': SimpleNamespace(total_amount=1)}
    mg.HighPrice = {'CU_NEW': 100.}; mg.LowPrice = {'CU_NEW': False}; mg.LastRealPrice = {'CU_NEW': 90.}
    mg.Times = {'CU_NEW': 100}; mg.Reentry_long = False; orders.clear(); ma['TrailingStop'](context, 'CU_NEW')
    require(mg.Reentry_long is True and mg.Times['CU_NEW'] == 100, 'Stop counter unexpectedly reset')
    mg.ClosePrice = pd.Series(np.full(50, 100.)); mg.CurrentPrice = 90.; ma['Dont_Re_entry'](context, 'CU_NEW')
    require(mg.Reentry_long is False, 'Accumulated counter reentry differs')
    rows.append({'case': 'MA_stop_does_not_reset_counter_and_old_counter_immediately_unblocks', 'orders': orders.copy(), 'counter': mg.Times['CU_NEW'], 'reentry_long': mg.Reentry_long})
    captures = []; requested = []
    def market_state():
        mg.instruments = ['CU', 'TA']; mg.MappingReal = {'CU': 'CU_NEW', 'TA': 'TA_NEW'}
        mg.MappingIndex = {'CU': 'CU8888.XSGE', 'TA': 'TA8888.XZCE'}
        mg.Times = {'CU_NEW': 0, 'TA_NEW': 0}; mg.Reentry_long = mg.Reentry_short = False
        ma['get_CCFX_end_date'] = lambda *a: datetime.date(2099, 1, 1)
        ma['Trade'] = lambda ctx, future: captures.append({'future': future, 'signal': mg.Signal, 'cross': mg.Cross})
        ma['Dont_Re_entry'] = lambda *a: None
        captures.clear(); requested.clear(); orders.clear()
    def history(code, count, *a):
        requested.append([code, count]); values = np.arange(count, dtype=float)+100.
        return pd.DataFrame({'close': values}, index=range(-count, 0))
    market_state(); ma['attribute_history'] = history; mg.Reentry_long = True; ma['market_open'](context)
    require([r['signal'] for r in captures] == [0, 0] and [r['cross'] for r in captures] == [1, 1], 'Global reentry coupling differs')
    rows.append({'case': 'MA_shared_reentry_flag_blocks_both_symbols', 'captured_intents': captures.copy(), 'explicit_negative_label_fixture': True})
    market_state(); ma['get_CCFX_end_date'] = lambda *a: context.current_dt.date(); ma['market_open'](context)
    require(not captures and not requested, 'Expiry return does not stop full loop')
    rows.append({'case': 'MA_first_symbol_expiry_returns_entire_callback', 'processed_signals': captures.copy()})
    market_state(); ma['attribute_history'] = lambda code, count, *a: history(code, min(count, 10))
    ma['market_open'](context)
    require(not captures and all(r[0] != 'TA_NEW' for r in requested), 'Short history return differs')
    rows.append({'case': 'MA_first_symbol_short_history_returns_before_second', 'history_requests': requested.copy()})
    market_state(); ma['attribute_history'] = lambda code, count, *a: pd.DataFrame({'close': np.full(count, 100.)},
        index=pd.date_range('2020-01-01', periods=count))
    failure('MA_original_datetime_Series_negative_label_fails', lambda: ma['market_open'](context), KeyError)
    require(len(rows) == 27, 'Diagnostic case count differs')
    return {'cases': rows, 'records_sha256': digest(rows), 'not_a_backtest': True, 'platform_equivalent': False,
        'synthetic_fixture': True, 'real_historical_operand_windows': 0,
        'limits': ['All prices/contracts/positions/cash/orders are explicit synthetic diagnostic fixtures, not historical trading data or fills',
            'Mean/numpy injection and negative integer labels are explicit diagnostic scaffolding, not hidden compatibility repairs',
            'Schedules and fee/margin/slippage declarations do not compute three-cost returns',
            'Standard ATR/RQAlpha/TA-Lib references do not replace original wrapped ATR or state behavior']}


def study(root, directory):
    binding(root, directory)
    with redirect_stdout(StringIO()): result = diagnostics(directory)
    save(directory / 'diagnostics.json', result)
    return archive(root, 'Batch60 original commodity ATR/state/rollover/coupling defects diagnosed; synthetic only')


def worker(root, directory):
    folder = directory / 'probes/shfe'; folder.mkdir(parents=True, exist_ok=False)
    row = {'endpoint': 'shfe', **QUERY, 'status': 'failed', 'attempts': [], 'files': [], 'wire_responses': [],
        'published': False, 'strict_usable': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(not row['attempts'], 'One-request limit reached'); row['attempts'].append({'method': method, 'url': url})
        session.trust_env = False; kwargs['timeout'] = (8, 10); kwargs['allow_redirects'] = False
        response = original(session, method, url, **kwargs); path = folder / 'response-000.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    def one_shot(url, encoding='utf-8', method='get', data=None, headers=None):
        require(method == 'get', 'Unexpected request method')
        response = requests.get(url, headers=headers); response.encoding = encoding; return response
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'API changed')
        import akshare as ak
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request), \
                patch.dict(ak.get_shfe_daily.__globals__, {'requests_link': one_shot}):
            frame = ak.get_shfe_daily(**QUERY['parameters'])
        path = Path(root) / 'raw/commodity_probe_batch60/shfe' / f'{directory.name}.parquet'; require(not path.exists(), 'Raw exists')
        path = raw.save(root, 'commodity_probe_batch60', 'shfe', directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    finally: save(folder / 'result.json', row)
    return row


def validate_probe(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'API changed')
    row = read(directory / 'probe-results.json')['result']
    require(row == read(directory / 'probes/shfe/result.json') and row['function'] == QUERY['function'] and
        row['parameters'] == QUERY['parameters'] and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and
        row['published'] is False and row['strict_usable'] is False and len(row['attempts']) <= 1 and
        row['status'] in ('failed', 'empty', 'success'), 'Probe binding changed')
    expected_url = read(directory / 'existing-apis.json')['shfe_url_template'] % QUERY['parameters']['date']
    require(all(a['method'].upper() == 'GET' and a['url'] == expected_url for a in row['attempts']), 'Probe request scope changed')
    for item in row['files']+row['wire_responses']: require(file_sha(item['file']) == item['sha256'], 'Probe response changed')
    if row['status'] in ('empty', 'success'):
        require(len(row['files']) == 1, 'Probe raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
        require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status']=='success'), 'Raw profile differs')
    else: require(not row['files'] and row.get('error'), 'Failure evidence missing')
    return row


def probe(root, directory):
    binding(root, directory); require(not (directory / 'probes/shfe').exists(), 'Probe already attempted')
    row = worker(root, directory); save(directory / 'probe-results.json', {'result': row, 'published': False})
    validate_probe(directory)
    return archive(root, 'Batch60 independent SHFE historical daily probe archived; not published')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch60 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        result = diagnostics(directory); dep = inventory(root); decision = supplements(); catalog = offline_catalog(root, directory)
        validate_probe(directory)
    save(directory / 'diagnostics-offline.json', result)
    require(before == implementation() and file_sha(directory / 'diagnostics-offline.json') == file_sha(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and decision == read(directory / 'supplement-decision.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch60 forbidden-network original diagnostics and three catalogs match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch60.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch60_research.py', 'tests/unit/test_batch59_research.py',
            'tests/unit/test_catalog_futures.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); validate_probe(directory)
    require(read(directory / 'supplement-decision.json') == supplements(), 'Decision changed')
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); validate_probe(directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and
        off['socket_network_disabled'] is True and off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics-offline.json'), 'Offline proof changed')
    with redirect_stdout(StringIO()): diag = diagnostics(directory)
    require(diag == read(directory / 'diagnostics.json') and inventory(root) == read(directory / 'dependency-inventory.json') and
        supplements() == read(directory / 'supplement-decision.json'), 'Recomputed research changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch60 commodity source diagnostics accepted; original futures trades remain data-gated')
    receipt = Path('docs/handoff/2026-10-07-batch60-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'supplements': read(directory / 'supplement-decision.json'), 'probe': read(directory / 'probe-results.json'),
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch60/20261007-commodity-turtle-MA')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
