"""Execute unchanged short80/limit78 definitions on explicit fixtures; never trades."""
import ast
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch101 import canonical
from scripts.research_strategy_batch102 import safe_definition
from scripts.verify_strategy_batch3 import require

PREREAD = Path('data/staging/strategies-batch103/20261011-next-source-preread')
SOURCES = ('2024年度精选策略2/80.短线策略-2022年两个半月收益45%-无未来函数.txt',
           '聚宽2025年精选/78首板低开策略-终极版 最大回撤15%，年化50%.txt')
SOURCE_SHA = ('780c0a0c8d22c12c6f27a3ddb91127b72d80e71da96152fde47b237b64071829',
              '317641ff8498a7e2893c410aa1a45ab00e0582684d65a3f1fb84efd8e05115ba')


def tree(number):
    require(number in (0, 1), 'Unknown source')
    path = PREREAD/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number] == file_sha(Path('repo/量化策略源代码')/SOURCES[number]),
            'Original source bytes changed')
    return ast.parse(read_source(path)[0])


def literal(number, namespace=None):
    nodes = [n for n in tree(number).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (11 if number == 0 else 20) and all(safe_definition(n) for n in nodes),
            'Original definition contract differs')
    ns = {'pd': pd, 'np': np, 'dt': dt, 'datetime': dt.datetime, 'g': SimpleNamespace(),
          'log': SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREREAD/f'{number}.source'), 'exec'), ns)
    return ns


def initialize(number):
    calls = []
    ns = literal(number, {'enable_profile': lambda: calls.append(['profile']),
        'set_benchmark': lambda s: calls.append(['benchmark', s]),
        'set_option': lambda *a: calls.append(['option', list(a)]),
        'set_order_cost': lambda *a, **k: calls.append(['cost', list(a), k]),
        'set_slippage': lambda *a: calls.append(['slippage', list(a)]),
        'run_daily': lambda f, *a, **k: calls.append(['daily', f.__name__, list(a), k]),
        'OrderCost': lambda **k: k, 'FixedSlippage': lambda v: v})
    ns['initialize'](SimpleNamespace())
    return ns, calls


def short_orders(*, holdings=(), buy=('A', 'B'), immediate=False, cash=1000., limit=False):
    """Cash/positions feedback is a fixture, without Book fills or cost simulation."""
    ns, _ = initialize(0); calls = []
    positions = {s: SimpleNamespace(closeable_amount=100, avg_cost=9.) for s in holdings}
    portfolio = SimpleNamespace(positions=positions, available_cash=cash)
    ns['g'].buy_list = list(buy)
    current = {s: SimpleNamespace(low_limit=10. if limit else 9., high_limit=11.) for s in set(holdings)|set(buy)}
    bars = {s: SimpleNamespace(close=10.) for s in current}
    def order(kind, symbol, value):
        calls.append([kind, symbol, float(value)])
        if immediate:
            if kind == 'sell': positions.pop(symbol, None)
            else:
                positions[symbol] = SimpleNamespace(closeable_amount=0, avg_cost=10.)
                portfolio.available_cash -= value
        return None
    ns.update(get_current_data=lambda: current, order_target=lambda s, v: order('sell', s, v),
              order_value=lambda s, v: order('buy', s, v))
    error = None; output = StringIO()
    with redirect_stdout(output):
        try: ns['handle_data'](SimpleNamespace(portfolio=portfolio), bars)
        except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'calls': calls, 'holding_keys': list(positions), 'cash': portfolio.available_cash,
            'error': error, 'immediate_feedback_stub': immediate,
            'orders_return_None_not_fills': True}  # exclude nondeterministic timing prints


def sell78(time='11:25:00', price=11., *, cost=10., limit=12., closeable=100):
    calls = []; stock = SimpleNamespace(avg_cost=cost, closeable_amount=closeable)
    ns = literal(1, {'get_previous_trade_day': lambda *a: dt.date(2024, 4, 8),
        'get_current_data': lambda: {'A': SimpleNamespace(last_price=price, high_limit=limit)},
        'get_security_info': lambda *a: SimpleNamespace(display_name='fixture-A'),
        'order_target_value': lambda *a: calls.append(list(a))})
    ns['get_previous_trade_day'] = lambda *a: dt.date(2024, 4, 8)
    context = SimpleNamespace(current_dt=dt.datetime.fromisoformat('2024-04-08T'+time),
        portfolio=SimpleNamespace(positions={'A': stock}))
    with redirect_stdout(StringIO()): ns['sell'](context)
    return calls


def opening78(number, opening, previous=100.):
    calls = []; frame = pd.DataFrame({'open': [opening], 'pre_close': [previous]})
    def price(*a, **k): calls.append([list(a), k]); return frame
    ns = literal(1, {'get_price': price})
    return {'selected': ns['filter_stocks_by_opening_range'+('2' if number == 2 else '')]('2024-04-08', ['A']),
            'queries': calls}


def gain78(number, previous, close, opening, *, negative_labels=False):
    index = [-2, -1] if negative_labels else pd.RangeIndex(2)
    frame = pd.DataFrame({'close': [previous, close], 'open': [previous, opening]}, index=index)
    ns = literal(1, {'get_price': lambda *a, **k: frame}); out = StringIO()
    with redirect_stdout(out): result = ns['filter_st_stock'+('2' if number == 2 else '')](['A'], dt.date(2024, 4, 5))
    return {'selected': result, 'stdout': out.getvalue(), 'negative_label_fixture': negative_labels,
            'real_ST_status_queried': False}


def diagnostics():
    cases = []
    def add(name, callback):
        logs = StringIO(); error = None; result = None
        with redirect_stdout(logs):
            try: result = callback()
            except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)}
        cases.append({'case': name, 'value': canonical(result), 'error': error, 'stdout': logs.getvalue()})

    for n in (0, 1): add(f'initialize_{n}', lambda n=n: initialize(n)[1])
    pool = pd.DataFrame({'display_name': ['普通', 'ST测试', '*测试', '退市', '创业', '科创', '北证', '未排689', '未排4']},
        index=['000001.XSHE', '000002.XSHE', '000003.XSHE', '000004.XSHE', '300001.XSHE',
               '688001.XSHG', '830001.XBEI', '689001.XSHG', '430001.XBEI'])
    add('short_pool_names_and_prefixes', lambda: literal(0, {'get_all_securities': lambda *a: pool})['get_stock_pool']('2024-04-05'))

    for name, amounts in [('opposite', [-1., 1.]), ('zero', [0., 1.]), ('same', [1., 2.])]:
        frame = pd.DataFrame({'sec_code': ['A', 'A'], 'date': ['2024-04-03', '2024-04-05'], 'change_pct': amounts})
        add('short_moneyflow_'+name, lambda frame=frame: literal(0, {'get_money_flow': lambda **k: frame})['stock_filter3'](['A'], '2024-04-05'))
    one = pd.DataFrame({'sec_code': ['A'], 'date': ['2024-04-05'], 'change_pct': [1.]})
    add('short_moneyflow_one_column_returns_None', lambda: literal(0, {'get_money_flow': lambda **k: one})['stock_filter3'](['A'], '2024-04-05'))
    for gap in (-.01, 0., .01):
        add('short_open_gap_'+str(gap), lambda gap=gap: literal(0, {
            'get_current_data': lambda: {'A': SimpleNamespace(day_open=10.+gap)},
            'get_price': lambda **k: pd.DataFrame({'code': ['A'], 'close': [10.]})})['stock_filter4'](['A'], '2024-04-05'))
    dif = {'positive': .15, 'equal': .1, 'upper_equal': .2, 'dea_zero': .05}
    dea = {'positive': .1, 'equal': .1, 'upper_equal': .1, 'dea_zero': 0.}
    add('short_MACD_difference_not_returned_histogram', lambda: literal(0, {
        'MACD': lambda **k: (dif, dea, dict.fromkeys(dif, 999.))})['stock_filter2'](list(dif), '2024-04-05'))
    empty = pd.DataFrame(columns=['time', 'code', 'close', 'volume'])
    add('short_empty31_price_table', lambda: literal(0, {'get_price': lambda **k: empty})['stock_filter1']([], '2024-04-05'))
    for immediate in (False, True):
        add('short_two_buys_'+str(immediate), lambda immediate=immediate: short_orders(immediate=immediate))
    add('short_live_sell_dictionary_mutation', lambda: short_orders(holdings=('old',), immediate=True))
    add('short_one_existing_holding_blocks_buys', lambda: short_orders(holdings=('A',)))
    add('short_limit_equal_blocks_buy', lambda: short_orders(limit=True))
    for time, price, limit, closeable in [('11:25:00', 10., 12., 100), ('11:25:00', 11., 12., 100),
            ('11:25:00', 12., 12., 100), ('14:30:00', 9., 12., 100), ('14:30:00', 11., 12., 0)]:
        add(f'limit_sell_{time}_{price}_{closeable}', lambda time=time, price=price, limit=limit, closeable=closeable:
            sell78(time, price, limit=limit, closeable=closeable))
    for number, opens in [(1, [100., 101., 99., 101.01]), (2, [96., 97., 97.01, 95.99])]:
        for opening in opens: add(f'limit_open_{number}_{opening}', lambda number=number, opening=opening: opening78(number, opening))
    for number, prev, close, opening in [(1, 100., 105., 105.), (1, 100., 104., 104.),
            (1, 100., 106., 106.), (2, 100., 110., 104.), (2, 100., 110., 110.)]:
        add(f'limit_gain_{number}_{close}_{opening}', lambda number=number, prev=prev, close=close, opening=opening:
            gain78(number, prev, close, opening, negative_labels=True))
    add('limit_RangeIndex_negative_labels_swallowed', lambda: gain78(1, 100., 105., 105.))

    def risk(close):
        calls = []
        ns = literal(1, {'get_previous_trade_day': lambda *a: dt.date(2024, 4, 5),
            'get_price': lambda *a, **k: pd.DataFrame({'close': [close], 'low': [close], 'low_limit': [90.]}),
            'order_target_value': lambda *a: calls.append(list(a))})
        ns['get_previous_trade_day'] = lambda *a: dt.date(2024, 4, 5)
        ns['sell_if_limit_down_yesterday'](SimpleNamespace(portfolio=SimpleNamespace(
            positions={'A': SimpleNamespace(avg_cost=100., closeable_amount=0)})))
        return {'calls': calls, 'closeable_amount_fixture': 0}
    add('limit_risk_minus4_and_no_sellable_check', lambda: risk(96.))
    add('limit_risk_above_minus4', lambda: risk(96.01))
    def duplicate_buy():
        calls = []
        ns = literal(1, {'get_previous_trade_day': lambda *a: dt.date(2024, 4, 8),
            'prepare_stock_list': lambda *a: ['A'], 'prepare_stock_list2': lambda *a: ['A'],
            'get_security_info': lambda *a: SimpleNamespace(display_name='fixture'),
            'order_target_value': lambda *a: calls.append(list(a))})
        ns.update(get_previous_trade_day=lambda *a: dt.date(2024, 4, 8),
                  prepare_stock_list=lambda *a: ['A'], prepare_stock_list2=lambda *a: ['A'])
        ns['buy'](SimpleNamespace(portfolio=SimpleNamespace(cash=1000.)))
        return calls
    add('limit_combined_candidates_not_deduplicated', duplicate_buy)
    def date_boundary(n):
        ns = literal(1, {'get_trade_days': lambda **k: [dt.date(2024, 4, 5), dt.date(2024, 4, 8)]})
        result = ns['get_previous_trade_day'](SimpleNamespace(current_dt=dt.datetime(2024, 4, 8)), n)
        return None if result is None else result.isoformat()
    for n in (0, 1, 2, 3): add('limit_trade_day_n_'+str(n), lambda n=n: date_boundary(n))
    for days in (60, 61):
        add('limit_listing_days_'+str(days), lambda days=days: literal(1, {'get_security_info': lambda *a:
            SimpleNamespace(start_date=dt.date(2024, 4, 8)-dt.timedelta(days=days))})['filter_new_stock'](['A'], dt.date(2024, 4, 8)))
    add('limit_flat_relative_position', lambda: literal(1, {'get_price': lambda *a, **k:
        pd.DataFrame({'high': [10.], 'low': [10.], 'close': [10.]})})['get_relative_position_stocks']('2024-04-05', 15, ['A']))
    return {'status': 'original_function_diagnostics_only', 'cases': cases, 'case_count': len(cases),
            'source_sha256': list(SOURCE_SHA), 'synthetic_operands': True,
            'complete_original_pool_signals': 0, 'new_trading_backtests': 0,
            'original_strategy_complete': False, 'economic_variant_approved': False}
