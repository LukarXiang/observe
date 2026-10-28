"""Execute pinned original selection and order paths on explicit synthetic API inputs."""
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from scripts import review_strategy_batch90 as stage
from scripts.verify_strategy_batch3 import require


def limit_selection(total=99, last=110., peak=115., pre=100.):
    stocks = ['600000.XSHG', '600001.XSHG']
    pool = stocks + [f'300{i:03d}.XSHE' for i in range(total-2)]
    calls = []
    def days(**kwargs):
        calls.append(['trade_days', str(kwargs['end_date']), kwargs['count']])
        return [dt.date(2019, 8, 1), dt.date(2021, 1, 4)]
    def securities(kind, day):
        calls.append(['securities', kind, str(day)])
        return pd.DataFrame(index=pool)
    def prices(selected, **kwargs):
        require(selected == stocks and kwargs['count'] == 120 and
                kwargs['fields'] == ['close', 'pre_close', 'high_limit'], 'Original Panel request changed')
        calls.append(['price', selected, kwargs['count'], str(kwargs['end_date'])])
        return {'close': pd.DataFrame({stocks[0]: [peak, last], stocks[1]: [100., 100.]}),
            'pre_close': pd.DataFrame({stocks[0]: [pre, pre], stocks[1]: [100., 100.]}),
            'high_limit': pd.DataFrame({stocks[0]: [last, last], stocks[1]: [110., 110.]})}
    ns = stage.namespace(0, {'get_trade_days': days, 'get_all_securities': securities, 'get_price': prices})
    with redirect_stdout(StringIO()):
        actual = ns['select'](NS(previous_date=dt.date(2021, 1, 4)))
    return {'members_sorted_for_comparison_only': sorted(actual), 'calls': calls,
            'original_set_order_not_fixed': True, 'synthetic_Panel_not_platform_adapter': True}


def shortline_selection(dif=1., dea=.95, flow=(1., -1.), day_open=9., ma_pass=True, second_macd=(999., -999.)):
    stocks = ['600001.XSHG', '600000.XSHG']; calls = []; g = NS()
    context = NS(previous_date=dt.date(2021, 1, 4), current_dt=dt.datetime(2021, 1, 5, 8, 55))
    def means(selected, end_date, count):
        require(selected == stocks and count in (5, 10, 20, 30), 'Original mean request changed')
        current = end_date == context.previous_date
        calls.append(['mean', str(end_date), count])
        values = {True: {5: 10., 10: 11., 20: 21., 30: 31.}, False: {5: 11., 10: 10., 20: 20., 30: 30.}}
        close = values[current][count]
        if not ma_pass and current and count == 5:
            close = 11.
        volume = (110. if count == 5 else 120.) if current else (101. if count == 5 else 100.)
        return pd.DataFrame({'code': selected, 'close': [close]*2, 'volume': [volume]*2})
    def macd(**kwargs):
        require((kwargs['SHORT'], kwargs['LONG'], kwargs['MID'], kwargs['unit']) == (12, 26, 9, '1d'),
                'Original MACD parameters changed')
        calls.append(['MACD', kwargs['security_list'], str(kwargs['check_date'])])
        x, y = (dif, dea) if kwargs['check_date'] == context.previous_date else second_macd
        return ({s: x for s in kwargs['security_list']}, {s: y for s in kwargs['security_list']}, {})
    def money(**kwargs):
        require(kwargs['count'] == 1 and kwargs['fields'] == ['change_pct'], 'Original flow request changed')
        calls.append(['flow', kwargs['security_list'], str(kwargs['end_date'])])
        value = flow[0 if kwargs['end_date'] == context.previous_date else 1]
        return pd.DataFrame({'change_pct': [] if value is None else [value]})
    ns = stage.namespace(1, {'g': g, 'datetime': dt,
        'get_all_securities': lambda *a, **k: pd.DataFrame({'display_name': ['fixture']*2}, index=stocks),
        'get_stock_price': means, 'MACD': macd, 'get_money_flow': money,
        'get_trade_days': lambda **k: [dt.date(2021, 1, 1), context.previous_date],
        'get_current_data': lambda: {s: NS(day_open=day_open) for s in stocks},
        'get_price': lambda **k: pd.DataFrame({'close': [10.]})})
    # Bind the synthetic mean request after loading unchanged definitions.
    ns['get_stock_price'] = means
    with redirect_stdout(StringIO()):
        ns['before_trading_start'](context)
    return {'buy_list': g.buy_list, 'calls': calls, 'synthetic_means_MACD_flow_quotes': True}


def limit_orders(pool=('a', 'b', 'c'), immediate=False, rejected=False):
    positions = {}; orders = []; portfolio = NS(positions=positions, available_cash=20000.)
    def order(stock, cash):
        orders.append([stock, cash])
        if immediate and not rejected:
            positions[stock] = NS(closeable_amount=0)
            portfolio.available_cash -= cash
        return None
    ns = stage.namespace(0, {'g': NS(pool=list(pool), max_holding=2),
        'get_current_data': lambda: {s: NS(last_price=110., name=s) for s in pool},
        'attribute_history': lambda *a: {'close': np.array([100.])},
        'order_target_value': order, 'log': NS(info=lambda *a: None)})
    ns['toBuy'](NS(portfolio=portfolio))
    return {'orders': orders, 'positions': list(positions), 'cash': portfolio.available_cash,
            'synthetic_mutation_policy': 'immediate' if immediate else 'deferred', 'rejected': rejected}


def shortline_orders(closeable=100, price=10., cash=2000., immediate_sell=False, held=True):
    positions = {'held': NS(closeable_amount=closeable, avg_cost=10.)} if held else {}
    orders = []; portfolio = NS(positions=positions, available_cash=cash)
    def target(stock, quantity):
        orders.append(['sell', stock, quantity])
        if immediate_sell:
            positions.pop(stock)
        return None
    names = ['held', 'a', 'b']
    ns = stage.namespace(1, {'g': NS(buy_stock_limit=1, buy_list=['a', 'b']),
        'get_current_data': lambda: {s: NS(low_limit=9., high_limit=11.) for s in names},
        'order_target': target, 'order_value': lambda *a: orders.append(['buy', *a])})
    with redirect_stdout(StringIO()):
        try:
            ns['handle_data'](NS(portfolio=portfolio, current_dt=dt.datetime(2021, 1, 5, 9, 31)),
                              {s: NS(close=price) for s in names})
        except RuntimeError as exc:
            require(immediate_sell and 'dictionary changed size' in str(exc), 'Unexpected lifecycle failure')
            return {'orders': orders, 'exception': type(exc).__name__, 'synthetic_dictionary_not_platform_mapping': True}
    return {'orders': orders, 'exception': None, 'positions': list(positions), 'cash': portfolio.available_cash,
            'synthetic_rejection_or_deferred_orders': True}


def diagnostics():
    rows = []
    for name, args, expected in (
        ('89_original_denominator_100_equal_one_percent', {'total': 100}, []),
        ('89_original_denominator_99_above_one_percent', {}, ['600000.XSHG']),
        ('89_current_maximum_excluded', {'peak': 110.}, []),
        ('89_near_high_exact_1_07_excluded', {'peak': 110.*1.07}, []),
        ('89_near_high_just_inside', {'peak': 110.*1.07-.001}, ['600000.XSHG']),
        ('89_limit_gain_exact_1_09_excluded', {'last': 109., 'peak': 115.}, [])):
        result = limit_selection(**args)
        require(result['members_sorted_for_comparison_only'] == expected, f'{name}: original select differs')
        rows.append({'case': name, 'actual': result, 'expected_members': expected})
    for name, args, expected in (
        ('8_full_injected_selection_sorted', {}, ['600000.XSHG', '600001.XSHG']),
        ('8_second_MACD_values_unused', {'second_macd': (-999., 999.)}, ['600000.XSHG', '600001.XSHG']),
        ('8_MA_strict_equal_blocks', {'ma_pass': False}, []),
        ('8_MACD_dif_zero_blocks', {'dif': 0.}, []),
        ('8_MACD_dea_zero_blocks', {'dea': 0.}, []),
        ('8_MACD_difference_zero_blocks', {'dea': 1.}, []),
        ('8_MACD_difference_negative_point1_blocks', {'dif': .2, 'dea': .1}, []),
        ('8_flow_reverse_signs_pass', {'flow': (-1., 1.)}, ['600000.XSHG', '600001.XSHG']),
        ('8_flow_same_signs_block', {'flow': (1., 1.)}, []),
        ('8_flow_zero_blocks', {'flow': (0., -1.)}, []),
        ('8_flow_missing_blocks', {'flow': (None, -1.)}, []),
        ('8_open_equal_previous_close_blocks', {'day_open': 10.}, []),
        ('8_preopen_zero_quote_would_pass_synthetic_only', {'day_open': 0.}, ['600000.XSHG', '600001.XSHG'])):
        result = shortline_selection(**args)
        require(result['buy_list'] == expected, f'{name}: original selection differs')
        rows.append({'case': name, 'actual': result, 'expected_buy_list': expected})
    for name, args, expected in (
        ('89_deferred_orders_can_exceed_two_slots', {}, [['a', 10000.], ['b', 10000.], ['c', 10000.]]),
        ('89_immediate_fill_stops_after_two', {'immediate': True}, [['a', 10000.], ['b', 10000.]]),
        ('89_rejection_does_not_reserve_slots', {'immediate': True, 'rejected': True}, [['a', 10000.], ['b', 10000.], ['c', 10000.]]),
        ('89_set_order_permutation_changes_intents', {'pool': ('c', 'b', 'a'), 'immediate': True}, [['c', 10000.], ['b', 10000.]])):
        result = limit_orders(**args); require(result['orders'] == expected, f'{name}: original orders differ')
        rows.append({'case': name, 'actual': result, 'expected_orders': expected})
    for name, args, expected, exception in (
        ('8_deferred_sell_holding_blocks_buy', {}, [['sell', 'held', 0]], None),
        ('8_Tplus1_no_sell', {'closeable': 0}, [], None),
        ('8_at_low_limit_no_sell_or_buy', {'price': 9.}, [], None),
        ('8_at_high_limit_no_sell_or_buy', {'price': 11.}, [], None),
        ('8_rejected_buy_still_breaks_after_first', {'held': False}, [['buy', 'a', 2000.]], None),
        ('8_cash_below_100shares_blocks', {'held': False, 'cash': 999.}, [], None),
        ('8_cash_equal_100shares_submits_all_no_fee_reserve', {'held': False, 'cash': 1000.}, [['buy', 'a', 1000.]], None),
        ('8_synchronous_plain_dict_removal_raises', {'immediate_sell': True}, [['sell', 'held', 0]], 'RuntimeError')):
        result = shortline_orders(**args)
        require(result['orders'] == expected and result['exception'] == exception, f'{name}: original lifecycle differs')
        rows.append({'case': name, 'actual': result, 'expected_orders': expected, 'expected_exception': exception})
    return {'cases': rows, 'case_count': len(rows), 'synthetic_only': True,
        'not_a_backtest': True, 'original_signal_rows': 0, 'new_trading_backtests': 0,
        'platform_equivalence_proven': False,
        'limits': ['Pinned original definitions on explicit synthetic Panel/means/MACD/flow/quote contracts',
            'No original historical full universe or MACD seed/flow/limit/minute input fabricated',
            'Set membership sorted only for comparison, actual buy order not stabilized',
            'Ordinary dictionary mutation demonstrates a boundary, not platform position-map behavior',
            'Rejected/deferred/immediate API stubs expose control flow, not actual fills or Book accounting']}
