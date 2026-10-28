"""Supplement unchanged macro callbacks at state boundaries, with explicit stubs."""
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import research_strategy_batch102 as original
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch102/20261011-state-boundaries')


def handle(mode=1, times=0, *, cash=1000., price=100., year_mean=0., recovery=False, right_side=False, immediate=False):
    calls = []; ns, _ = original.initialize(0)
    ns['g'].mode = mode; ns['g'].times = times; ns['g'].record = 100.
    portfolio = SimpleNamespace(total_value=1000., available_cash=cash, positions={})
    ns.update(get_PMI=lambda *a: [50.+i*.1 for i in range(13)] if recovery else [49.]*13,
        calc_provide_and_need=lambda *a: (49., 51.), calc_save=lambda *a: (49., 51.),
        change_to_yeak_k=lambda: {'injected': year_mean}, year_move_average=lambda *a: year_mean,
        get_bars=lambda symbol, *a, **k: pd.Series([price if symbol == '000300.XSHG' else 10.]),
        day_move_average=lambda n, now: ({(5, True): 105., (10, True): 104., (20, True): 103.,
            (30, True): 102., (30, False): 101.}[(n, now)] if right_side else 100.),
        order_target=lambda *a: calls.append(['target', *a]))
    def buy(symbol, value):
        calls.append(['value', symbol, float(value)])
        if immediate: portfolio.available_cash -= value
    ns['order_value'] = buy
    ns['handle'](SimpleNamespace(current_dt=dt.datetime(2024, 4, 9, 14, 55), previous_date=dt.date(2024, 4, 8), portfolio=portfolio))
    return {'calls': calls, 'mode': ns['g'].mode, 'times': ns['g'].times, 'record': ns['g'].record,
        'cash_stub_after': portfolio.available_cash, 'all_price_macro_MA_and_feedback_inputs_are_stubs': True}


def exit_state(*, score=97., latched=False, raw=51., inventory=49., index_price=100., prior_cross=True):
    calls = []; ns, _ = original.initialize(0); ns['g'].mode = 1; ns['g'].over_hot = latched
    def save_inputs(*a): calls.append(['inventory_query']); return raw, inventory
    ns.update(calc_provide_and_need=lambda *a: (score, score), calc_save=save_inputs,
        get_bars=lambda *a, **k: pd.Series([index_price]),
        day_move_average=lambda n, now: {(5, True): 98., (5, False): 102. if prior_cross else 100.,
            (20, True): 100., (20, False): 100., (120, True): 100.}[n, now],
        order_target=lambda *a: calls.append(['target', *a]), order_value=lambda *a: calls.append(['value', *a]))
    ns['sell_stock_security'](dt.date(2024, 4, 9), 1000., ['510300.XSHG'])
    return {'calls': calls, 'mode': ns['g'].mode, 'over_hot': ns['g'].over_hot, 'times': ns['g'].times,
        'all_macro_price_and_MA_inputs_are_stubs': True}


def diagnostics():
    with network_blocked():
        cases = {
            'cycle_recovers_to_long_first4percent': handle(mode=0, times=-1, recovery=True),
            'mode2_recovers_to_long_keeps_multiplier': handle(mode=2, times=3, recovery=True),
            'annual_entry_then_same_callback_drawdown_double_order': handle(mode=0, times=-1, price=80., year_mean=100.),
            'no_cash_no_order_but_multiplier_and_record_advance': handle(price=97.9, cash=0.),
            'first_buy_then_right_side_cash_deferred': handle(times=-1, right_side=True),
            'first_buy_then_right_side_cash_immediate': handle(times=-1, right_side=True, immediate=True),
            'overheat_exact97_no_inventory': exit_state(),
            'latched_overheat_below97_still_queries_inventory': exit_state(score=80., latched=True, raw=50.),
            'inventory_exact50_does_not_switch': exit_state(score=100., raw=50.),
            'MA120_price_equality_blocks_exit_after_switch': exit_state(score=100.),
            'prior_MA_equality_blocks_exit_after_switch': exit_state(score=100., index_price=101., prior_cross=False),
        }
    result = {'cases': cases, 'case_count': len(cases), 'synthetic_only': True, 'compatibility_repairs': False,
        'original_functions_unchanged': True, 'new_trading_backtests': 0}
    require(json.loads(json.dumps(result, allow_nan=False)) == result, 'Supplemental JSON shape differs')
    return result


if __name__ == '__main__':
    require(not DIRECTORY.exists(), 'Immutable boundary output exists'); DIRECTORY.mkdir(parents=True)
    doc = diagnostics(); save(DIRECTORY/'diagnostics.json', doc)
    print(json.dumps({'case_count': doc['case_count'], 'new_trading_backtests': 0}))
