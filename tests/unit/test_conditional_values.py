from types import SimpleNamespace

import pytest

from observe.loop import run_loop
from observe.portfolio import conditional_value_orders
from tests.unit.test_rule_portfolio import ZERO
from tests.unit.test_portfolio_loop import D, flat


def signal(enter = False, exit = False, skip = False):
    return {'A': {'target_value': 20000., 'enter_when_empty': enter, 'exit': exit, 'skip_when_empty': skip}}


@pytest.mark.parametrize('cash', [20000, 1000000])
def test_fixed_amount_no_refill_or_trim_on_price_change(cash):
    market = flat('A'); market[D[2]]['A']['close'] = 11
    book, orders = run_loop(D, market, {}, cash, ZERO, rebalance_every = 1, construction = 'conditional_values',
                            conditional_values_by_date = {d: signal(enter = True) for d in D})
    assert len(orders) == 1 and orders[0]['amount'] == 20000 and orders[0]['qty_filled'] == 2000
    assert book.pos('A').qty == 2000 and book.cash == cash - 20000


def test_actual_other_position_and_continue_suppress_entry():
    assert conditional_value_orders(signal(enter = True, skip = True), {}) == []
    assert conditional_value_orders(signal(enter = True), {'B': SimpleNamespace(qty = 100)}) == []
    assert conditional_value_orders(signal(exit = True), {'A': SimpleNamespace(qty = 100)})[0]['qty'] == 'all'
    assert conditional_value_orders(signal(), {'A': SimpleNamespace(qty = 100)}) == []


def test_partial_entry_never_refills_and_rejected_entry_can_retry():
    market = flat('A'); market[D[1]]['A']['avg_amount_20d'] = 20000
    book, orders = run_loop(D, market, {}, 1000000, ZERO, rebalance_every = 1, construction = 'conditional_values',
                            conditional_values_by_date = {d: signal(enter = True) for d in D})
    assert len(orders) == 1 and book.pos('A').qty == 100
    market[D[1]]['A']['suspended'] = True
    book, orders = run_loop(D, market, {}, 1000000, ZERO, rebalance_every = 1, construction = 'conditional_values',
                            conditional_values_by_date = {d: signal(enter = True) for d in D})
    assert len(orders) == 2 and orders[0]['qty_filled'] == 0 and orders[1]['qty_filled'] == 2000


@pytest.mark.parametrize('persistent,expected', [(True, 0), (False, 1900)])
def test_partial_exit_retries_only_under_new_exit_condition(persistent, expected):
    market = flat('A'); market[D[3]]['A']['avg_amount_20d'] = 20000
    signals = {d: signal() for d in D}; signals[D[0]] = signal(enter = True); signals[D[2]] = signal(exit = True)
    signals[D[3]] = signal(exit = persistent)
    book, orders = run_loop(D, market, {}, 1000000, ZERO, rebalance_every = 1, construction = 'conditional_values', conditional_values_by_date = signals)
    assert book.pos('A').qty == expected
    assert orders[1]['side'] == 'sell' and orders[1]['qty_filled'] == 100
    assert len(orders) == (3 if persistent else 2)


@pytest.mark.parametrize('changes', [{'target_value': -1}, {'target_value': float('nan')}, {'target_value': True},
                                    {'enter_when_empty': 1}, {'exit': None}, {'other': 1}, {'enter_when_empty': True, 'exit': True}])
def test_invalid_conditional_schema(changes):
    value = signal(); value['A'].update(changes)
    with pytest.raises(ValueError): conditional_value_orders(value, {})


def test_missing_dates_or_multiple_instruments_rejected():
    with pytest.raises(ValueError, match = '覆盖'):
        run_loop(D, {}, {}, 1000, ZERO, construction = 'conditional_values', conditional_values_by_date = {})
    with pytest.raises(ValueError, match = '单股'): conditional_value_orders({**signal(), 'B': signal()['A']}, {})
