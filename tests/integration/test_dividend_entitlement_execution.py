"""Frozen Store actions reach both loops without a second dividend implementation."""
from datetime import date

import pandas as pd
import pytest

from observe.data.store import Store
from observe.execution import InputBlocked, _action, build, load
from observe.loop import fixed_order_replay, run_loop

from tests.unit.test_dividend_entitlements import ACTION, D, I, ZERO


DAYS = [date(2024, 10, 17), *D]


def tables(actions=(ACTION,)):
    return {
        'calendar': pd.DataFrame({'date': DAYS, 'is_open': True}),
        'instruments': pd.DataFrame([{'instrument': I, 'kind': 'etf', 'board': 'other',
                                      'list_date': date(2013, 3, 25)}]),
        'bars_1d': pd.DataFrame([{'instrument': I, 'date': d, 'is_trading': True,
                                  'open': 10, 'close': 10, 'preclose': 10, 'amount': 1000000}
                                 for d in DAYS]),
        'corp_actions': pd.DataFrame(actions),
    }


def test_frozen_record_actions_run_and_replay_after_ex_date_sale(tmp_path):
    store = Store(tmp_path)
    parts = {name: {'all': store.write_partition(name, 'all', frame)}
             for name, frame in tables().items()}
    store.publish(store.write_batch(parts)); snapshot = store.snapshot('synthetic dividend entitlement')
    inputs = build(load(store, store.state(snapshot), DAYS[0], DAYS[-1], instruments=[I]),
                   DAYS[0], DAYS[-1], liquidity_window=1)
    assert inputs.info['actions'] == 1 and inputs.info['dividend_record_events'] == 1
    assert inputs.actions[D[0]][0] == inputs.actions[D[1]][0]
    assert all(not x for x in inputs.candidates.values())  # Does not grant ETF stock eligibility.
    signals = {d: {I: {'target_value': 1000., 'enter_when_empty': False,
                       'exit': False, 'skip_when_empty': False}} for d in DAYS}
    signals[DAYS[0]][I]['enter_when_empty'] = True
    signals[D[0]][I]['exit'] = True
    book, orders = run_loop(inputs.dates, inputs.market, {}, 1000, ZERO, actions=inputs.actions,
                            calendar=inputs.calendar, construction='conditional_values',
                            conditional_values_by_date=signals, rebalance_every=1)
    assert [(r['exec_date'], r['side'], r['qty_filled']) for r in orders] == [
        (D[0], 'buy', 100), (D[1], 'sell', 100)]
    assert book.equity_rows[2]['receivable'] == 8.04
    assert book.cash == 1008.04 and not book.receivable and book.pos(I).qty == 0
    replay, _ = fixed_order_replay(orders, inputs.dates, inputs.market, 1000, ZERO,
                                   actions=inputs.actions, calendar=inputs.calendar)
    assert replay.equity_rows == book.equity_rows
    assert replay.cash_events == book.cash_events


@pytest.mark.parametrize('patch', [
    {'record_date': None}, {'pay_date': None}, {'cash_entitlement': 'unknown'},
    {'record_date': D[1]}, {'pay_date': D[0]}, {'bonus_ratio': .5},
])
def test_adapter_blocks_invalid_explicit_entitlements(patch):
    with pytest.raises(InputBlocked, match='invalid_dividend_entitlement'):
        _action({**ACTION, **patch})


def test_duplicate_and_record_outside_run_are_blocked():
    with pytest.raises(InputBlocked, match='duplicate_dividend_entitlement'):
        build(tables((ACTION, ACTION)), DAYS[0], DAYS[-1], liquidity_window=1)
    with pytest.raises(InputBlocked, match='dividend_record_outside_run'):
        build(tables(), D[1], DAYS[-1], liquidity_window=1)
    with pytest.raises(InputBlocked, match='dividend_on_closed_day'):
        build(tables(({**ACTION, 'record_date': date(2024, 10, 19)},)),
              DAYS[0], DAYS[-1], liquidity_window=1)


def test_record_at_end_is_replayed_even_when_ex_date_is_outside_range():
    inputs = build(tables(), DAYS[0], D[0], liquidity_window=1)
    assert inputs.info['actions'] == 0 and inputs.info['dividend_record_events'] == 1
    assert list(inputs.actions) == [D[0]]


def test_optional_columns_from_other_actions_do_not_change_policy():
    converted = _action({**ACTION, 'convert_to': float('nan'), 'convert_ratio': float('nan')})
    assert converted['cash_entitlement'] == 'record_close' and 'convert_to' not in converted
    legacy = _action({**ACTION, 'cash_entitlement': float('nan')})
    assert 'cash_entitlement' not in legacy
