import math

import pytest

from observe.ledger.book import Book, Position
from observe.loop import run_loop
from observe.portfolio import slot_value_orders
from observe.strategy_rsi import POOL, flags, slot_batch
from tests.unit.test_portfolio_loop import D, flat, ZERO


def batch(buys = (), sells = (), slots = 2, pool = ('A', 'B', 'A')):
    return {'pool': list(pool), 'buys': list(buys), 'sells': list(sells), 'max_positions': slots}


@pytest.mark.parametrize('value,buy,sell', [(0, False, True), (9.99, False, True), (10, False, False), (15.99, False, False),
                                         (16, True, False), (24.99, True, False), (25, False, False), (85.99, False, False), (86, False, True)])
def test_integer_thresholds(value, buy, sell):
    assert flags(value) == (int(value), buy, sell)


def test_stable_duplicate_sort_and_schema():
    signals = [{'instrument': i, 'priority': 20 if i in POOL[:3] else 50, 'buy': i in POOL[:3], 'sell': False} for i in dict.fromkeys(POOL)]
    assert slot_batch(signals)['buys'] == [*POOL[:3], POOL[29]]
    for changed in ({'buy': 1}, {'priority': 20.0}, {'priority': 0}, {'other': 1}):
        invalid = [dict(r) for r in signals]; invalid[0].update(changed)
        with pytest.raises(ValueError): slot_batch(invalid)
    with pytest.raises(ValueError): slot_batch(signals[:-1])
    with pytest.raises(ValueError): flags(math.nan)


def test_sequential_postfill_cash_partial_and_duplicate_retries():
    quotes = flat('A')[D[0]] | flat('B')[D[0]]
    quotes['A']['avg_amount_20d'] = 20000
    book = Book(100000); generator = slot_value_orders(batch(['A', 'B', 'A']), book, quotes, ZERO, D[0])
    first = next(generator); assert first['amount'] == 50000
    book.execute(first, quotes['A'], D[0], ZERO)
    second = next(generator); assert second['instrument'] == 'B' and second['amount'] == book.cash == 99000
    book.execute(second, quotes['B'], D[0], ZERO)
    assert list(generator) == [] and sum(p.qty > 0 for p in book.positions.values()) == 2
    quotes['A']['suspended'] = True
    book = Book(100000); orders = []
    for o in slot_value_orders(batch(['A', 'A']), book, quotes, ZERO, D[0]):
        orders.append(book.execute(o, quotes['A'], D[0], ZERO))
    assert len(orders) == 2 and all(o['status'] == 'rejected' for o in orders)


@pytest.mark.parametrize('liquidity,remaining,free', [(20000, 900, 1), (0, 1000, 1), (1e9, 0, 2)])
def test_partial_or_failed_sale_retains_slot(liquidity, remaining, free):
    quotes = flat('A')[D[0]] | flat('B')[D[0]]; quotes['A']['avg_amount_20d'] = liquidity
    book = Book(10000); book.positions['A'] = Position(qty = 1000, last_price = 10)
    gen = slot_value_orders(batch(['B'], ['A']), book, quotes, ZERO, D[0]); sell = next(gen)
    book.execute(sell, quotes['A'], D[0], ZERO)
    buy = next(gen)
    assert book.pos('A').qty == remaining and buy['amount'] == book.cash / free


def test_source_opposite_limit_guards_t1_and_no_existing_trim():
    quotes = flat('A')[D[0]] | flat('B')[D[0]]; book = Book(10000)
    book.positions['A'] = Position(qty = 1000, last_price = 10)
    quotes['A']['open'] = 11; quotes['B']['open'] = 9
    assert list(slot_value_orders(batch(['B'], ['A']), book, quotes, ZERO, D[0])) == []
    quotes['A']['open'] = 10; quotes['B']['open'] = 10; book.pos('A').today_buy = 1000
    assert [o['side'] for o in slot_value_orders(batch(['B'], ['A']), book, quotes, ZERO, D[0])] == ['buy']
    assert list(slot_value_orders(batch(['A']), book, quotes, ZERO, D[0])) == []


def test_loop_uses_previous_signal_at_open_never_refills_and_rejects_missing():
    market = {d: flat('A')[d] | flat('B')[d] for d in D}
    signals = {d: batch(['A', 'B']) for d in D}
    book, orders = run_loop(D, market, {}, 100000, ZERO, construction = 'signal_slots',
                            slot_signals_by_date = signals, refill_between_rebalance = False)
    assert len(orders) == 2 and all(o['exec_date'] == D[1] and o['decision_date'] == D[0] for o in orders)
    assert book.pos('A').qty == book.pos('B').qty == 5000
    with pytest.raises(ValueError, match = '覆盖'):
        run_loop(D, market, {}, 100000, ZERO, construction = 'signal_slots', slot_signals_by_date = {})
