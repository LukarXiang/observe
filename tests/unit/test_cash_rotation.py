import pytest

from observe.ledger import Book
from observe.ledger.book import Position
from observe.loop import run_loop
from observe.portfolio import cash_rotation_orders, validate_cash_rotation
from observe.strategy_rotation import rotation_batch
from tests.unit.test_portfolio_loop import D, flat, ZERO


def test_rotation_partial_sale_still_buys_with_postfill_cash():
    book = Book(10000); book.positions['A'] = Position(qty = 1000, last_price = 10)
    quotes = flat('AB')[D[0]]; quotes['A']['avg_amount_20d'] = 20000
    gen = cash_rotation_orders({'sell': 'A', 'buy': 'B'}, book); sell = next(gen)
    book.execute(sell, quotes['A'], D[0], ZERO); buy = next(gen)
    assert book.pos('A').qty == 900 and buy['amount'] == book.cash == 11000
    book.execute(buy, quotes['B'], D[0], ZERO)
    assert book.pos('B').qty == 1100 and list(gen) == []


def test_failed_entry_has_no_retry_and_no_mean_exit_or_existing_trim():
    market = flat('AB'); market[D[1]]['A']['suspended'] = True
    signals = {d: {'buy': None, 'sell': None} for d in D}; signals[D[0]] = {'buy': 'A', 'sell': 'B'}
    book, orders = run_loop(D, market, {}, 100000, ZERO, construction = 'signal_cash_rotation', cash_rotations_by_date = signals, refill_between_rebalance = False)
    assert len(orders) == 1 and orders[0]['status'] == 'rejected' and book.cash == 100000
    market[D[1]]['A']['suspended'] = False
    book, orders = run_loop(D, market, {}, 100000, ZERO, construction = 'signal_cash_rotation', cash_rotations_by_date = signals, refill_between_rebalance = False)
    assert len(orders) == 1 and book.pos('A').qty == 10000


@pytest.mark.parametrize('batch', [{'buy': 'A', 'sell': 'A'}, {'buy': 'A', 'sell': None}, {'buy': 1, 'sell': 'A'}, {'buy': None, 'sell': None, 'other': 1}])
def test_invalid_rotation_schema(batch):
    with pytest.raises(ValueError): validate_cash_rotation(batch)


def test_rotation_full_pair_flags_required():
    pair = ['A', 'B']; rows = [{'instrument': 'A', 'buy': True, 'sell': False}, {'instrument': 'B', 'buy': False, 'sell': True}]
    assert rotation_batch(rows, pair) == {'buy': 'A', 'sell': 'B'}
    for invalid in (rows[:1], [rows[0], rows[0]], [rows[0], {'instrument': 'B', 'buy': False, 'sell': False}],
                    [{'instrument': 'A', 'buy': 1, 'sell': False}, rows[1]]):
        with pytest.raises(ValueError): rotation_batch(invalid, pair)
