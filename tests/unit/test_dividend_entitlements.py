"""Record-close dividend rights must survive disposal and cannot fund early orders."""
from copy import deepcopy
from datetime import date

import pytest

from observe.ledger import Book, LedgerError, Rule, RuleSet


D = [date(2024, 10, d) for d in (18, 21, 22, 23, 24)]
I = '510310.SH'
ZERO = RuleSet([Rule(start=date(2020, 1, 1), commission_rate=0, min_commission=0,
                     stamp_tax=0, transfer_fee=0, price_tick=.001)])
ACTION = {'instrument': I, 'record_date': D[0], 'ex_date': D[1], 'pay_date': D[4],
          'cash_per_share': .0804, 'cash_entitlement': 'record_close'}


def quote(price=10):
    return {'open': price, 'close': price, 'preclose': price}


def order(book, day, side, **extra):
    return book.execute({'instrument': I, 'side': side, **extra}, quote(), day, ZERO)


def test_record_day_purchase_gets_rights_but_record_day_sale_does_not():
    book = Book(2000, D)
    book.start_day(D[0], [ACTION])
    assert order(book, D[0], 'buy', amount=1000)['qty_filled'] == 100
    assert order(book, D[0], 'sell', qty='all')['reject_reason'] == 'not_sellable'
    book.close_day(D[0], {I: quote()})
    book.start_day(D[1], [ACTION])
    assert book.receivable[D[4]] == 8.04 and book.cash == 1000

    sold = Book(0, D)
    sold.pos(I).qty = 100
    sold.start_day(D[0], [ACTION])
    assert order(sold, D[0], 'sell', qty='all')['qty_filled'] == 100
    sold.close_day(D[0], {})
    sold.start_day(D[1], [ACTION])
    assert sum(sold.receivable.values()) == 0


def test_rights_use_record_close_not_ex_date_holdings_and_survive_sale():
    action = {**ACTION, 'ex_date': D[2]}
    book = Book(3000, D)
    book.start_day(D[0], [action])
    order(book, D[0], 'buy', amount=1000)
    book.close_day(D[0], {I: quote()})
    book.start_day(D[1]); order(book, D[1], 'sell', qty='all'); book.close_day(D[1], {})
    book.start_day(D[2], [action])
    assert book.receivable[D[4]] == 8.04
    order(book, D[2], 'buy', amount=2000)
    assert book.close_day(D[2], {I: quote()})['receivable'] == 8.04
    book.start_day(D[3])
    order(book, D[3], 'sell', qty='all')
    book.close_day(D[3], {})
    assert book.cash == 3000 and book.receivable[D[4]] == 8.04
    book.start_day(D[4])
    assert book.cash == 3008.04 and not book.receivable
    assert len([r for r in book.cash_events if r['kind'] == 'dividend_paid']) == 1


def test_zero_record_holdings_do_not_gain_dividend_from_ex_date_purchase():
    book = Book(1000, D)
    book.start_day(D[0], [ACTION]); book.close_day(D[0], {})
    book.start_day(D[1], [ACTION]); order(book, D[1], 'buy', amount=1000)
    assert sum(book.receivable.values()) == 0


def test_dividend_receivable_preserves_equity_and_is_not_spendable():
    action = {**ACTION, 'cash_per_share': 1}
    book = Book(1000, D)
    book.start_day(D[0], [action]); order(book, D[0], 'buy', amount=1000)
    assert book.close_day(D[0], {I: quote()})['equity'] == 1000
    book.start_day(D[1], [action])
    assert order(book, D[1], 'buy', amount=100)['reject_reason'] == 'cash'
    row = book.close_day(D[1], {I: quote(9)})
    assert (row['cash'], row['receivable'], row['market_value'], row['equity']) == (0, 100, 900, 1000)
    book.start_day(D[4])
    row = book.close_day(D[4], {I: quote(9)})
    assert (row['cash'], row['receivable'], row['equity']) == (100, 0, 1000)


@pytest.mark.parametrize('patch', [
    {'record_date': None}, {'pay_date': None}, {'record_date': D[1]},
    {'pay_date': D[0]}, {'cash_per_share': float('nan')}, {'cash_per_share': -1},
    {'cash_per_share': 0}, {'cash_per_share': 10**400}, {'bonus_ratio': .1}, {'rights_ratio': .1},
    {'convert_to': I, 'convert_ratio': .5}, {'cash_entitlement': 'unknown'},
])
def test_invalid_explicit_entitlements_fail_before_mutating_book(patch):
    book = Book(1000, D); before = deepcopy(book.__dict__)
    with pytest.raises(LedgerError, match='dividend|entitlement'):
        book.start_day(D[0], [{**ACTION, **patch}])
    assert book.__dict__ == before


def test_missing_record_close_and_duplicate_ex_date_fail_without_double_payment():
    book = Book(1000, D)
    with pytest.raises(LedgerError, match='record'):
        book.start_day(D[1], [ACTION])
    book.start_day(D[0], [ACTION]); order(book, D[0], 'buy', amount=1000)
    book.close_day(D[0], {I: quote()}); book.start_day(D[1], [ACTION])
    with pytest.raises(LedgerError, match='applied|duplicate'):
        book.start_day(D[1], [ACTION])
    assert sum(book.receivable.values()) == 8.04


def test_mutated_terms_and_unclosed_registration_are_rejected():
    book = Book(0, D); book.start_day(D[0], [ACTION])
    with pytest.raises(LedgerError, match='close'):
        book.start_day(D[1], [ACTION])
    with pytest.raises(LedgerError, match='date'):
        book.close_day(D[1], {})
    book.close_day(D[0], {})
    with pytest.raises(LedgerError, match='changed|terms'):
        book.start_day(D[1], [{**ACTION, 'cash_per_share': .1}])


def test_legacy_dividend_still_uses_ex_date_position():
    book = Book(0, D); book.pos(I).qty = 100
    legacy = {k: v for k, v in ACTION.items() if k != 'cash_entitlement'}
    book.start_day(D[1], [legacy])
    assert book.receivable[D[4]] == 8.04
