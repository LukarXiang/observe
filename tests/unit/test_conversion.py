from copy import deepcopy
from datetime import date

import pytest

from observe.execution import _action
from observe.ledger import Book, LedgerError

DAY = date(2024, 9, 23)
LATER = [date(2024, 9, 24), date(2024, 9, 25)]


def conversion(ratio = .5, rounding = 'floor', destination = '510310.SH'):
    return {'instrument': '510310.SH', 'ex_date': DAY, 'convert_to': destination,
            'convert_ratio': ratio, 'convert_rounding': rounding}


def holding(qty = 1000):
    book = Book(0, [DAY, *LATER]); p = book.pos('510310.SH')
    p.qty, p.cost, p.last_price = qty, 2, 2
    return book, p


def test_same_code_consolidation_preserves_holding_cost_and_equity():
    book, p = holding()
    book.start_day(DAY, [conversion()])
    assert (p.qty, p.sellable, p.cost, p.last_price) == (500, 500, 4, 4)
    assert book.close_day(DAY, {'510310.SH': {'close': 4}})['equity'] == 2000
    assert book.issues[-1]['qty'] == 500


def test_official_ceil_consolidation_and_explicit_fraction_record():
    book, p = holding()
    book.start_day(DAY, [conversion(.49977589, 'ceil')])
    assert p.qty == 500 and p.cost * p.qty == 2000
    assert p.last_price == pytest.approx(2 / .49977589)
    fractions = [a for a in book.assumptions if a['field'] == 'conversion_fraction']
    assert fractions[0]['value'] == pytest.approx(-.22411)
    assert fractions[0]['rounding'] == 'ceil'
    assert book.cash == 0 and not book.cash_events


@pytest.mark.parametrize('rounding,qty,pending,parts', [
    ('floor', 151, 101, [50, 51]), ('ceil', 152, 101, [50, 51]),
])
def test_conversion_transforms_pending_listings_and_keeps_them_locked(rounding, qty, pending, parts):
    book, p = holding(303); p.pending = 202
    for day in LATER: book.listing[day].append(('510310.SH', 101))
    book.start_day(DAY, [conversion(.5, rounding)])
    assert (p.qty, p.pending, p.sellable) == (qty, pending, qty - pending)
    assert [book.listing[d][0][1] for d in LATER] == parts
    book.start_day(LATER[0]); assert p.pending == 51
    book.start_day(LATER[1]); assert p.pending == 0 and p.sellable == qty


def test_other_code_conversion_legacy_floor_still_merges_destination_cost():
    book, old = holding(101); new = book.pos('NEW'); new.qty, new.cost = 10, 3
    action = conversion(.5, destination = 'NEW'); action.pop('convert_rounding')
    book.start_day(DAY, [action])
    assert old.qty == 0 and new.qty == 60
    assert new.cost * new.qty == pytest.approx(232)
    assert new.last_price == 4


@pytest.mark.parametrize('action', [conversion(0), conversion(-1), conversion(float('nan')),
                                   conversion(float('inf')), conversion(.5, 'unknown'), conversion(.001)])
def test_invalid_conversion_cannot_consume_holding(action):
    book, p = holding(1); before = deepcopy(vars(p))
    with pytest.raises(LedgerError, match = 'conversion'): book.start_day(DAY, [action])
    assert vars(p) == before and not book.issues and not book.assumptions


def test_pending_without_listing_evidence_blocks_before_conversion():
    book, p = holding(); p.pending = 100; before = deepcopy(vars(p))
    with pytest.raises(LedgerError, match = 'conversion'): book.start_day(DAY, [conversion()])
    assert vars(p) == before


def test_execution_action_preserves_conversion_fields_without_changing_old_shape():
    original = {'instrument': '510310.SH', 'ex_date': DAY, 'cash_per_share': 0, 'bonus_ratio': 0}
    old = _action(original)
    assert not {'convert_to', 'convert_ratio', 'convert_rounding'} & set(old)
    incoming = _action({**original, **conversion(.49977589, 'ceil')})
    assert incoming['convert_to'] == '510310.SH' and incoming['convert_ratio'] == .49977589
    assert incoming['convert_rounding'] == 'ceil'


@pytest.mark.parametrize('ratio,price,cost', [
    ('1e-1000', 2, 2), ('1e-308', 2, 2), ('1e308', 2, 2), (.5, 2, float('inf')),
])
def test_unrepresentable_conversion_blocks_before_mutating_positions(ratio, price, cost):
    book, p = holding(); p.last_price, p.cost = price, cost
    before = deepcopy(book.positions)
    with pytest.raises(LedgerError, match = 'conversion'):
        book._convert(DAY, conversion(ratio, 'ceil', 'NEW'))
    assert book.positions == before and not book.issues and not book.assumptions


def test_other_code_conversion_merges_existing_pending_queues():
    book, old = holding(303); old.pending = 202
    for day in LATER: book.listing[day].append(('510310.SH', 101))
    new = book.pos('NEW'); new.qty, new.pending, new.cost = 20, 10, 3
    book.listing[LATER[1]].append(('NEW', 10))
    book.start_day(DAY, [conversion(.5, 'ceil', 'NEW')])
    assert (old.qty, old.pending, new.qty, new.pending, new.sellable) == (0, 0, 172, 111, 61)
    assert new.cost * new.qty == pytest.approx(666)
    book.start_day(LATER[0]); assert new.pending == 61
    book.start_day(LATER[1]); assert new.pending == 0 and new.sellable == 172


def test_same_code_split_transforms_t1_lock_without_cash_flow():
    book, p = holding(101); p.today_buy = 100
    book._convert(DAY, conversion(2))
    assert (p.qty, p.today_buy, p.sellable, p.cost, p.last_price) == (202, 200, 2, 1, 1)
    assert not book.cash_events
