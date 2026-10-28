"""Hand-calculated unit NAV, cash reconciliation and rejected-flow atomicity."""
from copy import deepcopy
from datetime import date
import math

import pytest

from observe.ledger import Book, LedgerError, Position, Rule, RuleSet

D1, D2, D3 = (date(2024, 1, d) for d in (2, 3, 4))


def state(book):
    return deepcopy(book.__dict__)


def test_cash_deposits_are_not_returns_and_events_reconcile():
    b = Book(1000, external_flows=True)
    b.start_day(D1)
    for n in range(3):
        b.external_cash_flow(D1, 400, {}, event_id=f'callback-{n}')
    row = b.close_day(D1, {})
    assert (row['cash'], row['equity'], row['units'], row['unit_nav']) == (2200, 2200, 2200, 1)
    assert row['external_flow'] == row['external_flow_total'] == 1200
    assert row['daily_return'] == 0
    assert b.equity_curve() == [1, 1]
    assert sum(e['amount'] for e in b.cash_events) + b.initial == b.cash


def test_intraday_deposit_and_withdrawal_preserve_return():
    b = Book(1000, external_flows=True)
    b.positions['A'] = Position(qty=100, last_price=10)
    b.cash = 0
    b.cash_events.append({'date': D1, 'kind': 'buy', 'amount': -1000})
    b.start_day(D1)
    event = b.external_cash_flow(D1, 600, {'A': {'open': 12}}, event_id='deposit')
    assert (event['unit_nav'], event['units_after'], event['equity_before'], event['equity_after']) == (1.2, 1500, 1200, 1800)
    b.external_cash_flow(D1, -300, {'A': {'open': 12}}, event_id='withdraw')
    row = b.close_day(D1, {'A': {'close': 15}})
    # 1800 equity / 1250 units = 1.44; price gain before deposit participates correctly.
    assert row['units'] == 1250
    assert row['unit_nav'] == pytest.approx(1.44)
    assert row['daily_return'] == pytest.approx(.44)
    assert row['external_flow'] == 300
    b.start_day(D2)
    row = b.close_day(D2, {'A': {'close': 12}})
    assert row['unit_nav'] == pytest.approx(1.2)
    assert row['daily_return'] == pytest.approx(1.2 / 1.44 - 1)


def test_real_book_fills_fees_are_returns_not_external_flows():
    b = Book(2000, external_flows=True)
    rules = RuleSet([Rule(start=D1)])
    b.start_day(D1)
    b.external_cash_flow(D1, 400, {}, event_id='cash')
    fill = b.execute({'instrument': 'A', 'side': 'buy', 'amount': 1000}, {'open': 10}, D1, rules)
    row = b.close_day(D1, {'A': {'close': 10}})
    assert row['unit_nav'] == pytest.approx((2400 - fill['fee']) / 2400)
    assert row['daily_return'] == pytest.approx(-fill['fee'] / 2400)
    assert [e['kind'] for e in b.cash_events] == ['external_cash_flow', 'buy', 'fee']


def test_record_dividend_receivable_and_pending_shares_are_in_flow_valuation():
    b = Book(1000, external_flows=True)
    b.cash = 0
    b.cash_events.append({'date': D1, 'kind': 'buy', 'amount': -1000})
    b.positions['A'] = Position(qty=100, pending=20, last_price=10)
    b.receivable[D3] = 100
    b.start_day(D1)
    event = b.external_cash_flow(D1, 550, {'A': {'suspended': True, 'open': 999}}, event_id='funding')
    assert (event['market_value'], event['receivable'], event['equity_before'], event['unit_nav']) == (1000, 100, 1100, 1.1)
    assert event['units_after'] == 1500
    assert b.positions['A'].last_price == 10
    row = b.close_day(D1, {'A': {'suspended': True, 'close': 999}})
    assert row['unit_nav'] == pytest.approx(1.1)


def test_full_redemption_then_new_deposit_retains_historical_nav():
    b = Book(1000, external_flows=True)
    b.start_day(D1)
    b._cash(D1, 'realized_gain', 200)
    event = b.external_cash_flow(D1, -1200, {}, event_id='all')
    assert event['units_after'] == 0
    row = b.close_day(D1, {})
    assert row['unit_nav'] == pytest.approx(1.2)
    assert row['daily_return'] == pytest.approx(.2)
    b.start_day(D2)
    b.external_cash_flow(D2, 600, {}, event_id='restart')
    row = b.close_day(D2, {})
    assert row['units'] == pytest.approx(500)
    assert row['unit_nav'] == pytest.approx(1.2)
    assert row['daily_return'] == 0


def test_zero_initial_cash_and_bankruptcy_are_distinct():
    b = Book(0, external_flows=True)
    b.start_day(D1)
    assert b.close_day(D1, {})['unit_nav'] == 1
    b.start_day(D2)
    b.external_cash_flow(D2, 400, {}, event_id='seed')
    b._cash(D2, 'loss', -400)
    assert b.close_day(D2, {})['unit_nav'] == 0
    b.start_day(D3)
    before = state(b)
    with pytest.raises(LedgerError, match='positive unit NAV'):
        b.external_cash_flow(D3, 400, {}, event_id='recapitalize')
    assert state(b) == before


@pytest.mark.parametrize('amount', [True, False, float('nan'), float('inf'), -float('inf'), 0, .001, 10**1000, 1e13, '400', None, -1001])
def test_invalid_amounts_leave_all_state_unchanged(amount):
    b = Book(1000, external_flows=True)
    b.start_day(D1)
    before = state(b)
    with pytest.raises(LedgerError):
        b.external_cash_flow(D1, amount, {}, event_id='rejected')
    assert state(b) == before


@pytest.mark.parametrize('quote', [{}, {'open': None}, {'open': 0}, {'open': -1}, {'open': True}, {'open': math.nan}, {'open': math.inf}, {'suspended': True}])
def test_missing_or_invalid_quotes_are_atomic(quote):
    b = Book(1000, external_flows=True)
    b.positions['A'] = Position(qty=100)
    b.start_day(D1)
    before = state(b)
    with pytest.raises(LedgerError):
        b.external_cash_flow(D1, 400, {'A': quote}, event_id='missing')
    assert state(b) == before


def test_lifecycle_duplicate_ids_and_close_valuation():
    b = Book(1000, external_flows=True)
    with pytest.raises(LedgerError):
        b.external_cash_flow(D1, 400, {}, event_id='early')
    b.start_day(D1)
    b.external_cash_flow(D1, 400, {}, event_id='same')
    for kwargs in ({'event_id': 'same'}, {'event_id': ''}, {'event_id': True}, {'event_id': 'phase', 'price_field': 'preclose'}):
        before = state(b)
        with pytest.raises(LedgerError):
            b.external_cash_flow(D1, 400, {}, **kwargs)
        assert state(b) == before
    with pytest.raises(LedgerError):
        b.start_day(D2)
    with pytest.raises(LedgerError):
        b.close_day(D2, {})
    b.close_day(D1, {})
    with pytest.raises(LedgerError):
        b.external_cash_flow(D1, 400, {}, event_id='late')
    with pytest.raises(LedgerError):
        b.start_day(D1)
    b.start_day(D2)
    with pytest.raises(LedgerError):
        b.external_cash_flow(D2, 400, {}, event_id='same')


def test_close_price_flow_cent_rounding_and_event_copy():
    b = Book(1000, external_flows=True)
    b.start_day(D1)
    event = b.external_cash_flow(D1, 400.126, {}, event_id='cent', price_field='close')
    assert event['amount'] == 400.13
    event['amount'] = 999
    assert b.cash_events[0]['amount'] == 400.13
    assert b.external_flow_events[0]['amount'] == 400.13
    assert b.close_day(D1, {})['unit_nav'] == 1


def test_default_mode_keeps_legacy_schema_and_equity_curve():
    b = Book(1000)
    before = state(b)
    with pytest.raises(LedgerError, match='not enabled'):
        b.external_cash_flow(D1, 400, {}, event_id='off')
    assert state(b) == before
    b.start_day(D1)
    row = b.close_day(D1, {})
    assert set(row) == {'date', 'cash', 'market_value', 'receivable', 'equity', 'daily_return', 'stale_price'}
    assert b.equity_curve() == [1000, 1000]


def test_cash_dividend_is_income_even_with_external_deposit():
    b = Book(2000, external_flows=True)
    b.cash = 1000
    b.cash_events.append({'date': D1, 'kind': 'buy', 'amount': -1000})
    b.positions['A'] = Position(qty=100, last_price=10)
    action = {'instrument': 'A', 'record_date': D1, 'ex_date': D2,
        'pay_date': D3, 'cash_per_share': 1., 'cash_entitlement': 'record_close'}
    b.start_day(D1, [action])
    b.close_day(D1, {'A': {'close': 10}})
    b.start_day(D2, [action])
    event = b.external_cash_flow(D2, 400, {'A': {'open': 9}}, event_id='ex-date')
    assert (event['receivable'], event['equity_before'], event['unit_nav']) == (100, 2000, 1)
    assert b.close_day(D2, {'A': {'close': 9}})['daily_return'] == 0
    b.start_day(D3)
    row = b.close_day(D3, {'A': {'close': 9}})
    assert (row['receivable'], row['unit_nav'], row['external_flow']) == (0, 1, 0)


def test_close_failure_does_not_consume_day_or_dividend_rights():
    b = Book(1000, external_flows=True)
    b.positions['A'] = Position(qty=100, last_price=10)
    b.start_day(D1)
    before = state(b)
    with pytest.raises(LedgerError):
        b.close_day(D1, {'A': {'close': -1}})
    assert state(b) == before
    assert b.close_day(D1, {'A': {'close': 10}})['unit_nav'] == 2


def test_aggregate_limit_and_delisted_flow_rejected_atomically():
    b = Book(1e12, external_flows=True)
    b.start_day(D1)
    before = state(b)
    with pytest.raises(LedgerError):
        b.external_cash_flow(D1, .01, {}, event_id='limit')
    assert state(b) == before
    b.positions['A'] = Position(qty=100, last_price=10)
    before = state(b)
    with pytest.raises(LedgerError, match='delisted'):
        b.external_cash_flow(D1, -100, {'A': {'open': 10, 'delisted': True}}, event_id='delisted')
    assert state(b) == before


def test_many_small_deposits_and_exact_full_redemption():
    b = Book(1000, external_flows=True)
    b.start_day(D1)
    b._cash(D1, 'income', 37.19)
    for n in range(1000):
        b.external_cash_flow(D1, .01, {}, event_id=str(n))
    event = b.external_cash_flow(D1, -b.cash, {}, event_id='all')
    assert (b.cash, event['units_after']) == (0, 0)
    assert b.close_day(D1, {})['unit_nav'] == pytest.approx(1.03719)
