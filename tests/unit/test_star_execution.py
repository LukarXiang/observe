"""科创板数量边界；对照手算并保护既有主板规则指纹。"""
from datetime import date

import pytest

from observe.ledger import Book, Rule, RuleSet
from observe.portfolio import target_weight_orders

DAYS = [date(2024, 4, 1), date(2024, 4, 2)]
QUOTE = {'open': 10, 'close': 10, 'preclose': 10, 'board': 'star', 'avg_amount_20d': 1e9}
ZERO = RuleSet([Rule(start = date(2019, 7, 22), buy_unit = 1, min_buy_qty = 200, min_sell_qty = 200, max_order_qty = 50000,
                     limit_pct = .2, commission_rate = 0, min_commission = 0, stamp_tax = 0, transfer_fee = 0)])


def test_old_main_board_fingerprint_and_separate_star_profile():
    old = RuleSet.from_yaml('configs/rule_profiles/main_board.yaml')
    assert old.config_fingerprint() == 'cc21a3bd6d6db915d30b3e0e2326653792dd878a8630a39e64e6bd02ab2dd7df'
    with pytest.raises(ValueError, match = 'no trading rule'): old.on(DAYS[0], 'star')
    profile = RuleSet.from_yaml('configs/rule_profiles/csi800_daily_v1.yaml')
    for st in (False, True):
        r = profile.on(DAYS[0], 'star', st)
        assert (r.buy_unit, r.min_buy_qty, r.min_sell_qty, r.max_order_qty) == (1, 200, 200, 50000)
        assert profile.limit_prices(10, DAYS[0], 'star', st) == (8, 12)
        assert profile.scaled(commission_rate = 2).on(DAYS[0], 'star', st).min_buy_qty == 200


@pytest.mark.parametrize('amount,expected', [(1990, 0), (2000, 200), (2010, 201), (500010, 50000)])
def test_star_buy_minimum_one_share_increment_and_maximum(amount, expected):
    b = Book(1_000_000, DAYS); b.start_day(DAYS[0])
    result = b.execute({'instrument': '688001.SH', 'side': 'buy', 'amount': amount}, QUOTE, DAYS[0], ZERO)
    assert result['qty_filled'] == expected and b.cash == 1_000_000 - expected * 10
    if expected:
        assert b.execute({'instrument': '688001.SH', 'side': 'sell', 'qty': 'all'}, QUOTE, DAYS[0], ZERO)['reject_reason'] == 'not_sellable'


def test_star_cash_shrink_keeps_minimum_and_does_not_overspend():
    profile = RuleSet.from_yaml('configs/rule_profiles/csi800_daily_v1.yaml')
    for cash, expected in [(2015.02, 201), (2005.02, 200), (2004.99, 0)]:
        b = Book(cash, DAYS); b.start_day(DAYS[0])
        result = b.execute({'instrument': '688001.SH', 'side': 'buy', 'amount': 3000}, QUOTE, DAYS[0], profile)
        assert result['qty_filled'] == expected and b.cash >= 0
        if expected: assert result['fee'] == 5.02


def test_star_sell_minimum_odd_balance_and_participation():
    b = Book(0, DAYS); b.pos('688001.SH').qty = 401; b.start_day(DAYS[0])
    order = {'instrument': '688001.SH', 'side': 'sell', 'qty': 199}
    assert b.execute(order, QUOTE, DAYS[0], ZERO)['reject_reason'] == 'sell_unit'
    assert b.execute({**order, 'qty': 201}, QUOTE, DAYS[0], ZERO)['qty_filled'] == 201
    assert b.execute({**order, 'qty': 'all', 'participation': .5}, {**QUOTE, 'avg_amount_20d': 1000}, DAYS[0], ZERO)['reject_reason'] == 'participation_limit'
    assert b.execute({**order, 'qty': 200}, QUOTE, DAYS[0], ZERO)['qty_filled'] == 200
    b.pos('688001.SH').qty = 199
    assert b.execute({**order, 'qty': 198}, QUOTE, DAYS[0], ZERO)['reject_reason'] == 'sell_unit'
    assert b.execute({**order, 'qty': 'all'}, QUOTE, DAYS[0], ZERO)['qty_filled'] == 199
    assert b.cash == 6000


def test_star_target_rebalance_does_not_send_subminimum_orders():
    b = Book(0, DAYS); p = b.pos('688001.SH'); p.qty, p.last_price = 401, 10
    assert target_weight_orders({'688001.SH': 400 / 401}, b.positions, 4010, {'688001.SH': QUOTE}, ZERO, DAYS[0]) == []
    orders = target_weight_orders({'688001.SH': 201 / 401}, b.positions, 4010, {'688001.SH': QUOTE}, ZERO, DAYS[0])
    assert len(orders) == 1 and orders[0]['qty'] == 200 and orders[0]['side'] == 'sell'
    assert target_weight_orders({'688001.SH': 1}, {}, 1990, {'688001.SH': QUOTE}, ZERO, DAYS[0]) == []
    assert target_weight_orders({'688001.SH': 1}, {}, 2010, {'688001.SH': QUOTE}, ZERO, DAYS[0])[0]['amount'] == 2010
