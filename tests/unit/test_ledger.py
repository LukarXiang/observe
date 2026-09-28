"""账本用例（模块 16 测试 1–14、16、17）。预期值均为手算，写在注释里。"""
from datetime import date

import pandas as pd
import pytest

from observe.ledger import Book, LedgerError, Rule, RuleSet

D = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17)]
FEE = RuleSet([Rule(start = date(2020, 1, 1))])                                   # 佣金万 2.5 最低 5、印花税 0.05%、过户费 0.001%
ZERO = RuleSet([Rule(start = date(2020, 1, 1), stamp_tax = 0, transfer_fee = 0, commission_rate = 0, min_commission = 0)])
Q = {'open': 10, 'close': 10, 'preclose': 10}


def buy(b, i, amount, day, rules = ZERO, **q): return b.execute({'instrument': i, 'side': 'buy', 'amount': amount}, {**Q, **q}, day, rules)
def sell(b, i, day, rules = ZERO, **q): return b.execute({'instrument': i, 'side': 'sell', 'qty': 'all'}, {**Q, **q}, day, rules)


def test_t1_same_day_sell_rejected_next_day_ok():
    b = Book(1000, D); b.start_day(D[0]); buy(b, 'A', 1000, D[0])
    assert sell(b, 'A', D[0])['reject_reason'] == 'not_sellable'
    b.close_day(D[0], {'A': Q}); b.start_day(D[1]); assert sell(b, 'A', D[1])['qty_filled'] == 100


def test_suspension_and_limit_rejections():
    b = Book(10000, D); b.start_day(D[0])
    assert buy(b, 'A', 1000, D[0], suspended = True)['reject_reason'] == 'suspended'
    assert buy(b, 'A', 1000, D[0], open = None)['reject_reason'] == 'no_open_price'
    assert buy(b, 'A', 1000, D[0], open = 11.0)['reject_reason'] == 'limit_up'          # 涨停价 = 10 × 1.1 = 11.00
    b.pos('B').qty = 100
    assert sell(b, 'B', D[0], open = 9.0)['reject_reason'] == 'limit_down'              # 跌停价 9.00


def test_gap_up_shrinks_or_rejects_and_cash_never_negative():
    b = Book(1100, D); b.start_day(D[0])
    r = buy(b, 'A', 2000, D[0], FEE, open = 10.6, preclose = 10)   # 2000/10.6 = 188 → 100 股；1060 + 佣金 5 + 过户 0.01 = 1065.01 ≤ 1100
    assert r['qty_filled'] == 100 and b.cash == pytest.approx(1100 - 1065.01)
    assert buy(b, 'B', 2000, D[0], FEE, open = 10.6, preclose = 10)['reject_reason'] == 'cash' and b.cash >= 0


def test_cash_dividend_receivable_then_paid():
    b = Book(0, D); p = b.pos('A'); p.qty, p.last_price = 100, 10.0
    b.start_day(D[0], [{'instrument': 'A', 'ex_date': D[0], 'cash_per_share': 1.0, 'pay_date': D[2]}])
    r = b.close_day(D[0], {'A': {**Q, 'close': 9}}); assert (r['market_value'], r['receivable'], r['equity']) == (900, 100, 1000)
    b.start_day(D[1]); b.close_day(D[1], {'A': {**Q, 'close': 9}})
    b.start_day(D[2]); r = b.close_day(D[2], {'A': {**Q, 'close': 9}})
    assert (r['cash'], r['receivable'], r['equity'], r['daily_return']) == (100, 0, 1000, 0)   # 到账只是应收转现金，不重复计收益


def test_bonus_shares_pending_until_listing_and_equity_unchanged():
    b = Book(0, D); p = b.pos('A'); p.qty, p.last_price = 100, 10.0
    b.start_day(D[0], [{'instrument': 'A', 'ex_date': D[0], 'bonus_ratio': 1.0, 'bonus_list_date': D[1]}])
    assert (p.qty, p.pending, p.sellable) == (200, 100, 100)
    assert b.close_day(D[0], {'A': {**Q, 'close': 5}})['equity'] == 1000                 # 100×10 → 200×5
    assert sell(b, 'A', D[0], open = 5, preclose = 5)['qty_filled'] == 100 and p.qty == 100 and p.sellable == 0
    b.close_day(D[0], {'A': {**Q, 'close': 5}}); b.start_day(D[1]); assert p.sellable == 100


def test_min_commission_and_roundtrip_cost():
    assert FEE.fees(10000, 'buy', D[0])['commission'] == 5                              # 10000 × 万 2.5 = 2.5 → 最低 5
    b = Book(2000, D); b.start_day(D[0])
    b.execute({'instrument': 'A', 'side': 'buy', 'amount': 1100}, Q, D[0], FEE, slippage = 0.01)   # 成交 10.10×100=1010，费 5+0.01
    b.close_day(D[0], {'A': Q}); b.start_day(D[1])
    b.execute({'instrument': 'A', 'side': 'sell', 'qty': 'all'}, Q, D[1], FEE, slippage = 0.01)    # 成交 9.90×100=990，费 5+0.50+0.01
    assert b.cash == pytest.approx(2000 - 1015.01 + 984.49)                                    # 损失 30.52 = 滑点 20 + 费用 10.52


def test_stamp_tax_switch_on_2023_08_28():
    rules = RuleSet([Rule(start = date(2020, 1, 1), stamp_tax = 0.001), Rule(start = date(2023, 8, 28), stamp_tax = 0.0005)])
    assert rules.fees(10000, 'sell', date(2023, 8, 25))['stamp_tax'] == 10 and rules.fees(10000, 'sell', date(2023, 8, 28))['stamp_tax'] == 5
    with pytest.raises(ValueError): rules.on(date(2019, 12, 31))                             # 查不到规则就报错


def test_delisted_holding_blocks_and_stays_in_book():
    b = Book(0, D); p = b.pos('A'); p.qty, p.last_price = 100, 3.0
    b.start_day(D[0]); b.close_day(D[0], {'A': {'close': 3.0, 'delisted': True}})
    assert b.status == 'blocked' and p.qty == 100 and b.equity_rows[-1]['market_value'] == 300


def test_suspended_ex_date_adjusts_valuation_and_ignores_placeholder_close():
    b = Book(0, D); p = b.pos('A'); p.qty, p.last_price = 100, 10.0
    act = {'instrument': 'A', 'ex_date': D[0], 'cash_per_share': 1.0, 'bonus_ratio': 0.5, 'pay_date': D[0], 'bonus_list_date': D[1]}
    b.start_day(D[0], [act], {'A': {'suspended': True}})
    r = b.close_day(D[0], {'A': {'suspended': True, 'close': 10.0}})                     # 停牌行的占位收盘价 10 不能用
    assert p.qty == 150 and p.last_price == pytest.approx(6.0)                                 # (10 − 1) / 1.5 = 6
    assert (r['cash'], r['market_value'], r['equity'], r['stale_price']) == (100, 900, 1000, True)


def test_missing_dates_are_inferred_conservatively_and_recorded():
    old = [date(2013, 6, 3 + k) for k in range(5)] + [date(2013, 6, 10 + k) for k in range(5)] + [date(2013, 6, 17 + k) for k in range(3)]
    b = Book(0, old); b.pos('A').qty = 10
    b.start_day(old[0], [{'instrument': 'A', 'ex_date': old[0], 'cash_per_share': 1.0, 'bonus_ratio': 1.0}])
    pay, listed = [x['value'] for x in b.assumptions]
    assert pay == old[10] and listed == old[1]                  # 2014 年前：到账按除息后第 10 个交易日；上市按第 1 个交易日
    b2 = Book(0, D); b2.pos('A').qty = 10; b2.start_day(D[0], [{'instrument': 'A', 'ex_date': D[0], 'cash_per_share': 1.0}])
    assert b2.assumptions[0]['value'] == D[0] and b2.cash == 10


def test_cash_reconciliation_detects_untracked_change():
    b = Book(1000, D); b.start_day(D[0]); buy(b, 'A', 500, D[0]); b.cash += 1       # 绕过流水改现金
    with pytest.raises(LedgerError): b.close_day(D[0], {'A': Q})


def test_manual_cashflow_table_matches_step_by_step():
    t = pd.read_csv('tests/fixtures/synthetic_cashflows.csv'); b = Book(10000, D); b.start_day(D[0])
    assert b.cash == t.expected_cash[0]
    buy(b, 'A', 5000, D[0], FEE, open = 50, preclose = 50); assert b.cash == pytest.approx(t.expected_cash[1])
    b.close_day(D[0], {'A': {'close': 50}}); b.start_day(D[1])
    sell(b, 'A', D[1], FEE, open = 50, preclose = 50); assert b.cash == pytest.approx(t.expected_cash[2])
    assert sum(e['amount'] for e in b.cash_events) == pytest.approx(t.cash_delta[1:].sum())
