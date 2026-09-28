from datetime import date

import pandas as pd
import pytest

from observe.dataset import cross_sectional_preprocess
from observe.evaluation.portfolio import metrics
from observe.labels import adj_open_to_open_h
from observe.ledger.book import Book
from observe.ledger.rules import Rule, RuleSet
from observe.portfolio import target_list

RULES = RuleSet([Rule(start = date(2020, 1, 1), commission_rate = 0.00025, min_commission = 5, transfer_fee = 0.00001, stamp_tax = 0.0005)])

def test_buy_t1_sell_and_identity():
    b = Book(10000); d = date(2024, 1, 2)
    b.start_day(d); b.execute({"instrument": "A", "side": "buy", "amount": 5000}, {"open": 50}, d, RULES)
    assert b.position("A").sellable == 0
    row = b.mark_to_market(d, {"A": {"close": 50}}); assert row["equity"] == pytest.approx(b.cash + 5000)
    b.start_day(date(2024, 1, 3)); assert b.position("A").sellable == 100
    b.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 50}, date(2024, 1, 3), RULES)
    assert b.position("A").qty == 0

def test_cash_policy_difference():
    b = Book(10000); d = date(2024, 1, 2); b.start_day(d); b.position("A").qty = 100; b.position("A").last_price = 100
    b.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 100}, d, RULES)
    result = b.execute({"instrument": "B", "side": "buy", "amount": 10000}, {"open": 100}, d, RULES, open_cash = 10000)
    assert result["qty_filled"] == 0

def test_corporate_actions_pending_and_pay():
    b = Book(0); b.position("A").qty = 100
    b.start_day(date(2024, 1, 2), [{"instrument": "A", "ex_date": date(2024, 1, 2), "pay_date": date(2024, 1, 5), "cash_per_share": 1, "bonus_ratio": 1}])
    assert b.position("A").qty == 200 and b.position("A").pending == 100 and b.position("A").sellable == 100
    b.start_day(date(2024, 1, 3), [{"instrument": "A", "bonus_list_date": date(2024, 1, 3)}]); assert b.position("A").sellable == 200
    b.start_day(date(2024, 1, 5)); assert b.cash == 100

def test_target_priorities():
    r = target_list({"A": 1, "B": 2, "C": 3}, {"A": 100, "Z": 100}, n = 2, buffer = 0, max_sell = 0, candidates = {"A", "B", "C"})
    assert "Z" in r["forced"] and "Z" in r["exits"] and "A" in r["target"]
    r = target_list({"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 100, "C": 100}, n = 2, buffer = 0, max_sell = None)
    assert len(r["exits"]) == 1

def test_label_and_preprocess_future_independent():
    f = pd.DataFrame({"instrument": ["A", "B", "A", "B"], "date": [1, 1, 2, 2], "x": [1, 3, 2, 4], "adj_open": [10, 20, 11, 22]})
    assert adj_open_to_open_h(f, 1).iloc[0] == 1
    a = cross_sectional_preprocess(f, ["x"]); f.loc[3, "x"] = 999999; c = cross_sectional_preprocess(f, ["x"])
    assert a.loc[0, "x"] == c.loc[0, "x"]

def test_metrics_and_short_sharpe_undefined():
    e = [100, 101, 100, 102, 101, 103, 104, 103, 105, 106, 107]
    m = metrics(e, [100 + i for i in range(len(e))]); assert m["annualized_return"] > 0; assert m["sharpe"] is None
