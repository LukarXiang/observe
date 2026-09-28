from datetime import date, timedelta

import pandas as pd
import pytest

from observe.dataset import cross_sectional_preprocess
from observe.evaluation.portfolio import metrics
from observe.labels import adj_open_to_open_h
from observe.ledger.book import Book
from observe.ledger.rules import Rule, RuleSet
from observe.loop import rerun_scenario
from observe.portfolio import orders_for_targets, target_list

RULES = RuleSet([Rule(start = date(2020, 1, 1), commission_rate = .00025, min_commission = 5, stamp_tax = .0005, transfer_fee = .00001)])


def assert_identity(book):
    for row in book.equity_rows:
        assert row["equity"] == pytest.approx(row["cash"] + row["market_value"] + row["receivable"], abs = .01)


def test_sell_then_buy_and_preopen_cash_trajectories():
    sell_then_buy = Book(1000); sell_then_buy.position("A").qty = 100; sell_then_buy.start_day(date(2024, 1, 2))
    sell_then_buy.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 10}, date(2024, 1, 2), RULES)
    bought = sell_then_buy.execute({"instrument": "B", "side": "buy", "amount": 1000}, {"open": 10}, date(2024, 1, 2), RULES)
    assert bought["qty_filled"] == 100
    preopen = Book(1000); preopen.position("A").qty = 100; preopen.start_day(date(2024, 1, 2))
    preopen.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 10}, date(2024, 1, 2), RULES, open_cash = 1000)
    blocked = preopen.execute({"instrument": "B", "side": "buy", "amount": 1000}, {"open": 10}, date(2024, 1, 2), RULES, open_cash = 1000)
    assert blocked["qty_filled"] == 0


def test_failed_sell_stays_in_assets_until_next_rebalance():
    b = Book(0); b.position("A").qty = 100; b.start_day(date(2024, 1, 2))
    result = b.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 9, "preclose": 10}, date(2024, 1, 2), RULES)
    assert result["reject_reason"] == "limit_down" and b.position("A").qty == 100
    row = b.mark_to_market(date(2024, 1, 2), {"A": {"close": 9}}); assert row["market_value"] == 900
    assert_identity(b)


def test_split_on_execution_day_uses_post_split_price_and_pending_shares():
    b = Book(0); b.position("A").qty = 100; b.position("A").last_price = 10
    d = date(2024, 1, 2); b.start_day(d, [{"instrument": "A", "ex_date": d, "bonus_ratio": 1, "bonus_list_date": d + timedelta(days = 1)}])
    assert b.position("A").qty == 200 and b.position("A").pending == 100
    assert b.position("A").sellable == 100


def test_missing_dates_follow_conservative_assumptions():
    d = date(2013, 1, 2); calendar = [d + timedelta(days = i) for i in range(1, 15)]
    b = Book(0); b.position("A").qty = 10; b.start_day(d, [{"instrument": "A", "ex_date": d, "cash_per_share": 1, "trading_dates": calendar}])
    assert b.assumptions[0]["value"] == calendar[9]
    b.start_day(calendar[0]); assert b.position("A").sellable == 10


def test_label_and_ledger_rights_difference_is_explainable():
    frame = pd.DataFrame({"adj_open": [10., 12.], "raw_open": [10., 9.]})
    assert adj_open_to_open_h(frame).iloc[0] == pytest.approx(.2)
    assert frame.raw_open.iloc[1] / frame.raw_open.iloc[0] - 1 < 0


def test_future_label_validity_does_not_change_preprocessing():
    frame = pd.DataFrame({"date": [1, 1, 2, 2], "instrument": ["A", "B", "A", "B"], "x": [1., 3., 2., 4.]})
    first = cross_sectional_preprocess(frame, ["x"])
    changed = frame.copy(); changed["matured_at"] = [None, None, None, date(2099, 1, 1)]
    second = cross_sectional_preprocess(changed, ["x"])
    assert first["x"].tolist() == second["x"].tolist()


def test_cost_change_recomputes_second_order():
    d1, d2, d3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    config = {"dates": [d1, d2, d3], "market": {d1: {"A": {"open": 10, "close": 10}}, d2: {"A": {"open": 10, "close": 10}}, d3: {"A": {"open": 10, "close": 10}}}, "scores_by_date": {d1: {"A": 1}, d2: {"A": 1}}, "initial_cash": 2000, "rules": RULES, "n": 1, "buffer": 0, "max_sell": None, "rebalance_every": 1}
    base, base_orders = rerun_scenario(config, slippage = 0); costly, costly_orders = rerun_scenario(config, slippage = .1)
    assert base_orders != costly_orders or base.equity_rows != costly.equity_rows


def test_max_sell_null_zero_and_k_with_forced_exit():
    scores = {"A": 1, "B": 2, "C": 3, "D": .5}; held = {"A": 1, "B": 1, "D": 1, "X": 1}
    for setting, expected in ((None, 3), (0, 1), (1, 2)):
        result = target_list(scores, held, n = 2, buffer = 0, max_sell = setting, candidates = set(scores))
        assert len(result["exits"]) == expected and "X" in result["forced"]


def test_metrics_hand_calculation_and_missing_benchmark_days():
    equity = [100, 101, 100, 102, 103, 102, 104, 105, 104, 106, 107]
    result = metrics(equity, [100, 100, 101, 102, 102, 103, 104, 104, 105, 106, 107])
    assert result["annualized_return"] == pytest.approx((107 / 100) ** (242 / 10) - 1)
    assert result["max_drawdown"] == pytest.approx(-1 / 101)
    assert result["information_ratio"] is not None


def test_every_daily_mark_has_identity_after_dividend():
    b = Book(1000); b.position("A").qty = 100; b.position("A").last_price = 10
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    b.start_day(d1, [{"instrument": "A", "ex_date": d1, "cash_per_share": 1, "pay_date": d2}]); b.mark_to_market(d1, {"A": {"close": 9}})
    b.start_day(d2); b.mark_to_market(d2, {"A": {"close": 9}}); assert_identity(b)


def test_order_intents_are_amount_or_proportion_semantics():
    result = target_list({"A": 2}, {}, n = 1, buffer = 0, max_sell = None)
    orders = orders_for_targets(result, {}, {"A": 100}, True)
    assert orders[0]["side"] == "buy" and "amount" in orders[0] and "qty" not in orders[0]


def test_synthetic_rule_profile_is_loaded_by_date():
    rules = RuleSet.from_yaml("tests/fixtures/synthetic_rules.yaml")
    assert rules.for_date(date(2024, 1, 2)).buy_unit == 100


def test_missing_benchmark_is_reported_without_deleting_strategy_days():
    result = metrics([100, 101, 102, 101], [100, float("nan"), 101, 102])
    assert result["benchmark_missing_days"] == 2 and result["annualized_return"] is not None


def test_manual_cashflow_fixture_matches_book_step_by_step():
    table = pd.read_csv("tests/fixtures/synthetic_cashflows.csv")
    b = Book(10000); d1, d2 = date(2024, 1, 2), date(2024, 1, 3); b.start_day(d1)
    assert table.loc[0, "expected_cash"] == b.cash
    b.execute({"instrument": "A", "side": "buy", "amount": 5000}, {"open": 50}, d1, RULES)
    assert b.cash == pytest.approx(table.loc[1, "expected_cash"])
    b.start_day(d2); b.execute({"instrument": "A", "side": "sell", "qty": "all"}, {"open": 50}, d2, RULES)
    assert b.cash == pytest.approx(table.loc[2, "expected_cash"])
