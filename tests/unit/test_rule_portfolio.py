from datetime import date

import pandas as pd
import pytest

from observe.ledger.rules import Rule, RuleSet
from observe.loop import run_loop
from observe.portfolio import target_weight_orders
from observe.schedule import rebalance_dates
from observe.factors.expr import compute

ZERO = RuleSet([Rule(start = date(2000, 1, 1), limit_pct = 1, commission_rate = 0, min_commission = 0, stamp_tax = 0, transfer_fee = 0)])


def test_week_and_month_use_actual_holiday_calendar_and_previous_close():
    cal = list(pd.to_datetime(['2024-03-28', '2024-03-29', '2024-04-01', '2024-04-02', '2024-04-03', '2024-04-08', '2024-04-09']).date)
    assert rebalance_dates(cal, cal, 'monthly') == {date(2024, 3, 29)}
    assert rebalance_dates(cal, cal, 'weekly') == {date(2024, 3, 29), date(2024, 4, 3)}
    assert rebalance_dates(cal, cal, 'weekly', session = 2) == {date(2024, 3, 28), date(2024, 4, 1), date(2024, 4, 8)}
    assert rebalance_dates(cal, cal, 'sessions', every = 5) == {cal[0], cal[5]}


def test_full_target_rebalance_trims_existing_names_and_reserves_cash():
    days = list(pd.bdate_range('2024-01-02', periods = 5).date)
    market = {d: {i: {'open': 10 if k <= 2 else px, 'close': 10 if k < 2 else px, 'preclose': 10 if k <= 2 else px, 'avg_amount_20d': 1e9}
                  for i, px in [('A', 12), ('B', 8)]} for k, d in enumerate(days)}
    target = {d: {'A': .4, 'B': .4} for d in days}
    book, orders = run_loop(days, market, {}, 100000, ZERO, construction = 'target_weights', targets_by_date = target, rebalance_every = 2)
    assert book.pos('A').qty == 3400 and book.pos('B').qty == 5000
    assert book.cash == 19200 and book.equity_curve()[-1] == 100000
    trim = [o for o in orders if o['reason'] == 'target_trim']
    assert len(trim) == 1 and trim[0]['qty_filled'] == 600 and trim[0]['exec_date'] == days[3]
    target[days[0]] = {}
    cash_book, _ = run_loop(days[:2], market, {}, 100000, ZERO, construction = 'target_weights', targets_by_date = target)
    assert cash_book.cash == 100000


def test_invalid_or_missing_target_is_not_a_cash_signal():
    with pytest.raises(ValueError, match = '目标权重'): target_weight_orders({'A': .7, 'B': .4}, {}, 100, {}, ZERO, date(2024, 1, 2))
    with pytest.raises(ValueError, match = '目标权重'): target_weight_orders({'A': float('nan')}, {}, 100, {}, ZERO, date(2024, 1, 2))
    with pytest.raises(ValueError, match = '缺少规则策略目标'):
        run_loop([date(2024, 1, 2)], {}, {}, 1000, ZERO, construction = 'target_weights', targets_by_date = {})


def test_valuation_reciprocal_keeps_negative_values_and_excludes_zero_nonfinite():
    values = pd.DataFrame([[2, -4, 0, float('inf'), float('nan')]])
    factor = compute('1/pb_mrq', {'pb_mrq': values})
    assert factor.iloc[0, :2].tolist() == [.5, -.25] and factor.iloc[0, 2:].isna().all()
