"""标签、横截面预处理与组合指标（模块 13、17）。"""
import statistics as st
from datetime import date

import numpy as np
import pandas as pd
import pytest

from observe.dataset import cross_sectional_preprocess, training_rows
from observe.evaluation import metrics
from observe.labels import build_labels
from observe.ledger import Book, Rule, RuleSet

CAL = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 15), date(2024, 1, 16)]   # 1/8 与 1/15 之间是长假


def bars(prices, trading = None):
    rows = [{'date': d, 'instrument': i, 'adj_open': p[k], 'is_trading': True if trading is None else trading.get(i, [True] * len(CAL))[k]}
            for i, p in prices.items() for k, d in enumerate(CAL)]
    return pd.DataFrame(rows)


def test_label_endpoints_follow_calendar_not_rows():
    lab = build_labels(bars({'X': [10, 11, 12, 13, 14, 15, 16]}), CAL, h = 2).set_index('decision_date')
    r = lab.loc[CAL[3]]                                                   # 决策 1/5 → 1/8 开盘买、跨长假到 1/16 开盘卖
    assert (r.entry_date, r.exit_date) == (CAL[4], CAL[6]) and r.value == pytest.approx(16 / 14 - 1) and r.valid
    assert lab.loc[CAL[0], 'value'] == pytest.approx(13 / 11 - 1)         # k+1 与 k+1+h，不是 k 与 k+h
    assert lab.loc[CAL[5], 'invalid_reason'] == 'not_matured'


def test_label_invalid_when_exit_suspended_not_shifted_and_per_instrument():
    b = bars({'X': [10, 11, 12, 13, 14, 15, 16], 'Y': [20, 20, 20, 99, 20, 20, 20]}, {'Y': [True, True, True, False, True, True, True]})
    lab = build_labels(b, CAL, h = 2).set_index(['decision_date', 'instrument'])
    assert lab.loc[(CAL[0], 'Y'), 'invalid_reason'] == 'exit_suspended' and np.isnan(lab.loc[(CAL[0], 'Y'), 'value'])
    assert lab.loc[(CAL[0], 'X'), 'value'] == pytest.approx(13 / 11 - 1)   # 不同证券互不串行


def test_rights_issue_label_vs_ledger_without_subscription():
    """10 配 3、配股价 5：除权参考价 = (10 + 0.3×5)/1.3 = 8.846；除权日开盘 8.85。后复权因子把除权后价格乘 10/8.846。
    复权标签 ≈ +0.04%；不参与配股的账本 = 8.85/10 − 1 = −11.5%；差 = 0.885 × (10/8.846 − 1)"""
    ref = (10 + 0.3 * 5) / 1.3; f = 10 / ref; raw = [10, 10, 8.85, 8.85, 8.85, 8.85, 8.85]
    adj = [p * (f if k >= 2 else 1) for k, p in enumerate(raw)]
    label = build_labels(bars({'A': adj}), CAL, h = 1).set_index('decision_date').loc[CAL[0], 'value']
    rules = RuleSet([Rule(start = date(2020, 1, 1), stamp_tax = 0, transfer_fee = 0, commission_rate = 0, min_commission = 0)])
    b = Book(1000, CAL); b.start_day(CAL[1]); b.execute({'instrument': 'A', 'side': 'buy', 'amount': 1000}, {'open': 10, 'preclose': 10}, CAL[1], rules)
    b.close_day(CAL[1], {'A': {'close': 10}}); b.start_day(CAL[2]); held = b.close_day(CAL[2], {'A': {'close': 8.85}})['equity'] / 1000 - 1
    assert held == pytest.approx(-0.115) and label - held == pytest.approx(0.885 * (f - 1))


def frame():
    return pd.DataFrame({'date': [1, 1, 1, 1, 2, 2, 2, 2], 'instrument': list('ABCD') * 2, 'eligible': [True, True, True, False] * 2,
                         'x': [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0]})


def test_preprocess_uses_all_eligible_rows_and_ignores_future():
    base = cross_sectional_preprocess(frame(), ['x']); day1 = base[base.date == 1].x.tolist()
    assert day1[:3] == pytest.approx([-1, 0, 1]) and np.isnan(day1[3])     # D 不在候选：不参与也不输出
    changed = frame(); changed.loc[changed.date == 2, 'x'] = [0, 0, 999, 0]; changed.loc[changed.date == 2, 'eligible'] = [False, True, True, True]
    assert cross_sectional_preprocess(changed, ['x'])[lambda t: t.date == 1].x.tolist()[:3] == day1[:3]   # 改未来不改过去


def test_training_rows_filter_after_preprocess_by_maturity():
    pre = cross_sectional_preprocess(frame(), ['x'])
    labels = pd.DataFrame({'decision_date': [1, 1, 1], 'instrument': ['A', 'B', 'C'], 'value': [0.1, np.nan, 0.3], 'valid': [True, False, True],
                           'matured_at': [5, 5, 9]})
    rows = training_rows(pre, labels, asof = 6)
    assert rows.instrument.tolist() == ['A'] and rows.x.iloc[0] == pytest.approx(-1)   # 特征仍是三只候选一起标准化的结果


# 指标 --------------------------------------------------------------------------
R = [0.01, -0.005, 0.02, 0.0, -0.01, 0.015, 0.005, -0.002, 0.012, -0.008, 0.004, 0.009, -0.003, 0.007, 0.011, -0.006, 0.002, 0.013, -0.004, 0.006, 0.008]


def test_metrics_match_independent_hand_formulas():
    e = [100.0]; [e.append(e[-1] * (1 + r)) for r in R]; rf = 0.02; a = 242; rf_d = (1 + rf) ** (1 / a) - 1
    b = [100.0]; [b.append(b[-1] * (1 + r / 2)) for r in R]
    m = metrics(e, b, rf = rf)
    x = [r - rf_d for r in R]; act = [r - r / 2 for r in R]
    assert m['sharpe'] == pytest.approx(st.mean(x) / st.stdev(x) * a ** 0.5)
    assert m['information_ratio'] == pytest.approx(st.mean(act) / st.stdev(act) * a ** 0.5)
    assert m['annual_return'] == pytest.approx((e[-1] / 100) ** (a / 21) - 1)
    assert m['annual_return_diff'] == pytest.approx(m['annual_return'] - ((b[-1] / 100) ** (a / 21) - 1))
    peak, dd = e[0], 0.0
    for v in e: peak = max(peak, v); dd = min(dd, v / peak - 1)
    assert m['max_drawdown'] == pytest.approx(dd)


def test_metrics_short_series_undefined_and_missing_benchmark_kept():
    m = metrics([100, 101, 100, 102], [100, np.nan, 101, 102])
    assert m['sharpe'] is None and m['information_ratio'] is None and m['benchmark_missing_days'] == 2 and m['total_return'] == pytest.approx(0.02)


def test_first_day_fee_counts_in_first_return():
    rules = RuleSet([Rule(start = date(2020, 1, 1))]); b = Book(10000, CAL); b.start_day(CAL[0])
    b.execute({'instrument': 'A', 'side': 'buy', 'amount': 5000}, {'open': 50, 'preclose': 50}, CAL[0], rules)
    r = b.close_day(CAL[0], {'A': {'close': 50}})
    assert b.equity_curve()[:2] == [10000, 9994.95] and r['daily_return'] == pytest.approx(-0.000505)


# 切分计划 ------------------------------------------------------------------------------------------------
def test_splits_roll_without_overlap_and_respect_holdout():
    from observe.dataset import plan_splits
    days = list(pd.bdate_range('2015-01-01', periods = 1000).date)
    sp = plan_splits(days, 504, 126, 63, holdout_start = days[900])
    assert len(sp) == 4 and (sp.test_start.iloc[1:].to_numpy() > sp.test_end.iloc[:-1].to_numpy()).all()   # 测试窗首尾相接不重叠
    assert (sp.test_end < days[900]).all() and (sp.select_asof == sp.valid_start).all()
    assert all(days.index(r.test_start) == days.index(r.fit_asof) + 1 for r in sp.itertuples())           # fit_asof 为测试窗前一交易日


def test_samples_respect_maturity_at_select_and_fit():
    from observe.dataset import plan_splits, samples
    days = list(pd.bdate_range('2020-01-01', periods = 40).date); h = 5
    lab = build_labels(bars_n(days), days, h = h)
    sp = plan_splits(days, 20, 10, 5).iloc[0]
    sel, fit = samples(lab, sp, 'select'), samples(lab, sp, 'fit')
    assert sel.matured_at.max() <= sp.select_asof and fit.matured_at.max() <= sp.fit_asof
    assert sel.decision_date.max() == days[days.index(sp.select_asof) - h - 1]                            # 训练窗最后 h+1 天因未成熟被排除
    assert fit.decision_date.max() == days[days.index(sp.fit_asof) - h - 1]                               # 验证窗最后 h+1 天不参与重拟合


def bars_n(days):
    return pd.DataFrame({'date': days, 'instrument': 'X', 'adj_open': [10 + k * 0.1 for k in range(len(days))], 'is_trading': True})
