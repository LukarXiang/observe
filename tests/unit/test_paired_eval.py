"""成对比较的统计部分（模块 17）：块抽样区间、逐日差值、分窗口与分年份。手工构造的小例子。"""
import numpy as np
import pandas as pd
import pytest

from observe.evaluation.paired import block_ci, curve_pairs, diff_stats, model_pairs


def series(values, start = '2024-01-02'):
    idx = [d.date() for d in pd.bdate_range(start, periods = len(values))]; return pd.Series(values, index = idx, dtype = float)


def test_block_ci_is_deterministic_and_covers_the_mean():
    x = np.random.default_rng(0).normal(0.03, 0.1, 400); lo, hi = block_ci(x, 20, 500, seed = 1)
    assert (lo, hi) == tuple(block_ci(x, 20, 500, seed = 1)) and lo < x.mean() < hi and hi - lo < 0.06
    assert block_ci(x, 20, 500, seed = 2) != [lo, hi]
    assert block_ci(np.full(100, 0.5), 20) == [0.5, 0.5]
    assert block_ci(x[:10], 20) is None and block_ci([np.nan] * 50, 20) is None


def test_block_ci_widens_when_the_series_is_autocorrelated():
    rng = np.random.default_rng(3); e = rng.normal(0, 1, 600); ar = np.zeros(600)
    for k in range(1, 600): ar[k] = 0.9 * ar[k - 1] + e[k]
    iid = block_ci(e, 20, 800, seed = 0); auto = block_ci(ar, 20, 800, seed = 0)
    assert auto[1] - auto[0] > 2 * (iid[1] - iid[0])                       # 相邻日重叠标签造成的自相关不会被当成独立样本


def test_diff_stats_uses_only_days_both_sides_have():
    a = series([0.1, 0.2, np.nan, 0.4, 0.5]); b = series([0.2, 0.2, 0.9, 0.3, 0.7]); b.iloc[4] = np.nan
    s = diff_stats(a, b, 2, 100, 0)
    assert s['days'] == 3 and s['a_mean'] == pytest.approx((0.1 + 0.2 + 0.4) / 3) and s['b_mean'] == pytest.approx((0.2 + 0.2 + 0.3) / 3)
    assert s['diff_mean'] == pytest.approx(s['b_mean'] - s['a_mean']) and s['b_higher_share'] == pytest.approx(1 / 3)


def frame(scores, labels, model = 'm', split = 0):
    days = [d.date() for d in pd.bdate_range('2024-01-02', periods = len(scores))]; rows = []
    for d, (sc, lb) in zip(days, zip(scores, labels)):
        for i, (s_, l_) in enumerate(zip(sc, lb)): rows.append({'model_id': model, 'decision_date': d, 'instrument': f'{i:06d}.SH', 'score': s_, 'value': l_, 'split_id': split})
    df = pd.DataFrame(rows); return df[['model_id', 'decision_date', 'instrument', 'score', 'split_id']], df[['decision_date', 'instrument', 'value']].assign(valid = True)


def test_model_pairs_hand_computed():
    rng = np.random.default_rng(1); n_days, n = 40, 12; labels = [rng.normal(0, 1, n) for _ in range(n_days)]
    good = [l + rng.normal(0, 0.3, n) for l in labels]; bad = [rng.normal(0, 1, n) for _ in range(n_days)]
    pa, lab = frame(bad, labels); pb, _ = frame(good, labels)
    r = model_pairs(pa, pb, lab, 5, min_n = 10, top_n = 3, n_boot = 200, seed = 0)['m']
    assert r['rank_ic']['days'] == n_days and r['rank_ic']['diff_mean'] > 0.5 and r['rank_ic']['b_higher_share'] > 0.95 and r['rank_ic']['diff_ci95'][0] > 0
    assert r['top3_mean_label']['diff_mean'] > 0 and r['windows'] == 1 and r['window_wins_b'] == 1 and set(r['by_year']) == {2024} and not r['identical_scores']
    same = model_pairs(pa, pa, lab, 5, min_n = 10, top_n = 3, n_boot = 200, seed = 0)['m']
    assert same['rank_ic']['diff_mean'] == 0 and same['identical_scores'] and same['window_wins_b'] == 0


def test_curve_pairs_windows_and_daily_difference():
    days = [d.date() for d in pd.bdate_range('2024-01-01', periods=7)]
    a = [{'date': str(d), 'equity': e} for d, e in zip(days, [100, 101, 102, 103, 104, 105, 106])]
    b = [{'date': str(d), 'equity': e} for d, e in zip(days, [100, 102, 104, 106, 108, 110, 112])]
    r = curve_pairs(a, b, [(0, days[1], days[3]), (1, days[4], days[6])], 3, 100, 0)
    assert r['windows'] == 2 and r['window_wins_b'] == 2 and r['by_window'][0]['a'] == pytest.approx(103 / 100 - 1) and r['by_window'][0]['b'] == pytest.approx(106 / 100 - 1)      # 窗口内各日收益连乘（含窗口首日相对前一日）
    assert r['daily_return']['days'] == 6 and r['daily_return']['diff_mean'] > 0


def test_curve_pairs_truncates_at_the_first_unreliable_day():
    days = [d.date() for d in pd.bdate_range('2024-01-01', periods=8)]
    a = [{'date': str(d), 'equity': 100 + k} for k, d in enumerate(days)]; b = [{'date': str(d), 'equity': 100 + 2 * k} for k, d in enumerate(days)]
    r = curve_pairs(a, b, [(0, days[1], days[3]), (1, days[4], days[7])], 3, 100, 0, until = days[5])
    assert r['period'] == {'first': str(days[1]), 'last': str(days[4]), 'days': 4, 'truncated_before': str(days[5])} and r['daily_return']['days'] == 4
    assert [w['days'] for w in r['by_window']] == [3, 1] and r['a_total_return'] == pytest.approx(104 / 100 - 1) and r['b_total_return'] == pytest.approx(108 / 100 - 1)
    empty = curve_pairs(a, b, [(0, days[6], days[7])], 3, 100, 0, until = days[5]); assert empty['by_window'] == [] and empty['windows'] == 0
