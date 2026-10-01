"""独立手算因子评价：IC、并列分组、无效标签、覆盖率和 JSON 容差。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from observe.evaluation.research import factor_diagnostics
from observe.runs import compare_tables


def inputs(values, labels):
    days = [date(2024, 1, 2), date(2024, 1, 3)][:len(values)]; rows, lab = [], []
    for d, vs, ys in zip(days, values, labels):
        for i, (v, y) in enumerate(zip(vs, ys)):
            rows.append({'date': d, 'instrument': str(i), 'f': v})
            lab.append({'decision_date': d, 'instrument': str(i), 'value': y, 'valid': True, 'matured_at': d, 'invalid_reason': ''})
    return pd.DataFrame(rows), pd.DataFrame(lab), days


def test_hand_calculated_ic_quantiles_and_rank_autocorrelation():
    fac, lab, days = inputs([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], [[.01, .02, .03, .04, .05]] * 2)
    daily, groups, report = factor_diagnostics(fac, lab, ['f'], days, h = 1, min_n = 5, n_boot = 100, rebalance_every = 1)
    assert daily.ic.tolist() == pytest.approx([1., -1.]) and daily.rank_ic.tolist() == pytest.approx([1., -1.])
    assert daily.long_short_label.tolist() == pytest.approx([.04, -.04])
    assert daily.actual_quantiles.tolist() == [5, 5] and daily.rank_autocorrelation.iloc[1] == pytest.approx(-1.)
    assert groups[groups.date == days[0]].mean_label.tolist() == pytest.approx([.01, .02, .03, .04, .05])
    assert report['factors']['f']['ic_positive_share'] == .5 and report['factors']['f']['rank_ic_ci95'] is None


def test_ties_reduce_groups_and_invalid_labels_never_replace_members():
    fac, lab, days = inputs([[1] * 8 + [9, 10]], [np.arange(10) / 100])
    lab.loc[lab.instrument == '9', ['valid', 'value']] = [False, np.nan]
    daily, groups, _ = factor_diagnostics(fac, lab, ['f'], days, h = 1, min_n = 5, n_boot = 100)
    assert daily.actual_quantiles.iloc[0] < 5 and groups.selected_count.sum() == 10 and groups.valid_label_count.sum() == 9
    assert groups.iloc[-1].selected_count == 2 and groups.iloc[-1].mean_label == pytest.approx(.08)


def test_daily_equal_weight_coverage_and_hidden_holdout_labels():
    fac, lab, days = inputs([[1, 2, 3, 4, 5], [1] + [np.nan] * 14], [np.arange(5), np.arange(15)])
    _, _, report = factor_diagnostics(fac, lab, ['f'], days, h = 1, min_n = 2, n_boot = 100)
    assert report['factors']['f']['coverage_daily_mean'] == pytest.approx((1 + 1 / 15) / 2)
    hold = date(2024, 1, 4); lab['matured_at'] = hold
    a = factor_diagnostics(fac, lab, ['f'], days, 1, min_n = 2, n_boot = 100, holdout = hold)
    lab['value'] += 1e6
    b = factor_diagnostics(fac, lab, ['f'], days, 1, min_n = 2, n_boot = 100, holdout = hold)
    pd.testing.assert_frame_equal(a[0], b[0]); pd.testing.assert_frame_equal(a[1], b[1]); assert a[2] == b[2]


def test_nested_json_numeric_tolerance_preserves_structural_differences():
    a = {'model_eval': {'selection': [{'coef_norm': .00299097676770259, 'selected': 'ridge@1'}]}}
    b = {'model_eval': {'selection': [{'coef_norm': .00299097676770269, 'selected': 'ridge@1'}]}}
    assert compare_tables(a, b, abs_tol = 1e-9)[1]['model_eval']['differences'] == 0
    assert compare_tables(a, b, abs_tol = 1e-18)[1]['model_eval']['differences'] == 1
    b['model_eval']['selection'][0]['selected'] = 'ridge@10'
    assert compare_tables(a, b, abs_tol = 1.)[1]['model_eval']['differences'] == 1


def test_rank_autocorrelation_keeps_each_days_full_candidate_ranks():
    fac, lab, days = inputs([[1, 2, 3, 4], [1, 2, 3, 4]], [[.01, .02, .03, .04]] * 2)
    fac.loc[fac.date == days[1], 'instrument'] = ['0', '2', '3', '4']
    lab.loc[lab.decision_date == days[1], 'instrument'] = ['0', '2', '3', '4']
    daily, _, _ = factor_diagnostics(fac, lab, ['f'], days, 1, min_n = 2, n_boot = 100)
    assert daily.rank_autocorrelation.iloc[1] == pytest.approx(np.corrcoef([1, 3, 4], [1, 2, 3])[0, 1])
    strict, _, _ = factor_diagnostics(fac, lab, ['f'], days, 1, min_n = 4, n_boot = 100)
    assert pd.isna(strict.rank_autocorrelation.iloc[1])
