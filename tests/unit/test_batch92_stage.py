"""Check original notebook rules and causal windows without admitting index trades."""
import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch92 as batch


def test_full_source_is_pinned_and_plot_top_level_is_excluded():
    definitions = batch.definitions(batch.tree())
    assert len(definitions) == 7
    assert len(batch.SOURCE.read_text().splitlines()) == 807
    assert 'plot_ochl' not in batch.namespace()


@pytest.mark.parametrize('prices, expected', [(np.arange(1., 61.), 3136.),
    (np.arange(100., 40., -1.), 3136.), (np.full(60, 100.), 0.), (np.tile([100., 101.], 30), 56.)])
def test_original_absolute_score_known_paths_and_independent_sum(prices, expected):
    ns = batch.namespace()
    assert ns['calc_trend_score'](pd.Series(prices)) == expected
    assert batch.independent_score(prices.tolist()) == expected


def test_original_equal_price_sign_is_zero_and_initial_compound_unit_is_retained():
    normal = batch.namespace()['Normalize_Trend'](pd.Series([1., 2., 3., 4., 5.]))
    assert normal.normalize_monotone().tolist() == [0., 1., 2., 3., 4.]
    assert normal.normalize_compound(5).tolist() == [0., 1.]
    flat = batch.namespace()['Normalize_Trend'](pd.Series([2.]*5))
    assert flat.normalize_movingaverage(5).tolist() == [0., 0.]
    assert flat.normalize_compound(5).tolist() == [0., 0.]


def test_original_absolute_points_include_all_extrema_and_ends():
    score = batch.namespace()['Tren_Score'](pd.Series([0., 2., 1., 2., 0., 1.]))
    assert score._get_absolute_point() == [True, True, False, True, True, True]
    assert score.calc_trend_score('absolute') is None
    assert score.score['absolute'].trend_score == 9.
    assert score.score['absolute'].act_score == 1.


def test_relative_branch_current_pandas_error_is_preserved():
    score = batch.namespace()['Tren_Score'](pd.Series([0., 1., 0.]))
    with pytest.raises(TypeError, match='method'):
        score.calc_trend_score('opposite')


def test_mixed_score_none_returns_are_a_separate_synthetic_diagnosis(monkeypatch):
    score = batch.namespace()['Tren_Score'](pd.Series([0., 1., 0.]))
    # The real relative branch fails earlier; this injects None to isolate the later max expression.
    monkeypatch.setattr(score, 'calc_trend_score', lambda _: None)
    with pytest.raises(TypeError): score.calc_absolute_score()


def test_flag_exit_has_priority_when_both_crossings_are_true():
    frame = pd.DataFrame({'score': [5., 7., 1., 7.], 'lower_bound': [2., 8., 2., 2.],
        'upper_bound': [10., 6., 6., 6.]}, index=pd.date_range('2020-01-01', periods=4))
    flag = batch.namespace()['get_hold_flag'](frame)
    assert flag.tolist() == [0., 0., 0., 1.]
    pd.testing.assert_series_equal(flag, batch.independent_flags(frame))


def test_strict_previous_bound_and_inclusive_current_bound_are_preserved():
    frame = pd.DataFrame({'score': [6., 7., 5., 6., 1.], 'lower_bound': [1.]*5, 'upper_bound': [6.]*5})
    assert batch.namespace()['get_hold_flag'](frame).tolist() == [0., 0., 0., 1., 0.]


def test_later_prices_do_not_change_original_score_bounds_or_flags():
    ns = batch.namespace()
    prices = pd.Series(100+np.sin(np.arange(180)/5)*10+np.arange(180)/100)
    def evaluate(close):
        score = close.rolling(60).apply(ns['calc_trend_score'], raw=False)
        frame = pd.DataFrame({'score': score, 'lower_bound': score.rolling(20).quantile(.05),
            'upper_bound': score.rolling(20).quantile(.85)})
        return frame.assign(flag=ns['get_hold_flag'](frame))
    changed = prices.copy(); changed.iloc[110:] *= 5
    pd.testing.assert_frame_equal(evaluate(prices).iloc[:110], evaluate(changed).iloc[:110], check_exact=True)


def test_quantiles_use_linear_interpolation_and_index_is_not_a_tradeable_security():
    assert batch.independent_quantile(list(range(20)), .05) == pytest.approx(.95)
    assert batch.independent_quantile(list(range(20)), .85) == pytest.approx(16.15)
    with pytest.raises(ValueError, match='60 finite positive'):
        batch.independent_score([1.]*59)
