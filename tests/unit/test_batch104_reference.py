"""Independent rolling-window operands and source boundary checks."""
import numpy as np
import pandas as pd
import pytest

from scripts import reference_strategy_batch104 as reference


def test_nested_original_means_match_hand_calculated31_row_arithmetic():
    values = pd.Series(np.arange(1., 32.))
    original = reference.nested()
    assert original['caculate_close_ma_5_10'](values) == (29., 26.5, 28., 25.5)
    assert original['caculate_close_ma_20_30'](values) == (21.5, 16.5, 20.5, 15.5)
    assert reference.independent_means(values.to_numpy(), (5, 10, 20, 30)) == [29., 26.5, 21.5, 16.5, 28., 25.5, 20.5, 15.5]


def test_inclusive_relative_point3_and_earliest_windows_with_constant_bounds():
    dates = pd.bdate_range('2024-01-02', periods=31)
    frame = pd.DataFrame({'date': dates, 'instrument': 'A', 'open': 6., 'close': 6.,
        'high': 20., 'low': 0., 'volume': 100., 'is_trading': True})
    ma, relative, summary = reference.compute(frame)
    assert len(ma) == 1 and len(relative) == 19
    assert summary['relative_year_windows'] == {'15': {'2024': 17}, '30': {'2024': 2}}
    assert relative.original_selected.all() and relative.independent_ratio.eq(.3).all()
    assert not relative.disagreement.any()
    assert not ma.iloc[0].original_all_conditions and ma.iloc[0].condition_disagreements == 0


def test_native_nan_suspended_row_stays_missing_and_original_mean_skips_it():
    frame = pd.DataFrame({'date': pd.bdate_range('2024-01-02', periods=31), 'instrument': 'A',
        'open': 6., 'close': 6., 'high': 20., 'low': 0., 'volume': 100., 'is_trading': True})
    frame.loc[30, ['open', 'close', 'high', 'low']] = np.nan
    frame.loc[30, ['volume', 'is_trading']] = [0., False]
    ma, relative, summary = reference.compute(frame)
    assert pd.isna(frame.loc[30, 'close'])
    assert ma.iloc[0].close_means == [6.]*8 and ma.iloc[0].nontrading_rows == 1
    final = relative[relative.window_last.eq('2024-02-13')]
    assert len(final) == 2 and final.close.isna().all() and not final.original_selected.any()
    assert summary['profiles']['A']['nontrading_rows'] == 1


@pytest.mark.parametrize('close,second_accepted', [(8., True), (10., True), (10.02, False)])
def test_second_relative_branch_keeps_point5_and_first_keeps_point3(close, second_accepted):
    frame = pd.DataFrame({'date': pd.bdate_range('2024-01-02', periods=31), 'instrument': 'A',
        'open': close, 'close': close, 'high': 20., 'low': 0., 'volume': 100., 'is_trading': True})
    _, relative, _ = reference.compute(frame)
    assert not relative.loc[relative.length.eq(15), 'original_selected'].any()
    assert relative.loc[relative.length.eq(30), 'original_selected'].eq(second_accepted).all()
    assert not relative.disagreement.any()
