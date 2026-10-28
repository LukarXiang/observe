"""Original input dtype, MA warmup, zero/NaN and exact heat transition boundaries."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts import reference_strategy_batch103_emotion as emotion


def test_native_integer_input_keeps_original_exception_fallback_not_float_signal():
    values = np.asarray([1000]*95+[100]*5, dtype='int64')
    assert emotion.callback(values) == (1, 0.)
    assert emotion.callback(values.astype('float64')) == (-1, -72.)
    doc = emotion.dtype_diagnostics()
    assert doc['rows'][0]['direct_TA_Lib_error']['message'] == 'input array type is not double'
    assert doc['rows'][1]['direct_TA_Lib_error'] is None
    assert doc['economic_or_compatibility_fix_approved'] is False


@pytest.mark.parametrize('count,expected', [(4, 0), (5, -1), (6, -1)])
def test_exact_negative_run_boundary_independently_matches_original(count, expected):
    values = np.asarray([1000]*(100-count)+[100]*count, dtype=float)
    original, rate = emotion.callback(values)
    independent, independent_rate, crossing, _ = emotion.independent(values)
    assert original == independent == expected and rate == independent_rate
    assert crossing == count


@pytest.mark.parametrize('values', [np.full(100, 100.), np.zeros(100), np.full(100, np.nan)])
def test_flat_zero_or_nan_history_preserves_none_return(values):
    result, rate = emotion.callback(values)
    independent, independent_rate, crossing, _ = emotion.independent(values)
    assert result is independent is None and crossing is None
    if np.isfinite(values).all() and values[-1] > 0:
        assert rate == independent_rate == 0.
    else:
        assert math.isnan(rate) and independent_rate is None


def test_native_volume_projection_retains_integer_dtype_and_zero_values():
    dates = pd.bdate_range('2026-05-01', periods=110)
    frame = pd.DataFrame({'date': dates, 'volume': np.arange(110, dtype='int64')})
    projected, excluded = emotion.normalize_volume(frame)
    assert projected.volume.dtype == frame.volume.dtype and projected.volume.iloc[0] == 0
    assert len(projected)+excluded == 110 and projected.date.iloc[-1] <= '2026-09-29'


@pytest.mark.parametrize('kind', ['negative', 'nan', 'duplicate', 'unordered'])
def test_invalid_candidate_rejected_without_fill_or_implicit_type_conversion(kind):
    frame = pd.DataFrame({'date': pd.bdate_range('2024-01-02', periods=100), 'volume': np.ones(100)})
    if kind == 'negative': frame.loc[10, 'volume'] = -1
    elif kind == 'nan': frame.loc[10, 'volume'] = np.nan
    elif kind == 'duplicate': frame.loc[10, 'date'] = frame.loc[11, 'date']
    else: frame.loc[[10, 11], 'date'] = frame.loc[[11, 10], 'date'].to_numpy()
    with pytest.raises(ValueError): emotion.normalize_volume(frame)


def test_compute_preserves_native_and_explicit_float_outputs_and_calendar_unknowns():
    days = pd.bdate_range('2004-01-02', periods=102).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({'date': days, 'volume': [1000]*97+[100]*5})
    native = frame.copy(deep=True)
    result, profiles = emotion.compute({'000016.XSHG': frame}, days[1:])
    pd.testing.assert_frame_equal(frame, native)
    assert len(result) == 3 and result.native_result.tolist() == ['1']*3
    assert result.iloc[-1].float_reference_result == '-1' and result.iloc[-1].independent_result == '-1'
    assert result.iloc[-1].native_vs_float_state_disagreement
    assert not result.iloc[0].calendar_complete_for_window and result.iloc[1].calendar_complete_for_window
    assert result.float_state_disagreement.sum() == 0
    assert profiles['000016.XSHG']['year_windows'] == {'2004': 3}
