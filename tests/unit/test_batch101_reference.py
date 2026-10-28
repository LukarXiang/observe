"""Hand-return boundaries and preserved ES denominator; no fabricated portfolio NAV."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts import reference_strategy_batch101 as reference


@pytest.mark.parametrize('policy, expected', [
    ('explicit_forward_fill', [0., -.1, 0., 0., -.1]),
    ('no_forward_fill', [0., -.1, 0., 0., 0.])])
def test_natural_day_weekend_returns_match_hand_values(policy, expected):
    frame = pd.DataFrame({'date': ['2024-01-04', '2024-01-05', '2024-01-08'], 'close': [100., 90., 81.]})
    for function in (reference.reference_returns, reference.independent_returns):
        np.testing.assert_allclose(function(frame, policy), expected, rtol=0, atol=1e-15)


def test_no_fill_missing_trade_day_does_not_use_previous_observation():
    frame = pd.DataFrame({'date': ['2024-01-02', '2024-01-04'], 'close': [100., 80.]})
    np.testing.assert_array_equal(reference.reference_returns(frame, 'no_forward_fill'), [0., 0., 0.])
    np.testing.assert_allclose(reference.reference_returns(frame, 'explicit_forward_fill'), [0., 0., -.2], atol=1e-15)
    assert frame.close.tolist() == [100., 80.]


def test_unchanged_ES_array_inclusive_two_tail_fixed120_denominator():
    values = np.array([-.2, -.1]+[0.]*168)
    output = reference.original_ES_function()(values)
    assert output.shape == (1,) and math.isclose(output[0], .3, abs_tol=1e-15)
    assert reference.original_ES_function()([0.]*170)[0] == 0


def test_ES_reference_prefix_does_not_change_after_later_prices():
    first = pd.bdate_range('2024-01-02', periods=121)
    frame = pd.DataFrame({'date': first.strftime('%Y-%m-%d'), 'close': np.linspace(100., 90., 121)})
    calendar = first.strftime('%Y-%m-%d').tolist()
    before, _ = reference.components({'X': frame}, calendar)
    frame.loc[120, 'close'] = 1000.
    after, _ = reference.components({'X': frame}, calendar)
    pd.testing.assert_frame_equal(before.loc[before.asof_native_close.eq(calendar[119])],
                                  after.loc[after.asof_native_close.eq(calendar[119])], check_exact=True)


def test_crossing_missing_open_date_is_marked_and_not_a_portfolio():
    calendar = pd.bdate_range('2024-01-02', periods=122).strftime('%Y-%m-%d').tolist()
    observations = calendar[:60]+calendar[61:]
    frame = pd.DataFrame({'date': observations, 'close': np.linspace(100., 80., len(observations))})
    output, summary = reference.components({'X': frame}, calendar)
    assert len(output) == 4 and output.missing_open_count.tolist() == [1]*4
    assert output.missing_open_dates.tolist() == ['["'+calendar[60]+'"]']*4
    assert not output.original_return_input_proven.any()
    assert summary['profiles']['X']['gap_crossing_windows_per_policy'] == 2
    assert summary['new_trading_backtests'] == summary['full_original_portfolio_weights'] == 0


@pytest.mark.parametrize('mode', ['too_short', 'future', 'nonopen', 'duplicate'])
def test_invalid_reference_operand_is_rejected(mode):
    calendar = pd.bdate_range('2024-01-02', periods=120).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({'date': calendar, 'close': np.linspace(100., 90., 120)})
    if mode == 'too_short': frame = frame.iloc[:119]
    elif mode == 'future': frame.loc[119, 'date'] = '2026-09-30'
    elif mode == 'nonopen': calendar = calendar[1:]
    else: frame.loc[119, 'date'] = frame.loc[118, 'date']
    with pytest.raises(ValueError): reference.components({'X': frame}, calendar)


def test_unknown_reference_policy_is_rejected():
    for function in (reference.reference_returns, reference.independent_returns):
        with pytest.raises(ValueError, match='Unknown return'): function(pd.DataFrame(), 'implicit_default')
