import math

import numpy as np
import pandas as pd

from observe.strategy_means import window_fsum


def test_equal_price_boundary_is_independent_of_old_rolling_accumulator():
    window = [5.866102110000001, 5.866102110000001, 5.824276070000001, 5.907928150000001, 5.866102110000001]
    close = pd.Series([4.3, 4.3, *window])
    assert close.iloc[-1] < close.rolling(5).mean().iloc[-1]
    expected = math.fsum(window) / 5
    assert window_fsum(close, 5).iloc[-1] == expected == close.iloc[-1]
    assert window_fsum(pd.Series(window), 5).iloc[-1] == expected


def test_missing_window_and_future_append_do_not_change_past_mean():
    close = pd.Series([1., 2., np.nan, 4., 5., 6., 7.])
    means = window_fsum(close, 3)
    assert means.iloc[:5].isna().all() and means.iloc[5:].tolist() == [5., 6.]
    future = window_fsum(pd.concat([close, pd.Series([1000., 3000.])], ignore_index=True), 3)
    pd.testing.assert_series_equal(future.iloc[:len(close)], means)
