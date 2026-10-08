"""User-approved per-window summation without a cumulative rolling accumulator."""
import math


def window_fsum(close, window):
    return close.rolling(window, min_periods=window).apply(lambda values: math.fsum(values) / window, raw=True)
