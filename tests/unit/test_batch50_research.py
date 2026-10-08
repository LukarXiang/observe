import numpy as np
import pandas as pd
import pytest

from scripts.review_strategy_batch50 import continuous_reference, parse_monthly


def test_monthly_sorts_and_parses_without_filling():
    frame = pd.DataFrame({'月份': ['2020年02月份', '2020年01月份'], 'value': ['2', '1']})
    result = parse_monthly(frame, ['value'])
    assert result.index.tolist() == ['2020-01', '2020-02']
    assert result.value.tolist() == [1, 2]


@pytest.mark.parametrize('months,values', [(['2020年01月份']*2, ['1', '2']),
    (['2020年01月份', '2020年03月份'], ['1', '2']), (['2020年01月份'], [None]),
    (['2020年01月份'], ['inf']), (['2020年01月份'], ['invalid'])])
def test_monthly_rejects_gaps_duplicates_and_nonfinite(months, values):
    with pytest.raises((AssertionError, ValueError)):
        parse_monthly(pd.DataFrame({'月份': months, 'value': values}), ['value'])


def test_continuous_strict_float32_tie_and_delay():
    assert continuous_reference([1, 2, 3, 4], 2, 1, 'up') == [(3, 1)]
    tiny = float(np.nextafter(np.float64(2), np.float64(3)))
    assert continuous_reference([1, 2, tiny], 2, 0, 'up') == [(2, 0)]


def test_original_misspelled_down_branch_and_nan():
    assert continuous_reference([3, 2, 1], 2, 0, 'dowm') == [(2, 1)]
    assert continuous_reference([3, np.nan, 1], 2, 0, 'dowm') == [(2, 0)]


@pytest.mark.parametrize('n,delay,how', [(0, 0, 'up'), (2, -1, 'up'), (2, 0, 'invalid')])
def test_continuous_rejects_invalid_parameters(n, delay, how):
    with pytest.raises(ValueError): continuous_reference([1, 2, 3], n, delay, how)
