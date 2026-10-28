"""Check numeric adaptation against the original string-input failure."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import recover_strategy_batch75 as recovery
from scripts import review_strategy_batch75 as batch


def namespace(frame):
    return {'g': SimpleNamespace(stock_pool=list(batch.POOL), stock_num=1, momentum_day=29),
        'np': np, 'math': math, 'attribute_history': lambda stock, *a, **kw: frame[stock].copy()}


def test_original_string_input_crashes_and_numeric_adapter_matches_float_rank():
    numeric = {stock: pd.DataFrame({'close': np.exp(np.arange(29)*rate)})
        for stock, rate in zip(batch.POOL, [.001, .002, -.001], strict=True)}
    strings = {stock: frame.astype(str) for stock, frame in numeric.items()}
    before = {stock: frame.copy(deep=True) for stock, frame in strings.items()}
    original = batch.selected(batch.DIRECTORY, 1, ['get_rank'], namespace(strings))
    with pytest.raises(TypeError, match='log'):
        original['get_rank'](list(batch.POOL))
    adapted = recovery.numeric_selected(batch.DIRECTORY, 1, ['get_rank'], namespace(strings))
    expected = batch.selected(batch.DIRECTORY, 1, ['get_rank'], namespace(numeric))
    assert adapted['get_rank'](list(batch.POOL)) == expected['get_rank'](list(batch.POOL)) == [batch.POOL[1]]
    for stock in batch.POOL:
        pd.testing.assert_frame_equal(strings[stock], before[stock], check_exact=True)


def test_unparseable_price_is_rejected_without_filling():
    frames = {stock: pd.DataFrame({'close': ['broken']*29}) for stock in batch.POOL}
    adapted = recovery.numeric_selected(batch.DIRECTORY, 1, ['get_rank'], namespace(frames))
    with pytest.raises(ValueError, match='broken'):
        adapted['get_rank'](list(batch.POOL))


def test_adapter_preserves_dates_missing_values_and_original_numeric_precision():
    frame = pd.DataFrame({'close': ['2.604', None, '2.644'], 'date': ['A', 'B', 'C']})
    before = frame.copy(deep=True)
    frames = {stock: frame for stock in batch.POOL}
    adapted = recovery.numeric_selected(batch.DIRECTORY, 1, ['get_rank'], namespace(frames))
    result = adapted['attribute_history'](batch.POOL[0], 29)
    assert result.date.tolist() == ['A', 'B', 'C']
    np.testing.assert_array_equal(result.close.to_numpy(), np.array([2.604, np.nan, 2.644]))
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_recovery_checks_include_adapter_and_regression_test():
    assert 'scripts/recover_strategy_batch75.py' in recovery.FILES
    assert 'tests/unit/test_batch75_recovery.py' in recovery.FILES
    assert 'scripts/recover_strategy_batch75.py' in recovery.commands()[0]
    assert 'tests/unit/test_batch75_recovery.py' in recovery.commands()[1]
