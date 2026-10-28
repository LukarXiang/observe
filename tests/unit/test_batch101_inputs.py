"""Preserve native candidate values and reject corrupt historical operands."""
import numpy as np
import pandas as pd
import pytest

from scripts import prepare_strategy_batch101 as inputs


def test_native_close_and_future_cutoff_are_preserved():
    native = pd.DataFrame({'日期': ['2026-09-28', '2026-09-29', '2026-09-30'], '收盘': [1., 2., 99.]})
    result, excluded = inputs.normalize(native)
    assert excluded == 1 and result.close.tolist() == [1., 2.]
    assert result.date.tolist() == ['2026-09-28', '2026-09-29']
    assert native['收盘'].tolist() == [1., 2., 99.]


@pytest.mark.parametrize('mode', ['duplicate', 'unordered', 'zero', 'nonfinite', 'missing_close'])
def test_native_candidate_invalid_history_is_blocked(mode):
    frame = pd.DataFrame({'date': ['2020-01-02', '2020-01-03'], 'close': [1., 2.]})
    if mode=='duplicate': frame.loc[1, 'date']=frame.loc[0, 'date']
    elif mode=='unordered': frame=frame.iloc[::-1]
    elif mode=='zero': frame.loc[0, 'close']=0.
    elif mode=='nonfinite': frame.loc[0, 'close']=np.nan
    else: frame=frame.drop(columns='close')
    with pytest.raises(ValueError): inputs.normalize(frame)
