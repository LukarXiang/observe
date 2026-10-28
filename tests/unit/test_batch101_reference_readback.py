"""Saved-window readback must reject changed values, dates and evidence scope."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts import reference_strategy_batch101 as reference
from scripts import verify_strategy_batch101_reference as validation


def fixture():
    calendar = pd.bdate_range('2024-01-02', periods=121).strftime('%Y-%m-%d').tolist()
    native = pd.DataFrame({'date': calendar, 'close': np.linspace(100., 80., 121)})
    frames = {'X': native}; frame, summary = reference.components(frames, calendar)
    return frame, summary, frames, calendar


def test_saved_ES_readback_covers_every_row():
    result = validation.readback(*fixture())
    assert result['all_saved_ES_values_checked'] == 4 and result['new_trading_backtests'] == 0


@pytest.mark.parametrize('mode', ['value', 'date', 'missing', 'duplicate', 'scope'])
def test_saved_reference_corruption_is_rejected(mode):
    frame, summary, frames, calendar = fixture(); summary = copy.deepcopy(summary)
    if mode == 'value': frame.loc[0, ['ES', 'independent_ES']] = .99
    elif mode == 'date': frame.loc[0, 'window_first'] = calendar[1]
    elif mode == 'missing': frame.loc[0, 'missing_open_dates'] = '["2024-01-03"]'
    elif mode == 'duplicate': frame = pd.concat([frame.iloc[[0]], frame.iloc[[0]], frame.iloc[2:]], ignore_index=True)
    else: summary['strict_usable'] = True
    with pytest.raises(ValueError): validation.readback(frame, summary, frames, calendar)
