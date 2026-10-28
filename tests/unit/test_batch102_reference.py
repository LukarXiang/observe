"""Independent operands and faithful non-trading scope for macro48/ETF58."""
import numpy as np
import pandas as pd

from scripts import reference_strategy_batch102 as reference


def test_quadratic_terminal_slope_and_previous_stat_month():
    x = np.arange(13)
    values = .1*x*x + .2*x + 40
    frame = pd.DataFrame({'月份': pd.period_range('2023-01', periods=13, freq='M').strftime('%Y年%m月份'), '制造业-指数': values})
    result = reference.pmi_reference(frame).iloc[0]
    assert np.isclose(result.slope, 2.6) and np.isclose(result.independent_slope, 2.6)
    assert result.last_stat_month == '2024-01' and not result.original_point_in_time_signal_proven


def test_missing_trade_date_is_annotated_without_synthesizing_price():
    days = pd.bdate_range('2024-01-02', periods=21).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({'date': days[:10]+days[11:], 'close': np.linspace(100., 110., 20)})
    rows, profiles = reference.etf_reference({'X': frame}, days)
    assert len(rows) == 1 and rows.iloc[0].missing_open_count == 1
    assert np.isclose(rows.iloc[0].inc19_intervals, .1)
    assert profiles['X']['missing_open_dates'] == [days[10]]
    assert not rows.iloc[0].original_attribute_history_proven


def test_annual_transform_keeps109_months_and_fixed10_divisor():
    months = pd.date_range('2005-01-31', periods=109, freq='ME')
    frame = pd.DataFrame({'date': months.strftime('%Y-%m-%d'), 'close': np.arange(109.)+1000.})
    daily, annual = reference.index_reference(frame)
    assert daily.empty and len(annual) == 1
    row = annual.iloc[0]
    assert row.input_month_rows == 109 and row.annual_keys == 10
    assert not row.original1455_include_now_proven
