"""Lossless seconds-to-milliseconds promotion passes; altered values and unrelated types fail."""
import pandas as pd
import pytest

from scripts.recover_strategy_batch105_stage import assert_saved_frame


def frames():
    expected = pd.DataFrame({'date':pd.to_datetime(['2022-08-08','2022-09-02']).as_unit('s'),'value':[1.,2.]})
    saved = expected.copy(); saved['date'] = saved.date.dt.as_unit('ms')
    return saved,expected


def test_actual_parquet_precision_promotion_is_lossless():
    saved,expected = frames(); assert_saved_frame(saved,expected)


def test_millisecond_date_change_is_not_hidden_by_precision_conversion():
    saved,expected = frames(); saved.loc[0,'date'] += pd.Timedelta(milliseconds=1)
    with pytest.raises(AssertionError): assert_saved_frame(saved,expected)


def test_unrelated_numeric_dtype_change_remains_rejected():
    saved,expected = frames(); saved['value'] = saved.value.astype('int64')
    with pytest.raises(AssertionError): assert_saved_frame(saved,expected)
