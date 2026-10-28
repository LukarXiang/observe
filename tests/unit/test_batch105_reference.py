"""Reference semantics on native gaps, strict floating boundaries and controlled calendar ages."""
import math

import numpy as np
import pandas as pd

from scripts import reference_strategy_batch105 as reference


def test_native_row_operand_gaps_do_not_create_calendar_rows_or_replace_nan():
    dates = pd.bdate_range('2020-01-01',periods=43).delete([5,9,11])
    prices = pd.DataFrame({'date':dates,'instrument':'A','close':np.arange(1.,41.),'is_trading':True})
    prices.loc[22,['close','is_trading']] = [np.nan,False]
    output,profiles = reference.native_bias(prices)
    assert len(output) == 20 and profiles[0]['nontrading_rows'] == profiles[0]['missing_close_rows'] == 1
    assert output.date.tolist() == dates[20:].tolist()
    assert not output.loc[output.native_position >= 22,'original_bias'].any()
    assert output.loc[output.native_position == 22,'close'].isna().all()


def test_independent_full20_mean_uses_fsum_and_preserves_strict_equality():
    values = np.array([10.]*21)
    means = reference.independent_means(values)
    assert np.isnan(means[:19]).all() and means[19:].tolist() == [10.,10.]
    assert not (values > means).any()
    unstable = np.array([1e16,1.,-1e16]+[1.]*17)
    assert reference.independent_means(unstable)[-1] == math.fsum(unstable)/20


def test_calendar_counter_uses_controlled_exchange_and_all_available_open_days():
    instruments = pd.DataFrame({'instrument':['A','B','C'],'exchange':['SZ','SH','SZ'],
        'list_date':pd.to_datetime(['2020-01-01','2020-01-02',None])})
    calendar = pd.DataFrame({'date':pd.to_datetime(['2020-09-09','2020-09-10','2020-09-11']),
        'is_open':[True,True,False]})
    output,scope = reference.calendar_age(instruments,calendar)
    assert scope['checked_security_calendar_cells'] == 6 and scope['missing_listing_dates'] == 1
    assert scope['source_industry_mapping_proven'] is False and len(output) == 4
    assert output.source_compatibility_count.tolist() == [0,0,0,1]
    assert output.source_compatibility_count.equals(output.independent_count)
