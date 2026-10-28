"""Independent integer-denominator, source-query and strict-calendar-age expectations."""
import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch105 as study
from scripts.reference_strategy_batch105_age import function


def test_three_source_schemes_keep_distinct_denominators_and_integer_truncation():
    result = study.breadth_fixture()
    assert result['scheme1'] == [50]*20 and result['scheme2'] == [25]*20 and result['scheme3'] == [33]*20
    assert result['vmax'] == 200 and result['industry_last'] == {'one':50,'two':0}
    assert result['calls'][0] == ['calendar_parameters',{'end_date':'2022-09-02','count':40}]
    assert result['calls'][1] == ['securities_date',{'date':'2022-07-11T00:00:00'}]
    assert result['calls'][2][1]['kwargs'] == {'end_date':'2022-09-02','count':40,'fields':'close','panel':False}


def test_missing_close_stays_in_source_denominator_and_is_false():
    result = study.breadth_fixture(missing=True)
    assert result['scheme3'][-1] == 0 and result['scheme3'][-2] == 33
    assert result['last_bias'] == {'A':False,'B':False,'C':False}


def test_duplicate_industry_name_overwrites_without_changing_global_population():
    result = study.breadth_fixture(duplicate_names=True)
    assert result['vmax'] == 100 and result['scheme1'] == result['scheme2'] == [0]*20
    assert result['scheme3'] == [33]*20


def test_no_industry_does_not_manufacture_membership():
    result = study.breadth_fixture(no_industries=True)
    assert result['scheme1'] == result['scheme2'] == [] and result['scheme3'] == [33]*20


def test_pivot_duplicate_time_code_is_rejected_by_original():
    with pytest.raises(ValueError,match='duplicate entries'): study.breadth_fixture(duplicate_rows=True)


@pytest.mark.parametrize('rows,expected',[(0,0),(19,0),(20,0),(21,1),(40,20)])
def test_original_discards_first20_rows_even_though_MA20_exists_at_position19(rows,expected):
    result = study.original_bias(pd.DataFrame({'A':np.arange(float(rows))}))
    assert len(result) == expected


def test_original_generic_datetime_error_and_incomplete_first_fix_stay_visible():
    with pytest.raises(TypeError,match='unit-less'): study.age_fixture()
    with pytest.raises(TypeError,match='NoneType'): study.age_fixture(explicit=True)


def test_compatibility_counter_retains_strict252_natural_days_and_grouping():
    basic = pd.DataFrame({'code':['B','A','outside'],'list_date':pd.to_datetime(['2020-01-02','2020-01-01','2010-01-01'])})
    periods = pd.to_datetime(['2020-09-09','2020-09-10']).to_numpy(dtype='datetime64[ns]')
    result = function(basic,periods)({'A':'group','B':'group'},'first','last',252)
    assert result.to_numpy().tolist() == [[0],[1]]
    assert result.index.tolist() == list(pd.to_datetime(periods)) and result.columns.tolist() == ['group']


@pytest.mark.parametrize('codes',[['A','B'],['B','A']])
def test_compatibility_reference_handles_equal_labels_with_different_index_names(codes):
    dates = {'A':'2020-01-01','B':'2020-01-02'}
    basic = pd.DataFrame({'code':codes,'list_date':pd.to_datetime([dates[c] for c in codes])})
    periods = pd.to_datetime(['2020-09-09','2020-09-10']).to_numpy(dtype='datetime64[ns]')
    result = function(basic,periods)({'A':'group','B':'group'},'first','last',252)
    assert result.to_numpy().tolist() == [[0],[1]]


def test_missing_listing_date_is_not_counted_as_eligible():
    basic = pd.DataFrame({'code':['A','B'],'list_date':pd.to_datetime(['2020-01-01',None])})
    periods = pd.to_datetime(['2020-09-10']).to_numpy(dtype='datetime64[ns]')
    result = function(basic,periods)({'A':'group','B':'group'},'first','last',252)
    assert result.iloc[0,0] == 1


def test_compatibility_reference_does_not_hide_duplicate_stock_basic_codes():
    basic = pd.DataFrame({'code':['A','A'],'list_date':pd.to_datetime(['2020-01-01','2020-01-02'])})
    periods = pd.to_datetime(['2020-09-10']).to_numpy(dtype='datetime64[ns]')
    with pytest.raises(ValueError,match='duplicate labels'):
        function(basic,periods)({'A':'group'},'first','last',252)
