"""Original input/state boundaries; diagnostics are never execution evidence."""
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import review_strategy_batch96 as stage


@pytest.fixture
def directory(tmp_path):
    folder = tmp_path/'sources'; folder.mkdir()
    for number in range(2): shutil.copyfile(stage.PREREAD/f'{number}.source',folder/f'{number}.source')
    return tmp_path


def indexed(directory):
    return {r['case']:r for r in stage.diagnostics(directory)['cases']}


def test_original_csv_order_and_missing_date(directory):
    cases = indexed(directory)
    assert cases['csv_order']['value'] == ['600519.XSHG','600036.XSHG']
    assert cases['csv_missing_date']['error']['type'] == 'KeyError'
    # Duplicate labels change loc from Series to DataFrame: returned index is dates, not stock codes.
    assert cases['csv_duplicate_date']['value'] == []


@pytest.mark.parametrize('count',[-1,0,1,2,3])
def test_exact_one_percent_clears_and_resets(directory,count):
    cases = indexed(directory)
    row = cases[f'rf_clock_{count}_{[.01,.01]}']
    assert row['error'] is None
    assert row['value'] == {'count':0,'events':['clear']}


def test_rf_period_and_original_csv_not_truncated(directory):
    ns = stage.rf_namespace(directory); g = ns['g']
    assert (g.period,g.adjust_position_hour,g.adjust_position_minute,g.buy_stock_count)==(3,14,50,5)
    assert (g.index2,g.index8)==('000016.XSHG','399333.XSHE')
    assert not any(getattr(g,n) for n in ('is_rank_stock','pick_by_pe','pick_by_eps',
        'filter_gem','filter_blacklist','is_market_stop_loss_by_price','is_market_stop_loss_by_3_black_crows',
        'is_market_stop_loss_by_28_index','is_stock_stop_loss','is_stock_stop_profit','is_equity_curve_protect'))
    cases = indexed(directory)
    row = cases[f'rf_clock_0_{[np.nextafter(.01,math.inf),.01]}']
    assert row['value']['events']==[['adjust',['A','B','C','D','E','F']]]


def test_nan_growth_injected_into_clock_is_not_auto_clear(directory):
    cases = indexed(directory)
    assert cases['rf_clock_0_[nan, nan]']['value']['events'][0][0] == 'adjust'
    assert cases['rf_missing_isnan']['error']['type'] == 'NameError'
    assert cases['rf_history_datetime']['error']['type'] == 'KeyError'
    assert cases['rf_history_range']['value']==90.


def test_cross_shortage_and_partial_mutation(directory):
    cases = indexed(directory)
    assert cases['zhu_core_flat']['error']['type']=='IndexError'
    assert cases['zhu_core_flat']['state_lengths_after']==[199]*4
    assert cases['zhu_partial_state_on_error']['error']['type']=='IndexError'
    assert cases['zhu_partial_state_on_error']['state_lengths_after']==[200,200,199,199]
    assert cases['zhu_core_oscillating']['error'] is None
    assert cases['zhu_core_oscillating']['value']['lengths']==[199]*4


def test_fixed_barslast_and_strict_24_day_helper(directory):
    cases = indexed(directory)
    assert cases['barslast_199']['value']==[1]
    assert cases['barslast_200']['value']==[0]
    assert cases['tradeHelper1_23']['value'] is True
    assert cases['tradeHelper1_24']['value'] is None
    assert cases['tradeHelper_empty']['error']['type']=='IndexError'


def test_original_reference_index_switch(directory):
    cases = indexed(directory)
    assert cases['index_switch_2016-12-30']['value']=='000001.XSHG'
    assert cases['index_switch_2017-01-02']['value']=='399001.XSHE'


def test_zero_cross_boundaries_from_original_helper(directory):
    ns = stage.zhu_namespace(directory)
    ns['macdx']=lambda values:[None,None,np.array([-1.,0.,1.,0.,-1.])]
    dead,gold=ns['hasCross']([1.]*5)
    assert dead==[False,False,False,True]
    assert gold==[False,True,False,False]


@pytest.mark.parametrize('period',[9,12,25,26,90])
def test_expanding_seed_is_preserved(directory,period):
    values=[10.,20.,15.,18.,25.]*40
    ns=stage.zhu_namespace(directory)
    original=np.asarray(ns['ema'](values,period))
    assert original[0]==10.
    assert original[1]==pytest.approx(20.*2/3+10./3)
    np.testing.assert_allclose(original,stage.scalar_ema(values,period),rtol=1e-12,atol=1e-10)


def test_tampered_source_rejected_before_execution(directory):
    path=directory/'sources/1.source';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='identity'): stage.literal(directory,1,{})


def test_only_original_adjustment_time_invokes_clock(directory):
    import datetime
    ns=stage.rf_namespace(directory);events=[]
    ns['do_handle_data']=lambda *a:events.append('clock')
    for minute in (49,50,51): ns['handle_data'](SimpleNamespace(current_dt=datetime.datetime(2020,1,2,14,minute)),{})
    assert events==['clock']


def test_original_adjustment_feedback_and_position_mapping(directory):
    cases=indexed(directory)
    assert cases['rf_adjust_plain_empty']['error']['type']=='KeyError'
    assert cases['rf_adjust_plain_empty']['orders']==[]
    assert len(cases['rf_adjust_reject']['orders'])==6
    for mode in ('partial','full'):
        row=cases['rf_adjust_'+mode]
        assert row['error'] is None
        assert row['remaining_positions']==['A','B','C','D','E']
        assert row['orders']==[[s,200.] for s in ['A','B','C','D','E']]
    assert cases['rf_adjust_mutating_sell']['error']['type']=='RuntimeError'
    assert cases['rf_adjust_mutating_sell']['orders']==[['OLD',0]]


def test_original_fund_callback_requires_minute_and_does_not_guard_flat(directory):
    cases=indexed(directory);row=cases['zhu_market_flat_before_orders']
    assert row['error']['type']=='IndexError'
    assert row['orders']==[]
    assert row['history_queries']==[['399001.XSHE',1,'1m',['close','high','low']],
        ['399001.XSHE',199,'1d',['close','high','low']]]
    init=cases['zhu_initialize_stock_cost']['events']
    assert next(e for e in init if e[0]=='cost')[2]=={'type':'stock'}
    assert ['schedule','market_open',{'time':'14:30','reference_security':'000001.XSHG'}] in init
