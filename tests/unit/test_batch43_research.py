from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch43 as batch


@pytest.fixture
def sources(tmp_path):
    rows=[]
    for name in batch.SOURCES:
        path=Path('repo/量化策略源代码')/name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy':str(path),'source_sha256':file_sha(path)})
    write_json(tmp_path/'source-reviews/review.json',{'sources':rows})
    return tmp_path


def cases(directory):
    with redirect_stdout(io.StringIO()): result=batch.diagnostics(directory)
    assert json.loads(json.dumps(result,allow_nan=False))==result
    return {r['case']:r for r in result['cases']}


def test_candle_original_queue_and_time_defects(sources):
    r=cases(sources)
    assert r['candle_rejected_buy_removes_and_skips_with_stale_cash']['orders']==[['A',10000.],['C',10000.]]
    assert r['candle_rejected_buy_removes_and_skips_with_stale_cash']['remaining']==['B']
    assert r['candle_tail_clear_skips_same_list']['remaining']==['B','D']
    assert r['candle1440_else_calls_ticks_with_zero_sellable']['requests']==['ticks']
    assert r['candle_after1440_checks_sellable']['requests']==[]
    assert r['candle_before_open_appends_duplicate_queue']['queue']==['A','A','A']
    assert r['candle_listing_strict1080']['accepted']==['older']
    assert r['candle_standard_frame_not_legacy_panel']['error'].startswith('KeyError:')


def test_weekly_attempts_cash_and_actual_cap(sources):
    r=cases(sources)
    assert r['weekly_five_new_plus_old_not_total_cap']['selected']==['S0','S1','S2','S3','S4']
    assert r['weekly_five_new_plus_old_not_total_cap']['orders']==[[f'S{k}',2000.] for k in range(5)]
    assert r['weekly_failed_attempt_blocks_later_buy']['attempted']==['fail']
    assert r['weekly_failed_attempt_blocks_later_buy']['orders']==[]
    assert r['weekly_tuesday_does_not_reset_attempts']['attempted']==['old']
    assert r['weekly_monday_resets_attempts']['attempted']==[]


def test_weekly_original_current_bar_and_linear_weights(sources):
    r=cases(sources)
    request=r['weekly_original_current_bar_and_multiindex']['requests'][0]
    assert request['kwargs']['include_now'] is True and request['kwargs']['unit']=='1w'
    assert request['kwargs']['count']==48 and request['kwargs']['df'] is True
    assert r['weekly_original_current_bar_and_multiindex']['columns']==['code','level_1','close']
    assert r['weekly_original_linear_weight_not_ema']['values']==pytest.approx([107+2/3,106.])
    assert r['weekly_zero_range_no_sell']['result'] is None


def test_margin_original_gaps_repeated_additive_orders_and_two_calls(sources):
    r=cases(sources)
    for z in (-2.1,-2.,2.,2.1): assert r[f'margin_signal_{z}']['result'] is None
    for z in (-1.95,0.,1.95): assert r[f'margin_signal_{z}']['result']=='mid'
    assert r['margin_signal_-2.10000001']['result']=='buy2'
    assert r['margin_signal_2.10000001']['result']=='buy1'
    orders=r['margin_repeated_buy1_additive_intents']['orders']
    assert len(orders)==4 and orders[:2]==orders[2:]
    assert orders[0]['kind']=='marginsec_open' and orders[0]['args']==['510500.XSHG',100000]
    assert r['margin_fixed_share_close_and_unchecked_state']['state']=='even'
    assert r['margin_buy2_uses_asset_value_as_share_count']['orders'][0]['args']==['510220.XSHG',200000.]
    assert r['margin_handle_calls_twice_second_drives_orders']['signals']==['buy1','buy2']
    assert r['margin_handle_calls_twice_second_drives_orders']['orders'][0]['args'][0]=='510220.XSHG'


def test_margin_population_std_array_and_platform_injection(sources):
    r=cases(sources); row=r['margin_synthetic120_population_std_shape']
    assert row['shape']==[1] and row['value']==pytest.approx(row['reference'],abs=1e-12)
    assert r['margin_mean_requires_platform_injection']['error'].startswith('NameError:')


def test_original_candle_inclusive_green_and_six_row_valley(sources):
    code,_=batch.candle_kernel(sources)
    recent=pd.DataFrame({'open':[100.]*6,'close':[99.,99.,99.,100.,101.,102.]})
    past=pd.DataFrame({'high':[210.]*30,'low':[100.]*30,'close':[120.]*30})
    r=batch.original_candle(recent,past,code)
    assert r[:2]==[3,2]
    assert r[2:]==pytest.approx([1.1,(120.-99.)/99.])
    with pytest.raises(ValueError,match='Incomplete candle'):
        batch.original_candle(recent.iloc[:5],past,code)


def test_weekly48_ratios_keep_original_strict_gate(sources):
    sample=pd.DataFrame({'close_adj':[100.]*47+[110.],'open_adj':[100.]*48,
        'high_adj':[105.]*48,'low_adj':[90.]*48,'volume':[100.]*47+[150.]})
    r,flag=batch.original_ratios(sample,batch.ratio_kernel(sources))
    expected,ref_flag=batch.reference_ratios(sample)
    assert r==pytest.approx(expected) and flag is ref_flag is True
    sample.loc[47,'volume']=200.
    assert batch.original_ratios(sample,batch.ratio_kernel(sources))[1] is False


def calendar_and_prices():
    dates=pd.date_range('2021-01-01','2021-01-29')
    calendar=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'is_open':dates.dayofweek<5})
    opened=calendar[calendar.is_open].date.tolist()
    price=pd.DataFrame({'instrument':['A']*len(opened),'date':opened,'is_trading':[True]*len(opened),
        'open_adj':np.arange(len(opened))+9.,'high_adj':np.arange(len(opened))+11.,
        'low_adj':np.arange(len(opened))+8.,'close_adj':np.arange(len(opened))+10.,
        'volume':np.ones(len(opened))*100.,'back_factor':np.ones(len(opened))})
    return calendar,price


def test_completed_week_calendar_and_suspension_aggregation():
    calendar,price=calendar_and_prices(); price.loc[3,'is_trading']=False
    frame,proof=batch.completed_weeks(price,calendar)
    assert len(frame)==4
    assert frame.iloc[0][['open_adj','high_adj','low_adj','close_adj','volume']].tolist()==[10.,16.,9.,15.,400.]
    assert frame.iloc[0].source_daily_rows==5 and frame.iloc[0].traded_rows==4
    assert proof['max_aggregation_differences']==[0.]*5
    assert len(proof['excluded'])==1
    with pytest.raises(ValueError,match='Unknown missing daily'):
        batch.completed_weeks(price.drop(index=3),calendar)


def test_partial_boundary_weeks_never_become_operands():
    calendar,price=calendar_and_prices()
    calendar=calendar[calendar.date.between('2021-01-06','2021-01-27')]
    price=price[price.date.between('2021-01-06','2021-01-27')]
    frame,proof=batch.completed_weeks(price,calendar)
    assert frame.date.tolist()==['2021-01-15','2021-01-22']
    assert len(proof['excluded'])==2


def test_source_corruption_blocks_original_arithmetic(sources):
    doc=batch.read(sources/'source-reviews/review.json'); doc['sources'][0]['source_sha256']='0'*64
    write_json(sources/'source-reviews/review.json',doc)
    with pytest.raises(ValueError,match='Selected source changed'): batch.candle_kernel(sources)
    with pytest.raises(ValueError,match='Selected source changed'): batch.diagnostics(sources)


def test_probe_sample_unproven_and_new_namespace(tmp_path,monkeypatch):
    import akshare as ak
    api={'apis':[]}; calls=[]; write_json(tmp_path/'existing-apis.json',api)
    monkeypatch.setattr(batch,'api_evidence',lambda:api)
    monkeypatch.setattr(ak,'stock_margin_detail_sse',lambda **kw:pd.DataFrame({'shares':[1.]}))
    def save(root,upstream,endpoint,key,frame):
        calls.append([upstream,endpoint,key]); path=tmp_path/'raw.parquet'; frame.to_parquet(path,index=False); return path
    monkeypatch.setattr(batch.raw,'save',save)
    row=batch.worker('diagnostic-root',tmp_path,'margin_detail')
    assert row['status']=='sample' and row['strict_usable'] is False and row['published'] is False
    assert calls==[['batch43_dependency_probe','margin_detail',tmp_path.name]]


@pytest.mark.parametrize('change',['query','sha','publication','failed_data','status'])
def test_probe_substitution_rejected(tmp_path,monkeypatch,change):
    api={'apis':[]}; write_json(tmp_path/'existing-apis.json',api); monkeypatch.setattr(batch,'api_evidence',lambda:api)
    for endpoint,query in batch.QUERIES.items():
        row={'endpoint':endpoint,'query':query,'api_sha256':file_sha(tmp_path/'existing-apis.json'),
            'status':'failed','published':False,'files':[],'wire':[],'error':'synthetic failure'}
        if change=='query':row['query']={}
        if change=='sha':row['api_sha256']='0'*64
        if change=='publication':row['published']=True
        if change=='failed_data':row['files']=[{'file':'unaccepted','sha256':'0'*64}]
        if change=='status':row['status']='accepted'
        write_json(tmp_path/f'probe-{endpoint}.json',row)
    with pytest.raises((ValueError,FileNotFoundError)):batch.validate_probes(tmp_path)
