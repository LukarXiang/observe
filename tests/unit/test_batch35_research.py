import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch35 as batch


@pytest.fixture
def sources(tmp_path):
    rows=[]
    for name in batch.SOURCES:
        path=Path('repo/量化策略源代码')/name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy':str(path),'source_sha256':file_sha(path)})
    write_json(tmp_path/'source-reviews/review.json',{'sources':rows})
    return tmp_path


def frame(count):
    x=np.arange(float(count)); low=100+x*.1+np.sin(x/9)
    return pd.DataFrame({'low':low,'high':low+2+np.sin(x/3), 'close':low+1,'volume':np.ones(count)})


def test_accepted_directory_matches_receipt():
    if not batch.RECEIPT.is_file(): pytest.skip('Restore accepted receipt')
    doc=batch.read(batch.RECEIPT)
    assert doc['status']=='ok' and batch.ACCEPTED==Path(doc['checks']['commands'][0]['log']).parent


def test_original_pool_schedule_and_cost_settings(sources):
    a,acalls=batch.configuration(sources,0); b,bcalls=batch.configuration(sources,1); c,ccalls=batch.configuration(sources,2)
    assert len(a.ETF_list)==10 and a.unit=='30m' and a.lag1==5
    assert b.stock_pool==['000300.XSHG','000905.XSHG','399006.XSHE'] and b.M==dict(zip(b.stock_pool,[700,800,500]))
    assert b.score_threshold==dict(zip(b.stock_pool,[.7,1,.4]))
    assert next(row[2] for row in bcalls if row[0]=='cost')=={'type':'stock'}
    assert c.stock_pool==['510050.XSHG','510300.XSHG','159949.XSHE','159928.XSHE'] and (c.N,c.M,c.K)==(18,600,8)
    assert [row[2]['time'] for row in ccalls if row[0]=='schedule']==['9:30','open','13:00']
    assert [row[2]['time'] for row in acalls if row[0]=='schedule']==['9:15','11:15']


def test_review_correction_hash_corruption_rejected(tmp_path):
    write_json(tmp_path/'source-reviews/review.json',{'sources':[]})
    write_json(tmp_path/'review-correction.json',{'original_sha256':'0'*64,'corrected_sha256':'0'*64})
    with pytest.raises(ValueError,match='Review correction binding changed'): batch.accepted_review(tmp_path)


def test_successful_probe_uses_batch35_raw_namespace(tmp_path,monkeypatch):
    import akshare as ak
    from observe.data import raw
    calls=[]; api={'apis':[]}; write_json(tmp_path/'existing-apis.json',api)
    monkeypatch.setattr(batch,'api_evidence',lambda:api)
    monkeypatch.setattr(ak,'fund_etf_hist_sina',lambda **kw:pd.DataFrame({'close':[1.]}))
    def save(root,upstream,endpoint,key,data):
        calls.append([upstream,endpoint,key]); path=tmp_path/'raw.parquet'; data.to_parquet(path,index=False); return path
    monkeypatch.setattr(raw,'save',save)
    result=batch.worker('diagnostic-root',tmp_path,'fundconsumer')
    assert result['status']=='sample' and result['published'] is False
    assert calls==[['batch35_dependency_probe','fundconsumer',tmp_path.name]]


def test_v21_initialization_preserves_one_position_r2_misalignment(sources):
    data=frame(626); ns,_,_,_=batch.timing_namespace(sources,2,data,625)
    slopes,scores=ns['initial_slope_series']()
    assert len(slopes)==608 and len(scores)==8
    assert slopes[-1]==pytest.approx(batch.reference_ols(data.low.iloc[607:625],data.high.iloc[607:625])[1])
    original=ns['get_ols'](data.low.iloc[600:618],data.high.iloc[600:618])[2]
    assert scores[0]==pytest.approx(ns['get_zscore'](slopes[:600])*original)
    earlier=ns['get_ols'](data.low.iloc[599:617],data.high.iloc[599:617])[2]
    assert abs(scores[0]-ns['get_zscore'](slopes[:600])*earlier)>1e-6


@pytest.mark.parametrize('score,derivative,expected',[(.1,-1.,'SELL'),(0.,-1.,'BUY'),(-.7,0.,'SELL'),(-.699999,0.,'BUY'),(.7,0.,'BUY')])
def test_v21_priority_and_strict_threshold(score,derivative,expected):
    assert batch.v21_signal(score,derivative)==expected


def test_damped_original_full_loop_matches_independent_sample_formula(sources):
    data=frame(718); ns,_,_,_=batch.timing_namespace(sources,1,data,717)
    actual=ns['get_timing_signal'](None,'000300.XSHG'); reference=batch.reference_damped(data)
    assert actual==reference['signal']
    std=data.close.pct_change().rolling(18).std().tail(700)
    assert float(std.rank(pct=True).iloc[-1])==reference['quantile']


@pytest.mark.parametrize('bad', ['length','nan','zero'])
def test_damped_invalid_input_rejected(bad):
    data=frame(718)
    if bad=='length': data=data.iloc[:-1]
    if bad=='nan': data.loc[20,'low']=np.nan
    if bad=='zero': data.loc[20,'close']=0
    with pytest.raises(ValueError,match='Invalid damped'): batch.reference_damped(data)


def test_original_errors_orders_and_record_only_messages(sources):
    diagnostic=batch.diagnostics(sources); assert json.loads(json.dumps(diagnostic))==diagnostic
    cases={r['case']:r for r in diagnostic['cases']}
    assert cases['empty_index_order']['orders']==[['target','',0],['value','000300.XSHG',1000.],['value','511880.XSHG',1000.]]
    assert cases['concatenated_holdings']['orders'][0]==['target','onetwo',0]
    assert cases['duplicate_account_stop']['orders']==[['fund',0],['fund',0]]
    assert cases['mixed_targets']['orders']==[['fund',1000.],[.001,1000.]]
    assert cases['stop_orders_before_undefined_value']['orders']==[['fund',0]]
    assert cases['hold_stop_0']['orders']==[] and len(cases['hold_stop_0']['record_only_messages'])==1
    assert cases['hold_stop_100']['orders']==[['fund',0]]
    assert cases['ipo_once_permanent']['available']==['index'] and cases['ipo_once_permanent']['pending']=={}
    assert len(cases['bbi_orders_CLEAR_2']['orders'])==1
    assert cases['bbi_orders_BUY_1']['orders']==[]
    assert cases['bbi_orders_BUY_0']['orders']==[['value','fundA',1000.]]
    assert cases['timing_0.1_-1.0']['signal']=='SELL' and cases['timing_-0.7_0.0']['signal']=='SELL'
    assert cases['timing_0.0_-1.0']['signal']=='BUY'


def test_source_corruption_prevents_any_fragment(sources):
    doc=batch.read(sources/'source-reviews/review.json'); doc['sources'][0]['source_sha256']='0'*64
    write_json(sources/'source-reviews/review.json',doc)
    with pytest.raises(ValueError,match='Selected source changed'): batch.bbi_order_fragment(sources)


def test_input_corruption_blocks_before_read(tmp_path):
    write_json(tmp_path/'input-analysis.json',{'snapshot':batch.SNAPSHOT,'not_a_backtest':True,'platform_equivalent':False,
        'component_instrument':'000300.SH','profiles':{n:{'file':str(tmp_path/n),'sha256':'0'*64} for n in ('index-input.parquet','calendar-input.parquet')}})
    (tmp_path/'index-input.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Input binding changed'): batch.inputs(tmp_path)


@pytest.mark.parametrize('change',['query','sha','publication','failed_data','status'])
def test_probe_substitution_rejected(tmp_path,monkeypatch,change):
    api={'apis':[]}; write_json(tmp_path/'existing-apis.json',api); monkeypatch.setattr(batch,'api_evidence',lambda:api)
    for endpoint,query in batch.QUERIES.items():
        row={'endpoint':endpoint,'query':query,'api_sha256':file_sha(tmp_path/'existing-apis.json'),
            'status':'failed','published':False,'files':[],'wire':[],'error':'synthetic failure'}
        if change=='query': row['query']={}
        if change=='sha': row['api_sha256']='0'*64
        if change=='publication': row['published']=True
        if change=='failed_data': row['files']=[{'file':'unaccepted','sha256':'0'*64}]
        if change=='status': row['status']='accepted'
        write_json(tmp_path/f'probe-{endpoint}.json',row)
    with pytest.raises((ValueError,FileNotFoundError)): batch.validate_probes(tmp_path)
