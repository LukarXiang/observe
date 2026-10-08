from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha,write_json
from scripts import review_strategy_batch36 as batch


@pytest.fixture
def sources(tmp_path):
    rows=[]
    for name in batch.SOURCES:
        path=Path('repo/量化策略源代码')/name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy':str(path),'source_sha256':file_sha(path)})
    write_json(tmp_path/'source-reviews/review.json',{'sources':rows})
    return tmp_path


def test_accepted_directory_matches_receipt():
    if not batch.RECEIPT.is_file(): pytest.skip('Restore accepted receipt')
    receipt=batch.read(batch.RECEIPT)
    assert receipt['status']=='ok' and batch.ACCEPTED==Path(receipt['checks']['commands'][0]['log']).parent


def test_configuration_retains_static_180_pool_schedule_and_unused_zero_slippage(sources):
    g,calls=batch.configuration(sources)
    assert (g.tc,g.num_stocks,g.index,g.t,g.if_trade)==(7,5,'000010.XSHG',0,False)
    assert [c for c in calls if c[0]=='pool']==[['pool','000010.XSHG']]
    assert not any(c[0]=='slippage' for c in calls)
    assert [c[2]['time'] for c in calls if c[0]=='schedule']==['before_open','open','after_close']


def test_lr_uses24_strict_means_and240_warmup(sources):
    ns={'np':np,'pd':pd}; batch.selected(sources,1,('cala_LR',),ns)
    constant=pd.Series(np.ones(260),index=pd.date_range('2020-01-01',periods=260))
    result=ns['cala_LR'](constant)
    assert len(result)==21 and result.eq(0).all() and result.index[0]==constant.index[239]
    rising=pd.Series(np.arange(1.,261.),index=constant.index)
    assert ns['cala_LR'](rising).eq(1).all()
    assert batch.reference_lr(rising)==[1.]*21


@pytest.mark.parametrize('values',[[],[1.]*239,[1.]*239+[0.],[1.]*239+[np.nan]])
def test_invalid_lr_inputs_rejected(values):
    with pytest.raises(ValueError,match='Invalid LR'): batch.reference_lr(values)


def test_reverse_dampening_aligns_dates_and_preserves_endpoint_exponents(sources):
    cls,_=batch.original_class(sources,'RSRS_improve1')
    data=pd.DataFrame({'RSRS_z':[2.,-2.],'R_2':[.5,.5]},index=['a','b'])
    rank=pd.Series([1.,0.],index=['b','a'])
    assert cls.cala_passivation_RSRS(data.copy(),rank)['RSRS_passivation'].tolist()==[.5,-2.]
    partial=cls.cala_passivation_RSRS(data.copy(),rank.drop('a'))
    assert pd.isna(partial.loc['a','RSRS_passivation'])


@pytest.mark.parametrize('period',batch.PERIODS)
def test_original_multihorizon_mean_ratio_and_query_parameters(sources,period):
    frame=pd.DataFrame({'close':np.arange(1.,301.),'date':pd.date_range('2020-01-01',periods=300).strftime('%Y-%m-%d')})
    ns,_,calls,_=batch.multihorizon_namespace(sources,frame)
    result=ns['calAt']('index_operator_sample',frame.date.iloc[-1],period)
    assert result==pytest.approx((301.-period/2-.5)/300)
    assert calls[-1]['fq']=='pre' and calls[-1]['skip_paused'] is True and calls[-1]['count']==period


def test_diagnostics_preserve_real_faults_original_labels_and_breadth_denominators(sources):
    diagnostic=batch.diagnostics(sources); assert json.loads(json.dumps(diagnostic))==diagnostic
    cases={r['case']:r for r in diagnostic['cases']}
    assert cases['reversed_open_labels']['labels']==[-.5]*25
    requests=cases['reversed_open_labels']['requests']
    assert requests[0]['end_date']=='2020-01-08T00:00:00' and requests[1]['end_date']=='2020-01-01T00:00:00'
    assert all(r['fields']==['close','open'] and r['count']==5 and r['skip_paused'] for r in requests)
    assert cases['feature_natural_week_lag']['requests'][0]['date']=='2020-01-01T00:00:00'
    assert cases['seven_callback_schedule_and_equal_orders']['days']==[0,7,14]
    assert cases['seven_callback_schedule_and_equal_orders']['orders'][:6]==[['old',0],['a',2000.],['b',2000.],['c',2000.],['d',2000.],['e',2000.]]
    assert cases['flag_strict_threshold_nan_hold']['flags']==[0.,1.,1.,1.,0.,0.]
    assert cases['industry_missing_excluded']['stocks']==['a','c']
    assert cases['breadth_zero_and_nan']['industry']==33. and cases['breadth_zero_and_nan']['index_all']==25.
    assert cases['breadth_half_percent_rounding']['industry']==cases['breadth_half_percent_rounding']['index_all']==0.
    call=cases['original_BIAS_call_only']['calls'][0]
    assert call['kwargs']=={'check_date':'2020-01-01','N1':20,'N2':60,'N3':120,'include_now':True,'fq_ref_date':None}


def test_source_corruption_blocks_classes_and_fragments(sources):
    doc=batch.read(sources/'source-reviews/review.json'); doc['sources'][1]['source_sha256']='0'*64
    write_json(sources/'source-reviews/review.json',doc)
    with pytest.raises(ValueError,match='Selected source changed'): batch.original_class(sources,'RSRS_improve1')


def test_input_corruption_rejected_before_read(tmp_path):
    write_json(tmp_path/'input-analysis.json',{'snapshot':batch.SNAPSHOT,'not_a_backtest':True,'platform_equivalent':False,
        'component_instrument':'000300.SH','profiles':{n:{'file':str(tmp_path/n),'sha256':'0'*64} for n in ('index-input.parquet','calendar-input.parquet')}})
    (tmp_path/'index-input.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Input binding changed'): batch.inputs(tmp_path)


def test_successful_probe_uses_batch36_namespace_without_publication(tmp_path,monkeypatch):
    import akshare as ak
    from observe.data import raw
    api={'apis':[]}; calls=[]; write_json(tmp_path/'existing-apis.json',api); monkeypatch.setattr(batch,'api_evidence',lambda:api)
    monkeypatch.setattr(ak,'index_stock_cons_csindex',lambda **kw:pd.DataFrame({'code':['sample']}))
    def save(root,upstream,endpoint,key,frame):
        calls.append([upstream,endpoint,key]); path=tmp_path/'raw.parquet'; frame.to_parquet(path,index=False); return path
    monkeypatch.setattr(raw,'save',save); result=batch.worker('diagnostic-root',tmp_path,'pool180')
    assert result['status']=='sample' and result['published'] is False and result['strict_usable'] is False
    assert calls==[['batch36_dependency_probe','pool180',tmp_path.name]]


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
