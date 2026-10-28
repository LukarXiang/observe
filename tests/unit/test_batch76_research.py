"""Preserve original dynamic-reference, order and compatibility boundaries."""
import ast
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch76 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']:r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_eight_actual_pairs_and_two_intraday_schedules(cases):
    row=cases['28_actual_pool_schedules_cost']
    assert row['pool']==batch.PAIRS
    assert row['lags']==[6,7,13,13]
    assert [s[1]['time'] for s in row['schedules']]==['11:30','14:40']
    assert row['costs']==[{'buy_cost':.0005,'sell_cost':.0005,'min_cost':5}] and row['slips']==[0]


def test_third_index_and_unreachable_three_selection(cases):
    row=cases['28_dynamic_third_and_actual_two']
    assert row['reference']=='000010.XSHG'
    assert row['signal']=='BUY' and row['buy']==list(batch.PAIRS.values())[:2]
    assert cases['28_repeated_signal_accumulates_buy']['buy']==row['buy']*2
    assert cases['28_one_eligible']['buy']==['150019.XSHE']


def test_sell_and_cash_branches_preserve_state(cases):
    assert cases['28_cash_KEEP']['signal']=='KEEP'
    assert cases['28_holding_sell']['signal']==cases['28_emotion_sell']['signal']=='sell_the_stocks'
    assert cases['28_holding_sell']['last']==cases['28_emotion_sell']['buy']==[]
    assert cases['28_empty_KEEP_buys_fallback']['orders']==[['target','511880.XSHG',1234.]]


@pytest.mark.parametrize('run,signal',[(2,0),(3,1),(5,0),(6,-1)])
def test_original_emotion_run_thresholds(cases,run,signal):
    assert cases['28_emotion_run_'+str(run)]['signal']==signal


def test_same_sign_emotion_returns_none(cases):
    assert cases['28_emotion_no_cross_returns_None']['signal'] is None
    assert cases['28_emotion_no_cross_returns_None']['rate']==0
    assert cases['28_positive_run_without_negative_returns_None']['signal'] is None


def test_trim_strict_threshold_and_nested_duplicate_orders(cases):
    assert cases['28_trim_7500.0']['orders']==[]
    orders=cases['28_trim_7501.0']['orders']
    assert len(orders)==4
    assert orders==[['target',s,5000.] for s in list(batch.PAIRS.values())[:2]]*2


def test_original_incremental_cash_and_failure_cases(cases):
    assert cases['28_incremental_targets_ignore_available_cash']['orders']==[['value',s,5000.] for s in list(batch.PAIRS.values())[:2]]+[ ['target','511880.XSHG',1234.] ]
    assert cases['28_empty_BUY_division']['exception']=='ZeroDivisionError'
    assert cases['28_immediate_delete_during_sell']['exception']=='RuntimeError'


def test_52_actual_fallback_and_endpoint_formula(cases):
    row=cases['52_actual_parameters']
    assert row['fallback']=='150051.XSHE' and row['cycle']==20
    assert row['schedules'][0][1]['time']=='14:53'
    assert cases['52_power_20_endpoints']['power']==.19
    assert cases['52_both_nonpositive_fallback']['signal']=='150051.XSHE'
    assert cases['52_negative_held_clear_first']['signal']=='clear'


@pytest.mark.parametrize('date,buy,sell',[('2009-01-01',.003,.004),('2009-01-02',.002,.003),
    ('2011-01-01',.002,.003),('2011-01-02',.001,.002),('2013-01-01',.001,.002),('2013-01-02',.0003,.0013)])
def test_strict_historical_fee_dates(cases,date,buy,sell):
    assert cases['52_cost_'+date]['cost']==[{'buy_cost':buy,'sell_cost':sell,'min_cost':5}]


def test_52_compatibility_and_same_day_order_semantics(cases):
    assert cases['52_positive_held_sort_API_error']['exception']==cases['52_positive_cash_sort_API_error']['exception']=='TypeError'
    assert cases['52_label_minus_one_error']['exception']=='KeyError'
    assert cases['52_clear_no_same_day_winner']['orders']==[['target',s,0] for s in ('510300.XSHG','510500.XSHG')]
    expected=[['target','150051.XSHE',0],['value','510500.XSHG',1234.]]
    assert cases['52_sell_then_incremental_existing_cash']['orders']==cases['52_repeated_incremental_buy']['orders']==expected
    assert cases['52_immediate_delete_during_clear']['exception']=='RuntimeError'


def test_full_source_validation():
    batch.validate_sources(batch.DIRECTORY)
    rows=batch.read(batch.DIRECTORY/'source-reviews/review.json')['sources']
    assert [r['complete_lines_read'] for r in rows]==[237,97]


@pytest.mark.parametrize('body',['@boom()\ndef f(): pass','def f(x=boom()): pass','def f(x: boom()): pass'])
def test_selector_rejects_eager_definition_code(body):
    assert not batch.safe_function(ast.parse(body).body[0])


@pytest.mark.parametrize('name',['diagnostics.json','fund-history-components.parquet','index-components.parquet','paired-power-components.parquet','component-research.json'])
def test_study_repeat_is_blocked_before_any_write(tmp_path,name):
    path=tmp_path/name;path.write_bytes(b'original')
    with pytest.raises(ValueError):batch.study('unused',tmp_path)
    assert path.read_bytes()==b'original' and list(tmp_path.iterdir())==[path]


def test_hand_computed_independent_emotion():
    assert batch.independent_emotion([100.]*100) is None
    assert batch.independent_emotion([100.]*94+[50.]*6)==-1
    assert batch.independent_emotion([100.]*97+[200.]*3) is None
    assert batch.independent_emotion([100.]*96+[50.]+[200.]*3)==1


def test_components_never_replace_intraday_prices_and_exclude_future(tmp_path):
    shutil.copytree(batch.DIRECTORY/'source-reviews',tmp_path/'source-reviews')
    path=tmp_path/'fund.parquet';frame=pd.DataFrame({'date':pd.date_range('2026-09-01',periods=35),'close':np.arange(35)+100.})
    frame.to_parquet(path,index=False)
    batch.save(tmp_path/'reused-inputs.json',{'inputs':{'fund-150019':{'file':str(path),'sha256':batch.file_sha(path)}}})
    batch.save(tmp_path/'probe-results.json',{'results':[]})
    frames,summary=batch.components(tmp_path)
    assert len(frames[0])==17 and frames[1].empty and frames[2].empty
    assert frames[0].history_end.max()=='2026-09-29' and not frames[0].intraday_current_price_available.any()
    assert summary['inputs'][0]['post_cutoff_rows']==6
    assert summary['not_a_backtest'] and summary['new_trading_backtests']==0


def test_query_scope_is_missing_inputs_only():
    assert len(batch.QUERIES)==13
    assert not {'fund-159915','fund-510300','fund-510500','fund-510050','index-sh000016','index-sz399006'} & set(batch.QUERIES)


def test_candidate_audit_uses_batch76_binding_without_batch74_primary(tmp_path):
    shutil.copytree(batch.DIRECTORY/'source-reviews',tmp_path/'source-reviews')
    shutil.copyfile(batch.DIRECTORY/'input-binding.json',tmp_path/'input-binding.json')
    shutil.copyfile(batch.DIRECTORY/'existing-apis.json',tmp_path/'existing-apis.json')
    probes=tmp_path/'probes';probes.mkdir()
    with batch.patch.object(batch.native,'QUERIES',batch.QUERIES):
        batch.save(probes/'existing-apis.json',batch.native.api_evidence())
    rows=[{'case':case,'query':query,'status':'failed','error':'synthetic no response',
        'attempts':[],'wire_responses':[],'api_sha256':batch.file_sha(probes/'existing-apis.json')}
        for case,query in batch.QUERIES.items()]
    result={'results':rows,'new_original_input_rows':0,'new_trading_backtests':0}
    batch.save(probes/'results.json',result);batch.save(tmp_path/'probe-results.json',result)
    summary=batch.candidate_audit(tmp_path)
    assert len(summary['profiles'])==13
    assert all(p['error']=='synthetic no response' for p in summary['profiles'])
    assert summary['original_input_rows_admitted']==0
    assert not (probes/'input-binding.json').exists()
