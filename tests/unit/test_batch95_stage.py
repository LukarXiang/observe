"""Original SHA-bound callbacks: index/namespace boundaries and effects before exceptions."""
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import review_strategy_batch95 as stage


@pytest.fixture
def sources(tmp_path):
    folder=tmp_path/'sources'; folder.mkdir()
    for n,name in enumerate(stage.SOURCES):
        path=Path('repo/量化策略源代码')/name
        if not path.exists(): path=stage.DIRECTORY/'sources'/f'{n}.source'
        shutil.copyfile(path,folder/f'{n}.source')
    return tmp_path


@pytest.mark.parametrize('number',[0,1,2])
def test_initialize_preserves_pool_cost_seed_schedule(sources,number):
    row=stage.diagnostic_case(sources,number,'initialize')
    assert row['error'] is None
    assert row['result']['pool']==list(stage.POOL[number])
    assert row['result']['seed_length']==(600 if number==2 else 599)
    costs=[e[1] for e in row['events'] if e[0]=='cost']
    assert costs[0]['min_commission']==(0 if number==2 else 5)
    schedules=[e[1:] for e in row['events'] if e[0]=='schedule']
    assert [e[0] for e in schedules]==(['my_trade','check_lose','hold_check'] if number==2 else ['my_trade','check_lose','print_trade_info'])
    assert 'check_buy_back' not in stage.literal(sources,number,{})


@pytest.mark.parametrize('number',[0,1,2])
def test_date_label_rank_fails_before_timing_and_sell(sources,number):
    row=stage.diagnostic_case(sources,number,'rank_error_before_SELL')
    assert row['error']['type']=='KeyError' and not row['orders']
    assert ['timing'] not in row['events'] and len(row['held'])==1


def test_v2_range_index_also_fails_after_tail_slice(sources):
    row=stage.diagnostic_case(sources,2,'rank_range')
    assert row['error']['type']=='KeyError'
    controlled=stage.diagnostic_case(sources,2,'rank_controlled_positional')
    assert controlled['error'] is None and len(controlled['result'])==2


@pytest.mark.parametrize('number',[0,1,2])
def test_tie_is_stable_original_pool_order(sources,number):
    row=stage.diagnostic_case(sources,number,'rank_equal_scores')
    assert row['error'] is None
    assert row['result'][0]==stage.POOL[number][0]


def test_mean_namespace_failure_appends_before_error(sources):
    before=stage.diagnostic_case(sources,2,'mean_11')
    missing=stage.diagnostic_case(sources,2,'mean_12_missing')
    injected=stage.diagnostic_case(sources,2,'mean_12_injected')
    assert before['result']==0 and before['score_history_length']==11
    assert missing['error']['type']=='NameError' and missing['score_history_length']==12
    assert injected['error'] is None and injected['result']==pytest.approx(66/12)


@pytest.mark.parametrize('number',[0,1,2])
def test_rejected_first_buy_exposes_score_as_security(sources,number):
    row=stage.diagnostic_case(sources,number,'BUY_reject_numeric')
    assert row['error'] is None
    assert [o['security'] for o in row['orders']]==[stage.POOL[number][0],.123]
    full=stage.diagnostic_case(sources,number,'BUY_full_masks_numeric')
    assert len(full['orders'])==1 and len(full['held'])==1
    partial=stage.diagnostic_case(sources,number,'BUY_partial_masks_numeric')
    assert len(partial['orders'])==1 and len(partial['held'])==1


@pytest.mark.parametrize('number',[0,1,2])
def test_plain_positions_and_mutation_are_not_silently_fixed(sources,number):
    plain=stage.diagnostic_case(sources,number,'BUY_plain_empty')
    assert plain['error']['type']=='KeyError' and not plain['orders']
    selling=stage.diagnostic_case(sources,number,'SELL_mutating')
    assert selling['error']['type']=='RuntimeError' and len(selling['orders'])==1 and not selling['held']
    rotating=stage.diagnostic_case(sources,number,'KEEP_rotates')
    assert rotating['error']['type']=='RuntimeError' and rotating['orders'][0]['value']==0


@pytest.mark.parametrize('case',['partial_sell_blocks_rotation','deferred_sell_blocks_rotation'])
def test_unsold_holding_blocks_rotation(sources,case):
    row=stage.diagnostic_case(sources,0,case)
    assert row['error'] is None and len(row['orders'])==1 and row['orders'][0]['value']==0
    assert len(row['held'])==1


def test_stop_v2_submits_then_print_fails(sources):
    row=stage.diagnostic_case(sources,2,'stop_exact_minus90')
    assert row['error']['type']=='NameError' and len(row['orders'])==1 and not row['held']
    magic=stage.diagnostic_case(sources,0,'stop_exact_minus90')
    assert magic['error'] is None and len(magic['orders'])==1
    above=stage.diagnostic_case(sources,2,'stop_above_minus90')
    assert above['error'] is None and not above['orders']


@pytest.mark.parametrize('case',['hold_datetime','hold_range'])
def test_hold_last_label_is_not_position(sources,case):
    row=stage.diagnostic_case(sources,2,case)
    assert row['error']['type']=='KeyError' and not row['orders'] and row['hold_state']==''


def test_hold_controls_show_rejection_state_and_mutation_boundary(sources):
    rejected=stage.diagnostic_case(sources,2,'hold_below_injected_label')
    assert rejected['error'] is None and len(rejected['orders'])==1
    assert rejected['held']==[stage.POOL[2][0]] and rejected['hold_state']==stage.POOL[2][0]
    equal=stage.diagnostic_case(sources,2,'hold_equal_injected_label')
    assert equal['error'] is None and not equal['orders'] and equal['hold_state']==''
    wrong=stage.diagnostic_case(sources,2,'hold_wrong_time')
    assert wrong['error'] is None and not wrong['events']
    mutation=stage.diagnostic_case(sources,2,'hold_two_mutating')
    assert mutation['error']['type']=='RuntimeError' and len(mutation['orders'])==1
    assert mutation['hold_state']==stage.POOL[2][0] and len(mutation['held'])==1


@pytest.mark.parametrize('number',[0,2])
def test_literal_thresholds(sources,number):
    expected=({'score_minus07':'KEEP','score_above_minus07':'KEEP','score_plus07':'KEEP',
        'score_above_plus07':'BUY','score_below_minus07':'SELL','score_zero':'KEEP'} if number==0 else
        {'score_minus07':'SELL','score_above_minus07':'BUY','score_plus07':'BUY',
         'score_above_plus07':'BUY','score_below_minus07':'SELL','score_zero':'BUY',
         'positive_below_mean':'SELL','positive_equal_mean':'BUY'})
    for case,value in expected.items():
        row=stage.diagnostic_case(sources,number,case)
        assert row['error'] is None and row['result']==value


def test_independent_scalar_ols_and_z():
    intercept,slope,r2=stage.scalar_ols([0,1,2,3],[3,5,7,9])
    assert intercept==3 and slope==2 and r2==1
    assert stage.scalar_z([0.,1.,2.])==pytest.approx(np.sqrt(1.5))


def test_literal_rejects_changed_source(sources):
    path=sources/'sources/0.source'; path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='source identity'): stage.literal(sources,0,{})


def test_diagnostics_nonempty_and_duplicate_equal(sources):
    doc=stage.diagnostics(sources)
    assert doc['count']==len(stage.CASES)*3-18
    a=[{k:v for k,v in r.items() if k!='source'} for r in doc['cases'] if r['source']==0]
    b=[{k:v for k,v in r.items() if k!='source'} for r in doc['cases'] if r['source']==1]
    assert a==b and not doc['new_trading_backtests']
