"""Validate original DCA monetary units, valuation boundaries and native capture scope."""
import ast

import pytest

from scripts import review_strategy_batch77 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_original72_fund_cost_and_callbacks(cases):
    row = cases['72_defaults_schedule_cost']
    assert row['fund'] == '160706.XSHE' and row['last_cash'] == 1000
    assert row['costs'] == [{'buy_cost': .0015, 'sell_cost': .005}]
    assert [s[0] for s in row['schedules']] == ['fnud_main', 'fnud_close', 'fund_sell']


@pytest.mark.parametrize('q,value', [(.3999,1500),(.4,1000),(.6,500),(.8,-999000)])
def test_monthly_original72_budget_and_strict_boundaries(cases,q,value):
    row = cases['72_month_'+str(q)]
    assert row['orders'] == [['value','160706.XSHE',value]]
    assert row['last_cash'] == 1010 and row['total_month'] == 1


def test_original72_repeated_callback_and_sell_gate(cases):
    assert cases['72_repeat_month_increases_budget']['last_cash'] == 1020.1
    assert cases['72_sell_0.75_100.0_0.9']['queried'] == 0
    assert cases['72_sell_0.76_0.0_0.9']['queried'] == 0
    assert cases['72_sell_0.76_100.0_0.79']['orders'] == []
    assert cases['72_sell_0.76_100.0_0.8']['orders'] == [['target','160706.XSHE',0]]


def test_original_harmonic_PE_counts_negative_and_missing(cases):
    assert cases['72_harmonic_positive']['value'] == pytest.approx(40/3)
    assert cases['72_harmonic_negative_and_nan']['value'] == 30
    assert cases['72_harmonic_empty']['value'] is None
    assert cases['72_no_positive_PE_division']['exception'] == 'ZeroDivisionError'


def test_original_legacy_compatibility_and_bisect(cases):
    assert cases['72_ix_main_error']['exception'] == 'AttributeError'
    assert cases['72_month_alias_error']['exception'] == 'ValueError'
    assert cases['72_max_decile_bisect_error']['exception'] == cases['72_constant_decile_bisect_error']['exception'] == 'IndexError'
    assert cases['72_decile_5.0']['percentile'] == 40
    assert cases['72_decile_0.0']['percentile'] == 1


def test_original_cash_log_is_not_contribution_measure(cases):
    row = cases['72_order_cash_is_not_external_flow']
    assert row['total_cash'] == 500 and row['max_cash'] == 2000 and row['external_flows'] == 0


def test_original99_defaults_actual_mapping(cases):
    row = cases['99_defaults_mapping_cost_schedule']
    assert row['fund'] == '159919.XSHE' and row['fallback'] == '511010.XSHG'
    assert row['mapping']['000016.XSHG'] == '510800.XSHG'
    assert row['minimum'] == 10000 and row['maximum'] == 100000 and row['years'] == 7
    assert [s[1]['time'] for s in row['schedules']] == ['7:00','7:30','09:30']


@pytest.mark.parametrize('attr',['Take_Risk','Have_Work','Need_Pay','How_Long','Base_Money_Min','Base_Money_Max'])
def test_original_risk_failures_stop(cases,attr):
    assert not cases['99_risk_'+attr]['accepted']


def test_original99_integer_budget_and_cap(cases):
    assert cases['99_budget_0.1']['result'] == [1,40000,0]
    assert cases['99_budget_0.45']['result'] == [0,10000,0]
    assert cases['99_budget_0.6']['result'] == [0,10000,0]
    assert cases['99_budget_0.9']['result'] == [-1,10000,160000]
    assert cases['99_budget_cap']['result'] == [1,15000,0]


def test_quantity_and_wrong_sale_tuple_are_preserved(cases):
    row = cases['99_month_0.9']
    assert row['flows'] == [[10000,{'pindex':0}]]
    assert row['orders'] == [['quantity','511010.XSHG',10000],['quantity','159919.XSHE',-10000]]
    assert cases['99_month_0.1']['orders'] == [['quantity','159919.XSHE',40000]]
    assert cases['99_month_0.99']['orders'][-1] == ['target_value','159919.XSHE',0]


def test_real_index_shapes_are_not_repaired(cases):
    assert cases['99_date_label_minus_one_error']['exception'] == 'KeyError'
    assert cases['99_positions_total_amount_error']['exception'] == 'AttributeError'
    assert cases['99_clear_tuple_with_attribute_stub']['result'] == [2,10000,123]
    assert cases['99_increment_single_trade_day_error']['exception'] == 'IndexError'


def test_independent_strict_quartile_trim_and_negative_values(cases):
    assert cases['99_trim_strict_middle']['pe'] == cases['99_trim_strict_middle']['pb'] == 2.5
    assert cases['99_trim_negative_kept']['pe'] == -3
    assert cases['99_trim_constant_empty_trim']['pe'] is cases['99_trim_constant_empty_trim']['pb'] is None


def test_average_rank_and_original_whole_row_dropna(cases):
    assert cases['99_rank_all_tied']['rows'] == 1
    assert cases['99_rank_all_tied']['quantile'] == 1709/3416
    assert cases['99_rank_missing_PB_at_end']['rows'] == cases['99_rank_short']['rows'] == 0
    assert cases['99_inform_local_stub_only']['external_messages'] == 0


def test_frozen_sources_keep_header_and_ast_lines():
    batch.validate_sources(batch.DIRECTORY)
    rows = batch.read(batch.DIRECTORY/'source-reviews/review.json')['sources']
    assert [r['complete_lines_read'] for r in rows] == [145,282]
    tree = batch.source_tree(batch.read_source(batch.DIRECTORY/'source-reviews'/(batch.strategy_id(batch.SOURCES[0])+'.source'))[0],0)
    assert next(n for n in tree.body if isinstance(n,ast.FunctionDef)).lineno == 15


@pytest.mark.parametrize('body',['@boom()\ndef f(): pass','def f(x=boom()): pass','def f(x: boom()): pass'])
def test_source_selector_rejects_eager_code(body):
    assert not batch.safe_function(ast.parse(body).body[0])


def test_request_boundaries_redact_generated_tokens():
    kwargs = {'params':{'token':'a'*32,'indexCode':'000300.SH'}}
    url = 'https://legulegu.com/api/stockdata/index-basic-pe'
    assert batch.expected_request('index-PE',1,'GET',url,kwargs)['parameters']['token'] == '<redacted-generated>'
    with pytest.raises(ValueError): batch.expected_request('index-PE',0,'GET',url,kwargs)
    with pytest.raises(ValueError): batch.expected_request('index-PE',2,'GET',url,kwargs)
    with pytest.raises(ValueError): batch.expected_request('index-PE',1,'POST',url,kwargs)


def test_missing_supplier_pair_does_not_invent_index_PE(tmp_path):
    batch.save(tmp_path/'probe-results.json',{'results':[{'case':'index-PE','status':'failed'},{'case':'index-PB','status':'failed'}]})
    frame,summary = batch.components(tmp_path)
    assert frame.empty and summary['original72_historical_windows'] == summary['supplier_rank_windows'] == 0
    assert summary['not_a_backtest'] and summary['new_trading_backtests'] == 0


def test_study_repeat_refused_before_writes(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused',tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


def test_query_scope_keeps_three_actual_funds():
    assert set(batch.QUERIES) == {'fund-160706','fund-159919','fund-511010','index-PE','index-PB'}
    assert batch.QUERIES['fund-160706']['parameters']['symbol'] == 'sz160706'
