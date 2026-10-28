"""Verify original Python boundaries without defining missing wizard economics."""
import ast

import pytest

from scripts import review_strategy_batch85 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_parent_frozen_evidence_cannot_change(monkeypatch):
    original = batch.file_sha; target = batch.previous.DIRECTORY/'diagnostics.json'
    monkeypatch.setattr(batch, 'file_sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='Accepted84 evidence changed'): batch.binding('data')


def test_original_code_boundary_preserves_line_numbers():
    tree = batch.parsed_source('\n'*8+'from kuanke.wizard import *\ndef f():\n return 1')
    assert tree.body[0].lineno == 9 and tree.body[1].lineno == 10
    with pytest.raises(ValueError, match='boundary'): batch.parsed_source('from kuanke.wizard import *')


def test_selected_loader_does_not_evaluate_unsafe_defaults():
    with pytest.raises(ValueError, match='Unsafe'):
        batch.previous.previous.selected_nodes(ast.parse('def f(x=bad()):\n return x'), ['f'], {'bad': lambda: None})


@pytest.mark.parametrize('number,industries,filter_holded', [(0, 7, False), (1, 28, True)])
def test_full_original_defaults(cases, number, industries, filter_holded):
    row = cases[f'{number}_defaults']; state = row['state']
    assert len(state['industry_list']) == industries and state['filter_holded'] is filter_holded
    assert state['max_hold_stocknum'] == 5 and state['sell_will_buy'] is True
    assert state['order_style_str'] == 'by_cap_mean' and state['order_style_value'] == 100
    assert [e for e in row['events'] if e[0] == 'schedule'] == [
        ['schedule', 'sell_every_day', 'open'], ['schedule', 'risk_management', 'every_bar'],
        ['schedule', 'check_stocks', 'open'], ['schedule', 'trade', 'open'],
        ['schedule', 'selled_security_list_count', 'after_close']]
    assert ['slippage', .02] in row['events'] and ['option', 'order_volume_ratio', 1] in row['events']
    cost = next(e for e in row['events'] if e[0] == 'cost')
    assert cost[1] == {'open_tax': 0, 'close_tax': .001, 'open_commission': .0003, 'close_commission': .0003, 'min_commission': 5}
    assert cost[2] == {'type': 'stock'}


@pytest.mark.parametrize('number', [0, 1])
def test_holded_boolean_semantics_and_duplicates(cases, number):
    assert cases[f'{number}_holded_False']['result'] == ['new', 'new']
    assert cases[f'{number}_holded_True']['result'] == ['held', 'new', 'new']


def test_unknown_financial_helpers_only_record_actual_arguments(cases):
    first = cases['0_financial_helper_arguments']['trace']
    assert [r[2:] for r in first] == [['indicator.inc_revenue_year_on_year', 3],
        ['indicator.operation_profit_to_total_revenue', 3], ['valuation.circulating_market_cap', 3], ['valuation.capitalization', 5]]
    second = cases['1_financial_helper_arguments']['trace']
    assert [r[0] for r in second] == ['financial_data_filter_xiaoyu', 'financial_data_filter_qujian', 'financial_data_filter_qujian']
    assert [r[2:] for r in second] == [['valuation.pe_ratio_lyr', 20], ['indicator.roe', [3, 40]], ['valuation.pb_ratio', [3, 14]]]
    assert cases['0_IPO_helper_arguments']['trace'] == [['a', 600], ['b', 600]]
    assert cases['1_IPO_helper_arguments']['trace'] == []


@pytest.mark.parametrize('number', [0, 1])
def test_pool_sort_intersection_and_dormant_string_expansion(cases, number):
    row = cases[f'{number}_pool_order_and_user_string']
    assert row['pool'] == row['concepts'] == ['a', 'b'] and row['filtered'] == ['a']
    assert row['user_string_expansion'] == ['A', 'B', 'C']
    assert row['trace'] == [['index', '000300.XSHG'], ['industry', '801780'], ['industry', '801110']]
    assert cases[f'{number}_empty_sort_keeps_input']['same_object'] is True
    assert cases[f'{number}_empty_sort_keeps_input']['result'] == ['z', 'a', 'z']
    assert cases[f'{number}_inactive_nonempty_sort_compatibility']['error'] in ('TypeError', 'AttributeError')


@pytest.mark.parametrize('number', [0, 1])
def test_mutable_keys_loses_immediate_sold_history(cases, number):
    immediate = cases[f'{number}_sell_keys_immediate']
    assert immediate['positions'] == {} and immediate['sold'] == {}
    assert [r[1] for r in immediate['trace'] if r[0] == 'sell'] == ['a', 'b']
    assert cases[f'{number}_sell_keys_failed']['retry'] == ['a', 'b']
    assert cases[f'{number}_sell_keys_partial']['positions'] == {'a': 1, 'b': 1}
    assert cases[f'{number}_snapshot_sold_and_close_count']['sold'] == {'a': 1}


def test_exit_helper_parameters_preserved(cases):
    first = cases['0_sell_keys_immediate']['trace']
    assert [r for r in first if r[0] == 'exit_predicate'] == [
        ['exit_predicate', ['a', 8, 89], {}], ['exit_predicate', ['b', 8, 89], {}]]
    second = cases['1_sell_keys_immediate']['trace']
    assert [r for r in second if r[0] == 'exit_predicate'] == [
        ['exit_predicate', ['a', 'net_pct_main', 30], {'which_day': 'previous'}],
        ['exit_predicate', ['b', 'net_pct_main', 30], {'which_day': 'previous'}]]


@pytest.mark.parametrize('number', [0, 1])
def test_negative_slots_allocate_but_do_not_order(cases, number):
    row = cases[f'{number}_buy_6_False']
    assert row['trace'] == [['allocation', ['a', 'b', 'c', 'd'], 5, 'by_cap_mean', 100]] if number == 0 else row['trace'] == [
        ['allocation', ['p0', 'a', 'b', 'c', 'd'], 5, 'by_cap_mean', 100]]
    assert cases[f'{number}_buy_5_False']['trace'] == []


def test_original_repeat_buy_and_cash_helper_calls(cases):
    first = cases['0_buy_1_False']['trace']; second = cases['1_buy_1_False']['trace']
    assert first[0][1] == ['a', 'b', 'c', 'd'] and second[0][1] == ['p0', 'a', 'b', 'c']
    assert all(r[2:] == [10, None, None] for r in first if r[0] == 'size')
    assert [r[1] for r in second if r[0] == 'order'][0] == 'p0'


@pytest.mark.parametrize('number', [0, 1])
def test_retries_only_keep_actual_positions(cases, number):
    assert cases[f'{number}_retry_False']['orders'] == [['a', 0]]
    assert cases[f'{number}_retry_False']['retry'] == ['a']
    assert cases[f'{number}_retry_True']['retry'] == []
    assert cases[f'{number}_buy_gate_risk_management_signal']['no_helpers_required'] is True
    assert cases[f'{number}_buy_gate_daily_risk_management']['no_helpers_required'] is True


@pytest.mark.parametrize('number', [0, 1])
def test_literal_state_filters_not_historical_evidence(cases, number):
    row = cases[f'{number}_literal_state_filters']['results']
    assert 'paused' not in row['paused_filter'] and 'st' not in row['st_filter']
    assert 'delisted' not in row['delisted_filter'] and 'star' not in row['delisted_filter']
    assert 'limit' not in row['high_limit_filter']


def test_entry_helpers_and_sell_before_buy(cases):
    first = cases['0_trade_helper_and_sell_buy_order']['trace']; second = cases['1_trade_helper_and_sell_buy_order']['trace']
    assert first[4:6] == [['MA_entry', 'a', 8, 89], ['MA_entry', 'b', 8, 89]]
    assert second[4:6] == [['flow_entry', 'a', 'net_pct_main', 20, {'which_day': 'previous'}],
        ['flow_entry', 'b', 'net_pct_main', 20, {'which_day': 'previous'}]]
    assert first[-2:] == second[-2:] == [['sell', ['a']], ['buy', ['a']]]
    for number in range(2):
        row = cases[f'{number}_trade_skipped_counters']
        assert row['trace'] == [] and row['buy_days'] == row['sell_days'] == 2


def test_native_requests_are_bounded_and_not_historical_substitutes():
    assert len(batch.QUERIES) == 2
    url, params = batch.expected_request('money-flow-600519')
    assert params['secid'] == '1.600519' and 'start_date' not in params and 'end_date' not in params
    batch.validate_request('money-flow-600519', 'GET', url, {**params, '_': 1})
    with pytest.raises(ValueError): batch.validate_request('money-flow-600519', 'GET', url, {**params, '_': 1, 'beg': '20000101'})
    with pytest.raises(ValueError): batch.validate_request('money-flow-600519', 'POST', url, {**params, '_': 1})
    with pytest.raises(ValueError): batch.expected_request('unknown')


def test_unknown_helpers_and_archive_candidates_never_admit_trades():
    result = batch.component_summary(batch.DIRECTORY)
    assert result['original_MA_windows'] == result['original_money_flow_windows'] == result['original_pool_financial_rank_days'] == 0
    assert result['cost_backtests'] == [] and result['new_trading_backtests'] == 0
