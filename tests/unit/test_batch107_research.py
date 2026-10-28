"""Literal source edge cases and independent native-operand regression checks."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch107 as study
from scripts import probe_strategy_batch107_pool as probe


def cases():
    return {r['case']: r for r in study.diagnostics()['cases']}


@pytest.fixture(scope='module')
def observations():
    with study.network_blocked():
        return cases()


def test_actual_pools_schedules_and_fee_classes():
    ns, _, calls = study.initialize(0)
    assert ns['g'].stock_pool == list(study.POOL)
    assert ns['g'].momentum_day == 29 and ns['g'].stock_num == 1
    assert [(c[1], c[2]['time']) for c in calls if c[0] == 'daily'] == [
        ('my_trade', '9:30'), ('check_lose', 'open'), ('print_trade_info', '15:30')]
    ns, _, calls = study.initialize(1)
    assert len(ns['g'].all_stks) == 24 and ns['g'].all_stks[0] == '399306.XSHE'
    assert ns['g'].money == 100000. and ns['g'].max_number == 1
    assert [(c[1], c[2]['time']) for c in calls if c[0] == 'daily'] == [
        ('xuangu_buy', '2:00'), ('xuangu_sell', '3:00'), ('buy', '10:00'), ('sell', '10:00')]
    assert [c[2] for c in calls if c[0] == 'cost'] == [{'type': 'stock'}]


@pytest.mark.parametrize('name,expected', [
    ('27_global_pool_ignores_argument', [study.POOL[0]]),
    ('27_all_negative_still_selects', [study.POOL[2]]),
    ('27_tied_scores_original_pool_order', [study.POOL[0]]),
    ('27_new_up_equal', []), ('27_held_up_equal', [study.POOL[0]]),
    ('27_new_down_equal', []), ('27_held_down_equal', [study.POOL[0]]),
    ('27_filtered_top_no_second_choice', [[]]),
    ('27_open_filled_0', False), ('27_open_filled_1', True),
    ('27_close_complete', True), ('27_close_partial', False), ('27_close_cancelled', False),
    ('27_stop_at_minus80', [[study.POOL[0], 0]]), ('27_stop_above_minus80', []),
    ('61_sell_current_equality_previous_less', True), ('61_sell_previous_equality_no_cross', False)])
def test_original_branch_results(observations, name, expected):
    assert observations[name]['error'] is None
    assert observations[name]['value'] == expected


@pytest.mark.parametrize('name,error', [('27_math_implicit_failure', 'NameError'),
    ('27_current_series_negative_index', 'KeyError'), ('27_missing_position_key', 'KeyError'),
    ('61_dict_has_key_buy_failure', 'AttributeError'), ('61_dict_has_key_sell_failure', 'AttributeError')])
def test_original_defects_are_retained(observations, name, error):
    assert observations[name]['error']['type'] == error


def test_constant_history_is_not_a_valid_signal(observations):
    values = observations['27_constant_window_original_nonfinite']['value']
    assert any(isinstance(v, dict) and 'nonfinite' in v for v in values)
    assert study.independent_score(np.ones(29)*100) is None


def test_ref_window_and_insufficient_history_are_distinct(observations):
    complete = observations['61_ma60_ref2_uses_first60_of62']['value']
    partial = observations['61_ma60_missing_history_not_blocked']['value']
    assert complete['mean'] == 30.5 and complete['requests'][0]['count'] == 62
    assert partial['mean'] == 2. and partial['requests'][0]['count'] == 62


@pytest.mark.parametrize('name', ['61_paused_skip_before_has_key', '61_held_skip_before_has_key', '61_full_slot_skip_before_has_key'])
def test_skip_branches_do_not_require_legacy_mapping(observations, name):
    assert observations[name]['error'] is None


def test_exponential_price_slope_and_scale_invariance():
    original = study.score_function()
    x = np.arange(29)
    values = 12*np.exp(.002*x)
    a = original(pd.DataFrame({'close': values}))
    b = original(pd.DataFrame({'close': values*3.5}))
    independent = study.independent_score(values)
    assert math.isclose(a[0], .002, abs_tol=1e-14)
    assert math.isclose(a[3], 1., abs_tol=1e-14)
    assert math.isclose(a[4], math.expm1(.5), abs_tol=1e-12)
    assert np.allclose(a, independent, rtol=1e-10, atol=1e-12)
    assert math.isclose(a[4], b[4], abs_tol=1e-12)


def test_calendar_gap_is_reported_and_operands_remain_native():
    dates = pd.bdate_range('2022-01-03', periods=40).strftime('%Y-%m-%d').tolist()
    inputs = {s: pd.DataFrame({'date': dates, 'close': np.exp(np.arange(40)*(.001+j*.001))})
        for j, s in enumerate(study.POOL)}
    inputs[study.POOL[2]] = inputs[study.POOL[2]].drop(index=20).reset_index(drop=True)
    calendar = pd.DataFrame({'date': pd.to_datetime(dates), 'is_open': True})
    component, ranking, profiles, _ = study.operands(inputs, calendar, study.score_function())
    assert len(component) == 12+12+11 and len(ranking) == 11
    assert profiles[study.POOL[2]]['calendar_missing_dates'] == [dates[20]]
    assert component.loc[component.instrument.eq(study.POOL[2]), 'crosses_calendar_gap'].all()
    assert ranking.crosses_calendar_gap.all()
    assert not ranking.independent_order_difference.any()


def test_probe_uses_only_original_pool_fund_codes():
    ns, _, _ = study.initialize(1)
    assert set(probe.FUNDS) == {s[:6] for s in ns['g'].all_stks if s != '399306.XSHE'}
    with pytest.raises(ValueError): probe.expected_url('399306')
    assert '/sz159608/' in probe.expected_url('159608')


def test_source_text_contains_two_identical_name_definitions():
    import ast
    from observe.strategy_catalog import read_source
    tree = ast.parse(read_source(study.source_path(1))[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_name']
    assert len(nodes) == 2
    assert ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False)
