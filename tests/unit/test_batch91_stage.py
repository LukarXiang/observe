"""Original futures failure paths and retrospective candidate statistics stay explicit."""
import ast
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch91 as batch


def test_all_four_original_if_bodies_are_identical():
    trees = [batch.source_tree(number) for number in range(1, 5)]
    assert len({hashlib.sha256(ast.dump(t, include_attributes=False).encode()).hexdigest() for t in trees}) == 1
    assert [row['name'] for row in batch.definitions(trees[0])] == [
        'initialize', 'before_market_open', 'set_slip_fee', 'market_open']


@pytest.fixture(scope='module')
def original_cases():
    return batch.network_disabled(batch.diagnostics)


def test_original_time_threshold_and_two_leg_feedback(original_cases):
    for number in range(1, 5):
        rows = {r['case']: r for r in original_cases['cases'] if r['source'] == number}
        assert rows['time_0932_fetches_but_no_orders']['orders'] == []
        assert len(rows['time_0932_fetches_but_no_orders']['requests']) == 2
        assert len(rows['time_0933_opens']['orders']) == 2
        assert len(rows['rejected_first_leg_still_submits_second']['orders']) == 2
        assert len(rows['first_leg_exception_prevents_second']['orders']) == 1
        assert rows['first_leg_exception_prevents_second']['error'] == 'RuntimeError'
        assert len(rows['unchanged_empty_value_duplicates_two_legs']['orders']) == 4


def test_original_last_key_and_empty_code_behavior(original_cases):
    for number in range(1, 5):
        rows = {r['case']: r for r in original_cases['cases'] if r['source'] == number}
        assert [o['code'] for o in rows['close_only_last_key']['orders']] == ['old2', 'old4']
        assert [o['code'] for o in rows['close_empty_dicts_submits_empty_codes']['orders']] == ['', '']
        assert rows['close_only_last_key']['orders'][0]['close_today'] is False
        assert 'close_today' not in rows['close_only_last_key']['orders'][1]


def test_size_and_compatibility_fail_before_time_guard(original_cases):
    rows = {r['case']: r for r in original_cases['cases'] if r['source'] == 1}
    assert rows['infinite_cash_fails_before_time_guard']['error'] == 'OverflowError'
    assert rows['datetime_series_minus1_failure']['error'] == 'KeyError'
    assert [o['quantity'] for o in rows['NaN_cash_zero_targets']['orders']] == [0, 0]
    assert [o['quantity'] for o in rows['small_cash_zero_targets']['orders']] == [0, 0]
    assert all(o['quantity'] < 0 for o in rows['negative_cash_negative_targets']['orders'])
    assert original_cases['case_count'] == 158
    assert original_cases['new_trading_backtests'] == 0
    assert original_cases['IF_unique_executable_bodies'] == 1


def test_population_zscore_not_sample_std():
    zscore = batch.namespace(0, {'np': np})['zscore']
    values = zscore(pd.Series([1., 2., 3.]))
    assert math.isclose(values.iloc[-1], math.sqrt(1.5), rel_tol=1e-15)
    assert zscore(pd.Series([3., 3., 3.])).isna().all()


def test_retrospective_regression_preserves_intercept_and_population_scale():
    x = np.arange(10., 20.)
    y = 5.+2.*x+np.array([1., -1.]*5)
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10), 'C': x, 'CS': y})
    zscore = batch.namespace(0, {'np': np})['zscore']
    values, summary = batch.regression(frame, 'synthetic', zscore)
    assert math.isclose(summary['spread_mean'], -summary['intercept_numpy'], abs_tol=1e-12)
    assert abs(values.original_zscore_on_candidate.mean()) < 1e-14
    assert math.isclose(np.std(values.original_zscore_on_candidate), 1., rel_tol=1e-14)
    assert summary['retrospective_full_period_fit']
    assert not summary['original_statsmodels_OLS_executed']
    assert not summary['coint_executed']


@pytest.mark.parametrize('bad', ['duplicates', 'negative', 'missing'])
def test_regression_rejects_invalid_candidate_rows(bad):
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=4),
        'C': [1., 2., 3., 4.], 'CS': [2., 3., 4., 6.]})
    if bad == 'duplicates': frame.loc[1, 'date'] = frame.loc[0, 'date']
    if bad == 'negative': frame.loc[0, 'C'] = -1.
    if bad == 'missing': frame.loc[0, 'CS'] = math.nan
    with pytest.raises(ValueError):
        batch.regression(frame, 'invalid', batch.namespace(0, {'np': np})['zscore'])


def test_installed_and_reference_minute_api_have_no_historical_date_parameter():
    evidence = batch.api_evidence()
    assert evidence['function_AST_equal']
    assert evidence['historical_date_parameter'] is False
    assert evidence['queries'][0] == {'symbol': 'IF2003', 'type': '1'}
