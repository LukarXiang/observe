"""Frozen ETF original rules and candidate gates; synthetic cases are not trades."""
import ast
import math

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch84 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_parent_frozen_evidence_checked(monkeypatch):
    original = batch.file_sha; target = batch.previous.DIRECTORY/'diagnostics.json'
    monkeypatch.setattr(batch, 'file_sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='Accepted83 evidence changed'): batch.binding('data')


def test_complete_source_rules_and_schedule(cases):
    first = cases['0_defaults']
    assert first['state']['etf_pool'] == list(batch.POOL) and first['state']['m_days'] == 25
    assert first['schedules'] == [['trade', ['9:40'], {}]] and first['slippage'] == [0.]
    assert first['fees'][0][0]['open_commission'] == .0002 and first['fees'][0][0]['min_commission'] == 5
    second = cases['1_defaults']
    assert second['state']['security1'] == '513030.XSHG'
    assert second['schedules'][2] == ['after_market_close', [], {'time': '14:59'}]
    assert second['fees'][0][0]['open_commission'] == .00005 and second['fees'][0][0]['min_commission'] == .1
    assert second['slippage'] == [.001]


def test_rotation_missing_math_and_series_logs_remain_explicit(cases):
    assert cases['rotation_missing_math']['error'] == 'NameError'
    assert cases['rotation_positive']['ranking'] == list(batch.POOL)
    assert cases['rotation_positive']['logged_types'] == ['Series']*14
    assert cases['rotation_negative_still_selects']['first'] == batch.POOL[0]
    assert cases['rotation_constant']['logged_scores'] == ['NaN']*14


def test_positions_platform_semantics_not_silently_assumed(cases):
    assert cases['rotation_empty_dict']['error'] == 'KeyError'
    assert cases['rotation_zero_missing']['orders'] == [[batch.POOL[0], 1000.]]
    assert cases['rotation_held_target']['orders'] == []
    assert cases['rotation_failed_sale']['orders'] == [[batch.POOL[1], 0]]
    assert cases['rotation_immediate_sale']['orders'] == [[batch.POOL[1], 0], [batch.POOL[0], 1500.]]


@pytest.mark.parametrize('gap,expected', [(math.nextafter(.003, -math.inf), False), (.003, True),
    (math.nextafter(.003, math.inf), True), (.1, True), (math.nextafter(.5, -math.inf), True), (.5, True),
    (math.nextafter(.5, math.inf), False), (-.01, False)])
def test_t0_absolute_inclusive_boundaries(cases, gap, expected):
    assert bool(cases['T0_gap_'+repr(gap)]['orders']) is expected


def test_binary_subtraction_at_nominal_threshold_preserved(cases):
    row = cases['T0_decimal_subtraction_boundary']
    assert row['actual_gap'] < .003 and row['orders'] == []


def test_any_position_triggers_only_target_exit(cases):
    exits = [r for name, r in cases.items() if name.startswith('T0_exit_')]
    assert exits[0]['orders'] == []
    assert exits[1]['keys'] == ['unrelated'] and exits[1]['orders'] == [['513030.XSHG', 0]]
    assert exits[2]['orders'] == [['513030.XSHG', 0]]


@pytest.mark.parametrize('slope', [-.01, .001, .05])
def test_original250_annualization_and_independent_ols(slope):
    values = np.exp(np.arange(25)*slope)
    a = batch.score(values); b = batch.independent_score(values)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    assert math.isclose(a[1], math.exp(slope*250)-1, rel_tol=1e-10)


def test_only_missing_daily_symbols_requested_and_minute_contract():
    assert len(batch.QUERIES) == 14
    assert 'daily-518880' not in batch.QUERIES and 'daily-513100' not in batch.QUERIES
    url, params = batch.expected_request('minute-513030')
    assert url.endswith('/trends2/get') and params['ndays'] == '5' and params['secid'] == '1.513030'
    assert batch.QUERIES['minute-513030']['parameters']['period'] == '1'


def test_selected_loader_rejects_unsafe_defaults():
    with pytest.raises(ValueError, match='Unsafe'):
        batch.previous.selected_nodes(ast.parse('def f(x=bad()):\n return x'), ['f'], {'bad': lambda: None})


def test_partial_pool_and_future_rows_do_not_create_original_ranking(tmp_path, monkeypatch):
    dates = pd.date_range('2026-09-01', periods=35)
    path = tmp_path/'input.parquet'
    pd.DataFrame({'date': dates, 'open': np.exp(np.arange(35)*.001),
        'close': np.exp(np.arange(35)*.001)}).to_parquet(path, index=False)
    row = {'file': str(path), 'sha256': batch.file_sha(path)}
    monkeypatch.setattr(batch, 'read', lambda p: [{'instrument': batch.POOL[0], 'payload': row}]
        if p.name == 'reused-inputs.json' else {'results': []})
    output, gaps, summary = batch.compute(tmp_path)
    assert summary['covered_pool_size'] == 1 and len(summary['missing_pool']) == 13
    assert summary['full_pool_25_row_days'] == 0 and summary['new_trading_backtests'] == 0
    assert len(output) == 4 and gaps.empty and summary['cost_backtests'] == []
    assert output.window_end.lt(output.date).all() and output.date.max() == pd.Timestamp('2026-09-29')


def test_duplicate_candidate_dates_block_components(tmp_path, monkeypatch):
    path = tmp_path/'input.parquet'
    pd.DataFrame({'date': ['2026-09-01']*2, 'open': [1., 1.], 'close': [1., 1.]}).to_parquet(path, index=False)
    row = {'file': str(path), 'sha256': batch.file_sha(path)}
    monkeypatch.setattr(batch, 'read', lambda p: [{'instrument': batch.POOL[0], 'payload': row}]
        if p.name == 'reused-inputs.json' else {'results': []})
    with pytest.raises(ValueError, match='Bad candidate dates'): batch.compute(tmp_path)


def test_complete_pool_ranking_is_frozen_and_independently_verified(tmp_path, monkeypatch):
    rows = []
    for j, instrument in enumerate(batch.POOL):
        path = tmp_path/(instrument+'.parquet'); close = np.exp(np.arange(26)*(.01-j*.0001))
        pd.DataFrame({'date': pd.date_range('2026-09-01', periods=26), 'open': close, 'close': close}).to_parquet(path, index=False)
        rows.append({'instrument': instrument, 'payload': {'file': str(path), 'sha256': batch.file_sha(path)}})
    monkeypatch.setattr(batch, 'read', lambda p: rows if p.name == 'reused-inputs.json' else {'results': []})
    output, _, summary = batch.compute(tmp_path)
    assert len(output) == 14 and summary['missing_pool'] == [] and summary['full_pool_25_row_days'] == 1
    ranking = summary['full_pool_rankings'][0]
    assert ranking['original_rank'] == ranking['independent_rank'] == list(batch.POOL)
    assert ranking['independent_order_verified'] is True
    assert summary['new_trading_backtests'] == 0


def test_gap_uses_observed_prior_date_without_filling_missing_session(tmp_path, monkeypatch):
    path = tmp_path/'input.parquet'
    pd.DataFrame({'date': ['2026-09-01', '2026-09-03'], 'open': [1., 1.02], 'close': [1.01, 1.03]}).to_parquet(path, index=False)
    row = {'case': 'daily-513030', 'query': batch.QUERIES['daily-513030'], 'status': 'success',
        'file': str(path), 'sha256': batch.file_sha(path)}
    monkeypatch.setattr(batch, 'read', lambda p: [] if p.name == 'reused-inputs.json' else {'results': [row]})
    output, gaps, summary = batch.compute(tmp_path)
    assert output.empty and len(gaps) == 1 and gaps.prior_date.iloc[0] == pd.Timestamp('2026-09-01')
    assert gaps.threshold_hit.iloc[0] and summary['T0_threshold_hits'] == 1
    assert summary['new_trading_backtests'] == 0 and summary['cost_backtests'] == []


def test_offline_creates_catalog_before_validation(tmp_path, monkeypatch):
    for name in ('binding', 'validate_sources'): monkeypatch.setattr(batch, name, lambda *a: None)
    monkeypatch.setattr(batch, 'implementation', lambda: {})
    for name in ('diagnostics', 'candidate_audit', 'inventory', 'registered_coverage'):
        monkeypatch.setattr(batch, name, lambda *a: {})
    monkeypatch.setattr(batch, 'compute', lambda d: (pd.DataFrame(), pd.DataFrame(), {}))
    monkeypatch.setattr(batch, 'offline_catalog', lambda *a: {'result': 'match'})
    monkeypatch.setattr(batch, 'read', lambda *a: {})
    monkeypatch.setattr(batch.pd, 'read_parquet', lambda *a: pd.DataFrame())
    monkeypatch.setattr(batch, 'archive', lambda *a: {'status': 'ok'})
    def validate(*args): assert (tmp_path/'offline-catalog.json').is_file()
    monkeypatch.setattr(batch, 'validate_catalog', validate)
    assert batch.offline('data', tmp_path)['status'] == 'ok'


def test_temporal_unit_comparison_preserves_values_and_original_frames():
    left = pd.DataFrame({'date': pd.to_datetime(['2026-09-01']).as_unit('s'), 'score': [1.]})
    right = pd.DataFrame({'date': pd.to_datetime(['2026-09-01']).as_unit('ms'), 'score': [1.]})
    with pytest.raises(AssertionError): pd.testing.assert_frame_equal(left, right, check_exact=True)
    batch.compare_temporal_frames(left, right, ('date',))
    assert str(left.date.dtype) == 'datetime64[s]' and str(right.date.dtype) == 'datetime64[ms]'
    right.loc[0, 'date'] += pd.Timedelta(seconds=1)
    with pytest.raises(AssertionError): batch.compare_temporal_frames(left, right, ('date',))
