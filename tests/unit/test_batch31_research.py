from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch31 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_stable_ols_independently_matches_scipy(seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=120); y = 2 * x + rng.normal(size=120)
    actual = batch.reference_ols(x, y); reference = stats.linregress(x, y)
    np.testing.assert_allclose(actual, [reference.intercept, reference.slope, reference.rvalue ** 2], atol=1e-12)


@pytest.mark.parametrize('x,y', [([], []), ([1], [2]), ([1, 2], [3]), ([1, 1], [2, 3]), ([1, 2], [3, 3]), ([1, np.nan], [2, 3])])
def test_invalid_or_degenerate_ols_blocks(x, y):
    with pytest.raises(ValueError): batch.reference_ols(x, y)


def test_weekly_clock_counts_verified_sessions_and_excludes_execution_close():
    dates = pd.bdate_range('2005-01-03', periods=620).strftime('%Y-%m-%d').tolist()
    dates += ['2008-01-08', '2008-01-10', '2008-01-11']
    clock = batch.weekly_clock(pd.DataFrame({'date': dates}))
    assert clock[-1] == {'execution_date': '2008-01-11', 'price_end': '2008-01-10', 'end_index': 621}
    assert all(row['execution_date'] > row['price_end'] and row['end_index'] >= 617 for row in clock)


def test_original_capm_inverse_weights_empty_and_numeric_types(sources):
    row, _, _ = batch.capm_weight_case(sources, list(map(np.float64, range(1, 21))))
    assert list(row['buy_targets']) == [f's{k:02d}' for k in range(4, 20)]
    assert row['buy_targets']['s19'] == 0
    assert sum(row['buy_targets'].values()) == pytest.approx(100000.)
    empty, _, _ = batch.capm_weight_case(sources, [])
    assert empty['buy_targets'] == {} and empty['error'] is None
    single, _, _ = batch.capm_weight_case(sources, [np.float64(1)])
    assert single['buy_targets'] == {'s00': 'nonfinite'} and single['warnings']
    python, _, _ = batch.capm_weight_case(sources, [1.])
    assert python['error'].startswith('ZeroDivisionError:')


def test_original_diagnostics_preserve_state_and_boundaries(sources):
    rows = {row['case']: row for row in batch.diagnostics(sources)['cases']}
    assert rows['capm_sellable_value_and_rejected_trade_flag']['if_trade'] is False
    assert rows['capm_short_history_passes_eligibility']['approved'] == ['short']
    assert rows['rsrs_initial_loop_tail']['slopes'] == 599
    assert rows['rsrs_initial_loop_tail']['last_retained_pair'] == [598, 615]
    assert rows['rsrs_rank3_can_remain']['orders'] == []
    assert rows['rsrs_keep_empty_can_buy']['orders'] == [['e0', 450.], ['e1', 450.]]
    assert rows['rsrs_rank4_replaces']['orders'] == [['e4', 0], ['e0', 450.], ['e1', 450.]]
    assert rows['rsrs_degenerate_inputs_not_replaced']['z_finite'] is False
    assert rows['open_all_candidates_and_list_mutation']['remaining_buylist'] == ['b']
    assert not rows['eye_brain_minus10_exact']['sold']
    assert rows['eye_brain_minus10_below']['sold']
    assert not rows['eye_brain_plus50_exact']['sold']
    assert rows['eye_brain_plus50_above']['sold']
    assert rows['eye_brain_minus5_binary_boundary']['sold']
    assert rows['brain_only_today_eye_not_comment_five_days']['buylist'] == ['a']


def test_original_rsrs_appends_only_when_called(sources):
    rng = np.random.default_rng(9); low = np.cumsum(rng.normal(size=630)) + 100
    frame = pd.DataFrame({'date': pd.bdate_range('2005-01-03', periods=630).strftime('%Y-%m-%d'),
        'low': low, 'high': low + rng.uniform(.5, 3, 630), 'close': low + .2})
    ns, holder, _ = batch.rsrs_namespace(sources, frame); holder['end'] = 617; ns['initial_config']()
    assert len(ns['g'].slope_series) == 599
    holder['end'] = 622; ns['get_signal']()
    assert len(ns['g'].slope_series) == 600
    holder['end'] = 627; ns['get_signal']()
    assert len(ns['g'].slope_series) == 601


def test_source_drift_blocks_before_function_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.fragment(sources, 0, 'stop')


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_binding_refuses_substitution(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0' * 64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted', 'sha256': '0' * 64}]
        if change == 'status': row['status'] = 'accepted'
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises((ValueError, FileNotFoundError)): batch.validate_probes(tmp_path)
