from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch44 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def cases(directory):
    with redirect_stdout(io.StringIO()): result = batch.diagnostics(directory)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result['not_a_backtest'] is True and result['platform_equivalent'] is False
    return {r['case']: r for r in result['cases']}


def test_original_pool_lengths_and_shared_codes(sources):
    pools = batch.pools(sources)
    assert list(map(len, pools)) == [4, 10, 30]
    assert len({s for pool in pools for s in pool}) == 37
    assert pools[0] == ['518880.XSHG', '513100.XSHG', '159915.XSHE', '510880.XSHG']


def test_epo1200_original_request_and_analytic_anchor(sources):
    r = cases(sources)['epo52_original1200_request_and_inverse_variance_anchor']
    assert r['requests'][0]['kwargs'] == {'count': 1200, 'end_date': '2021-01-04', 'frequency': 'daily', 'fields': ['close']}
    assert r['weights'] == pytest.approx(r['reference'], abs=1e-12)
    assert sum(r['weights']) == pytest.approx(1)


def test_epo_zero_signal_and_zero_variance_preserve_failure(sources):
    rows = cases(sources)
    r = rows['epo52_w1_still_evaluates_zero_signal_gamma']
    assert r['weights'] == ['nan'] * 10 and r['warnings']
    assert rows['epo_zero_variance_not_regularized']['error'].startswith('ValueError:')
    assert rows['epo_all_negative_normalization_can_be_nonfinite']['weights'] == ['nan', 'nan']


def test_epo32_retains_actual_solver_matrix(sources):
    r = cases(sources)['epo32_uses_cov_tilde_not_computed_shrunk_cov']
    assert np.array(r['solve_matrix']) == pytest.approx(np.array(r['reference']), abs=1e-14)
    assert r['diff_from_unused'] > 0
    assert np.max(np.abs(np.array(r['solve_matrix']) - np.array(r['unused_matrix']))) == pytest.approx(r['diff_from_unused'])


def test_epo_original_order_and_filtered_old_position(sources):
    rows = cases(sources)
    assert rows['epo52_original_pool_order_without_sell_first']['orders'] == [['A', 4000.], ['B', 6000.]]
    assert rows['epo32_filtered_old_holding_not_explicitly_cleared']['orders'] == [['B', 10000.]]
    assert rows['epo32_listing30_inclusive']['accepted'] == ['equal']
    assert rows['epo32_datetime_platform_injection']['error'].startswith('NameError:')


def test_correlation_missing_column_and_stable_self_ties(sources):
    rows = cases(sources)
    assert rows['corr729_abs_self_correlation_ties_keep_input_order']['selected'] == ['A', 'B']
    assert rows['corr729_abs_self_correlation_ties_keep_input_order']['reference'] == ['A', 'B']
    assert rows['corr729_missing_one_price_drops_whole_column']['selected'] == ['B']
    assert rows['corr_math_platform_injection']['error'].startswith('NameError:')


def test_correlation_rejected_sell_and_missing_holding_proxy(sources):
    rows = cases(sources)
    assert rows['corr_rejected_sell_blocks_new_buy']['orders'] == [['old', 0]]
    assert rows['corr_plain_empty_positions_requires_platform_zero_proxy']['error'].startswith('KeyError:')
    assert rows['corr_plain_empty_positions_requires_platform_zero_proxy']['orders'] == []
    assert rows['rank25_constant_log_can_be_nonfinite']['values'][1:] == ['-inf', 'nan']


def test_score_original_full_window_and_strict_bounds(sources):
    code, _ = batch.rank_kernel(sources)
    values = 100 * np.exp(.001 * np.arange(25) + .0001 * np.sin(np.arange(25)))
    actual = batch.original_score(values, code); reference = batch.reference_score(values)
    assert actual == pytest.approx(reference, abs=1e-10)
    assert (-.5 < actual[2] < 4.5) == (-.5 < reference[2] < 4.5)
    with pytest.raises(ValueError, match='Incomplete25-row'): batch.original_score(values[:-1], code)


def test_complete_matrix_count_and_no_missing_price_substitution(sources):
    ns = batch.namespaces(sources)
    small = pd.DataFrame({'A': [100., 101.], 'B': [90., 92.]})
    with pytest.raises(ValueError, match='Incomplete EPO'): batch.original_epo(small, ns[0], 1200)
    with pytest.raises(ValueError, match='Incomplete729-row'): batch.original_correlation(small, ns[2])
    small.loc[0, 'A'] = np.nan
    with pytest.raises(ValueError, match='Incomplete EPO'): batch.original_epo(small, ns[1], 2)


def test_covariance_ddof1_and_invalid_operands():
    values = np.array([[1., 2.], [3., 1.], [2., 5.]])
    cov, mean = batch.covariance(values)
    assert cov == pytest.approx(np.cov(values, rowvar=False, ddof=1))
    assert mean == pytest.approx(values.mean(axis=0))
    with pytest.raises(ValueError, match='Incomplete covariance'): batch.covariance(values[:1])
    with pytest.raises(ValueError, match='Incomplete covariance'): batch.covariance([[1., float('nan')], [2., 3.]])


def test_correlation_details_preserve_original_filters_and_scores(sources):
    t = np.arange(729, dtype=float)
    prices = pd.DataFrame({'A': 100 * np.exp(.01 * np.sin(t)), 'B': 100 * np.exp(.012 * np.sin(t + .5)), 'flat': 100.})
    ns = batch.namespaces(sources)[2]
    details = batch.original_correlation(prices, ns, details=True)
    selected, volatility, scores = batch.reference_correlation(prices)
    assert details['selected'] == batch.original_correlation(prices, ns) == selected
    assert list(details['volatility'].index) == ['A', 'B']
    assert list(details['volatility']) == pytest.approx(volatility[:2], abs=1e-12)
    assert list(details['scores'].values()) == pytest.approx(scores, abs=1e-12)


def test_source_sha_corruption_blocks_original_functions(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.namespaces(sources)
    with pytest.raises(ValueError, match='Selected source changed'): batch.rank_kernel(sources)


def test_probe_sample_unproven_and_isolated_namespace(tmp_path, monkeypatch):
    import akshare as ak
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_em', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(batch.raw, 'save', save)
    row = batch.worker('diagnostic-root', tmp_path, 'gold_daily')
    assert row['status'] == 'sample' and row['strict_usable'] is False and row['published'] is False
    assert calls == [['batch44_dependency_probe', 'gold_daily', tmp_path.name]]


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_substitution_rejected(tmp_path, monkeypatch, change):
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
