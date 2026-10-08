from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch45 as batch


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


def test_original_pools_and_shared_numerical_ast(sources):
    pools = batch.pools(sources)
    assert list(map(len, pools)) == [13, 30, 30] and pools[1] == pools[2]
    assert len({s for pool in pools for s in pool}) == 36
    assert len(batch.shared_kernel_evidence(sources)) == 5
    assert batch.rank_kernel(sources, 1)[1] == batch.rank_kernel(sources, 2)[1]


def test_epo1200_anchored_w02_reference_and_lambda(sources):
    rows = cases(sources); r = rows['epo17_original1200_request_and_w02']
    assert r['requests'][0]['kwargs'] == {'count': 1200, 'end_date': '2021-01-04', 'frequency': 'daily', 'fields': ['close']}
    assert r['weights'] == pytest.approx(r['reference'], abs=1e-12)
    assert sum(r['weights']) == pytest.approx(1)
    r = rows['epo17_endogenous_anchored_ignores_lambda']
    assert r['weights10'] == r['weights40']


def test_epo_degenerate_signal_is_not_repaired(sources):
    r = cases(sources)['epo17_zero_signal_gamma_not_regularized']
    assert r['weights'] == ['nan'] * 3 and r['warnings']


def test_rank17_preserves_positive_only_without_upper_bound(sources):
    assert cases(sources)['rank17_positive_only_no_upper_score_cap']['rank'] == ['strong']
    code, _ = batch.rank_kernel(sources, 0)
    values = 100 * np.exp(.01 * np.arange(34))
    result = batch.original_score(values, code, 34)
    assert result[2] > 4.5 and result == pytest.approx(batch.reference_score(values), abs=1e-10)
    with pytest.raises(ValueError, match='Incomplete score'): batch.original_score(values[:-1], code, 34)


def test_epo_sell_before_three_targets_and_empty_selection(sources):
    rows = cases(sources)
    assert rows['epo17_sell_then_three_nav_targets_without_fill_check']['orders'] == [['old', 0], ['B', 2000.], ['C', 3000.], ['D', 5000.]]
    r = rows['epo17_empty_selection_still_optimizes_after_sell']
    assert r['orders'] == [['old', 0]] and r['requested'] == [[]] and r['error']


def test_trend_triangular_ratio_and_datetime_failure(sources):
    rows = cases(sources)
    for number in (1, 2):
        r = rows[f'trend{number}_3500_triangular_not_average_run']
        assert r['ratio'] == r['reference'] == 1735.5
        assert rows[f'trend{number}_datetime_integer_label_incompatible']['error'].startswith('KeyError:')
        assert rows[f'trend{number}_30_rows_zero_denominator']['error'].startswith('ZeroDivisionError:')


def test_trend3500_pre_paused_and_strict_filter(sources):
    r = cases(sources)['trend_history3500_pre_skip_paused_and_strict_gt3']
    assert r['selected'] == ['above']
    assert r['requests'] == [{'args': [3500], 'kwargs': {'unit': '1d', 'field': 'close', 'security_list': ['equal', 'above'], 'df': True, 'skip_paused': True, 'fq': 'pre'}}]


def test_independent_run_segments_and_mean_reference():
    assert batch.cumulative_trend([True, True, False, True]) == 1.
    assert batch.cumulative_trend([False] * 4) == 0.
    assert batch.cumulative_trend([True] * 4) == 2.5
    with pytest.raises(ValueError, match='Empty trend'): batch.cumulative_trend([])
    values = [1e16, 1., 1., 1e16]
    assert batch.stable_ma(values, 2)[1:].tolist() == [5e15, 1., 5e15]


def test_corr243_original_ties_and_missing_whole_column(sources):
    rows = cases(sources)
    assert rows['corr243_absolute_self_tie_stable_order']['selected'] == ['A', 'B']
    assert rows['corr243_one_missing_price_drops_column']['selected'] == ['B']
    assert rows['rank_platform_math_injection']['error'].startswith('NameError:')


def test_correlation_detail_return_preserves_original_scores(sources):
    t = np.arange(243, dtype=float)
    prices = pd.DataFrame({'A': 100 * np.exp(.01 * np.sin(t)), 'B': 100 * np.exp(.012 * np.sin(t + .5))})
    ns = batch.namespaces(sources)[1]; ns['history'] = lambda *a, **kw: prices.copy()
    r = ns['min_corr_details'](list(prices)); names, scores = batch.reference_correlation(prices)
    assert r['selected'] == ns['min_corr'](list(prices)) == names
    assert list(r['scores'].values()) == pytest.approx(scores, abs=1e-12)
    with pytest.raises(ValueError, match='Degenerate correlation'): batch.reference_correlation(pd.DataFrame({'A': [1., 1., 1.]}))


def test_daily_and_monthly_cache_are_distinct_state_machines(sources):
    rows = cases(sources); r = rows['daily74_recomputes_monthly97_reads_cache']
    assert r['trend_calls'] == [1]
    assert r['correlation_candidates'] == [[1, ['current']], [2, ['cached']]]
    assert rows['monthly97_up_strength_replaces_cache']['cache'] == ['updated']
    r = rows['monthly97_initialize_then_monthfirst9_and_daily10']
    assert r['cache'] == ['initialized']
    assert r['schedules'] == [['run_daily', 'trade', ['10:00'], {}], ['run_monthly', 'up_strength', [1, '9:00'], {}]]


def test_rejected_sell_and_zero_holding_proxy(sources):
    rows = cases(sources)
    assert rows['trend_rejected_sell_blocks_new_buy']['orders'] == [['old', 0]]
    r = rows['trend_empty_positions_requires_zero_proxy']
    assert r['error'].startswith('KeyError:') and r['orders'] == []


def test_source_corruption_blocks_all_original_kernels(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.namespaces(sources)
    with pytest.raises(ValueError, match='Selected source changed'): batch.shared_kernel_evidence(sources)


def test_probe_sample_isolated_and_unproven(tmp_path, monkeypatch):
    import akshare as ak
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_em', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(batch.raw, 'save', save)
    row = batch.worker('diagnostic-root', tmp_path, 'innovation_daily')
    assert row['status'] == 'sample' and row['strict_usable'] is False and row['published'] is False
    assert calls == [['batch45_dependency_probe', 'innovation_daily', tmp_path.name]]


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
