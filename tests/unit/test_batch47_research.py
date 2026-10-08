import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch47 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


@pytest.fixture
def cases(sources):
    doc = batch.diagnostics(sources)
    assert json.loads(json.dumps(doc, allow_nan=False)) == doc
    assert doc['not_a_backtest'] is True and doc['platform_equivalent'] is False
    return {row['case']: row for row in doc['cases']}


def test_complete_source_and_all_original_ast_ranges(sources):
    assert batch.original_pool(sources) == ['510050.XSHG', '159928.XSHE', '510300.XSHG', '159915.XSHE']
    for number in range(3):
        for name in batch.RANGES[number]: assert len(batch.block(sources, number, name)[1]) == 64
    assert batch.tree(sources, 2).body[0].lineno == 12


@pytest.mark.parametrize('number,expected', [(0, ['A', 'B', 'C', 'D']), (1, ['A', 'B'])])
def test_candidate_listing_and_delisting_filters(cases, number, expected):
    row = cases[f'pool{number}_original_start_end_filters']
    assert row['codes'] == expected and row['requests'] == [[['etf']]]


@pytest.mark.parametrize('number,count,threshold', [(0, 1000, 5e7), (1, 300, 1e7)])
def test_original_liquidity_window_and_strict_threshold(cases, number, count, threshold):
    row = cases[f'pool{number}_strict_liquidity_and_count']
    assert row['selected'] == ['above'] and row['threshold'] == threshold
    assert [r['kwargs']['count'] for r in row['requests']] == [count] * 3
    assert all(r['kwargs']['end_date'] == '2026-09-29' for r in row['requests'])


@pytest.mark.parametrize('number,clusters', [(0, 30), (1, 24)])
def test_original_cluster_counts_are_not_reduced(cases, number, clusters):
    assert f'n_clusters={clusters}' in cases[f'pool{number}_ten_assets_cannot_fit_original_clusters']['error']
    params = cases[f'pool{number}_installed_kmeans_and_silhouette_features']['params']
    assert params['n_clusters'] == clusters and params['random_state'] == 42 and params['n_init'] == 'auto'


def test_silhouette_86_contains_label_feature_66_does_not(cases):
    normal = cases['pool0_installed_kmeans_and_silhouette_features']
    optimized = cases['pool1_installed_kmeans_and_silhouette_features']
    assert normal['calls'][0]['columns'] == 60 and normal['calls'][0]['contains_cluster_id'] is False
    assert normal['calls'][0]['score'] == pytest.approx(normal['unlabelled_score'], abs=1e-14)
    assert optimized['calls'][0]['columns'] == 61 and optimized['calls'][0]['contains_cluster_id'] is True
    assert abs(optimized['calls'][0]['score'] - optimized['unlabelled_score']) > .01


@pytest.mark.parametrize('number', [0, 1])
def test_overlapping_sets_are_not_replaced_by_components(cases, number):
    row = cases[f'pool{number}_overlapping_sets_not_merged']
    assert row['groups'] == [['A', 'C', 'D'], ['A', 'B', 'C', 'D']]
    assert cases[f'pool{number}_strict_positive085']['groups'] == []
    assert cases[f'pool{number}_negative_correlation_not_absolute']['groups'] == []


@pytest.mark.parametrize('number', [0, 1])
def test_short_histories_remain_ragged(cases, number):
    assert cases[f'pool{number}_ragged_history_not_aligned']['error'].startswith('ValueError:')


def test_only_66_has_executable_post2020_removal(cases):
    row = cases['pool66_post2020_filter_only']
    assert row['selected'] == ['A', 'B'] and row['pool86_has_executable_age_block'] is False


@pytest.mark.parametrize('name', ['flat_slopes', 'flat_labels'])
def test_original_flat_assignments_remain_incompatible(cases, name):
    row = cases[f'factor43_{name}_2d_to_column_incompatible']
    assert row['error_type'] == 'TypeError' and 'dtype' in row['error']


def test_platform_names_are_not_silently_repaired(cases):
    assert cases['pool_np_platform_injection']['error'] == "NameError: name 'np' is not defined"
    assert cases['factor43_plot_acf_not_imported']['error'] == "NameError: name 'plot_acf' is not defined"


def test_future_ratios_tail_and_startup_nan_are_preserved(cases):
    row = cases['factor43_labels_are_ratios_and_tail_missing']
    assert row['first_ratio'] == row['reference'] == 1.2 and row['tail_nan_rows'] == 90
    scores = cases['factor43_leader20_startup_nan_kept']['first20_scores']
    assert scores[:19] == ['nan'] * 19 and isinstance(scores[-1], float)


@pytest.mark.parametrize('count', [3, 5, 10, 20, 30, 60])
def test_original_normalized_price_slope_against_independent_ols(sources, count):
    values = 100 + np.arange(count) * 2.5
    actual = batch.slope_kernel(sources)(values)
    assert actual == pytest.approx(2.5, abs=1e-12)
    assert actual == pytest.approx(batch.reference_slope(values), abs=1e-12)
    assert batch.slope_kernel(sources)(values * 17) == pytest.approx(actual, abs=1e-12)


def test_reference_pearson_pairwise_missing_and_degeneracy():
    assert batch.reference_correlation([1., 2., np.nan, 4.], [4., 3., 9., 1.]) == pytest.approx(-1)
    assert np.isnan(batch.reference_correlation([1., 1.], [2., 3.]))
    assert np.isnan(batch.reference_correlation([np.nan], [1.]))


def test_dated_pipeline_requires_exact_original_four_asset_shape(sources):
    panel = pd.DataFrame(np.arange(120).reshape(30, 4) + 100.)
    slopes, labels, ns = batch.dated_matrices(sources, panel)
    assert len(slopes) == 6 and len(labels) == 7 and len(ns['test']) == 30
    assert ns['slps_date'].shape == (30, 24) and ns['rets_date'].shape == (30, 28)
    with pytest.raises(ValueError, match='four ordered assets'): batch.dated_matrices(sources, panel.iloc[:, :3])


def test_corrupted_source_blocks_original_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.block(sources, 0, 'liquidity')
