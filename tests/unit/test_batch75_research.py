"""Preserve original ETF source boundaries and immutable study inputs."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch75 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_html_missing_original_site_and_parser_errors(cases):
    assert cases['HTML_two_pairs']['values'] == {'代码': '515950', 'name': 'ETF', 'kind': 'index'}
    assert cases['redacted_URL_never_requested']['local_stub_calls'] == ['http://.../515950.html?spm=search']
    assert cases['HTML_missing_node']['exception'] == 'AttributeError'
    assert cases['HTML_no_separator']['exception'] == cases['HTML_odd_parts']['exception'] == 'IndexError'


def test_actual_pool_cost_and_seed(cases):
    row = cases['actual_pool_parameters_seed_and_schedules']
    assert row['pool'] == list(batch.POOL)
    assert (row['N'], row['M'], row['seed_len'], row['seed_last']) == (18, 600, 599, 598)
    assert [r[1]['time'] for r in row['schedules']] == ['9:30', 'open', '15:30']
    assert row['slips'] == [.001]
    assert row['costs'][0][0]['open_commission'] == row['costs'][0][0]['close_commission'] == .0001
    assert row['costs'][0][0]['min_commission'] == 5


def test_rank_and_nonfinite_original_diagnostics(cases):
    assert cases['ranking_uses_global_pool']['rank'] == [batch.POOL[0]]
    assert cases['implicit_math_missing']['exception'] == 'NameError'
    assert cases['zero_variance_zscore']['finite'] is False
    assert cases['constant_regression_R2']['finite'] is False


@pytest.mark.parametrize('score,signal', [(.7, 'KEEP'), (.7001, 'BUY'), (-.7, 'KEEP'), (-.7001, 'SELL')])
def test_strict_timing_boundaries(cases, score, signal):
    assert cases['timing_boundary_'+str(score)]['signal'] == signal


def test_filter_then_trade_no_replacement_and_keep_adjustment(cases):
    assert cases['new_limit_up_excluded']['retained'] == []
    assert cases['held_limit_up_retained']['retained'] == cases['held_limit_down_retained']['retained'] == [batch.POOL[0]]
    assert all(cases['filter_'+name]['retained'] == [] for name in ('paused', 'is_st', 'name'))
    assert cases['KEEP_still_adjusts']['calls'] == [['adjust', [batch.POOL[0]]]]
    assert cases['510300_forces_SELL']['calls'] == [['close', batch.POOL[0]]]
    assert cases['filtered_top_not_replaced']['calls'] == [['adjust', []]]


def test_missing_key_and_immediate_mutation_failures(cases):
    assert cases['missing_position_key']['exception'] == 'KeyError'
    assert cases['immediate_position_delete_during_adjust']['exception'] == 'RuntimeError'
    assert cases['immediate_position_delete_during_SELL']['exception'] == 'RuntimeError'


def test_partial_order_success_is_not_full_close(cases):
    assert cases['order_state_0']['open'] is False
    assert cases['order_state_50']['open'] is True
    assert cases['order_state_50']['close'] is False
    assert cases['order_state_100']['open'] is cases['order_state_100']['close'] is True


def test_raw_stoploss_float_boundary_is_not_silently_repaired(cases):
    assert cases['stoploss_80.0']['orders'] == []
    assert cases['stoploss_79.99']['orders'] == [['A', 0]]
    assert cases['stoploss_80.01']['orders'] == []
    assert cases['zero_cost_stoploss']['exception'] == 'ZeroDivisionError'


@pytest.mark.parametrize('body', ['@boom()\ndef f(): pass', 'def f(x=boom()): pass', 'def f(x: boom()): pass'])
def test_selector_rejects_eager_definition_execution(body):
    assert not batch.previous.previous.safe_function(ast.parse(body).body[0])


def test_full_source_binding_and_definition_counts():
    batch.validate_sources(batch.DIRECTORY)
    rows = batch.read(batch.DIRECTORY / 'source-reviews/review.json')['sources']
    assert [r['complete_lines_read'] for r in rows] == [168, 290]
    assert [len([n for n in ast.parse(batch.read_source(Path(r['source_copy']))[0]).body if isinstance(n, ast.FunctionDef)]) for r in rows] == [2, 17]


def test_independent_ols_hand_computed_linear_case():
    intercept, slope, r2 = batch.independent_ols(range(18), [2*x+3 for x in range(18)])
    assert (intercept, slope, r2) == (3., 2., 1.)


@pytest.mark.parametrize('name', ['diagnostics.json', 'index-components.parquet', 'ETF-components.parquet', 'component-research.json'])
def test_study_rejects_any_existing_output_before_touching_it(tmp_path, name):
    target = tmp_path / name; target.write_bytes(b'original archive')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert target.read_bytes() == b'original archive'
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def test_ranking_archive_repeat_is_rejected(tmp_path):
    target = tmp_path / 'ranking-components.parquet'; target.write_bytes(b'original archive')
    with pytest.raises(ValueError): batch.candidate_study('unused', tmp_path)
    assert target.read_bytes() == b'original archive'


def test_missing_candidates_do_not_create_synthetic_rankings(tmp_path):
    batch.save(tmp_path / 'reused-ETF.json', {'payload': {'file': 'unused'}})
    batch.save(tmp_path / 'probe-results.json', {'results': [{'case': n, 'status': 'failed'} for n in batch.QUERIES]})
    frame, summary = batch.ranking_components(tmp_path)
    assert frame.empty and summary['missing'] == ['159915', '510500']
    assert summary['not_a_backtest'] is True


def test_ETF_reference_excludes_future_and_blocks_missing(tmp_path):
    path = tmp_path / 'candidate.parquet'; probe = tmp_path / 'probe.json'; batch.save(probe, {'status': 'success'})
    frame = pd.DataFrame({'date': pd.date_range('2026-08-01', periods=65), 'close': np.arange(65, dtype=float)+10})
    frame.loc[29, 'close'] = np.nan; frame.to_parquet(path, index=False)
    batch.save(tmp_path / 'reused-ETF.json', {'probe': str(probe), 'probe_sha256': batch.file_sha(probe),
        'payload': {'file': str(path), 'sha256': batch.file_sha(path)}})
    result, summary = batch.ETF_components(tmp_path)
    assert summary['after_cutoff'] == 5
    assert summary['blocked_windows'] == 29
    assert len(result) == 3 and result.history_end.max() == '2026-09-29'


def test_probe_identity_and_version_are_explicit():
    assert batch.QUERIES == {'159915': 'sz159915', '510500': 'sh510500'}
    assert batch.expected_request('159915') == 'https://finance.sina.com.cn/realstock/company/sz159915/hisdata_klc2/klc_kl.js'
