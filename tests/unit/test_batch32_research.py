from pathlib import Path

import numpy as np
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch32 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original strategy sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def test_original_boll_mixes_25_mean_20_sample_std_and_eight_old_lows(sources):
    code, _ = batch.boll_kernel(sources); values = np.arange(25.) + 100
    row = batch.boll_case(code, values)
    assert row['meandelta'] == np.mean(values) and row['meandelta'] != np.mean(values[-20:])
    assert row['stddev'] == pytest.approx(np.std(values[-20:], ddof=1))
    assert row['stddev'] != pytest.approx(np.std(values[-20:], ddof=0))
    assert row['prev_low'] == min(values[-10:-2])


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_independent_boll_equations_match_original_source(sources, seed):
    code, _ = batch.boll_kernel(sources); values = np.random.default_rng(seed).uniform(50, 100, 25)
    row = batch.boll_case(code, values); expected, entry = batch.reference_boll(values)
    np.testing.assert_allclose([row[n] for n in ('meandelta', 'stddev', 'upperbound', 'lowerbound', 'prev_low')], expected, rtol=1e-12)
    assert bool(row['final_list']) == entry


@pytest.mark.parametrize('values', [[], [1.] * 24, [1.] * 26, [1.] * 24 + [np.nan], [1.] * 24 + [0.]])
def test_invalid_independent_boll_window_blocks(values):
    with pytest.raises(ValueError, match='Invalid BOLL window'): batch.reference_boll(values)


def test_legacy_boll_indices_and_missing_stop_state_are_not_silently_repaired(sources):
    code, _ = batch.boll_kernel(sources); values = np.arange(25.) + 100
    with pytest.raises(KeyError): batch.boll_case(code, values, positional=False)
    with pytest.raises(KeyError): batch.boll_case(code, values, held=True)
    assert batch.boll_case(code, values, held=True, loss=values[-1])['final_list'] == ['component']
    assert batch.boll_case(code, values, held=True, loss=values[-1] + .01)['final_list'] == []


@pytest.mark.parametrize('value,expected', [(14.99, -1), (15., 0), (20., 0), (20.01, 1), (49.99, 1), (50., 0),
    (55., 0), (55.01, -1), (79.99, -1), (80., 0), (85., 0), (85.01, 1), (np.nan, 0)])
def test_original_rsi_strict_boundaries(sources, value, expected):
    ns, calls = batch.trix_namespace(sources, rsi=value)
    assert ns['RS']('component') == expected
    assert calls == [{'indicator': 'RSI', 'date': '2020-01-02', 'parameters': {'N1': 6}}]


def test_original_two_buy_checks_repeat_both_indicators_and_sell_uses_either(sources):
    ns, calls = batch.trix_namespace(sources)
    assert ns['btj1']('component') and ns['btj2']('component')
    assert [r['indicator'] for r in calls] == ['TRIX', 'RSI', 'TRIX', 'RSI']
    assert calls[0]['parameters'] == {'N': 12, 'M': 20}
    ns, _ = batch.trix_namespace(sources, trix=0., matrix=1., rsi=60.)
    assert ns['stj1']('component') is True
    ns, _ = batch.trix_namespace(sources, trix=2., matrix=1., rsi=40.)
    assert ns['stj1']('component') is True


def test_original_diagnostics_preserve_debt_order_slots_and_cash(sources):
    rows = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert rows['bluechip_false_filter_holded_excludes_existing']['stocks'] == ['new']
    assert rows['bluechip_empty_sort_preserves_order']['stocks'] == ['z', 'a']
    assert rows['bluechip_pool_sorted_deduped']['stocks'] == ['a', 'b']
    assert rows['bluechip_negative_slots_slice_without_orders']['allocation_lists'] == [['a', 'b']]
    assert rows['trix_financial_selects_above_median_debt']['stocks'] == ['c']
    assert rows['trix_single_candidate_uses_half_budget']['orders'] == [['a', 10000.]]
    assert rows['trix_partial_sell_deletes_high_state']['remaining'] == {}
    assert rows['boll_price_filter_asymmetric_held_exception']['stocks'] == ['held', 'one', 'hundred']
    assert rows['boll_cash_rounding']['orders'] == [['a', 5000], ['b', 5000]]
    assert rows['boll_share110_equal_allowed']['orders'] == [['a', 1100]]
    assert rows['boll_share_below110_skips']['orders'] == []
    assert rows['boll_rejected_sell_occupies_slot']['orders'] == [['old', 0]]
    assert rows['boll_current_upper_used_for_past_close_profit_exit']['kept'] is False
    assert rows['boll_filtered_candidate_retains_pretrade_loss']['selected'] == ['component']
    assert rows['boll_filtered_candidate_retains_pretrade_loss']['accepted'] == []
    assert rows['boll_filtered_candidate_retains_pretrade_loss']['loss_state'] == {'component': 90.}


def test_source_drift_blocks_before_ast_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.boll_kernel(sources)


def test_frozen_input_changed_bytes_block_before_parquet_read(tmp_path):
    path = tmp_path / 'price-input.parquet'; path.write_bytes(b'changed-input')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'pool': list(batch.previous.stock_binding.INSTRUMENTS), 'file': str(path), 'sha256': '0' * 64})
    with pytest.raises(ValueError, match='Input binding changed'): batch.inputs(tmp_path)


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
