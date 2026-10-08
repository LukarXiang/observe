from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch37 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def test_accepted_directory_matches_receipt():
    if not batch.RECEIPT.is_file(): pytest.skip('Restore accepted receipt')
    receipt = batch.read(batch.RECEIPT)
    assert receipt['status'] == 'ok'
    assert batch.ACCEPTED == Path(receipt['checks']['commands'][0]['log']).parent


def test_rps_original_uses120_rows119_intervals_and_dynamic_scale(sources):
    code, sha = batch.ratio_expression(sources)
    close = np.arange(1., 121.)
    assert eval(code, {'rps_data': {'close': close}}) == 119.
    assert eval(code, {'rps_data': {'close': close / 8.}}) == 119.
    assert len(sha) == 64


def test_original_rps_ranking_caps_and_missing_wizard_are_retained(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['rps_stable_ties_no_ten_cap']['count'] == 15
    assert cases['rps_stable_ties_no_ten_cap']['selected'] == [f's{k:03d}' for k in range(15)]
    assert cases['rps_exclusion_after_rank_strict_half']['selected'] == ['b']
    assert cases['rps_slots_and_duplicate']['orders'] == [['a', 200.], ['b', 250.]]
    assert 'n_day_chg_xiaoyu' in cases['missing_wizard_sell']['error']
    assert cases['rps_sorted_unique_pool']['pool'] == ['a', 'b']


def test_lof_float_full_signal_and_failures_not_repaired(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['lof_all_five_float_branch']['equal_one'] is True
    assert cases['lof_all_five_float_branch']['index'] == 1.
    assert cases['lof_all_five_float_branch']['original_weight_one_fund'] == 1.
    assert cases['lof_removed_append']['weight_before_error'] == 1.
    assert 'append' in cases['lof_removed_append']['error']
    assert cases['lof_empty_pool']['error']
    assert cases['lof_adjust_and_unconditional_cash_target']['orders'] == [['value', 'fund', 300.], ['value', '511880.XSHG', 700.]]
    assert cases['lof_clear_then_cash_target']['orders'] == [['target', 'fund', 0], ['value', '511880.XSHG', 700.]]


def test_chase_collisions_unlimited_total_value_and_pool_are_retained(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['chase_equal_return_overwrites_first']['orders'] == [['b', 1000.]]
    assert cases['chase_multiple_total_value_orders']['orders'] == [['b', 1000.], ['a', 1000.]]
    assert cases['chase_datetime_integer_index_fault']['error']
    assert cases['chase_not_true_limit']['selected'] == ['a']
    assert cases['chase_pool_keeps_duplicates_and_star']['pool'] == ['600001.XSHG', '688001.XSHG'] * 2
    assert cases['chase_pool_keeps_duplicates_and_star']['requests'] == ['000002.XSHG', '399106.XSHE']


@pytest.mark.parametrize('closes,expected', [([100., 109.5, 109.5], False), ([100., 110., 110.], True),
    ([100., 110., 121.], False), ([100., 109.6, 109.6], True)])
def test_pattern_strict_and_not_high_limit(sources, closes, expected):
    ns = {'get_current_data': lambda: {}, 'attribute_history': lambda *a, **kw: pd.DataFrame({'close': closes, 'high_limit': [1., 1., 1.]})}
    batch.selected(sources, 2, ('high_limit_filter',), ns)
    assert bool(ns['high_limit_filter'](['sample'])) is expected


def test_source_corruption_blocks_original_expression(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.ratio_expression(sources)


def test_review_correction_preserves_source_provenance(tmp_path):
    original = {'sources': [{'source_sha256': 'sample', 'review': {'old': True}}], 'catalog': {'catalog_id': 'old'}, 'snapshot': batch.SNAPSHOT}
    corrected = {'sources': [{'source_sha256': 'sample', 'review': {'corrected': True}}], 'catalog': {'catalog_id': 'new'}, 'snapshot': batch.SNAPSHOT}
    a = tmp_path / 'source-reviews/review.json'; b = tmp_path / 'source-reviews/review.corrected.json'
    write_json(a, original); write_json(b, corrected)
    write_json(tmp_path / 'source-clarification.json', {'original_sha256': file_sha(a), 'corrected_sha256': file_sha(b)})
    assert batch.reviewed(tmp_path) == corrected
    corrected['sources'][0]['source_sha256'] = 'substituted'; write_json(b, corrected)
    write_json(tmp_path / 'source-clarification.json', {'original_sha256': file_sha(a), 'corrected_sha256': file_sha(b)})
    with pytest.raises(ValueError, match='source provenance'): batch.reviewed(tmp_path)


def test_unbound_correction_is_rejected(tmp_path):
    write_json(tmp_path / 'source-reviews/review.json', {})
    write_json(tmp_path / 'source-reviews/review.corrected.json', {})
    write_json(tmp_path / 'source-clarification.json', {'original_sha256': '0' * 64, 'corrected_sha256': '0' * 64})
    with pytest.raises(ValueError, match='Review correction changed'): batch.reviewed(tmp_path)


def test_unknown_input_bytes_rejected_before_parquet_read(tmp_path):
    path = tmp_path / 'price-input.parquet'; path.write_bytes(b'changed')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'file': str(path), 'sha256': '0' * 64,
        'not_a_backtest': True, 'platform_equivalent': False})
    with pytest.raises(ValueError, match='Input binding differs'): batch.inputs(tmp_path)


def test_real_operator_counts_exclude_known_pauses_and_use119_intervals(sources, monkeypatch):
    rows = 122; frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=rows).strftime('%Y-%m-%d'),
        'instrument': 'sample', 'close_adj': np.arange(1., rows + 1), 'back_factor': 2., 'is_trading': True, 'adjustment_status': 'usable'})
    frame.loc[0, 'is_trading'] = False
    path = sources / 'price-input.parquet'; frame.to_parquet(path, index=False)
    write_json(sources / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'file': str(path), 'sha256': file_sha(path),
        'rows': rows, 'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max(), 'pool': ['sample'],
        'not_a_backtest': True, 'platform_equivalent': False})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    result = batch.compute(sources); stock = result['stocks'][0]
    assert result['excluded_known_paused_rows'] == 1
    assert stock['traded_rows'] == 121 and stock['rps_windows'] == 2 and stock['pattern_windows'] == 119
    assert stock['rps_first_end_date'] == frame.date.iloc[120]
    assert stock['rps_max_difference'] < 1e-12 and stock['boundaries'] == []


def test_successful_probe_cannot_publish_or_admit_historical_equivalence(tmp_path, monkeypatch):
    import akshare as ak
    from observe.data import raw
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_sina', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(raw, 'save', save)
    result = batch.worker('diagnostic-root', tmp_path, 'lof161903')
    assert result['status'] == 'sample' and result['published'] is False and result['strict_usable'] is False
    assert calls == [['batch37_dependency_probe', 'lof161903', tmp_path.name]]


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
