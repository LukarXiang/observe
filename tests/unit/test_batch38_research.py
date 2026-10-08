from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch38 as batch


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


def test_complete_original_trend_arithmetic_and_price_scale(sources):
    close = np.arange(100., 200.); high = close[-30:]; volume = np.ones(180)
    actual, flags = batch.original_trend(sources, close, high, volume)
    expected, ref_flags = batch.reference_trend(close, high, volume)
    assert actual == pytest.approx(expected) and flags == ref_flags == [True] * 7
    assert actual[:6] == pytest.approx([149.5, 169.5, 184.5, 1., 100., 1.])
    scaled, scaled_flags = batch.original_trend(sources, close*2, high*2, volume)
    assert scaled[3] == 2*actual[3] and scaled[3]/scaled[4] == actual[3]/actual[4]
    assert scaled_flags == flags


def test_reference_flags_are_native_json_booleans():
    close = np.arange(100., 200.)
    flags = batch.reference_trend(close, close[-30:], np.ones(180))[1]
    assert all(type(flag) is bool for flag in flags)
    assert json.loads(json.dumps(flags)) == flags


def test_component_correction_only_normalises_reference_flags(tmp_path):
    original = {'stocks': [{'boundaries': [{'reference_flags': [True, 'True', 'False'], 'actual': [1.]}]}]}
    fixed, count = batch.normalised_components(original)
    assert count == 2 and original['stocks'][0]['boundaries'][0]['reference_flags'] == [True, 'True', 'False']
    assert fixed == {'stocks': [{'boundaries': [{'reference_flags': [True, True, False], 'actual': [1.]}]}]}
    write_json(tmp_path / 'component-research.json', original); write_json(tmp_path / 'component-research.corrected.json', fixed)
    proof = {'original_sha256': file_sha(tmp_path / 'component-research.json'),
        'corrected_sha256': file_sha(tmp_path / 'component-research.corrected.json'), 'changed_boolean_flags': count,
        'before_fix_sha256': {}, 'not_a_backtest': True, 'arithmetic_changed': False}
    write_json(tmp_path / 'component-clarification.json', proof)
    assert batch.component_file(tmp_path).name == 'component-research.corrected.json'
    fixed['stocks'][0]['boundaries'][0]['actual'] = [2.]
    write_json(tmp_path / 'component-research.corrected.json', fixed)
    proof['corrected_sha256'] = file_sha(tmp_path / 'component-research.corrected.json')
    write_json(tmp_path / 'component-clarification.json', proof)
    with pytest.raises(ValueError, match='correction scope'): batch.component_file(tmp_path)


@pytest.mark.parametrize('flag', ['yes', 1, None])
def test_component_correction_rejects_unknown_flags(flag):
    with pytest.raises(ValueError, match='Unknown boundary flag'):
        batch.normalised_components({'stocks': [{'boundaries': [{'reference_flags': [flag]}]}]})


@pytest.mark.parametrize('kind', ['short_close', 'short_high', 'short_volume', 'nan', 'zero_close', 'negative_volume'])
def test_incomplete_or_unknown_trend_operands_blocked(sources, kind):
    close = np.arange(100., 200.); high = close[-30:]; volume = np.ones(180)
    if kind == 'short_close': close = close[:-1]
    if kind == 'short_high': high = high[:-1]
    if kind == 'short_volume': volume = volume[:-1]
    if kind == 'nan': close[0] = np.nan
    if kind == 'zero_close': close[0] = 0
    if kind == 'negative_volume': volume[0] = -1
    with pytest.raises(ValueError, match='Invalid complete trend'): batch.original_trend(sources, close, high, volume)


def test_original_volatility_is_sample_std_after_three_sigma_clipping(sources):
    ns = {'np': np, 'math': math}; batch.selected(sources, 2, ('get_volatility',), ns)
    close = np.exp(np.r_[np.zeros(39), 1.])
    actual = ns['get_volatility'](pd.DataFrame({'close': close}), False)
    assert actual == pytest.approx(batch.reference_volatility(close), abs=1e-10)
    assert actual < np.std(np.r_[np.zeros(38), 1.], ddof=1)*math.sqrt(250)*100
    assert ns['get_volatility'](pd.DataFrame({'close': np.ones(40)}), False) == 0.


def test_original_downside_branch_zeros_positive_returns(sources):
    ns = {'np': np, 'math': math}; batch.selected(sources, 2, ('get_volatility',), ns)
    close = np.exp(np.arange(40)*.01)
    assert ns['get_volatility'](pd.DataFrame({'close': close}), True) == 0.


def test_diagnostics_preserve_actual_factor_and_fill_direction_and_orders(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['trend_cross_stock_ffill_direction']['after_b'] == [1., 4.]
    assert cases['trend_removed_fillna_method']['error'] and cases['trend_datetime_column99_fault']['error']
    assert cases['trend_sell_adjust_buy_equal_total']['orders'] == [['old', 0], ['keep', 500.], ['new', 500.]]
    assert cases['trend_empty_targets_clear']['orders'] == [['old', 0], ['keep', 0]]
    assert cases['alpha022_actual_call_removed_order']['calls'] == [['2021-01-01', '000300.XSHG']]
    assert 'order' in cases['alpha022_actual_call_removed_order']['error']
    assert cases['alpha_sell_then_new_cash_only']['orders'] == [['target', 'old', 0], ['value', 'a', 400.], ['value', 'b', 400.]]
    assert cases['alpha_no_new_targets_cash_zero']['cash'] == 0 and cases['alpha_no_new_targets_cash_zero']['orders'] == []
    assert cases['alpha_current_pause_filter']['selected'] == ['a']


def test_diagnostics_roundtrip_without_implicit_date_cast(sources):
    result = batch.diagnostics(sources)
    assert json.loads(json.dumps(result)) == result


def test_balance_missing_positions_and_friday_pool_retained(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['balance_missing_target_ignored']['result'] is False
    assert cases['balance_sell_overweight_first']['orders'] == [['a', 500.], ['b', 500.]]
    assert cases['balance_monday_with_positions_skips']['history_calls'] == 0
    friday = cases['balance_friday_original_pool_weights']
    assert friday['weights'] == [15., 20., 2., 15., 7.5, 4., 25., 20.]
    assert friday['requests'][0]['kwargs']['security_list'] == ['161005.XSHE', '163412.XSHE', '511010.XSHG', '513100.XSHG',
        '513500.XSHG', '518880.XSHG', '159928.XSHE', '512010.XSHG']
    assert friday['requests'][0]['kwargs']['fq'] == 'post' and friday['position_sum'] == pytest.approx(1.)
    assert cases['wave_empty_np_NaN_removed']['error']
    assert cases['balance_zero_wave_invalid_weights'] == {'case': 'balance_zero_wave_invalid_weights', 'zero_waves': 8,
        'nan_weights': 8, 'need_balance': False}


def test_source_corruption_blocks_trend_kernel(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.trend_kernel(sources)


def test_input_corruption_blocks_before_parquet_read(tmp_path):
    path = tmp_path / 'price-input.parquet'; path.write_bytes(b'changed')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'file': str(path), 'sha256': '0' * 64,
        'not_a_backtest': True, 'platform_equivalent': False})
    with pytest.raises(ValueError, match='Input binding differs'): batch.inputs(tmp_path)


def test_compute_counts_warmups_and_excludes_known_pauses(sources, monkeypatch):
    count = 182; prices = np.arange(100., 100.+count)
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=count).strftime('%Y-%m-%d'), 'instrument': 'sample',
        'close_adj': prices, 'high_adj': prices*1.001, 'back_factor': 2., 'volume': 100., 'is_trading': True, 'adjustment_status': 'usable'})
    frame.loc[0, 'is_trading'] = False
    path = sources / 'price-input.parquet'; frame.to_parquet(path, index=False)
    write_json(sources / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'file': str(path), 'sha256': file_sha(path),
        'rows': count, 'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max(), 'pool': ['sample'],
        'not_a_backtest': True, 'platform_equivalent': False})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    result = batch.compute(sources); row = result['stocks'][0]
    assert result['excluded_known_paused_rows'] == 1 and row['traded_rows'] == 181
    assert row['trend_windows'] == 2 and row['wave_windows'] == 142 and row['boundaries'] == []
    assert row['trend_first_end_date'] == frame.date.iloc[180]
    assert row['wave_max_difference'] < 1e-10


def test_successful_probe_only_archives_unproven_sample(tmp_path, monkeypatch):
    import akshare as ak
    from observe.data import raw
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_sina', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(raw, 'save', save); result = batch.worker('diagnostic-root', tmp_path, 'lof161005')
    assert result['status'] == 'sample' and result['published'] is False and result['strict_usable'] is False
    assert calls == [['batch38_dependency_probe', 'lof161005', tmp_path.name]]


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
