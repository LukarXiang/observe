from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch39 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def test_accepted_input_directory_matches_receipt():
    if not batch.RECEIPT.is_file(): pytest.skip('Restore accepted receipt')
    receipt = batch.read(batch.RECEIPT)
    assert receipt['status'] == 'ok'
    assert batch.ACCEPTED == Path(receipt['checks']['commands'][0]['log']).parent


def test_atr_original_restarts_twenty_row_wilder_seed(sources):
    ns = {'tb': batch.talib}; batch.selected(sources, 2, ('ATR',), ns)
    close = np.arange(100., 140.); high = close+2; low = close-2
    value, unit, amount = ns['ATR'](high[-20:], low[-20:], close[-20:], 1000000.)
    assert value == 4. and unit == 2500. and amount == 347500.
    high[-3] += 20.
    actual = ns['ATR'](high[-20:], low[-20:], close[-20:], 1000000.)
    assert actual[0] == pytest.approx(batch.reference_atr(high[-20:], low[-20:], close[-20:]))
    assert actual[0] > value


def test_indicator_change_blocks(monkeypatch):
    monkeypatch.setattr(batch.talib, 'get_unstable_period', lambda *a: 1)
    with pytest.raises(ValueError, match='ATR backend changed'): batch.indicator()


def test_original_rsrs_algebra_uses_last_1100_population_std(sources):
    code, _ = batch.score_kernel(sources)
    from types import SimpleNamespace
    values = list(np.arange(1200., dtype=float))
    ns = {'g': SimpleNamespace(ans=values, M=1100), 'np': np, 'beta': 2., 'r2': .5}
    exec(code, ns)
    expected = (1199.-649.5)/math.sqrt((1100**2-1)/12)
    assert ns['zscore_rightdev'] == pytest.approx(expected)
    assert len(ns['section']) == 1100 and ns['section'][0] == 100.


def test_original_grid_and_cost_state_diagnostics(sources):
    result = batch.diagnostics(sources)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    cases = {r['case']: r for r in result['cases']}
    config = cases['grid84_configuration']
    assert config['cash'] == 10000. and config['buy'] == .9 and config['sell'] == 1.
    assert config['step'] == pytest.approx(.1)
    strict = [r for r in result['cases'] if r['case'] == 'grid84_strict_price']
    assert strict[0]['orders'] == strict[1]['orders'] == []
    assert strict[2]['orders'] == [['value', '512900.XSHG', 10000.]]
    assert strict[3]['orders'] == [['value', '512900.XSHG', -10000.]]
    moved = cases['grid84_unrelated_cash_changes_grid']
    assert moved['orders'] == [] and moved['buy'] == pytest.approx(.8) and moved['sell'] == pytest.approx(.9)
    assert cases['grid84_date_integer_fault']['error'] and cases['grid44_date_integer_fault']['error']
    assert cases['grid44_configuration']['max_net'] == 5
    assert cases['grid44_none_initial_order_fault']['error']
    assert cases['grid44_partial_held_adds_layer']['amounts'] == [100.]
    assert cases['grid44_partial_held_deletes_layer']['net'] == {}
    assert cases['grid44_partial_held_deletes_layer']['position_keys'] == ['510500.XSHG']
    sold = cases['grid44_sell_uses_first_amount_nonlot_quantity']
    assert sold['orders'][0][0] == 'quantity' and sold['orders'][0][2] < -20000 and sold['orders'][0][2] % 100 != 0
    assert sold['amounts'] == [20000.] and sold['net'] == {'510500.XSHG': 1}
    assert cases['grid44_zero_cash_still_requests_fixed_buy']['orders'] == [['value', '510500.XSHG', 20000]]
    adjusted = cases['grid44_cost_ratio_adjusts_base_only']
    assert adjusted['base'] == .5 and adjusted['amounts_before'] == adjusted['amounts_after'] == [20000.]
    assert cases['grid44_missing_prior_cost_fault']['error']


def test_original_seed_and_intent_state_diagnostics(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    seed = cases['rsrs_original_seed_request_loop_only']
    assert seed['init'] is False and seed['requests'][0][2] == '2021-01-02 14:45:00'
    assert [r['first'] for r in seed['windows']] == [1, 2, 3, 4]
    assert all(r['rows'] == 18 for r in seed['windows'])
    assert cases['rsrs_named_params_integer_fault']['error']
    buy = cases['rsrs_rejected_index_buy_updates_intent']
    assert buy['orders'] == [['value', '000300.XSHG', 380.]] and buy['sys'] == 3 and buy['level'] == 1
    assert cases['rsrs_add_inclusive_boundary']['level'] == 4
    assert cases['rsrs_reduce_inclusive_boundary']['level'] == 3
    assert cases['rsrs_strict_buy_threshold']['orders'] == cases['rsrs_strict_sell_threshold']['orders'] == []
    assert cases['rsrs_clear_always_targets_index']['orders'] == [['target', '000300.XSHG', 0], ['target', '000300.XSHG', 0]]
    assert cases['rsrs_empty_positions_leave_stale_intent']['sys'] == 10
    assert cases['rsrs_fractional_unit_sys_stays_zero']['sys'] == 0 and cases['rsrs_fractional_unit_sys_stays_zero']['level'] == 6
    assert cases['rsrs_zero_atr_nonfinite_unit']['unit_is_finite'] is False


def test_source_corruption_blocks_kernel(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.score_kernel(sources)


def test_input_corruption_blocks_before_parquet_read(tmp_path, monkeypatch):
    path = tmp_path / 'index-input.parquet'; path.write_bytes(b'changed')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'component_instrument': '000300.SH', 'profiles': {
            'index-input.parquet': {'file': str(path), 'sha256': '0' * 64}, 'calendar-input.parquet': {}}})
    def unexpected(*a, **kw): raise AssertionError('Corrupted input was parsed')
    monkeypatch.setattr(batch.pd, 'read_parquet', unexpected)
    with pytest.raises(ValueError, match='Input binding changed'): batch.inputs(tmp_path)


def test_component_counts_and_json_types(sources, monkeypatch):
    count = 1120; x = np.arange(count); low = 100.+x*.01+np.sin(x/5)
    frame = pd.DataFrame({'date': pd.bdate_range('2005-01-05', periods=count).strftime('%Y-%m-%d'),
        'low': low, 'high': low+2.+.05*np.sin(x/7), 'close': low+1.})
    monkeypatch.setattr(batch, 'inputs', lambda *a: (frame, [])); monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    write_json(sources / 'existing-apis.json', {'indicator': batch.indicator()})
    result = batch.compute(sources)
    assert result['atr_windows'] == count-19 and result['regression_windows'] == count-17
    assert result['rsrs_algebra_windows'] == count-1116 and result['rsrs_boundaries'] == []
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result['atr_max_differences'][0] < 1e-10 and result['rsrs_max_difference'] < 1e-10
    monkeypatch.setattr(batch, 'inputs', lambda *a: (frame.iloc[:100], []))
    with pytest.raises(ValueError, match='Incomplete ATR/RSRS'): batch.compute(sources)


def test_successful_probe_archives_only_unproven_batch39_sample(tmp_path, monkeypatch):
    import akshare as ak
    from observe.data import raw
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_sina', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(raw, 'save', save); result = batch.worker('diagnostic-root', tmp_path, 'fund512900')
    assert result['status'] == 'sample' and result['published'] is False and result['strict_usable'] is False
    assert calls == [['batch39_dependency_probe', 'fund512900', tmp_path.name]]


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
