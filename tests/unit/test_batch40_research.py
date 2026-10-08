from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch40 as batch


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
    return {row['case']: row for row in result['cases']}


def test_active_futures_lots_ignores_same_name_in_string(sources):
    ns, _, _ = batch.futures_namespace(sources); g = ns['g']; code = ns['get_future_code']('RB')
    ns['attribute_history'] = lambda *a: pd.DataFrame({'open': [100.]})
    assert ns['get_lots'](50000., 'RB') == 0
    g.ATR[code] = 3.
    assert ns['get_lots'](50000., 'RB') == pytest.approx(50000*.05/(3*10))
    assert ns['get_lots'](50000., 'RB') != pytest.approx(50000*.33/(100*10))


def test_original_array_and_cut_state(sources):
    result = cases(sources); array = result['future_array_accumulates_then_shifts']
    assert array['arrays'] == {'close': [0., 0., 12.], 'open': [0., 0., 10.], 'high': [0., 0., 13.], 'low': [0., 0., 8.]}
    assert not any(array['vars'].values())
    cut = result['future_cut_omits_current_bar_and_dispatches_all']
    assert cut['events'] == [{'RB8888.XSGE': 100., 'I8888.XDCE': 0.}, {'RB8888.XSGE': 100., 'I8888.XDCE': 200.}]
    assert all(not any(row.values()) for row in cut['vars'].values())
    assert result['future_no_night0915_ignores_current']['events'] == []
    assert result['future_date_integer_fault']['error']


def test_original_future_stops_roll_and_synthetic_atr(sources):
    result = cases(sources)
    assert result['future_missing_high_low_fault']['error']
    stop = result['future_stop_counter_not_reset_immediate_reentry_clear']
    assert stop['stopped'] is True and stop['after'] is False and stop['times'] == 40
    assert stop['orders'] == [[['RBTEST', 0], {'side': 'long'}]]
    assert result['future_false_short_low_stays_zero']['low'] is False
    assert result['future_false_short_low_stays_zero']['orders'] == []
    assert result['future_fractional_lots']['lots'] == pytest.approx(2500/30)
    assert result['future_roll_intent_both_sides']['orders'] == [
        [['RBTEST', 0], {'side': 'long'}], [['RBNEW', 2], {'side': 'long'}],
        [['RBTEST', 0], {'side': 'short'}], [['RBNEW', 3], {'side': 'short'}]]
    for row in result['future_original_channel_synthetic_only']['samples']:
        assert row['atr'] == pytest.approx(row['reference']) and row['signal'] == 1


def test_original_grid_rejection_reset_date_and_revaluation(sources):
    result = cases(sources)
    pool = result['grid_declared_duplicate_pool']; assert pool['entries'] == pool['unique']+1
    rejection = result['grid_rejected_buys_record_eleven_intents']
    assert rejection['recorded'] == len(rejection['orders']) == 11
    assert result['grid_initial_record_nonlot_plus_one']['implied_quantity'] % 100 == 1
    reset = result['grid_reset_uses_residual_stock_key']
    assert reset['mapped_symbol'] == reset['requested_reset'] != reset['other_key']
    assert result['grid_unpadded_date_blocks_next_month_sell']['orders'] == []
    assert result['grid_unpadded_date_blocks_next_month_sell']['lexical_later'] is False
    residual = result['grid_half_sell_residual_revalued_current_price']
    assert residual['first']['100.0']['2021-1-1'][0] == residual['second']['100.0']['2021-1-1'][0] == 20000.
    assert [r[1] for r in residual['orders']] == [-20000., -20000.]


def test_original_etf_pool_compatibility_array_and_stop(sources):
    result = cases(sources); pool = result['etf_fund_list_removes_index_mapping']
    assert pool['declared'] == 24 and pool['fund_filtered'] == 23
    assert pool['index_retained'] is False and pool['chip_local_eligible'] is False
    for name in ('etf_original_append_fault', 'etf_original_date_integer_fault', 'etf_stop_date_integer_fault'):
        assert result[name]['error']
    assert result['etf_target_is_array_and_sell_first']['orders'] == [
        {'stock': 'old', 'quantity': 0, 'type': 'int'}, {'stock': 'new', 'quantity': [100], 'type': 'ndarray'}]
    assert result['etf_stop_message_without_closeable_order']['orders'] == []
    assert len(result['etf_stop_message_without_closeable_order']['messages']) == 1
    assert result['etf_stop_sell_only_when_closeable']['orders'] == [['fund', 0]]


def test_original_etf_fourteen_rows_use_twelve_row_gap(sources):
    code, _ = batch.arithmetic_kernel(sources)
    ns = {'price_data': pd.DataFrame({'close': np.arange(100., 114.)}, index=range(-14, 0)), 'ta': batch.talib,
        'g': SimpleNamespace(moment_period=13, ma_period=10, type_num=1), 'total_value': 1000000.}
    exec(code, ns)
    assert ns['moment'] == pytest.approx(12/101*100)
    assert ns['ma_filter'] == 108.5 and ns['ma_status'] == 4.5 and ns['amount'] == 8800


def test_source_corruption_blocks_active_class(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0'*64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.futures_namespace(sources)


def test_source_corruption_blocks_etf_math(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][2]['source_sha256'] = '0'*64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.arithmetic_kernel(sources)


def test_component_backend_and_window_gates(sources, monkeypatch):
    count = 50; frame = pd.DataFrame({'date': pd.bdate_range('2005-01-05', periods=count).strftime('%Y-%m-%d'),
        'close': 100.+np.arange(count)*.1})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch.previous, 'inputs', lambda *a: (frame, []))
    write_json(sources / 'existing-apis.json', {'indicator': batch.indicator()})
    result = batch.compute(sources)
    assert result['windows'] == count-13 and result['strict_positive_boundaries'] == []
    assert max(result['max_differences']) < 1e-10
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    monkeypatch.setattr(batch.previous, 'inputs', lambda *a: (frame.iloc[:13], []))
    with pytest.raises(ValueError, match='Incomplete ETF'): batch.compute(sources)
    monkeypatch.setattr(batch.talib, 'get_compatibility', lambda: 1)
    with pytest.raises(ValueError, match='ATR backend changed'): batch.indicator()


def test_input_corruption_blocks_before_parquet(tmp_path, monkeypatch):
    path = tmp_path / 'index-input.parquet'; path.write_bytes(b'corrupt')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'component_instrument': '000300.SH', 'profiles': {'index-input.parquet': {'file': str(path), 'sha256': '0'*64}, 'calendar-input.parquet': {}}})
    def unexpected(*a, **kw): raise AssertionError('Corrupt input parsed')
    monkeypatch.setattr(batch.pd, 'read_parquet', unexpected)
    with pytest.raises(ValueError, match='Input binding changed'): batch.previous.inputs(tmp_path)


def test_probe_success_is_batch40_unproven_sample(tmp_path, monkeypatch):
    import akshare as ak
    from observe.data import raw
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'futures_zh_daily_sina', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(raw, 'save', save); result = batch.worker('diagnostic-root', tmp_path, 'futures_daily')
    assert result['status'] == 'sample' and result['strict_usable'] is False and result['published'] is False
    assert calls == [['batch40_dependency_probe', 'futures_daily', tmp_path.name]]


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_substitution_rejected(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0'*64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted', 'sha256': '0'*64}]
        if change == 'status': row['status'] = 'accepted'
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises((ValueError, FileNotFoundError)): batch.validate_probes(tmp_path)
