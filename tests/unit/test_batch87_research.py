"""Preserve original Chan callbacks, confirmation clocks and incremental-order defects."""
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest

from scripts import review_strategy_batch87 as batch


@pytest.fixture(scope='module')
def cases(): return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_parent_frozen_evidence_cannot_change(monkeypatch):
    original = batch.file_sha; target = batch.previous.DIRECTORY/'diagnostics.json'
    monkeypatch.setattr(batch, 'file_sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='Accepted86 evidence changed'): batch.binding('data')


def test_active_security_and_every_bar(cases):
    row = cases['active_symbol_and_schedules']
    assert row['security'] == '300354.XSHE'
    schedules = [r for r in row['events'] if r[0] == 'schedule']
    assert [r[1] for r in schedules] == ['before_market_open', 'market_open', 'after_market_close']
    assert all(r[2] == {'time': 'every_bar', 'reference_security': '000300.XSHG'} for r in schedules)


def test_original_history_request_and_duplicate_rows(cases):
    row = cases['market_open_duplicate_and_omitted_open']
    assert row['query'][0] == [['300354.XSHE'], {'count': 1, 'unit': '1m', 'fields': ['date', 'open', 'high', 'low', 'close']}]
    assert len(row['bars']) == 2 and row['bars'][0] == row['bars'][1]
    assert row['bars'][0]['open_price'] == 0 and row['bars'][0]['close_price'] == 15.5
    assert cases['none_history_no_reset']['bars'] == cases['initialize_does_not_reset_chan']['bars'] == 2


def test_partial_sma_and_first_value_ema(cases):
    row = cases['indicator_seed_and_rounding']
    assert row['sma'] == [1., 1.5, 2.3333, 4.6667]
    assert row['ema'] == [1., 1.5, 2.75, 5.375]


def test_inclusion_delays_first_fractal(cases):
    assert cases['first_three_no_fx']['fx'] == []
    assert len(cases['no_include_third_fx']['fx']) == 1


@pytest.mark.parametrize('label,expected', [('up_include', (12., 7.5)), ('down_include', (9., 5.))])
def test_inclusion_direction_and_input_copy(cases, label, expected):
    row = cases[label]
    assert len(row['chan']) == 3
    assert (row['chan'][-1]['high_price'], row['chan'][-1]['low_price']) == expected


def test_both_fractals_append_but_only_down_reaches_stroke(cases):
    row = cases['double_fractal_only_last_to_stroke']
    assert [r[3] for r in row['fx']] == ['up', 'down']
    assert [r[3] for r in row['strokes']] == ['down']


@pytest.mark.parametrize('price,quantity', [('10.0', 30000), ('1200.0', 200), ('2000.0', 200), ('6000.0', 0), ('10000.0', 0)])
def test_original_rounding_large_increment_and_zero_orders(cases, price, quantity):
    assert cases['increment_quantity_'+price]['orders'] == [['300354.XSHE', quantity], ['300354.XSHE', -quantity]]
    if price == '10.0': assert cases['increment_quantity_'+price]['amount'] == 300000.


def test_invalid_signal_has_no_order_reversal_or_dedup(cases):
    row = cases['duplicate_signal_and_invalid_no_reversal']
    assert row['last_orders'] == [['300354.XSHE', 30000], ['300354.XSHE', 30000]]
    assert len(row['invalid']) == 1 and row['buy'] == 7
    assert cases['zero_price_appends_before_error']['buy'] == 8


def test_short_distance_stroke_correction_drops_last_stroke(cases):
    row = cases['stroke_correction_short_distance_pops']
    assert len(row['strokes']) == 1 and row['strokes'][0][0] == 22


def test_three_stroke_correction_self_distance_drops_last(cases):
    row = cases['stroke_correction_three_strokes_self_distance_pops']
    assert len(row['strokes']) == 2 and row['strokes'][-1][0] == 22 and row['strokes'][-1][4] == 9


def test_signal_pivot_time_differs_from_confirmation(cases):
    row = cases['trend_signal_confirmation_not_pivot_date']; signal = row['signals'][0]
    assert signal[2] == 'S1' and signal[0] < signal[3]
    assert row['orders'] == [['300354.XSHE', -27300]]


def test_macd_resets_seed_per_segment(cases):
    assert cases['macd_segment_seed_and_reverse_dates']['reverse'] == 0
    ns = batch.namespace(batch.DIRECTORY); prices = np.array([11., 12., 13., 14.])
    assert cases['macd_segment_seed_and_reverse_dates']['value'] == round(sum(ns['MACD'](prices)[2]), 4)


def test_reference_equal_inclusion_and_fractal_boundaries():
    merged = [[10., 5., 0], [11., 6., 1], [12., 7., 2]]
    result, fx = batch.reference_inclusion(merged, [12., 7., 3], True)
    assert result[-1] == [12., 7., 3] and len(result) == 3 and fx == []
    assert merged[-1][2] == 2


def test_source_class_has_no_top_level_platform_import():
    ns = batch.namespace(batch.DIRECTORY)
    assert 'jqdata' not in ns and 'datetime' not in ns and ns['chan'].k_list == []


def test_order_failure_retains_signal_append():
    ns = batch.namespace(batch.DIRECTORY); c = ns['chan']; c.k_list = [batch.bar(ns, 0)]
    def failed(*args): raise RuntimeError('inert rejected order')
    ns['order'] = failed
    with redirect_stdout(StringIO()), pytest.raises(RuntimeError, match='inert rejected order'):
        c.on_buy_sell([c.k_list[-1].datetime, 11., 'B1', c.k_list[-1].datetime])
    assert len(c.buy_list) == 1


def test_independent_rounded_macd_preserves_intermediate_rounding():
    ns = batch.namespace(batch.DIRECTORY); values = np.array([11.135, 12.247, 9.81, 16.02])
    independent = batch.independent_indicators(values)
    for got, expected in zip((ns['SMA'](values), ns['EMA'](values), *ns['MACD'](values)), independent, strict=True):
        np.testing.assert_allclose(got, expected, atol=.00010000001, rtol=0)


def test_actual_instrument_input_missing_minute_cannot_be_replaced():
    doc = batch.read(batch.DIRECTORY/'dependency-inventory.json')
    assert doc['instrument'] == '300354.SZ'
    assert doc['instrument_tables']['bars_1d']['rows'] == 1392
    assert doc['instrument_tables']['bars_5m']['rows'] == doc['instrument_tables']['minute_universe']['rows'] == 0
    assert 'bars_1m' not in doc['registered_tables']


def test_frozen_date32_input_reaches_numeric_stage(monkeypatch):
    def numeric_stage(directory): raise RuntimeError('input_dates_verified')
    monkeypatch.setattr(batch, 'namespace', numeric_stage)
    with pytest.raises(RuntimeError, match='input_dates_verified'): batch.component(batch.DIRECTORY)


def test_sources_two_files_one_content():
    rows = batch.read(batch.DIRECTORY/'sources-frozen.json')
    assert len(rows) == 2 and len({r['source_sha256'] for r in rows}) == 1
    assert all(r['complete_lines_read'] == 582 and not r['top_level_executed'] for r in rows)


def test_supplement_limits_no_blind_retry():
    doc = batch.read(batch.DIRECTORY/'supplement-decision.json')
    assert doc['new_requests'] == doc['new_rows'] == 0 and not doc['published']
    assert 'ndays=5' in doc['decision'] and len(doc['restore_required']) == 4


def test_legacy_probe_nested_schema_is_not_current_coverage():
    doc = batch.minute_capabilities()
    assert doc['historical_host_scan'] == {'n_minutes_ok': 14, 'n_bars_ok': 0}
    assert doc['not_a_backtest'] and 'get_history_transaction_data' in doc['native_methods']
