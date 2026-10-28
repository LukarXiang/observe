import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, read_source
from scripts import review_strategy_batch69 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.parsed_source(read_source(p)[0], i) for i, p in enumerate(paths)]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_all_active_definitions_and_original_line_numbers(monkeypatch):
    trees = originals(monkeypatch)
    assert [len(batch.definitions(t)) for t in trees] == [13, 11]
    assert [t.body[0].lineno for t in trees] == [10, 12]
    for name in batch.SOURCES:
        assert REVIEWS[name]['review_status'] == '人工审查完成'
        assert REVIEWS[name]['status'] == '待数据'
        assert REVIEWS[name]['implementations'] == []


def test_original_diagnostics_preserve_economic_and_compatibility_defects(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused'))
    cases = {row['case']: row for row in result['cases']}
    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert result['real_historical_operand_windows'] == 0
    assert len(cases) == 28
    assert [r['signal'] for r in cases['adaptive_shared_reentry_blocks_two_products']['captured_intents']] == [0, 0]
    assert cases['adaptive_stop_counter100_immediately_unblocks']['reentry_long'] is False
    assert cases['adaptive_stop_equal_4ATR_no_order']['orders'] == []
    assert cases['adaptive_empty50_accesses_last_before_length_guard']['error'] == 'KeyError'
    assert cases['momentum_15rows_10th_from_end_is_9_intervals']['elapsed_intervals'] == 9
    assert cases['momentum_len5_passes_len1_guard_then_KeyError']['error'] == 'KeyError'
    assert cases['momentum_zero_signal_keeps_position_no_stop']['orders'] == []
    assert not cases['momentum_zero_signal_keeps_position_no_stop']['active_trailing_stop_definition']
    assert cases['momentum_reverse_None_orders_records_high_price']['orders'][1]['args'][1] == 2.5
    for prefix in ('adaptive', 'momentum'):
        assert cases[prefix+'_short_roll_reads_new_missing_position']['error'] == 'KeyError'
        assert len(cases[prefix+'_mixed_roll_stops_after_long_intents']['orders']) == 2
        assert cases[prefix+'_roll_None_orders_still_changes_mapping']['orders'][1]['args'] == ['CU_NEW', 0]
        assert cases[prefix+'_zero_numpy_ATR_infinite_lot_intent']['nonfinite_lot']
        assert cases[prefix+'_expiry_returns_entire_callback']['history_requests'] == []


def test_KAMA_fixture_uses_original_TALib_seed_and_window(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused'))
    case = next(row for row in result['cases'] if row['case'] == 'adaptive_50row_KAMA15_ATR15_last15_STDDEV_independent')
    assert case['ATR'] == 3.
    assert case['last_KAMA_difference'] == pytest.approx(1., abs=1e-7)
    assert case['filter'] == pytest.approx((224./12.)**.5, rel=1e-6)
    assert case['filter_multiplier'] == .3


@pytest.mark.parametrize('code', [
    '@unsafe\ndef f():\n    pass',
    'def f(value=open("secret")):\n    pass',
    'def f(*, value=open("secret")):\n    pass',
])
def test_selected_rejects_eager_execution(monkeypatch, code):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(code))
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected(Path('unused'), 0, ['f'], {})


def test_momentum_comment_quantity_definition_does_not_override_active_function(monkeypatch):
    tree = originals(monkeypatch)[1]
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_lots']
    assert len(defs) == 1
    assert 'g.ATR' in ast.unparse(defs[0])
    trade = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'Trade')
    assert 'Trailing' not in ast.unparse(trade)


def test_worker_failure_is_archived_and_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    def broken(case): raise ValueError('Explicit provider failure')
    monkeypatch.setattr(batch, 'invoke_api', broken)
    result = batch.worker(tmp_path, 'rb_daily')
    assert result['status'] == 'failed' and 'Explicit provider failure' in result['error']
    assert result == batch.read(tmp_path / batch.PROBE_FOLDER / 'rb_daily/result.json')
    assert result['files'] == result['responses'] == []
    assert not result['published'] and not result['strict_usable']
    with pytest.raises(FileExistsError): batch.worker(tmp_path, 'rb_daily')


def test_api_evidence_survives_JSON_freeze():
    try:
        import akshare  # noqa: F401
    except ImportError:
        pytest.skip('Optional data interface dependencies are not installed')
    evidence = batch.api_evidence()
    assert evidence == json.loads(json.dumps(evidence, allow_nan=False))


def numeric_fixture():
    close = np.arange(60, dtype=float)+100.
    return pd.DataFrame({'date': pd.date_range('2026-07-01', periods=60), 'open': close,
        'high': close+2, 'low': close-1, 'close': close})


def test_numeric_component_preserves_50_15_windows_9_interval_lag_and_filter():
    values, summary = batch.numeric_components(numeric_fixture(), 'synthetic')
    adaptive = values[values.model.eq('adaptive50')]; momentum = values[values.model.eq('momentum15')]
    assert len(adaptive) == 11 and len(momentum) == 46
    assert adaptive.raw_direction.eq(0).all()
    assert momentum.raw_direction.eq(1).all()
    assert momentum.close.sub(momentum.comparison_close).eq(9).all()
    assert values.atr.eq(3.).all()
    assert summary['independent_numeric_comparisons'] == 68


def test_invalid_OHLC_window_is_blocked_without_interpolation():
    frame = numeric_fixture(); frame.loc[0, 'close'] = np.nan
    values, summary = batch.numeric_components(frame, 'synthetic')
    assert summary['invalid_OHLC_rows'] == 1 and summary['invalid_windows_blocked'] == 2
    assert len(values) == 55


def test_cutoff_prevents_later_data_changing_earlier_components():
    frame = numeric_fixture(); expected, _ = batch.numeric_components(frame, 'synthetic')
    extra = frame.iloc[[0]].copy(); extra['date'] = pd.Timestamp('2026-10-08'); extra['close'] = 999999.
    actual, summary = batch.numeric_components(pd.concat([frame, extra], ignore_index=True), 'synthetic')
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    assert summary['rows_after_cutoff_excluded'] == 1
