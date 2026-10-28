"""Original all-weather callback behavior under explicit synthetic API contracts."""
from pathlib import Path

import pytest

from scripts import diagnose_strategy_batch89 as batch


@pytest.fixture
def source():
    if not batch.stage.DIRECTORY.exists():
        pytest.skip('Restore SHA-pinned all-weather source stage')
    path = batch.source_path()
    if not path.exists():
        pytest.skip('Restore original all-weather source payload')
    return path


@pytest.fixture
def results(source):
    return batch.offline_diagnostics(source)


def test_all_cases_are_synthetic_and_return_repair_stays_pending(results):
    assert results['case_count'] == 32
    assert results['not_a_backtest'] and results['synthetic_only'] and results['socket_HTTP_disabled']
    assert results['return_semantics'] == 'pending_D-allweather-fill'
    assert not results['real_ES_or_original_weights_computed']
    assert not results['platform_history_shape_proven'] and not results['platform_immediate_fill_proven']


def test_original_source_rejects_tampering_before_exec(tmp_path):
    path = tmp_path/'changed.source'
    path.write_text('raise RuntimeError("must not execute")', encoding='utf-8')
    with pytest.raises(ValueError, match='Unreviewed callback source'):
        batch.original_definitions(path, {})


def test_literal_callback_cycle_and_strict_deviation(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    assert rows['synthetic_weights_63_callback_cycle']['recalcs'] == [1, 32, 63]
    assert rows['synthetic_weights_63_callback_cycle']['trade_calls'] == 63
    assert rows['initial_zero_anchor_skips_price_move']['return'] is None
    assert rows['strict_15pct_115.0']['return'] is None
    assert rows['strict_15pct_85.0']['return'] is None
    assert rows['strict_15pct_115.01']['return'] is True
    assert rows['strict_15pct_84.99']['return'] is True
    assert rows['untracked_position_anchor_key']['exception'] == 'KeyError'


def test_thresholds_cash_and_profit_are_original_rules(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    stock = batch.stage.POOL[0]
    assert rows['delta_1000_equal_block']['orders'] == []
    assert rows['delta_1010_buy']['orders'] == [[stock, 4000.]]
    assert rows['relative_25pct_and_cash_equal_pass']['orders'] == [[stock, 8000.]]
    for case in ('relative_below_25pct_block', 'cash_below_quarter_block',
                 'price_equals_internal_cost_block', 'price_below_internal_cost_block'):
        assert rows[case]['orders'] == []
    assert rows['sell_does_not_require_profit_or_cash']['orders'] == [[stock, 4000.]]


def test_numpy_zero_target_differs_from_python_float(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    assert rows['zero_target_numpy_holding0']['exception'] is None
    assert rows['zero_target_numpy_holding0']['warnings']
    assert rows['zero_target_numpy_holding1000']['orders'] == [[batch.stage.POOL[0], 0.]]
    assert rows['zero_target_list_holding0']['exception'] == 'ZeroDivisionError'
    assert rows['zero_target_list_holding1000']['exception'] == 'ZeroDivisionError'


def test_special_branch_uses_actual_513100_and_strict_100share_cash(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    assert rows['special_delta_100shares_equal_block']['orders'] == []
    assert rows['special_cash_100shares_equal_block']['orders'] == []
    assert rows['special_cash_above_pass']['orders'] == [['513100.XSHG', 2000.]]
    assert rows['special_sell_no_cash_profit_check']['orders'] == [['513100.XSHG', 0.]]


def test_immediate_position_read_exposes_fill_contract(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    assert rows['order_record_deferred_None_20000.0']['exception'] == 'KeyError'
    assert rows['order_record_deferred_100.0_20000.0']['record']['amount'] == 100.
    assert rows['order_record_immediate_keep_zero_None_20000.0']['record']['amount'] == 2000.
    assert rows['order_record_immediate_keep_zero_100.0_0.0']['record']['amount'] == 0.
    assert rows['order_record_immediate_delete_zero_100.0_0.0']['exception'] == 'KeyError'


def test_original_partial_sale_cost_is_signal_state_not_ledger_pnl(results):
    rows = {r['case']: r['actual'] for r in results['cases']}
    row = rows['record_partial_sale_buy_full_loss']
    assert row['costs'] == [8., 10., 0.]
    assert row['margin'] == row['max_loss'] == -300.
    assert row['buy_times'] == row['loss'] == 1 and row['sell_times'] == 2
    assert rows['reduction_first_extra_holding_not_liquidated']['orders'] == [
        [batch.stage.POOL[0], 20000.], [batch.stage.POOL[1], 30000.]]


def test_harness_refuses_real_risk_history(source):
    h = batch.Harness(source)
    with pytest.raises(ValueError, match='Real risk-history calculation forbidden'):
        h.ns['fun_calc_trade_ratio'](h.context)
    assert len(h.calls) == 1 and h.calls[0]['lag'] == 120
    assert not h.orders


def test_no_platform_import_or_top_level_execution(source):
    h = batch.Harness(source)
    assert 'jqdata' not in h.ns and 'np' not in h.ns and 'math' not in h.ns
    assert batch.FUNCTIONS <= h.ns.keys()
    assert h.g.transactionRecord.keys() == set(batch.stage.POOL)
    assert h.context.portfolio.positions == {}
    assert Path(source).is_file()
