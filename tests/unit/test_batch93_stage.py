"""Original economic and compatibility boundaries, not real platform factor tests."""
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from scripts import review_strategy_batch93 as stage


def test_original_source_hashes_and_full_definitions():
    for n, count in ((0, 11), (1, 4)):
        ns = stage.namespace(n)
        tree = stage.source_tree(n)
        import ast
        assert len([v for v in tree.body if isinstance(v, (ast.ClassDef, ast.FunctionDef))]) == count
        assert 'initialize' in ns


def test_ROA2_precedence_and_six_selected_scores():
    result = stage.evaluate_fixture(stage.synthetic_operands())
    assert result['basic']['A']['ROA2'] == 100.1
    assert result['scores']['A'] == 6
    # Original ratio rewards a positive ratio change even with negative profits.
    assert result['scores']['B'] == 1
    assert result['scores']['C'] == 2
    assert result['independent_metric_cells'] == 42


def test_all_missing_and_zero_division_cannot_create_positive_scores():
    results = stage.factor_diagnostics()
    assert results[1]['scores']['C'] == results[2]['scores']['C'] == 0
    assert results[1]['basic']['C']['ROA2'] != results[1]['basic']['C']['ROA2']
    assert results[3]['basic']['A']['ROA2'] == 100000.1
    assert results[4]['exception'] == 'ValueError'


def test_original_dataframe_mean_not_half_of_pair():
    ns = stage.namespace(0); ns['set_params'](); factor = ns['F13_Score']()
    data = stage.synthetic_operands(); factor.calc(data)
    assert factor.basic.loc['A', 'DELTA_CATURN'] == 1.
    assert factor.basic.loc['A', 'ROA2'] != (100.+20.)/100.


def test_filters_preserve_dtype_pause_ipo_and_missing_industry_boundaries():
    rows = {r['case']: r for r in stage.filter_diagnostics()}
    assert rows['string_end_date']['selected'] == ['ended']
    assert rows['datetime64_end_date']['selected'] == ['live', 'ended']
    assert rows['date_object_end_date']['selected'] == ['live', 'ended']
    assert rows['paused22_threshold21_and_missing_sum']['selected'] == ['nan_paused', 'paused20']
    assert rows['IPO180_strict_boundary']['selected'] == ['age181']
    assert rows['ST_missing_not_false']['selected'] == ['ok']
    assert rows['industry_exclusion_and_missing']['selected'] == ['industry_missing', 'other']


def test_PB_pipeline_call_order_and_date_argument():
    row = stage.pb_diagnostics()[0]
    assert [r['function'] for r in row['calls']] == ['winsorize_med', 'standardlize', 'neutralize']
    assert row['calls'][-1]['keywords'] == {'how': 'sw_l1', 'date': dt.date(2024, 6, 3)}
    assert not row['real_platform_transformation']


def test_empty_FFScore_target_sells_then_divides_by_zero():
    rows = stage.ff_trade_diagnostics()
    assert rows[0]['order_intents'] == [['target', 'old', 0]]
    assert rows[0]['exception'] == 'ZeroDivisionError'
    assert rows[1]['order_intents'] == [['target', 'old', 0], ['target_value', 'A', 100000.]]
    assert rows[1]['exception'] is None


def test_commission_strict_midnight_boundary_without_schedule_claim():
    rows = stage.commission_diagnostics()
    costs = [row['direct_callback_events'][1][1]['buy_cost'] for row in rows]
    assert costs == [.003, .002, .002, .001, .001, .0003]
    assert not any(r['platform_auto_schedule_proven'] for r in rows)


def test_threefactor_growth_and_momentum_do_not_drive_rank_and_month_number_only():
    rows = {r['case']: r for r in stage.threefactor_selection_diagnostics()}
    assert rows['month_change']['selected'] == [f'S{i:02d}' for i in range(50)]
    assert rows['same_month']['selected'] == rows['same_month_other_year']['selected'] == ['held_prior']
    assert rows['permuted_res_columns']['selected'] == [f'S{i:02d}' for i in range(53, 3, -1)]
    calls = rows['month_change']['calls']
    assert [r['function'] for r in calls] == ['get_factor_values']+['standardlize', 'winsorize']*4
    assert all(r['keywords']['qrange'] == [.05, .93] for r in calls if r['function'] == 'winsorize')


def test_threefactor_cash_divisor_missing_keys_and_retained_positions():
    rows = {r['case']: r for r in stage.threefactor_order_diagnostics()}
    assert rows['zero_default_positions']['order_intents'] == [['target', 'old', 0], ['value', 'A', 200.]]
    assert rows['plain_dict_positions']['exception'] == 'KeyError:A'
    assert rows['same_month_no_orders']['order_intents'] == []
    assert rows['sell_updates_cash_before_read']['order_intents'][-1] == ['value', 'A', 300.]


def test_immutable_writer_and_nan_receipt_encoding(tmp_path):
    p = tmp_path/'receipt.json'
    stage.save_new(p, {'nan': np.nan})
    assert stage.read(p) == {'nan': None}
    with pytest.raises(FileExistsError): stage.save_new(p, {})


def test_stage_rejects_existing_destination_before_any_binding_read(tmp_path):
    with pytest.raises(ValueError, match='already exists'):
        stage.stage('data', tmp_path)


def test_offline_callbacks_exact_and_network_forbidden():
    import socket
    with stage.no_network():
        with pytest.raises(AssertionError, match='network forbidden'):
            socket.create_connection(('example.invalid', 80))
        one = stage.encoded(stage.diagnostics())
        two = stage.encoded(stage.diagnostics())
    assert one == two
    assert stage.read_source(Path('repo/量化策略源代码')/stage.SOURCES[1])[0]
