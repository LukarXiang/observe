"""Original CSV timing and GP failure semantics, plus safe definition loading."""
import ast

import numpy as np
import pytest

from scripts import review_strategy_batch83 as batch


def test_parent_frozen_evidence_is_rechecked(monkeypatch):
    original = batch.file_sha
    target = batch.previous.DIRECTORY/'diagnostics.json'
    monkeypatch.setattr(batch, 'file_sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='Accepted82 frozen evidence changed'):
        batch.binding('data')


def test_annotation_loader_preserves_body_without_evaluating_types():
    tree = ast.parse('raise RuntimeError("top-level")\ndef f(x: Missing.Type) -> Unknown:\n    return x + 1\n')
    ns = batch.selected_nodes(tree, ['f'], {})
    assert ns['f'](4) == 5 and ns['f'].__annotations__ == {}
    assert tree.body[1].args.args[0].annotation is not None


@pytest.mark.parametrize('code', [
    'def f(x: dangerous()):\n    return x\n',
    'def f(x=dangerous()):\n    return x\n',
    '@dangerous\ndef f(x):\n    return x\n',
    'def f(x):\n    def nested(y=dangerous()):\n        return y\n    return x\n',
])
def test_loader_rejects_definition_side_effects(code):
    called = []
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected_nodes(ast.parse(code), ['f'], {'dangerous': lambda: called.append(True)})
    assert called == []


def test_duplicate_definitions_require_explicit_occurrence():
    tree = ast.parse('def f():\n    return 1\ndef f():\n    return 2\n')
    with pytest.raises(ValueError, match='ambiguous'):
        batch.selected_nodes(tree, ['f'], {})
    assert batch.selected_nodes(tree, ['f'], {}, {'f': 0})['f']() == 1
    assert batch.selected_nodes(tree, ['f'], {}, {'f': 1})['f']() == 2


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_csv_dates_drive_state_and_same_day_exit_wins(cases):
    assert cases['91_defaults']['fast'] == 30 and cases['91_defaults']['slow'] == 100
    assert cases['91_defaults']['to_buy'] is False
    assert cases['91_defaults']['schedules'] == [
        ['get_signal', ['9:00'], {'reference_security': '000300.XSHG'}],
        ['trade', ['open'], {'reference_security': '000300.XSHG'}]]
    assert cases['91_same_day_exit_priority']['to_buy'] is False
    assert cases['91_entry']['to_buy'] is True and cases['91_no_date_keeps_true']['to_buy'] is True
    assert cases['91_exit']['to_buy'] is False and cases['91_no_date_keeps_false']['to_buy'] is False


def test_original_checks_any_long_position_and_submits_only_target_etf(cases):
    assert cases['91_order_True_False']['orders'] == [['value', '510300.XSHG', 100000.]]
    assert cases['91_order_True_True']['orders'] == []
    assert cases['91_order_False_False']['orders'] == []
    assert cases['91_order_False_True']['orders'] == [['quantity', '510300.XSHG', 0]]
    assert cases['91_failed_order_state']['to_buy'] is True


@pytest.mark.parametrize('year,earlier,later', [(2009, .003, .002), (2011, .002, .001), (2013, .001, .0003)])
def test_original_strict_fee_boundaries_and_absolute_slippage(cases, year, earlier, later):
    assert cases[f'91_fee_{year}_exact']['fee']['buy_cost'] == earlier
    assert cases[f'91_fee_{year}_later']['fee']['buy_cost'] == later
    assert cases[f'91_fee_{year}_later']['slippage'] == .002


@pytest.mark.parametrize('case,kind', [('91_missing_csv', 'FileNotFoundError'), ('91_bad_csv_dates', 'AttributeError'),
    ('91_hma_wrong_type', 'ValueError'), ('91_hma_missing_talib', 'NameError'), ('91_netbuy_recursive', 'RecursionError'),
    ('40_square_missing_arg', 'TypeError'), ('40_alpha2_last_missing_sigmoid', 'NameError')])
def test_original_errors_not_replaced_by_signals(cases, case, kind):
    assert cases[case]['error'] == kind


def test_flat_operators_cross_stock_boundary(cases):
    assert cases['40_delay']['result'] == [0., 1., 2., 3., 10., 20.]
    assert cases['40_delta']['result'] == [0., 1., 1., 7., 10., 10.]
    assert cases['40_ts_sum_int']['result'] == [0., 0., 6., 15., 33., 60.]


def test_notebook_length_understates_tail_and_mask_is_not_purging(cases):
    assert cases['40_boundary_singleton_tail']['length'] == [2]
    assert cases['40_boundary_singleton_tail']['labels_after_nan_to_num'] == [1., 2., 0.]
    assert cases['40_boundary_repeated_tail']['length'] == [2, 1]
    assert cases['40_boundary_repeated_tail']['labels_after_nan_to_num'] == [1., 2., 0., 0.]


@pytest.mark.parametrize('name', ['ts_sum', 'sma', 'ts_rank', 'ts_argmin', 'ts_argmax', 'ts_min', 'ts_max', 'ts_argmaxmin'])
def test_float_windows_and_missing_rank_window_become_zero_vectors(cases, name):
    assert cases['40_'+name+'_float']['result'] == [0.] * 6
    assert cases['40_ts_rank_int']['result'] == [0.] * 6
    assert cases['40_'+name+'_nonconstant']['result'] == [0.] * 6


def test_corr_omits_last_point_and_stddev_uses_mean_of_all_n(cases):
    assert cases['40_corr_int']['result'][:2] == ['NaN', 'NaN']
    assert cases['40_corr_int']['dtype'] == 'object'
    np.testing.assert_allclose(cases['40_corr_int']['result'][2:], [-1.] * 4)
    assert cases['40_corr_float']['result'] == [0.] * 6
    np.testing.assert_allclose(cases['40_stddev_int']['result'][2], 1.)
    np.testing.assert_allclose(cases['40_stddev_uses_all_window_values']['result'][-1], np.std([1., 2., 3., 10., 20., 30.], ddof=1))


def test_fitness_ignores_weights_and_can_be_nonfinite(cases):
    assert cases['40_metric_weights_ignored']['ones'] == cases['40_metric_weights_ignored']['zeros'] == 13.
    assert cases['40_metric_small']['value'] == 'NaN'
    assert cases['40_metric_zero_low']['value'] == 'Infinity'


def test_first_alpha2_not_later_definition_has_hardcoded_inverse_high_formula(cases):
    assert cases['40_alpha2_first']['result'] == [[9.], [1.]]


def test_reference_scope_retains_missing_observations_without_original_training():
    frame, result = batch.compute(batch.DIRECTORY)
    assert len(result['reference_stocks']) == 10
    assert result['scopes'][0]['rows'] == 500 and result['scopes'][0]['end'] == '2020-03-20'
    assert result['scopes'][1]['rows'] == 5280 and result['scopes'][1]['start'] == '2005-01-05'
    assert result['scopes'][1]['end'] == '2026-09-29'
    assert frame.loc[frame.valid, ['value', 'independent']].notna().all().all()
    np.testing.assert_allclose(frame.loc[frame.valid, 'value'], frame.loc[frame.valid, 'independent'], rtol=1e-12, atol=1e-14)
    assert frame.loc[~frame.valid, 'value'].isna().all()
    assert result['original_training_runs'] == result['original_timing_signals'] == result['new_trading_backtests'] == 0
    assert result['cost_backtests'] == [] and result['not_a_backtest'] is True
