"""Independent expectations for the original isolated all-weather arithmetic."""
import math
from pathlib import Path

import pytest

from scripts import verify_strategy_batch89 as batch


@pytest.fixture
def source():
    if not batch.stage.DIRECTORY.exists():
        pytest.skip('Restore SHA-pinned original source stage')
    path = batch.callbacks.source_path()
    if not path.exists():
        pytest.skip('Restore SHA-pinned all-weather original source')
    return path


@pytest.fixture
def rows(source):
    result = batch.network_disabled(lambda: batch.risk_diagnostics(source))
    assert result['synthetic_only'] and result['not_a_backtest']
    assert result['helper_daily_returns_synthetically_injected']
    assert result['original_full_weight_function_executed'] and result['synthetic_price_return_API_adapter']
    assert not result['original_prices_or_pandas_returns_computed']
    assert not result['real_ES_or_original_weights_computed']
    assert result['return_semantics'] == 'pending_D-allweather-fill'
    assert result['case_count'] == 29
    return {r['case']: r for r in result['cases']}


def test_ES99_selects_two_of_166_but_divides_by_120_probability(rows):
    case = rows['ES_confidence_2.58']
    assert case['calendar_return_rows'] == 166 and case['lag'] == 120
    assert case['selected_count_from_equal_operands'] == 2
    assert case['actual'] == pytest.approx(.02/1.2)
    assert case['actual'] != pytest.approx(.01)


@pytest.mark.parametrize('confidence,count', [(1.96, 7), (2.06, 5), (2.18, 4), (2.34, 3), (5.0, 1), (0.0, 7)])
def test_literal_probability_float_boundaries_preserved(rows, confidence, count):
    assert rows[f'ES_confidence_{confidence}']['selected_count_from_equal_operands'] == count


def test_ES95_float_boundary_has_seven_operands_not_nominal_six(rows):
    case = rows['ES_confidence_1.96']
    assert float.fromhex(case['denominator_hex']) > 6.
    assert case['actual'] == pytest.approx(.07/6.)


def test_first_stock_and_dictionary_order_ignore_nominal_portfolio_weights(rows):
    assert rows['ES_only_first_ratio_key']['actual'] == pytest.approx(.03/1.2)
    assert rows['ES_key_order_changes_operand']['actual'] == pytest.approx(.3/1.2)
    assert rows['risk_stock_list_uses_first_only']['ratio'] == {batch.stage.POOL[0]: 1}


def test_short_and_empty_operands_do_not_invent_extra_data(rows):
    assert rows['ES_short_array_still_divides_by_lag_probability']['actual'] == pytest.approx(.03/6.)
    assert rows['ES_empty_ratio_no_history']['actual'] == 0.
    assert rows['risk_empty_class']['ratio'] == {}
    assert rows['weights_empty_class_no_division']['actual'] == {}


def test_zero_nan_negative_infinite_risks_retain_original_behavior(rows):
    assert rows['risk_all_zero']['actual'] == rows['risk_nan_is_reset_to_zero']['actual'] == 0.
    assert rows['risk_all_positive_is_negative']['actual'] < 0.
    assert rows['risk_infinite_is_retained']['actual_hex'] == 'inf'
    assert not rows['risk_infinite_is_retained']['trade_usable']
    assert rows['weights_zero_total_raises']['exception'] == 'ZeroDivisionError'


def test_rounded_weight_total_can_exceed_one_and_existing_key_accumulates(rows):
    assert math.isclose(rows['weights_round_total_below_one']['rounded_total'], .999)
    assert math.isclose(rows['weights_round_total_above_one']['rounded_total'], 1.001)
    assert rows['weights_existing_key_accumulates']['actual'] == {batch.stage.POOL[0]: .5}
    assert rows['weights_multi_stock_class_mapping_missing']['exception'] == 'KeyError'


def test_risk_source_tamper_is_rejected_before_execution(tmp_path):
    source = tmp_path/'changed.source'; source.write_text('raise RuntimeError("do not execute")', encoding='ascii')
    with pytest.raises(ValueError, match='Unreviewed risk source'):
        batch.helpers(source, {})


def test_isolated_helpers_never_read_provider_prices(source):
    stock = batch.stage.POOL[0]
    ns, calls = batch.helpers(source, {stock: [-.1, .2]})
    assert ns['__fun_get_portfolio_ES']({stock: 1}, '1d', 120, 2.58) == pytest.approx(-.1/1.2)
    assert calls == [{'stock': stock, 'freq': '1d', 'lag': 120}]
    assert 'history' not in ns and 'order_target_value' not in ns and 'jqdata' not in ns
    assert Path(source).is_file()


def test_full_original_weights_run_only_on_synthetic_API_chain(rows):
    case = rows['full_synthetic_four_class_inverse_risk']
    assert case['actual']['weights'] == dict(zip(batch.stage.POOL, [.48, .24, .16, .12], strict=True))
    assert case['actual']['history_stock_calls'] == list(batch.stage.POOL)
    assert len(case['actual']['synthetic_operations']) == 20
    assert case['actual']['synthetic_operations'][:5] == [
        ['resample', 'D'], ['last'], ['pct_change_default'], ['fillna', 0, None, 0], ['iloc', 'all_rows', 0]]
    assert not case['native_pandas_computation']
    assert math.isclose(sum(rows['full_synthetic_round_below_one']['actual']['weights'].values()), .999)
    assert math.isclose(sum(rows['full_synthetic_round_above_one']['actual']['weights'].values()), 1.001)


def test_full_original_zero_infinite_and_negative_risk_paths(rows):
    assert rows['full_synthetic_zero_risk_raises']['actual']['exception'] == 'ZeroDivisionError'
    case = rows['full_synthetic_infinite_risk_nan_weights']
    assert case['actual']['weights'] == {stock: 'nan' for stock in batch.stage.POOL}
    assert case['actual']['warnings'] and not case['trade_usable']
    assert rows['full_synthetic_negative_risks_not_rejected']['actual']['exception'] is None
