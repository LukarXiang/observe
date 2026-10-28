"""Validate calendar dependency detection and original intraday strategy semantics."""
import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies
from scripts import review_strategy_batch82 as batch


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_all_trade_days_registered_without_changing_trading_status(tmp_path, prefix, legacy):
    source = tmp_path/'source'; source.mkdir()
    code = f'def initialize(context):\n    days = {prefix}get_all_trade_days()\n    order_value("600000.XSHG", 1000)\n'
    if legacy: code += '    print context.current_dt\n'
    (source/'calendar.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path/'data', source)
    row = json.loads((Path(result['output'])/'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_all_trade_days' in row['apis']
    assert row['status'] == '待审查' and row['gaps'] == []
    assert bool(row['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['get_all_trade_days_extra', 'my_get_all_trade_days'])
def test_similar_calendar_names_do_not_register(tmp_path, name):
    source = tmp_path/'source'; source.mkdir()
    (source/'other.py').write_text(f'def initialize(context):\n    {name}()\n', encoding='utf-8')
    result = catalog_strategies(tmp_path/'data', source)
    row = json.loads((Path(result['output'])/'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_all_trade_days' not in row['apis']


@pytest.fixture(scope='module')
def cases():
    return {row['case']: row for row in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_callback_counter_is_not_a_ten_day_rebalance(cases):
    assert cases['64_callback_clock']['trade_callbacks'] == [1, 10, 20]
    assert cases['64_defaults']['initial_if_trade'] is True
    assert cases['64_defaults']['schedules'][1][1]['time'] == 'every_bar'
    assert cases['64_trade_failure_keeps_flag']['if_trade'] is True
    assert cases['64_trade_failure_keeps_flag']['clock'] == 1


def test_historical_pool_uses_current_decision_date(cases):
    assert cases['64_pool_date']['arguments'] == ['000906.XSHG', '2020-02-05']


@pytest.mark.parametrize('label,kind', [('implicit_array', 'NameError'), ('obsolete_sort', 'TypeError'),
    ('float_slice', 'TypeError'), ('short_calendar', 'ValueError'), ('unused_undefined_factor', 'NameError')])
def test_original_reversal_defects_are_retained(cases, label, kind):
    assert cases['64_'+label]['error']['type'] == kind


def test_original_22_day_endpoint_product_and_float_boundaries(cases):
    row = cases['64_float_slice']; calls = row['price_calls']
    assert len(calls) == 44
    assert all(r['frequency'] == '1m' and r['start'] == r['end'] and r['fields'] == ['close'] for r in calls)
    assert all('10:30' in r['start'] for r in calls[::2])
    assert all('15:00' in r['start'] for r in calls[1::2])
    assert len({r['start'][:10] for r in calls}) == 22
    assert row['observed']['rates'] == {'A': 2.**22, 'B': 1., 'C': .5**22}
    assert row['observed']['slice']['start'] == 0.
    assert row['observed']['slice']['stop'] == pytest.approx(.075)
    assert row['original_signal_or_target_generated'] is False


def test_skipna_and_zero_divisor_do_not_become_valid_signals(cases):
    assert cases['64_missing_skipna']['observed']['rates']['A'] == 1.
    assert cases['64_zero_divisor']['observed']['rates']['A'] == 'Infinity'
    assert cases['64_missing_skipna']['original_signal_or_target_generated'] is False


@pytest.mark.parametrize('prefix', ['64', '95'])
def test_immediate_dict_removal_aborts_original_iteration(cases, prefix):
    row = cases[prefix+'_orders_immediate'+('_OLD' if prefix == '64' else '')]
    assert row['error']['type'] == 'RuntimeError'
    assert row['orders'] == [['order_target', 'OLD', 0.]]


def test_reversal_keeps_old_target_and_only_adds_empty_slots(cases):
    assert cases['64_orders_immediate_A']['orders'] == [['order_value', 'B', 100.]]
    assert cases['64_orders_immediate_empty']['orders'] == [['order_value', 'A', 50.], ['order_value', 'B', 50.]]
    assert cases['64_orders_delayed_OLD']['orders'] == [
        ['order_target', 'OLD', 0.], ['order_value', 'A', 100.], ['order_value', 'B', 100.]]
    assert cases['64_empty_targets_sell_only']['orders'] == [['order_target', 'OLD', 0.]]


def test_csv_missing_aborts_initialization_before_scheduling(cases):
    assert cases['95_initialize_missing_csv']['error']['type'] == 'FileNotFoundError'
    assert cases['95_initialize_missing_csv']['schedules'] == []
    row = cases['95_initialize_synthetic_csv']
    assert row['factor_name'] == 'pv_corr' and row['is_in_index'] is None and row['hold_num'] == 200
    assert row['index_names'] == ['date', 'code'] and row['rows'] == 2
    assert row['schedules'] == [['TradeFunc', {'time': '10:30', 'reference_security': '000300.XSHG'}]]


def test_timestamp_versus_date_can_silently_skip_original_factor_day(cases):
    assert [r[1] for r in cases['95_ascending']['orders']] == ['B', 'C', 'A']
    assert cases['95_date_type']['orders'] == [] and cases['95_date_type']['error'] is None
    assert cases['95_absent_date']['orders'] == [] and cases['95_absent_date']['error'] is None
    assert cases['95_retained_empty_level']['error']['type'] == 'KeyError'


@pytest.mark.parametrize('label', ['zero_target_count', 'empty_index_pool'])
def test_empty_cpv_target_divides_by_zero(cases, label):
    assert cases['95_'+label]['error']['type'] == 'ZeroDivisionError'


def test_short_pool_can_select_missing_factors_and_missing_index_codes(cases):
    assert [r[1] for r in cases['95_nan_under200']['orders']] == ['B', 'C', 'A']
    assert [r[1] for r in cases['95_all_nan_under200']['orders']] == ['A', 'B', 'C']
    assert [r[1] for r in cases['95_index_missing_code']['orders']] == ['B', 'MISSING']
    assert [r[1] for r in cases['95_duplicate_codes']['orders']] == ['A', 'B', 'A']


def test_lowest_200_not_highest_or_replaced_with_a_full_pool(cases):
    row = cases['95_over200']; assert len(row['orders']) == 200
    assert [r[1] for r in row['orders']] == [f'S{i:03d}' for i in range(204, 4, -1)]
    assert all(r[2] == .5 for r in row['orders'])
    assert [r[1] for r in cases['95_index_subset']['orders']] == ['B', 'C']


def test_pca_is_separate_from_default_and_fills_nan_before_fit(cases):
    assert cases['95_pca_fillzero']['factor'] == 'pca_pv'
    assert cases['95_pca_fillzero']['error'] is None
    assert [r[1] for r in cases['95_pca_fillzero']['orders']] == ['C', 'A', 'B']
    assert cases['95_missing_column']['error']['type'] == 'KeyError'


@pytest.mark.parametrize('stamp,buy,sell', [
    ('2009-01-01 00:00:00', .003, .004), ('2009-01-01 00:00:01', .002, .003),
    ('2011-01-01 00:00:00', .002, .003), ('2011-01-01 00:00:01', .001, .002),
    ('2013-01-01 00:00:00', .001, .002), ('2013-01-01 00:00:01', .0003, .0013)])
def test_cost_boundaries_are_strict_and_slippage_is_absolute(cases, stamp, buy, sell):
    row = cases['95_cost_'+stamp]
    assert row['commissions'] == [{'buy_cost': buy, 'sell_cost': sell, 'min_cost': 5}]
    assert row['fixed_absolute_slippage'] == [.002]
    assert cases['95_implicit_datetime']['error']['type'] == 'NameError'
    assert cases['95_implicit_datetime']['slippage_called_before_failure'] is True


def test_native_1m_query_is_only_a_five_day_request():
    _, params = batch.expected_request('minute-600519')
    assert params['ndays'] == '5' and 'beg' not in params and 'end' not in params
    assert batch.QUERIES['minute-600519']['parameters']['period'] == '1'


def test_repeat_study_refuses_overwriting(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


def test_functions_only_loader_skips_untrusted_top_level(monkeypatch, tmp_path):
    source = tmp_path/'frozen.source'
    source.write_text('raise RuntimeError("top_level")\n'+''.join(f'def f{i}():\n    return {i}\n' for i in range(6)))
    monkeypatch.setattr(batch, 'read', lambda *a: {'sources': [{}, {'source_copy': str(source)}]})
    monkeypatch.setattr(batch, 'SOURCE_SHA', ('unused', batch.file_sha(source)))
    assert batch.namespace(tmp_path, 1)['f5']() == 5
    source.write_text('@missing_decorator\n'+''.join(f'def f{i}():\n    return {i}\n' for i in range(6)))
    monkeypatch.setattr(batch, 'SOURCE_SHA', ('unused', batch.file_sha(source)))
    with pytest.raises(ValueError, match='Unsafe'): batch.namespace(tmp_path, 1)


def test_tie_order_uses_input_order_and_duplicate_targets_are_retained():
    frame = batch.synthetic_factor((1., 1., 1.), ('Z', 'A', 'B'))
    result = batch.controlled95(batch.DIRECTORY, frame, count=2)
    assert result['orders'] == [['order_target_value', 'Z', 50.], ['order_target_value', 'A', 50.]]
    assert result['not_a_backtest'] is True


@pytest.mark.parametrize('key,value', [('new_trading_backtests', 1), ('original64_factor_windows', 1),
    ('original95_factor_rankings', 1), ('newly_reviewed_sources', 4), ('source_contents', 3),
    ('diagnostic_cases', 0), ('not_a_backtest', False), ('original_strategy_complete', True), ('cost_backtests', ['base'])])
def test_finish_rejects_fabricated_inputs_or_trading_scope(monkeypatch, tmp_path, key, value):
    sha = 'frozen'; code = {'entry.py': sha}
    checked = {'status': 'passed', 'implementation_sha256': code, 'evidence_sha256': {n: sha for n in batch.CORE},
        'commands': [{'command': cmd, 'returncode': 0, 'log': 'log', 'sha256': sha} for cmd in batch.commands()]}
    summary = {'new_trading_backtests': 0, 'original64_factor_windows': 0, 'original95_factor_rankings': 0,
        'newly_reviewed_sources': 3, 'source_contents': 2, 'diagnostic_cases': 48,
        'not_a_backtest': True, 'original_strategy_complete': False, 'cost_backtests': [], key: value}
    documents = {'checked-state.json': checked, 'offline-verification.json': {'implementation_sha256': code}, 'research-summary.json': summary}
    monkeypatch.setattr(batch, 'binding', lambda *a: None)
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'validate_catalog', lambda *a: None)
    monkeypatch.setattr(batch, 'implementation', lambda: code)
    monkeypatch.setattr(batch, 'file_sha', lambda *a: sha)
    monkeypatch.setattr(batch, 'read', lambda path: documents[Path(path).name])
    with pytest.raises(ValueError, match='Research/trading scope'): batch.finish('unused', tmp_path)
