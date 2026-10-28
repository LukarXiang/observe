"""Preserve original seed shifts, KEEP actions, nonstandard execution and gates."""
import ast
import math

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch79 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


@pytest.mark.parametrize('number,slip,commission,minimum', [(0, .004, .0005, 5), (1, .001, .0001, 0)])
def test_actual_costs_override_misleading_comments(cases, number, slip, commission, minimum):
    row = cases[f'{number}_defaults']; cost = row['costs'][0][0]
    assert row['pool'] == list(batch.POOLS[number]) and row['slippage'] == [slip]
    assert cost['open_commission'] == cost['close_commission'] == commission
    assert cost['min_commission'] == minimum and cost['close_tax'] == 0


def test_real_schedules(cases):
    assert cases['0_defaults']['schedules'] == [['trade', {'time': '9:45'}]]
    assert [row[1]['time'] for row in cases['1_defaults']['schedules']] == ['7:00', '9:30', '9:35', 'open', '11:25', '11:27']


@pytest.mark.parametrize('number', [0, 1])
def test_original_ols_and_population_zscore(cases, number):
    assert cases[f'{number}_ols']['values'] == pytest.approx([3., 1.25, 1.], abs=1e-12)
    assert cases[f'{number}_zscore']['value'] == pytest.approx(math.sqrt(1.5))


def test_KEEP_buys_empty_portfolio_but_holding_is_not_rebalanced(cases):
    assert cases['31_empty_KEEP']['orders'] == cases['31_empty_BUY']['orders'] == [[batch.POOLS[0][0], 100.]]
    assert cases['31_held_KEEP']['orders'] == []
    assert cases['31_empty_SELL']['orders'] == []
    assert cases['31_held_SELL']['orders'] == cases['31_empty_pool_KEEP']['orders'] == [[batch.POOLS[0][0], 0]]


@pytest.mark.parametrize('name', ['initial_stock_motion', 'get_rank'])
def test_current_Series_indexing_error_is_not_silently_fixed(cases, name):
    assert cases[f'42_{name}_position_error']['exception'] == 'KeyError'


def test_three_element_rank_sends_numeric_securities_when_fills_delayed(cases):
    assert cases['42_triple_list_numeric_orders']['orders'] == [[batch.POOLS[1][0], 100.], [10., 100.], [2., 100.]]


def test_undefined_stoploss_log_variable_fails_after_order_intent(cases):
    row = cases['42_stoploss_undefined_value']
    assert row['exception'] == 'NameError' and row['orders'] == [['X', 0]]


@pytest.mark.parametrize('name,expected', [('WR_priority', 'BUY'), ('falling_positive', 'SELL'),
    ('falling_index', 'SELL'), ('strong_index', 'BUY'), ('score_BUY', 'BUY'), ('score_equal_SELL', 'SELL')])
def test_synthetic_WR_branch_order_does_not_supply_history(cases, name, expected):
    row = cases['42_'+name]
    assert row['signal'] == expected and row['synthetic_WR']
    assert row['slopes'] == 608 and row['zscores'] == 8


def test_prefix_keeps_state_updates_but_excludes_WR_and_signal_return():
    ns = batch.namespace(batch.DIRECTORY, 1)
    _, dump = batch.extracted(batch.DIRECTORY, 'prefix', ns)
    assert 'rsrs_score' in dump and 'idex_slope' in dump and 'WR' not in dump
    assert 'pop' in dump and 'append' in dump


def test_bias_assignment_uses_explicit_array_operand_only():
    ns = batch.namespace(batch.DIRECTORY, 1); fn, dump = batch.extracted(batch.DIRECTORY, 'bias', ns)
    bias = 1+np.arange(20)*.002
    assert fn(bias) == pytest.approx(20., abs=1e-10)
    assert 'polyfit' in dump and 'get_rank' not in dump


def test_constant_zscore_is_rejected_not_replaced():
    with pytest.raises(ValueError, match='Constant'):
        batch.independent_z([1.]*600)


def test_safe_selector_rejects_eager_defaults():
    assert not batch.safe_function(ast.parse('def f(x=boom()): pass').body[0])


def test_fixed_minute_request_hides_vendor_marker_and_rejects_scope_changes():
    actual = {**batch.PARAMETERS, 'ut': 'not-recorded'}
    row = batch.expected_request('GET', batch.ENDPOINT, {'params': actual}, 0)
    assert 'ut' not in row['parameters'] and not row['vendor_marker_recorded']
    for method, url, params, number in [('POST', batch.ENDPOINT, actual, 0), ('GET', batch.ENDPOINT, actual, 1),
        ('GET', 'https://example.com', actual, 0), ('GET', batch.ENDPOINT, {**actual, 'secid': '1.510300'}, 0)]:
        with pytest.raises(ValueError): batch.expected_request(method, url, {'params': params}, number)


def test_repeat_study_refused_without_overwriting_existing_artifact(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('values', [[3.]*8, [1.]*20, [2+3*x for x in range(8)]])
def test_slope_only_reference_accepts_constant_dependent_values(values):
    expected = 3. if values[-1] != values[0] else 0.
    assert batch.independent_slope(range(len(values)), values) == pytest.approx(expected, abs=1e-12)


def test_positive_nonconstant_prices_can_have_constant_bias():
    prices = pd.Series(2.**np.arange(110))
    bias = (prices/prices.rolling(90).mean()).iloc[-20:].to_numpy()
    assert np.ptp(prices) > 0 and np.ptp(bias) == 0
    fn, _ = batch.extracted(batch.DIRECTORY, 'bias', batch.namespace(batch.DIRECTORY, 1))
    assert fn(bias) == pytest.approx(0., abs=1e-10)
    assert batch.independent_slope(range(20), bias/bias[0])*10000 == 0.


@pytest.mark.parametrize('number', [0, 1])
def test_original_seed_window_shift_and_next_append_are_preserved(number):
    ns = batch.namespace(batch.DIRECTORY, number); batch.selected(batch.DIRECTORY, number, ns); g = ns['g']
    count = 626 if number else 618
    data = pd.DataFrame({'low': np.arange(count, dtype=float), 'high': np.arange(count)+1., 'close': np.arange(count)+10.})
    ns['attribute_history'] = lambda stock, width, *a, **kw: data.iloc[-width:].reset_index(drop=True).copy()
    ns['get_ols'] = lambda x, y: (0., float(x.iloc[0]), float(x.iloc[0])+10.)
    ns['get_zscore'] = lambda values: values[-1]
    if number:
        g.slope_series, g.rsrs_score_history = ns['initial_slope_series']()
        assert g.slope_series == list(range(608))
        assert g.rsrs_score_history == [(i+600)*(i+610) for i in range(8)]
        fn, _ = batch.extracted(batch.DIRECTORY, 'prefix', ns); fn(None, g.ref_stock)
        assert g.slope_series == list(range(1, 609))
        assert g.rsrs_score_history == [(i+600)*(i+610) for i in range(1, 9)]
    else:
        g.slope_series = ns['initial_slope_series']()[:-1]
        assert g.slope_series == list(range(599))
        assert ns['get_signal']() == 'BUY'
        assert g.slope_series == list(range(599))+[600]
