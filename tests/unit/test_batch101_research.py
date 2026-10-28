"""Original monthly state, ES arithmetic and futures selection boundaries."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import research_strategy_batch101 as study


def test_source_sha_and_namespace_isolation():
    a, b = study.literal(0), study.literal(0)
    assert a['g'] is not b['g'] and a['market_open'] is not b['market_open']
    with pytest.raises(ValueError, match='Unknown original'): study.literal(2)


def test_original_pool_datetimes_are_archived_with_actual_JSON_shape():
    import json
    doc = study.diagnostics()
    row = next(r for r in doc['cases'] if r['case']=='allweather_initial_rules_and_schedule')
    assert row['error'] is None
    assert row['value']['pool']['stock']['codes'][0]['513500.XSHG'] == '2014-01-15T00:00:00'
    assert json.loads(json.dumps(doc, allow_nan=False)) == doc


def test_group_uses_any_eligible_date_before_other_funds_listed():
    import datetime as dt
    ns, _ = study.init_allweather()
    targets = ns['get_trade_target'](SimpleNamespace(current_dt=dt.datetime(2013, 3, 25)))
    assert targets == ['510310.XSHG', '513100.XSHG', '513500.XSHG', '511010.XSHG', '510170.XSHG']
    assert ns['g'].pool['stock']['codes'][0]['513500.XSHG'] > dt.datetime(2013, 3, 25)


def test_actual_period_is_twelve_month_callbacks_not_twelve_days():
    rows = {r['case']: r for r in study.diagnostics()['cases']}
    result = rows['rebalance_on_callbacks_0_12_24']['value']
    assert [v['callback_count'] for v in result['calls']] == [0, 12, 24]
    assert result['final_count'] == 25


def test_original_security_baseline_cannot_match_asset_class_keys():
    ns, _ = study.init_allweather(); ns['g'].rebalanced_asset_values = {'510310.XSHG': 100.}
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={'510310.XSHG': SimpleNamespace(value=140.)}))
    assert ns['calc_asset_max_raise'](context) == 0
    assert ns['g'].raise_rate == -1


def test_original_global_inverse_ES_ignores_class_rates_with_injected_returns():
    values = {'510310.XSHG': [-.1, -.05]+[0.]*118, '518880.XSHG': [-.1, -.05]+[0.]*118}
    left, right = study.injected_inverse_ES(values), study.injected_inverse_ES(values, True)
    assert set(left) == set(right) == set(values)
    for key in values:
        assert left[key].shape == (1,) and left[key][0] == right[key][0] == .5


def test_original_zero_ES_retains_nonfinite_array_weight_shape():
    actual = study.injected_inverse_ES({'510310.XSHG': [0.]*120})
    assert actual['510310.XSHG'].shape == (1,) and np.isnan(actual['510310.XSHG'][0])


@pytest.mark.parametrize('confidence, denominator', [(1.96, 6), (2.06, 4), (2.18, 3), (2.34, 2), (2.58, 1), (0, 6)])
def test_original_ES_includes_endpoint_and_keeps_fixed_denominator(confidence, denominator):
    values = np.linspace(-.2, .1, 120)
    actual = study.es_kernel(values, confidence)
    expected = -math.fsum(sorted(values)[:denominator+1])/denominator
    assert actual.shape == (1,)
    assert math.isclose(actual[0], expected, abs_tol=1e-15)


def test_zero_denominator_is_not_silently_corrected():
    assert np.isinf(study.es_kernel([-.1]+[0.]*119, 5)[0])
    assert study.es_kernel([.01]*120)[0] < 0


@pytest.mark.parametrize('index_price, held, equity, expected_orders', [
    (106., (), 1_000_000., []), (108., (), 1_000_000., [['order', 'IF1', 33]]),
    (108., ('IF1',), 1_000_000., []), (108., (), 100., [['order', 'IF1', 0]])])
def test_original_futures_threshold_holdings_and_integer_order(index_price, held, equity, expected_orders):
    prices = dict.fromkeys(study.INDICES, index_price); prices['IF1'] = 100.
    ns, c, calls = study.futures_namespace(prices, {'IF': ['IF1']}, held, equity)
    ns['Trader_buy'](c)
    assert ns['g'].future_contract == 'IF1'
    assert [call for call in calls if call[0] == 'order'] == expected_orders
    assert all(call[1:4] == [1, '1m', 'close'] for call in calls if call[0] == 'history')


def test_lifecycle_and_empty_pool_are_preserved():
    ns, c, _ = study.futures_namespace(initialize_state=False)
    with pytest.raises(AttributeError): ns['Trader_buy'](c)
    ns, c, _ = study.futures_namespace(prices=dict.fromkeys(study.INDICES, 100.))
    with pytest.raises(IndexError): ns['Trader_buy'](c)


def test_original_daily_resample_failure_is_not_repaired():
    row = next(r for r in study.diagnostics()['cases'] if r['case']=='original_daily_resample_how_current_pandas_fails')
    assert row['error'] == 'TypeError' and 'how' in row['message']


def test_original_sell_uses_selected_state_not_new_live_ranking():
    ns, c, calls = study.futures_namespace(holdings=('IF1', 'IF2'))
    ns['g'].future_contract = 'IF1'; ns['Trader_sell'](c)
    assert calls == [['target', 'IF2', 0]]
