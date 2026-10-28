"""Independent boundary expectations for original callbacks and missing feedback."""
import datetime as dt
import json

import pandas as pd
import pytest

from scripts import research_strategy_batch104 as study


def test_fee_type_and_avoiding_future_data_are_actual_calls():
    ns, short = study.initialize(0)
    _, limit = study.initialize(1)
    assert ns['g'].buy_stock_limit == 1
    assert ['option', ['avoid_future_data', True]] not in short
    assert [c for c in short if c[0] == 'cost'][0][2] == {'type': 'fund'}
    assert [c for c in limit if c[0] == 'daily'] == [
        ['daily', 'sell_if_limit_down_yesterday', ['09:30'], {}],
        ['daily', 'buy', ['09:30'], {}], ['daily', 'sell', ['11:25'], {}],
        ['daily', 'sell', ['14:30'], {}]]
    assert not any(c[0] in ('cost', 'slippage') for c in limit)


def test_delayed_feedback_can_emit_two_full_cash_buy_intents_despite_one_slot():
    delayed = study.short_orders()
    immediate = study.short_orders(immediate=True)
    assert delayed['calls'] == [['buy', 'A', 1000.], ['buy', 'B', 1000.]]
    assert delayed['holding_keys'] == [] and delayed['cash'] == 1000.
    assert immediate['calls'] == [['buy', 'A', 1000.]] and immediate['cash'] == 0.


def test_immediate_sell_mutates_original_live_dictionary():
    result = study.short_orders(holdings=('old',), immediate=True)
    assert result['calls'] == [['sell', 'old', 0.]]
    assert result['error']['type'] == 'RuntimeError' and result['holding_keys'] == []


def test_one_existing_holding_and_limit_equal_block_buy():
    assert study.short_orders(holdings=('A',))['calls'] == []
    assert study.short_orders(limit=True)['calls'] == []


@pytest.mark.parametrize('branch,opening,expected', [(1, 100., ['A']), (1, 101., ['A']),
    (1, 99., []), (1, 101.01, []), (2, 96., ['A']), (2, 97., ['A']),
    (2, 97.01, []), (2, 95.99, [])])
def test_opening_ranges_keep_original_signed_inclusive_thresholds(branch, opening, expected):
    result = study.opening78(branch, opening)
    assert result['selected'] == expected
    assert result['queries'][0][1]['fields'] == ['open', 'pre_close']
    assert result['queries'][0][1]['start_date'] == result['queries'][0][1]['end_date'] == '2024-04-08'


def test_rangeindex_negative_labels_are_swallowed_without_a_compatibility_fix():
    result = study.gain78(1, 100., 105., 105.)
    assert result['selected'] == [] and '根据涨幅过滤ST出错 A: -1' in result['stdout']
    positional_fixture = study.gain78(1, 100., 105., 105., negative_labels=True)
    assert positional_fixture['selected'] == ['A'] and not positional_fixture['real_ST_status_queried']


def test_gain_filter_is_a_return_proxy_with_strict_endpoints():
    for close in (104., 106.):
        assert study.gain78(1, 100., close, close, negative_labels=True)['selected'] == []
    assert study.gain78(2, 100., 110., 104., negative_labels=True)['selected'] == ['A']
    assert study.gain78(2, 100., 110., 110., negative_labels=True)['selected'] == []


def test_profit_sell_and_afternoon_stop_obey_sellability_and_limit_boundary():
    assert study.sell78(price=10.) == []
    assert study.sell78(price=11.) == [['A', 0]]
    assert study.sell78(price=12., limit=12.) == []
    assert study.sell78('14:30:00', price=9.) == [['A', 0]]
    assert study.sell78('14:30:00', closeable=0) == []


def test_trade_day_guard_leaves_exact_length_indexerror():
    ns = study.literal(1, {'get_trade_days': lambda **k: [dt.date(2024, 4, 5), dt.date(2024, 4, 8)]})
    context = type('Context', (), {'current_dt': dt.datetime(2024, 4, 8)})()
    assert ns['get_previous_trade_day'](context, 0) == dt.date(2024, 4, 8)
    assert ns['get_previous_trade_day'](context, 3) is None
    with pytest.raises(IndexError): ns['get_previous_trade_day'](context, 2)


def test_empty_price_pivot_failure_stays_visible():
    ns = study.literal(0, {'get_price': lambda **k: pd.DataFrame(columns=['time', 'code', 'close', 'volume'])})
    with pytest.raises(ValueError, match='Columns must be same length'): ns['stock_filter1']([], '2024-04-05')


def test_all_diagnostics_JSON_preserves_failures_and_no_trade_claims():
    result = study.diagnostics(); decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded == result and result['case_count'] == 46 and result['new_trading_backtests'] == 0
    assert not result['original_strategy_complete'] and not result['economic_variant_approved']
    errors = {r['case']: r['error']['type'] for r in result['cases'] if r['error']}
    assert errors == {'short_empty31_price_table': 'ValueError', 'limit_trade_day_n_2': 'IndexError'}
    rows = {r['case']: r['value'] for r in result['cases']}
    assert rows['limit_risk_minus4_and_no_sellable_check']['calls'] == [['A', 0]]
    assert rows['limit_combined_candidates_not_deduplicated'] == [['A', 500.], ['A', 500.]]
    assert rows['short_MACD_difference_not_returned_histogram'] == ['positive']
