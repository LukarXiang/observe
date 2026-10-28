"""Protect the actual source semantics, especially failures and ignored order feedback."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch103 as study


def test_registered_time_does_not_change_when_intraday_signal_changes():
    ns, calls = study.initialize(0)
    ns['get_signal'] = lambda context: 'sell_the_stocks'
    ns['ETFtrade1'](object())
    assert ns['g'].signal == 'sell_the_stocks'
    assert [c for c in calls if c[0] == 'daily'] == [
        ['daily', 'ETFtrade1', {'time': '11:30'}], ['daily', 'ETFtrade2', {'time': '14:40'}]]
    assert tuple(ns['g'].ETFList.values()) == study.REAL_POOL
    assert ns['g'].IdxList['161005.XSHE'] == '161005.XSHE'


def test_fund_fees_and_slippage_are_distinct_original_rules():
    ns, real = study.initialize(0); ns['before_trading_start'](object())
    _, momentum = study.initialize(1)
    assert ['commission', {'buy_cost': .0005, 'sell_cost': .0005, 'min_cost': 5}] in real
    assert ['slippage', [0], {}] in real
    assert ['cost', [{'close_tax': 0., 'open_commission': .0001, 'close_commission': .0001, 'min_commission': 5}], {'type': 'fund'}] in momentum
    assert ['slippage', [.001], {'type': 'fund'}] in momentum


def test_all11_pass_still_select_only1_and_worst_mapping_controls_heat():
    base = {s: np.full(13, 100.) for s in study.REAL_POOL}
    prices = dict.fromkeys(study.REAL_POOL, 100.2)
    first = study.real_signal(base, prices)
    assert first['signal'] == 'BUY' and len(first['buy']) == 1 and len(first['rows']) == 11
    prices[study.REAL_POOL[-1]] = 98.
    result = study.real_signal(base, prices)
    assert result['target_market'] == '399673.XSHE' and len(result['buy']) == 1
    assert all(c[0] in study.REAL_POOL and c[1] == 13 and c[3] == {'df': False} for c in result['history_calls'])


@pytest.mark.parametrize('count,expected', [(4, 0), (5, -1), (6, -1)])
def test_volume_run_uses5_days_not_comment6(count, expected):
    assert study.emotion_monitor([1000.]*(100-count)+[100.]*count)['result'] == expected


def test_volume_no_cross_and_exception_have_different_results():
    assert study.emotion_monitor([100.]*100)['result'] is None
    assert study.emotion_monitor(fail=True)['result'] == 1


def test_nested_rebalance_repeats_targets_and_strict_boundary_is_retained():
    result = study.real_orders('BUY', ('A', 'B'), {'A': 800., 'B': 800.})
    assert result['calls'] == [['target', 'A', 500.], ['target', 'B', 500.]]*2
    equal = study.real_orders('BUY', ('A', 'B'), {'A': 750., 'B': 750.})
    assert equal['calls'] == []


def test_immediate_sell_mutates_live_keys_and_stops_with_original_runtime_error():
    result = study.real_orders('sell_the_stocks', holdings={'A': 900.}, immediate=True)
    assert result['calls'] == [['target', 'A', 0.]]
    assert result['error']['type'] == 'RuntimeError'
    assert result['holding_keys'] == []


def test_empty_buy_and_stale_clear_are_recorded_not_silently_repaired():
    assert study.real_orders('BUY')['error']['type'] == 'ZeroDivisionError'
    assert study.real_orders('BUY', ('A',), {'A': 900.}, clear=('old',))['error']['type'] == 'ValueError'


def test_last_records_intended_buy_despite_none_order_and_cash_fallback():
    result = study.real_orders('BUY', ('A',))
    assert result['last'] == ['A'] and result['buy'] == [] and result['holding_keys'] == []
    assert result['calls'] == [['value', 'A', 1000.], ['target', '511880.XSHG', 100.]]


@pytest.mark.parametrize('boundary,status,holding,direction', [(1.001, 0, '0', np.inf), (.999, 1, 'A', -np.inf)])
def test_momentum_entry_exit_strict_thresholds(boundary, status, holding, direction):
    histories = {'A': np.full(30, 100.)}
    equal = study.momentum(histories, {'A': 100.*boundary}, status=status, holding=holding)
    beyond = study.momentum(histories, {'A': np.nextafter(100.*boundary, direction)}, status=status, holding=holding)
    assert equal['calls'] == [] and len(beyond['calls']) == 1
    assert beyond['calls'][0][2]['STATUS'] == status  # state is updated after order intent, not after fill
    assert beyond['position']['STATUS'] == 1-status


@pytest.mark.parametrize('index', [pd.RangeIndex(30), pd.bdate_range('2024-01-02', periods=30)])
def test_current_pandas_integer_label_access_raises_without_compatibility_fix(index):
    with pytest.raises(KeyError): study.momentum({'A': pd.Series(np.full(30, 100.), index=index)}, {'A': 101.})


def test_short_window_has_no_explicit30_row_check():
    assert study.momentum({'A': np.full(20, 100.)}, {'A': 101.})['calls'] == []
    assert study.momentum({'A': np.full(21, 100.)}, {'A': 101.})['calls'][0][:2] == ['A', 100.]


def test_equal_score_random_bump_makes_iteration_order_economic():
    histories = dict.fromkeys(('A', 'B'), np.full(30, 100.)); prices = dict.fromkeys(histories, 101.)
    ab = study.momentum(histories, prices, ordered=('A', 'B'))
    ba = study.momentum(histories, prices, ordered=('B', 'A'))
    assert ab['position']['ETF_HOLD'] == 'B' and ba['position']['ETF_HOLD'] == 'A'
    zero = study.momentum(histories, prices, random_values=(0.,))
    assert zero['position']['ETF_HOLD'] == 'B'


def test_switch_uses_actual_cash_feedback_and_changes_state_without_fill():
    histories = {'A': np.full(30, 100.)}; prices = {'A': 101.}
    delayed = study.momentum(histories, prices, status=1, holding='old')
    immediate = study.momentum(histories, prices, status=1, holding='old', immediate=True)
    assert [c[:2] for c in delayed['calls']] == [['old', 0.], ['A', 100.]]
    assert [c[:2] for c in immediate['calls']] == [['old', 0.], ['A', 1000.]]
    assert delayed['position']['ETF_HOLD'] == immediate['position']['ETF_HOLD'] == 'A'


def test_diagnostic_JSON_readback_retains_errors_and_does_not_claim_trades():
    doc = study.diagnostics(); decoded = json.loads(json.dumps(doc, allow_nan=False))
    assert decoded == doc and doc['case_count'] == 45 and doc['new_trading_backtests'] == 0
    errors = {r['case']: r['error'] for r in doc['cases'] if r['error']}
    assert errors == {'real_empty_history_IndexError': 'IndexError',
        'momentum_datetime_index_KeyError': 'KeyError', 'momentum_RangeIndex_negative_KeyError': 'KeyError'}
