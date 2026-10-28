"""Verify source selection and rejection paths without treating fixtures as backtests."""
from pathlib import Path

import pytest

from scripts import diagnose_strategy_batch90 as batch


@pytest.fixture(autouse=True)
def restored_sources():
    if not all((Path('repo/量化策略源代码')/name).is_file() for name in batch.stage.SOURCES):
        pytest.skip('Restore SHA-pinned original batch90 sources')


def test_original_unfiltered_pool_controls_one_percent_gate():
    assert batch.limit_selection(total=100)['members_sorted_for_comparison_only'] == []
    result = batch.limit_selection(total=99)
    assert result['members_sorted_for_comparison_only'] == ['600000.XSHG']
    assert result['calls'][:2] == [['trade_days', '2021-01-04', 365], ['securities', 'stock', '2019-08-01']]
    assert result['calls'][2][1] == ['600000.XSHG', '600001.XSHG']


def test_unused_second_MACD_and_distinct_natural_day_requests():
    a = batch.shortline_selection(second_macd=(999., -999.))
    b = batch.shortline_selection(second_macd=(-999., 999.))
    assert a == b
    assert a['buy_list'] == ['600000.XSHG', '600001.XSHG']
    dates = [row[2] for row in a['calls'] if row[0] == 'MACD']
    assert dates == ['2021-01-04', '2021-01-01']
    flow_dates = {row[2] for row in a['calls'] if row[0] == 'flow'}
    assert flow_dates == {'2021-01-04', '2021-01-03'}


def test_deferred_and_rejected_orders_do_not_reserve_slots():
    assert len(batch.limit_orders()['orders']) == 3
    assert len(batch.limit_orders(immediate=True)['orders']) == 2
    assert len(batch.limit_orders(immediate=True, rejected=True)['orders']) == 3
    assert batch.limit_orders(pool=('c', 'b', 'a'), immediate=True)['positions'] == ['c', 'b']


def test_rejected_buy_breaks_and_exact_lot_cash_has_no_fee_reserve():
    assert batch.shortline_orders(held=False, cash=999.)['orders'] == []
    assert batch.shortline_orders(held=False, cash=1000.)['orders'] == [['buy', 'a', 1000.]]
    assert batch.shortline_orders()['orders'] == [['sell', 'held', 0]]


def test_plain_dict_immediate_removal_preserves_original_exception():
    result = batch.shortline_orders(immediate_sell=True)
    assert result['exception'] == 'RuntimeError'
    assert result['synthetic_dictionary_not_platform_mapping']


def test_lifecycle_diagnostics_have_explicit_nontrading_scope():
    result = batch.stage.network_disabled(batch.diagnostics)
    assert result['case_count'] == 31
    assert len({c['case'] for c in result['cases']}) == 31
    assert result['original_signal_rows'] == result['new_trading_backtests'] == 0
    assert not result['platform_equivalence_proven']
