"""Independent expectations for the original slot, value-average and window boundaries."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch106 as study


@pytest.fixture
def sources(tmp_path):
    folder = tmp_path/'sources'; folder.mkdir()
    for k, name in enumerate(study.SOURCES):
        shutil.copyfile(Path('repo/量化策略源代码')/name, folder/f'{k}.source')
    return tmp_path


def test_rejects_source_mutation_and_unknown_source(sources):
    with pytest.raises(ValueError, match='Unknown'): study.literal(2, directory=sources)
    with (sources/'sources/0.source').open('ab') as stream: stream.write(b'\n')
    with pytest.raises(ValueError, match='identity'): study.literal(0, directory=sources)


def test_monthly_schedule_and_final_value_average_asset(sources):
    _, _, calls = study.initialize(0, sources)
    assert ['monthly', 'on_start', [1, '9:35'], {}] in calls
    result = study.value_case('initial', sources)
    assert (result['security'], result['index']) == ('159905.XSHE', '399324.XSHE')
    assert result['money'] == 2500.
    assert ['monthly', 'target_mv', [1], {'time': 'open+30m'}] in result['schedule']
    assert ['monthly', 'compute_value', [2], {'time': 'open'}] in result['schedule']


def test_duplicate_slots_keep_distinct_quantities_and_order_sequence(sources):
    result = study.slot_case((1, 0, 0, 0), directory=sources)
    assert result['stocks'][:3] == ['510880.XSHG']*3
    assert result['slot_amounts'] == [1200, 1200, 2500, 2500, 2500]
    assert result['orders'][:3] == [['510880.XSHG', 1200], ['510880.XSHG', 1200], ['510880.XSHG', 2500]]


@pytest.mark.parametrize('zf, expected', [(-6., '511010.XSHG'), (-5.99, '510880.XSHG')])
def test_rounded_return_boundary_is_strict(sources, zf, expected):
    result = study.slot_case(zf=zf, directory=sources)
    assert result['stocks'][0] == result['stocks'][1] == expected


def test_failed_sell_reassigns_slot_without_confirmed_fills(sources):
    result = study.slot_case((0,1,0,0), held=True, reject=True, directory=sources)
    assert result['orders'][:2] == [['159949.XSHE', -100], ['513100.XSHG', 100]]
    assert result['stocks'][0] == '513100.XSHG'
    assert result['slot_amounts'][0] == 100


def test_switch_uses_all_cash_rather_than_original_slot_fraction(sources):
    result = study.slot_case((0,1,0,0), held=True, cash_after_sale=20000., directory=sources)
    assert result['orders'][:2] == [['159949.XSHE', -100], ['513100.XSHG', 2000]]
    assert result['slot_amounts'][0] == 2000


def test_implicit_math_and_negative_label_failures_remain_visible(sources):
    rows = {r['case']: r for r in study.diagnostics(sources)['cases']}
    assert rows['slots_implicit_math_not_imported']['error']['type'] == 'NameError'
    assert rows['slots_original_series_negative_label_fails']['error']['type'] == 'KeyError'
    assert rows['value_original_float_count375']['error']['type'] == 'ValueError'
    assert rows['value_removed_numpy_irr']['error']['type'] == 'AttributeError'
    assert rows['value_missing_position']['error']['type'] == 'KeyError'
    assert rows['slots_year_first_day_not_week']['value'] == [1,0,0]


def test_original_MACD_uses_excluded_current_month(sources):
    rows = {r['case']: r for r in study.diagnostics(sources)['cases']}
    call = rows['slots_original_monthly_MACD_contract']['value']['calls'][0]
    assert call[1]['unit'] == '1M' and call[1]['include_now'] is False
    assert [call[1][k] for k in ('SHORT','LONG','MID')] == [12,26,9]
    assert study.diagnostics(sources)['monthly_MACD_computed'] is False


@pytest.mark.parametrize('name, amount, money', [('initial_70_percent',70000.,2000.), ('half_reentry',50000.,3125.)])
def test_actual_entry_fractions_not_comment_values(sources, name, amount, money):
    result = study.value_case(name, sources)
    assert result['orders'] == [['159905.XSHE', amount]]
    assert result['money'] == money and result['last_month_value'] == amount*1.005


@pytest.mark.parametrize('name', ['profit_equal40','profit_equal9','yearly_equal75','cash_gt1000','profit_month_skip'])
def test_strict_thresholds_and_month_skip_emit_no_order(sources, name):
    assert study.value_case(name, sources)['orders'] == []


def test_90_percent_branch_changes_reentry_state(sources):
    exact = study.value_case('profit_exact90', sources); over = study.value_case('profit_over90', sources)
    assert exact['take_profit_days'] == 1 and exact['init_status'] is True
    assert over['take_profit_days'] == 0 and over['init_status'] is False
    assert exact['orders'] == over['orders'] == [['159905.XSHE', -20000.]]


@pytest.mark.parametrize('name, amount', [('buy_cap5',4000.), ('sell_cap5',-4000.)])
def test_value_deviation_is_capped_at_five_periods(sources, name, amount):
    assert study.value_case(name, sources)['orders'] == [['159905.XSHE', amount]]


def test_zero_range_and_short_prior_peak_are_not_fabricated(sources):
    f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1000).strftime('%Y-%m-%d'), 'close': [10.]*1000})
    result, skipped = study.operands(f, sources)
    assert len(result) == 751 and skipped == 751
    assert not result.two_horizons_complete.any() and result.rate1y.eq(0).all()


def test_known_peak_decline_and_long_prior_min_are_independent(sources):
    prices = np.full(1000,10.); prices[900] = 20.; prices[999] = 12.
    f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1000).strftime('%Y-%m-%d'), 'close': prices})
    result, _ = study.operands(f, sources); row = result.iloc[-1]
    assert row.rate1y == 100. and row.rate0 == 100.
    assert row.position_ratio == 0. and row.ratio_lt0318
    assert row.peak_date == f.date.iloc[900] and row.prior_min_date == f.date.iloc[276]


def test_annual_reference_uses_previous_day_and_prior_year_last_open():
    dates = pd.bdate_range('2021-11-01','2022-02-01')
    f = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'close': [100.]*len(dates)})
    f.loc[f.date.eq('2022-01-31'), 'close'] = 94.
    cal = pd.DataFrame({'date': dates, 'is_open': True})
    result, _ = study.annual_operands(f, cal)
    row = result[result.decision_date.eq('2022-02-01')].iloc[0]
    assert row.previous_date == '2022-01-31' and row.prior_year_last_date == '2021-12-31'
    assert row.annual_pct_round2 == -6. and not row.strict_gt_minus6


def test_network_blocker_rejects_socket_and_http():
    with study.network_blocked():
        with pytest.raises(AssertionError, match='network forbidden'): study.socket.create_connection(('localhost',1234))
        with pytest.raises(AssertionError, match='network forbidden'): study.requests.get('https://example.invalid')


def test_annual_missing_last_open_is_skipped():
    dates = pd.bdate_range('2021-11-01','2022-02-01')
    f = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'close': [100.]*len(dates)})
    f = f.loc[f.date.ne('2021-12-31')]
    result, skipped = study.annual_operands(f, pd.DataFrame({'date':dates, 'is_open':True}))
    assert skipped >= 2 and result.empty
