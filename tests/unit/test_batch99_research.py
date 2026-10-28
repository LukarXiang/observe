"""Source behavior, economic separation, and persisted diagnostic shapes."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch99 as stage


def namespace(number):
    return stage.definitions(Path('repo/量化策略源代码')/stage.SOURCE_PATHS[number], number)


def test_source_sha_and_top_level_exclusion(tmp_path):
    source = tmp_path/'source'; source.write_text('print("unexpected")')
    with pytest.raises(ValueError, match='Source SHA'):
        stage.definitions(source, 0)
    ns = namespace(0)
    assert 'set_variables' not in ns and 'ts' not in ns and 'enable_profile' not in ns
    assert set(k for k in ns['CH_VIX'].__dict__ if k in stage.METHODS) == stage.METHODS


@pytest.fixture
def diagnostics():
    return stage.diagnostics([namespace(0), namespace(1)])


def cases(diag):
    return {r['case']: r for r in diag['cases']}


def test_complete_json_roundtrip_and_list_shapes(diagnostics, tmp_path):
    stage.save(tmp_path/'diagnostics.json', diagnostics)
    restored = stage.read(tmp_path/'diagnostics.json')
    assert restored == diagnostics
    assert restored['case_count'] == len(restored['cases'])
    assert restored['new_trading_backtests'] == 0
    result = cases(restored)['call_put_order']['value']
    assert result == [['low', 'mid', 'high'], ['high', 'mid', 'low']]
    assert stage.canonical(pd.array(['low', 'high'], dtype='str')) == ['low', 'high']
    with pytest.raises(TypeError, match='Unreviewed archive type'):
        stage.canonical(object())


def test_expiry_duplicate_and_nonstandard_K0_preserved(diagnostics):
    rows = cases(diagnostics)
    assert rows['five_day_expiry_included']['value']['jy'] == 5/365
    assert rows['K0_is_min_diff_not_largest_strike_below_F']['value'] == 2.
    assert rows['duplicate_strike_type_mean']['value'][0]['call'] == pytest.approx(.45)
    assert rows['less_than_two_expiries']['error'] == 'UnboundLocalError'
    assert rows['missing_put_field']['error'] == 'NameError'
    assert rows['single_strike_deltaK']['error'] == 'IndexError'


def test_original_VIX_and_SKEW_scalar_oracles(diagnostics):
    values = cases(diagnostics)['scalar_vix_skew_comparison']['value']
    assert len(values) == 2
    assert max(v['sigma_error'] for v in values) < 1e-12
    assert max(v['skew_error'] for v in values) < 1e-10
    # Domain failures stay visible rather than acquiring invented option prices.
    assert cases(diagnostics)['negative_variance_returns_nan']['value'] == {'nonfinite': 'nan'}
    assert cases(diagnostics)['equal_maturity_division_by_zero']['error'] == 'ZeroDivisionError'


def test_long_SHIBOR_first_page_failure_and_current_pandas(diagnostics):
    rows = cases(diagnostics)
    assert rows['shibor_2001days_unbound_local']['error'] == 'UnboundLocalError'
    request = rows['shibor_first_page_is_one_day']['value']
    assert request == [{'start_date': '20050101', 'end_date': '20050101'}]
    assert rows['shibor_string_date_attribute_error']['error'] == 'AttributeError'
    rate = rows['shibor_cubic_flat_percent_to_fraction']['value']
    assert len(rate) == 1 and len(rate[0]) == 361
    assert all(float(v) == pytest.approx(.03) for k, v in rate[0].items() if k.isdigit())
    assert rows['GetCBOE_VIX_pandas3_pad_compatibility']['error'] == 'TypeError'


def test_EMA_is_state_not_crossing(diagnostics):
    rows = cases(diagnostics)
    assert rows['EMA_equal_buys']['value'] is True
    assert rows['EMA_falling_buys']['value'] is True
    assert rows['EMA_rising_sells']['value'] is False
    assert rows['EMA_short_window_false']['value'] is False


def test_preopen_contract_filter_and_original_indicator_defect(diagnostics):
    rows = cases(diagnostics)
    standard = rows['preopen_standard_contracts']['value']
    assert standard == [['10000000.XSHG', '10000001.XSHG', '10000002.XSHG'],
                        ['10000012.XSHG', '10000011.XSHG', '10000010.XSHG']]
    assert rows['preopen_empty_None_strings']['value'] == ['None', 'None']
    assert rows['preopen_short_TradingCode']['error'] == 'IndexError'
    assert rows['preopen_first_expiry_adjusted_only_no_next_fallback']['error'] == 'KeyError'
    assert rows['Indicators_sqrt_not_imported']['error'] == 'NameError'


@pytest.mark.parametrize('direction', ['up', 'down'])
@pytest.mark.parametrize('flag', [('Buy', 'Sell'), ('Sell', 'Buy'), ('Buy', 'Buy'), ('Sell', 'Sell')])
def test_paper_first_breakout_and_four_sides(diagnostics, direction, flag):
    row = cases(diagnostics)['paper_'+direction+'_'+flag[0]+'_'+flag[1]]
    assert row['scope'] == 'paper_callback_not_Book' and row['error'] is None
    is_buy = flag[0 if direction == 'up' else 1] == 'Buy'
    premium_in, premium_out = ((.02, .03) if direction == 'up' and is_buy else
                              (.03, .01) if direction == 'up' else
                              (.03, .01) if is_buy else (.02, .03))
    contracts = int(1000/premium_in/10000)
    profit = ((premium_out-premium_in) if is_buy else (premium_in-premium_out))-.001
    assert row['value'][1] == pytest.approx([1., (10000+contracts*10000*profit)/10000])


def test_paper_boundaries_no_fake_Book_run(diagnostics):
    rows = cases(diagnostics)
    for key in ('paper_zero_contracts', 'paper_equal_no_breakout', 'paper_None_string_skips'):
        assert rows[key]['value'][1] == [1., 1.]
    assert rows['paper_zero_premium']['error'] in {'OverflowError', 'ZeroDivisionError'}
    assert rows['paper_missing_entry']['error'] == 'ValueError'
    assert rows['paper_state_resets_next_day']['value'][1] == pytest.approx([1., 1.045, .99])
    assert rows['edge_strike_only_two']['value'] == ['low', 'mid']
    assert rows['equidistant_prefers_lower_strike']['value'] == ['low', 'mid', 'high']


def test_range_excludes_current_high_low_close_and_records_rounding():
    frame = pd.DataFrame({'date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
        'open': [2., 2., 2., 2.], 'high': [2.1, 2.2, 2.3, 100.],
        'low': [1.9, 1.8, 1.7, .01], 'close': [2., 2.1, 1.9, 99.]})
    output, summary = stage.components(namespace(1), frame)
    assert output.iloc[0].range == pytest.approx(.4)
    assert output.iloc[0].original_up == 2.12
    assert output.iloc[0].original_down == 1.88
    assert summary['new_trading_backtests'] == 0 and summary['rows'] == 1
    assert not math.isnan(output.iloc[0].range)
    assert json.loads(json.dumps(summary)) == summary


def test_nonfinite_encoding_never_falls_back_to_string():
    result = stage.canonical(np.array([np.nan, np.inf, -np.inf]))
    assert result == [{'nonfinite': x} for x in ('nan', 'inf', '-inf')]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
