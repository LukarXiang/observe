"""Original-source defects and arithmetic boundaries, not portfolio backtests."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch98 as stage


def ns(number):
    return stage.definitions(Path('repo/量化策略源代码')/stage.SOURCE_PATHS[number], number)


def test_source_sha_rejects_changed_bytes(tmp_path):
    source = tmp_path/'source.txt'; source.write_text('def x(): return 1')
    with pytest.raises(ValueError, match='Source SHA'):
        stage.definitions(source, 0)


def test_only_reviewed_definitions_execute():
    data = ns(0)
    assert 'TuShare' not in data and 'my_ts' not in data
    assert set(k for k in data if k in ('VT_Factor', 'BackTesting', 'Statistical_indicator')) == {
        'VT_Factor', 'BackTesting', 'Statistical_indicator'}
    assert not hasattr(data['BackTesting'], 'plot_net_value')


def test_population_kernel_and_missing_turnover_not_replaced():
    ctor = ns(0)['VT_Factor']; obj = ctor('index', '', '', 3)
    frame = pd.DataFrame({'pct_chg': [1., 2., 3.], 'turnover_rate_f': [2., 2., 2.]})
    assert obj._Calc_func(frame).iloc[-1] == pytest.approx(math.sqrt(2/3)/2)
    assert not np.isclose(obj._Calc_func(frame).iloc[-1], frame.pct_chg.std(ddof=1)/2)
    assert obj._Calc_func(frame.assign(turnover_rate_f=np.nan)).isna().all()


def test_original_interval_is_inclusive_and_hold_resets():
    ctor = ns(0)['BackTesting']
    assert ctor(pd.DataFrame(), 'a', None, value=(1., 2.)).Strategy_c(
        pd.DataFrame({'SINGAL': [.99, 1., 2., 2.01]})) == [0, 1, 1, 0]
    assert ctor(pd.DataFrame(), 'a', None, hold=3, value=.5).Strategy_a(
        pd.DataFrame({'SINGAL': [1., 0., 1., 0., 0., 0.]})) == [1, 1, 1, 1, 1, 0]


def test_forward_label_units_differ():
    obj = ns(0)['VT_Factor']('index', '', '', 3, [2])
    frame = pd.DataFrame({'close': [100., 110., 121.], 'pct_chg': [0., 10., 10.], 'kernel_3': 1.})
    labels = obj._Cal_forward_ret(frame, 3)
    assert labels['未来1日收益率'].iloc[0] == 10.
    assert labels['未来2日收益率'].iloc[0] == pytest.approx(.21)


@pytest.mark.parametrize('method', ['MA', 'BBANDS'])
def test_close_first_leaks_inversion_to_kernel(method):
    ctor = ns(0)['BackTesting']
    vals = np.r_[np.linspace(100., 10., 90), np.linspace(10., 100., 90)]
    vals[30] += 200.; vals[100] -= 200.
    frame = pd.DataFrame({'close': vals, 'kernel': vals})
    close_first = ctor(frame, ['close', 'kernel'], method).back_testing
    kernel_first = ctor(frame, ['kernel', 'close'], method).back_testing
    assert close_first.close_MARK.equals(close_first.kernel_MARK)
    assert not kernel_first.close_MARK.equals(kernel_first.kernel_MARK)


def test_original_summary_and_statistical_defects():
    data = ns(0)
    with pytest.raises(NameError):
        data['BackTesting'](pd.DataFrame({'a': [1., 1.], 'pct_chg': [0., 1.]}), 'a', None, value=0.).summary
    result = data['Statistical_indicator'](pd.DataFrame({'a': [1., 2., 9.]}), 'a', False)
    assert result['stdRet'] == 2. and result['stdRet'] != pytest.approx(np.std([1., 2., 9.]))


def test_17_returns_cannot_fill_15_by_15_quantile():
    data = ns(1); data['g'].quantile_N = data['g'].quantile_M = 15
    for index in (pd.RangeIndex(17), pd.date_range('2020-01-01', periods=17)):
        assert math.isnan(data['cal_ret_quantile'](pd.Series(np.linspace(.01, .1, 17), index=index)))
    with pytest.raises(KeyError):
        data['cal_ret_quantile'](pd.Series(np.linspace(.01, .1, 29)))


def test_first_score_skips_stock_and_second_mutates_shared_pool():
    data = ns(1); g = data['g']; g.N = 18; g.M = 600
    g.quantile_N = g.quantile_M = 15; g.ans = [.8, 1., 1.2]; g.init = True; g.ans_rightdev = []
    seen = []
    def history(stock, *args):
        seen.append(stock); x = np.arange(1., 19.)
        return pd.DataFrame({'low': x, 'high': x+1., 'close': x+2.})
    data['attribute_history'] = history
    assert data['calc_zscore_Rdev']('A') == 0. and seen == [] and len(g.ans) == 3
    assert math.isnan(data['calc_zscore_Rdev']('B'))
    assert len(g.ans) == 4 and seen == ['B']


@pytest.mark.parametrize('method', ['MA', 'BBANDS'])
def test_scalar_oracle_on_exact_flat_and_nonflat_fixture(method):
    data = ns(0); ctor = data['BackTesting']
    for vals in ([100.]*100, list(np.r_[np.linspace(40., 20., 100), np.linspace(20., 60., 100)])):
        obj = ctor(pd.DataFrame(), 'close', method)
        frame = pd.DataFrame({'SINGAL': vals})
        marks = obj._Cal_MA(frame) if method == 'MA' else obj._Cal_BBANDS(frame)
        assert marks == stage.stable_marks(vals, method)


def test_independent_OLS_affine_and_squared_transform():
    data = ns(1); x = np.arange(1., 19.); y = 2*x+3
    assert data['get_ols'](x, y) == pytest.approx((2., 1.))
    assert stage.stable_ols(x.tolist(), y.tolist()) == pytest.approx((2., 1.))
    beta, r2 = data['get_ols'](x*x, y*y)
    assert (beta, r2) == pytest.approx(stage.stable_ols((x*x).tolist(), (y*y).tolist()))
    assert beta != pytest.approx(2.)
