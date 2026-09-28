"""因子表达式：白名单解析、回看长度推导、停牌窗口、横截面、除零、分段与全量一致（模块 12）。"""
import numpy as np
import pandas as pd
import pytest

from observe.factors import ExprError, compute, parse

rng = np.random.default_rng(3)
IDX = pd.bdate_range('2020-01-01', periods = 300); COLS = list('ABCDE')
PANEL = {'close_adj': pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (300, 5)), axis = 0) * 10, IDX, COLS),
         'volume': pd.DataFrame(rng.lognormal(10, 1, (300, 5)), IDX, COLS)}
PANEL['ret'] = PANEL['close_adj'].pct_change()


@pytest.mark.parametrize('expr, lookback', [('ts_mean(ts_delta(close_adj, 5), 20)', 24), ('ts_corr(ret, ts_delay(volume, 1), 10)', 10),
                                            ('cs_rank(ret)', 0), ('ts_std(ret, 20) / ts_mean(ret, 60)', 59), ('where(ts_delay(ret, 3) > 0, ts_max(close_adj, 5), 0)', 4)])
def test_lookback_rules(expr, lookback):
    assert parse(expr).lookback == lookback


def test_lookback_equals_first_valid_row_on_complete_data():
    for e in ('ts_mean(ts_delta(close_adj, 5), 20)', 'ts_corr(ret, ts_delay(volume, 1), 10)', 'ts_slope(close_adj, 15)'):
        out = compute(e, {k: v.iloc[1:] for k, v in PANEL.items()})     # 去掉 ret 首行缺失，使数据完整
        first = out.notna().any(axis = 1).to_numpy().argmax(); assert first == parse(e).lookback, e


@pytest.mark.parametrize('expr', ["__import__('os')", 'close_adj.values', 'close_adj[0]', 'lambda: 1', 'ts_delay(close_adj, -1)', 'ts_mean(close_adj, 0)',
                                  'ts_mean(close_adj, 501)', 'ts_mean(close_adj, 2.5)', 'ts_mean(close_adj, w)', 'open', 'ts_mean(x = close_adj, w = 5)', 'close_adj ** 2'])
def test_non_whitelisted_syntax_is_rejected(expr):
    with pytest.raises(ExprError): parse(expr)


def test_window_counts_trading_days_and_min_obs():
    x = pd.DataFrame({'A': np.arange(1.0, 21.0)}, pd.bdate_range('2021-01-01', periods = 20)); x.iloc[5:8] = np.nan   # 停牌 3 天
    assert np.isnan(compute('ts_mean(close_adj, 20)', {'close_adj': x}).iloc[-1, 0])                                      # 只有 17 个有效值
    assert compute('ts_mean(close_adj, 20)', {'close_adj': x}, min_obs_ratio = 0.8).iloc[-1, 0] == pytest.approx(x.A.mean())


def test_cross_sectional_only_on_eligible_and_ties_and_zero_division():
    x = pd.DataFrame([[1.0, 1.0, 3.0, 100.0]], columns = list('ABCD')); mask = pd.DataFrame([[True, True, True, False]], columns = list('ABCD'))
    r = compute('cs_rank(close_adj)', {'close_adj': x}, mask).iloc[0]
    assert r.A == r.B == pytest.approx(0.5) and r.C == pytest.approx(1.0) and np.isnan(r.D)                                 # 并列取平均名次；D 不在候选
    z = compute('close_adj / (close_adj - 1)', {'close_adj': x}).iloc[0]; assert np.isnan(z.A) and np.isfinite(z.C)          # 分母为 0 → 缺失


def test_segmented_equals_full_with_warmup():
    e = 'cs_zscore(ts_mean(ts_delta(close_adj, 5), 20) / ts_std(ret, 30)) + cs_rank(ts_corr(ret, volume, 10))'
    lb = parse(e).lookback; full = compute(e, PANEL)
    parts = []
    for a, b in ((0, 120), (120, 210), (210, 300)):
        s = max(0, a - lb); seg = compute(e, {k: v.iloc[s:b] for k, v in PANEL.items()}); parts.append(seg.iloc[a - s:])
    pd.testing.assert_frame_equal(pd.concat(parts), full)


def test_slope_of_linear_series_is_one():
    x = pd.DataFrame({'A': np.arange(30.0)}); assert compute('ts_slope(close_adj, 10)', {'close_adj': x}).iloc[-1, 0] == pytest.approx(1.0)
