"""分钟聚合特征（模块 12）：与手算逐值核对、按所需时间段判断完整性、不跨午休、表达式可引用。离线，合成数据。"""
import numpy as np
import pandas as pd
import pytest

from observe.factors import ExprError, parse
from observe.factors.intraday import INTRADAY_FIELDS, MIN_BARS, daily_features, gap_reasons, slot_of

DAY = pd.Timestamp('2024-03-15')
ENDS = [DAY + pd.Timedelta(minutes = 575 + 5 * k) for k in range(24)] + [DAY + pd.Timedelta(minutes = 785 + 5 * k) for k in range(24)]   # 09:35..11:30、13:05..15:00


def day(seed = 0, inst = '600001.SH', base = 10.0, drop = (), day_ = DAY):
    """一天 48 根：开盘价 = 上一根收盘价（午休后第一根 = 上午收盘 + 跳空），成交量、成交额随机；drop 是要删掉的槽位"""
    rng = np.random.default_rng(seed); close = np.round(base + np.cumsum(rng.normal(0, 0.05, 48)), 2); open_ = np.r_[close[0] - 0.01, close[:-1]]
    open_[24] = np.round(close[23] + 0.3, 2)                                   # 午休跳空：不应进入已实现波动
    vol = 100 * rng.integers(1, 30, 48); ends = [e + (day_ - DAY) for e in ENDS]
    df = pd.DataFrame({'bar_end': ends, 'instrument': inst, 'open': open_, 'high': np.maximum(open_, close) + 0.01, 'low': np.minimum(open_, close) - 0.01,
                       'close': close, 'volume': vol, 'amount': np.round(vol * close, 2)})
    return df.drop(index = list(drop)).reset_index(drop = True)


def manual(df):
    c, o = df.close.to_numpy(), df.open.to_numpy()
    r = np.r_[np.log(c[0] / o[0]), np.diff(np.log(c[:24])), np.log(c[24] / o[24]), np.diff(np.log(c[24:]))]
    up, dn = (r[r > 0] ** 2).sum(), (r[r < 0] ** 2).sum()
    return {'rv_5m': np.sqrt((r ** 2).sum()), 'rskew_5m': np.sqrt(48) * (r ** 3).sum() / (r ** 2).sum() ** 1.5, 'rsj_5m': (up - dn) / (up + dn),
            'open30_ret': c[5] / o[0] - 1, 'tail30_ret': c[47] / o[42] - 1, 'tail_amount_share': df.amount.iloc[42:].sum() / df.amount.sum(),
            'vwap_dev': c[47] / (df.amount.sum() / df.volume.sum()) - 1}


def test_slots_follow_the_five_minute_grid():
    s = slot_of(pd.Series(ENDS + [DAY + pd.Timedelta(hours = 12), DAY + pd.Timedelta(hours = 9, minutes = 31), DAY + pd.Timedelta(hours = 15, minutes = 5)]))
    assert list(s[:48]) == list(range(48)) and list(s[48:]) == [-1, -1, -1]      # 午休与不在网格上的时刻被忽略


def test_features_match_hand_computation():
    df = day(1); f = daily_features(df); assert len(f) == 1 and f.n_bars[0] == 48
    for k, v in manual(df).items(): assert f[k][0] == pytest.approx(v, rel = 1e-12), k
    assert set(INTRADAY_FIELDS) <= set(f.columns) and f.date[0] == DAY.date() and f.instrument[0] == '600001.SH'


def test_lunch_gap_and_previous_day_do_not_enter_returns():
    a = day(2); b = a.copy(); b.loc[24:, ['open', 'high', 'low', 'close']] += 5.0     # 整个下午抬高 5 元：只有跨午休的收益会变
    fa, fb = daily_features(a), daily_features(b)
    assert fa.rv_5m[0] == pytest.approx(manual(a)['rv_5m']) and fb.rv_5m[0] == pytest.approx(manual(b)['rv_5m'])
    d1, d2 = day(3), day(3, day_ = DAY + pd.Timedelta(days = 1), base = 30.0); g = daily_features(pd.concat([d1, d2]))          # 次日价格水平相差很大，但不会串进当天
    assert len(g) == 2 and g.rv_5m[0] == pytest.approx(manual(d1)['rv_5m']) and g.rv_5m[1] == pytest.approx(manual(d2)['rv_5m'])


def test_completeness_is_judged_per_feature():
    full = daily_features(day(4)).iloc[0]
    mid = daily_features(day(4, drop = [10])).iloc[0]                                 # 少一根中间的：全天特征仍可算，开盘 / 尾盘窗口完整
    assert mid.n_bars == 47 and np.isfinite(mid.rv_5m) and mid.rv_5m == pytest.approx(full.rv_5m, rel = 0.3) and mid.open30_ret == full.open30_ret
    open_gap = daily_features(day(4, drop = [3])).iloc[0]                             # 开盘 30 分钟少一根：只有开盘特征缺失
    assert np.isnan(open_gap.open30_ret) and np.isfinite(open_gap.rv_5m) and open_gap.tail30_ret == full.tail30_ret
    tail_gap = daily_features(day(4, drop = [45])).iloc[0]                            # 尾盘少一根：尾盘收益与成交额占比缺失
    assert np.isnan(tail_gap.tail30_ret) and np.isnan(tail_gap.tail_amount_share) and np.isfinite(tail_gap.vwap_dev) and np.isfinite(tail_gap.rv_5m)
    few = daily_features(day(4, drop = list(range(10, 10 + 48 - MIN_BARS + 1)))).iloc[0]          # 不足 44 根：全天类特征缺失
    assert few.n_bars == MIN_BARS - 1 and np.isnan(few.rv_5m) and np.isnan(few.rskew_5m) and np.isnan(few.vwap_dev)
    g = gap_reasons(daily_features(pd.concat([day(4), day(4, inst = '600002.SH', drop = [3]), day(4, inst = '600003.SH', drop = [45]), day(4, inst = '600004.SH', drop = list(range(10, 15)))])))
    assert g == {'stock_days': 4, 'too_few_bars': 1, 'open_window_incomplete': 1, 'tail_window_incomplete': 1, 'flat_price': 0}


def test_flat_price_day_has_no_shape_features():
    df = day(5); df[['open', 'high', 'low', 'close']] = 10.0; df['amount'] = df.volume * 10.0      # 全天一字板：波动为 0，偏度与涨跌波动比没有定义
    f = daily_features(df).iloc[0]
    assert f.rv_5m == 0 and np.isnan(f.rskew_5m) and np.isnan(f.rsj_5m) and f.open30_ret == 0 and f.vwap_dev == pytest.approx(0)
    assert gap_reasons(daily_features(df))['flat_price'] == 1


def test_zero_volume_day_has_no_vwap():
    df = day(6); df['volume'] = 0; df['amount'] = 0.0
    f = daily_features(df).iloc[0]
    assert np.isnan(f.vwap_dev) and np.isnan(f.tail_amount_share) and np.isfinite(f.rv_5m)


def test_expressions_can_use_intraday_fields():
    p = parse('ts_mean(rv_5m, 20) / ts_mean(rv_5m, 60)'); assert p.lookback == 59 and p.fields == {'rv_5m'}
    with pytest.raises(ExprError): parse('ts_mean(rv_1m, 20)')
