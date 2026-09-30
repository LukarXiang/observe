"""分钟聚合特征（模块 12「分钟聚合特征」）：由 5 分钟线按「证券 × 交易日」算出的日频字段，可像 close_adj 一样写进因子表达式。

口径（区间结束时刻标签，一天 48 根：09:35..11:30 共 24 根、13:05..15:00 共 24 根）：
- 5 分钟收益不跨午休、不跨交易日：每个时段的第一根用 ln(收盘 / 开盘)，其余用相邻两根收盘价之比的对数；缺根则涉及的收益缺失；
- 价格只在当天内相除，复权与否结果相同，因此直接用原始价；
- 完整性按特征所需的时间段判断，不满足时该特征当天为缺失，其他特征不受影响。"""
import numpy as np
import pandas as pd

INTRADAY_FIELDS = ('rv_5m', 'rskew_5m', 'rsj_5m', 'open30_ret', 'tail30_ret', 'tail_amount_share', 'vwap_dev')
SLOTS, MIN_BARS, OPEN_SLOTS, TAIL_SLOTS = 48, 44, slice(0, 6), slice(42, 48)
_MORNING, _AFTERNOON = 24, 24


def slot_of(bar_end):
    """结束时刻 → 0..47 的槽位；不在 09:35..11:30、13:05..15:00 的 5 分钟网格上的记 -1"""
    m = (bar_end.dt.hour * 60 + bar_end.dt.minute).to_numpy(); s = np.full(len(m), -1)
    am = (m >= 575) & (m <= 690) & (m % 5 == 0); pm = (m >= 785) & (m <= 900) & (m % 5 == 0)
    s[am] = (m[am] - 575) // 5; s[pm] = _MORNING + (m[pm] - 785) // 5
    return s


def _grid(bars):
    """长表 → (键表, 各字段 (键数, 48) 数组，缺根为 NaN)；同一根重复时取最后一行"""
    b = bars[['bar_end', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    b['bar_end'] = pd.to_datetime(b.bar_end); b['slot'] = slot_of(b.bar_end); b = b[b.slot >= 0]
    b['date'] = b.bar_end.dt.normalize(); b = b.drop_duplicates(['instrument', 'date', 'slot'], keep = 'last')
    ic, inst = pd.factorize(b.instrument.astype(str)); dc, dates = pd.factorize(b.date)
    keys, inv = np.unique(ic.astype('int64') * len(dates) + dc, return_inverse = True)
    key_table = pd.DataFrame({'instrument': inst.to_numpy()[keys // len(dates)], 'date': dates[keys % len(dates)].date})
    out = {}
    for c in ('open', 'high', 'low', 'close', 'volume', 'amount'):
        a = np.full((len(keys), SLOTS), np.nan); a[inv, b.slot.to_numpy()] = b[c].to_numpy(float); out[c] = a
    return key_table, out


def daily_features(bars):
    """bars: bar_end / instrument / open / high / low / close / volume / amount 的 5 分钟长表（可以是任意一批证券日）。
    返回 date / instrument / n_bars / 各分钟特征，每个「证券 × 交易日」一行"""
    key, g = _grid(bars); o, c, vol, amt = g['open'], g['close'], g['volume'], g['amount']
    n_bars = np.isfinite(c).sum(1)
    with np.errstate(all = 'ignore'):
        # 5 分钟对数收益：时段首根 = ln(c / o)，其余 = ln(c_k / c_{k-1})
        r = np.log(c / np.concatenate([o[:, :1], c[:, :-1]], 1)); r[:, _MORNING] = np.log(c[:, _MORNING] / o[:, _MORNING])
        r[:, 0] = np.log(c[:, 0] / o[:, 0]); r[~np.isfinite(r)] = np.nan
        n_ret = np.isfinite(r).sum(1); ok = (n_bars >= MIN_BARS) & (n_ret >= MIN_BARS - 2)
        r2 = np.nansum(r ** 2, 1) * SLOTS / n_ret; rv2 = np.where(ok, r2, np.nan)      # 缺根时按有效收益个数放大到 48 根
        rv = np.sqrt(rv2)
        skew = np.sqrt(n_ret) * np.nansum(r ** 3, 1) / np.nansum(r ** 2, 1) ** 1.5
        up, dn = np.nansum(np.where(r > 0, r ** 2, 0.0), 1), np.nansum(np.where(r < 0, r ** 2, 0.0), 1)
        rsj = (up - dn) / (up + dn)
        open_ok = np.isfinite(o[:, 0]) & np.isfinite(c[:, OPEN_SLOTS]).all(1); tail_ok = np.isfinite(o[:, 42]) & np.isfinite(c[:, TAIL_SLOTS]).all(1)
        open30 = np.where(open_ok, c[:, 5] / o[:, 0] - 1, np.nan); tail30 = np.where(tail_ok, c[:, 47] / o[:, 42] - 1, np.nan)
        total_amt = np.nansum(amt, 1); total_vol = np.nansum(vol, 1)
        share = np.where(tail_ok & (n_bars >= MIN_BARS) & (total_amt > 0), np.nansum(amt[:, TAIL_SLOTS], 1) / total_amt, np.nan)
        vwap = total_amt / total_vol; dev = np.where(np.isfinite(c[:, 47]) & (n_bars >= MIN_BARS) & (total_vol > 0), c[:, 47] / vwap - 1, np.nan)
    out = key.assign(n_bars = n_bars, rv_5m = rv, rskew_5m = np.where(ok, skew, np.nan), rsj_5m = np.where(ok, rsj, np.nan),
                     open30_ret = open30, tail30_ret = tail30, tail_amount_share = share, vwap_dev = dev)
    for f in INTRADAY_FIELDS: out[f] = out[f].where(np.isfinite(out[f]))                 # 0 / 0 等产生的无穷大一律记缺失
    return out.sort_values(['date', 'instrument']).reset_index(drop = True)


def gap_reasons(daily):
    """特征缺失原因计数：too_few_bars（不足 44 根）/ open_window / tail_window / flat_price（全天价格不变，偏度与涨跌波动比无定义）"""
    n = daily.n_bars; few = n < MIN_BARS
    return {'stock_days': int(len(daily)), 'too_few_bars': int(few.sum()),
            'open_window_incomplete': int((~few & daily.open30_ret.isna()).sum()), 'tail_window_incomplete': int((~few & daily.tail30_ret.isna()).sum()),
            'flat_price': int((~few & daily.rv_5m.notna() & daily.rskew_5m.isna()).sum())}
