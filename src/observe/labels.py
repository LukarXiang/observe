"""统一标签 adj_open_to_open_h：决策日 k 收盘后决策，k+1 开盘买入、k+1+h 开盘卖出，用后复权开盘价。
它相当于假设分红除息时立即再投资、配股按复权因子处理，只作排序目标，不等于账本收益。"""
import numpy as np
import pandas as pd


def build_labels(bars, calendar, h = 5):
    """bars: 长表 date / instrument / adj_open / is_trading；calendar: 全部交易日（有序）。返回每个「决策日 × 证券」一行"""
    cal = pd.Index(sorted(calendar)); price_col = 'adj_open' if 'adj_open' in bars else 'open_adj'
    px = bars.pivot(index = 'date', columns = 'instrument', values = price_col).reindex(cal)
    trading = bars.pivot(index = 'date', columns = 'instrument', values = 'is_trading').reindex(cal).fillna(False).astype(bool)
    entry, exit_ = px.shift(-1), px.shift(-(1 + h)); tr_in, tr_out = trading.shift(-1, fill_value = False), trading.shift(-(1 + h), fill_value = False)
    out = pd.DataFrame({'value': (exit_ / entry - 1).stack(future_stack = True), 'entry_ok': tr_in.stack(), 'exit_ok': tr_out.stack()}).reset_index()
    out.columns = ['decision_date', 'instrument', 'value', 'entry_ok', 'exit_ok']
    pos = {d: j for j, d in enumerate(cal)}; j = out.decision_date.map(pos)
    out['entry_date'] = [cal[x + 1] if x + 1 < len(cal) else pd.NaT for x in j]; out['exit_date'] = [cal[x + 1 + h] if x + 1 + h < len(cal) else pd.NaT for x in j]
    out['matured_at'] = out['exit_date']                                       # 取退出日收盘后，保守口径
    reason = np.select([out.exit_date.isna(), ~out.entry_ok, ~out.exit_ok, ~np.isfinite(out.value)], ['not_matured', 'entry_suspended', 'exit_suspended', 'no_price'], '')
    out['invalid_reason'] = reason; out['valid'] = reason == ''; out.loc[~out.valid, 'value'] = np.nan
    return out[['decision_date', 'instrument', 'entry_date', 'exit_date', 'value', 'matured_at', 'valid', 'invalid_reason']]
