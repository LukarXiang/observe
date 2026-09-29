"""股票池（模块 11）：每个决策日给出研究候选及每只证券被排除的原因，只用决策日收盘后已知的信息。

历史在市证券 = 当天日线里有记录的股票（含停牌），不用今天的证券清单倒推。停牌、涨跌停、可卖数量是下单时的可成交检查，不在这里。"""
import hashlib, json

import numpy as np
import pandas as pd

from .data.standardize import board as board_of, flag

DEFAULT = {'boards': ['main'], 'min_listed_sessions': 120, 'exclude_st': True, 'suspend_window': 20, 'max_suspended': 10,
           'liquidity_window': 20, 'min_avg_amount': 2e7}
REASONS = ('board_off', 'not_listed', 'too_new', 'st', 'no_data', 'suspended_long', 'illiquid')


def version(cfg): return hashlib.sha256(json.dumps(cfg, sort_keys = True, default = str).encode()).hexdigest()[:12]


def build_universe(bars, instruments, sessions, cfg = None, start = None, end = None):
    """bars：标准化日线；instruments：证券资料（kind、list_date、delist_date、board）；sessions：有序交易日。
    返回 universe_members：decision_date、instrument、eligible、reason、universe_version（只含当天有日线记录的股票）"""
    cfg = {**DEFAULT, **(cfg or {})}; ver = version(cfg)
    days = pd.Index(sorted(pd.to_datetime(pd.Series(list(sessions))).dt.date))
    meta = instruments.drop_duplicates('instrument', keep = 'last').set_index('instrument')
    stocks = set(meta.index[meta['kind'] == 'stock'])
    b = bars.loc[bars.instrument.isin(stocks), ['date', 'instrument', 'is_trading', 'is_st', 'amount']].copy()
    b['date'] = pd.to_datetime(b.date).dt.date; b = b[b.date.isin(set(days))]
    for c in ('is_trading', 'is_st'):
        if b[c].dtype != bool: b[c] = b[c].map(flag).astype(bool)
    cols = sorted(set(b.instrument))
    if not cols: return pd.DataFrame(columns = ['decision_date', 'instrument', 'eligible', 'reason', 'universe_version'])
    piv = lambda c: b.pivot(index = 'date', columns = 'instrument', values = c).reindex(index = days, columns = cols)
    present = piv('is_trading').notna().to_numpy()
    trading = piv('is_trading').fillna(False).to_numpy(bool) & present
    st = piv('is_st').fillna(False).to_numpy(bool)
    amount = piv('amount').apply(pd.to_numeric, errors = 'coerce').to_numpy(float)

    day64 = pd.to_datetime(pd.Series(days)).to_numpy()[:, None]
    ld = pd.to_datetime(meta['list_date'].reindex(cols)).to_numpy()[None, :]
    dd = pd.to_datetime(meta['delist_date'].reindex(cols) if 'delist_date' in meta else pd.Series(pd.NaT, index = cols)).to_numpy()[None, :]
    listed = ~pd.isna(ld) & (day64 >= ld) & (pd.isna(dd) | (day64 < dd))
    first = np.searchsorted(pd.to_datetime(pd.Series(days)).to_numpy(), np.where(pd.isna(ld), np.datetime64('NaT', 'ns'), ld)[0])
    # 上市以来的交易日数（按交易日历，包含当天）；上市日早于日历起点时只能从起点算下界，因此研究区间须从日历第 min_listed_sessions 个交易日之后开始
    sessions_listed = np.arange(len(days))[:, None] - first[None, :] + 1
    boards = np.array([(meta['board'].get(i) if 'board' in meta else None) or board_of(i) for i in cols])[None, :]

    roll = lambda x, w, mp: pd.DataFrame(x).rolling(w, min_periods = mp)
    suspended = roll((present & ~trading).astype(float), cfg['suspend_window'], 1).sum().to_numpy()
    missing = roll((listed & ~present).astype(float), cfg['liquidity_window'], 1).sum().to_numpy()
    avg = roll(np.where(present, amount, np.nan), cfg['liquidity_window'], cfg['liquidity_window']).mean().to_numpy()

    reason = np.select([~np.isin(boards, cfg['boards']) & present, ~listed, sessions_listed < cfg['min_listed_sessions'], st & bool(cfg['exclude_st']),
                        (missing > 0) | np.isnan(avg), suspended > cfg['max_suspended'], avg < cfg['min_avg_amount']],
                       list(REASONS), default = '')
    out = pd.DataFrame({'decision_date': np.repeat(np.asarray(days, dtype = object), len(cols)), 'instrument': np.tile(np.asarray(cols, dtype = object), len(days)),
                        'reason': reason.ravel(), '_present': present.ravel()})
    out = out[out._present].drop(columns = '_present')
    if start is not None: out = out[out.decision_date >= pd.Timestamp(start).date()]
    if end is not None: out = out[out.decision_date <= pd.Timestamp(end).date()]
    out['eligible'] = out.reason.eq(''); out['reason'] = pd.Series(np.where(out.eligible, None, out.reason), index = out.index, dtype = object); out['universe_version'] = ver
    return out[['decision_date', 'instrument', 'eligible', 'reason', 'universe_version']].reset_index(drop = True)


def members(universe, day):
    u = universe[(universe.decision_date == day) & universe.eligible]; return sorted(u.instrument)
