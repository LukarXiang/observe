"""策略字段：快照里的日线、复权因子、证券资料 → 按需计算的宽表（交易日 × 证券）。

所有字段都是「该交易日收盘后可知」的值；决策日 t 的选股只读到 t 行。停牌日的价量为缺失，不当作 0。
金额单位：amount 为元；float_mcap 为亿元（与聚宽 valuation 表的市值单位一致，便于移植阈值）。
"""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..data.prices import with_adjusted
from ..data.standardize import board as board_of, flag

RAW = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turnover', 'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm')
ADJ = ('open_adj', 'high_adj', 'low_adj', 'close_adj', 'ret')
DERIVED = ('float_shares', 'float_mcap', 'is_st', 'paused', 'listed_days', 'limit_up', 'limit_down', 'limit_up_price', 'limit_down_price')
FIELDS = frozenset(RAW + ADJ + DERIVED)
INDEX_FIELDS = frozenset(('open', 'high', 'low', 'close', 'volume', 'amount', 'ret'))
FLOAT_SHARES_FFILL = 60                                     # 停牌期间沿用最近一次由换手推出的流通股本，最多 60 个交易日


class StockPanel(Mapping):
    """按需计算并缓存字段宽表。bars：日线长表（含 is_trading、is_st、board）；meta：证券资料按 instrument 索引"""

    def __init__(self, bars, adj, coverage, meta, days, rules = None):
        self.days = pd.Index(days); b = bars.copy(); b['date'] = pd.to_datetime(b.date).dt.date
        self.cols = pd.Index(sorted(set(b.instrument)))
        self._bars, self._adj, self._coverage, self._meta, self._rules = b, adj, coverage, meta, rules
        self._cache = {}
        self.trading = self._pivot('is_trading').fillna(False).map(flag).astype(bool)

    def _pivot(self, col, frame = None):
        f = self._bars if frame is None else frame
        return f.pivot(index = 'date', columns = 'instrument', values = col).reindex(index = self.days, columns = self.cols)

    def __iter__(self): return iter(FIELDS)
    def __len__(self): return len(FIELDS)

    def __getitem__(self, name):
        if name not in FIELDS: raise KeyError(name)
        if name not in self._cache: self._cache[name] = self._compute(name)
        return self._cache[name]

    def _compute(self, name):
        t = self.trading
        if name in RAW:
            if name in ('ps_ttm', 'pcf_ncf_ttm') and name not in self._bars:
                return pd.DataFrame(np.nan, index=self.days, columns=self.cols)
            return self._pivot(name).apply(pd.to_numeric, errors = 'coerce').astype(float).where(t)
        if name in ADJ:
            if 'close_adj' not in self._cache:
                v = with_adjusted(self._bars, self._adj, self._coverage); v['date'] = pd.to_datetime(v.date).dt.date
                for c in ADJ: self._cache[c] = self._pivot(c, v).astype(float).where(t)
            return self._cache[name]
        if name == 'float_shares':
            s = (self['volume'] / self['turnover'].where(self['turnover'] > 0)).where(lambda x: np.isfinite(x))
            return s.ffill(limit = FLOAT_SHARES_FFILL)
        if name == 'float_mcap':
            close = self['close'].ffill(limit = FLOAT_SHARES_FFILL)           # 停牌日按最近收盘估值，与聚宽市值口径一致
            return close * self['float_shares'] / 1e8
        if name == 'is_st': return self._pivot('is_st').map(flag).astype(float)
        if name == 'paused': return (~t).astype(float)
        if name == 'listed_days':
            ld = pd.to_datetime(self._meta['list_date'].reindex(self.cols))
            listed = np.where(ld.isna(), np.nan, ld.to_numpy().astype('datetime64[D]').astype('int64').astype(float))
            day = pd.to_datetime(self.days).to_numpy().astype('datetime64[D]').astype('int64').astype(float)
            return pd.DataFrame(day[:, None] - listed[None, :], index = self.days, columns = self.cols)
        if name in ('limit_up_price', 'limit_down_price'):
            up, down = self._limits(); self._cache['limit_up_price'], self._cache['limit_down_price'] = up, down
            return self._cache[name]
        if name == 'limit_up': return (self['close'] >= self['limit_up_price'] - 1e-9).astype(float).where(t)
        if name == 'limit_down': return (self['close'] <= self['limit_down_price'] + 1e-9).astype(float).where(t)
        raise KeyError(name)

    def _limits(self):
        """涨跌停价：前收盘 × (1 ± 当日该板块 / ST 状态的涨跌幅)，按价格最小变动单位四舍五入"""
        if self._rules is None: raise ValueError('计算涨跌停需要执行规则集')
        pre = self['preclose']; st = self['is_st'].fillna(0) > 0
        board = pd.Series([self._meta['board'].get(i) if 'board' in self._meta else None for i in self.cols], index = self.cols)
        board = board.where(board.notna(), pd.Series([board_of(i) for i in self.cols], index = self.cols))
        pct = pd.DataFrame(np.nan, index = self.days, columns = self.cols)
        for b in sorted(set(board)):
            cols = board.index[board == b]
            for s in (False, True):
                rate = pd.Series([self._rate(d, b, s) for d in self.days], index = self.days)
                mask = st[cols] if s else ~st[cols]
                pct[cols] = pct[cols].where(~mask, pd.DataFrame(np.repeat(rate.to_numpy()[:, None], len(cols), 1), index = self.days, columns = cols))
        up = np.floor(pre * (1 + pct) * 100 + 0.5) / 100; down = np.floor(pre * (1 - pct) * 100 + 0.5) / 100
        return up, down

    def _rate(self, day, board, st):
        try: return self._rules.on(day, board, st).limit_pct
        except ValueError: return np.nan


class IndexPanel(Mapping):
    """单个指数的字段序列，表示成只有一列的宽表，以便复用同一套表达式"""

    def __init__(self, index_bars, code, days):
        x = index_bars[index_bars.instrument == code].copy(); x['date'] = pd.to_datetime(x.date).dt.date
        x = x.set_index('date').reindex(pd.Index(days))
        self._f = {f: x[[f]].rename(columns = {f: code}).astype(float) for f in INDEX_FIELDS if f != 'ret'}
        self._f['ret'] = self._f['close'] / self._f['close'].shift(1) - 1

    def __iter__(self): return iter(INDEX_FIELDS)
    def __len__(self): return len(INDEX_FIELDS)
    def __getitem__(self, k): return self._f[k]
