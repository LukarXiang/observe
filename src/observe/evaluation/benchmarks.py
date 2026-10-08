"""基准只读评价：策略日期不删除，基准缺口不填充，不跨缺失交易日算日收益。"""
import numpy as np
import pandas as pd

from .portfolio import metrics
from ..runs import canonical
from ..data.prices import with_adjusted


def price_levels(index, calendar, dates, name = '000300.SH'):
    days = sorted(dates); prior = [d for d in calendar if d < days[0]] if days else []
    anchor = prior[-1] if prior else None; wanted = [anchor, *days]
    values = pd.Series(dtype = float)
    if len(index):
        if not {'date', 'index', 'close'}.issubset(index.columns): raise ValueError('index_1d 必须包含 date / index / close')
        rows = index[index['index'] == name].copy(); rows['date'] = pd.to_datetime(rows.date).dt.date
        if rows.date.duplicated().any(): raise ValueError(f'指数 {name} 的日期主键重复')
        values = pd.to_numeric(rows.set_index('date').close, errors = 'coerce')
        values = values.where(np.isfinite(values) & (values > 0))
    levels = values.reindex(wanted).to_numpy(float)
    return levels, {'index': name, 'description': '沪深 300 价格指数，不含分红' if name == '000300.SH' else f'{name} 价格指数，不含分红',
                    'anchor_date': str(anchor) if anchor else None, 'convention': '首日收益以前一交易日收盘为起点；只读取实验快照，不前向填充、不跨缺失交易日',
                    'missing_levels': int((~np.isfinite(levels)).sum())}


def stock_price_levels(bars, adj, coverage, calendar, dates, name):
    selected = bars[bars.instrument.eq(name)].copy() if 'instrument' in bars else pd.DataFrame()
    if selected.empty:
        levels, info = price_levels(pd.DataFrame(), calendar, dates, name)
    else:
        view = with_adjusted(selected, adj, coverage)
        prices = view[['date', 'instrument', 'close_adj']].rename(columns = {'instrument': 'index', 'close_adj': 'close'})
        levels, info = price_levels(prices, calendar, dates, name)
    info.update(kind = 'stock', description = f'{name} 后复权股票价格基准',
                adjustment = '冻结后复权因子；非含现金分红再投资的买入持有账本')
    return levels, info


def benchmark_comparison(equity_rows, initial, levels, benchmark_id, model, scenario):
    eq = np.array([initial, *[r['equity'] for r in equity_rows]], dtype = float); levels = np.asarray(levels, dtype = float)
    result = metrics(eq, levels); r = eq[1:] / eq[:-1] - 1; b = levels[1:] / levels[:-1] - 1; valid = np.isfinite(b)
    days = [r['date'] for r in equity_rows]; relative = result.pop('relative_nav')[1:]
    table = pd.DataFrame({'date': days, 'model_id': model, 'scenario': scenario, 'benchmark_id': benchmark_id, 'strategy_return': r,
                          'benchmark_return': b, 'active_return': r - b, 'relative_nav': relative})
    active = {k: v for k, v in result.items() if k.startswith(('benchmark_', 'common_', 'active_')) or k in ('information_ratio', 'annual_return_diff')}
    active.update(available = bool(valid.any()), common_start = str(np.array(days)[valid][0]) if valid.any() else None,
                  common_end = str(np.array(days)[valid][-1]) if valid.any() else None, missing_dates = [str(d) for d, ok in zip(days, valid) if not ok],
                  note = '策略自身收益保留完整区间；主动日收益为策略减基准；年化收益率差只在共同有效日分别复利年化后相减，缺失日不拼成多日收益')
    return table, canonical(active)
