"""后复权价视图：原始价 × 当日生效的后复权因子（决策 6）。不单独存表，读取时计算。"""
import pandas as pd


def with_adjusted(bars, adj):
    """bars: bars_1d；adj: adj_factors。因子取「该日或之前最近一次」；在首条记录之前为 1。
    完全没有复权记录的证券因子为缺失（可能只是还没做全量初始化），不默认成 1"""
    b = bars.assign(_d = pd.to_datetime(bars.date)).sort_values('_d'); a = adj.assign(_d = pd.to_datetime(adj.ex_date)).sort_values('_d')[['instrument', '_d', 'back_factor']]
    m = pd.merge_asof(b, a, on = '_d', by = 'instrument', direction = 'backward')
    known = m.instrument.isin(set(adj.instrument)); m['back_factor'] = m.back_factor.where(~(known & m.back_factor.isna()), 1.0)
    for c in ('open', 'high', 'low', 'close', 'preclose'): m[f'{c}_adj'] = m[c] * m.back_factor
    m = m.sort_values(['instrument', '_d'])
    m['ret'] = m.groupby('instrument').close_adj.transform(lambda s: s / s.ffill().shift() - 1)   # 停牌日收益为缺失，复牌日相对停牌前最后收盘
    return m.drop(columns = '_d').sort_values(['date', 'instrument']).reset_index(drop = True)
