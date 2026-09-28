"""后复权价视图：原始价 × 当日生效的后复权因子（决策 6）。不单独存表，读取时计算。"""
import pandas as pd
import numpy as np


def with_adjusted(bars, adj, coverage = None):
    """计算后复权视图。

    ``coverage`` 可为 ``{instrument: status}`` 或含 instrument/status 的表。
    只有 ``complete`` 或 ``no_events`` 才允许在首个事件前使用初始基准；
    未知/部分覆盖保留缺失，避免把一次局部下载伪装成完整历史。
    """
    b = bars.assign(_d = pd.to_datetime(bars.date)).sort_values('_d'); a = adj.assign(_d = pd.to_datetime(adj.ex_date)).sort_values('_d')[['instrument', '_d', 'back_factor']]
    m = pd.merge_asof(b, a, on = '_d', by = 'instrument', direction = 'backward')
    if coverage is None:
        status = {}
    elif isinstance(coverage, dict):
        status = coverage
    else:
        status = coverage.set_index('instrument')['status'].to_dict()
    allow_initial = m.instrument.map(status).isin(['complete', 'no_events'])
    m['back_factor'] = m.back_factor.where(~(m.back_factor.isna() & allow_initial), 1.0)
    for c in ('open', 'high', 'low', 'close', 'preclose'):
        if c in m: m[f'{c}_adj'] = m[c] * m.back_factor
    m = m.sort_values(['instrument', '_d'])
    if 'close_adj' in m:
        prev = m.groupby('instrument').close_adj.transform(lambda s: s.ffill().shift())
        m['ret'] = (m.close_adj / prev - 1).where(m.close_adj.notna() & prev.notna())
    else: m['ret'] = np.nan
    m['adjusted_unavailable_reason'] = np.where(m.back_factor.isna(), 'adjustment_coverage_unknown', None)
    return m.drop(columns = '_d').sort_values(['date', 'instrument']).reset_index(drop = True)
