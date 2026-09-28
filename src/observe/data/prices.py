"""后复权价视图：原始价 × 当日生效的后复权因子（决策 6）。不单独存表，读取时计算。"""
import pandas as pd
import numpy as np


def with_adjusted(bars, adj, coverage = None):
    """计算后复权视图。

    ``coverage`` 可为 ``{instrument: status}`` 或含 instrument/status 的表。
    只有 ``complete`` 或 ``no_events`` 才允许在首个事件前使用初始基准；
    未知/部分覆盖保留缺失，避免把一次局部下载伪装成完整历史。
    """
    required = ['instrument', 'ex_date', 'back_factor']
    a = adj.copy() if adj is not None else pd.DataFrame()
    for c in required:
        if c not in a: a[c] = pd.Series(dtype = 'object' if c != 'back_factor' else 'float64')
    a = a[required]
    b = bars.copy()
    if 'date' not in b or 'instrument' not in b:
        raise ValueError('bars 必须包含 date 和 instrument')
    if b.empty:
        for c in ('open', 'high', 'low', 'close', 'preclose'):
            if c in b: b[f'{c}_adj'] = pd.Series(index = b.index, dtype = 'float64')
        b['back_factor'] = pd.Series(index = b.index, dtype = 'float64'); b['ret'] = pd.Series(index = b.index, dtype = 'float64')
        b['adjustment_status'] = pd.Series(index = b.index, dtype = 'object'); b['adjustment_verified_through'] = pd.Series(index = b.index, dtype = 'object'); b['adjustment_source'] = pd.Series(index = b.index, dtype = 'object')
        b['adjusted_unavailable_reason'] = pd.Series(index = b.index, dtype = 'object')
        return b
    b['_d'] = pd.to_datetime(b.date)
    a['_d'] = pd.to_datetime(a.ex_date, errors = 'coerce')
    a['back_factor'] = pd.to_numeric(a.back_factor, errors = 'coerce')
    b = b.sort_values('_d'); a = a.dropna(subset = ['_d']).sort_values('_d')
    if a.empty:
        m = b.copy(); m['back_factor'] = np.nan
    else:
        m = pd.merge_asof(b, a, on = '_d', by = 'instrument', direction = 'backward')
    if coverage is None:
        status = {}
    elif isinstance(coverage, dict):
        status = coverage
    else:
        status = coverage.set_index('instrument')['status'].to_dict()
    coverage_rows = coverage.set_index('instrument').to_dict('index') if hasattr(coverage, 'set_index') and len(coverage) else {}
    allow_initial = m.instrument.map(status).isin(['complete', 'no_events'])
    if coverage_rows:
        allow_initial &= m.instrument.map(lambda x: bool(coverage_rows.get(x, {}).get('has_start_basis', False) or coverage_rows.get(x, {}).get('status') == 'no_events'))
    m['back_factor'] = m.back_factor.where(~(m.back_factor.isna() & allow_initial), 1.0)
    for c in ('open', 'high', 'low', 'close', 'preclose'):
        if c in m: m[f'{c}_adj'] = m[c] * m.back_factor
    m = m.sort_values(['instrument', '_d'])
    if 'close_adj' in m:
        prev = m.groupby('instrument').close_adj.transform(lambda s: s.ffill().shift())
        m['ret'] = (m.close_adj / prev - 1).where(m.close_adj.notna() & prev.notna())
    else: m['ret'] = np.nan
    m['adjustment_status'] = m.instrument.map(status).fillna('unknown')
    if coverage_rows:
        m['adjustment_verified_through'] = m.instrument.map(lambda x: coverage_rows.get(x, {}).get('verified_through'))
        m['adjustment_source'] = m.instrument.map(lambda x: coverage_rows.get(x, {}).get('source'))
    else:
        m['adjustment_verified_through'] = None; m['adjustment_source'] = None
    m['adjusted_unavailable_reason'] = np.where(m.back_factor.isna(), 'adjustment_coverage_unknown', None)
    return m.drop(columns = '_d').sort_values(['date', 'instrument']).reset_index(drop = True)
