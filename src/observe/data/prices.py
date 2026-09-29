"""后复权价视图：原始价 × 当日生效的后复权因子（决策 6）。不单独存表，读取时计算。"""
import pandas as pd
import numpy as np

from .standardize import flag


COVERAGE_COLUMNS = ('instrument', 'status', 'requested_start', 'requested_end',
                    'verified_from', 'verified_through', 'has_gap', 'has_start_basis',
                    'confirmed_no_events', 'verified_at', 'source', 'evidence')


def normalize_coverage(coverage):
    """Return the explicit adjustment coverage contract.

    Old rows are conservatively migrated: missing verification fields are unknown,
    never truthy by accident (in particular, NaN is not a valid boolean).
    """
    if coverage is None:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    if isinstance(coverage, dict):
        rows = [{'instrument': k, **(v if isinstance(v, dict) else {'status': v})} for k, v in coverage.items()]
        coverage = pd.DataFrame(rows)
    c = coverage.copy()
    for name in COVERAGE_COLUMNS:
        if name not in c:
            c[name] = pd.NA
    for name in ('requested_start', 'requested_end', 'verified_from', 'verified_through', 'verified_at'):
        c[name] = pd.to_datetime(c[name], errors='coerce').dt.date
    for name in ('has_gap', 'has_start_basis', 'confirmed_no_events'):
        c[name] = c[name].map(flag).astype(bool)                               # 'False' 字符串是 False；无法识别的值报错
    c['status'] = c['status'].where(c['status'].notna(), 'unknown').astype(str)
    return c[list(COVERAGE_COLUMNS)].drop_duplicates('instrument', keep='last')


def with_adjusted(bars, adj, coverage=None, allow_estimated=False):
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
    a = a.dropna(subset = ['_d'])
    ok = np.isfinite(a.back_factor) & (a.back_factor > 0)                    # 因子必须有限且大于零；否则该证券整段不可用
    invalid = set(a.loc[~ok, 'instrument']); a = a[ok]
    b = b.sort_values('_d'); a = a.sort_values('_d')
    if a.empty:
        m = b.copy(); m['back_factor'] = np.nan
    else:
        m = pd.merge_asof(b, a, on = '_d', by = 'instrument', direction = 'backward')
    cov = normalize_coverage(coverage)
    coverage_rows = cov.set_index('instrument').to_dict('index')
    m['_coverage'] = m.instrument.map(coverage_rows)
    def usable(row):
        c = row['_coverage'] if isinstance(row['_coverage'], dict) else {}
        day = row['_d'].date()
        if not c or bool(c.get('has_gap', False)):
            return False
        through = c.get('verified_through')
        if through is None or pd.isna(through) or day > through:
            return False
        start = c.get('verified_from') or c.get('requested_start')
        if start is not None and not pd.isna(start) and day < start:
            return False
        return bool(c.get('has_start_basis', False) or c.get('confirmed_no_events', False) or c.get('status') == 'no_events')
    # 覆盖声明「确认无事件」，但覆盖区间内却有复权事件：两者冲突，显式报告而不是任选其一
    conflict = set()
    for inst, c in coverage_rows.items():
        if not (c.get('confirmed_no_events') or c.get('status') == 'no_events'): continue
        lo, hi = (None if x is None or pd.isna(x) else x for x in (c.get('verified_from'), c.get('verified_through')))
        if any((lo is None or d >= lo) and (hi is None or d <= hi) for d in a.loc[a.instrument == inst, '_d'].dt.date): conflict.add(inst)
    m['_bad'] = m.instrument.isin(invalid | conflict)
    m['_coverage_usable'] = m.apply(usable, axis=1) & ~m['_bad']
    m['_factor_known'] = m.back_factor.notna()
    m['_available'] = m['_factor_known'] & m['_coverage_usable']
    m['_no_event_basis'] = ~m['_factor_known'] & m['_coverage_usable']       # 覆盖完整、首个事件之前：初始基准因子为 1
    m['_estimated'] = m['_factor_known'] & ~m['_coverage_usable'] & ~m['_bad'] & bool(allow_estimated)   # 只在显式估计模式下保留
    m['back_factor'] = m.back_factor.where(m['_available'] | m['_estimated']).mask(m['_no_event_basis'], 1.0)
    for c in ('open', 'high', 'low', 'close', 'preclose'):
        if c in m: m[f'{c}_adj'] = m[c] * m.back_factor
    m = m.sort_values(['instrument', '_d'])
    if 'close_adj' in m:
        prev = m.groupby('instrument').close_adj.transform(lambda s: s.ffill().shift())
        m['ret'] = (m.close_adj / prev - 1).where(m.close_adj.notna() & prev.notna())
    else: m['ret'] = np.nan
    m['adjustment_status'] = np.select([m['_available'] | m['_no_event_basis'], m['_estimated']], ['usable', 'estimated'], default='unavailable')
    if coverage_rows:
        m['adjustment_verified_through'] = m.instrument.map(lambda x: coverage_rows.get(x, {}).get('verified_through'))
        m['adjustment_source'] = m.instrument.map(lambda x: coverage_rows.get(x, {}).get('source'))
    else:
        m['adjustment_verified_through'] = None; m['adjustment_source'] = None
    # 状态、数值与原因一致：usable 有值无原因；estimated 有值并说明未经覆盖核实；unavailable 无值且必有原因
    m['adjusted_unavailable_reason'] = np.select([
        m['_available'] | m['_no_event_basis'], m.instrument.isin(invalid), m.instrument.isin(conflict), m['_coverage'].isna(),
        m['_coverage'].map(lambda x: bool(x.get('has_gap', False)) if isinstance(x, dict) else False),
        m['_factor_known'], pd.Series(True, index = m.index)
    ], [None, 'invalid_factor', 'coverage_conflict', 'coverage_unknown', 'unknown_gap', 'outside_verified_coverage', 'coverage_not_verified'], default=None)
    return m.drop(columns = ['_d', '_coverage', '_bad', '_coverage_usable', '_factor_known', '_available', '_no_event_basis', '_estimated']).sort_values(['date', 'instrument']).reset_index(drop = True)
