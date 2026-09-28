"""组合指标（决策 17 / 模块 17）。equity 为含期初净值 E0 的序列，第一天费用已计入第一天收益。"""
import numpy as np


def _annual(levels, a):
    levels = np.asarray(levels, dtype = float); t = len(levels) - 1
    if t <= 0 or not np.isfinite(levels[0]) or not np.isfinite(levels[-1]) or levels[0] <= 0 or levels[-1] <= 0: return None
    return float((levels[-1] / levels[0]) ** (a / t) - 1)


def metrics(equity, benchmark = None, rf = 0.0, annualization = 242, min_days = 20):
    e = np.asarray(equity, dtype = float); a = annualization
    if e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or (e <= 0).any():
        raise ValueError('equity 必须是至少两天的有限正净值序列')
    r = e[1:] / e[:-1] - 1
    rf_d = (1 + rf) ** (1 / a) - 1; x = r - rf_d; sd = x.std(ddof = 1) if len(x) > 1 else 0.0
    peak = np.maximum.accumulate(e); dd = e / peak - 1; low = int(dd.argmin()); high = int(e[:low + 1].argmax())
    out = {'total_return': float(e[-1] / e[0] - 1), 'annual_return': _annual(e, a), 'annual_vol': float(r.std(ddof = 1) * np.sqrt(a)) if len(r) > 1 else None,
           'sharpe': float(x.mean() / sd * np.sqrt(a)) if len(x) >= min_days and sd > 0 else None,
           'max_drawdown': float(dd.min()), 'drawdown_peak': high, 'drawdown_trough': low}
    if benchmark is not None:
        b = np.asarray(benchmark, dtype = float)
        if b.ndim != 1 or len(b) != len(e): raise ValueError('benchmark 与 equity 长度必须一致')
        finite_b = np.isfinite(b)
        if finite_b.any() and not np.isfinite(b[finite_b]).all(): raise ValueError('benchmark 含无效值')
        br = b[1:] / b[:-1] - 1; ok = np.isfinite(br) & np.isfinite(r); act = r[ok] - br[ok]
        asd = act.std(ddof = 1) if len(act) > 1 else 0.0
        bench_annual = _annual(b, a)
        out.update(benchmark_missing_days = int((~ok).sum()), information_ratio = float(act.mean() / asd * np.sqrt(a)) if len(act) >= min_days and asd > 0 else None,
                   annual_return_diff = None if bench_annual is None else out['annual_return'] - bench_annual, relative_nav = (e / e[0]) / (b / b[0]))
    return out
