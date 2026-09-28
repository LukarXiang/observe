"""组合指标（决策 17 / 模块 17）。equity 为含期初净值 E0 的序列，第一天费用已计入第一天收益。"""
import numpy as np


def _annual(levels, a):
    levels = np.asarray(levels, dtype = float); t = len(levels) - 1
    return float((levels[-1] / levels[0]) ** (a / t) - 1) if t > 0 else None


def metrics(equity, benchmark = None, rf = 0.0, annualization = 242, min_days = 20):
    e = np.asarray(equity, dtype = float); r = e[1:] / e[:-1] - 1; a = annualization
    rf_d = (1 + rf) ** (1 / a) - 1; x = r - rf_d; sd = x.std(ddof = 1) if len(x) > 1 else 0.0
    peak = np.maximum.accumulate(e); dd = e / peak - 1; low = int(dd.argmin()); high = int(e[:low + 1].argmax())
    out = {'total_return': float(e[-1] / e[0] - 1), 'annual_return': _annual(e, a), 'annual_vol': float(r.std(ddof = 1) * np.sqrt(a)) if len(r) > 1 else None,
           'sharpe': float(x.mean() / sd * np.sqrt(a)) if len(x) >= min_days and sd > 0 else None,
           'max_drawdown': float(dd.min()), 'drawdown_peak': high, 'drawdown_trough': low}
    if benchmark is not None:
        b = np.asarray(benchmark, dtype = float); br = b[1:] / b[:-1] - 1; ok = np.isfinite(br); act = r[ok] - br[ok]
        asd = act.std(ddof = 1) if len(act) > 1 else 0.0; lv = b[np.isfinite(b)]
        bench_annual = _annual(lv, a) if np.isfinite(b[0]) and np.isfinite(b[-1]) else None
        out.update(benchmark_missing_days = int((~ok).sum()), information_ratio = float(act.mean() / asd * np.sqrt(a)) if len(act) >= min_days and asd > 0 else None,
                   annual_return_diff = None if bench_annual is None else out['annual_return'] - bench_annual, relative_nav = (e / e[0]) / (b / b[0]))
    return out
