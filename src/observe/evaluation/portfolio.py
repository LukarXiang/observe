import numpy as np


def metrics(equity, benchmark = None, rf = 0.0, annualization = 242):
    e = np.asarray(equity, dtype = float); r = e[1:] / e[:-1] - 1
    rf_d = (1 + rf) ** (1 / annualization) - 1; excess = r - rf_d
    sharpe = float(excess.mean() / excess.std(ddof = 1) * np.sqrt(annualization)) if len(excess) >= 20 and excess.std(ddof = 1) else None
    result = {"sharpe": sharpe, "annualized_return": float((e[-1] / e[0]) ** (annualization / len(r)) - 1) if len(r) else None, "max_drawdown": float(np.min(e / np.maximum.accumulate(e) - 1))}
    if benchmark is not None:
        b = np.asarray(benchmark, dtype = float); n = min(len(e), len(b)); strategy_r = e[1:n] / e[:n - 1] - 1; benchmark_r = b[1:n] / b[:n - 1] - 1
        valid = np.isfinite(benchmark_r); active = strategy_r[valid] - benchmark_r[valid]
        deviation = active.std(ddof = 1) if len(active) > 1 else 0.0
        result["information_ratio"] = float(active.mean() / deviation * np.sqrt(annualization)) if deviation else None
        result["benchmark_missing_days"] = int((~valid).sum())
        valid_levels = b[np.isfinite(b)]
        result["annualized_return_diff"] = result["annualized_return"] - ((valid_levels[-1] / valid_levels[0]) ** (annualization / (len(valid_levels) - 1)) - 1) if len(valid_levels) > 1 else None
    return result
