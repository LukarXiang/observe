import numpy as np


def metrics(equity, benchmark = None, rf = 0.0, annualization = 242):
    e = np.asarray(equity, dtype = float); r = e[1:] / e[:-1] - 1
    rf_d = (1 + rf) ** (1 / annualization) - 1; excess = r - rf_d
    sharpe = float(excess.mean() / excess.std(ddof = 1) * np.sqrt(annualization)) if len(excess) >= 20 and excess.std(ddof = 1) else None
    result = {"sharpe": sharpe, "annualized_return": float((e[-1] / e[0]) ** (annualization / len(r)) - 1) if len(r) else None, "max_drawdown": float(np.min(e / np.maximum.accumulate(e) - 1))}
    if benchmark is not None:
        b = np.asarray(benchmark, dtype = float); n = min(len(e), len(b)); active = e[1:n] / e[:n-1] - 1 - (b[1:n] / b[:n-1] - 1)
        result["information_ratio"] = float(active.mean() / active.std(ddof = 1) * np.sqrt(annualization)) if len(active) > 1 and active.std(ddof = 1) else None
        result["annualized_return_diff"] = result["annualized_return"] - ((b[n - 1] / b[0]) ** (annualization / (n - 1)) - 1)
    return result
