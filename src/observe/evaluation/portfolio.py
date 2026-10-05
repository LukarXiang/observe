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
        if np.isinf(b).any() or (b[np.isfinite(b)] <= 0).any(): raise ValueError('benchmark 的已知净值必须有限且为正，缺失用 NaN')
        br = b[1:] / b[:-1] - 1; ok = np.isfinite(br) & np.isfinite(r); act = r[ok] - br[ok]
        asd = act.std(ddof = 1) if len(act) > 1 else 0.0
        bench_annual = _annual([1.0, *np.cumprod(1 + br[ok])], a)
        strategy_annual = _annual([1.0, *np.cumprod(1 + r[ok])], a)
        out.update(benchmark_missing_days = int((~ok).sum()), information_ratio = float(act.mean() / asd * np.sqrt(a)) if len(act) >= min_days and asd > 0 else None,
                   annual_return_diff = None if bench_annual is None else strategy_annual - bench_annual, relative_nav = (e / e[0]) / (b / b[0]),
                   benchmark_common_days = int(ok.sum()), common_strategy_annual_return = strategy_annual, common_benchmark_annual_return = bench_annual,
                   active_return_mean = float(act.mean()) if len(act) else None, active_annualization_basis = 'matched_daily_returns_compounded')
    return out


def trading_stats(equity_rows, fills, orders, positions_daily, initial):
    """组合层交易统计（模块 17）：换手、费用分项、成交情况、现金占比、持仓集中度。输入是账本产物的行列表"""
    eq = {r['date']: r for r in equity_rows}; days = sorted(eq); prev = dict(zip(days, [initial] + [eq[d]['equity'] for d in days[:-1]]))
    traded = {}
    for f in fills: traded[f['date']] = traded.get(f['date'], 0.0) + f['value']
    turnover = np.array([traded.get(d, 0.0) / prev[d] for d in days]) if days else np.array([])
    fee = {k: round(sum(f.get(k, 0.0) for f in fills), 2) for k in ('commission', 'stamp_tax', 'transfer_fee', 'fee')}
    reasons = {}
    for o in orders:
        if o.get('status') == 'rejected': reasons[o.get('reject_reason')] = reasons.get(o.get('reject_reason'), 0) + 1
    weights = {}
    for p in positions_daily:
        if p['last_price'] is not None and eq[p['date']]['equity'] > 0: weights.setdefault(p['date'], []).append(p['qty'] * p['last_price'] / eq[p['date']]['equity'])
    top1 = np.array([max(weights.get(d, [0.0])) for d in days]); top5 = np.array([sum(sorted(weights.get(d, []), reverse = True)[:5]) for d in days])
    cash = np.array([eq[d]['cash'] / eq[d]['equity'] for d in days if eq[d]['equity'] > 0])
    mean = lambda x: float(x.mean()) if len(x) else None
    return {'turnover_two_sided_daily_mean': mean(turnover), 'turnover_half_daily_mean': mean(turnover / 2), 'fees': fee, 'fee_ratio_to_initial': fee['fee'] / initial if initial else None,
            'orders': len(orders), 'filled_orders': sum(1 for o in orders if o.get('qty_filled')), 'partial_fills': sum(1 for o in orders if o.get('status') == 'partial'),
            'fill_rate': sum(1 for o in orders if o.get('qty_filled')) / len(orders) if orders else None, 'reject_reasons': dict(sorted(reasons.items(), key = lambda x: str(x[0]))),
            'cash_share_mean': mean(cash), 'max_single_weight_mean': mean(top1), 'top5_weight_mean': mean(top5), 'stale_price_days': sum(1 for d in days if eq[d].get('stale_price'))}
