"""成对比较（模块 17「模型层」与「组合层」）：同一样本、同一切分下，B 侧（加分钟特征）减 A 侧（日频基础）的逐日差值、分窗口、分年份，
以及差值均值的块抽样 95% 区间。只读已保存的预测、标签与净值，不重新训练、不重新成交。

块抽样：按连续日期块（块长 = 标签持有期的 4 倍，至少 20 天）有放回地重抽，重复 n_boot 次，随机种子固定，区间可复现。
持有期 h > 1 时相邻日的标签重叠、秩 IC 自相关，逐日独立的 t 检验会低估不确定性，所以不用。"""
import numpy as np
import pandas as pd

from ..models import rank_ic


def block_ci(x, block, n_boot = 1000, seed = 0, level = 0.95):
    """环形连续日期块抽样下均值的百分位置信区间；样本不足一个块时返回 None"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < max(block, 2): return None
    rng = np.random.default_rng(seed); k = -(-n // block)
    starts = rng.integers(0, n, size = (n_boot, k)); idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]      # 环形块：序列两端的观测与中间被抽到的机会相同
    means = x[idx].mean(1); a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def _f(x): return None if x is None or not np.isfinite(x) else float(x)


def diff_stats(a, b, block, n_boot = 1000, seed = 0):
    """a、b：按日期索引的序列。只用两侧都有值的日子；差 = b − a"""
    j = pd.concat([a.rename('a'), b.rename('b')], axis = 1, join = 'inner').dropna(); d = j.b - j.a
    return {'days': int(len(j)), 'a_mean': _f(j.a.mean()), 'b_mean': _f(j.b.mean()), 'diff_mean': _f(d.mean()), 'diff_ci95': block_ci(d.to_numpy(), block, n_boot, seed),
            'b_higher_share': _f((d > 0).mean()) if len(d) else None, 'block_days': block}


def _label(lab): return lab[lab.valid][['decision_date', 'instrument', 'value']]


def _top(j, n):
    return j.sort_values(['decision_date', 'score', 'instrument'], ascending = [True, False, True]).groupby('decision_date').head(n).groupby('decision_date').value.mean()


def model_pairs(pred_a, pred_b, lab, block, min_n = 30, top_n = 20, n_boot = 1000, seed = 0):
    """逐模型比较测试窗上的每日秩 IC 与前 N 名平均标签。返回 {model_id: {...}}"""
    y = _label(lab); out = {}
    for model in sorted(set(pred_a.model_id) & set(pred_b.model_id)):
        ja = pred_a[pred_a.model_id == model].merge(y, on = ['decision_date', 'instrument']); jb = pred_b[pred_b.model_id == model].merge(y, on = ['decision_date', 'instrument'])
        ia, ib = rank_ic(ja, min_n = min_n), rank_ic(jb, min_n = min_n); ta, tb = _top(ja, top_n), _top(jb, top_n)
        split = pred_a[pred_a.model_id == model].drop_duplicates('decision_date').set_index('decision_date').split_id
        both = pd.concat([ia.rename('a'), ib.rename('b')], axis = 1, join = 'inner').dropna(); d = both.b - both.a
        windows = [{'split_id': int(s), 'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())} for s, g in both.groupby(split.reindex(both.index))]
        years = pd.to_datetime(pd.Series(both.index, index = both.index)).dt.year
        out[model] = {'rank_ic': diff_stats(ia, ib, block, n_boot, seed), f'top{top_n}_mean_label': diff_stats(ta, tb, block, n_boot, seed), 'by_window': windows,
                      'window_wins_b': int(sum(1 for w in windows if w['diff'] is not None and w['diff'] > 0)), 'windows': len(windows),
                      'by_year': {int(y_): {'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())} for y_, g in both.groupby(years)},
                      'identical_scores': bool(len(ja) == len(jb) and np.array_equal(ja.sort_values(['decision_date', 'instrument']).score.to_numpy(), jb.sort_values(['decision_date', 'instrument']).score.to_numpy()))}
    return out


def factor_pairs(fac, lab, names, directions, dev_dates, block, min_n = 30, n_boot = 1000, seed = 0):
    """新增因子在开发区间的逐日秩 IC（原始因子值，不含模型）：均值、块抽样区间、方向调整后的均值、按年"""
    y = _label(lab); j = fac[fac.date.isin(set(dev_dates))].merge(y, left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument'])
    out = {}
    for n in names:
        ic = rank_ic(j.assign(score = j[n]), min_n = min_n); ok = ic.dropna(); sign = directions[n]
        years = pd.to_datetime(pd.Series(ok.index, index = ok.index)).dt.year
        out[n] = {'direction': sign, 'rank_ic_mean': _f(ok.mean()), 'rank_ic_ci95': block_ci(ok.to_numpy(), block, n_boot, seed), 'direction_adjusted_ic': _f(ok.mean() * sign),
                  'days': int(len(ok)), 'coverage': _f(fac[fac.date.isin(set(dev_dates))][n].notna().mean()), 'by_year': {int(k): _f(v) for k, v in ok.groupby(years).mean().items()}}
    return out


def curve_pairs(eq_a, eq_b, windows, block, n_boot = 1000, seed = 0, until = None):
    """两条净值曲线（{date, equity} 行列表）的日收益差与分窗口累计收益。windows：[(split_id, 首日, 末日)]。
    until：只比较早于该日的日子（一侧账本在该日起因持仓退市等原因不可信时，取双方都干净的区间）；没有任何日子的窗口不列出"""
    def rets(eq):
        e = pd.DataFrame(eq).set_index('date').equity.astype(float); e.index = pd.to_datetime(e.index).date; r = e.pct_change().dropna()
        return r if until is None else r[r.index < pd.Timestamp(until).date()]
    ra, rb = rets(eq_a), rets(eq_b); rows = []
    comp = lambda r: float((1 + r).prod() - 1)
    for sid, lo, hi in windows:
        a, b = (r[(r.index >= lo) & (r.index <= hi)] for r in (ra, rb))
        if len(a) and len(b): rows.append({'split_id': int(sid), 'days': int(len(a)), 'a': comp(a), 'b': comp(b), 'diff': comp(b) - comp(a)})
    both = ra.index.intersection(rb.index)
    return {'period': {'first': str(both.min()) if len(both) else None, 'last': str(both.max()) if len(both) else None, 'days': int(len(both)), 'truncated_before': None if until is None else str(until)},
            'a_total_return': comp(ra.loc[both]) if len(both) else None, 'b_total_return': comp(rb.loc[both]) if len(both) else None,
            'daily_return': diff_stats(ra, rb, block, n_boot, seed), 'by_window': rows, 'window_wins_b': int(sum(1 for r in rows if r['diff'] > 0)), 'windows': len(rows)}
