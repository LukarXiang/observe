"""成对比较（模块 17「模型层」与「组合层」）：同一样本、同一切分下，B 侧（加分钟特征）减 A 侧（日频基础）的逐日差值、分窗口、分年份，
以及差值均值的块抽样 95% 区间。只读已保存的预测、标签与净值，不重新训练、不重新成交。

块抽样：按交易日**位置**上连续的日期块（块长 = 标签持有期的 4 倍、至少 20 天）有放回（环形）重抽，重复 n_boot 次，随机种子固定。
指标不可定义的日子保留在原位置（缺失），不把不相邻的交易日拼成连续块；每次重抽的均值只对可定义的日子取。
这里的区间是**固定预测条件下**的抽样不确定性：预测、模型选择与拟合都当作已知，区间不含重新选模、重新拟合带来的变化。
相邻日的标签重叠使逐日秩 IC 自相关，逐日独立的 t 检验会低估不确定性，所以不用。只保留预先声明的主块长和至多一个敏感性块长。"""
import numpy as np
import pandas as pd

from ..dataset import dev_labels
from .ranking import rank_ic_table, topn_table, undefined_reasons

KEY = ['decision_date', 'instrument']


def block_ci(x, block, n_boot = 1000, seed = 0, level = 0.95):
    """x：按交易日位置排列的序列，缺失位置保留为 nan。环形块重抽下（可定义日子的）均值的百分位区间；可定义的日子少于一个块时返回 None"""
    x = np.asarray(x, float); n = len(x)
    if np.isfinite(x).sum() < max(block, 2): return None
    rng = np.random.default_rng(seed); k = -(-n // block)
    starts = rng.integers(0, n, size = (n_boot, k)); idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    v = x[idx]; f = np.isfinite(v); cnt = f.sum(1); means = np.where(cnt > 0, np.where(f, v, 0.0).sum(1) / np.maximum(cnt, 1), np.nan); a = (1 - level) / 2
    return [float(np.nanquantile(means, a)), float(np.nanquantile(means, 1 - a))]


def _f(x): return None if x is None or not np.isfinite(x) else float(x)


def diff_stats(a, b, block, n_boot = 1000, seed = 0, grid = None, alt_block = None):
    """a、b：按日期索引的序列（不可定义的日子为缺失）。差 = b − a，只在两侧都可定义的日子上算均值；grid：完整的交易日位置（默认取两侧日期的并集）"""
    grid = pd.Index(sorted(set(a.index) | set(b.index))) if grid is None else pd.Index(grid)
    ra, rb = a.reindex(grid).astype(float), b.reindex(grid).astype(float); both = ra.notna() & rb.notna(); d = (rb - ra).where(both)
    out = {'days': int(both.sum()), 'grid_days': int(len(grid)), 'undefined_days': int(len(grid) - both.sum()), 'only_a_defined': int((ra.notna() & ~rb.notna()).sum()), 'only_b_defined': int((~ra.notna() & rb.notna()).sum()),
           'a_mean': _f(ra[both].mean()), 'b_mean': _f(rb[both].mean()), 'diff_mean': _f(d.mean()), 'diff_ci95': block_ci(d.to_numpy(), block, n_boot, seed),
           'b_higher_share': _f((d[both] > 0).mean()) if both.any() else None, 'block_days': block}
    if alt_block: out['diff_ci95_alt'] = block_ci(d.to_numpy(), alt_block, n_boot, seed); out['alt_block_days'] = alt_block
    return out


def pairing_report(pa, pb):
    """两侧预测表（单个模型）的主键、唯一性、fit_asof、证据级别、有限值状态。差异先报告，不靠内连接悄悄消失"""
    ka, kb = pa[KEY].drop_duplicates(), pb[KEY].drop_duplicates(); m = ka.merge(kb, on = KEY, how = 'outer', indicator = True)
    both = pa.drop_duplicates(KEY).merge(pb.drop_duplicates(KEY), on = KEY, suffixes = ('_a', '_b'))
    out = {'rows_a': int(len(pa)), 'rows_b': int(len(pb)), 'common_keys': int((m._merge == 'both').sum()), 'only_in_a': int((m._merge == 'left_only').sum()), 'only_in_b': int((m._merge == 'right_only').sum()),
           'duplicates_a': int(pa.duplicated(KEY).sum()), 'duplicates_b': int(pb.duplicated(KEY).sum()),
           'nonfinite_score_a': int((~np.isfinite(pa.score.to_numpy(float))).sum()), 'nonfinite_score_b': int((~np.isfinite(pb.score.to_numpy(float))).sum()),
           'fit_asof_mismatch': int((pd.to_datetime(both.fit_asof_a) != pd.to_datetime(both.fit_asof_b)).sum()) if 'fit_asof_a' in both else None,
           'evidence_mismatch': int((both.evidence_level_a != both.evidence_level_b).sum()) if 'evidence_level_a' in both else None}
    out['identical_keys'] = out['only_in_a'] == 0 and out['only_in_b'] == 0 and out['duplicates_a'] == 0 and out['duplicates_b'] == 0
    return out


def model_pairs(pred_a, pred_b, lab, block, min_n = 30, top_n = 20, n_boot = 1000, seed = 0, holdout = None, alt_block = None):
    """逐模型比较测试窗上的每日秩 IC 与前 N 名平均标签。标签只用开发区间可用的（dataset.dev_labels）。
    预测主键不完全相同时，该模型的指标只在共同主键上算并标 restricted_to_common_keys。返回 {model_id: {...}}"""
    labd = dev_labels(lab, holdout); y = labd[KEY + ['value']]; out = {}
    grid = pd.Index(sorted(set(pred_a.decision_date) | set(pred_b.decision_date)))
    for model in sorted(set(pred_a.model_id) & set(pred_b.model_id)):
        a, b = pred_a[pred_a.model_id == model], pred_b[pred_b.model_id == model]; rep = pairing_report(a, b)
        a, b = a.drop_duplicates(KEY), b.drop_duplicates(KEY)
        if not rep['identical_keys']:
            common = a[KEY].merge(b[KEY], on = KEY); a, b = a.merge(common, on = KEY), b.merge(common, on = KEY)
        ja, jb = a.merge(y, on = KEY, how = 'left'), b.merge(y, on = KEY, how = 'left')
        ta, tb = rank_ic_table(ja, min_n = min_n), rank_ic_table(jb, min_n = min_n); ia, ib = ta.ic, tb.ic
        pa_, pb_ = topn_table(a, labd, top_n), topn_table(b, labd, top_n)
        split = a.drop_duplicates('decision_date').set_index('decision_date').split_id
        both = pd.concat([ia.rename('a'), ib.rename('b')], axis = 1, join = 'inner').dropna(); windows = []
        for s, g in both.groupby(split.reindex(both.index)): windows.append({'split_id': int(s), 'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())})
        years = pd.to_datetime(pd.Series(both.index, index = both.index)).dt.year
        cover = lambda t: {'valid_label_share': _f(t.valid_label_count.sum() / t.selected_count.sum()) if len(t) else None, 'undefined_days': int((~t.defined).sum()), 'days': int(len(t))}
        out[model] = {'pairing': rep, 'restricted_to_common_keys': not rep['identical_keys'],
                      'rank_ic': {**diff_stats(ia, ib, block, n_boot, seed, grid, alt_block), 'undefined_a': undefined_reasons(ta), 'undefined_b': undefined_reasons(tb)},
                      f'top{top_n}_mean_label': {**diff_stats(pa_.mean_label, pb_.mean_label, block, n_boot, seed, grid, alt_block), 'coverage_a': cover(pa_), 'coverage_b': cover(pb_)},
                      'by_window': windows, 'window_wins_b': int(sum(1 for w in windows if w['diff'] is not None and w['diff'] > 0)), 'windows': len(windows),
                      'by_year': {int(y_): {'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())} for y_, g in both.groupby(years)},
                      'identical_scores': bool(len(ja) == len(jb) and np.array_equal(ja.sort_values(KEY).score.to_numpy(), jb.sort_values(KEY).score.to_numpy()))}
    return out


def factor_pairs(fac, lab, names, directions, dev_dates, block, min_n = 30, n_boot = 1000, seed = 0, holdout = None):
    """新增因子在开发区间的逐日秩 IC（原始因子值，不含模型）：均值、块抽样区间、方向调整后的均值、按年。标签只用开发区间可用的（dataset.dev_labels）"""
    labd = dev_labels(lab, holdout); dev = sorted(set(dev_dates)); grid = pd.Index(dev)
    j = fac[fac.date.isin(set(dev))].merge(labd[KEY + ['value']], left_on = ['date', 'instrument'], right_on = KEY, how = 'left'); out = {}
    for n in names:
        t = rank_ic_table(j.assign(score = j[n]), min_n = min_n); ic = t.ic.reindex(grid); ok = ic.dropna(); sign = directions[n]
        years = pd.to_datetime(pd.Series(ok.index, index = ok.index)).dt.year
        out[n] = {'direction': sign, 'rank_ic_mean': _f(ok.mean()), 'rank_ic_ci95': block_ci(ic.to_numpy(), block, n_boot, seed), 'direction_adjusted_ic': _f(ok.mean() * sign),
                  'days': int(len(ok)), 'undefined': undefined_reasons(t), 'coverage': _f(fac[fac.date.isin(set(dev))][n].notna().mean()), 'by_year': {int(k): _f(v) for k, v in ok.groupby(years).mean().items()}}
    return out


def curve_pairs(eq_a, eq_b, windows, block, n_boot = 1000, seed = 0, until = None):
    """两条净值曲线（{date, equity} 行列表）的日收益差与分窗口累计收益。windows：[(split_id, 首日, 末日)]。
    until：只比较早于该日的日子（一侧账本在该日起因持仓退市等原因不可信时，取双方都干净的区间，仅作诊断）；没有任何日子的窗口不列出"""
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
