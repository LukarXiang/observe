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


REQUIRED = ['model_id', 'decision_date', 'instrument', 'score', 'split_id', 'fit_asof', 'evidence_level']
MODEL_IDS = ('single_factor', 'equal_blend', 'ridge')


def pairing_report(pa, pb, plan = None, evidence = None):
    """两侧预测表（单个模型）能否作正式成对比较：必要字段、主键唯一与集合相同、分数有限、split_id / fit_asof / 证据级别一致，
    fit_asof 早于决策日并与切分计划一致。任一项不满足记入 invalid_reasons；缺必要字段不视为通过。差异先报告，不靠内连接悄悄消失"""
    reasons = []; miss_a, miss_b = sorted(set(REQUIRED) - set(pa.columns)), sorted(set(REQUIRED) - set(pb.columns))
    if miss_a: reasons.append(f'missing_columns_a:{miss_a}')
    if miss_b: reasons.append(f'missing_columns_b:{miss_b}')
    if miss_a or miss_b: return {'rows_a': int(len(pa)), 'rows_b': int(len(pb)), 'identical_keys': False, 'invalid_reasons': reasons}
    ka, kb = pa[KEY].drop_duplicates(), pb[KEY].drop_duplicates(); m = ka.merge(kb, on = KEY, how = 'outer', indicator = True)
    both = pa.drop_duplicates(KEY).merge(pb.drop_duplicates(KEY), on = KEY, suffixes = ('_a', '_b'))
    count = lambda mask: int(mask.sum())
    out = {'rows_a': int(len(pa)), 'rows_b': int(len(pb)), 'common_keys': int((m._merge == 'both').sum()), 'only_in_a': int((m._merge == 'left_only').sum()), 'only_in_b': int((m._merge == 'right_only').sum()),
           'duplicates_a': int(pa.duplicated(KEY).sum()), 'duplicates_b': int(pb.duplicated(KEY).sum()),
           'nonfinite_score_a': count(~np.isfinite(pa.score.to_numpy(float))), 'nonfinite_score_b': count(~np.isfinite(pb.score.to_numpy(float))),
           'split_id_mismatch': count(both.split_id_a != both.split_id_b), 'fit_asof_mismatch': count(pd.to_datetime(both.fit_asof_a) != pd.to_datetime(both.fit_asof_b)),
           'evidence_mismatch': count(both.evidence_level_a != both.evidence_level_b)}
    for side, p in (('a', pa), ('b', pb)):
        out[f'fit_asof_not_before_decision_{side}'] = count(pd.to_datetime(p.fit_asof) >= pd.to_datetime(p.decision_date))
        out[f'fit_asof_differs_from_plan_{side}'] = out[f'outside_test_window_{side}'] = out[f'evidence_unexpected_{side}'] = 0
        if plan is not None:
            pl = plan.set_index('split_id'); known = p.split_id.isin(pl.index); q = p[known]
            out[f'unknown_split_{side}'] = count(~known)
            out[f'fit_asof_differs_from_plan_{side}'] = count(pd.to_datetime(q.fit_asof).to_numpy() != pd.to_datetime(pl.fit_asof.reindex(q.split_id).to_numpy()))
            d = pd.to_datetime(q.decision_date).to_numpy(); out[f'outside_test_window_{side}'] = count((d < pd.to_datetime(pl.test_start.reindex(q.split_id).to_numpy())) | (d > pd.to_datetime(pl.test_end.reindex(q.split_id).to_numpy())))
        if evidence is not None: out[f'evidence_unexpected_{side}'] = count(p.evidence_level != evidence)
    flags = {'duplicate_keys': out['duplicates_a'] + out['duplicates_b'], 'key_sets_differ': out['only_in_a'] + out['only_in_b'], 'nonfinite_scores': out['nonfinite_score_a'] + out['nonfinite_score_b'],
             'split_id_differs': out['split_id_mismatch'], 'fit_asof_differs_between_sides': out['fit_asof_mismatch'], 'evidence_level_differs_between_sides': out['evidence_mismatch'],
             'fit_asof_not_before_decision': out['fit_asof_not_before_decision_a'] + out['fit_asof_not_before_decision_b'], 'fit_asof_differs_from_plan': out['fit_asof_differs_from_plan_a'] + out['fit_asof_differs_from_plan_b'],
             'decision_outside_test_window': out['outside_test_window_a'] + out['outside_test_window_b'], 'evidence_level_unexpected': out['evidence_unexpected_a'] + out['evidence_unexpected_b'],
             'unknown_split': out.get('unknown_split_a', 0) + out.get('unknown_split_b', 0)}
    reasons += [k for k, v in flags.items() if v]
    out['identical_keys'] = not (flags['duplicate_keys'] or flags['key_sets_differ']); out['invalid_reasons'] = reasons
    return out


def pairing_gate(pred_a, pred_b, lab_a, lab_b, expected_models = None):
    """整体门槛：模型集合两侧一致（只在一侧出现的模型不能靠取交集忽略）、期望的模型都在、两侧标签主键唯一且逐值相同。返回原因列表"""
    reasons = []; ma, mb = set(pred_a.model_id), set(pred_b.model_id)
    if ma != mb: reasons.append(f'model_set_differs:only_a={sorted(ma - mb)},only_b={sorted(mb - ma)}')
    if expected_models and set(expected_models) - (ma & mb): reasons.append(f'expected_models_missing:{sorted(set(expected_models) - (ma & mb))}')
    for side, lab in (('a', lab_a), ('b', lab_b)):
        if lab.duplicated(KEY).any(): reasons.append(f'label_duplicate_keys_{side}')
    cols = KEY + ['value', 'valid', 'matured_at']
    if set(cols) <= set(lab_a.columns) and set(cols) <= set(lab_b.columns):
        x, y = lab_a[cols].sort_values(KEY).reset_index(drop = True), lab_b[cols].sort_values(KEY).reset_index(drop = True)
        if not x.equals(y): reasons.append('labels_differ_between_sides')
    else: reasons.append('label_columns_missing')
    return reasons


def _primary(a, b, y, labd, grid, block, min_n, top_n, n_boot, seed, alt_block):
    """正式主比较：两侧预测完整且成对，各自用自己的原始预测名单"""
    ja, jb = a.merge(y, on = KEY, how = 'left'), b.merge(y, on = KEY, how = 'left')
    ta, tb = rank_ic_table(ja, min_n = min_n), rank_ic_table(jb, min_n = min_n); ia, ib = ta.ic, tb.ic
    pa_, pb_ = topn_table(a, labd, top_n), topn_table(b, labd, top_n)
    split = a.drop_duplicates('decision_date').set_index('decision_date').split_id
    both = pd.concat([ia.rename('a'), ib.rename('b')], axis = 1, join = 'inner').dropna(); windows = []
    for s, g in both.groupby(split.reindex(both.index)): windows.append({'split_id': int(s), 'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())})
    years = pd.to_datetime(pd.Series(both.index, index = both.index)).dt.year
    cover = lambda t: {'valid_label_share': _f(t.valid_label_count.sum() / t.selected_count.sum()) if len(t) else None, 'undefined_days': int((~t.defined).sum()), 'days': int(len(t))}
    return {'rank_ic': {**diff_stats(ia, ib, block, n_boot, seed, grid, alt_block), 'undefined_a': undefined_reasons(ta), 'undefined_b': undefined_reasons(tb)},
            f'top{top_n}_mean_label': {**diff_stats(pa_.mean_label, pb_.mean_label, block, n_boot, seed, grid, alt_block), 'coverage_a': cover(pa_), 'coverage_b': cover(pb_)},
            'by_window': windows, 'window_wins_b': int(sum(1 for w in windows if w['diff'] is not None and w['diff'] > 0)), 'windows': len(windows),
            'by_year': {int(y_): {'days': int(len(g)), 'a': _f(g.a.mean()), 'b': _f(g.b.mean()), 'diff': _f((g.b - g.a).mean())} for y_, g in both.groupby(years)},
            'identical_scores': bool(len(ja) == len(jb) and np.array_equal(ja.sort_values(KEY).score.to_numpy(), jb.sort_values(KEY).score.to_numpy()))}


def _common_subset(a, b, y, labd, grid, block, min_n, top_n, n_boot, seed, alt_block):
    """受限诊断：只在「主键在两侧都出现且唯一、score_A、score_B、label 同时有限」的证券上算双方 IC，并报告删除数量。
    这里的前 N 名名单在共同子集上**重新定义**，是另一个样本上的名单，不能当作原组合的前 N 名，也不进入正式主摘要"""
    a1, b1 = a[~a.duplicated(KEY, keep = False)], b[~b.duplicated(KEY, keep = False)]
    j = a1[KEY + ['score']].merge(b1[KEY + ['score']], on = KEY, suffixes = ('_a', '_b')).merge(y, on = KEY, how = 'left')
    ok = np.isfinite(j.score_a.to_numpy(float)) & np.isfinite(j.score_b.to_numpy(float)) & np.isfinite(j.value.to_numpy(float)); k = j[ok]
    removed = {'rows_a': int(len(a)), 'rows_b': int(len(b)), 'dropped_duplicate_keys_a': int(len(a) - len(a1)), 'dropped_duplicate_keys_b': int(len(b) - len(b1)), 'not_in_both': int(len(a1) + len(b1) - 2 * len(j)),
               'nonfinite_score_a': int((~np.isfinite(j.score_a.to_numpy(float))).sum()), 'nonfinite_score_b': int((~np.isfinite(j.score_b.to_numpy(float))).sum()),
               'label_not_finite': int((~np.isfinite(j.value.to_numpy(float))).sum()), 'kept_rows': int(ok.sum())}
    ia = rank_ic_table(k.assign(score = k.score_a), min_n = min_n).ic; ib = rank_ic_table(k.assign(score = k.score_b), min_n = min_n).ic
    ta = topn_table(k[KEY + ['score_a']].rename(columns = {'score_a': 'score'}), labd, top_n); tb = topn_table(k[KEY + ['score_b']].rename(columns = {'score_b': 'score'}), labd, top_n)
    return {'kind': 'common_subset_diagnostic', 'rows_removed': removed, 'rank_ic': diff_stats(ia, ib, block, n_boot, seed, grid, alt_block),
            f'top{top_n}_common_subset': {**diff_stats(ta.mean_label, tb.mean_label, block, n_boot, seed, grid, alt_block), 'list_redefined': True},
            'note': '受限诊断：不进入正式主摘要；前 N 名名单已在共同证券子集上重新定义'}


def model_pairs(pred_a, pred_b, lab, block, min_n = 30, top_n = 20, n_boot = 1000, seed = 0, holdout = None, alt_block = None, plan = None, evidence = None):
    """逐模型比较测试窗上的每日秩 IC 与前 N 名平均标签。标签只用开发区间可用的（dataset.dev_labels）。
    每个模型先过 pairing_report：通过才输出正式主结果（valid_primary_comparison = true，双方用各自完整的原始预测）；
    不通过则主结果为 None、列出 invalid_reasons，另给单独命名的共同子集诊断。只在一侧出现的模型也逐个列出，不取交集忽略"""
    labd = dev_labels(lab, holdout); y = labd[KEY + ['value']]; out = {}
    grid = pd.Index(sorted(set(pred_a.decision_date) | set(pred_b.decision_date)))
    for model in sorted(set(pred_a.model_id) | set(pred_b.model_id)):
        a, b = pred_a[pred_a.model_id == model], pred_b[pred_b.model_id == model]
        if not len(a) or not len(b):
            out[model] = {'pairing': {'rows_a': int(len(a)), 'rows_b': int(len(b))}, 'valid_primary_comparison': False, 'invalid_reasons': ['model_only_in_a' if len(a) else 'model_only_in_b'], 'rank_ic': None, 'diagnostic': None}; continue
        rep = pairing_report(a, b, plan, evidence); entry = {'pairing': rep, 'valid_primary_comparison': not rep['invalid_reasons'], 'invalid_reasons': rep['invalid_reasons']}
        if entry['valid_primary_comparison']: entry.update(_primary(a, b, y, labd, grid, block, min_n, top_n, n_boot, seed, alt_block), diagnostic = None)
        else: entry.update(rank_ic = None, diagnostic = _common_subset(a, b, y, labd, grid, block, min_n, top_n, n_boot, seed, alt_block) if not rep.get('invalid_reasons', [''])[0].startswith('missing_columns') else None)
        out[model] = entry
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
    until：只比较早于该日的日子（一侧账本在该日起因持仓退市等原因不可信时，取双方都干净的区间，仅作诊断）；没有任何日子的窗口不列出。
    返回计划区间（截断前双方共有的日子）、实际可评价区间与被排除的天数"""
    def rets(eq):
        e = pd.DataFrame(eq).set_index('date').equity.astype(float); e.index = pd.to_datetime(e.index).date; return e.pct_change().dropna()
    fa, fb = rets(eq_a), rets(eq_b); planned = fa.index.intersection(fb.index)
    ra, rb = (fa, fb) if until is None else (fa[fa.index < pd.Timestamp(until).date()], fb[fb.index < pd.Timestamp(until).date()]); rows = []
    comp = lambda r: float((1 + r).prod() - 1)
    for sid, lo, hi in windows:
        a, b = (r[(r.index >= lo) & (r.index <= hi)] for r in (ra, rb))
        if len(a) and len(b): rows.append({'split_id': int(sid), 'days': int(len(a)), 'a': comp(a), 'b': comp(b), 'diff': comp(b) - comp(a)})
    both = ra.index.intersection(rb.index)
    return {'planned_period': {'first': str(planned.min()) if len(planned) else None, 'last': str(planned.max()) if len(planned) else None, 'days': int(len(planned))},
            'period': {'first': str(both.min()) if len(both) else None, 'last': str(both.max()) if len(both) else None, 'days': int(len(both)), 'truncated_before': None if until is None else str(until)},
            'excluded_days': int(len(planned) - len(both)),
            'a_total_return': comp(ra.loc[both]) if len(both) else None, 'b_total_return': comp(rb.loc[both]) if len(both) else None,
            'daily_return': diff_stats(ra, rb, block, n_boot, seed), 'by_window': rows, 'window_wins_b': int(sum(1 for r in rows if r['diff'] > 0)), 'windows': len(rows)}
