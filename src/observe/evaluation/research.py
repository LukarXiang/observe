"""已冻结研究产物的详细评价：参考 alphalens 的因子诊断，模型比较复用现有成对评价。
因子分组与名单先由特征固定再关联标签；每日等权，不把重叠标签连乘为净值。"""
import numpy as np
import pandas as pd

from ..dataset import dev_labels
from .paired import block_ci, model_pairs
from .ranking import rank_ic_table


def finite(x): return None if x is None or not np.isfinite(x) else float(x)


def holdout_start(plan):
    h = plan.holdout_start.iloc[0]
    return None if pd.isna(h) else pd.Timestamp(h).date()


def model_comparisons(pred, lab, plan, h, min_n = 30, top_n = 20, n_boot = 1000, seed = 0):
    """复杂模型与两个基线在同一个测试窗、同一候选名单上比较，分别报告配对差值及块抽样区间。"""
    out, block = {}, max(20, 4 * h)
    for model in ('ridge', 'lgbm'):
        if model not in set(pred.model_id): continue
        for base in ('single_factor', 'equal_blend'):
            if base not in set(pred.model_id): raise ValueError(f'复杂模型 {model} 缺少基线 {base}')
            a, b = (pred[pred.model_id == k].assign(model_id = 'comparison') for k in (base, model))
            out[f'{model}_vs_{base}'] = model_pairs(a, b, lab, block, min_n, top_n, n_boot, seed, holdout_start(plan), 2 * block, plan)['comparison']
    return {'comparisons': out, 'block_days': block, 'n_boot': n_boot, 'seed': seed,
            'note': '固定预测条件下的连续日期块抽样区间，不含重新选参和拟合的不确定性；区间包含 0 时没有可确认的增量'}


def quantiles(values, q = 5):
    """并列值不随机拆开；分不满五组时保留实际组数。"""
    if values.nunique() <= 1: return pd.Series(1, index = values.index, dtype = int)
    codes = pd.qcut(values, q, labels = False, duplicates = 'drop')
    mapping = {c: i + 1 for i, c in enumerate(sorted(codes.dropna().unique()))}
    return codes.map(mapping)


def factor_diagnostics(fac, lab, names, days, h, min_n = 30, n_boot = 1000, seed = 0, holdout = None, rebalance_every = 5):
    """返回每日明细与汇总。所有候选进入分组，标签无效不替补；置信区间保留无效日的交易日位置。"""
    days = sorted(set(days)); labels = dev_labels(lab, holdout)
    f = fac[fac.date.isin(days)].copy()
    j = f.merge(labels[['decision_date', 'instrument', 'value', 'valid']], left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument'], how = 'left')
    j.loc[~j.valid.fillna(False).astype(bool), 'value'] = np.nan
    rows, groups, report, block = [], [], {}, max(20, 4 * h)
    day_pos = {d: k for k, d in enumerate(days)}
    for name in names:
        rt = rank_ic_table(j.assign(score = j[name]), min_n = min_n, by = 'date')
        previous, rebalanced = None, None
        for d, g in j.groupby('date', sort = True):
            values = g.set_index('instrument')[name]; values = values[np.isfinite(values.astype(float))]
            ic = rt.loc[d]; paired = g[np.isfinite(g[name].to_numpy(float)) & np.isfinite(g.value.to_numpy(float))]
            pearson = finite(np.corrcoef(paired[name], paired.value)[0, 1]) if not ic.reason else None
            current = values.rank(); common = pd.concat([previous.rename('previous'), current.rename('current')], axis = 1, join = 'inner') if previous is not None else pd.DataFrame()
            auto = np.corrcoef(common.previous, common.current)[0, 1] if len(common) >= min_n and common.previous.nunique() > 1 and common.current.nunique() > 1 else np.nan
            bins = quantiles(values); memberships = {int(q): set(bins.index[bins == q]) for q in sorted(bins.dropna().unique())}
            means, turnovers = {}, {}
            for q, members in memberships.items():
                y = g[g.instrument.isin(members)].value; valid = np.isfinite(y.to_numpy(float)); share = float(valid.mean())
                means[q] = finite(y[valid].mean()) if not ic.reason and share >= 0.5 else None
                turnover = None if rebalanced is None else finite(1 - len(members & rebalanced.get(q, set())) / len(members))
                if day_pos[d] % rebalance_every == 0: turnovers[q] = turnover
                groups.append({'factor': name, 'date': d, 'quantile': q, 'selected_count': len(members), 'valid_label_count': int(valid.sum()), 'valid_label_share': share,
                               'mean_label': means[q], 'turnover': turnover if day_pos[d] % rebalance_every == 0 else None})
            spread = finite(means[max(means)] - means[min(means)]) if len(means) > 1 and means[max(means)] is not None and means[min(means)] is not None else None
            rows.append({'factor': name, 'date': d, 'candidate_count': len(g), 'finite_factor_count': len(values), 'coverage': len(values) / len(g), 'ic': pearson,
                         'rank_ic': finite(ic.ic), 'n_valid': int(ic.n_valid), 'undefined_reason': ic.reason, 'actual_quantiles': len(memberships), 'long_short_label': spread,
                         'rank_autocorrelation': finite(auto), 'quantile_turnover': finite(np.mean([v for v in turnovers.values() if v is not None])) if any(v is not None for v in turnovers.values()) else None})
            if day_pos[d] % rebalance_every == 0: rebalanced = memberships
            previous = current
        daily = pd.DataFrame([r for r in rows if r['factor'] == name]).set_index('date').reindex(days)
        year = pd.to_datetime(pd.Series(days, index = days)).dt.year
        report[name] = {'ic_mean': finite(daily.ic.mean()), 'ic_std': finite(daily.ic.std()), 'ic_positive_share': finite((daily.ic.dropna() > 0).mean()),
                        'rank_ic_mean': finite(daily.rank_ic.mean()), 'rank_ic_ci95': block_ci(daily.rank_ic, block, n_boot, seed), 'ic_ci95': block_ci(daily.ic, block, n_boot, seed),
                        'coverage_daily_mean': finite(daily.coverage.mean()), 'rank_autocorrelation_mean': finite(daily.rank_autocorrelation.mean()),
                        'quantile_turnover_mean': finite(daily.quantile_turnover.mean()), 'long_short_label_mean': finite(daily.long_short_label.mean()),
                        'defined_days': int(daily.rank_ic.notna().sum()), 'by_year': {int(y): {'ic_mean': finite(g.ic.mean()), 'rank_ic_mean': finite(g.rank_ic.mean()), 'coverage': finite(g.coverage.mean())} for y, g in daily.groupby(year)}}
    similarity = f.groupby('date')[names].apply(lambda g: g.rank().corr()).groupby(level = 1).mean()
    return pd.DataFrame(rows), pd.DataFrame(groups), {'factors': report, 'mean_daily_rank_correlation': {n: {m: finite(similarity.loc[n, m]) for m in names} for n in names},
            'block_days': block, 'n_boot': n_boot, 'seed': seed, 'rebalance_every': rebalance_every, 'label_min_valid_share': 0.5,
            'note': '每日等权；分组基于当天有限因子值，标签无效不替补；多空差仅作诊断，复权标签不是可实现收益，不连乘成净值'}
