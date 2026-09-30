"""数据集视图：先在当天全部研究候选上做横截面预处理，训练时再按标签有效性与成熟时点过滤。"""
import numpy as np
import pandas as pd


def cross_sectional_preprocess(frame, features, eligible = 'eligible'):
    """去极值（中位数 ± 5 倍绝对中位差）→ 标准化 → 缺失填 0；只用当天候选的特征，不看标签"""
    out = frame.copy(); mask = out[eligible].astype(bool) if eligible in out else pd.Series(True, index = out.index)
    for c in features:
        x = out.loc[mask, c]; g = x.groupby(out.loc[mask, 'date'])
        med = g.transform('median'); mad = (x - med).abs().groupby(out.loc[mask, 'date']).transform('median')
        x = x.where(mad == 0, x.clip(med - 5 * mad, med + 5 * mad)); g = x.groupby(out.loc[mask, 'date'])
        std = g.transform('std'); z = ((x - g.transform('mean')) / std).where(std > 0, 0.0)
        out[c] = z.reindex(out.index).where(mask).fillna(0).where(mask)
    return out


def training_rows(frame, labels, asof):
    """训练样本 = 预处理后的特征 ∩ 有效且 matured_at <= asof 的标签"""
    ok = labels[labels.valid & (labels.matured_at <= asof)]
    eligible = frame[frame.eligible.astype(bool)] if 'eligible' in frame else frame
    return eligible.merge(ok[['decision_date', 'instrument', 'value']], left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument']).drop(columns = 'decision_date')


def plan_splits(dates, train = 504, valid = 126, test = 63, holdout_start = None):
    """按决策日期滚动：训练窗固定长度随测试窗前移；测试窗首尾相接不重叠；进入最终留出区间的日子不做测试窗。
    select_asof = 验证窗第一个决策日（其收盘后）；fit_asof = 测试窗前一交易日（其收盘后）"""
    d_dev = [x for x in sorted(dates) if holdout_start is None or x < holdout_start]; rows, s = [], 0   # 最终留出区间不做测试窗
    while s + train + valid + test <= len(d_dev):
        tr, va, te = d_dev[s:s + train], d_dev[s + train:s + train + valid], d_dev[s + train + valid:s + train + valid + test]
        rows.append({'split_id': len(rows), 'train_start': tr[0], 'train_end': tr[-1], 'valid_start': va[0], 'valid_end': va[-1],
                     'test_start': te[0], 'test_end': te[-1], 'select_asof': va[0], 'fit_asof': va[-1]})
        s += test
    return pd.DataFrame(rows)


def samples(labels, split, stage):
    """stage = 'select'：训练窗内 matured_at <= select_asof；'valid'：验证窗内 matured_at <= fit_asof；'fit'：训练 + 验证窗内 matured_at <= fit_asof"""
    d, m = labels.decision_date, labels.matured_at; ok = labels.valid
    if stage == 'select': win, asof = (d >= split.train_start) & (d <= split.train_end), split.select_asof
    elif stage == 'valid': win, asof = (d >= split.valid_start) & (d <= split.valid_end), split.fit_asof
    elif stage == 'fit': win, asof = (d >= split.train_start) & (d <= split.valid_end), split.fit_asof
    else: raise ValueError(stage)
    return labels[ok & win & (m <= asof)]


def dev_labels(labels, holdout_start):
    """开发区间分析（因子评价、方向核对、缺失诊断、测试窗评价）共用的标签可用性规则：
    决策日与成熟时点（退出日）都严格早于最终留出起点。不满足的行保留，标为无效、原因 holdout_boundary，值置缺失；训练逻辑另有 matured_at <= asof 的规则，不经过这里。"""
    out = labels.copy()
    if holdout_start is None: return out
    h = pd.Timestamp(holdout_start)
    late = (pd.to_datetime(out.decision_date) >= h) | (pd.to_datetime(out.matured_at).fillna(pd.Timestamp.max) >= h)
    hit = late & out.valid.astype(bool)
    out.loc[hit, 'invalid_reason'] = 'holdout_boundary'; out.loc[late, 'valid'] = False; out.loc[late, 'value'] = np.nan
    return out
