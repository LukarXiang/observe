"""数据集视图：先在当天全部研究候选上做横截面预处理，训练时再按标签有效性与成熟时点过滤。"""
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
    return frame.merge(ok[['decision_date', 'instrument', 'value']], left_on = ['date', 'instrument'], right_on = ['decision_date', 'instrument']).drop(columns = 'decision_date')
