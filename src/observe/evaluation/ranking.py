"""排序评价的唯一实现（模块 17）：每日秩 IC 与前 N 名。因子层、模型层、成对比较都只调用这里，不再各写一份。

秩 IC：每天先筛出 score 与 label 同时有限的证券，用有效配对数检查 min_n，两侧任一为常数记不可定义，
再在同一个配对子集上取 Spearman（平均名次的皮尔逊相关，与 scipy.stats.spearmanr 一致，测试中逐日对照）。
前 N 名：先按当日全部有限预测分数固定名单（分数降序、代码升序打破并列），再关联未来标签；标签无效的证券留在名单里，
不用第 N+1 名替补，也不记零收益；名单中可评价的部分给平均标签，覆盖不足的日子不可定义。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

IC_COLS = ['ic', 'n_valid', 'reason']
TOP_MIN_VALID_SHARE = 0.5      # 名单中标签有效的比例低于它，当日前 N 名平均标签记为不可定义（受限）


def spearman(x, y):
    """已是有限值的两列的 Spearman。任一侧为常数返回 nan"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _one(score, label, min_n):
    s, y = np.asarray(score, float), np.asarray(label, float); ok = np.isfinite(s) & np.isfinite(y); n = int(ok.sum())
    if n == 0: return np.nan, 0, 'no_valid_pairs'
    if n < min_n: return np.nan, n, 'too_few_valid'
    s, y = s[ok], y[ok]
    if np.ptp(s) == 0: return np.nan, n, 'constant_score'
    if np.ptp(y) == 0: return np.nan, n, 'constant_label'
    return spearman(s, y), n, ''


def rank_ic_table(frame, score = 'score', label = 'value', min_n = 30, by = 'decision_date'):
    """每日秩 IC 明细：索引为日期，列 ic / n_valid（有效配对数）/ reason（不可定义原因，可定义时为空串）"""
    rows = {d: _one(g[score].to_numpy(), g[label].to_numpy(), min_n) for d, g in frame.groupby(by)}
    out = pd.DataFrame.from_dict(rows, orient = 'index', columns = IC_COLS) if rows else pd.DataFrame(columns = IC_COLS)
    out.index.name = by; out['ic'] = out.ic.astype(float); out['n_valid'] = out.n_valid.astype(int); return out


def rank_ic(frame, score = 'score', label = 'value', min_n = 30, by = 'decision_date'):
    """每日秩 IC 序列（不可定义的日子为缺失）；明细见 rank_ic_table"""
    return rank_ic_table(frame, score, label, min_n, by).ic


def undefined_reasons(table):
    """不可定义原因计数，附有效配对数的均值"""
    bad = table[table.ic.isna()]
    return {'undefined_days': int(len(bad)), 'reasons': {k: int(v) for k, v in bad.reason.value_counts().sort_index().items()}, 'n_valid_mean': None if not len(table) else float(table.n_valid.mean())}


def topn_table(pred, labels, n = 20, model = None, min_valid_share = TOP_MIN_VALID_SHARE):
    """前 N 名评价，每个「决策日」一行。
    pred：decision_date / instrument / score（可含 model_id，model 指定时先筛）；labels：decision_date / instrument / value / valid / invalid_reason。
    名单先由预测固定，之后才关联标签。返回列：selected_names、selected_count、valid_label_count、valid_label_share、invalid_reason_counts、mean_label（名单中有效标签的平均，
    有效比例低于 min_valid_share 或没有有效标签时为缺失）、defined"""
    p = pred if model is None else pred[pred.model_id == model]
    p = p.drop(columns = [c for c in ('value', 'valid', 'invalid_reason') if c in p])       # 预测表里若已带标签列，忽略：名单只由分数决定
    p = p[np.isfinite(p.score.to_numpy(float))].sort_values(['decision_date', 'score', 'instrument'], ascending = [True, False, True])
    top = p.groupby('decision_date').head(n)
    lab = labels[['decision_date', 'instrument', 'value', 'valid', 'invalid_reason']].drop_duplicates(['decision_date', 'instrument'])
    j = top.merge(lab, on = ['decision_date', 'instrument'], how = 'left')
    j['valid'] = j.valid.fillna(False).astype(bool); j['invalid_reason'] = np.where(j.valid, '', j.invalid_reason.fillna('no_label_row').replace('', 'unknown'))
    rows = {}
    for d, g in j.groupby('decision_date'):
        ok = g[g.valid & np.isfinite(g.value.astype(float))]; share = len(ok) / len(g)
        rows[d] = {'selected_names': g.instrument.tolist(), 'selected_count': int(len(g)), 'valid_label_count': int(len(ok)), 'valid_label_share': float(share),
                   'invalid_reason_counts': {k: int(v) for k, v in g[~g.valid].invalid_reason.value_counts().sort_index().items()},
                   'mean_label': float(ok.value.mean()) if len(ok) and share >= min_valid_share else np.nan}
    out = pd.DataFrame.from_dict(rows, orient = 'index') if rows else pd.DataFrame(columns = ['selected_names', 'selected_count', 'valid_label_count', 'valid_label_share', 'invalid_reason_counts', 'mean_label'])
    out.index.name = 'decision_date'; out['defined'] = out.mean_label.notna(); return out


def topn_summary(table):
    """前 N 名日表的汇总：平均标签（只含可定义的日子）与覆盖"""
    t = table
    reasons = {}
    for c in t.invalid_reason_counts:
        for k, v in c.items(): reasons[k] = reasons.get(k, 0) + v
    return {'days': int(len(t)), 'defined_days': int(t.defined.sum()), 'undefined_days': int((~t.defined).sum()), 'mean_label': None if not t.defined.any() else float(t.mean_label.mean()),
            'selected_count_total': int(t.selected_count.sum()), 'valid_label_count_total': int(t.valid_label_count.sum()),
            'valid_label_share': None if not len(t) or not t.selected_count.sum() else float(t.valid_label_count.sum() / t.selected_count.sum()),
            'invalid_reason_counts': dict(sorted(reasons.items())), 'min_valid_share': TOP_MIN_VALID_SHARE}
