"""统一评价函数（模块 17）：秩 IC 与前 N 名的黄金样本、与 SciPy 逐日对照、留出边界标签规则、成对预测检查、块抽样位置。手工构造的小例子。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from observe.dataset import dev_labels
from observe.evaluation.paired import block_ci, diff_stats, model_pairs, pairing_report
from observe.evaluation.ranking import rank_ic, rank_ic_table, spearman, topn_summary, topn_table
from observe.labels import build_labels

D = date(2024, 1, 2)


def one_day(score, label, day = D):
    return pd.DataFrame({'decision_date': day, 'instrument': [f'{i:06d}.SH' for i in range(len(score))], 'score': score, 'value': label})


def test_golden_rank_ic_uses_the_joint_valid_pairs():
    f = one_day([1, 2, 3, np.nan], [10, 30, 20, 15])
    t = rank_ic_table(f, min_n = 3).loc[D]
    assert t.ic == pytest.approx(0.5) and t.n_valid == 3 and t.reason == ''                # 旧实现先分别排名再丢缺失，得到约 0.65465367
    u = rank_ic_table(f, min_n = 4).loc[D]; assert np.isnan(u.ic) and u.n_valid == 3 and u.reason == 'too_few_valid'                 # min_n 按有效配对数检查
    assert rank_ic(f, min_n = 3).loc[D] == pytest.approx(0.5)


def test_missing_in_different_positions_inf_and_constants():
    f = one_day([1, np.nan, 3, 4, 5], [2, 1, np.nan, 4, 3])
    t = rank_ic_table(f, min_n = 3).loc[D]; assert t.n_valid == 3 and t.ic == pytest.approx(spearmanr([1, 4, 5], [2, 4, 3])[0])         # 两侧缺失位置不同：只留同时有限的三个
    g = one_day([1, np.inf, 3, 4, 5], [2, 1, 6, 4, -np.inf]); t = rank_ic_table(g, min_n = 3).loc[D]; assert t.n_valid == 3            # inf 不是有效值
    assert rank_ic_table(one_day([1, 1, 1, 1], [1, 2, 3, 4]), min_n = 3).loc[D].reason == 'constant_score'
    assert rank_ic_table(one_day([1, 2, 3, 4], [5, 5, 5, 5]), min_n = 3).loc[D].reason == 'constant_label'
    assert rank_ic_table(one_day([np.nan] * 3, [1, 2, 3]), min_n = 1).loc[D].reason == 'no_valid_pairs'
    assert rank_ic_table(one_day([1, 2, np.nan, np.nan], [np.nan, np.nan, 3, 4]), min_n = 1).loc[D].reason == 'no_valid_pairs'          # 全部缺失：两侧从不同时有值
    assert not np.isnan(rank_ic_table(one_day([1, 2, 3], [3, 1, 2]), min_n = 3).loc[D].ic)                                                # 刚好达到门槛
    assert np.isnan(rank_ic_table(one_day([1, 2, 3], [3, 1, 2]), min_n = 4).loc[D].ic)


def test_ties_and_scipy_reference_day_by_day():
    rng = np.random.default_rng(0); frames = []
    for k in range(30):
        n = 40; s = rng.integers(0, 6, n).astype(float); y = np.round(rng.normal(0, 1, n) + 0.3 * s, 1)
        s[rng.random(n) < 0.15] = np.nan; y[rng.random(n) < 0.15] = np.nan; frames.append(one_day(s, y, date(2024, 1, 2 + k)))
    f = pd.concat(frames); t = rank_ic_table(f, min_n = 5)
    for d, g in f.groupby('decision_date'):
        ok = g.score.notna() & g.value.notna(); assert t.loc[d].n_valid == ok.sum() and t.loc[d].ic == pytest.approx(spearmanr(g.score[ok], g.value[ok])[0], abs = 1e-12)
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(spearmanr([1, 2, 2, 3], [1, 3, 2, 4])[0])


def labels(rows):
    return pd.DataFrame(rows, columns = ['decision_date', 'instrument', 'value', 'valid', 'invalid_reason'])


def test_top_n_list_is_fixed_before_labels_and_never_backfilled():
    pred = pd.DataFrame({'decision_date': D, 'instrument': ['A', 'B', 'C'], 'score': [3.0, 2.0, 1.0]})
    lab = labels([(D, 'A', np.nan, False, 'exit_suspended'), (D, 'B', 0.10, True, ''), (D, 'C', -0.20, True, '')])
    t = topn_table(pred, lab, n = 1).loc[D]
    assert t.selected_names == ['A'] and t.selected_count == 1 and t.valid_label_count == 0 and t.valid_label_share == 0 and t.invalid_reason_counts == {'exit_suspended': 1}
    assert np.isnan(t.mean_label) and not t.defined                                            # 不能变成 B 的 +10%，也不能记零收益
    t2 = topn_table(pred, lab, n = 2).loc[D]                                                   # 名单 A、B：只有 B 可评价，覆盖 50%，达到门槛
    assert t2.selected_names == ['A', 'B'] and t2.valid_label_count == 1 and t2.mean_label == pytest.approx(0.10) and t2.valid_label_share == 0.5
    t3 = topn_table(pred, lab, n = 2, min_valid_share = 0.75).loc[D]; assert np.isnan(t3.mean_label) and t3.valid_label_count == 1         # 覆盖不足：不可定义，不是 +10%
    s = topn_summary(topn_table(pred, lab, n = 1)); assert s['defined_days'] == 0 and s['mean_label'] is None and s['invalid_reason_counts'] == {'exit_suspended': 1}


def test_top_n_ties_missing_rows_and_nonfinite_scores():
    pred = pd.DataFrame({'decision_date': D, 'instrument': ['B', 'A', 'C', 'D'], 'score': [1.0, 1.0, np.nan, 0.5]})
    lab = labels([(D, 'A', 0.01, True, ''), (D, 'B', 0.03, True, '')])                        # D 没有标签行；C 分数缺失不入选
    t = topn_table(pred, lab, n = 3).loc[D]
    assert t.selected_names == ['A', 'B', 'D'] and t.invalid_reason_counts == {'no_label_row': 1} and t.mean_label == pytest.approx(0.02)          # 并列按代码升序


def test_golden_top_n_matches_model_eval_path():
    """研究评价与成对比较用同一实现：A 标签无效时前 1 名不可定义"""
    pa = pd.DataFrame({'model_id': 'm', 'decision_date': D, 'instrument': ['A', 'B', 'C'], 'score': [3.0, 2.0, 1.0], 'split_id': 0, 'fit_asof': date(2023, 12, 29), 'evidence_level': 'x'})
    lab = labels([(D, 'A', np.nan, False, 'entry_suspended'), (D, 'B', 0.1, True, ''), (D, 'C', -0.2, True, '')]).assign(matured_at = date(2024, 1, 9))
    r = model_pairs(pa, pa, lab, 2, min_n = 1, top_n = 1, n_boot = 100)['m']['top1_mean_label']
    assert r['days'] == 0 and r['a_mean'] is None and r['coverage_a']['undefined_days'] == 1 and r['coverage_a']['valid_label_share'] == 0


def price_path(days, shock_from = None):
    px = 10 + np.arange(len(days)) * 0.1
    if shock_from is not None: px = np.where(np.arange(len(days)) >= shock_from, px * 1.3, px)
    return pd.DataFrame({'date': days, 'instrument': 'X', 'adj_open': px, 'is_trading': True})


@pytest.mark.parametrize('h', [2, 5])
def test_dev_labels_never_reach_into_the_holdout(h):
    days = [d.date() for d in pd.bdate_range('2024-01-02', periods = 30)]; hold = days[20]
    lab = build_labels(price_path(days), days, h = h); lab2 = build_labels(price_path(days, shock_from = 20), days, h = h)       # 只改留出期（含留出起点）的价格
    a, b = dev_labels(lab, hold), dev_labels(lab2, hold)
    pd.testing.assert_frame_equal(a.drop(columns = ['value']).reset_index(drop = True), b.drop(columns = ['value']).reset_index(drop = True))
    ok = a[a.valid]; assert (pd.to_datetime(ok.matured_at) < pd.Timestamp(hold)).all() and (pd.to_datetime(ok.decision_date) < pd.Timestamp(hold)).all()
    assert a.value.equals(b.value) or np.allclose(a.value.dropna(), b.value.dropna())               # 允许的开发分析输出不受留出期价格影响
    naive = lab[lab.valid & (pd.to_datetime(lab.decision_date) < pd.Timestamp(hold))]; naive2 = lab2[lab2.valid & (pd.to_datetime(lab2.decision_date) < pd.Timestamp(hold))]
    assert len(naive) == len(ok) + h + 1 and not np.allclose(naive.value, naive2.value)         # 旧规则（只看决策日）会放进 h+1 个成熟时点在留出期内的标签，且它们随留出期价格变化
    late = a[(a.invalid_reason == 'holdout_boundary')]; assert len(late) >= h and (pd.to_datetime(late.decision_date) < pd.Timestamp(hold)).sum() == h + 1


def test_dev_labels_without_holdout_is_identity():
    days = [d.date() for d in pd.bdate_range('2024-01-02', periods=12)]; lab = build_labels(price_path(days), days, h = 2)
    pd.testing.assert_frame_equal(dev_labels(lab, None), lab)


def test_pairing_report_and_common_key_restriction():
    base = pd.DataFrame({'model_id': 'm', 'decision_date': D, 'instrument': list('ABCD'), 'score': [4.0, 3.0, 2.0, 1.0], 'split_id': 0, 'fit_asof': date(2023, 12, 29), 'evidence_level': 'x'})
    b = base.iloc[:3].copy(); b = pd.concat([b, b.iloc[[0]]]); b.loc[b.index[1], 'fit_asof'] = date(2023, 12, 28); b.loc[b.index[2], 'score'] = np.inf
    rep = pairing_report(base, b)
    assert (rep['only_in_a'], rep['only_in_b'], rep['common_keys'], rep['duplicates_b'], rep['nonfinite_score_b'], rep['fit_asof_mismatch'], rep['identical_keys']) == (1, 0, 3, 1, 1, 1, False)
    lab = labels([(D, i, v, True, '') for i, v in zip('ABCD', [0.4, 0.3, 0.2, 0.1])]).assign(matured_at = date(2024, 1, 9))
    r = model_pairs(base, b, lab, 2, min_n = 2, top_n = 2, n_boot = 100)['m']
    assert r['restricted_to_common_keys'] and r['pairing']['only_in_a'] == 1 and r['rank_ic']['days'] == 1          # 缺一行不靠内连接静默消失：先报告，再只在共同主键上算


def test_block_bootstrap_keeps_trading_day_positions():
    idx = [d.date() for d in pd.bdate_range('2024-01-02', periods = 120)]
    x = pd.Series(np.random.default_rng(1).normal(0.02, 0.05, 120), index = idx); x.iloc[40:70] = np.nan            # 30 个不可定义的日子
    keep = block_ci(x.to_numpy(), 20, 500, seed = 3); squeezed = block_ci(x.dropna().to_numpy(), 20, 500, seed = 3)
    assert keep is not None and squeezed is not None and keep != squeezed                                            # 缺失日期占位，块不会把不相邻的日子拼在一起
    a, b = x, x + 0.01; s = diff_stats(a, b, 20, 200, 0, grid = idx, alt_block = 40)
    assert (s['days'], s['grid_days'], s['undefined_days']) == (90, 120, 30) and s['diff_mean'] == pytest.approx(0.01) and s['diff_ci95'] == pytest.approx([0.01, 0.01]) and s['alt_block_days'] == 40
    assert block_ci(np.r_[np.full(10, np.nan), np.arange(15.0)], 20) is None                                           # 可定义的日子少于一个块
