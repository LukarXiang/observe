"""统一评价函数（模块 17）：秩 IC 与前 N 名的黄金样本、与 SciPy 逐日对照、留出边界标签规则、成对预测检查、块抽样位置。手工构造的小例子。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from observe.dataset import dev_labels
from observe.evaluation.paired import block_ci, diff_stats, model_pairs, pairing_gate, pairing_report
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


def test_pairing_problems_invalidate_the_primary_result_and_only_feed_a_diagnostic():
    base = pd.DataFrame({'model_id': 'm', 'decision_date': D, 'instrument': list('ABCD'), 'score': [4.0, 3.0, 2.0, 1.0], 'split_id': 0, 'fit_asof': date(2023, 12, 29), 'evidence_level': 'x'})
    b = base.iloc[:3].copy(); b = pd.concat([b, b.iloc[[0]]]); b.loc[b.index[1], 'fit_asof'] = date(2023, 12, 28); b.loc[b.index[2], 'score'] = np.inf
    rep = pairing_report(base, b)
    assert (rep['only_in_a'], rep['only_in_b'], rep['common_keys'], rep['duplicates_b'], rep['nonfinite_score_b'], rep['fit_asof_mismatch'], rep['identical_keys']) == (1, 0, 3, 1, 1, 1, False)
    assert {'duplicate_keys', 'key_sets_differ', 'nonfinite_scores', 'fit_asof_differs_between_sides'} <= set(rep['invalid_reasons'])
    lab = labels([(D, i, v, True, '') for i, v in zip('ABCD', [0.4, 0.3, 0.2, 0.1])]).assign(matured_at = date(2024, 1, 9))
    r = model_pairs(base, b, lab, 2, min_n = 2, top_n = 2, n_boot = 100)['m']
    assert r['valid_primary_comparison'] is False and r['rank_ic'] is None and r['invalid_reasons']             # 不能靠共同主键悄悄得到主结果
    assert r['diagnostic']['kind'] == 'common_subset_diagnostic' and r['diagnostic']['rows_removed']['dropped_duplicate_keys_b'] == 2 and r['diagnostic']['top2_common_subset']['list_redefined'] is True


def pair_frame(scores, split = 0, fit = date(2023, 12, 29), model = 'm', ev = 'x', keys = 'ABCD'):
    return pd.DataFrame({'model_id': model, 'decision_date': D, 'instrument': list(keys), 'score': scores, 'split_id': split, 'fit_asof': fit, 'evidence_level': ev})


PLAN = pd.DataFrame({'split_id': [0], 'fit_asof': [date(2023, 12, 29)], 'test_start': [D], 'test_end': [D]})


def test_golden_pairing_sample_formal_invalid_diagnostic_zero():
    lab = labels([(D, i, v, True, '') for i, v in zip('ABCD', [4, 1, 3, 2])]).assign(matured_at = date(2024, 1, 9))
    a, b = pair_frame([1.0, 2, 3, 4]), pair_frame([1.0, 2, 3, np.nan])
    joined = lambda p: p.merge(lab[['decision_date', 'instrument', 'value']], on = ['decision_date', 'instrument'])
    direct = rank_ic_table(joined(b), min_n = 2).ic.iloc[0] - rank_ic_table(joined(a), min_n = 2).ic.iloc[0]
    assert direct == pytest.approx(-0.1)                                                                             # 旧的直接比较：两侧在不同证券集上算 IC，差约 −0.10
    r = model_pairs(a, b, lab, 2, min_n = 2, top_n = 2, n_boot = 50)['m']
    assert r['valid_primary_comparison'] is False and r['invalid_reasons'] == ['nonfinite_scores'] and r['rank_ic'] is None
    d = r['diagnostic']; assert d['rank_ic']['diff_mean'] == pytest.approx(0) and d['rows_removed']['nonfinite_score_b'] == 1 and d['rows_removed']['kept_rows'] == 3
    ok = model_pairs(a, pair_frame([1.0, 2, 3, 4.5]), lab, 2, min_n = 2, top_n = 2, n_boot = 50)['m']; assert ok['valid_primary_comparison'] is True and ok['rank_ic']['days'] == 1 and ok['diagnostic'] is None


@pytest.mark.parametrize('mutate,reason', [
    (lambda b: b.assign(fit_asof = date(2023, 12, 28)), 'fit_asof_differs_between_sides'),
    (lambda b: b.assign(split_id = 1), 'split_id_differs'),
    (lambda b: b.assign(evidence_level = 'y'), 'evidence_level_differs_between_sides'),
    (lambda b: pd.concat([b, b.iloc[[0]]]), 'duplicate_keys'),
    (lambda b: b.iloc[:3], 'key_sets_differ'),
    (lambda b: b.drop(columns = 'fit_asof'), 'missing_columns_b'),
])
def test_each_pairing_problem_blocks_the_formal_comparison(mutate, reason):
    a = pair_frame([4.0, 3, 2, 1]); b = mutate(pair_frame([4.0, 3, 2, 1]))
    rep = pairing_report(a, b, PLAN); assert any(r.startswith(reason) for r in rep['invalid_reasons'])


def test_fit_asof_must_precede_decision_and_match_the_plan():
    late = pair_frame([4.0, 3, 2, 1], fit = D); rep = pairing_report(late, late, PLAN, 'x')
    assert 'fit_asof_not_before_decision' in rep['invalid_reasons'] and 'fit_asof_differs_from_plan' in rep['invalid_reasons']
    other = pair_frame([4.0, 3, 2, 1], fit = date(2023, 12, 28)); assert 'fit_asof_differs_from_plan' in pairing_report(other, other, PLAN)['invalid_reasons']
    assert 'evidence_level_unexpected' in pairing_report(pair_frame([4.0, 3, 2, 1]), pair_frame([4.0, 3, 2, 1]), PLAN, 'exploratory')['invalid_reasons']
    assert 'decision_outside_test_window' in pairing_report(pair_frame([4.0, 3, 2, 1]), pair_frame([4.0, 3, 2, 1]), PLAN.assign(test_end = date(2023, 12, 30), test_start = date(2023, 12, 30)))['invalid_reasons']
    assert 'unknown_split' in pairing_report(pair_frame([4.0, 3, 2, 1], split = 5), pair_frame([4.0, 3, 2, 1], split = 5), PLAN)['invalid_reasons']
    assert pairing_report(pair_frame([4.0, 3, 2, 1]), pair_frame([4.0, 3, 2, 1]), PLAN, 'x')['invalid_reasons'] == []


def test_model_present_on_one_side_only_is_reported_not_intersected_away():
    a = pd.concat([pair_frame([4.0, 3, 2, 1], model = 'ridge'), pair_frame([4.0, 3, 2, 1], model = 'equal_blend')]); b = pair_frame([4.0, 3, 2, 1], model = 'ridge')
    lab = labels([(D, i, v, True, '') for i, v in zip('ABCD', [4, 1, 3, 2])]).assign(matured_at = date(2024, 1, 9))
    gate = pairing_gate(a, b, lab, lab, ('ridge', 'equal_blend')); assert gate[0].startswith('model_set_differs') and gate[1].startswith('expected_models_missing')
    r = model_pairs(a, b, lab, 2, min_n = 2, top_n = 2, n_boot = 50)
    assert set(r) == {'ridge', 'equal_blend'} and r['equal_blend']['invalid_reasons'] == ['model_only_in_a'] and r['equal_blend']['rank_ic'] is None and r['ridge']['valid_primary_comparison'] is True
    assert pairing_gate(b, b, lab, lab.assign(value = lab.value + 1), ('ridge',)) == ['labels_differ_between_sides']


def test_original_top_n_is_not_reselected_when_the_other_side_lacks_a_security():
    a = pair_frame([1.0, 2, 3, 4, 5], keys = 'ABCDE'); b = pair_frame([1.0, 2, 3, 4], keys = 'ABCD')                # B 侧缺 E，而 E 是 A 侧分数最高的
    lab = labels([(D, i, v, True, '') for i, v in zip('ABCDE', [0.0, 0.0, 0.0, 0.1, 0.5])]).assign(matured_at = date(2024, 1, 9))
    orig = topn_table(a, lab, 1).loc[D]; assert orig.selected_names == ['E'] and orig.mean_label == pytest.approx(0.5)             # 本侧原始名单不受另一侧缺失影响
    r = model_pairs(a, b, lab, 2, min_n = 2, top_n = 1, n_boot = 50)['m']
    assert r['valid_primary_comparison'] is False and 'top1_mean_label' not in r                                                       # 不进入正式主摘要
    d = r['diagnostic']['top1_common_subset']; assert d['list_redefined'] is True and d['a_mean'] == pytest.approx(0.1)             # 共同子集上的名单已重新定义（E 被删，名单变成 D）


def test_block_bootstrap_keeps_trading_day_positions():
    idx = [d.date() for d in pd.bdate_range('2024-01-02', periods = 120)]
    x = pd.Series(np.random.default_rng(1).normal(0.02, 0.05, 120), index = idx); x.iloc[40:70] = np.nan            # 30 个不可定义的日子
    keep = block_ci(x.to_numpy(), 20, 500, seed = 3); squeezed = block_ci(x.dropna().to_numpy(), 20, 500, seed = 3)
    assert keep is not None and squeezed is not None and keep != squeezed                                            # 缺失日期占位，块不会把不相邻的日子拼在一起
    a, b = x, x + 0.01; s = diff_stats(a, b, 20, 200, 0, grid = idx, alt_block = 40)
    assert (s['days'], s['grid_days'], s['undefined_days']) == (90, 120, 30) and s['diff_mean'] == pytest.approx(0.01) and s['diff_ci95'] == pytest.approx([0.01, 0.01]) and s['alt_block_days'] == 40
    assert block_ci(np.r_[np.full(10, np.nan), np.arange(15.0)], 20) is None                                           # 可定义的日子少于一个块
