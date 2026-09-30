"""选参阶段的边界防护：全部候选验证秩 IC 不可定义时阻断该窗口，而不是用 max(-inf, ...) 选第一个 alpha；并记录有效验证日与不可定义原因。"""
import numpy as np
import pandas as pd
import pytest

from observe.dataset import plan_splits
from observe.research import ModelConfig, SelectionUndefined, selection_stage

NAMES = ['f1', 'f2']; DIRS = {'f1': 1, 'f2': 1}


def world(n_days = 40, n = 30, seed = 0, valid = True):
    rng = np.random.default_rng(seed); dates = [d.date() for d in pd.bdate_range('2024-01-02', periods = n_days)]; rows, labs = [], []
    for d in dates:
        f = rng.normal(0, 1, (n, 2)); y = 0.05 * f[:, 0] + rng.normal(0, 0.1, n)
        for i in range(n):
            rows.append({'date': d, 'instrument': f'{i:06d}.SH', 'f1': f[i, 0], 'f2': f[i, 1]})
            labs.append({'decision_date': d, 'instrument': f'{i:06d}.SH', 'value': y[i], 'valid': valid, 'matured_at': dates[min(dates.index(d) + 2, n_days - 1)] if dates.index(d) + 2 < n_days else pd.NaT})
    plan = plan_splits(dates, 20, 10, 5); return pd.DataFrame(rows), pd.DataFrame(labs), next(plan.itertuples(index = False))


def mc(**kw): return ModelConfig(baseline_factor = 'f1', ridge_alphas = [1.0, 10.0], correlation_threshold = 0.99, min_names = kw.pop('min_names', 10), top_n = 5, **kw)


def test_normal_window_selects_a_ridge_alpha_and_records_validation_evidence():
    X, lab, sp = world(); *_, tried, best = selection_stage(sp, X, lab, NAMES, DIRS, mc())
    ridge = [t for t in tried if t['candidate'].startswith('ridge@')]
    assert best in {'ridge@1', 'ridge@10'} and all(t['valid_days'] > 0 and t['valid_pairs_mean'] >= 10 and t['select_rows'] > 0 and t['valid_rows'] > 0 for t in ridge)
    assert best == max(ridge, key = lambda t: t['valid_rank_ic'])['candidate']


@pytest.mark.parametrize('case', ['no_valid_labels', 'too_few_names', 'constant_predictions'])
def test_undefined_validation_blocks_the_window_instead_of_picking_the_first_alpha(case):
    X, lab, sp = world(valid = case != 'no_valid_labels'); m = mc(min_names = 1000 if case == 'too_few_names' else 10)
    if case == 'constant_predictions': X[NAMES] = 0.0
    with pytest.raises(SelectionUndefined) as e: selection_stage(sp, X, lab, NAMES, DIRS, m)
    reasons = {r for t in e.value.tried if t['candidate'].startswith('ridge@') for r in t['undefined']}
    assert reasons == {'no_valid_labels': set(), 'too_few_names': {'too_few_valid'}, 'constant_predictions': {'constant_score'}}[case]
    assert all(t['valid_rank_ic'] is None and t['valid_days'] == 0 for t in e.value.tried if t['candidate'].startswith('ridge@')) and (case != 'no_valid_labels' or all(t['valid_rows'] == 0 for t in e.value.tried))     # 标签全无效：验证样本为空，没有任何一天可评价


def test_empty_validation_sample_is_undefined_not_a_crash():
    X, lab, sp = world(); lab = lab[lab.decision_date < sp.valid_start]                         # 验证窗内一个标签都没有
    with pytest.raises(SelectionUndefined) as e: selection_stage(sp, X, lab, NAMES, DIRS, mc())
    assert all(t['valid_rows'] == 0 for t in e.value.tried)
