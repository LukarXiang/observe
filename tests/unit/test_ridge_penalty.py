"""归一化 Ridge 正则口径（决策 23）：目标 = 加权平均损失 + lambda·‖beta‖²，alpha = lambda × sum_weight；旧 alpha 口径含义不变。手工构造的小例子，Ridge 求解一律直接调用 scikit-learn。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from observe.models import RidgeModel, _Model, day_weights
from observe.research import ModelConfig, candidate_models

FEATS = ['a', 'b', 'c']; DIRS = {f: 1 for f in FEATS}


def data(n_days = 30, per = 20, seed = 0):
    rng = np.random.default_rng(seed); dates = np.repeat(np.arange(n_days), per); X = pd.DataFrame(rng.normal(0, 1, (n_days * per, 3)), columns = FEATS)
    y = 0.3 * X.a - 0.2 * X.b + rng.normal(0, 1, len(X)); return X, y.to_numpy(), day_weights(dates)


def test_normalized_penalty_passes_lambda_times_sum_weight_to_scikit_learn():
    X, y, w = data(); m = RidgeModel(FEATS, DIRS, lam = 0.5).fit(X, y, w)
    ref = Ridge(alpha = 0.5 * w.sum()).fit(X.to_numpy(), y, sample_weight = w)
    assert m.penalty['alpha_used'] == pytest.approx(0.5 * 30) and m.penalty['sum_weight'] == pytest.approx(30) and m.penalty['lambda'] == pytest.approx(0.5) and m.penalty['penalty_mode'] == 'normalized'
    assert np.allclose(m.coef, ref.coef_, atol = 1e-12) and m.intercept == pytest.approx(ref.intercept_)


def test_solution_minimises_the_stated_objective():
    X, y, w = data(seed = 1); lam = 2.0; m = RidgeModel(FEATS, DIRS, lam = lam).fit(X, y, w); beta, b = np.array(m.coef), m.intercept
    err = y - X.to_numpy() @ beta - b; grad_beta = -2 * X.to_numpy().T @ (w * err) / w.sum() + 2 * lam * beta; grad_b = -2 * (w * err).sum() / w.sum()      # J = Σw·err²/Σw + lam·‖beta‖²，截距不惩罚
    assert np.abs(grad_beta).max() < 1e-9 and abs(grad_b) < 1e-9


def test_scaling_all_weights_leaves_normalized_predictions_unchanged_but_not_the_old_mode():
    X, y, w = data(seed = 2); test = X.iloc[:50]
    n1, n2 = RidgeModel(FEATS, DIRS, lam = 3.0).fit(X, y, w), RidgeModel(FEATS, DIRS, lam = 3.0).fit(X, y, 7.0 * w)
    assert np.allclose(n1.predict(test), n2.predict(test), atol = 1e-10) and n2.penalty['alpha_used'] == pytest.approx(7 * n1.penalty['alpha_used'])     # 权重乘常数、alpha 同比换算：预测一致
    o1, o2 = RidgeModel(FEATS, DIRS, alpha = 90.0).fit(X, y, w), RidgeModel(FEATS, DIRS, alpha = 90.0).fit(X, y, 7.0 * w)
    assert not np.allclose(o1.predict(test), o2.predict(test), atol = 1e-6)                                                                         # 旧口径数值 alpha 固定，随权重尺度改变强度
    ref = Ridge(alpha = 90.0).fit(X.to_numpy(), y, sample_weight = w); assert np.allclose(o1.coef, ref.coef_, atol = 1e-12) and o1.penalty['penalty_mode'] == 'alpha'   # 旧口径含义不变


def test_selection_fit_and_refit_are_converted_with_their_own_sum_weight():
    X, y, w = data(n_days = 20); m = RidgeModel(FEATS, DIRS, lam = 1.0); m.fit(X, y, w); first = dict(m.penalty)
    X2, y2, w2 = data(n_days = 30, seed = 3); m.fit(X2, y2, w2)                                                     # 同一个对象先按选参样本拟合、再按 train+valid 重拟合
    assert first['sum_weight'] == pytest.approx(20) and m.penalty['sum_weight'] == pytest.approx(30) and m.penalty['alpha_used'] == pytest.approx(30.0) and first['alpha_used'] == pytest.approx(20.0)
    assert m.penalty['lambda'] == pytest.approx(1.0) and m.info['sum_weight'] == pytest.approx(30) and m.info['alpha_used'] == pytest.approx(30.0)     # lambda 保持，alpha 随样本权重换算


def test_save_load_round_trip_keeps_mode_lambda_and_predictions(tmp_path):
    X, y, w = data(seed = 4); m = RidgeModel(FEATS, DIRS, lam = 0.1).fit(X, y, w); path = tmp_path / 'm.json'; m.save(path)
    m2 = _Model.load(path); assert isinstance(m2, RidgeModel) and m2.params == {'lam': 0.1} and m2.penalty == m.penalty and np.array_equal(m2.predict(X), m.predict(X))
    old = RidgeModel(FEATS, DIRS, alpha = 5.0).fit(X, y, w); old.save(path); assert _Model.load(path).params == {'alpha': 5.0}


def test_exactly_one_penalty_parameter_and_candidate_names_by_mode():
    with pytest.raises(ValueError, match = 'alpha.*lam'): RidgeModel(FEATS, DIRS)
    with pytest.raises(ValueError, match = 'alpha.*lam'): RidgeModel(FEATS, DIRS, alpha = 1.0, lam = 1.0)
    old = candidate_models(ModelConfig(), FEATS, DIRS); assert [k for k, _ in old if k.startswith('ridge')] == ['ridge@1', 'ridge@10', 'ridge@100', 'ridge@1000', 'ridge@10000']        # 默认仍是旧口径
    new = candidate_models(ModelConfig(penalty_mode = 'normalized'), FEATS, DIRS); assert [k for k, _ in new if k.startswith('ridge')] == [f'ridge_norm@{x:g}' for x in (0.01, 0.1, 1, 10, 100, 1000)]
    assert all('lam' in m.params for k, m in new if k.startswith('ridge')) and all('alpha' in m.params for k, m in old if k.startswith('ridge'))
