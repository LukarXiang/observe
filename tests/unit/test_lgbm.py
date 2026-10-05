"""LightGBM 的确定性、冻结迭代次数与训练时间边界。"""
import numpy as np
import pandas as pd
import pytest

from observe.models import FeatureMismatch, LGBMModel, _Model, day_weights
from observe.research import ModelConfig, selection_stage
from tests.unit.test_selection_guard import DIRS, NAMES, world

pytest.importorskip('lightgbm')


def test_save_load_determinism_and_fixed_iteration_refit(tmp_path):
    rng = np.random.default_rng(8); X = pd.DataFrame(rng.normal(size = (600, 3)), columns = ['a', 'b', 'c']); y = X.a ** 2 - X.b + rng.normal(scale = .1, size = len(X))
    params = {'num_threads': 1, 'seed': 7, 'num_leaves': 7, 'min_data_in_leaf': 10, 'learning_rate': .1, 'num_boost_round': 40, 'early_stopping_rounds': 5}
    weights = day_weights(np.repeat(np.arange(30), 20))
    m = LGBMModel(list(X), dict.fromkeys(X, 1), **params).fit(X[:400], y[:400], weights[:400], X_valid = X[400:], y_valid = y[400:], w_valid = weights[400:])
    iterations = m.best_iteration
    assert 1 <= iterations <= 40 and m.info['early_stopping'] and m.info['deterministic'] and m.info['force_col_wise']
    m.fit(X, y, weights, num_boost_round = iterations)
    assert not m.info['early_stopping'] and m.info['requested_iterations'] == iterations and m.best_iteration <= iterations
    path = tmp_path / 'lgbm.json'; m.save(path); restored = _Model.load(path)
    assert np.array_equal(m.predict(X), restored.predict(X)) and m.importance == restored.importance
    repeat = LGBMModel(list(X), dict.fromkeys(X, 1), **params).fit(X, y, weights, num_boost_round = iterations)
    assert np.array_equal(m.predict(X), repeat.predict(X))
    with pytest.raises(FeatureMismatch): restored.predict(X[['b', 'a', 'c']])
    with pytest.raises(FeatureMismatch): restored.predict(X[['a', 'b']])
    assert restored.predict(X[:0]).shape == (0,)


def test_test_labels_cannot_change_selection_or_early_stopping():
    X, lab, sp = world()
    mc = ModelConfig(baseline_factor = 'f1', min_names = 10, ridge_alphas = [1.0], num_threads = 1,
                     lgbm = [{'num_boost_round': 20, 'early_stopping_rounds': 3, 'num_leaves': 4, 'min_data_in_leaf': 10}])
    original = selection_stage(sp, X, lab, NAMES, DIRS, mc)
    future = lab.copy(); hit = future.decision_date >= sp.test_start; future.loc[hit, 'value'] += 1e6; future.loc[hit, 'valid'] = False
    modified = selection_stage(sp, X, future, NAMES, DIRS, mc)
    assert original[-2:] == modified[-2:]
    for a, b in zip(original[-3], modified[-3]):
        assert a[0] == b[0] and np.array_equal(a[1].predict(X[a[1].features]), b[1].predict(X[b[1].features]))


def test_grid_and_runtime_parameters_are_constrained():
    with pytest.raises(ValueError): ModelConfig(lgbm = [{}] * 13)
    with pytest.raises(ValueError): ModelConfig(lgbm = [{'deterministic': False}])
    with pytest.raises(ValueError): ModelConfig(num_threads = 0)
