"""模型（模块 14）：两个基线、Ridge 与可选 LightGBM，接口统一为 fit / predict / save / load。

输入特征是横截面预处理后的值（去极值 → 标准化 → 缺失填 0），方向在因子登记时固定。
predict 时特征名或顺序与训练时不一致直接报错，不静默补列。"""
import importlib.metadata, json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation.ranking import rank_ic      # noqa: F401  秩 IC 只有 evaluation.ranking 一份实现，这里保留旧的导入位置


class FeatureMismatch(ValueError): pass


class _Model:
    kind = ''

    def __init__(self, features, directions, **params):
        self.features, self.directions, self.params, self.info = list(features), dict(directions), params, {}

    def _x(self, X):
        if list(X.columns) != self.features: raise FeatureMismatch(f'特征名或顺序不一致：训练 {self.features}，预测 {list(X.columns)}')
        return X.to_numpy(float)

    def fit(self, X, y, w = None, **info):
        self._x(X); self.info = {**info, 'samples': int(len(X))}; return self

    def describe(self):
        return {'kind': self.kind, 'features': self.features, 'directions': self.directions, 'params': self.params, **self.state(), 'info': self.info,
                'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn')}}

    def state(self): return {}

    def save(self, path): Path(path).write_text(json.dumps(self.describe(), ensure_ascii = False, indent = 1, default = str), encoding = 'utf-8')

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text(encoding = 'utf-8')); cls = KINDS[d['kind']]
        m = cls(d['features'], d['directions'], **d['params']); m.info = d['info']; m.restore(d); return m

    def restore(self, d): pass


class SingleFactor(_Model):
    """基线：直接用一个因子（按登记方向）当分数"""
    kind = 'single_factor'

    def predict(self, X): return self.directions[self.params['factor']] * self._x(X)[:, self.features.index(self.params['factor'])]


class EqualBlend(_Model):
    """基线：全部特征按登记方向等权相加"""
    kind = 'equal_blend'

    def predict(self, X): return self._x(X) @ np.array([self.directions[f] for f in self.features], float) / len(self.features)


class RidgeModel(_Model):
    """scikit-learn Ridge；系数随模型说明一起保存，加载后逐位复现预测。两种惩罚口径，构造时二选一：
    - `alpha = a`：数值 alpha 原样传给 scikit-learn（旧口径，含义不变）；
    - `lam = l`：归一化口径，目标 = Σ w·err² / Σ w + l·‖beta‖²，传给 scikit-learn 的 alpha = l × 本次拟合的 sum_weight（选参拟合与重拟合各按各自的样本权重换算）"""
    kind = 'ridge'

    def __init__(self, features, directions, **params):
        if ('alpha' in params) == ('lam' in params): raise ValueError('RidgeModel 需要且只需要 alpha（旧口径）或 lam（归一化口径）之一')
        super().__init__(features, directions, **params); self.penalty = None

    def fit(self, X, y, w = None, **info):
        from sklearn.linear_model import Ridge
        sw = float(np.sum(w)) if w is not None else float(len(X))
        alpha = self.params['lam'] * sw if 'lam' in self.params else float(self.params['alpha'])
        r = Ridge(alpha = alpha, fit_intercept = True).fit(self._x(X), np.asarray(y, float), sample_weight = None if w is None else np.asarray(w, float))
        self.coef, self.intercept = [float(c) for c in r.coef_], float(r.intercept_)
        self.penalty = {'penalty_mode': 'normalized' if 'lam' in self.params else 'alpha', 'alpha_used': float(alpha), 'sum_weight': sw, 'lambda': float(alpha / sw), 'coef_norm': float(np.linalg.norm(self.coef)), 'features': len(self.features)}
        out = super().fit(X, y, w, **info); self.info.update(self.penalty); return out

    def predict(self, X): return self._x(X) @ np.array(self.coef) + self.intercept

    def state(self): return {'coef': dict(zip(self.features, self.coef)), 'intercept': self.intercept, 'penalty': self.penalty}

    def restore(self, d): self.coef, self.intercept, self.penalty = [d['coef'][f] for f in self.features], d['intercept'], d.get('penalty')


def lightgbm():
    """延迟加载可选依赖；数据读取和三个原有模型不需要 LightGBM。"""
    try: import lightgbm as lgb
    except (ImportError, OSError) as exc:
        raise RuntimeError('LightGBM 不可用：先执行 uv sync --extra ml；macOS 还需要 Homebrew 的 libomp。原始错误：' + str(exc)) from exc
    return lgb


class LGBMModel(_Model):
    """参考 qlib LGBModel.fit：训练与验证分别构建 Dataset，验证窗早停；重拟合用冻结的迭代次数，不再早停。
    Booster 原生文本与说明一起存成 JSON，不使用 pickle；方向只供说明，回归直接拟合统一收益标签。"""
    kind = 'lgbm'

    def __init__(self, features, directions, **params):
        super().__init__(features, directions, **{'seed': 20261001, 'num_threads': 4, **params})

    def fit(self, X, y, w = None, X_valid = None, y_valid = None, w_valid = None, num_boost_round = None, **info):
        lgb = lightgbm(); params = dict(self.params)
        rounds = int(num_boost_round or params.pop('num_boost_round', 200)); params.pop('num_boost_round', None)
        patience = int(params.pop('early_stopping_rounds', 20))
        params.update(objective = 'regression', metric = 'l2', device_type = 'cpu', deterministic = True, force_col_wise = True, verbosity = -1)
        train = lgb.Dataset(self._x(X), label = np.asarray(y, float), weight = None if w is None else np.asarray(w, float), feature_name = self.features)
        valid, callbacks = None, []
        if X_valid is not None:
            if y_valid is None or not len(X_valid): raise ValueError('LightGBM 早停需要非空验证特征与标签')
            valid = [lgb.Dataset(self._x(X_valid), label = np.asarray(y_valid, float), weight = w_valid, reference = train, feature_name = self.features)]
            callbacks = [lgb.early_stopping(patience, first_metric_only = True, verbose = False)]
        self.booster = lgb.train(params, train, num_boost_round = rounds, valid_sets = valid, valid_names = ['valid'] if valid else None, callbacks = callbacks)
        self.best_iteration = int(self.booster.best_iteration or self.booster.current_iteration())
        self.importance = dict(zip(self.features, map(float, self.booster.feature_importance(importance_type = 'gain'))))
        super().fit(X, y, w, **info)
        self.info.update(seed = params['seed'], num_threads = params['num_threads'], deterministic = True, force_col_wise = True,
                         requested_iterations = rounds, best_iteration = self.best_iteration, early_stopping = X_valid is not None,
                         early_stopping_metric = 'validation_weighted_l2' if valid else None)
        return self

    def predict(self, X):
        x = self._x(X)
        return np.asarray(self.booster.predict(x, num_iteration = self.best_iteration, num_threads = self.params['num_threads']), float) if len(x) else np.empty(0)

    def describe(self):
        d = super().describe(); d['versions']['lightgbm'] = importlib.metadata.version('lightgbm'); return d

    def state(self):
        return {'model_text': self.booster.model_to_string(num_iteration = self.best_iteration), 'best_iteration': self.best_iteration, 'importance_gain': self.importance}

    def restore(self, d):
        self.booster = lightgbm().Booster(model_str = d['model_text']); self.best_iteration, self.importance = d['best_iteration'], d['importance_gain']


KINDS = {c.kind: c for c in (SingleFactor, EqualBlend, RidgeModel, LGBMModel)}


def day_weights(dates):
    """每个决策日总权重相同，同一天的证券平分"""
    d = pd.Series(list(dates)); return 1.0 / d.map(d.value_counts()).to_numpy(float)
