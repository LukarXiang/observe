"""模型（模块 14）：两个基线（单因子、等权合成）与 Ridge，接口统一为 fit / predict / save / load。

输入特征是横截面预处理后的值（去极值 → 标准化 → 缺失填 0），方向在因子登记时固定。
predict 时特征名或顺序与训练时不一致直接报错，不静默补列。"""
import importlib.metadata, json
from pathlib import Path

import numpy as np
import pandas as pd


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
    """scikit-learn Ridge；系数随模型说明一起保存，加载后逐位复现预测"""
    kind = 'ridge'

    def fit(self, X, y, w = None, **info):
        from sklearn.linear_model import Ridge
        r = Ridge(alpha = self.params['alpha'], fit_intercept = True).fit(self._x(X), np.asarray(y, float), sample_weight = None if w is None else np.asarray(w, float))
        self.coef, self.intercept = [float(c) for c in r.coef_], float(r.intercept_)
        return super().fit(X, y, w, **info)

    def predict(self, X): return self._x(X) @ np.array(self.coef) + self.intercept

    def state(self): return {'coef': dict(zip(self.features, self.coef)), 'intercept': self.intercept}

    def restore(self, d): self.coef, self.intercept = [d['coef'][f] for f in self.features], d['intercept']


KINDS = {c.kind: c for c in (SingleFactor, EqualBlend, RidgeModel)}


def day_weights(dates):
    """每个决策日总权重相同，同一天的证券平分"""
    d = pd.Series(list(dates)); return 1.0 / d.map(d.value_counts()).to_numpy(float)


def rank_ic(frame, score = 'score', label = 'value', min_n = 30):
    """每日秩相关（斯皮尔曼）；有效证券少于 min_n 或任一侧为常数的日子记缺失。返回按日期的序列"""
    def one(g):
        if len(g) < min_n or g[score].nunique() < 2 or g[label].nunique() < 2: return np.nan
        return g[score].rank().corr(g[label].rank())
    return frame.groupby('decision_date')[[score, label]].apply(one)
