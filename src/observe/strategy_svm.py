"""Source98's lagged SVC, with only the one-sample prediction shape corrected."""
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .execution import InputBlocked

FEATURES = ('close_mean', 'volume_mean', 'high_mean', 'low_mean', 'volume_first_ratio', 'close_first_ratio', 'close_std')
SVC_DEFAULTS = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
                'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                'probability': 'deprecated', 'random_state': None, 'shrinking': True, 'tol': .001, 'verbose': False}


def lagged_samples(history):
    """252 rows; source deliberately starts at offset22 and holds out offset246."""
    if len(history) != 252: raise ValueError('SVM98 history must contain 252 sessions')
    raw = history[['close', 'high', 'low', 'volume']].to_numpy(float)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise InputBlocked([{'kind': 'svm_invalid_history', 'date': str(history.index[-1]), 'detail': 'Required price/volume window is missing or nonpositive; no filling or no-trade substitute'}])
    features, labels, rows = [], [], []
    for k in range(22, 247):
        close, high, low, volume = raw[k - 21:k + 1].T
        x = [close[-1] / np.mean(close), volume[-1] / np.mean(volume), high[-1] / np.mean(high), low[-1] / np.mean(low),
             volume[-1] / volume[0], close[-1] / close[0], np.std(close, ddof = 0)]
        y = int(raw[k + 5, 0] > raw[k, 0]); features.append(x); labels.append(y)
        rows.append({'feature_date': history.index[k], 'feature_start': history.index[k - 21], 'label_end': history.index[k + 5],
                     'label': y, 'role': 'prediction_only' if k == 246 else 'train', **dict(zip(FEATURES, x, strict = True))})
    return np.array(features), np.array(labels), rows


def fit_lagged(history, decision_day):
    if history.index[-1] != decision_day: raise ValueError('SVM history must end on decision day')
    x, y, rows = lagged_samples(history)
    if len(set(y[:-1])) != 2:
        raise InputBlocked([{'kind': 'svm_single_class', 'date': str(decision_day), 'detail': 'Original SVC cannot fit a one-class training set; no constant-prediction substitute'}])
    clf = SVC(**SVC_DEFAULTS).fit(x[:-1], y[:-1])
    prediction = int(clf.predict(x[-1:])[0])
    model = {'decision_date': str(decision_day), 'feature_date': str(rows[-1]['feature_date']),
             'train_label_end': str(rows[-2]['label_end']), 'prediction_label_end': str(rows[-1]['label_end']),
             'train_samples': len(x) - 1, 'prediction': prediction, 'parameters': clf.get_params(),
             'sklearn_version': importlib.metadata.version('scikit-learn'), 'features': FEATURES,
             'gamma_used': float(clf._gamma), 'support': clf.support_.tolist(), 'support_vectors': clf.support_vectors_.tolist(),
             'dual_coef': clf.dual_coef_.tolist(), 'intercept': clf.intercept_.tolist(), 'n_support': clf.n_support_.tolist(),
             'n_iter': clf.n_iter_.tolist()}
    return prediction, model, [{'decision_date': decision_day, **row} for row in rows]


def generate(wide, mask, days, decision_days, instrument, max_weight):
    history = pd.DataFrame({name: wide[column][instrument] for name, column in
                            (('close', 'close_adj'), ('high', 'high_adj'), ('low', 'low_adj'), ('volume', 'volume'))})
    factors, scores, targets, coverage, models, samples = [], [], [], [], [], []
    weight = 0.; last_feature = None; last_prediction = None
    for day in days:
        fitting = day in decision_days
        if fitting:
            if not mask.at[day, instrument]: raise InputBlocked([{'kind': 'svm_ineligible', 'date': str(day), 'detail': 'Original fixed stock unavailable on fit day'}])
            k = history.index.get_loc(day); prediction, model, rows = fit_lagged(history.iloc[k - 251:k + 1], day)
            models.append(model); samples.extend(rows); weight = max_weight if prediction else 0.
            last_feature = rows[-1]['feature_date']; last_prediction = prediction
            scores.append({'decision_date': day, 'instrument': instrument, 'score': float(prediction)})
        factors.append({'date': day, 'instrument': instrument, 'is_fit': fitting, 'feature_date': last_feature,
                        'prediction': last_prediction, 'weight': weight})
        targets += [{'decision_date': day, 'instrument': i, 'weight': w, 'cash_weight': 1 - weight, 'rebalance': fitting}
                    for i, w in ((instrument, weight), ('CASH', 1 - weight))]
        coverage.append({'date': day, 'candidates': int(mask.at[day, instrument]), 'fit_samples': 224 if fitting else 0,
                         'is_fit': fitting, 'selected': int(weight > 0), 'cash_weight': 1 - weight})
    if not models: raise InputBlocked([{'kind': 'svm_no_schedule', 'detail': 'No weekly third-session execution in requested interval'}])
    return factors, scores, targets, coverage, models, samples
