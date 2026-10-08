import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from observe.execution import InputBlocked
from observe.strategy_svm import FEATURES, SVC_DEFAULTS, fit_lagged, lagged_samples


def history():
    days = list(pd.bdate_range('2023-01-02', periods = 252).date)
    k = np.arange(252); close = 20 + 3 * np.sin(k / 12)
    return pd.DataFrame({'close': close, 'high': close + .2, 'low': close - .2, 'volume': 1000 + k * 10}, index = days)


def test_original_offsets_feature_formulas_and_label_cutoffs():
    data = history(); x, y, rows = lagged_samples(data)
    assert x.shape == (225, 7) and y.shape == (225,)
    assert rows[0]['feature_date'] == data.index[22] and rows[0]['feature_start'] == data.index[1]
    assert rows[-1]['feature_date'] == data.index[-6] and rows[-1]['label_end'] == data.index[-1]
    assert rows[-2]['label_end'] == data.index[-2] and sum(r['role'] == 'train' for r in rows) == 224
    for offset, row in zip(range(22, 247), rows, strict = True):
        window = data.iloc[offset - 21:offset + 1]; last = window.iloc[-1]
        expected = [last.close / window.close.mean(), last.volume / window.volume.mean(), last.high / window.high.mean(), last.low / window.low.mean(),
                    last.volume / window.iloc[0].volume, last.close / window.iloc[0].close, window.close.std(ddof = 0)]
        assert np.allclose([row[f] for f in FEATURES], expected, rtol = 1e-12)
        assert row['label'] == int(data.iloc[offset + 5].close > last.close)


def test_one_sample_shape_defect_and_last_observed_label_excluded():
    data = history(); x, y, _ = lagged_samples(data)
    model = SVC(**SVC_DEFAULTS).fit(x[:-1], y[:-1])
    with pytest.raises(ValueError, match = '2D'): model.predict(x[-1])
    prediction, trace, _ = fit_lagged(data, data.index[-1])
    assert prediction == model.predict(x[-1:])[0] and trace['train_samples'] == 224
    assert trace['parameters'] == SVC_DEFAULTS and trace['train_label_end'] == str(data.index[-2])
    changed = data.copy(); changed.iloc[-1, changed.columns.get_loc('close')] *= 100
    again, other, _ = fit_lagged(changed, changed.index[-1])
    assert again == prediction and trace == other


@pytest.mark.parametrize('column,value', [('volume', 0), ('close', np.nan), ('high', np.inf)])
def test_required_training_values_block_without_filling(column, value):
    data = history(); data.iloc[100, data.columns.get_loc(column)] = value
    with pytest.raises(InputBlocked, match = 'missing or nonpositive'): fit_lagged(data, data.index[-1])


def test_one_class_training_and_history_length_block():
    data = history(); data['close'] = np.arange(252) + 20
    with pytest.raises(InputBlocked, match = 'one-class'): fit_lagged(data, data.index[-1])
    with pytest.raises(ValueError, match = '252'): lagged_samples(data.iloc[1:])
    with pytest.raises(ValueError, match = 'decision'): fit_lagged(history(), data.index[-2])
