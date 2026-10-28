"""Boundary and data-isolation checks for original Batch100 source diagnostics."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch100 as study
from scripts.run_strategy_batch4 import read


def namespace(number):
    return study.literal(Path('repo/量化策略源代码')/study.SOURCE_PATHS[number], number)


def test_source_sha_rejects_other_original(tmp_path):
    path = tmp_path/'source'; path.write_text('def innocent(): return 1\n')
    with pytest.raises(ValueError, match='Source SHA'): study.literal(path, 1)


def test_original_namespace_independent():
    left, right = namespace(0), namespace(0)
    left['getStockPrice'] = lambda *a: (3., 2.)
    assert right['getStockPrice'] is not left['getStockPrice']
    assert right['g'] is not left['g']


def test_original_rolling_rank_is_not_scalar():
    ns = namespace(1)
    assert ns['ts_rank']([1., 2., 2.]).tolist() == [2.5]
    with pytest.raises(TypeError, match='Series'):
        pd.Series(np.arange(20.)).rolling(10).apply(ns['ts_rank'])


def test_rank_count_is_exact_not_highest_percentile():
    ns = namespace(1)
    assert ns['ts_rankeq10'](np.array([10., 9.5, 10., np.nan])) == 2


@pytest.fixture
def operand():
    return study.candidate()[0]


def test_400_row_prefix_and_original_failure(operand, monkeypatch):
    ns = namespace(1)
    # The compiler reads SHA-checked frozen text in production; fixture uses the same original bytes.
    monkeypatch.setattr(study, 'DIRECTORY', Path('data/staging/strategies-batch100/20261011-original-feature-stage'))
    prefix = study.feature_prefix(ns); window = operand.iloc[:400].set_index('date')
    study.set_window(ns, window); context = SimpleNamespace(etf='510050.XSHG')
    out = prefix(context)
    assert len(out) == 400 and 'alpha004' in out and 'alpha005' not in out
    assert out.index.tolist() == window.index.tolist()
    with pytest.raises(TypeError, match='Series'): ns['create_fea'](context)
    cells, error = study.scalar_checks(ns, window, out.iloc[-1])
    assert cells == len(out.columns) and error < 2e-8


def test_future_row_not_exposed_to_daily_callbacks(operand):
    ns = namespace(1); window = operand.iloc[:400].set_index('date'); study.set_window(ns, window)
    opening, closing, high, volume, low = ns['get_price']('510050.XSHG')
    assert all(len(s) == 400 and s.index[-1] < operand.date.iloc[400] for s in (opening, closing, high, volume, low))
    with pytest.raises(ValueError, match='input request'):
        ns['get_price']('wrong_etf')


def test_independent_scalar_audit_rejects_tail_corruption(operand):
    ns = namespace(1); prefix = study.feature_prefix(ns); window = operand.iloc[:400].set_index('date')
    study.set_window(ns, window); tail = prefix(SimpleNamespace(etf='510050.XSHG')).iloc[-1].copy()
    tail['close_mean_rl_20D'] += .01
    with pytest.raises(ValueError, match='Independent feature differs'): study.scalar_checks(ns, window, tail)


def test_original_diagnostics_persist_with_list_shape(tmp_path):
    result = study.diagnostics(); path = tmp_path/'diagnostics.json'; study.save(path, result)
    assert read(path) == result
    rows = {r['case']: r for r in result['cases']}
    assert rows['sector_identity_mapping']['value']['ETFs'] == {
        key: key for key in rows['sector_identity_mapping']['value']['stocks']}
    assert rows['wrong_minute_no_action']['value'] == []
    assert rows['strict_mean_equality_risk_off']['value']['signal'] == 'sell_the_stocks'
    assert rows['all_nan_risk_off']['value']['signal'] == 'sell_the_stocks'
    assert rows['rank_returns_series']['value'] == [2.5]
    assert rows['empty_minute_price']['error'] == 'IndexError'
    assert rows['rolling_rank_current_pandas_fails']['error'] == 'TypeError'


def test_cash_and_partial_sell_are_order_intents():
    rows = {r['case']: r['value'] for r in study.diagnostics()['cases']}
    assert rows['risk_off_two_positions'] == [['target', 'old1', 0], ['unused_price', '511880.XSHG'],
                                             ['target', 'old1', 0], ['buy', '511880.XSHG', 1234.]]
    assert rows['risk_off_immediate_double_sell'][-1] == ['buy', '511880.XSHG', 1434.]
    assert rows['risk_off_money_held'] == []
    assert rows['risk_on_same_held'] == [['unused_price', '511880.XSHG']]


def test_model_thresholds_selected_columns_and_ensemble():
    rows = {r['case']: r for r in study.diagnostics()['cases']}
    for name in ('xgb_upper_equal', 'xgb_lower_equal', 'xgb_ensemble_mean_hold', 'xgb_nan_hold'):
        assert rows[name]['value'] == [['matrix', ['b', 'a'], [[3., 1.]]]]
    assert rows['xgb_buy']['value'][-1] == ['buy', '510050.XSHG', 1234.]
    assert rows['xgb_sell']['value'][-1] == ['target', '510050.XSHG', 0]
    assert rows['xgb_empty_ensemble_error']['error'] == 'IndexError'
    for name in study.MODELS: assert rows['xgb_missing_'+name]['error'] == 'FileNotFoundError'


def test_source_400_row_ma_reseeding_is_observable(operand):
    ns = namespace(1)
    all_values = operand.iloc[:401].set_index('date')
    full = ns['devfea'](all_values[['close', 'open', 'high', 'low', 'volume']].copy(),
        all_values.high, all_values.low, all_values.close, all_values.open, all_values.volume)
    subset = all_values.iloc[1:]
    truncated = ns['devfea'](subset[['close', 'open', 'high', 'low', 'volume']].copy(),
        subset.high, subset.low, subset.close, subset.open, subset.volume)
    # Cumulative AD/OBV encode the window start; continuous-history values would be a different input.
    assert full.AD.iloc[-1] != truncated.AD.iloc[-1]
    assert full.OBV.iloc[-1] != truncated.OBV.iloc[-1]
