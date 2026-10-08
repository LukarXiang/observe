import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch62 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / n for n in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only strategy checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_original_diagnostics_preserve_future_masks_and_failures(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert result['real_historical_operand_windows'] == 0
    assert cases['future_append_changes_past_equal_volume_bar_endpoints']['past_endpoint_changed']
    assert cases['next_day_paused_changes_prior_day_factor_label_mask']['changed'] is None
    assert cases['constructor_M_fails_after_writing_future_labels']['factor_df_missing']
    assert cases['skipped_all_missing_OLS_date_fails_reindex']['error'] == 'KeyError'
    assert cases['sample_std_ddof1_constant_row_NaN']['first'] == [-1., 0., 1.]
    assert cases['residual_BP_variable_undefined_in_fresh_execution']['error'] == 'NameError'


def test_first_assignments_do_not_select_later_neutralized_or_csv_values(monkeypatch):
    trees = originals(monkeypatch); nodes = batch.original_assignments(trees[0])
    assert len(nodes) == 10 and all(n.lineno < 201 for n in nodes.values())
    close = pd.DataFrame({'A': np.arange(501.)+1, 'B': np.arange(501.)+1})
    close.loc[50, 'B'] = np.nan
    actual = batch.compute_values(close, trees[0])
    expected = batch.reference_values(close.B)
    for name in batch.METRICS:
        np.testing.assert_allclose(actual[name].B, expected[name], rtol=1e-12, atol=1e-12, equal_nan=True)
    assert actual['mom_1m'].A.iloc[20] == 20.
    assert np.isnan(actual['mom_1m'].B.iloc[70])
    assert np.isfinite(actual['ma_20'].B.iloc[51])
    assert actual['ma_20'].A.iloc[19] == 10.5 / 20


def test_original_model_parameters_distinguish_training_and_search(monkeypatch):
    trees = originals(monkeypatch); result = batch.model_parameters(trees[1])
    assert len(result['external_imports']) == 4 and not result['training_performed']
    assert result['scalars']['standardize_window'] == result['scalars']['vol_forward_window'] == 21
    active = next(r for r in result['calls'] if r['function'] == 'dict' and 'input_size' in r['keywords'])
    search = next(r for r in result['calls'] if r['function'] == 'optimize_multi_hyperparameters')
    assert active['keywords']['target_vol'] == '1' and search['keywords']['target_vol'] == '0.5'
    assert active['keywords']['transcation_cost'] == '0.0003'
    assert batch.definitions(trees[1]) == {}


def test_selected_rejects_definition_time_execution(monkeypatch):
    for text in ('@unsafe\ndef f(): pass', 'def f(x=open("secret")): pass', 'def f(*, x=open("secret")): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), ['f'], {})


def test_inventory_absent_etf_tables_are_explicit(monkeypatch):
    class Store:
        def __init__(self, root): pass
        def state(self, snapshot): return {'tables': {}}
        def load_state(self, state, table, filters=None): return pd.DataFrame()
    monkeypatch.setattr(batch, 'Store', Store); monkeypatch.setattr(batch, 'dependency_state', lambda r: {})
    result = batch.inventory('unused')
    assert all(not sum(rows.values()) for rows in result['ETF_rows'].values())
    assert result['not_a_backtest']
