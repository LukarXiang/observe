import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch63 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / n for n in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse('\n'.join(read_source(p)[0].splitlines()[8:])) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_original_financial_diagnostics(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['real_model_fits'] == 0 and result['synthetic_model_fits'] == 1
    assert cases['original_SVR_set_indexer_fails_before_fit']['days'] == 0
    assert cases['five_year_FCF_keeps_only_oldest_nonempty_year_not_intersection']['selected'] == ['B']
    assert cases['float_quarter_suffix_and_removed_Panel']['requests'] == ['2025q1.0', '2025q1.-1']
    assert cases['value_before_open_repeats_entire_selection']['calls'] == ['selection', 'selection']
    assert cases['EPS_level_not_growth_strict_008_05']['selected'] == ['B', 'C']


def test_real_arithmetic_does_not_fill_unknown_model_features(monkeypatch):
    trees = originals(monkeypatch)
    frame = pd.DataFrame({'total_assets': [10., 10., 1., np.nan], 'total_liabilities': [5., 0., 2., 1.],
        'net_profit_ytd': [0., -2., -1., np.nan], 'operating_cashflow_ytd': [10., 2., -1., np.nan],
        '开发支出': ['0', '2', '-1', ''], '投资活动产生的现金流量净额': ['-5', '3', '0', ''],
        '流动资产合计': ['10', '5', '', '2'], '流动负债合计': ['2', '0', '1', '1']})
    with np.errstate(all='ignore'):
        actual, flags, _, parsed = batch.component_frame(frame, trees); expected = batch.reference_frame(frame, parsed)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert list(actual) == batch.METRICS
    assert actual.NI_p.iloc[0] == -10000. and actual.NI_n.iloc[2] == 0.
    assert actual.LEV.iloc[1] == 10000. and np.isnan(actual.CR.iloc[1])
    assert actual.FCF.iloc[0] == 15. and actual.FCF.iloc[1] == -1.
    assert flags.input_missing_log_NC.iloc[3] and actual.log_NC.iloc[3] == 0.


def test_selected_rejects_definition_time_side_effects(monkeypatch):
    for text in ('@unsafe\ndef f(): pass', 'def f(x=open("secret")): pass', 'def f(*, x=open("secret")): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_original_definitions_are_complete(monkeypatch):
    trees = originals(monkeypatch)
    assert len(batch.definitions(trees[0])) == 4
    assert len(batch.definitions(trees[1])) == 9


def test_financial_unit_evidence_has_json_null_and_roundtrips(monkeypatch):
    originals(monkeypatch)
    folder = Path('data/staging/strategies-batch63/20261007-SVR-value-financial')
    if not (folder / 'financial-input.parquet').is_file(): pytest.skip('Frozen financial projection is not installed')
    frame = pd.read_parquet(folder / 'financial-input.parquet').head(8)
    fields = pd.read_parquet(folder / 'financial-fields.parquet')
    monkeypatch.setattr(pd, 'read_parquet', lambda p: fields if Path(p).name == 'financial-fields.parquet' else frame)
    result, _ = batch.component(folder)
    assert any(r['standard_name'] is None for r in result['unit_evidence'])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
