import ast
from pathlib import Path

import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch56 as batch


def test_original_diagnostics_reach_bond_price_and_five_row_label(monkeypatch):
    sources = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(path.is_file() for path in sources):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(path) for path in sources] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(path)[0]) for path in sources]
    monkeypatch.setattr(batch, 'validate_sources', lambda directory: None)
    monkeypatch.setattr(batch, 'source_tree', lambda directory, number: trees[number])
    result = batch.diagnostics(Path('unused'))
    cases = {row['case']: row for row in result['cases']}
    assert result['not_a_backtest'] and not result['platform_equivalent']
    assert cases['premium_unsorted_last_row_and_inverse_fit']['rates_are_explicit_synthetic_ratios']
    assert cases['correlation_object_date_aggregation_failure']['error'] == 'TypeError'
    assert cases['correlation_fixed_december_start_and_five_row_label']['label_pairs'] == 5
    assert len(cases['correlation_fixed_december_start_and_five_row_label']['queries']) == 24
    assert cases['fixed_pair_two_rows_still_uses_quantiles']['actual_rows'] == 2
