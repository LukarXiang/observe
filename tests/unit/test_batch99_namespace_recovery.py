"""Ensure diagnostic paper stubs cannot replace original real-history arithmetic."""
from pathlib import Path

import pandas as pd
import pytest

from scripts import research_strategy_batch99 as stage
from scripts import recover_strategy_batch99_namespace as recovery


def loader(number):
    return stage.definitions(Path('repo/量化策略源代码')/stage.SOURCE_PATHS[number], number)


def operand():
    return pd.DataFrame({'date': ['2018-07-17', '2018-07-18', '2018-07-19', '2018-07-20'],
        'open': [2.47, 2.47, 2.47, 2.474], 'high': [2.487, 2.48, 2.47, 2.553],
        'low': [2.46, 2.445, 2.45, 2.452], 'close': [2.47, 2.46, 2.46, 2.540]})


def test_old_shared_namespace_is_rejected_by_historical_example():
    shared = [loader(0), loader(1)]
    stage.diagnostics(shared)
    frame, summary = stage.components(shared[1], operand())
    assert frame.iloc[0].original_up == 3. and frame.iloc[0].original_down == 2.
    assert summary['rounding_difference_days']['scalar'] == 1


def test_recovery_reloads_definition_before_real_history():
    diagnostics, frame, summary = recovery.compute(operand(), loader)
    assert frame.iloc[0].original_up == pytest.approx(2.482)
    assert frame.iloc[0].original_down == pytest.approx(2.466)
    assert summary['rounding_difference_days']['scalar'] == 0
    assert summary['new_trading_backtests'] == diagnostics['new_trading_backtests'] == 0
    assert diagnostics['case_count'] == 44


def test_recovery_fixture_mutation_cannot_affect_next_compute():
    first = recovery.compute(operand(), loader)
    second = recovery.compute(operand().assign(high=[2.487, 2.48, 2.47, 200.]), loader)
    assert first[0] == second[0]
    assert first[1].original_up.tolist() == second[1].original_up.tolist()
    assert first[1].original_down.tolist() == second[1].original_down.tolist()
