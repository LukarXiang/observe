import ast
from pathlib import Path

import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch60 as batch


def original_sources(monkeypatch):
    paths = [Path('repo/量化策略源代码') / n for n in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.parse_source(read_source(p)[0], start) for p, start in zip(paths, batch.CODE_START, strict=True)]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])


def test_original_commodity_diagnostics_preserve_state_and_roll_defects(monkeypatch):
    original_sources(monkeypatch)
    result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert result['real_historical_operand_windows'] == 0
    assert cases['turtle_ATR_first_row_wraps_to_last_close_and_uses_all_rows']['wrapped_true_ranges'] == [11., 3., 9.]
    assert cases['turtle_zero_lot_intent_still_counts_position']['position'] == 1
    assert len(cases['turtle_roll_adjusts_mark_only_and_can_add_same_bar_after_None_orders']['orders']) == 3
    assert cases['MA_short_roll_reads_new_contract_position_and_raises']['error'] == 'KeyError'
    assert cases['MA_mixed_roll_fails_after_long_intents_before_mapping_update']['orders'][0]['kwargs']['side'] == 'long'
    assert cases['MA_stop_does_not_reset_counter_and_old_counter_immediately_unblocks']['reentry_long'] is False
    assert [r['signal'] for r in cases['MA_shared_reentry_flag_blocks_both_symbols']['captured_intents']] == [0, 0]


def test_selected_rejects_decorators_and_eager_defaults(monkeypatch):
    for text in ('@unsafe\ndef f():\n    pass', 'def f(value=open("secret")):\n    pass',
            'def f(*, value=open("secret")):\n    pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'):
            batch.selected(Path('unused'), 0, ['f'], {})


def test_article_prefix_and_all_original_definitions_are_preserved(monkeypatch):
    original_sources(monkeypatch)
    assert len(batch.definitions(batch.source_tree(Path('unused'), 0))) == 11
    assert len(batch.definitions(batch.source_tree(Path('unused'), 1))) == 13
    assert batch.CODE_START == (10, 12)
