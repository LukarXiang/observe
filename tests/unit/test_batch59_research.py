import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.data.store import fingerprint
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch59 as batch


def original_sources(monkeypatch):
    paths = [Path('repo/量化策略源代码') / n for n in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])


def test_original_IF_diagnostics_preserve_reverse_and_none_order_defects(monkeypatch):
    original_sources(monkeypatch)
    result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert cases['td_none_entry_still_updates_flag_and_hold']['flag'] == 1
    assert len(cases['td_long_reversal_duplicates_close_intent']['orders']) == 3
    assert cases['td_reverse_exit_setup_1_wrong_side']['orders'][0]['args'][-1] == 'long'
    assert cases['td_reverse_exit_setup_-1_wrong_side']['orders'][0]['args'][-1] == 'short'
    assert cases['td_increment_to_thirteen_skips_cancellation']['after']['count'] == 13
    assert cases['intraday_contract_index_three_requires_four']['error'] == 'IndexError'
    assert cases['intraday_close_helper_returns_after_first_eligible']['remaining_eligible'] == ['B']


def test_TD_seeds_match_original_and_do_not_promote_prices_to_trades(monkeypatch, tmp_path):
    original_sources(monkeypatch)
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=30).strftime('%Y-%m-%d'),
        'close': 100+np.sin(np.arange(30))*10, 'high': 111+np.sin(np.arange(30))*10, 'low': 89+np.sin(np.arange(30))*10})
    frame.to_parquet(tmp_path / 'index-input.parquet', index=False)
    summary, states, ratios = batch.arithmetic(tmp_path)
    assert summary['td_state_cases'] == 18*7 and summary['td_state_differences'] == 0
    assert summary['ma_windows'] == 25 and len(ratios)==25
    assert summary['not_a_backtest'] and not summary['platform_equivalent']
    frame.loc[15, 'close'] = np.nan; frame.to_parquet(tmp_path / 'index-input.parquet', index=False)
    summary, states, ratios = batch.arithmetic(tmp_path)
    assert len(summary['blocked_dates']['TD13']) == 13
    assert len(summary['blocked_dates']['MA6']) == 6
    assert len(states)==5*7 and len(ratios)==19


def test_selected_rejects_eager_default_calls(monkeypatch):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse('def unsafe(value=open("secret")):\n    pass'))
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected(Path('unused'), 0, ['unsafe'], {})


def test_index_binding_distinguishes_registered_content_hash_from_file_hash(monkeypatch, tmp_path):
    frame = pd.DataFrame({'date': pd.date_range('2005-01-05', periods=5280, freq='B').date,
        'index': ['000300.SH']*5280, 'close': np.arange(5280, dtype=float)+100.})
    path = tmp_path / 'index.parquet'; frame.to_parquet(path, index=False)
    content_sha = fingerprint(pd.read_parquet(path))
    assert content_sha != file_sha(path)
    state = {'tables': {'index_1d': {'all': {'file': path.name, 'sha': content_sha, 'rows': 5280}}}}
    store = SimpleNamespace(state=lambda *a: state, load_state=lambda st, table: frame.copy() if table=='index_1d' else pd.DataFrame())
    monkeypatch.setattr(batch, 'Store', lambda root: store)
    monkeypatch.setattr(batch, 'sessions', lambda frame: pd.date_range('2005-01-05', periods=5280, freq='B').date)
    result, bound = batch.index_operand(tmp_path)
    assert len(result)==5280
    assert bound['partitions'][0]['sha256'] == file_sha(path)
    assert bound['partitions'][0]['content_fingerprint'] == content_sha
