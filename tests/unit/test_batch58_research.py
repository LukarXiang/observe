import ast
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch58 as batch


def test_original_gold_stock_diagnostics_preserve_dependency_failures(monkeypatch):
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
    assert cases['probability_singleton_first_character_order']['orders'][-1][1] == '6'
    assert cases['csv_hk_filter_months_and_duplicate_rate']['recommendation_rate'] == 1.5
    assert cases['q_count_float_hardcoded_30m']['query']['frequency'] == '30m'
    assert cases['q_count_float_hardcoded_30m']['count_type'] == 'float'
    assert cases['unused_combination_method_still_blocks']['error'] == 'RuntimeError'
    assert cases['empty_dict_clears_old_holdings']['orders'] == [['target', 'OLD', 0]]


def test_selected_rejects_unreviewed_class_side_effect(monkeypatch):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse('class Q_factor:\n    raise RuntimeError("side effect")'))
    with pytest.raises(ValueError, match='Unsafe class'):
        batch.selected(Path('unused'), 0, ['Q_factor'], {})


def test_vectorized_windows_match_individual_original_windows_and_block_pause(monkeypatch, tmp_path):
    source = Path('repo/量化策略源代码') / batch.SOURCES[0]
    if not source.is_file():
        pytest.skip('Read-only strategy source checkout is not installed')
    tree = ast.parse(read_source(source)[0])
    monkeypatch.setattr(batch, 'validate_sources', lambda directory: None)
    monkeypatch.setattr(batch, 'source_tree', lambda directory, number: tree)
    prices = 10 + np.sin(np.arange(125)/3) + np.arange(125)/30
    frame = pd.DataFrame({'instrument': ['fixture']*125, 'date': pd.date_range('2020-01-01', periods=125).strftime('%Y-%m-%d'),
        'close_adj': prices, 'high_adj': prices*1.03, 'low_adj': prices*.97, 'is_trading': [True]*125})
    frame.loc[1, 'is_trading'] = False
    frame.to_parquet(tmp_path / 'price-input.parquet', index=False)
    summary, calculated = batch.arithmetic(tmp_path)
    ns = batch.selected(tmp_path, 0, ['AF_factor', 'RetN_momentum'], {'pd': pd, 'np': np})
    for length, component in ((22, 'AF20'), (121, 'RM120')):
        for row in calculated[calculated.component.eq(component)].itertuples():
            end = frame.index[frame.date.eq(row.date)][0]
            window = frame.iloc[end-length+1:end+1]
            data = {k: pd.DataFrame({'A': window[k+'_adj'].to_numpy()}) for k in ('close', 'high', 'low')}
            data['paused'] = pd.DataFrame({'A': np.zeros(length)})
            obj = ns['AF_factor']([], 'fixture', 20) if component == 'AF20' else ns['RetN_momentum']([], 'fixture', 120)
            obj.data = data
            expected = obj.v_factor(.5).iloc[0] if component == 'AF20' else obj.calc('lamb', .3).iloc[0]
            assert row.value == expected
    assert summary['profiles'][0]['blocked_windows'] == 2
    assert summary['profiles'][1]['blocked_windows'] == 2
    assert len(calculated) == 105
