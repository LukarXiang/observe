import ast
from pathlib import Path

import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch61 as batch


def original_sources(monkeypatch):
    paths = [Path('repo/量化策略源代码') / n for n in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])


def test_original_diffusion_and_leader_diagnostics(monkeypatch):
    original_sources(monkeypatch)
    result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert result['real_historical_operand_windows'] == 0
    assert cases['ROC100_means_99_calendar_intervals_and_excludes_zero_nan']['ROC'] == {'A': 1., 'D': -.5}
    assert cases['mktcap_is_positive_ROC_magnitude_not_positive_cap_share']['weighted'] == .3125
    assert cases['missing_positive_valuation_silently_skipped']['weighted'] == .5
    assert cases['initial_prepare_then_first_trade_repeats_previous_date_endpoint']['last_values'] == [119., 119.]
    assert cases['leader_original_OLS_array_float_fails']['error'] == 'TypeError'
    assert len(cases['daily_buy_one_can_submit_two_accepted_orders_in_same_loop']['mock_orders']) == 2
    assert cases['leader_counts_at_most_11_non_one_price_limit_days']['max_days'] == 11
    assert cases['leader_negative_label_fails_with_datetime_index']['message'] == '-1'
    assert cases['today_limit_chained_assignment_does_not_update_x1']['retained_limit'] == 11.


def test_selected_rejects_decorators_and_eager_defaults(monkeypatch):
    for text in ('@unsafe\ndef f():\n    pass', 'def f(value=open("secret")):\n    pass',
            'def f(*, value=open("secret")):\n    pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'):
            batch.selected(Path('unused'), 0, ['f'], {})


@pytest.mark.parametrize('text_dates', [False, True])
def test_ROC_component_uses_99_intervals_and_does_not_fill_missing_endpoints(monkeypatch, text_dates):
    original_sources(monkeypatch)
    dates = pd.bdate_range('2020-01-01', periods=101).date
    prices = pd.DataFrame({'date': [str(d) for d in dates] if text_dates else dates, 'instrument': 'A', 'close_adj': range(1, 102)})
    prices['close_adj'] = prices.close_adj.astype(float); prices.loc[1, 'close_adj'] = float('nan')
    monkeypatch.setattr(pd, 'read_parquet', lambda path: pd.DataFrame({'date': dates, 'is_open': True}) if path.name == 'calendar-input.parquet' else prices)
    result = batch.component(Path('unused'))
    assert result['endpoint_windows'] == 1 and result['unavailable_endpoint_windows'] == 1
    assert result['intervals'] == 99 and result['not_a_backtest'] and not result['platform_equivalent']


def test_all_original_definitions_are_frozen(monkeypatch):
    original_sources(monkeypatch)
    assert len(batch.definitions(batch.source_tree(Path('unused'), 0))) == 14
    assert len(batch.definitions(batch.source_tree(Path('unused'), 1))) == 6


def test_inventory_handles_absent_members_and_queries_index_schema(monkeypatch):
    class Store:
        def __init__(self, root): pass
        def state(self, snapshot): return {'tables': {}}
        def load_state(self, state, table, filters=None):
            if table == 'index_1d':
                assert filters == [('index', 'in', ['510300.SH', '000001.SH'])]
                return pd.DataFrame({'index': ['000001.SH'], 'date': [pd.Timestamp('2020-01-01').date()]})
            return pd.DataFrame()
    monkeypatch.setattr(batch, 'Store', Store)
    monkeypatch.setattr(batch, 'dependency_state', lambda root: {})
    result = batch.inventory('unused')
    assert result['HS300_members']['rows'] == 0 and result['HS300_members']['first'] is None
    assert result['original_asset_rows']['index_1d']['000001.SH'] == 1
