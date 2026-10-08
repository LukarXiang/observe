import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib
from talib import abstract

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch64 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.parsed_source(read_source(p)[0], i)[0] for i, p in enumerate(paths)]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, i: trees[i])
    return trees


def prices(dates):
    base = np.linspace(10, 12, len(dates))
    frame = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'instrument': 'A', 'is_trading': True})
    for name, offset in zip(batch.OHLC, (0, 1, -1, .25), strict=True):
        frame[name] = base+offset; frame[f'{name}_adj'] = base+offset
    return frame


def test_original_definitions_normalization_and_pattern_calls(monkeypatch):
    trees = originals(monkeypatch)
    assert len(batch.definitions(trees[0])) == 7 and len(batch.definitions(trees[1])) == 5
    text = read_source(Path('repo/量化策略源代码') / batch.SOURCES[0])[0]
    with pytest.raises(SyntaxError): ast.parse(text)
    _, normalization = batch.parsed_source(text, 0)
    assert [r['line'] for r in normalization['edits']] == [563]
    assert set(batch.pattern_names(trees[1])) == set(talib.get_function_groups()['Pattern Recognition'])


def test_original_price_shape_and_pattern_diagnostics(monkeypatch):
    originals(monkeypatch)
    folder = Path('data/staging/strategies-batch64/20261007-price-shape-candles')
    if not (folder / 'source-reviews/review.json').exists():
        from scripts.run_strategy_batch4 import read
        original_read = read
        monkeypatch.setattr(batch, 'read', lambda p: {'sources': [{'source_copy': str(Path('repo/量化策略源代码') / batch.SOURCES[0])}]}
            if Path(p).name == 'review.json' else original_read(p))
    result = batch.diagnostics(folder); cases = {r['case']: r for r in result['cases']}
    assert result['not_a_backtest'] and result['real_historical_trade_windows'] == 0
    assert cases['legacy_print_syntax']['line'] == 563
    assert cases['IPO_filter_strict_90_calendar_days']['selected'] == ['B']
    assert cases['IC_persistent_frame_drops_new_universe_assets']['correlation_indices'][-1] == ['B', 'C']
    assert cases['find_pattern_first_nonzero_not_latest_and_keeps_negative']['stock'] == 'A'
    assert cases['discern_pattern_datetime_series_minus1_is_label']['error'] == 'KeyError'


def test_shape_preserves_original_nan_skipping_and_tracks_partial_windows(monkeypatch):
    tree = originals(monkeypatch)[0]; dates = pd.bdate_range('2020-01-01', periods=22)
    frame = prices(dates).drop(index=[3])
    values, audit = batch.shape_values(frame, dates, tree)
    assert list(values.columns) == ['window', 'date', 'instrument', 'window_rows', 'HighOpen', 'HighOpen_finite_rows', 'CloseLow', 'CloseLow_finite_rows']
    assert audit['max_abs_error'] < 1e-12 and 'VwapClose' not in values
    row = values[(values.window == 20) & (values.date == dates[19].strftime('%Y-%m-%d'))].iloc[0]
    assert row.HighOpen_finite_rows == row.CloseLow_finite_rows == 19
    assert row.HighOpen == pytest.approx(np.log(frame.iloc[:19].high/frame.iloc[:19].open).mean())


def test_pattern_window_skips_only_known_pauses_and_checks_two_library_apis():
    dates = pd.bdate_range('2020-01-01', periods=103); frame = prices(dates); frame.loc[2, 'is_trading'] = False
    names = ['CDLDOJI', 'CDLENGULFING']; values, result = batch.pattern_values(frame, dates, names)
    assert result['statuses']['known_paused_skipped'] == 1
    assert result['statuses']['eligible'] == 3 and result['direct_abstract_matched_cells'] == 6
    assert values.iloc[0].window_status == 'insufficient_100_rows' and pd.isna(values.iloc[0].CDLDOJI)
    window = frame[frame.is_trading].tail(100)
    for name in names:
        expected = abstract.Function(name)({n: window[f'{n}_adj'].to_numpy() for n in batch.OHLC})[-1]
        assert values[name].iloc[-1] == expected


def test_unknown_calendar_gap_blocks_patterns_without_zero_signal_substitution():
    dates = pd.bdate_range('2020-01-01', periods=105); frame = prices(dates).drop(index=[50])
    values, result = batch.pattern_values(frame, dates, ['CDLDOJI'])
    assert result['statuses']['unknown_internal_calendar_day'] == 5
    assert result['statuses']['eligible'] == 0 and values.CDLDOJI.isna().all()


def test_selected_rejects_definition_time_side_effects(monkeypatch):
    for text in ('@unsafe\ndef f(): pass', 'def f(x=open("secret")): pass', 'def f(*, x=open("secret")): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_selected_allows_only_literal_numeric_default_multiplication(monkeypatch):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse('def f(n=30*3): return n'))
    assert batch.selected(Path('unused'), 0, ['f'], {})['f']() == 90
    for text in ('def f(n=30*unsafe()): pass', 'def f(n="x"*3): pass', 'def f(n=30*external): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})
