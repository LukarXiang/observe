import ast
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, read_source
from scripts import review_strategy_batch68 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.parsed_source(read_source(p)[0], i)[0] for i, p in enumerate(paths)]
    monkeypatch.setattr(batch, 'source_tree', lambda d, i: trees[i])
    return trees


def operands(day, n=100):
    return pd.DataFrame({'date': [day]*n, 'instrument': [f'S{i:03d}' for i in range(n)],
        'pb_mrq': np.arange(n)-3., 'turnover': np.arange(n, 0, -1)/100,
        'volume': np.arange(1, n+1)*100., 'close': 10., 'is_trading': True})


def test_complete_sources_and_preamble_preserves_line_numbers(monkeypatch):
    trees = originals(monkeypatch)
    assert len(batch.definitions(trees[0])) == 9
    assert len(batch.definitions(trees[1])) == 18
    text = read_source(Path('repo/量化策略源代码') / batch.SOURCES[1])[0]
    _, normalization = batch.parsed_source(text, 1)
    assert len(normalization['removed_preamble']) == 8
    assert normalization['original_line_numbers_retained']
    assert trees[1].body[0].lineno == 9
    for name in batch.SOURCES:
        assert REVIEWS[name]['review_status'] == '人工审查完成'
        assert REVIEWS[name]['status'] == '待数据'


def test_original_synthetic_diagnostics_and_json_roundtrip(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused'))
    cases = {row['case']: row for row in result['cases']}
    assert result == json.loads(json.dumps(result))
    assert cases['filter_after35_no_refill']['count'] == 34
    assert cases['sell_all_including_retained_then_read_cash_once']['buy_intents'] == [['A', 600.], ['B', 600.]]
    assert cases['empty_targets_no_division_because_loop_empty']['orders'] == []
    assert cases['date_series_zero_is_label']['error'] == 'KeyError'
    assert cases['range30_and_drawdown4_not_entry_profit_gate']['orders'] == [['A', 0]]
    assert cases['optional_risk_branch_unbound_sell_lists']['error'] == 'UnboundLocalError'
    assert cases['hold_day83_delegates_unknown_unit_and_sells_retained']['helper_argument'] == 83
    assert cases['removed_DataFrame_sort_after_external_scores']['error'] == 'AttributeError'
    assert result['not_a_backtest'] and result['real_historical_trade_windows'] == 0


def test_monthly_dates_use_real_calendar_and_previous_session():
    calendar = pd.DataFrame({'date': pd.to_datetime(['2021-01-29', '2021-01-30', '2021-02-01',
        '2021-02-02', '2021-02-26', '2021-03-01']), 'is_open': [True, False, True, True, True, True]})
    assert batch.monthly_schedule(calendar) == [(dt.date(2021, 1, 29), dt.date(2021, 2, 1)),
        (dt.date(2021, 2, 26), dt.date(2021, 3, 1))]


def test_monthly_duplicate_calendar_rejected():
    calendar = pd.DataFrame({'date': ['2021-02-01']*2, 'is_open': True})
    with pytest.raises(ValueError, match='duplicate'): batch.monthly_schedule(calendar)


def test_original_PB70_then_turnover35_includes_nonpositive_without_day_filter(monkeypatch):
    tree = originals(monkeypatch)[0]; before = dt.date(2021, 1, 29); day = dt.date(2021, 2, 1)
    frame = operands(before); frame['turnover'] = np.arange(100)/100+.01
    values, summary = batch.numeric_values(frame, [(before, day)], tree)
    selected = values[values.branch.eq('original_PB_prescreen')]
    assert list(selected.instrument) == [f'S{i:03d}' for i in range(35)]
    assert selected.pb_mrq.le(0).sum() == 4
    assert summary['coverage'][0]['nonpositive_pb'] == 4
    assert summary['not_a_backtest'] and not summary['original_strategy_complete']


def test_top70_truncation_precedes_low_turnover_rank(monkeypatch):
    tree = originals(monkeypatch)[0]; before = dt.date(2021, 1, 29)
    values, _ = batch.numeric_values(operands(before), [(before, dt.date(2021, 2, 1))], tree)
    selected = values[values.branch.eq('original_PB_prescreen')]
    assert list(selected.instrument) == [f'S{i:03d}' for i in range(69, 34, -1)]
    assert 'S099' not in set(selected.instrument)


def test_invalid_proxy_is_missing_and_never_called_total_cap(monkeypatch):
    tree = originals(monkeypatch)[0]; before = dt.date(2021, 1, 29); frame = operands(before, 4)
    frame.loc[0, 'turnover'] = 0.; frame.loc[1, 'is_trading'] = False
    values, summary = batch.numeric_values(frame, [(before, dt.date(2021, 2, 1))], tree)
    assert summary['coverage'][0]['proxy_missing'] == 2
    assert set(values.branch) == {'original_PB_prescreen', 'float_cap_proxy_prescreen'}
    assert values.loc[values.instrument.isin(['S000', 'S001']), 'float_cap_proxy'].isna().all()
    assert 'market_cap' not in values.columns


def test_future_month_does_not_change_earlier_candidates(monkeypatch):
    tree = originals(monkeypatch)[0]; before = dt.date(2021, 1, 29); day = dt.date(2021, 2, 1)
    frame = operands(before); one, _ = batch.numeric_values(frame, [(before, day)], tree)
    future = operands(dt.date(2021, 2, 26)); future[['pb_mrq', 'volume']] = -999.
    again, _ = batch.numeric_values(pd.concat([frame, future]), [(before, day)], tree)
    pd.testing.assert_frame_equal(one, again, check_exact=True)


def test_missing_month_remains_empty_and_duplicate_rows_block(monkeypatch):
    tree = originals(monkeypatch)[0]; before = dt.date(2021, 1, 29); frame = operands(before)
    values, summary = batch.numeric_values(frame, [(dt.date(2021, 2, 26), dt.date(2021, 3, 1))], tree)
    assert values.empty and summary['coverage'][0]['stored_rows'] == 0
    with pytest.raises(ValueError, match='Duplicate'): batch.numeric_values(pd.concat([frame, frame]), [(before, dt.date(2021, 2, 1))], tree)
    with pytest.raises(ValueError, match='Future'): batch.numeric_values(frame, [(before, before)], tree)


def test_selected_rejects_definition_time_side_effects(monkeypatch):
    for text in ('@unsafe\ndef f(): pass', 'def f(x=open("secret")): pass', 'def f(*, x=open("secret")): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_only_empty_list_default_admitted_without_weakening_side_effect_guard(monkeypatch):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse('def f(result=[]): return result'))
    function = batch.selected(Path('unused'), 0, ['f'], {})['f']
    assert function() == []
    assert function() is function()
    for text in ('def f(result=[open("secret")]): pass', 'def f(result=list()): pass'):
        monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(text))
        with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_sparse_total_cap_does_not_fill_required_dates_or_use_future():
    calendar = pd.DataFrame({'date': ['2021-01-29', '2021-02-01', '2021-02-26', '2021-03-01'], 'is_open': True})
    schedule = batch.monthly_schedule(calendar)
    frame = pd.DataFrame({'date': ['2021-01-28', '2021-02-26', '2026-10-08'], 'value': [10., 12., 99.]})
    audit = batch.cap_coverage(frame, schedule, calendar)
    assert audit['required_monthly_dates'] == 2 and audit['matched_monthly_dates'] == 1
    assert audit['missing_monthly_dates'] == ['2021-01-29']
    assert audit['matched_trading_dates'] == 1 and audit['missing_trading_dates'] == 3
    assert audit['future_rows_excluded'] == 1 and not audit['strict_usable']
