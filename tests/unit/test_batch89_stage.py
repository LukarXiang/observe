"""Verify the source-stage error and distinguish candidate coverage from trading."""
import json
from pathlib import Path

import pandas as pd
import pytest

from observe.runs import file_sha
from scripts import review_strategy_batch89 as batch


@pytest.fixture
def frozen(tmp_path):
    source = Path('repo/量化策略源代码')/batch.SOURCES[0]
    if not source.exists():
        pytest.skip('Restore original source payload for literal checks')
    dest = tmp_path/'original.source'; dest.write_bytes(source.read_bytes())
    (tmp_path/'sources-frozen.json').write_text(json.dumps([{'source_copy': str(dest)}]), encoding='utf-8')
    return tmp_path


def test_native_original_failure_and_monday_return_difference(frozen):
    result = batch.compatibility(frozen)
    assert result['original_failure']['exception'] == 'TypeError'
    assert result['history_calls_before_failure'] == [
        {'lag': 120, 'freq': '1d', 'field': 'close', 'stock': '510300.XSHG', 'df': True}]
    assert result['synthetic_calendar_rows'] == 166
    assert result['changed_return_rows'] == 23
    assert result['first_example']['date'] == '2021-01-11'
    assert result['first_example']['current_default_return'] == 0
    assert result['first_example']['explicit_forward_filled_return'] < 0
    assert result['decision'] == 'pending_D-allweather-fill'
    assert result['synthetic_only'] and result['not_a_backtest']


def test_native_calc_rejects_changed_source(frozen):
    (frozen/'original.source').write_text('def fun_calc_trade_ratio(context): return {}', encoding='utf-8')
    with pytest.raises(ValueError, match='Unreviewed source'):
        batch.original_calc(frozen, lambda *a, **kw: None)


def test_candidate_intersection_cutoff_and_jump_are_not_trades(tmp_path):
    inputs = {}
    for instrument, dates, closes in (
        ('A', ['2020-01-02', '2020-01-03', '2026-09-30'], [10., 2., 100.]),
        ('B', ['2020-01-03', '2020-01-06'], [20., 21.])):
        path = tmp_path/(instrument+'.parquet')
        pd.DataFrame({'date': pd.to_datetime(dates), 'close': closes}).to_parquet(path, index=False)
        inputs[instrument] = {'payload': {'file': str(path), 'sha256': file_sha(path)}}
    result = batch.candidate_coverage(inputs)
    assert result['common_observed_dates'] == 1
    assert result['common_range'] == ['2020-01-03', '2020-01-03']
    assert result['inputs']['A']['rows_through_cutoff'] == 2
    assert result['inputs']['A']['raw_jumps_over_50pct'][0]['raw_relative_change'] == pytest.approx(-.8)
    assert all(not r['trade_usable'] for r in result['inputs'].values())


@pytest.mark.parametrize('dates,closes', [(['2020-01-02', '2020-01-02'], [10., 11.]),
    (['2020-01-03', '2020-01-02'], [10., 11.]), (['2020-01-02'], [0.])])
def test_candidate_invalid_dates_or_prices_are_blocked(tmp_path, dates, closes):
    path = tmp_path/'bad.parquet'
    pd.DataFrame({'date': pd.to_datetime(dates), 'close': closes}).to_parquet(path, index=False)
    with pytest.raises(ValueError):
        batch.candidate_coverage({'A': {'payload': {'file': str(path), 'sha256': file_sha(path)}}})
