from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import talib

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch30 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows}); return tmp_path


def five(day='2022-01-04'):
    return pd.DataFrame({'bar_end': batch.minute_grid(day), 'instrument': batch.MINUTE_STOCK,
        'open': np.arange(48.) + 100, 'high': np.arange(48.) + 102, 'low': np.arange(48.) + 99,
        'close': np.arange(48.) + 101, 'volume': 100, 'amount': 10000., 'back_factor': 2., 'adjustment_status': 'usable'})


@pytest.mark.parametrize('change', ['none', 'rows', 'content', 'legacy', 'subsecond'])
def test_initial_minute_content_must_match_snapshot(change):
    frame = five(); frame['bar_end'] = frame.bar_end.dt.as_unit('ms')
    original = frame.copy()
    if change in ('legacy', 'subsecond'): original['bar_end'] = original.bar_end.dt.as_unit('s')
    entry = {'rows': len(frame), 'sha': batch.fingerprint(original)}
    if change == 'rows': entry['rows'] -= 1
    if change == 'content': frame.loc[0, 'close'] += .01
    if change == 'subsecond': frame.loc[0, 'bar_end'] += pd.Timedelta(milliseconds=1)
    if change in ('rows', 'content', 'subsecond'):
        with pytest.raises(ValueError, match='Minute partition'): batch.verify_minute_frame(frame, entry, 'bars_5m')
    else:
        proof = batch.verify_minute_frame(frame, entry, 'bars_5m')
        assert proof['accepted_fingerprint'] == entry['sha']
        assert proof['legacy_seconds_columns'] == (['bar_end'] if change == 'legacy' else [])


def test_ten_minute_pairing_keeps_lunch_and_ohlcv():
    actual, info = batch.aggregate_ten(five(), ['2022-01-04'])
    assert len(actual) == 24 and info['accepted_days'] == 1
    assert actual.bar_end.dt.strftime('%H:%M').tolist() == [f'{h:02d}:{m:02d}' for h, m in
        [(9, 40), (9, 50), (10, 0), (10, 10), (10, 20), (10, 30), (10, 40), (10, 50), (11, 0), (11, 10), (11, 20), (11, 30),
         (13, 10), (13, 20), (13, 30), (13, 40), (13, 50), (14, 0), (14, 10), (14, 20), (14, 30), (14, 40), (14, 50), (15, 0)]]
    assert actual.iloc[0][['open', 'high', 'low', 'close', 'volume', 'amount', 'close_adj']].tolist() == [100., 103., 99., 102., 200, 20000., 204.]
    assert actual.iloc[12]['open'] == 124.


@pytest.mark.parametrize('change', ['missing', 'off_grid', 'nan', 'negative', 'inverted', 'unknown', 'factor_conflict'])
def test_invalid_minute_day_is_recorded_and_never_filled(change):
    frame = five()
    if change == 'missing': frame = frame.iloc[1:]
    if change == 'off_grid': frame.loc[0, 'bar_end'] += pd.Timedelta(minutes=1)
    if change == 'nan': frame.loc[0, 'close'] = np.nan
    if change == 'negative': frame.loc[0, 'volume'] = -1
    if change == 'inverted': frame.loc[0, 'high'] = 1
    if change == 'unknown': frame.loc[0, 'adjustment_status'] = 'unavailable'
    if change == 'factor_conflict': frame.loc[0, 'back_factor'] = 3.
    actual, info = batch.aggregate_ten(frame, ['2022-01-04'])
    assert actual.empty and len(info['rejected_days']) == 1


def test_absent_day_breaks_kama_warmup_segment():
    frame = pd.concat([five('2022-01-04'), five('2022-01-06')], ignore_index=True)
    actual, info = batch.aggregate_ten(frame, ['2022-01-04', '2022-01-05', '2022-01-06'])
    assert actual.segment.unique().tolist() == [0, 1]
    assert info['absent_verified_dates'] == ['2022-01-05']


def test_duplicate_or_other_stock_minute_rows_block():
    with pytest.raises(ValueError, match='scope/duplicates'): batch.aggregate_ten(pd.concat([five(), five()]), ['2022-01-04'])
    frame = five(); frame['instrument'] = 'other'
    with pytest.raises(ValueError, match='scope/duplicates'): batch.aggregate_ten(frame, ['2022-01-04'])


def test_interval_complete_does_not_hide_uncovered_frozen_tail(tmp_path):
    frame = five(); _, interval = batch.aggregate_ten(frame, ['2022-01-04'])
    assert interval['absent_verified_dates'] == []
    price = pd.DataFrame({'date': ['2022-01-04', '2022-01-05', '2022-01-06'], 'instrument': batch.MINUTE_STOCK, 'is_trading': True})
    price.to_parquet(tmp_path / 'price-input.parquet', index=False); frame.to_parquet(tmp_path / 'five-minute-input.parquet', index=False)
    tail = batch.minute_tail(tmp_path, {'price': price, 'five-minute': frame})
    assert tail['missing_tail_dates'] == ['2022-01-05', '2022-01-06']
    assert tail['frozen_verified_daily_last'] == '2022-01-06'


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_independent_kama_seed_and_efficiency_agree_with_talib(seed):
    x = np.cumsum(np.random.default_rng(seed).normal(size=360)) + 100.
    np.testing.assert_allclose(batch.reference_kama(x), talib.KAMA(x, timeperiod=312), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_kama_flat_history_and_positive_scale():
    x = np.full(360, 10.)
    np.testing.assert_allclose(batch.reference_kama(x)[312:], 10.)
    x = np.linspace(100., 200., 360)
    np.testing.assert_allclose(batch.reference_kama(x * 2)[312:], 2 * batch.reference_kama(x)[312:])


@pytest.mark.parametrize('values', [[], [1.] * 312, [1.] * 359 + [np.nan], [1.] * 359 + [0.]])
def test_bad_kama_inputs_block(values):
    with pytest.raises(ValueError, match='Invalid KAMA'): batch.reference_kama(values)


def test_adhesion_original_slice_is_one_oldest_of_twelve(sources):
    code, _ = batch.technical_kernel(sources)
    close = pd.DataFrame({'component': np.arange(250.) + 100.}); opening = pd.DataFrame({'component': [349.]})
    ns = {'pd': pd, 'np': np, 'stock_list1': ['component'], 'context': SimpleNamespace(current_dt='2022-01-04'), 'g': SimpleNamespace(previous_buylist={}),
        'get_price': lambda stocks, **kw: {kw['fields']: (close if kw['fields'] == 'close' else opening).tail(kw['count']).copy()}}
    exec(code, ns)
    row = ns['All'].loc['component']
    assert row.highest == row.lowest == close.iloc[-12, 0]
    assert row.highest != close.iloc[-30:, 0].max()


def test_pullback_original_score_loop_agrees_with_independent_reference(sources):
    (_, _), (code, _) = batch.pullback_kernel(sources); close = np.arange(260.) + 100.; means = batch.reference_means(close, (13, 21, 55, 120))
    ns = {'np': np, 'talib': talib, 'df': pd.DataFrame({'code': 'component', 'high': close + 1, 'low': close - 1, 'close': close}),
        'tick_list': ['component'], 'score': {}, 'avg_score': {}}
    ns['data_frame'] = ns['df']
    batch.selected(sources, 2, ('get_ma', 'get_std_percentage', 'get_avg_array'), ns); exec(code, ns)
    _, reference = batch.pullback_reference(close, close, means, 259)
    assert ns['score']['component'] == pytest.approx(reference, abs=1e-12)


def test_original_states_and_risk_boundaries_are_not_repaired(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['short_kama_returns_neutral']['signal'] == cases['missing_kama_returns_neutral']['signal'] == 0
    assert cases['two_day_clock']['selection_calls'] == 1 and cases['two_day_clock']['counter'] == 1
    assert cases['empty_sell_does_not_clear']['sell_list'] == ['stale']
    assert cases['empty_sell_does_not_clear']['previous_buylist'] == {'new': 30}
    assert cases['six_positions_still_submit_and_mark']['orders'] == [['candidate', 100.]]
    assert cases['six_positions_still_submit_and_mark']['marked']
    assert cases['five_of_six_with_two_percent_tolerance']['count'] == 5
    assert cases['ascending_ma_has_zero_array_score']['value'] == 0
    assert not cases['strict_minus7_boundary']['sold']
    assert cases['drawdown_current_price_denominator']['sold']
    assert all(cases[n]['sold'] for n in ('first_day_losing', 'third_day_below5', 'empty_peak_still_age_exit'))
    assert cases['target_existing_stock_can_add']['orders'] == [['already_held', 50.], ['new', 50.]]


def test_source_drift_blocks_before_ast_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][1]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.technical_kernel(sources)


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data'])
def test_probe_binding_refuses_substitution(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0' * 64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted'}]
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises(ValueError): batch.validate_probes(tmp_path)
