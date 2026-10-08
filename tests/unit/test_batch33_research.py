from pathlib import Path
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch33 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original strategy sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def test_shared_functions_do_not_make_distinct_active_pools_identical(sources):
    _, _, hashes = batch.kernels(sources)
    assert set(hashes) == {'get_signal', 'EmotionMonitor', 'ETFtrade', 'before_market_open', 'get_before_after_trade_days'}
    a, b = [batch.params(sources, k) for k in (1, 2)]
    assert len(a.ETF_targets) == 9 and len(b.ETF_targets) == 8
    assert set(a.ETF_targets) - set(b.ETF_targets) == {'399976.XSHE', '399441.XSHE'}
    assert set(b.ETF_targets) - set(a.ETF_targets) == {'000015.XSHG'}
    assert a.ETF_targets['399001.XSHE'] == b.ETF_targets['399001.XSHE'] == '150019.XSHE'


def test_original_score_uses_live_value_and_oldest_completed_close(sources):
    code, _, _ = batch.kernels(sources); x = np.arange(13.) + 10
    actual = batch.score_case(code, x, 30.)
    assert actual == pytest.approx([200., 16., 87.5])
    assert actual[0] != pytest.approx((30./x[-1]-1)*100)


@pytest.mark.parametrize('values', [[], [1.] * 12, [1.] * 14, [1.] * 12 + [np.nan], [1.] * 12 + [0.]])
def test_invalid_score_samples_block(sources, values):
    code, _, _ = batch.kernels(sources)
    with pytest.raises(ValueError, match='Invalid score'): batch.score_case(code, values, 1.)


@pytest.mark.parametrize('values,signal', [(np.ones(13), 1), (np.arange(13., 0., -1), -1), (np.arange(1., 14.), 1)])
def test_original_talib_emotion_and_independent_sma_rules(sources, values, signal):
    actual, rate, calls = batch.emotion_case(sources, values)
    expected, expected_rate, _ = batch.reference_emotion(values)
    assert actual == expected == signal and rate == expected_rate
    assert calls == [{'security': 'component', 'count': 13, 'unit': '1d', 'fields': 'volume'}]


def test_zero_volume_is_original_positive_with_undefined_rate(sources):
    signal, rate, _ = batch.emotion_case(sources, np.zeros(13))
    assert signal == 1 and np.isnan(rate)


@pytest.mark.parametrize('values', [[], [1.] * 12, [1.] * 14, [1.] * 12 + [np.nan], [1.] * 12 + [-1.]])
def test_invalid_emotion_inputs_block(sources, values):
    with pytest.raises(ValueError, match='Invalid emotion'): batch.emotion_case(sources, values)


@pytest.mark.parametrize('increase,deviation,signal', [(.1, 0., 'BUY'), (.0999, 0., 'CLEAR'), (.1, -.0001, 'CLEAR'), (1., 1., 'BUY')])
def test_actual_single_target_thresholds_are_inclusive(sources, increase, deviation, signal):
    result = batch.suffix_case(sources, [('best', 'idx_best', increase, deviation), ('worst', 'idx_worst', -.5, -1.)])
    assert result['signal'] == signal and result['target_market'] == 'idx_worst'
    assert result['buy'] == (['best'] if signal == 'BUY' else ['stale'])


def test_negative_emotion_precedes_qualifying_fund_and_preserves_stale_targets(sources):
    result = batch.suffix_case(sources, [('best', 'idx', 10., 10.)], emotion=-1)
    assert result['signal'] == 'CLEAR' and result['buy'] == ['stale']
    result = batch.suffix_case(sources, [])
    assert result['signal'] == 'CLEAR' and result['buy'] == ['stale']


def test_original_rebalance5000_is_strict_and_failed_sell_blocks_safe_fund(sources):
    assert batch.trade_case(sources, 'BUY', ['a'], {'a': 5000.}) == []
    assert batch.trade_case(sources, 'BUY', ['a'], {'a': 4999.}) == [['value', 'a', 10000.]]
    assert batch.trade_case(sources, 'CLEAR', [], {'old': 10000.}) == [['target', 'old', 0]]
    assert batch.trade_case(sources, 'CLEAR', [], {'old': 10000.}, accept_sell=True) == [['target', 'old', 0], ['value', '511880.XSHG', 1000.]]
    with pytest.raises(ZeroDivisionError): batch.trade_case(sources, 'BUY', [], {})


def test_source_diagnostics_preserve_wizard_clock_union_retry_and_compatibility(sources):
    diagnostic = batch.diagnostics(sources)
    assert json.loads(json.dumps(diagnostic)) == diagnostic
    rows = {r['case']: r for r in diagnostic['cases']}
    assert rows['bank_300_callback_clock']['refresh_indices'] == [0, 300, 600]
    assert len(rows['bank_300_callback_clock']['industry_list']) == 28
    assert rows['bank_300_callback_clock']['allocation'] == ['by_market_cap_percent', 100]
    assert rows['bank_industry_union']['stocks'] == ['bank', 'nonbank']
    assert rows['bank_rejected_retry']['orders'] == [['held', 0]]
    assert rows['bank_rejected_retry']['pending'] == ['held']
    assert rows['bank_accepted_retry']['pending'] == []
    assert 'financial_data_filter_qujian' in rows['bank_missing_finance_wizard']['error']
    assert rows['bank_sum_compatibility']['result'] == ['a']
    assert rows['bank_sum_compatibility']['warnings'][0]['category'] == 'Pandas4Warning'
    assert 'append' in rows['original_append_error']['error']
    assert 'original_dated_negative_index_error' in rows


def test_dated_series_negative_index_failure_remains_visible(sources):
    with pytest.raises(KeyError): batch.emotion_case(sources, np.ones(13), dated=True)


def test_original_listing_gate_counts13_sessions_including_yesterday(sources):
    days = list(pd.bdate_range('2020-01-01', periods=20).date); yesterday = days[-1]
    g = batch.params(sources); index, fund = next(iter(g.ETF_targets.items())); boundary = days[-13]
    funds = pd.DataFrame({'start_date': [boundary]}, index=[fund])
    indexes = pd.DataFrame({'start_date': [boundary]}, index=[index]); requested = []
    def securities(types, date):
        requested.append([types, date]); return funds if types == 'fund' else indexes
    ns = {'g': g, 'pd': pd, 'datetime': datetime, 'get_all_trade_days': lambda: days, 'get_all_securities': securities}
    batch.selected(sources, 1, ('get_before_after_trade_days', 'before_market_open'), ns)
    ctx = batch.SimpleNamespace(previous_date=yesterday)
    assert ns['get_before_after_trade_days'](yesterday, 13) == boundary
    ns['before_market_open'](ctx); assert g.ETFList == {index: fund}
    assert requested == [['fund', yesterday], ['index', yesterday]]
    funds.loc[fund, 'start_date'] = days[-12]
    ns['before_market_open'](ctx); assert g.ETFList == {}
    funds.loc[fund, 'start_date'] = boundary; indexes.drop(index, inplace=True)
    ns['before_market_open'](ctx); assert g.ETFList == {}


def test_long_operator_rejects_windows_crossing_a_missing_verified_session(sources, monkeypatch):
    days = list(pd.bdate_range('2020-01-01', periods=20).date)
    frame = pd.DataFrame({'date': days, 'close': np.arange(20.)+100., 'volume': np.arange(20.)+1000.}).drop(index=8).reset_index(drop=True)
    monkeypatch.setattr(batch, 'validate_sources', lambda directory: None)
    monkeypatch.setattr(batch, 'inputs', lambda directory: (frame, days))
    for name in ('index-input.parquet', 'calendar-input.parquet'): (sources / name).write_bytes(b'diagnostic-input')
    result = batch.compute(sources)
    assert result['windows'] == 0 and result['calendar_gap_windows_rejected'] == 6


def test_original_source_drift_blocks_extraction(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][1]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.kernels(sources)


def test_input_byte_drift_blocks_before_parquet_read(tmp_path):
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'component_instrument': '000300.SH', 'profiles': {n: {'file': str(tmp_path / n), 'sha256': '0' * 64}
            for n in ('index-input.parquet', 'calendar-input.parquet')}})
    (tmp_path / 'index-input.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input binding changed'): batch.inputs(tmp_path)


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_substitution_and_failed_data_rejected(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0' * 64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted', 'sha256': '0' * 64}]
        if change == 'status': row['status'] = 'accepted'
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises((ValueError, FileNotFoundError)): batch.validate_probes(tmp_path)
