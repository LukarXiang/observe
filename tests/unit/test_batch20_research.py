import json

import numpy as np
import pandas as pd
import pytest
import talib

from observe.runs import file_sha
from scripts.review_strategy_batch20 import QUERIES, SNAPSHOT, frames, reference_macd, sample


@pytest.mark.parametrize('periods,scale', [((12, 26, 9), 2), ((5, 15, 7), 1), ((3, 7, 7), 1)])
def test_independent_macd_seeds_and_warmup(periods, scale):
    prices = 20 + .1 * np.arange(200) + 2 * np.sin(np.arange(200) / 3)
    expected = talib.MACDEXT(prices, fastperiod=periods[0], fastmatype=1,
        slowperiod=periods[1], slowmatype=1, signalperiod=periods[2], signalmatype=1)
    actual = reference_macd(prices, *periods, scale)
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(a, b * (scale if i == 2 else 1), rtol=0, atol=1e-12)
        assert np.isnan(a[:sum(periods[1:]) - 2]).all()


@pytest.mark.parametrize('prices,periods', [([1., np.inf], (12, 26, 9)), ([1., np.nan], (12, 26, 9)), ([1.], (26, 12, 9))])
def test_reference_rejects_nonfinite_or_invalid_periods(prices, periods):
    with pytest.raises(ValueError): reference_macd(prices, *periods)


def provider_frame():
    dates = pd.date_range('2025-01-01', periods=201)
    return pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'code': 'sz.000001', 'tradestatus': '1',
        'open': 10., 'close': 11., 'high': 12., 'low': 9.})


@pytest.mark.parametrize('defect', ['short', 'duplicate', 'security', 'nonfinite', 'ohlc'])
def test_bad_provider_samples_block(defect):
    frame = provider_frame()
    if defect == 'short': frame = frame.iloc[:199]
    elif defect == 'duplicate': frame.loc[200, 'date'] = frame.loc[199, 'date']
    elif defect == 'security': frame.loc[200, 'code'] = 'sh.000001'
    elif defect == 'nonfinite': frame.loc[200, 'close'] = np.inf
    elif defect == 'ohlc': frame.loc[200, 'high'] = 10.
    with pytest.raises(ValueError): sample(frame, 'daily')


def test_sample_skips_explicit_pause_and_freezes_200_trades():
    frame = provider_frame(); frame.loc[100, 'tradestatus'] = '0'
    result = sample(frame, 'daily')
    assert len(result) == 200
    assert frame.loc[100, 'date'] not in result.timestamp.tolist()


@pytest.mark.parametrize('flag', ['', None, 'unknown', 1])
def test_unknown_trade_flag_does_not_become_a_skipped_pause(flag):
    frame = provider_frame(); frame['tradestatus'] = frame.tradestatus.astype(object); frame.loc[100, 'tradestatus'] = flag
    with pytest.raises(ValueError, match='Unknown raw trading status'): sample(frame, 'daily')


def test_monthly_incomplete_period_is_not_used():
    frame = pd.concat([provider_frame(), provider_frame().iloc[[0]].assign(date='2026-09-01')], ignore_index=True)
    result = sample(frame, 'monthly')
    assert '2026-09-01' not in result.timestamp.tolist()


@pytest.mark.parametrize('defect', ['snapshot', 'endpoint_set', 'terminal', 'parameters'])
def test_probe_evidence_rejects_scope_or_terminal_mismatch(tmp_path, defect):
    api = tmp_path / 'existing-apis.json'; api.write_text('{}')
    rows = []
    for endpoint in QUERIES:
        terminal = tmp_path / 'probes' / endpoint / 'result.json'; terminal.parent.mkdir(parents=True)
        result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'wire_responses': []}
        terminal.write_text(json.dumps(result))
        rows.append({'endpoint': endpoint, 'result': result})
    if defect == 'terminal':
        (tmp_path / 'probes/daily/result.json').write_text('{}')
    if defect == 'parameters':
        result = rows[0]['result']; result['status'] = 'success'
        result['files'] = [{'parameters': {'code': 'sh.000001'}}]
        (tmp_path / 'probes/daily/result.json').write_text(json.dumps(result))
    probe = {'snapshot': SNAPSHOT if defect != 'snapshot' else 'old', 'published': False,
        'api_sha256': file_sha(api), 'results': rows if defect != 'endpoint_set' else rows[:-1]}
    (tmp_path / 'probe-results.json').write_text(json.dumps(probe))
    with pytest.raises(ValueError): frames(tmp_path)
