import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch70 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.parsed_source(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_active_definitions_keep_original_line_numbers(monkeypatch):
    trees = originals(monkeypatch)
    assert [len(batch.definitions(t)) for t in trees] == [13, 17]
    assert [t.body[0].lineno for t in trees] == [12, 14]
    assert len(batch.POOL) == 26 and len(set(batch.POOL)) == 26
    assert 'PB' in batch.POOL and 'A' not in batch.POOL
    assert len(batch.QUERIES) == 25
    assert 'RB' not in batch.QUERIES and 'CU' not in batch.QUERIES


def test_diagnostics_cover_original_rank_and_order_state(monkeypatch):
    originals(monkeypatch)
    result = batch.diagnostics(Path('unused'))
    cases = {row['case']: row for row in result['cases']}
    assert len(cases) == 22
    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert result['not_a_backtest'] and result['synthetic_fixture']
    assert not result['platform_equivalent'] and result['real_historical_trade_windows'] == 0
    assert cases['cross_no_momentum_fixed_pool_rank_IndexError']['error'] == 'IndexError'
    assert cases['cross_six_candidates_still_requires_seven']['error'] == 'IndexError'
    overlap = cases['cross_seven_candidates_overlap_all_buy_priority']
    assert len(overlap['open_intents']) == 7 and overlap['global_reentry_true_did_not_block']
    assert cases['cross_short_roll_reads_new_key']['error'] == 'KeyError'
    assert cases['cross_short_roll_reads_new_key']['orders'] == []
    assert cases['cross_1ATR_stop_is_not_percent_and_counter_stays_zero']['orders'][0]['args'][1] == 0
    roll = cases['turtle_roll_equality_typo_followed_by_real_reset_before_expiry']
    assert roll['position'] == 0 and roll['last_price_retained'] == roll['mark_retained'] == 100.
    assert cases['turtle_None_add_intent_still_increments_counter']['add_time'] == 2
    assert cases['turtle_None_ATR_stop_still_resets_internal_position']['position'] == 0
    assert cases['turtle_nonempty_short_history_not_guarded']['error'] == 'IndexError'
    assert not cases['turtle_position_price95percent_inclusive']['price_field_semantics_proved']


@pytest.mark.parametrize('code', [
    '@unsafe\ndef f():\n    pass',
    'def f(value=open("secret")):\n    pass',
    'def f(*, value=open("secret")):\n    pass',
])
def test_selected_rejects_eager_execution(monkeypatch, code):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(code))
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected(Path('unused'), 0, ['f'], {})


def fixture():
    close = np.arange(40, dtype=float) + 100.
    return pd.DataFrame({'date': pd.date_range('2026-07-01', periods=40),
        'open': close, 'high': close+2, 'low': close-1, 'close': close})


def test_original_numeric_functions_preserve_lags_channel_and_risk_unit(monkeypatch):
    originals(monkeypatch)
    values, summary = batch.numeric_windows(fixture(), 'SR', Path('unused'))
    cross = values[values.model.eq('cross20')]
    turtle = values[values.model.eq('turtle21')]
    assert len(cross) == 21 and len(turtle) == 20
    assert cross.ATR.eq(3.).all() and turtle.ATR.eq(3.).all()
    assert cross.momentum.iloc[0] == 119./105.-1
    assert turtle.channel_upper.iloc[0] == 120.
    assert turtle['break'].eq(0).all()
    assert turtle.risk_unit.iloc[0] == pytest.approx(1000000.*.01/3./10)
    assert summary['independent_numeric_comparisons'] == 82
    contract, _ = batch.numeric_windows(fixture(), 'SR2001', Path('unused'))
    assert set(contract.model) == {'turtle21'}


def test_invalid_OHLC_blocks_full_windows_without_interpolation(monkeypatch):
    originals(monkeypatch)
    frame = fixture()
    frame.loc[0, 'close'] = np.nan
    values, summary = batch.numeric_windows(frame, 'SR', Path('unused'))
    assert summary['invalid_OHLC_rows'] == 1 and summary['invalid_windows_blocked'] == 2
    assert len(values) == 39


def test_cutoff_excludes_future_rows_and_duplicate_dates_fail(monkeypatch):
    originals(monkeypatch)
    frame = fixture()
    expected, _ = batch.numeric_windows(frame, 'CU', Path('unused'))
    extra = frame.iloc[[0]].copy()
    extra['date'] = pd.Timestamp('2026-10-08')
    extra['close'] = 999999.
    actual, summary = batch.numeric_windows(pd.concat([frame, extra], ignore_index=True), 'CU', Path('unused'))
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    assert summary['rows_after_cutoff_excluded'] == 1
    with pytest.raises(ValueError, match='uniqueness'):
        batch.numeric_windows(pd.concat([frame, frame.iloc[[-1]]]), 'CU', Path('unused'))


def test_cross_sections_require_all_26_and_keep_stable_ties():
    values = pd.DataFrame([{'case': ins, 'model': 'cross20', 'date': '2020-01-02', 'momentum': 0.}
        for ins in batch.POOL])
    ranks, skipped = batch.cross_sections(values)
    assert skipped == 0 and len(ranks) == 14
    assert list(ranks[ranks.side.eq('buy')]['case']) == list(batch.POOL[:7])
    assert list(ranks[ranks.side.eq('sell')]['case']) == list(batch.POOL[-1:-8:-1])
    missing, skipped = batch.cross_sections(values.iloc[:-1])
    assert missing.empty and skipped == 1


def test_worker_failure_is_frozen_and_not_published(tmp_path, monkeypatch):
    ak = pytest.importorskip('akshare')
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    def broken(**kwargs):
        raise ValueError('Explicit supplier failure')
    monkeypatch.setattr(ak, 'futures_zh_daily_sina', broken)
    result = batch.worker(tmp_path, 'SR2001')
    assert result['status'] == 'failed' and 'Explicit supplier failure' in result['error']
    assert result['attempts'] == result['responses'] == result['files'] == []
    assert not result['published'] and not result['strict_usable']
    assert result == batch.read(tmp_path / 'probes/SR2001/result.json')
    with pytest.raises(FileExistsError):
        batch.worker(tmp_path, 'SR2001')


def test_wire_is_saved_before_parsing_and_replayed_offline(tmp_path, monkeypatch):
    pytest.importorskip('akshare')
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    calls = []
    records = [{'d': '2020-01-02', 'o': '100', 'h': '103', 'l': '99',
        'c': '102', 'v': '10', 'p': '20', 's': '101'}]
    wire = ('var_sample=('+json.dumps(records)+');').encode()
    def response(session, method, url, **kwargs):
        assert session.trust_env is False
        assert kwargs['timeout'] == (8, 10) and kwargs['allow_redirects'] is False
        calls.append((method, url, kwargs['params']))
        result = batch.requests.Response()
        result.status_code = 200
        result._content = wire
        result.encoding = 'utf-8'
        result.url = url
        return result
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', response)
    row = batch.worker(tmp_path, 'SR2001')
    assert row['status'] == 'success' and len(calls) == len(row['attempts']) == 1
    assert Path(row['responses'][0]['file']).read_bytes() == wire
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', lambda *a, **kw: pytest.fail('Unexpected network'))
    pd.testing.assert_frame_equal(pd.read_parquet(row['files'][0]['file']), batch.replay_frame(row), check_exact=True)


def test_worker_repeated_request_is_rejected_with_first_wire_preserved(tmp_path, monkeypatch):
    ak = pytest.importorskip('akshare')
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    def response(*args, **kwargs):
        result = batch.requests.Response()
        result.status_code = 200
        result._content = b'first response'
        result.url = batch.previous.SINA_URL
        return result
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', response)
    def twice(**kwargs):
        for _ in range(2):
            batch.requests.get(batch.previous.SINA_URL, params={'symbol': 'SR2001', 'type': '2021_04_12'})
    monkeypatch.setattr(ak, 'futures_zh_daily_sina', twice)
    row = batch.worker(tmp_path, 'SR2001')
    assert row['status'] == 'failed' and 'repeated request' in row['error']
    assert len(row['attempts']) == len(row['responses']) == 1 and not row['files']
    assert Path(row['responses'][0]['file']).read_bytes() == b'first response'


def test_api_evidence_survives_JSON_freeze():
    pytest.importorskip('akshare')
    evidence = batch.api_evidence()
    assert evidence == json.loads(json.dumps(evidence, allow_nan=False))
