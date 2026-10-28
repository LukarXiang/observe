"""Independent interval mapping, gaps, operand identity and native input protection."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from scripts import reference_strategy_batch103 as reference


def test_exact_old3_recent3_and21_row_base_offsets_are_independently_verified():
    days = pd.bdate_range('2024-01-02', periods=31).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({'date': days, 'close': np.arange(100., 131.)})
    real, momentum, profiles = reference.price_references({'510050.XSHG': frame}, days)
    assert len(real) == 18 and len(momentum) == 1
    row = momentum.iloc[0]
    assert row.old_mean == 108. and row.base_price == 109.
    assert math.isclose(row.score, ((128.+129.+130.)/(107.+108.+109.)-1)*100, abs_tol=1e-12)
    first = real.iloc[0]
    assert first.cp_increase == (113./100.-1)*100 and first.MA13 == 106.
    assert first.asof_native_close == days[13] and first.window_first == days[0]
    assert row.history_rows == 30 and row.current_is_daily_close_proxy
    assert profiles['510050.XSHG']['year_windows'] == {'real': {'2024': 18}, 'momentum': {'2024': 1}}


def test_missing_day_preserved_without_fill_or_silent_shortening():
    days = pd.bdate_range('2024-01-02', periods=32).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({'date': days, 'close': np.arange(100., 132.)}).drop(index=15).reset_index(drop=True)
    original = frame.copy(deep=True)
    real, momentum, profiles = reference.price_references({'510050.XSHG': frame}, days)
    pd.testing.assert_frame_equal(frame, original)
    assert profiles['510050.XSHG']['missing_open_dates'] == [days[15]]
    assert int(momentum.iloc[0].missing_open_count) == 1
    assert real.missing_open_count.gt(0).sum() > 0
    assert momentum.iloc[0].asof_native_close == days[-1]


def registry(tmp_path, *, function='fund_etf_hist_sina', exchange='sh', duplicate=False):
    raw = tmp_path/'raw.parquet'; wire = tmp_path/'response.bin'; wire.write_bytes(b'explicit-test-wire')
    pd.DataFrame({'date': ['2024-01-02', '2024-01-03', '2026-10-01'], 'close': [1., 2., 3.]}).to_parquet(raw)
    row = {'status': 'success', 'case': 'fund-510050', 'rows': 3, 'file': str(raw), 'sha256': file_sha(raw),
        'query': {'function': function, 'parameters': {'symbol': exchange+'510050'}},
        'wire_responses': [{'file': str(wire), 'sha256': file_sha(wire), 'status_code': 200}]}
    reg = tmp_path/'results.json'; reg.write_text(json.dumps({'results': [row, row] if duplicate else [row]}))
    return reg, raw


def test_registered_native_bytes_kept_and_future_rows_explicitly_excluded(tmp_path, monkeypatch):
    reg, raw = registry(tmp_path); monkeypatch.setattr(reference, 'REGISTRIES', (str(reg),))
    frames, bindings, hashes, rows = reference.candidates()
    assert frames['510050.XSHG'].close.tolist() == [1., 2.]
    assert bindings['510050.XSHG']['future_excluded'] == 1
    assert bindings['510050.XSHG']['native_rows'] == 3
    assert hashes[str(raw)] == file_sha(raw) and len(rows) == 1


@pytest.mark.parametrize('overrides', [{'function': 'stock_zh_index_daily'}, {'exchange': 'sz'}, {'duplicate': True}])
def test_candidate_wrong_api_exchange_or_duplicate_rejected(tmp_path, monkeypatch, overrides):
    reg, _ = registry(tmp_path, **overrides); monkeypatch.setattr(reference, 'REGISTRIES', (str(reg),))
    with pytest.raises(ValueError): reference.candidates()


def test_changed_native_bytes_rejected_before_computation(tmp_path, monkeypatch):
    reg, raw = registry(tmp_path); monkeypatch.setattr(reference, 'REGISTRIES', (str(reg),))
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError): reference.candidates()
