from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch21 as batch
from scripts.review_strategy_batch21 import SNAPSHOT, etf_sample, reference_rsi, selected, upstream_receipt


@pytest.mark.parametrize('values', [np.arange(100, dtype=float), -np.arange(100, dtype=float), np.ones(100),
    np.sin(np.arange(100) / 3), np.log((100 + np.arange(100)) / (103 + np.arange(100)))])
def test_independent_wilder_rsi_seed_and_flat_boundary(values):
    actual = reference_rsi(values)
    np.testing.assert_allclose(actual, talib.RSI(values, 6), atol=1e-12, rtol=0, equal_nan=True)
    assert np.isnan(actual[:6]).all()


@pytest.mark.parametrize('values', [[1., np.nan], [1., np.inf]])
def test_invalid_rsi_inputs_block(values):
    with pytest.raises(ValueError): reference_rsi(values)


def sample_prices():
    return pd.DataFrame({'date': pd.date_range(end='2026-09-29', periods=100).strftime('%Y-%m-%d'),
        'open': 10., 'close': 11., 'high': 12., 'low': 9.})


@pytest.mark.parametrize('defect', ['short', 'duplicate', 'nonfinite', 'ohlc', 'stale'])
def test_invalid_etf_price_sample_blocks(defect):
    frame = sample_prices()
    if defect == 'short': frame = frame.iloc[1:]
    elif defect == 'duplicate': frame.loc[98, 'date'] = frame.loc[97, 'date']
    elif defect == 'nonfinite': frame.loc[99, 'close'] = np.nan
    elif defect == 'ohlc': frame.loc[99, 'high'] = 10.
    elif defect == 'stale': frame.loc[99, 'date'] = '2026-09-28'
    with pytest.raises(ValueError): etf_sample(frame)


def test_function_slice_excludes_python2_print_and_top_level_side_effect(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text("print 'old syntax'\ndef f():\n    if True:\n        return 3\nraise RuntimeError('must not run')\n")
    review = tmp_path / 'source-reviews/review.json'
    write_json(review, {'sources': [{'source_copy': str(source), 'source_sha256': file_sha(source)}]})
    namespace = {}; fingerprint = selected(tmp_path, 0, ('f',), namespace)
    assert len(fingerprint) == 64 and namespace['f']() == 3


def test_selected_function_rejects_changed_source(tmp_path):
    source = tmp_path / 'source.txt'; source.write_text('def f():\n    return 1\n')
    write_json(tmp_path / 'source-reviews/review.json', {'sources': [{'source_copy': str(source), 'source_sha256': file_sha(source)}]})
    source.write_text('def f():\n    return 2\n')
    with pytest.raises(ValueError, match='Selected source changed'): selected(Path(tmp_path), 0, ('f',), {})


def test_self_consistent_replaced_upstream_cannot_override_accepted_receipt(tmp_path):
    data = tmp_path / 'prices.parquet'; pd.DataFrame({'close': [1.]}).to_parquet(data)
    terminal = tmp_path / 'result.json'; receipt = tmp_path / 'receipt.json'
    write_json(terminal, {'file': str(data), 'sha256': file_sha(data)})
    write_json(receipt, {'status': 'ok', 'reviews': {'snapshot': SNAPSHOT},
        'checks': {'evidence_sha256': {'probes/etf510300/result.json': file_sha(terminal)}}})
    assert upstream_receipt(receipt, terminal)['terminal_sha256'] == file_sha(terminal)
    pd.DataFrame({'close': [2.]}).to_parquet(data)
    write_json(terminal, {'file': str(data), 'sha256': file_sha(data)})
    with pytest.raises(ValueError, match='accepted receipt'): upstream_receipt(receipt, terminal)


def test_undefined_body_ratio_blocks_without_an_invented_signal(tmp_path, monkeypatch):
    prices = tmp_path / 'prices.parquet'
    pd.DataFrame({k: np.full(100, 10.) for k in ('open', 'high', 'low', 'close')}).to_parquet(prices)
    write_json(tmp_path / 'input-analysis.json', {'input_file': str(prices), 'input_sha256': file_sha(prices)})
    monkeypatch.setattr(batch, 'bound_upstream', lambda *args: {})
    monkeypatch.setattr(batch, 'selected', lambda *args: 'reviewed')
    with pytest.raises(ValueError, match='undefined on flat last bar'): batch.compute(tmp_path)
