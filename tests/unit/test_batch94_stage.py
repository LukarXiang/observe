"""Independent DFT/OLS and literal original contracts for Batch94 numerical stage."""
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch94 as batch


@pytest.fixture
def source_directory(tmp_path):
    (tmp_path/'sources').mkdir()
    for number in range(3):
        shutil.copyfile(batch.READINESS/f'{number}.source', tmp_path/'sources'/f'{number}.source')
    return tmp_path


@pytest.mark.parametrize('frequency,kept', [(0, True), (60, True), (61, False), (189, False), (190, True), (249, True)])
def test_fft_exact_spectral_boundary(source_directory, frequency, kept):
    signal = np.cos(2*np.pi*frequency*np.arange(250)/250)
    literal = batch.literal(source_directory, 0, {'np': np})['fftFilter'](60, 250, signal)
    expected = signal if kept else np.zeros(250)
    np.testing.assert_allclose(literal.real, expected, rtol=0, atol=2e-13)
    assert np.max(np.abs(literal.imag)) < 2e-13


def test_dft_projection_without_fft_or_polyfit(source_directory, monkeypatch):
    values = np.log(1.1+.01*np.arange(250)+.05*np.sin(np.arange(250)))
    literal = batch.literal(source_directory, 0, {'np': np})['fftFilter'](60, 250, values)
    expected = float(np.polyfit(np.arange(20), literal[-20:], 1)[0].real)
    def forbidden(*args, **kwargs): raise AssertionError('Independent DFT must not use FFT/polyfit')
    monkeypatch.setattr(np.fft, 'fft', forbidden); monkeypatch.setattr(np.fft, 'ifft', forbidden)
    monkeypatch.setattr(np, 'polyfit', forbidden)
    kernel = batch.fft_slope_kernel()
    actual = math.fsum(float(v)*w for v, w in zip(values, kernel, strict=True))
    assert actual == pytest.approx(expected, abs=2e-13)
    assert math.fsum(kernel) == pytest.approx(0, abs=2e-14)


def test_scalar_ols_has_known_intercept_slope_r2():
    x = [1., 2., 3.]; y = [3., 5., 8.]
    intercept, slope, r2 = batch.scalar_ols(x, y)
    assert intercept == pytest.approx(1/3)
    assert slope == pytest.approx(2.5)
    assert r2 == pytest.approx(75/76)


def test_scalar_population_z_does_not_use_sample_variance():
    assert batch.scalar_z([1., 2., 3.]) == pytest.approx(math.sqrt(1.5))


@pytest.mark.parametrize('number', [0, 1, 2])
def test_all_literal_sources_use_same_ols_and_population_z(source_directory, number):
    ns = batch.literal(source_directory, number, {'np': np})
    x = np.array([1., 2., 3.]); y = np.array([3., 5., 8.])
    result = ns['get_ols'](x, y)
    assert result == pytest.approx((1/3, 2.5, 75/76), abs=1e-13)
    assert ns['get_zscore']([1., 2., 3.]) == pytest.approx(math.sqrt(1.5))


@pytest.mark.parametrize('score,expected', [(-.7, 'KEEP'), (.7, 'KEEP'), (-.700001, 'SELL'), (.700001, 'BUY')])
def test_original_strict_threshold(score, expected):
    assert batch.timing(score) == expected


def test_shape_and_smooth_spies_do_not_claim_kalman_numerics(source_directory):
    result = batch.diagnostics(source_directory)
    kalman = next(r for r in result['cases'] if r['case'] == 'Kalman_constructor_and_smooth_call_spy')
    assert kalman['score_shape'] == [1]
    assert not kalman['Kalman_numerics_reproduced']
    assert not result['platform_top_level_executed']
    assert result['case_count'] == 9
    assert result['new_trading_backtests'] == 0


def test_source_corruption_rejected_before_definitions_run(source_directory):
    with (source_directory/'sources/0.source').open('ab') as stream: stream.write(b'\nraise RuntimeError()')
    with pytest.raises(ValueError, match='Source identity'):
        batch.literal(source_directory, 0, {})


def test_alternate_root_rejected_before_any_output(tmp_path):
    directory = tmp_path/'output'
    with pytest.raises(ValueError, match='actual data root'):
        batch.prepare(tmp_path, directory)
    assert not directory.exists()


def test_native_string_parsing_keeps_bytes_and_cutoff(tmp_path):
    path = tmp_path/'input.parquet'
    original = pd.DataFrame({'date': ['2026-09-29', '2026-09-30'], **{n: ['1.2', '1.3'] for n in ('open', 'high', 'low', 'close')}})
    original.to_parquet(path)
    sha = batch.file_sha(path)
    parsed, raw = batch.native_frame({'file': str(path), 'rows': 2, 'sha256': sha})
    assert parsed.date.tolist() == ['2026-09-29']
    assert parsed.close.tolist() == [1.2]
    pd.testing.assert_frame_equal(raw, original)
    assert batch.file_sha(path) == sha


def test_duplicate_dates_rejected(tmp_path):
    path = tmp_path/'input.parquet'
    pd.DataFrame({'date': ['2026-09-29', '2026-09-29']}).to_parquet(path)
    with pytest.raises(ValueError, match='Native dates ambiguous'):
        batch.native_frame({'file': str(path), 'rows': 2, 'sha256': batch.file_sha(path)})


def test_unverified_native_byte_change_rejected(tmp_path):
    path = tmp_path/'input.parquet'; path.write_bytes(b'not a parquet')
    with pytest.raises(ValueError, match='Candidate byte SHA'):
        batch.native_frame({'file': str(path), 'rows': 0, 'sha256': 'wrong'})


def test_rsrs_seed_gap_and_future_rows_not_visible(source_directory, monkeypatch):
    index = np.arange(621.)
    frame = pd.DataFrame({'date': pd.date_range('2000-01-01', periods=621).strftime('%Y-%m-%d'),
        'low': 100+index*.2+np.sin(index*.4), 'high': 104+index*.21+np.cos(index*.31)})
    path = source_directory/'index-input.parquet'; frame.to_parquet(path, index=False)
    def bind():
        return {'frozen_sha256': batch.file_sha(path)}
    monkeypatch.setattr(batch, 'read', lambda p: bind())
    actual, summary = batch.rsrs_components(source_directory)
    assert summary['discarded_window_start'] == 599
    assert summary['first_appended_window_start'] == 600
    assert summary['seed_slopes_after_discard'] == 599
    # First daily population is windows0..598 plus600, preserving original skipped599.
    slopes = [batch.scalar_ols(frame.low.iloc[i:i+18], frame.high.iloc[i:i+18])[1] for i in range(599)]
    slopes.append(batch.scalar_ols(frame.low.iloc[600:618], frame.high.iloc[600:618])[1])
    assert actual.zscore.iloc[0] == pytest.approx(batch.scalar_z(slopes), abs=1e-10)
    frame.loc[619:, 'high'] *= 3
    frame.to_parquet(path, index=False)
    changed, _ = batch.rsrs_components(source_directory)
    pd.testing.assert_frame_equal(actual.iloc[:2], changed.iloc[:2], check_exact=True)
    assert not actual.iloc[2:].equals(changed.iloc[2:])


def test_dft_bad_window_rejected():
    with pytest.raises(ValueError, match='parameters invalid'):
        batch.fft_slope_kernel(size=20, momentum=21)


def test_readiness_paths_are_only_sources():
    assert len(list(Path(batch.READINESS).glob('*.source'))) == 3
