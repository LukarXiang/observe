"""Check independent Batch88 finite-price arithmetic and original replay boundaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import verify_strategy_batch88 as verify


@pytest.fixture
def original(tmp_path, monkeypatch):
    path = Path('repo/量化策略源代码')/verify.stage.SOURCES[0]
    if not path.exists():
        pytest.skip('Restore original source payload for literal checks')
    dest = tmp_path/'original.source'; dest.write_bytes(path.read_bytes())
    (tmp_path/'sources-frozen.json').write_text(json.dumps([{'source_copy': str(dest)}]), encoding='utf-8')
    monkeypatch.setattr(verify.stage, 'DIRECTORY', tmp_path)
    return verify.original_namespace()


def array(high, low):
    out = np.empty(len(high), dtype=[('date', 'datetime64[ns]'), ('open', 'f8'),
                                    ('high', 'f8'), ('low', 'f8'), ('close', 'f8')])
    out['date'] = pd.bdate_range('2020-01-01', periods=len(high)).to_numpy(dtype='datetime64[ns]')
    out['high'] = high; out['low'] = low; out['open'] = out['close'] = (out['high']+out['low'])/2
    return out


@pytest.mark.parametrize('high,low', [([10], [9]), ([10, 11], [9, 10]), ([10, 11, 12], [9, 10, 11]),
    ([10]*8, [9]*8), ([10, 11, 12, 13, 12, 11, 10, 9, 10, 11], [9, 10, 11, 12, 11, 10, 9, 8, 9, 10]),
    ([10, 12, 11, 13, 12, 14, 13, 12, 15], [9, 10, 10.5, 11, 11.5, 12, 12.5, 11, 13]),
    ([10, 10.000001, 11, 10.5, 12, 13, 12, 11], [9, 9.000001, 10, 10.2, 11, 12, 11, 10])])
def test_default_independent_inclusion_and_fractals(original, high, low):
    data = array(high, low); obj = original['KBarChan'](data.copy())
    obj.standardize(); obj.markTopBot()
    expected = verify.standard_rows(data); types = verify.fractals(expected)
    assert len(expected) == len(obj.kDataFrame_standardized)
    for row, expected_row, tb in zip(obj.kDataFrame_standardized, expected, types, strict=True):
        assert all(row[name] == value for name, value in expected_row.items())
        assert row['tb'] == tb


def test_independent_serializer_keeps_bits_and_date():
    a = array([1., np.nextafter(1., 2.)], [.5, .6])
    assert verify.serial(a) == verify.stage.payload(a)
    assert verify.serial(a)['rows'][0][2] != verify.serial(a)['rows'][1][2]


@pytest.mark.parametrize('a,b', [(100.0005, 100.), (100.002, 100.), (.01000000001, .01),
                               (0., 1e-9), (1e-9, 0.), (1e8, 1e8+1000.01)])
def test_independent_finite_comparisons_match_native_defaults(a, b):
    assert verify.close(a, b) == bool(np.isclose(a, b))
    assert verify.more(a, b) == bool(a > b and not np.isclose(a, b))
    assert verify.less(a, b) == bool(a < b and not np.isclose(a, b))


def test_independent_window_scalar_factor_anchor():
    bars = pd.DataFrame({'date': [pd.Timestamp('2020-01-01').date(), pd.Timestamp('2020-01-02').date()],
        'independent_factor': [2., 4.], **{k: [10., 6.] for k in ('open', 'high', 'low', 'close')}})
    np.testing.assert_array_equal(verify.independent_window(bars)['close'], [5., 6.])


def test_independent_original_import_prefix_stops_before_chart(original):
    assert 'KBarChan' in original and 'draw_chan' not in original
    assert 'get_bars' not in original and 'stock_df' not in original


def test_fractal_tail_is_assumed_and_trend_storage_truncates(original):
    data = array([10, 11, 12, 13, 12, 11, 10, 9, 10], [9, 10, 11, 12, 11, 10, 9, 8, 9])
    obj = original['KBarChan'](data.copy())
    assert obj.kDataFrame_standardized.dtype['trend_type'].kind == 'i'
    obj.kDataFrame_standardized['trend_type'][0] = .5
    assert obj.kDataFrame_standardized['trend_type'][0] == 0
    obj.standardize(); obj.markTopBot()
    assert obj.kDataFrame_standardized['tb'][-1] == 1
