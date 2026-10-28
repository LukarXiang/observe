"""Check original intraday source boundaries without inventing full trading inputs."""
import datetime as dt

import pandas as pd
import pytest

from scripts import review_strategy_batch90 as batch


@pytest.fixture
def original():
    if not (batch.Path('repo/量化策略源代码')/batch.SOURCES[0]).is_file():
        pytest.skip('Restore original SHA-pinned batch90 sources')
    return batch


def test_original_time_and_price_boundaries(original):
    result = original.network_disabled(original.diagnostics)
    cases = {r['case']: r for r in result['cases']}
    assert result['case_count'] == 13
    assert cases['89_time_930']['events'] == cases['89_time_1031']['events'] == []
    assert cases['89_time_931']['events'] == cases['89_time_1030']['events'] == ['buy']
    assert cases['89_time_1450']['events'] == ['sell']
    assert cases['89_stock_price_109.0']['orders'] == []
    assert cases['89_stock_price_109.01']['orders'] == [['s', 10000.]]
    assert cases['89_datetime_series_minus1']['exception'] == 'KeyError'
    assert not result['platform_equivalence_proven'] and result['new_trading_backtests'] == 0


def test_original_filter_uses_series_index_not_stock_name(original):
    ns = original.namespace(1)
    pool = pd.DataFrame({'display_name': ['ST fixture', '*退 fixture', 'normal']},
                        index=['600000.XSHG', '600001.XSHG', '688001.XSHG'])
    assert ns['filter_stock'](pool) == ['600000.XSHG', '600001.XSHG']
    pool.index = ['ST', '600001.XSHG', '688001.XSHG']
    assert ns['filter_stock'](pool) == []


def test_natural_day_helper_does_not_synthesize_a_trading_day(original):
    ns = original.namespace(1, {'datetime': dt})
    assert ns['getday3'](dt.date(2021, 1, 4), -1) == dt.date(2021, 1, 3)


@pytest.mark.parametrize('key,value', [('p1_ma5_close', 11.), ('p2_ma5_close', 10.),
    ('p2_ma5_volume', 100.), ('p1_ma5_volume', 100.), ('p1_ma20_close', 20.), ('p1_ma30_close', 30.)])
def test_original_six_conditions_are_strict(original, key, value):
    inputs = {'p1_ma5_close': 10., 'p1_ma10_close': 11., 'p2_ma5_close': 11., 'p2_ma10_close': 10.,
        'p2_ma5_volume': 101., 'p2_ma10_volume': 100., 'p1_ma5_volume': 110., 'p1_ma10_volume': 120.,
        'p1_ma20_close': 21., 'p2_ma20_close': 20., 'p1_ma30_close': 31., 'p2_ma30_close': 30.}
    code = original.ma_predicate()
    assert eval(code, {'__builtins__': {}}, inputs)
    assert not eval(code, {'__builtins__': {}}, {**inputs, key: value})


def test_source_tamper_is_blocked_before_loading(original, monkeypatch):
    monkeypatch.setattr(original, 'file_sha', lambda path: 'changed')
    with pytest.raises(ValueError, match='Original source changed'):
        original.namespace(0)


def test_division_fixture_is_not_exact_equality(original):
    inputs = {'p1_ma5_close': 10., 'p1_ma10_close': 11., 'p2_ma5_close': 11., 'p2_ma10_close': 10.,
        'p2_ma5_volume': 101., 'p2_ma10_volume': 100., 'p1_ma5_volume': 100./1.2, 'p1_ma10_volume': 100.,
        'p1_ma20_close': 21., 'p2_ma20_close': 20., 'p1_ma30_close': 31., 'p2_ma30_close': 30.}
    assert inputs['p1_ma5_volume']*1.2 > 100.
    assert eval(original.ma_predicate(), {'__builtins__': {}}, inputs)
