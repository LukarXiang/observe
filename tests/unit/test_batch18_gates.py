import pandas as pd
import pytest

from scripts.review_strategy_batch18 import missing_stock_components, raw_quality


@pytest.mark.parametrize('absent', ['bars', 'factors'])
@pytest.mark.parametrize('empty', [False, True])
def test_downloaded_stock_requires_both_nonempty_components(absent, empty):
    frames = {f'300014.SZ_{k}': pd.DataFrame({'value': [1]}) for k in ('bars', 'factors')}
    if empty:
        frames[f'300014.SZ_{absent}'] = pd.DataFrame()
    else:
        del frames[f'300014.SZ_{absent}']
    assert missing_stock_components(frames, '300014.SZ') == [absent]
    assert missing_stock_components({}, '000651.SZ') == []


@pytest.mark.parametrize('column,value', [('tradestatus', ''), ('isST', 'unknown'), ('volume', '-1'), ('amount', 'nan'), ('volume', 'inf')])
def test_input_quality_rejects_unknown_flags_or_invalid_flow(column, value):
    frame = pd.DataFrame({'tradestatus': ['1'], 'isST': ['0'], 'volume': ['100'], 'amount': ['200']})
    assert raw_quality(frame) == {'invalid_raw_flags': {'tradestatus': 0, 'isST': 0}, 'invalid_flow_cells': 0}
    frame.loc[0, column] = value
    quality = raw_quality(frame)
    assert sum(quality['invalid_raw_flags'].values()) + quality['invalid_flow_cells'] == 1
