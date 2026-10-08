import ast
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.review_strategy_batch51 import ema_reference, emotion_reference, momentum_kernels, parse_daily, weekly_score


def daily(dates, closes):
    return pd.DataFrame({'date': dates, 'open': closes, 'high': closes, 'low': closes, 'close': closes, 'volume': ['0']*len(dates)})


def test_daily_string_columns_sorted_and_anchor_explicit():
    frame = daily(['2026-09-30', '2026-09-28', '2026-09-29'], ['3', '1', '2'])
    result = parse_daily(frame)
    assert result.date.tolist() == ['2026-09-28', '2026-09-29']
    assert result.close.tolist() == [1., 2.]
    assert frame.close.tolist() == ['3', '1', '2']


@pytest.mark.parametrize('dates,closes', [(['2026-09-29']*2, ['1', '2']),
    (['2026-09-30']*2, ['1', '2']), (['2026-09-29'], ['inf']),
    (['2026-09-29'], [None]), (['2026-09-29'], ['bad']),
    (['2026-09-29'], ['0']), (['2026-09-30'], ['1'])])
def test_daily_rejects_bad_operands_even_after_anchor(dates, closes):
    with pytest.raises(ValueError): parse_daily(daily(dates, closes))


def test_daily_rejects_negative_volume():
    frame = daily(['2026-09-29'], ['1']); frame['volume'] = '-1'
    with pytest.raises(ValueError): parse_daily(frame)


def test_weighted_weekly_score_retains_negative_values():
    assert weekly_score([100.]*8+[90.]) == pytest.approx(-.1)
    assert weekly_score([100.]*9) == 0.


@pytest.mark.parametrize('values', [[1.]*8, [1.]*8+[0.], [1.]*8+[math.inf]])
def test_weekly_rejects_incomplete_or_invalid_window(values):
    with pytest.raises(ValueError): weekly_score(values)


def test_ema_mean_seed_and_recursion():
    result = ema_reference([1., 2., 3., 6.], 3)
    assert all(math.isnan(v) for v in result[:2])
    assert result[2:] == [2., 4.]
    assert ema_reference([5.]*15, 12)[11:] == [5.]*4


@pytest.mark.parametrize('positive,length,expected', [(True, 2, 0), (True, 3, 1), (False, 5, 0), (False, 6, -1)])
def test_emotion_strict_cross_lengths(positive, length, expected):
    old, new = (-1., 1.) if positive else (1., -1.)
    assert emotion_reference([old]*(30-length)+[new]*length) == expected


def test_emotion_no_cross_and_zero_are_not_fail_open():
    assert emotion_reference([1.]*30) is None
    assert emotion_reference([0.]*30) is None
    assert emotion_reference([np.nan]*30) is None


def test_original_loop_subscript_assignments_do_not_break_expression_extraction():
    tree = ast.parse('''def get_signal(context):
    for row in g.ETFList:
        cp_increase = 100*(current_price/close_data['close'][0]-1)
        ma_n1 = (current_price+close_data['close'].sum()-close_data['close'][0])/g.lag2
        pre_price = current_price-ma_n1*g.ma_threshold
        df.loc[0, 'code'] = row[1]
''')
    code, digest = momentum_kernels(tree)
    env = {'current_price': 110., 'close_data': {'close': np.array([100.]*10)}, 'g': SimpleNamespace(lag2=10, ma_threshold=.9)}
    assert eval(code['cp_increase'], env) == pytest.approx(10.)
    env['ma_n1'] = eval(code['ma_n1'], env)
    assert env['ma_n1'] == 101.
    assert eval(code['pre_price'], env) == pytest.approx(19.1)
    assert len(digest) == 64
