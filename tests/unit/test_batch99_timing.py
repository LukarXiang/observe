"""Original decision cutoff, held-position branches and dated fees."""
from pathlib import Path

from scripts import diagnose_strategy_batch99_timing as timing
from scripts import research_strategy_batch99 as stage


def cases():
    def loader(n): return stage.definitions(Path('repo/量化策略源代码')/stage.SOURCE_PATHS[n], n)
    return {c['case']: c['calls'] for c in timing.study(loader)['cases']}


def test_trade_uses_previous_date_and_fifteen_rows():
    for name, calls in cases().items():
        if name.startswith('fees'): continue
        assert calls[:2] == [['get_trade_days', {'end_date': '2018-07-20', 'count': 15}],
                             ['CH_VIX', ['510050.XSHG', '2018-07-02', '2018-07-20']]]


def test_equal_EMA_buys_total_value_only_when_not_held():
    rows = cases()
    assert rows['flat_buy_empty'][-1] == ['order_target_value', ['510050.XSHG', 123456.]]
    assert len(rows['flat_buy_held']) == 3


def test_rising_EMA_sells_only_existing_position():
    rows = cases()
    assert rows['rising_sell_held'][-1] == ['order_target', ['510050.XSHG', 0]]
    assert len(rows['rising_sell_empty']) == 3


def test_original_dated_commissions_zero_slippage():
    rows = cases()
    for date, buy, sell in [('2008-12-31', .003, .004), ('2010-01-04', .002, .003),
                            ('2012-01-04', .001, .002), ('2015-02-09', .0003, .0013)]:
        assert rows['fees_'+date] == [['slippage', 0],
                                    ['commission', {'buy_cost': buy, 'sell_cost': sell, 'min_cost': 5}]]
