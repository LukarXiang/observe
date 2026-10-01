from datetime import date

import numpy as np
import pandas as pd
import pytest

from observe.evaluation.benchmarks import benchmark_comparison, price_levels
from observe.evaluation.portfolio import metrics
from observe.ledger.book import Book
from observe.ledger.rules import Rule, RuleSet
from observe.loop import run_loop

D = list(pd.bdate_range('2024-01-02', periods = 5).date)
ZERO = RuleSet([Rule(start = date(2000, 1, 1), limit_pct = 1, commission_rate = 0, min_commission = 0, stamp_tax = 0, transfer_fee = 0)])


def test_index_gap_keeps_strategy_days_and_never_bridges_the_missing_session():
    prices = pd.DataFrame({'date': [D[0], D[1], D[3], D[4]], 'index': '000300.SH', 'close': [100, 101, 103, 104]})
    levels, meta = price_levels(prices, D, D[1:])
    rows = [{'date': d, 'equity': e} for d, e in zip(D[1:], [110, 120, 100, 105])]
    table, report = benchmark_comparison(rows, 100, levels, '000300.SH', 'ridge', 'base')
    assert len(table) == 4 and table.strategy_return.tolist() == pytest.approx([.1, 120 / 110 - 1, 100 / 120 - 1, .05])
    assert report['benchmark_missing_days'] == 2 and report['benchmark_common_days'] == 2
    assert table.benchmark_return.iloc[[0, 3]].tolist() == pytest.approx([.01, 104 / 103 - 1]) and table.benchmark_return.iloc[1:3].isna().all()
    assert report['annual_return_diff'] == pytest.approx((1.1 * 1.05) ** 121 - (1.01 * 104 / 103) ** 121)
    assert meta['anchor_date'] == str(D[0]) and report['information_ratio'] is None
    assert metrics([100, 110, 120, 100, 105], levels)['total_return'] == pytest.approx(.05)


def test_missing_anchor_and_invalid_prices_remain_missing():
    prices = pd.DataFrame({'date': D, 'index': '000300.SH', 'close': [np.nan, 101, 0, np.inf, 104]})
    levels, _ = price_levels(prices, D, D[1:])
    assert np.isnan(levels[[0, 2, 3]]).all()
    report = metrics([100, 101, 102, 103, 104], levels)
    assert report['benchmark_missing_days'] == 4 and report['annual_return_diff'] is None
    with pytest.raises(ValueError, match = '必须有限且为正'): metrics([100, 101], [100, 0])
    with pytest.raises(ValueError, match = '主键重复'): price_levels(pd.concat([prices, prices.iloc[:1]]), D, D[1:])


def test_universe_equal_holds_every_candidate_and_trims_without_liquidating():
    market = {d: {i: {'open': (10 if k <= 2 else p), 'close': (10 if k < 2 else p), 'preclose': (10 if k <= 2 else p), 'avg_amount_20d': 1e9}
                       for i, p in [('A', 12), ('B', 8)]} for k, d in enumerate(D)}
    eligible = {d: {'A', 'B'} for d in D}
    book, orders = run_loop(D, market, {d: {'A': 999, 'B': -999} for d in D}, 100000, ZERO, eligible_by_date = eligible,
                            construction = 'universe_equal', n = 1, max_weight = 1, rebalance_every = 2)
    assert (book.pos('A').qty, book.pos('B').qty) == (4200, 6200)
    trims = [o for o in orders if o['reason'] == 'equal_weight_trim']
    assert len(trims) == 1 and trims[0]['qty_requested'] == trims[0]['qty_filled'] == 800 and trims[0]['exec_date'] == D[3]
    assert book.cash == 0 and book.equity_curve()[-1] == 100000


def test_explicit_partial_sell_obeys_t_plus_one_lots_and_participation():
    book = Book(10000, D); q = {'open': 10, 'close': 10, 'preclose': 10, 'avg_amount_20d': 1000}
    book.start_day(D[0]); book.execute({'instrument': 'A', 'side': 'buy', 'amount': 3000}, q, D[0], ZERO)
    assert book.execute({'instrument': 'A', 'side': 'sell', 'qty': 100}, q, D[0], ZERO)['reject_reason'] == 'not_sellable'
    book.start_day(D[1])
    assert book.execute({'instrument': 'A', 'side': 'sell', 'qty': 50}, q, D[1], ZERO)['reject_reason'] == 'sell_unit'
    fill = book.execute({'instrument': 'A', 'side': 'sell', 'qty': 200, 'participation': 1}, q, D[1], ZERO)
    assert fill['qty_filled'] == 100 and fill['remaining_qty'] == 100 and fill['status'] == 'partial' and book.pos('A').qty == 200
