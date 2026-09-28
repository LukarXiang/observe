from datetime import date

import pandas as pd
import pytest

from observe.data.standardize import market_input
from observe.data.store import Store
from observe.evaluation import metrics
from observe.factors import compute
from observe.labels import build_labels
from observe.ledger import Rule, RuleSet
from observe.loop import run_loop


def test_small_published_research_loop_is_repeatable(tmp_path):
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    bars = pd.DataFrame({'date': days * 2, 'instrument': ['A'] * 4 + ['B'] * 4,
                         'open': [10, 10, 5, 6, 20, 20, 20, 21],
                         'high': [10.5, 10.5, 5.5, 6.5, 20.5, 20.5, 21.5, 21.5],
                         'low': [9.5, 9.5, 4.5, 5.5, 19.5, 19.5, 19.5, 20.5],
                         'close': [10, 10, 5, 6, 20, 20, 21, 21], 'preclose': [10, 10, 10, 5, 20, 20, 20, 21],
                         'is_trading': True, 'is_st': False, 'board': 'main', 'volume': 1000, 'amount': 10000})
    adj = pd.DataFrame({'instrument': ['A'], 'ex_date': [days[2]], 'back_factor': [2.0]})
    store = Store(tmp_path); bid = store.write_batch({'bars_1d': {'2024': store.write_partition('bars_1d', '2024', bars)}, 'adj_factors': {'all': store.write_partition('adj_factors', 'all', adj)}}); store.publish(bid); sid = store.snapshot()
    view = market_input(store.load('bars_1d', snapshot = sid), store.load('adj_factors', snapshot = sid))
    panel = {'close_adj': view.pivot(index = 'date', columns = 'instrument', values = 'close_adj'), 'ret': view.pivot(index = 'date', columns = 'instrument', values = 'ret')}
    factor = compute('max2(ret, 0)', panel)
    labels = build_labels(view, days, h = 1)
    rules = RuleSet([Rule(start = date(2020, 1, 1), stamp_tax = 0, transfer_fee = 0, commission_rate = 0, min_commission = 0)])
    market = {d: {i: {'open': float(r.open), 'close': float(r.close), 'preclose': float(r.preclose), 'avg_amount_20d': 1e9} for i, r in view[view.date == d].set_index('instrument').iterrows()} for d in days}
    scores = {d: {'A': float(factor.loc[d, 'A']) if pd.notna(factor.loc[d, 'A']) else 0.0} for d in days}
    actions = {days[2]: [{'instrument': 'A', 'ex_date': days[2], 'bonus_ratio': 1.0}]}
    first, first_orders = run_loop(days, market, scores, 1000, rules, eligible_by_date = {d: {'A'} for d in days}, actions = actions, n = 1, max_weight = 1.0, rebalance_every = 2)
    second, _ = run_loop(days, market, scores, 1000, rules, eligible_by_date = {d: {'A'} for d in days}, actions = actions, n = 1, max_weight = 1.0, rebalance_every = 2)
    # Hand calculation: Jan 4 raw open = 5 and Jan 5 raw open = 6; factor 2 applies to both, so h=1 return is 20%.
    assert labels.loc[(labels.instrument == 'A') & (labels.decision_date == days[1]), 'value'].iloc[0] == pytest.approx(0.2)
    fills = [x for x in first_orders if x.get('qty_filled')]
    assert fills and fills[0]['fill_price'] == 10 and fills[0]['fee'] == 0 and first.cash == 0
    assert labels.valid.any() and first.equity_curve() == second.equity_curve()
    assert first.positions['A'].qty == 200
    assert first.equity_curve()[-1] == pytest.approx(1200.0)
    assert metrics(first.equity_curve())['total_return'] == pytest.approx(0.2)
