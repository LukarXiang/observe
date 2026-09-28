"""组合构建与逐日循环（模块 15、16 用例 1、5、12、15、16、18）。价格恒定、无费用时现金轨迹可直接手算。"""
from datetime import date

import pytest

from observe.ledger import Rule, RuleSet
from observe.loop import rerun_scenario, run_loop
from observe.portfolio import plan_rebalance, refill_orders

D = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10)]
ZERO = RuleSet([Rule(start = date(2020, 1, 1), stamp_tax = 0, transfer_fee = 0, commission_rate = 0, min_commission = 0)])
FEE = RuleSet([Rule(start = date(2020, 1, 1))])


def flat(names, price = 10.0, days = D): return {d: {i: {'open': price, 'close': price, 'preclose': price, 'avg_amount_20d': 1e9} for i in names} for d in days}
def cash_path(book): return [r['cash'] for r in book.equity_rows]
def holding(book): return {i: p.qty for i, p in book.positions.items() if p.qty}


# 组合构建 ---------------------------------------------------------------------
def test_buffer_keeps_rank_25_and_sells_rank_31():
    scores = {f'S{k:02d}': 100 - k for k in range(1, 41)}                                      # S01 排第 1
    plan = plan_rebalance(scores, set(scores), {'S25', 'S31'}, n = 20, buffer = 10, max_sell = None)
    assert 'S25' in plan['keep'] and plan['sells'] == ['S31']


def test_score_missing_kept_and_forced_exit_not_counted():
    scores = {'A': 1, 'B': 2, 'C': 3, 'D': 4}; held = {'A', 'B', 'M', 'X'}                  # M 在候选但没分数；X 不在候选
    plan = plan_rebalance(scores, {'A', 'B', 'C', 'D', 'M'}, held, n = 2, buffer = 0, max_sell = 0)
    assert plan['forced'] == ['X'] and plan['sells'] == [] and 'M' in plan['keep']            # max_sell = 0 仍然强制退出 X
    assert plan['buys'] == []                                                                  # 保留 A、B、M 已超过 N = 2


@pytest.mark.parametrize('max_sell, sells', [(None, ['A', 'B']), (0, []), (1, ['A'])])
def test_max_sell_null_zero_k(max_sell, sells):
    plan = plan_rebalance({'A': 1, 'B': 2, 'C': 3, 'D': 4}, {'A', 'B', 'C', 'D'}, {'A', 'B'}, n = 2, buffer = 0, max_sell = max_sell)
    assert plan['sells'] == sells and len(plan['target']) == 2


def test_fewer_candidates_than_n_keeps_cash():
    plan = plan_rebalance({f'S{k}': k for k in range(12)}, {f'S{k}' for k in range(12)}, set(), n = 20)
    assert len(plan['buys']) == 12


def test_refill_does_not_rerank_and_skips_sub_lot_gaps():
    orders = refill_orders({'A', 'B'}, {'A': 1000, 'B': 1000}, {'A': 950, 'X': 500}, {'X': 'exit_rank'}, {'A': 1, 'B': 2}, {'A': 1000, 'B': 1000})
    assert [(o['instrument'], o['side']) for o in orders] == [('X', 'sell'), ('B', 'buy')]    # A 只差 50，不足一手不补


# 逐日循环 ---------------------------------------------------------------------
BASE = dict(dates = D[:5], market = flat('ABCD', days = D[:5]), initial_cash = 2000, rules = ZERO, n = 2, buffer = 0, max_sell = None, max_weight = 1.0,
            rebalance_every = 2, scores_by_date = {D[0]: {'A': 4, 'B': 3, 'C': 2, 'D': 1}, D[2]: {'C': 4, 'D': 3, 'A': 2, 'B': 1}, D[4]: {'C': 4, 'D': 3, 'A': 2, 'B': 1}},
            eligible_by_date = {d: set('ABCD') for d in D})


def test_full_rotation_cash_path_under_both_policies():
    """D0 决策买 A、B；D1 成交满仓；D2 决策换成 C、D；D3 开盘先卖 A、B"""
    b1, o1 = run_loop(**BASE)
    assert cash_path(b1) == [2000, 0, 0, 0, 0] and holding(b1) == {'C': 100, 'D': 100}      # 先卖后买：回款当场买入
    b2, _ = run_loop(**{**BASE, 'open_cash_policy': 'preopen_cash_only'})
    assert cash_path(b2) == [2000, 0, 0, 2000, 0] and holding(b2) == {'C': 100, 'D': 100}   # 回款隔一天由补单买入
    b3, _ = run_loop(**{**BASE, 'open_cash_policy': 'preopen_cash_only', 'refill_between_rebalance': False})
    assert cash_path(b3) == [2000, 0, 0, 2000, 2000] and holding(b3) == {}                   # 不补单：现金闲置到下次调仓
    assert all(o['exec_date'] > o['decision_date'] for o in o1)                                   # 最早下一交易日成交
    assert holding(b1)                                                                            # 区间结束不自动清仓


def test_failed_sell_is_retried_and_funds_the_next_buy():
    m = flat('AB', 10.0, D[:5]); m.update({d: {**m[d], 'B': {'open': 9, 'close': 9, 'preclose': 9, 'avg_amount_20d': 1e9}} for d in D[:5]})
    m[D[3]]['A'] = {'open': 9, 'close': 9, 'preclose': 10, 'avg_amount_20d': 1e9}; m[D[4]]['A'] = {'open': 9, 'close': 9, 'preclose': 9, 'avg_amount_20d': 1e9}   # D3 开盘跌停
    book, orders = run_loop(D[:5], m, {D[0]: {'A': 2, 'B': 1}, D[2]: {'B': 2, 'A': 1}, D[4]: {'B': 2, 'A': 1}}, 1000, ZERO,
                            eligible_by_date = {d: {'A', 'B'} for d in D}, rebalance_every = 2, n = 1, buffer = 0, max_sell = None, max_weight = 1.0)
    d3 = [(o['instrument'], o.get('reject_reason')) for o in orders if o['exec_date'] == D[3]]
    assert d3 == [('A', 'limit_down'), ('B', 'cash')]                     # 卖不掉的 A 仍占着全部资金
    d4 = [(o['instrument'], o['side'], o['qty_filled']) for o in orders if o['exec_date'] == D[4]]
    assert d4 == [('A', 'sell', 100), ('B', 'buy', 100)]                  # 补卖 A 得 900，补买 B 100 股 × 9
    assert holding(book) == {'B': 100} and book.cash == 0


def test_bonus_across_pending_sell_order_and_listing_date():
    m = flat('AB', 10.0, D[:5]); m[D[3]]['A'] = m[D[4]]['A'] = {'open': 5, 'close': 5, 'preclose': 5, 'avg_amount_20d': 1e9}
    acts = {D[3]: [{'instrument': 'A', 'ex_date': D[3], 'bonus_ratio': 1.0, 'bonus_list_date': D[4]}]}
    book, orders = run_loop(D[:5], m, {D[0]: {'A': 1}}, 1000, ZERO, eligible_by_date = {D[0]: {'A'}, D[2]: {'B'}, D[4]: {'B'}},
                            actions = acts, rebalance_every = 2, n = 1, buffer = 0, max_sell = None, max_weight = 1.0)
    sells = [(o['exec_date'], o['qty_filled'], o['reason']) for o in orders if o['side'] == 'sell']
    assert sells == [(D[3], 100, 'exit_universe'), (D[4], 100, 'exit_universe')]   # 除权日只能卖原有 100 股，新增 100 股上市日才卖
    assert book.cash == 1000 and holding(book) == {}


def test_buy_order_across_ex_date_uses_post_split_price():
    m = flat('A', 10.0, D[:3]); m[D[1]]['A'] = {'open': 5, 'close': 5, 'preclose': 5, 'avg_amount_20d': 1e9}
    book, orders = run_loop(D[:3], m, {D[0]: {'A': 1}}, 2000, ZERO, actions = {D[1]: [{'instrument': 'A', 'ex_date': D[1], 'bonus_ratio': 1.0, 'bonus_list_date': D[2]}]},
                            n = 1, buffer = 0, max_weight = 1.0)
    assert orders[0]['qty_filled'] == 400 and book.pos('A').pending == 0        # 金额 2000 / 除权后 5 元；除权当天才买入，不享受送股


def test_scores_changing_between_rebalances_do_not_trigger_trades():
    scores = {d: {'A': 2, 'B': 1} for d in D}; scores[D[1]] = {'B': 9, 'A': 1}                  # 非调仓日分数翻转
    book, orders = run_loop(D[:3], flat('AB', days = D[:3]), scores, 1000, ZERO, eligible_by_date = {d: {'A', 'B'} for d in D},
                            rebalance_every = 5, n = 1, buffer = 0, max_weight = 1.0)
    assert [o['instrument'] for o in orders] == ['A'] and holding(book) == {'A': 100}


def test_cost_scenario_reruns_loop_and_changes_second_rebalance_amount():
    """D1 买 A：1e6/10 → 100000 股放不下费用，缩到 99900 股 = 999000；基础佣金 249.75 + 过户 9.99 → 现金 740.26。
    佣金加倍：499.50 + 9.99 → 现金 490.51。D1 收盘调仓换 B，目标金额 = 当日净值。"""
    inputs = dict(dates = D[:3], market = flat('AB', days = D[:3]), scores_by_date = {D[0]: {'A': 2, 'B': 1}, D[1]: {'B': 2, 'A': 1}},
                  initial_cash = 1_000_000, rules = FEE, eligible_by_date = {d: {'A', 'B'} for d in D}, rebalance_every = 1, n = 1, buffer = 0, max_sell = None, max_weight = 1.0)
    for rules, cash in ((FEE, 740.26), (FEE.scaled(commission_rate = 2), 490.51)):
        book, orders = rerun_scenario(inputs, rules = rules)
        assert book.equity_rows[1]['cash'] == pytest.approx(cash)
        b = [o for o in orders if o['instrument'] == 'B'][0]
        assert b['reason'] == 'enter_top' and b['amount'] == pytest.approx(999_000 + cash)   # 第二次调仓的目标金额随成本变化
        assert book.equity_rows[1]['equity'] == pytest.approx(999_000 + cash)


def test_share_conversion_moves_position_and_cost():
    from observe.ledger import Book
    b = Book(0, D); p = b.pos('OLD'); p.qty, p.cost, p.last_price = 1000, 8.0, 10.0
    b.start_day(D[0], [{'instrument': 'OLD', 'ex_date': D[0], 'convert_to': 'NEW', 'convert_ratio': 1.2345}])
    n = b.pos('NEW'); assert (b.pos('OLD').qty, n.qty) == (0, 1234) and n.cost == pytest.approx(8000 / 1234) and n.last_price == pytest.approx(10 / 1.2345)
    assert b.assumptions[0]['field'] == 'conversion_fraction' and b.issues[0]['kind'] == 'converted'
    assert b.close_day(D[0], {'NEW': {'close': 10 / 1.2345}})['equity'] == pytest.approx(1234 * 10 / 1.2345, abs = 0.01)


def test_fixed_order_replay_reproduces_run_and_differs_from_full_rerun_under_new_cost():
    from observe.loop import fixed_order_replay
    # 半仓、股价 1 元：买单金额不受现金约束，两套费率下的金额差会落到不同的手数上
    inputs = dict(dates = D[:3], market = flat('AB', 1.0, D[:3]), scores_by_date = {D[0]: {'A': 2, 'B': 1}, D[1]: {'B': 2, 'A': 1}},
                  initial_cash = 1_000_000, rules = FEE, eligible_by_date = {d: {'A', 'B'} for d in D}, rebalance_every = 1, n = 1, buffer = 0, max_sell = None, max_weight = 0.5)
    base, orders = run_loop(**inputs)
    same, _ = fixed_order_replay(orders, D[:3], inputs['market'], 1_000_000, FEE)
    assert [r['equity'] for r in same.equity_rows] == [r['equity'] for r in base.equity_rows]           # 原费率重放 = 原运行
    costly = FEE.scaled(commission_rate = 2); rerun, _ = run_loop(**{**inputs, 'rules': costly}); replay, _ = fixed_order_replay(orders, D[:3], inputs['market'], 1_000_000, costly)
    qty = lambda book: book.pos('B').qty
    assert qty(replay) == qty(base) != qty(rerun)                                                         # 重放沿用基础情景的买单金额；完整重跑按更高成本后的净值重算
    assert replay.equity_rows[-1]['equity'] != rerun.equity_rows[-1]['equity']
