"""逐日循环：盘前公司行动 → 开盘执行上一决策的订单（先卖后买）→ 撤单 → 收盘估值勾稽 → 决策生成下一交易日订单。"""
from .ledger.book import Book
from .portfolio import plan_rebalance, rebalance_orders, refill_orders

POLICIES = ('sell_then_buy', 'preopen_cash_only')


def run_loop(dates, market, scores_by_date, initial_cash, rules, eligible_by_date = None, actions = None, rebalance_every = 5,
             open_cash_policy = 'sell_then_buy', slippage = 0.0, n = 20, buffer = 10, max_sell = 5, max_weight = 0.10, refill_between_rebalance = True, calendar = None):
    """market: {日: {证券: {'open','close','preclose','suspended','avg_amount_20d',...}}}；返回 (账本, 订单记录)"""
    if open_cash_policy not in POLICIES: raise ValueError(f'open_cash_policy must be one of {POLICIES}')
    book = Book(initial_cash, calendar = calendar or dates); actions = actions or {}; pending, orders = [], []   # calendar 用于推断缺失日期
    target, target_amount, sell_reason, rank = set(), {}, {}, {}
    for k, day in enumerate(dates):
        quotes = market.get(day, {}); book.start_day(day, actions.get(day, ()), quotes)
        budget = book.cash if open_cash_policy == 'preopen_cash_only' else None
        for o in pending:
            r = book.execute(o, quotes.get(o['instrument'], {}), day, rules, slippage, budget)
            if budget is not None and o['side'] == 'buy' and r['qty_filled']: budget -= r['value'] + r['fee']
            orders.append({**r, 'exec_date': day})                            # 未成交部分当天撤销
        row = book.close_day(day, quotes)
        held = {i for i, p in book.positions.items() if p.qty}
        if k % rebalance_every == 0:
            scores = scores_by_date.get(day, {}); eligible = eligible_by_date.get(day, set()) if eligible_by_date is not None else set(scores)
            plan = plan_rebalance(scores, eligible, held, n, buffer, max_sell); amount = round(row['equity'] * min(1 / n, max_weight), 2)
            target, rank = plan['target'], plan['rank']
            target_amount = {**{i: target_amount[i] for i in plan['keep'] if i in target_amount}, **{i: amount for i in plan['buys']}}
            sell_reason = {**{i: 'exit_universe' for i in plan['forced']}, **{i: 'exit_rank' for i in plan['sells']}}
            pending = [{**o, 'decision_date': day} for o in rebalance_orders(plan, amount)]
        elif refill_between_rebalance:
            value = {i: book.positions[i].qty * book.positions[i].last_price for i in held}
            lot = {i: q['close'] * rules.on(day, q.get('board', 'main'), q.get('is_st', False)).buy_unit for i in target if (q := quotes.get(i, {})).get('close')}
            pending = [{**o, 'decision_date': day} for o in refill_orders(target, target_amount, value, sell_reason, rank, lot)]
        else: pending = []
    return book, orders


def rerun_scenario(inputs, **changes):
    """成本情景：同一份预测与组合配置，改变费率 / 滑点后完整重跑逐日循环（不是固定订单重放）"""
    return run_loop(**{**inputs, **changes})
