"""逐日循环：盘前公司行动 → 开盘执行上一决策的订单（先卖后买）→ 撤单 → 收盘估值勾稽 → 决策生成下一交易日订单。"""
from .ledger.book import Book
from .portfolio import DEFAULT_PARTICIPATION, equal_weight_orders, plan_rebalance, rebalance_orders, refill_orders

POLICIES = ('sell_then_buy', 'preopen_cash_only')


def run_loop(dates, market, scores_by_date, initial_cash, rules, eligible_by_date = None, actions = None, rebalance_every = 5,
             open_cash_policy = 'sell_then_buy', slippage = 0.0, n = 20, buffer = 10, max_sell = 5, max_weight = 0.10, refill_between_rebalance = True, calendar = None,
             participation = DEFAULT_PARTICIPATION, on_close = None, construction = 'topn'):
    """market: {日: {证券: {'open','close','preclose','suspended','avg_amount_20d',...}}}；返回 (账本, 订单记录)。
    on_close(日, 账本, 当日净值行) 在收盘勾稽之后、决策之前调用，用于记录逐日持仓等状态；抛出异常即中止循环"""
    if open_cash_policy not in POLICIES: raise ValueError(f'open_cash_policy must be one of {POLICIES}')
    if construction not in ('topn', 'universe_equal'): raise ValueError('未知组合构建方式')
    book = Book(initial_cash, calendar = calendar or dates); actions = actions or {}; pending, orders = [], []   # calendar 用于推断缺失日期
    target, target_amount, target_filled, sell_reason, rank = set(), {}, {}, {}, {}
    for k, day in enumerate(dates):
        quotes = market.get(day, {}); book.start_day(day, actions.get(day, ()), quotes)
        budget = book.cash if open_cash_policy == 'preopen_cash_only' else None
        remaining = []
        for o in pending:
            r = book.execute(o, quotes.get(o['instrument'], {}), day, rules, slippage, budget)
            if budget is not None and o['side'] == 'buy' and r['qty_filled']: budget -= r['value'] + r['fee']
            orders.append({**r, 'exec_date': day})                            # 未成交部分当天撤销
            if o['side'] == 'buy' and r.get('qty_filled'): target_filled[o['instrument']] = target_filled.get(o['instrument'], 0.0) + r['value']
            if construction == 'universe_equal':
                if o['side'] == 'buy':
                    left = round(o['amount'] - r.get('value', 0.0), 2)
                    if left > 0: remaining.append({**o, 'amount': left})
                elif o['qty'] == 'all':
                    if book.pos(o['instrument']).qty: remaining.append(o)
                else:
                    left = o['qty'] - r.get('qty_filled', 0)
                    if left > 0: remaining.append({**o, 'qty': left})
        row = book.close_day(day, quotes)
        if on_close: on_close(day, book, row)
        held = {i for i, p in book.positions.items() if p.qty}
        if construction == 'universe_equal':
            if k % rebalance_every == 0:
                eligible = eligible_by_date.get(day, set()) if eligible_by_date is not None else set(scores_by_date.get(day, {}))
                intents = equal_weight_orders(eligible, book.positions, row['equity'], quotes, rules, day, max_weight, participation)
            else:
                intents = remaining if refill_between_rebalance else []
                def can_refill(o):
                    if o['side'] == 'sell': return True
                    q = quotes.get(o['instrument'], {}); price = q.get('close')
                    return bool(price and o['amount'] >= price * rules.on(day, q.get('board', 'main'), q.get('is_st', False)).buy_unit)
                intents = [o for o in intents if can_refill(o)]
            pending = [{**o, 'decision_date': day} for o in intents]
            continue
        if k % rebalance_every == 0:
            scores = scores_by_date.get(day, {}); eligible = eligible_by_date.get(day, set()) if eligible_by_date is not None else set(scores)
            plan = plan_rebalance(scores, eligible, held, n, buffer, max_sell); amount = round(row['equity'] * min(1 / n, max_weight), 2)
            target, rank = plan['target'], plan['rank']
            target_amount = {**{i: target_amount[i] for i in plan['keep'] if i in target_amount}, **{i: amount for i in plan['buys']}}
            target_filled = {i: target_filled.get(i, 0.0) for i in target_amount}
            sell_reason = {**{i: 'exit_universe' for i in plan['forced']}, **{i: 'exit_rank' for i in plan['sells']}}
            pending = [{**o, 'decision_date': day} for o in rebalance_orders(plan, amount, participation)]
        elif refill_between_rebalance:
            value = {i: book.positions[i].qty * book.positions[i].last_price for i in held}
            lot = {i: q['close'] * rules.on(day, q.get('board', 'main'), q.get('is_st', False)).buy_unit for i in target if (q := quotes.get(i, {})).get('close')}
            pending = [{**o, 'decision_date': day} for o in refill_orders(target, target_amount, value, sell_reason, rank, lot, target_filled, participation)]
        else: pending = []
    return book, orders


def fixed_order_replay(orders, dates, market, initial_cash, rules, actions = None, open_cash_policy = 'sell_then_buy', slippage = 0.0, calendar = None):
    """诊断项：把一次运行已产生的订单按原执行日原样重放（例如换一套费率）。订单不随持仓与现金变化，不能替代完整重跑"""
    book = Book(initial_cash, calendar = calendar or dates); actions = actions or {}; by_day, out = {}, []
    for o in orders: by_day.setdefault(o['exec_date'], []).append({k: o[k] for k in ('instrument', 'side', 'reason') if k in o} | ({'amount': o['amount']} if o['side'] == 'buy' else {'qty': o.get('qty', 'all')}))
    for day in dates:
        quotes = market.get(day, {}); book.start_day(day, actions.get(day, ()), quotes); budget = book.cash if open_cash_policy == 'preopen_cash_only' else None
        for o in by_day.get(day, []):
            r = book.execute(o, quotes.get(o['instrument'], {}), day, rules, slippage, budget)
            if budget is not None and o['side'] == 'buy' and r['qty_filled']: budget -= r['value'] + r['fee']
            out.append({**r, 'exec_date': day})
        book.close_day(day, quotes)
    return book, out


def rerun_scenario(inputs, **changes):
    """成本情景：同一份预测与组合配置，改变费率 / 滑点后完整重跑逐日循环（不是固定订单重放）"""
    return run_loop(**{**inputs, **changes})
