"""组合构建：调仓日定目标名单（强制退出 → 分数缺失保留 → 排名缓冲 → 普通换出 → 补足新买入），非调仓日只补单。"""


def plan_rebalance(scores, eligible, held, n = 20, buffer = 10, max_sell = 5):
    """scores: {证券: 分数}；eligible: 当天研究候选；held: 当前持仓证券集合。max_sell: None 不限，0 不做普通换出，k 为上限"""
    ranked = sorted((i for i in eligible if i in scores), key = lambda i: (-scores[i], i)); rank = {i: k for k, i in enumerate(ranked, 1)}
    forced = sorted(i for i in held if i not in eligible)                       # 不受 max_sell 限制、不计数
    missing = {i for i in held if i in eligible and i not in scores}
    buffered = {i for i in held if i in rank and rank[i] <= n + buffer}
    ordinary = sorted((i for i in held if i in rank and i not in buffered), key = lambda i: (scores[i], i))
    k = len(ordinary) if max_sell is None else max(0, int(max_sell)); sells, capped = ordinary[:k], set(ordinary[k:])
    keep = missing | buffered | capped
    buys = [i for i in ranked if i not in held][:max(0, n - len(keep))]
    return {'keep': keep, 'missing': missing, 'forced': forced, 'sells': sells, 'buys': buys, 'target': keep | set(buys), 'rank': rank}


DEFAULT_PARTICIPATION = 0.05


def rebalance_orders(plan, amount, participation = DEFAULT_PARTICIPATION):
    return [{'instrument': i, 'side': 'sell', 'qty': 'all', 'participation': participation, 'reason': 'exit_universe'} for i in plan['forced']] + \
           [{'instrument': i, 'side': 'sell', 'qty': 'all', 'participation': participation, 'reason': 'exit_rank'} for i in plan['sells']] + \
           [{'instrument': i, 'side': 'buy', 'amount': amount, 'participation': participation, 'reason': 'enter_top'} for i in plan['buys']]


def refill_orders(target, target_amount, held_value, sell_reason, rank, lot_value, completed = None, participation = DEFAULT_PARTICIPATION):
    """非调仓日：目标外持仓继续卖；目标内持有市值不足的补买差额（不足一个买入单位不补）；不重新排名、不再平衡"""
    sells = [{'instrument': i, 'side': 'sell', 'qty': 'all', 'participation': participation, 'reason': sell_reason.get(i, 'exit_rank')} for i in sorted(held_value) if i not in target]
    basis = completed if completed is not None else held_value
    gaps = {i: target_amount[i] - basis.get(i, 0.0) for i in target if i in target_amount}
    buys = [{'instrument': i, 'side': 'buy', 'amount': round(g, 2), 'participation': participation, 'reason': 'refill'} for i, g in sorted(gaps.items(), key = lambda x: rank.get(x[0], 10 ** 9))
            if g >= lot_value.get(i, float('inf'))]
    return sells + buys


def equal_weight_orders(eligible, positions, equity, quotes, rules, day, max_weight = 1.0, participation = DEFAULT_PARTICIPATION):
    """完整候选等权；调仓时重设所有目标，用决策收盘价算减持整手与增持金额。"""
    amount = round(equity * min(1 / len(eligible), max_weight), 2) if eligible else 0.0
    sells, buys = [], []
    for i in sorted(set(positions) | set(eligible)):
        p = positions.get(i); qty = p.qty if p else 0
        if i not in eligible:
            if qty: sells.append({'instrument': i, 'side': 'sell', 'qty': 'all', 'reason': 'exit_universe', 'participation': participation})
            continue
        q = quotes.get(i, {}); price = p.last_price if qty else q.get('close')
        if not price or price <= 0: continue
        unit = rules.on(day, q.get('board', 'main'), q.get('is_st', False)).buy_unit
        gap = round(amount - qty * price, 2)
        trim = int(max(0, -gap) / price // unit * unit)
        if trim: sells.append({'instrument': i, 'side': 'sell', 'qty': trim, 'reason': 'equal_weight_trim', 'participation': participation})
        elif gap >= unit * price: buys.append({'instrument': i, 'side': 'buy', 'amount': gap, 'reason': 'equal_weight_add', 'participation': participation})
    return sells + buys
