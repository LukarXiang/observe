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


def rebalance_orders(plan, amount):
    return [{'instrument': i, 'side': 'sell', 'qty': 'all', 'reason': 'exit_universe'} for i in plan['forced']] + \
           [{'instrument': i, 'side': 'sell', 'qty': 'all', 'reason': 'exit_rank'} for i in plan['sells']] + \
           [{'instrument': i, 'side': 'buy', 'amount': amount, 'reason': 'enter_top'} for i in plan['buys']]


def refill_orders(target, target_amount, held_value, sell_reason, rank, lot_value):
    """非调仓日：目标外持仓继续卖；目标内持有市值不足的补买差额（不足一个买入单位不补）；不重新排名、不再平衡"""
    sells = [{'instrument': i, 'side': 'sell', 'qty': 'all', 'reason': sell_reason.get(i, 'exit_rank')} for i in sorted(held_value) if i not in target]
    gaps = {i: target_amount[i] - held_value.get(i, 0.0) for i in target if i in target_amount}
    buys = [{'instrument': i, 'side': 'buy', 'amount': round(g, 2), 'reason': 'refill'} for i, g in sorted(gaps.items(), key = lambda x: rank.get(x[0], 10 ** 9))
            if g >= lot_value.get(i, float('inf'))]
    return sells + buys
