"""组合构建：排名名单、完整等权基准及显式目标权重，共用唯一账本。"""
import math


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


def validate_cash_rotation(batch):
    if set(batch) != {'buy', 'sell'}: raise ValueError('Invalid cash rotation batch')
    if batch['buy'] is None and batch['sell'] is None: return
    if any(not isinstance(i, str) or not i for i in batch.values()) or batch['buy'] == batch['sell']:
        raise ValueError('Rotation requires distinct buy and sell instruments')


def cash_rotation_orders(batch, book, participation = DEFAULT_PARTICIPATION):
    """Signal state changes independently of fills; rotate only on a new switch."""
    validate_cash_rotation(batch)
    if batch['buy'] is None: return
    instrument = batch['sell']
    if instrument in book.positions and book.positions[instrument].qty:
        yield {'instrument': instrument, 'side': 'sell', 'qty': 'all', 'participation': participation, 'reason': 'rotation_exit'}
    yield {'instrument': batch['buy'], 'side': 'buy', 'amount': book.cash, 'participation': participation, 'reason': 'rotation_enter'}


def validate_slot_batch(batch):
    required = {'pool', 'buys', 'sells', 'max_positions'}
    extension = {'cash_divisor', 'order_guard_policy'}
    if set(batch) not in (required, required | extension) or type(batch['max_positions']) is not int or batch['max_positions'] < 1:
        raise ValueError('Invalid slot batch')
    if extension <= set(batch) and (type(batch['cash_divisor']) not in (int, float) or batch['cash_divisor'] != 1.5 or batch['order_guard_policy'] != 'ledger_only_v1'):
        raise ValueError('Invalid slot execution policy')
    for key in ('pool', 'buys', 'sells'):
        if not isinstance(batch[key], list) or any(not isinstance(i, str) or not i for i in batch[key]): raise ValueError('Invalid slot instruments')
    if not batch['pool'] or set(batch['buys']) & set(batch['sells']) or (set(batch['buys']) | set(batch['sells'])) - set(batch['pool']):
        raise ValueError('Invalid slot signal scope')
    if len(set(batch['sells'])) != len(batch['sells']) or any(batch['buys'].count(i) > batch['pool'].count(i) for i in batch['buys']):
        raise ValueError('Invalid slot multiplicity')


def slot_value_orders(batch, book, quotes, rules, day, participation = DEFAULT_PARTICIPATION):
    """Resume after each Book.execute so partial fills retain occupied slots."""
    validate_slot_batch(batch)
    ledger_only = batch.get('order_guard_policy') == 'ledger_only_v1'
    def at_limit(instrument, upper):
        q = quotes.get(instrument, {}); price, pre = q.get('open'), q.get('preclose')
        if price is None or pre is None: return False
        down, up = rules.limit_prices(pre, day, q.get('board', 'main'), bool(q.get('is_st', False)))
        return price >= up if upper else price <= down
    for instrument, position in list(book.positions.items()):
        if not position.qty: continue
        forced = instrument not in batch['pool']
        if forced or (instrument in batch['sells'] and (ledger_only or (position.sellable > 0 and not at_limit(instrument, True)))):
            yield {'instrument': instrument, 'side': 'sell', 'qty': 'all', 'participation': participation,
                   'reason': 'slot_exit_universe' if forced else 'slot_exit'}
    for instrument in batch['buys']:
        if instrument in book.positions and book.positions[instrument].qty: continue
        free = batch['max_positions'] - sum(p.qty > 0 for p in book.positions.values())
        if free > 0 and (ledger_only or not at_limit(instrument, False)):
            yield {'instrument': instrument, 'side': 'buy', 'amount': book.cash / (free * batch.get('cash_divisor', 1)), 'participation': participation, 'reason': 'slot_enter'}


def conditional_value_orders(signals, positions, participation = DEFAULT_PARTICIPATION):
    """固定金额入场只在实际空仓时触发；持仓后不补买，不保留未成交意图。"""
    if len(signals) != 1: raise ValueError('conditional_values 当前只支持单股信号')
    instrument, signal = next(iter(signals.items()))
    flags = ('enter_when_empty', 'exit', 'skip_when_empty')
    if not isinstance(instrument, str) or set(signal) != {*flags, 'target_value'}:
        raise ValueError('条件金额信号字段无效')
    if any(type(signal[f]) is not bool for f in flags): raise ValueError('条件金额信号必须为布尔值')
    if signal['enter_when_empty'] and signal['exit']: raise ValueError('条件入场与退出不可同时触发')
    value = signal['target_value']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('target_value 必须为有限正金额')
    flat = not any(p.qty for p in positions.values())
    if flat and signal['skip_when_empty']: return []
    qty = positions[instrument].qty if instrument in positions else 0
    if signal['exit']:
        return [{'instrument': instrument, 'side': 'sell', 'qty': 'all', 'participation': participation, 'reason': 'conditional_exit'}] if qty else []
    if flat and signal['enter_when_empty']:
        return [{'instrument': instrument, 'side': 'buy', 'amount': value, 'participation': participation, 'reason': 'conditional_enter'}]
    return []


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
        rule = rules.on(day, q.get('board', 'main'), q.get('is_st', False))
        gap = round(amount - qty * price, 2)
        trim = rule.sell_quantity(max(0, -gap) / price, qty)
        if trim: sells.append({'instrument': i, 'side': 'sell', 'qty': trim, 'reason': 'equal_weight_trim', 'participation': participation})
        elif rule.buy_quantity(gap / price): buys.append({'instrument': i, 'side': 'buy', 'amount': gap, 'reason': 'equal_weight_add', 'participation': participation})
    return sells + buys


def target_weight_orders(weights, positions, equity, quotes, rules, day, participation = DEFAULT_PARTICIPATION):
    """显式只做多权重，剩余为现金；每次调仓也调整仍在目标名单内的已有持仓。"""
    if any(not isinstance(i, str) or not math.isfinite(float(w)) or not 0 <= w <= 1 for i, w in weights.items()) or sum(weights.values()) > 1 + 1e-10:
        raise ValueError('目标权重必须有限、非负，合计不超过1')
    sells, buys = [], []
    for instrument in sorted(set(positions) | set(weights)):
        p = positions.get(instrument); qty = p.qty if p else 0
        weight = weights.get(instrument, 0.0)
        if weight == 0:
            if qty: sells.append({'instrument': instrument, 'side': 'sell', 'qty': 'all', 'reason': 'target_exit', 'participation': participation})
            continue
        q = quotes.get(instrument, {}); price = p.last_price if qty else q.get('close')
        if not price or price <= 0: continue
        rule = rules.on(day, q.get('board', 'main'), q.get('is_st', False))
        gap = round(equity * weight - qty * price, 2)
        trim = rule.sell_quantity(max(0, -gap) / price, qty)
        if trim: sells.append({'instrument': instrument, 'side': 'sell', 'qty': trim, 'reason': 'target_trim', 'participation': participation})
        elif rule.buy_quantity(gap / price): buys.append({'instrument': instrument, 'side': 'buy', 'amount': gap, 'reason': 'target_add', 'participation': participation})
    return sells + buys
