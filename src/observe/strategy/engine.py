"""声明式策略的逐日循环：收盘后（决策日）选股 / 退出检查 → 下一交易日开盘按账本规则成交。

与研究回放（loop.run_loop）共用账本、执行规则集和行情输入；不同之处在于目标名单由策略规格逐日生成，
并支持择时仓位、空仓月份、止损止盈等退出规则。
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..factors.expr import compute, parse
from ..ledger.book import Book
from ..portfolio import DEFAULT_PARTICIPATION
from .fields import FIELDS, INDEX_FIELDS

MIN_OBS_RATIO = 0.8


def schedule_days(calendar, dates, sched):
    """返回执行日集合。按完整交易日历分组（周、月），再截取回测区间，避免区间起点落在月中时把首日当成当月第一个交易日"""
    if sched.freq == 'daily': return set(dates)
    if sched.freq == 'every_n': return set(dates[::sched.n])
    cal = pd.Series(pd.to_datetime(calendar), index = calendar)
    key = cal.dt.isocalendar().week.astype(int) + cal.dt.isocalendar().year.astype(int) * 100 if sched.freq == 'weekly' else cal.dt.month + cal.dt.year * 100
    out = set()
    for _, g in cal.groupby(key.to_numpy()):
        days = list(g.index); k = sched.day - 1 if sched.day > 0 else sched.day
        if -len(days) <= k < len(days): out.add(days[k])
    return out & set(dates)


class Selector:
    """预先在整段宽表上计算规格里的全部表达式，调仓时按决策日取一行"""

    def __init__(self, spec, panel, candidates, index_panel = None):
        self.spec, self.panel, self.candidates = spec, panel, candidates
        u = spec.universe
        base = pd.DataFrame(False, index = panel.days, columns = panel.cols)
        for d, names in candidates.items():
            if d in base.index: base.loc[d, base.columns.intersection(sorted(names))] = True
        if u.exclude_paused: base &= panel['paused'].fillna(1) == 0
        if u.exclude_st: base &= panel['is_st'].fillna(1) == 0
        if u.min_listed_days: base &= panel['listed_days'].fillna(-1) >= u.min_listed_days
        self.base = base
        self.values = {}
        for e in dict.fromkeys(spec.expressions()):
            self.values[e] = compute(parse(e, FIELDS), panel, eligible = base, min_obs_ratio = MIN_OBS_RATIO, fields = FIELDS)
        self.eligible = base.copy()
        for f in u.filters: self.eligible &= self.values[f].fillna(0) > 0
        self.signal = None
        x = spec.exposure
        if x.expr is not None:
            if index_panel is None: raise ValueError(f'择时需要指数 {x.index} 的日线')
            self.signal = compute(parse(x.expr, INDEX_FIELDS), index_panel, min_obs_ratio = MIN_OBS_RATIO, fields = INDEX_FIELDS).iloc[:, 0]

    def targets(self, d):
        """决策日 d 的目标名单（有序）"""
        if d not in self.eligible.index: return []
        row = self.eligible.loc[d]; pool = list(row.index[row.to_numpy()])
        chosen = OrderedDict()
        for pipe in self.spec.select.pipelines:
            names = pool
            for s in pipe:
                v = self.values[s.filter or s.sort].loc[d, names]
                if s.filter is not None: names = list(v.index[(v.fillna(0) > 0).to_numpy()])
                else: names = list(v.dropna().sort_values(ascending = s.asc, kind = 'mergesort').index[:s.take])
            for i in names: chosen.setdefault(i, None)
        return list(chosen)[:self.spec.select.n]

    def exposure(self, d, exec_day):
        x = self.spec.exposure
        if exec_day.month in x.empty_months: return 0.0
        if self.signal is None: return x.on
        v = self.signal.get(d)
        return x.off if v is None or not np.isfinite(v) or v <= 0 else x.on

    def still_eligible(self, d, i):
        return bool(self.base.at[d, i]) and all(self.values[f].at[d, i] > 0 for f in self.spec.universe.filters) if i in self.base.columns and d in self.base.index else False


def _exit_reason(spec, panel, d, prev, i, p):
    x = spec.exits
    close = panel['close'].at[d, i] if i in panel.cols else np.nan
    if not np.isfinite(close) or p.cost <= 0: return None
    if x.stop_loss is not None and close / p.cost - 1 <= -x.stop_loss: return 'stop_loss'
    if x.take_profit is not None and close / p.cost - 1 >= x.take_profit: return 'take_profit'
    if x.limit_up_break and prev is not None:
        lu = panel['limit_up']
        if lu.at[prev, i] == 1 and lu.at[d, i] == 0: return 'limit_up_break'
    return None


def run_strategy_loop(spec, inputs, panel, selector, rules, on_close = None):
    """inputs：execution.build 的结果（行情、公司行动、交易日历）。返回 (账本, 订单记录, 每次调仓的目标名单)"""
    dates = inputs.dates; book = Book(spec.execution.initial_cash, calendar = inputs.calendar)
    exec_days = schedule_days(inputs.calendar, dates, spec.schedule)
    slip = spec.execution.slippage; pending, orders, decisions = [], [], []
    pre = [d for d in panel.days if d < dates[0]]
    if dates[0] in exec_days and pre:                     # 区间首日也是调仓日：用前一交易日收盘数据决策（预热区间内），首日开盘建仓
        x = selector.exposure(pre[-1], dates[0]); target = selector.targets(pre[-1]) if x > 0 else []
        decisions.append({'decision_date': pre[-1], 'exec_date': dates[0], 'exposure': x, 'targets': target})
        if target:
            each = round(book.cash * x / len(target), 2)
            pending = [{'instrument': i, 'side': 'buy', 'amount': each, 'reason': 'rebalance_in', 'participation': DEFAULT_PARTICIPATION, 'decision_date': pre[-1]} for i in target]
    for k, day in enumerate(dates):
        quotes = inputs.market.get(day, {}); book.start_day(day, inputs.actions.get(day, ()), quotes)
        for i, p in list(book.positions.items()):
            if p.qty and quotes.get(i, {}).get('delisted'): book.settle_delisted(day, i)
        for o in pending:
            orders.append({**book.execute(o, quotes.get(o['instrument'], {}), day, rules, slip), 'exec_date': day})
        row = book.close_day(day, quotes)
        if on_close: on_close(day, book, row)
        if k + 1 >= len(dates): break
        nxt, prev = dates[k + 1], dates[k - 1] if k else None
        held = {i: p for i, p in book.positions.items() if p.qty}
        sells = OrderedDict()
        for i, p in held.items():
            r = _exit_reason(spec, panel, day, prev, i, p)
            if r is None and spec.exits.drop_from_target and not selector.still_eligible(day, i): r = 'drop_from_universe'
            if r: sells[i] = r
        buys = []
        if nxt in exec_days:
            x = selector.exposure(day, nxt); target = selector.targets(day) if x > 0 else []
            decisions.append({'decision_date': day, 'exec_date': nxt, 'exposure': x, 'targets': target})
            for i in held:
                if i not in target: sells.setdefault(i, 'rebalance_out' if x > 0 else 'exposure_off')
            buys = _buys(spec, book, quotes, held, sells, target, x, row['equity'], rules, day)
            if spec.rebalance.mode == 'reweight': buys, partial = buys; sells.update(partial)
        pending = [{'instrument': i, 'side': 'sell', 'qty': r[1] if isinstance(r, tuple) else 'all', 'reason': r[0] if isinstance(r, tuple) else r, 'decision_date': day}
                   for i, r in sells.items()] + [{**b, 'decision_date': day} for b in buys]
    return book, orders, decisions


def _value(p, quotes, i):
    px = quotes.get(i, {}).get('close') or p.last_price or 0.0
    return p.qty * px


def _buys(spec, book, quotes, held, sells, target, x, equity, rules, day):
    """keep：保留仍在目标里的持仓，用（现金 + 预计卖出所得）等分买入新进名单，总仓位不超过 equity × x。
    reweight：每个目标调到 equity × x / 目标数，偏离超过 band 才调整；返回 (买单, 减仓卖单)"""
    part = {'participation': DEFAULT_PARTICIPATION}
    if not target or x <= 0: return ([], {}) if spec.rebalance.mode == 'reweight' else []
    sell_value = sum(_value(held[i], quotes, i) for i in sells if i in held)
    if spec.rebalance.mode == 'keep':
        kept = [i for i in target if i in held and i not in sells]; new = [i for i in target if i not in held]
        if not new: return []
        room = equity * x - sum(_value(held[i], quotes, i) for i in kept)
        budget = min(book.cash + sell_value * (1 - 0.0015), room)       # 卖出所得按约 0.15% 费用与滑点折算后计入可用资金
        if budget <= 0: return []
        each = budget / len(new)
        return [{'instrument': i, 'side': 'buy', 'amount': round(each, 2), 'reason': 'rebalance_in', **part} for i in new]
    each = equity * x / len(target); band = spec.rebalance.band; buys, partial = [], {}
    for i in target:
        if i in sells: continue
        have = _value(held[i], quotes, i) if i in held else 0.0; diff = each - have
        if i in held and abs(diff) <= band * each: continue
        if diff > 0: buys.append({'instrument': i, 'side': 'buy', 'amount': round(diff, 2), 'reason': 'rebalance_in' if i not in held else 'reweight_up', **part})
        else:
            q = quotes.get(i, {}); px = q.get('close')
            if not px: continue
            unit = rules.on(day, q.get('board', 'main'), q.get('is_st', False)).buy_unit
            lots = int(-diff / px // unit * unit)
            if lots > 0: partial[i] = ('reweight_down', lots)
    return buys, partial


def summarize_targets(decisions):
    return [{'decision_date': d['decision_date'], 'exec_date': d['exec_date'], 'exposure': d['exposure'], 'n_targets': len(d['targets']),
             'targets': ','.join(d['targets'])} for d in decisions]
