"""执行输入适配器：把同一快照里的交易日历、原始日线、证券资料和公司行动转换成账本行情（决策 6）。
run、reproduce、队列和公开入口测试都经过这里，没有第二套执行口径。

- 下单、成交、费用、资金账只用原始价与对应的前收盘价；复权价不进入这里（研究视图见 data.prices）。
- 交易日循环由交易日历驱动；执行区间内整日缺失日线直接阻断，不缩短回放。
- 停牌行保留为 suspended，已有持仓继续按最近有效价估值；研究候选与账本行情是两个集合。
- 参考成交额来自决策时点之前 N 个交易日的真实成交额，预热区间不足时明确不可用。
"""
import math
from bisect import bisect_left
from dataclasses import dataclass, field

import pandas as pd

from .data.standardize import board as board_of, flag

PRICE_TOL = 0.005        # 前收盘与上一有效收盘相差超过半个报价单位，说明发生了除权除息
TABLES = ('calendar', 'bars_1d', 'instruments', 'corp_actions')


class InputBlocked(RuntimeError):
    """输入不足以执行：不是程序错误，运行状态记为 blocked"""
    def __init__(self, issues):
        self.issues = issues; super().__init__('; '.join(f"{x['kind']}: {x.get('detail', '')}" for x in issues))


@dataclass
class ExecutionInput:
    dates: list                                   # 执行区间内的全部交易日（来自交易日历）
    calendar: list                                # 快照里的全部交易日：账本推断到账日、红股上市日用
    market: dict                                  # {交易日: {证券: 行情}}，只含原始价
    candidates: dict                              # {决策日: 研究候选集合}
    actions: dict                                 # {除权除息日: [公司行动]}
    unexplained: set = field(default_factory = set)   # {(交易日, 证券)}：前收盘被重设，但快照里没有对应的公司行动
    limitations: list = field(default_factory = list)
    info: dict = field(default_factory = dict)


def _date(x):
    if x is None: return None
    try:
        if pd.isna(x): return None
    except (TypeError, ValueError): pass
    return pd.Timestamp(x).date()


def _num(x):
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None


def sessions(calendar):
    if calendar is None or not len(calendar) or not {'date', 'is_open'}.issubset(calendar.columns): return []
    return sorted({_date(d) for d, o in zip(calendar.date, calendar.is_open) if flag(o)})


def load(store, state, start = None, end = None, liquidity_window = 20):
    """从同一个已冻结状态读取执行需要的表；日线只读执行区间与预热区间涉及的年份分区"""
    cal = store.load_state(state, 'calendar'); days = sessions(cal); parts = None
    s, e = _date(start), _date(end)
    if s is not None and days:
        k = max(bisect_left(days, s) - liquidity_window, 0); first = days[k].year
        parts = [p for p in state['tables'].get('bars_1d', {}) if not p.isdigit() or (int(p) >= first and (e is None or int(p) <= e.year))]
    used = {t: {p: v for p, v in state['tables'].get(t, {}).items() if t != 'bars_1d' or parts is None or p in parts} for t in TABLES}
    return {'calendar': cal, 'bars_1d': store.load_state(state, 'bars_1d', parts = parts), 'instruments': store.load_state(state, 'instruments'),
            'corp_actions': store.load_state(state, 'corp_actions'), 'partitions': used}


def _action(r):
    num = lambda k: _num(r.get(k)) or 0.0
    return {'instrument': r['instrument'], 'ex_date': _date(r['ex_date']), 'cash_per_share': num('cash_per_share'), 'bonus_ratio': num('bonus_ratio'),
            'rights_ratio': num('rights_ratio'), 'rights_price': num('rights_price'), 'record_date': _date(r.get('record_date')),
            'pay_date': _date(r.get('pay_date')), 'bonus_list_date': _date(r.get('bonus_list_date')), 'source': r.get('source')}


def build(tables, start = None, end = None, boards = ('main',), liquidity_window = 20, liquidity_override = None):
    """tables: load() 的结果。返回 ExecutionInput；输入不成立时抛 InputBlocked。

    liquidity_override 只给隔离的合成测试用：显式设定参考成交额，并记为限制，不是真实快照的默认值。"""
    if liquidity_window < 1: raise ValueError('liquidity_window must be >= 1')
    cal = sessions(tables.get('calendar'))
    if not cal: raise InputBlocked([{'kind': 'calendar_missing', 'detail': '快照没有交易日历'}])
    inst = tables.get('instruments')
    if inst is None or not len(inst) or not {'instrument', 'kind'}.issubset(inst.columns):
        raise InputBlocked([{'kind': 'instruments_missing', 'detail': '快照没有证券资料（证券类型）'}])
    b = tables.get('bars_1d')
    if b is None or not len(b): raise InputBlocked([{'kind': 'bars_missing', 'detail': '快照没有日线'}])
    b = b.copy(); b['date'] = b.date.map(_date); have = sorted(set(b.date)); on_cal = set(cal)
    s, e = _date(start), _date(end)
    if e is None: e = have[-1]
    if s is None:                                   # 默认从预热区间满足之后的第一个交易日开始
        k = bisect_left(cal, have[0]) + liquidity_window
        if k >= len(cal) or cal[k] > e: raise InputBlocked([{'kind': 'insufficient_history', 'detail': f'日线不足 {liquidity_window} 个交易日的预热区间'}])
        s = cal[k]
    if s > e: raise InputBlocked([{'kind': 'empty_range', 'detail': f'{s} > {e}'}])
    if s < cal[0] or e > cal[-1]: raise InputBlocked([{'kind': 'calendar_does_not_cover', 'detail': f'交易日历 {cal[0]}..{cal[-1]} 不覆盖 {s}..{e}'}])
    dates = [d for d in cal if s <= d <= e]
    if not dates: raise InputBlocked([{'kind': 'empty_range', 'detail': f'{s}..{e} 没有交易日'}])
    issues = []
    missing = [d for d in dates if d not in set(have)]
    if missing: issues.append({'kind': 'missing_session', 'detail': f'{len(missing)} 个交易日整日没有日线', 'dates': [str(d) for d in missing]})
    closed = sorted({d for d in have if s <= d <= e and d not in on_cal})
    if closed: issues.append({'kind': 'bar_on_closed_day', 'detail': f'{len(closed)} 个非交易日有日线', 'dates': [str(d) for d in closed]})
    if issues: raise InputBlocked(issues)

    limitations = []
    ks = bisect_left(cal, s); kw = ks - liquidity_window
    warmup_start = cal[max(kw, 0)]
    if kw < 0 or cal[kw] < have[0]:
        limitations.append({'kind': 'liquidity_warmup_short', 'detail': f'{s} 之前不足 {liquidity_window} 个交易日的成交额，前几个交易日没有参考成交额'})
    span = [d for d in cal if warmup_start <= d <= e]
    b = b[b.date.isin(set(span))]
    if liquidity_override is None:
        amount = b.pivot(index = 'date', columns = 'instrument', values = 'amount').reindex(span).apply(pd.to_numeric, errors = 'coerce')
        ref = amount.rolling(liquidity_window, min_periods = liquidity_window).mean().shift(1).loc[dates[0]:]
        ref = {k: float(v) for k, v in ref.stack().dropna().items()}
        liquidity = {'mode': 'history', 'window': liquidity_window}
    else:
        ref = None; liquidity = {'mode': 'override', 'value': float(liquidity_override)}
        limitations.append({'kind': 'synthetic_liquidity', 'detail': f'参考成交额被显式设为 {liquidity_override}，仅用于合成测试'})

    meta = inst.drop_duplicates('instrument', keep = 'last').set_index('instrument')
    list_date = meta['list_date'].map(_date).to_dict() if 'list_date' in meta else {}
    delist = {i: d for i, d in (meta['delist_date'].map(_date).to_dict() if 'delist_date' in meta else {}).items() if d is not None}
    kind = meta['kind'].to_dict(); board = meta['board'].to_dict() if 'board' in meta else {}
    allowed = set(boards)

    in_range = set(dates); market = {d: {} for d in dates}; bad, no_ref = [], 0
    for r in b[b.date.isin(in_range)].itertuples(index = False):
        d, i = r.date, r.instrument
        avg = liquidity_override if ref is None else ref.get((d, i))
        if avg is None: no_ref += 1
        q = {'board': board.get(i) or getattr(r, 'board', None) or board_of(i), 'is_st': flag(getattr(r, 'is_st', False)), 'preclose': _num(r.preclose), 'avg_amount_20d': avg}
        if not flag(r.is_trading):
            q.update(suspended = True, open = None, close = None)
        else:
            o, c = _num(r.open), _num(r.close)
            q.update(suspended = False, open = o if o and o > 0 else None, close = c if c and c > 0 else None)
            if q['open'] is None or q['close'] is None: bad.append((d, i))
        market[d][i] = q
    if bad: limitations.append({'kind': 'invalid_quote', 'detail': f'{len(bad)} 条交易中的日线缺少有效开盘价或收盘价', 'rows': [[str(d), i] for d, i in bad[:50]]})
    seen = set(b.instrument)
    for i, gone in delist.items():                       # 退市后没有日线：持仓不能静默消失，交给账本按退市持仓阻断
        if i not in seen: continue
        for d in dates:
            if d >= gone and i not in market[d]: market[d][i] = {'delisted': True, 'suspended': True, 'board': board.get(i) or board_of(i), 'is_st': False}

    candidates = {}                                     # 研究候选 = 当时在市、启用板块的股票；停牌、涨跌停是下单时的可成交检查，不在这里过滤
    invalid = set(bad)
    for d in dates:
        pick = set()
        for i, q in market[d].items():
            if q.get('delisted') or (d, i) in invalid: continue
            if kind.get(i) != 'stock' or q['board'] not in allowed: continue
            ld = list_date.get(i); gone = delist.get(i)
            if ld is None or ld > d or (gone is not None and d >= gone): continue
            pick.add(i)
        candidates[d] = pick

    acts, by_inst = {}, {}
    ca = tables.get('corp_actions')
    for r in (ca.to_dict('records') if ca is not None and len(ca) else []):
        a = _action(r)
        if a['ex_date'] is None:
            limitations.append({'kind': 'action_date_missing', 'detail': f"{a['instrument']} 公司行动缺少除权除息日"}); continue
        by_inst.setdefault(a['instrument'], []).append(a['ex_date'])
        if not dates[0] <= a['ex_date'] <= dates[-1]: continue
        if a['ex_date'] not in on_cal:
            limitations.append({'kind': 'action_on_closed_day', 'detail': f"{a['instrument']} 除权除息日 {a['ex_date']} 不是交易日"}); continue
        for k in ('pay_date', 'bonus_list_date'):
            if a[k] is not None and a[k] < a['ex_date']: raise InputBlocked([{'kind': 'action_dates_invalid', 'detail': f"{a['instrument']} {k} {a[k]} 早于除权除息日 {a['ex_date']}"}])
        acts.setdefault(a['ex_date'], []).append(a)

    unexplained = set()
    t = b[b.is_trading.map(flag)].sort_values(['instrument', 'date'])
    t = t.assign(prev_close = t.groupby('instrument').close.shift(1), prev_date = t.groupby('instrument').date.shift(1))
    for r in t[t.date.isin(in_range)].itertuples(index = False):
        pre, last = _num(r.preclose), _num(r.prev_close)
        if pre is None or last is None or abs(pre - last) <= PRICE_TOL: continue
        if not any(r.prev_date < x <= r.date for x in by_inst.get(r.instrument, ())): unexplained.add((r.date, r.instrument))

    info = {'start': str(dates[0]), 'end': str(dates[-1]), 'sessions': len(dates), 'warmup_start': str(warmup_start), 'boards': sorted(allowed),
            'liquidity': liquidity, 'quotes': sum(len(x) for x in market.values()), 'quotes_without_liquidity_reference': no_ref,
            'unexplained_price_resets': len(unexplained), 'actions': sum(len(x) for x in acts.values())}
    return ExecutionInput(dates, cal, market, candidates, acts, unexplained, limitations, info)
