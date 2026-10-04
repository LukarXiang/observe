"""资金账本：持仓（总股数 / 今日买入 / 待上市股份）、现金、应收分红、现金流水与每日勾稽。"""
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import math


@dataclass
class Position:
    qty: int = 0
    today_buy: int = 0                  # T+1：今天买入的不可卖
    pending: int = 0                    # 送转新增、尚未到红股上市日的不可卖
    cost: float = 0.0                   # 加权平均成本
    last_price: float | None = None
    stale: bool = False

    @property
    def sellable(self): return self.qty - self.today_buy - self.pending


class LedgerError(RuntimeError): pass


class Book:
    PAY_SPLIT, PAY_LAG, LIST_LAG = date(2014, 1, 1), 10, 1   # 到账日 / 红股上市日缺失时的从严推断（决策 18）

    def __init__(self, cash, calendar = ()):
        self.initial = self.cash = round(float(cash), 2)
        if not math.isfinite(self.initial) or self.initial < 0: raise LedgerError('initial cash must be finite and non-negative')
        self.calendar = sorted(calendar)
        self.positions, self.receivable, self.listing = {}, defaultdict(float), defaultdict(list)   # listing: {上市日: [(证券, 股数)]}
        self.cash_events, self.fills, self.equity_rows, self.assumptions, self.issues = [], [], [], [], []
        self._order_seq = 0; self._fill_seq = 0

    # 工具 -----------------------------------------------------------------
    def pos(self, i): return self.positions.setdefault(i, Position())

    def _after(self, day, n):
        k = bisect_right(self.calendar, day) + n - 1
        if k >= len(self.calendar): raise LedgerError(f'calendar too short to infer {n} sessions after {day}')
        return self.calendar[k]

    def _cash(self, day, kind, amount, **ref):
        if not isinstance(amount, (int, float)) or not math.isfinite(amount): raise LedgerError(f'invalid cash amount on {day}: {amount}')
        self.cash = round(self.cash + amount, 2); self.cash_events.append({'date': day, 'kind': kind, 'amount': round(amount, 2), **ref})
        if self.cash < -0.005: raise LedgerError(f'negative cash {self.cash} on {day}: {kind} {ref}')

    # 盘前 -----------------------------------------------------------------
    def start_day(self, day, actions = (), quotes = None):
        quotes = quotes or {}
        for p in self.positions.values(): p.today_buy = 0
        for d in [d for d in self.listing if d <= day]:
            for i, q in self.listing.pop(d): self.pos(i).pending -= q
        for a in actions:
            if a['ex_date'] != day: continue
            if a.get('convert_to'): self._convert(day, a)
            else: self._ex_date(day, a, quotes.get(a['instrument'], {}))
        for d in [d for d in self.receivable if d <= day]:
            self._cash(day, 'dividend_paid', self.receivable.pop(d))

    def _ex_date(self, day, a, quote):
        i = a['instrument']; p = self.pos(i)
        if not p.qty: return
        cash, bonus = a.get('cash_per_share', 0.0), a.get('bonus_ratio', 0.0); rights_price, rights_ratio = a.get('rights_price', 0.0), a.get('rights_ratio', 0.0)
        if cash:
            pay = a.get('pay_date')
            if pay is None:
                pay = day if day >= self.PAY_SPLIT else self._after(day, self.PAY_LAG)
                self.assumptions.append({'date': day, 'instrument': i, 'field': 'pay_date', 'value': pay})
            self.receivable[pay] += round(p.qty * cash, 2)
        if bonus:
            added = int(p.qty * bonus); listed = a.get('bonus_list_date')
            if listed is None:
                listed = self._after(day, self.LIST_LAG); self.assumptions.append({'date': day, 'instrument': i, 'field': 'bonus_list_date', 'value': listed})
            old_qty = p.qty; p.qty += added; p.cost = p.cost * old_qty / p.qty
            if listed > day: p.pending += added; self.listing[listed].append((i, added))
        if quote.get('suspended') and p.last_price is not None:   # 停牌期间除权：估值价按除权参考价调整（与是否参与配股无关）
            ref = a.get('ref_price')
            p.last_price = ref if ref is not None else (p.last_price - cash + rights_price * rights_ratio) / (1 + bonus + rights_ratio); p.stale = True

    def _convert(self, day, a):
        """吸收合并换股：旧代码持仓按比例转为新代码，成本整体转移；不足 1 股的零头舍去并记为假设（实际多以现金补偿）"""
        old = self.pos(a['instrument'])
        if not old.qty: return
        new = self.pos(a['convert_to']); r = a['convert_ratio']; q = int(old.qty * r)
        if old.qty * r - q > 1e-9: self.assumptions.append({'date': day, 'instrument': a['instrument'], 'field': 'conversion_fraction', 'value': old.qty * r - q})
        total_cost = old.cost * old.qty; new.cost = (new.cost * new.qty + total_cost) / (new.qty + q); new.qty += q
        if new.last_price is None and old.last_price is not None: new.last_price = old.last_price / r
        self.issues.append({'date': day, 'instrument': a['instrument'], 'kind': 'converted', 'to': a['convert_to'], 'qty': q})
        old.qty = old.today_buy = old.pending = 0

    # 开盘成交 -------------------------------------------------------------
    def execute(self, order, quote, day, rules, slippage = 0.0, budget = None):
        """order: {'instrument','side','amount'(买)|'qty':'all'(卖)}；budget 为 preopen_cash_only 下本事件可用现金"""
        side, i = order['side'], order['instrument']; p = self.pos(i); raw, pre = quote.get('open'), quote.get('preclose'); bd, st = quote.get('board', 'main'), bool(quote.get('is_st', False))
        self._order_seq += 1; oid = order.get('order_id', f'order-{self._order_seq}')
        def reject(why): return {**order, 'order_id': oid, 'qty_filled': 0, 'status': 'rejected', 'reject_reason': why}
        if quote.get('suspended'): return reject('suspended')
        if raw is None or not isinstance(raw, (int, float)) or raw <= 0 or raw != raw or raw in (float('inf'), float('-inf')): return reject('no_open_price')
        if pre is not None:
            down, up = rules.limit_prices(pre, day, bd, st)
            if side == 'buy' and raw >= up: return reject('limit_up')
            if side == 'sell' and raw <= down: return reject('limit_down')
        rule = rules.on(day, bd, st)
        price, unit = rules.fill_price(raw, side, slippage, day, bd, st), rule.buy_unit
        if not isinstance(price, (int, float)) or price <= 0 or price != price or price in (float('inf'), float('-inf')): return reject('invalid_fill_price')
        if pre is not None:
            down, up = rules.limit_prices(pre, day, bd, st)
            if not down <= price <= up: return reject('slippage_outside_limit')
        requested = None
        if side == 'sell':
            qty = p.sellable
            if qty <= 0: return reject('not_sellable')
            requested = qty
            if order.get('qty', 'all') != 'all':
                desired = order['qty']
                if type(desired) is not int or desired <= 0: return reject('invalid_qty')
                if desired < p.sellable and desired % unit: return reject('sell_unit')
                if rule.min_sell_qty is not None and rule.sell_quantity(min(desired, p.sellable), p.sellable) != min(desired, p.sellable): return reject('sell_unit')
                requested, qty = desired, min(desired, p.sellable)
            if rule.max_order_qty is not None: qty = min(qty, rule.max_order_qty)
            if 'participation' in order:
                cap = quote.get('avg_amount_20d'); limit = order['participation']
                if cap is None: return reject('no_liquidity_reference')
                cap_qty = rule.sell_quantity(cap * limit / price, p.sellable)
                qty = min(qty, cap_qty)
                if qty <= 0: return reject('participation_limit')
        else:
            qty = rule.buy_quantity(order['amount'] / price)
            requested = qty; cap = quote.get('avg_amount_20d'); limit = order.get('participation')
            if limit is not None:
                if cap is None: return reject('no_liquidity_reference')
                qty = min(qty, rule.buy_quantity(cap * limit / price))
            money = self.cash if budget is None else min(budget, self.cash)
            while qty and price * qty + rules.fees(price * qty, 'buy', day)['fee'] > money: qty -= unit
            if qty < (rule.min_buy_qty or unit): qty = 0
            if not qty: return reject('cash')
        value = round(price * qty, 2); f = rules.fees(value, side, day)
        if side == 'buy':
            p.cost = (p.cost * p.qty + value + f['fee']) / (p.qty + qty); p.qty += qty; p.today_buy += qty
            self._cash(day, 'buy', -value, instrument = i); self._cash(day, 'fee', -f['fee'], instrument = i)
        else:
            p.qty -= qty; self._cash(day, 'sell', value, instrument = i); self._cash(day, 'fee', -f['fee'], instrument = i)
        self._fill_seq += 1
        remaining = max(0, requested - qty) if requested is not None else 0
        fill = {**order, 'order_id': oid, 'fill_id': f'fill-{self._fill_seq}', 'date': day, 'qty_requested': requested, 'qty_filled': qty, 'remaining_qty': remaining, 'fill_price': price, 'value': value, **f, 'status': 'partial' if remaining else 'filled'}
        self.fills.append(fill); return fill

    # 收盘 -----------------------------------------------------------------
    def close_day(self, day, quotes):
        market, stale = 0.0, False
        for i, p in self.positions.items():
            if not p.qty: continue
            q = quotes.get(i, {})
            if q.get('delisted'): self.issues.append({'date': day, 'instrument': i, 'kind': 'delisted_holding', 'qty': p.qty})
            if q.get('suspended'): p.stale = p.last_price is not None   # 停牌行的占位收盘价不用于估值
            elif q.get('close') is None: raise LedgerError(f'no valuation price for {i} on {day}')
            else: p.last_price, p.stale = q['close'], False
            if p.last_price is None or not math.isfinite(float(p.last_price)): raise LedgerError(f'non-finite valuation price for {i} on {day}')
            market += p.qty * p.last_price; stale = stale or p.stale
        receivable = round(sum(self.receivable.values()), 2); equity = round(self.cash + market + receivable, 2)
        flow = round(self.initial + sum(e['amount'] for e in self.cash_events), 2)          # 现金流水核对
        if abs(flow - self.cash) > 0.01: raise LedgerError(f'cash {self.cash} != initial + events {flow} on {day}')
        prev = self.equity_rows[-1]['equity'] if self.equity_rows else self.initial
        if not math.isfinite(equity): raise LedgerError(f'non-finite equity on {day}')
        row = {'date': day, 'cash': self.cash, 'market_value': round(market, 2), 'receivable': receivable, 'equity': equity, 'daily_return': equity / prev - 1 if prev else None, 'stale_price': stale}
        self.equity_rows.append(row); return row

    @property
    def status(self): return 'blocked' if any(x['kind'] == 'delisted_holding' for x in self.issues) else 'success'

    def equity_curve(self): return [self.initial] + [r['equity'] for r in self.equity_rows]
