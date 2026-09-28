"""资金账本：持仓（总股数 / 今日买入 / 待上市股份）、现金、应收分红、现金流水与每日勾稽。"""
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date


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
        self.initial = self.cash = round(float(cash), 2); self.calendar = sorted(calendar)
        self.positions, self.receivable, self.listing = {}, defaultdict(float), defaultdict(list)   # listing: {上市日: [(证券, 股数)]}
        self.cash_events, self.fills, self.equity_rows, self.assumptions, self.issues = [], [], [], [], []

    # 工具 -----------------------------------------------------------------
    def pos(self, i): return self.positions.setdefault(i, Position())

    def _after(self, day, n):
        k = bisect_right(self.calendar, day) + n - 1
        if k >= len(self.calendar): raise LedgerError(f'calendar too short to infer {n} sessions after {day}')
        return self.calendar[k]

    def _cash(self, day, kind, amount, **ref):
        self.cash = round(self.cash + amount, 2); self.cash_events.append({'date': day, 'kind': kind, 'amount': round(amount, 2), **ref})
        if self.cash < -0.005: raise LedgerError(f'negative cash {self.cash} on {day}: {kind} {ref}')

    # 盘前 -----------------------------------------------------------------
    def start_day(self, day, actions = (), quotes = None):
        quotes = quotes or {}
        for p in self.positions.values(): p.today_buy = 0
        for d in [d for d in self.listing if d <= day]:
            for i, q in self.listing.pop(d): self.pos(i).pending -= q
        for a in actions:
            if a['ex_date'] == day: self._ex_date(day, a, quotes.get(a['instrument'], {}))
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
            p.qty += added; p.pending += added; p.cost = p.cost * (p.qty - added) / p.qty; self.listing[listed].append((i, added))
        if quote.get('suspended') and p.last_price is not None:   # 停牌期间除权：估值价按除权参考价调整（与是否参与配股无关）
            ref = a.get('ref_price')
            p.last_price = ref if ref is not None else (p.last_price - cash + rights_price * rights_ratio) / (1 + bonus + rights_ratio); p.stale = True

    # 开盘成交 -------------------------------------------------------------
    def execute(self, order, quote, day, rules, slippage = 0.0, budget = None):
        """order: {'instrument','side','amount'(买)|'qty':'all'(卖)}；budget 为 preopen_cash_only 下本事件可用现金"""
        side, i = order['side'], order['instrument']; p = self.pos(i); raw, pre = quote.get('open'), quote.get('preclose')
        def reject(why): return {**order, 'qty_filled': 0, 'status': 'rejected', 'reject_reason': why}
        if quote.get('suspended'): return reject('suspended')
        if raw is None or raw <= 0: return reject('no_open_price')
        if pre is not None:
            down, up = rules.limit_prices(pre, day)
            if side == 'buy' and raw >= up: return reject('limit_up')
            if side == 'sell' and raw <= down: return reject('limit_down')
        price, unit = rules.fill_price(raw, side, slippage, day), rules.on(day).buy_unit
        if side == 'sell':
            qty = p.sellable
            if qty <= 0: return reject('not_sellable')
        else:
            qty = int(order['amount'] / price // unit * unit)
            cap = quote.get('avg_amount_20d'); limit = order.get('participation', 0.05)
            if cap is not None: qty = min(qty, int(cap * limit / price // unit * unit))
            money = self.cash if budget is None else min(budget, self.cash)
            while qty and price * qty + rules.fees(price * qty, 'buy', day)['fee'] > money: qty -= unit
            if not qty: return reject('cash')
        value = round(price * qty, 2); f = rules.fees(value, side, day)
        if side == 'buy':
            p.cost = (p.cost * p.qty + value + f['fee']) / (p.qty + qty); p.qty += qty; p.today_buy += qty
            self._cash(day, 'buy', -value, instrument = i); self._cash(day, 'fee', -f['fee'], instrument = i)
        else:
            p.qty -= qty; self._cash(day, 'sell', value, instrument = i); self._cash(day, 'fee', -f['fee'], instrument = i)
        fill = {**order, 'date': day, 'qty_filled': qty, 'fill_price': price, 'value': value, **f, 'status': 'filled'}   # amount 保留为买单的目标金额
        self.fills.append(fill); return fill

    # 收盘 -----------------------------------------------------------------
    def close_day(self, day, quotes):
        market, stale = 0.0, False
        for i, p in self.positions.items():
            if not p.qty: continue
            q = quotes.get(i, {})
            if q.get('delisted'): self.issues.append({'date': day, 'instrument': i, 'kind': 'delisted_holding', 'qty': p.qty})
            if q.get('suspended') or q.get('close') is None: p.stale = p.last_price is not None   # 停牌行的占位收盘价不用于估值
            else: p.last_price, p.stale = q['close'], False
            if p.last_price is None: raise LedgerError(f'no valuation price for {i} on {day}')
            market += p.qty * p.last_price; stale = stale or p.stale
        receivable = round(sum(self.receivable.values()), 2); equity = round(self.cash + market + receivable, 2)
        flow = round(self.initial + sum(e['amount'] for e in self.cash_events), 2)          # 现金流水核对
        if abs(flow - self.cash) > 0.01: raise LedgerError(f'cash {self.cash} != initial + events {flow} on {day}')
        prev = self.equity_rows[-1]['equity'] if self.equity_rows else self.initial
        row = {'date': day, 'cash': self.cash, 'market_value': round(market, 2), 'receivable': receivable, 'equity': equity, 'daily_return': equity / prev - 1 if prev else None, 'stale_price': stale}
        self.equity_rows.append(row); return row

    @property
    def status(self): return 'blocked' if any(x['kind'] == 'delisted_holding' for x in self.issues) else 'success'

    def equity_curve(self): return [self.initial] + [r['equity'] for r in self.equity_rows]
