"""资金账本：持仓（总股数 / 今日买入 / 待上市股份）、现金、应收分红、现金流水与每日勾稽。"""
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
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


def cash_entitlement_spec(action):
    """Validate opt-in cash-only record-close terms; legacy actions return None."""
    policy = action.get('cash_entitlement')
    if policy is None: return None
    if not isinstance(policy, str) or policy != 'record_close': raise LedgerError('unknown cash dividend entitlement policy')
    fields = ('instrument', 'record_date', 'ex_date', 'pay_date', 'cash_per_share')
    terms = {k: action.get(k) for k in fields}
    if not isinstance(terms['instrument'], str) or not terms['instrument']:
        raise LedgerError('cash dividend entitlement requires an instrument')
    if any(type(terms[k]) is not date for k in ('record_date', 'ex_date', 'pay_date')):
        raise LedgerError('cash dividend entitlement requires explicit record/ex/pay dates')
    if not terms['record_date'] < terms['ex_date'] <= terms['pay_date']:
        raise LedgerError('cash dividend entitlement dates must satisfy record < ex <= pay')
    cash = terms['cash_per_share']
    try: valid_cash = type(cash) in (int, float) and math.isfinite(cash) and cash > 0
    except OverflowError: valid_cash = False
    if not valid_cash:
        raise LedgerError('cash dividend entitlement requires a positive finite amount')
    if action.get('convert_to') or any(action.get(k, 0) for k in ('bonus_ratio', 'rights_ratio', 'rights_price')):
        raise LedgerError('cash dividend entitlement cannot combine conversion, bonus or rights')
    return terms


class Book:
    PAY_SPLIT, PAY_LAG, LIST_LAG = date(2014, 1, 1), 10, 1   # 到账日 / 红股上市日缺失时的从严推断（决策 18）

    def __init__(self, cash, calendar = (), *, external_flows = False):
        if type(external_flows) is not bool: raise LedgerError('external_flows must be boolean')
        if external_flows: cash = self._flow_amount(cash, allow_zero = True)
        self.initial = self.cash = round(float(cash), 2)
        if not math.isfinite(self.initial) or self.initial < 0: raise LedgerError('initial cash must be finite and non-negative')
        self.calendar = sorted(calendar)
        self.positions, self.receivable, self.listing = {}, defaultdict(float), defaultdict(list)   # listing: {上市日: [(证券, 股数)]}
        self.cash_events, self.fills, self.equity_rows, self.assumptions, self.issues = [], [], [], [], []
        self._order_seq = 0; self._fill_seq = 0
        self.dividend_entitlements = {}
        self._record_due, self._record_day = [], None
        self.external_flows = external_flows
        self.external_flow_events = []
        self._flow_day, self._flow_last_day = None, None
        self._flow_ids = set()
        self._units = Decimal(str(self.initial))
        self._unit_nav = Decimal(1)
        self._external_total = 0.0
        self._external_day_total = 0.0

    @staticmethod
    def _flow_amount(amount, *, allow_zero = False):
        try:
            valid = type(amount) in (int, float) and math.isfinite(amount) and abs(amount) <= 1e12
        except OverflowError: valid = False
        if not valid: raise LedgerError('external amount must be finite numeric within 1e12')
        amount = round(float(amount), 2)
        if not allow_zero and not amount: raise LedgerError('external amount rounds to zero')
        return amount

    def _flow_active(self, day):
        if not self.external_flows: raise LedgerError('external cash flows not enabled')
        if type(day) is not date or day != self._flow_day:
            raise LedgerError('external flow requires the active unclosed day')

    def _flow_valuation(self, quotes, price_field):
        """Value immediately before a flow without changing marks or settlement state."""
        values = []
        for i, p in self.positions.items():
            if not p.qty: continue
            q = quotes.get(i, {})
            price = p.last_price if q.get('suspended') else q.get(price_field)
            try:
                valid = type(p.qty) is int and p.qty > 0 and type(price) in (int, float) and \
                    math.isfinite(price) and price > 0 and math.isfinite(p.qty * price)
            except OverflowError: valid = False
            if not valid: raise LedgerError(f'invalid external-flow valuation for {i}')
            if q.get('delisted'): raise LedgerError(f'delisted external-flow holding {i}')
            values.append(p.qty * price)
        try:
            market = sum(values)
            receivable = round(sum(self.receivable.values()), 2)
            equity = round(self.cash + market + receivable, 2)
        except (OverflowError, ValueError) as exc: raise LedgerError('invalid external-flow equity') from exc
        if not all(math.isfinite(v) and v >= 0 for v in (self.cash, market, receivable, equity)) or equity > 1e12:
            raise LedgerError('external-flow equity must be finite, non-negative and within 1e12')
        return market, receivable, equity

    def _flow_nav(self, equity):
        if self._units == 0:
            if equity != 0: raise LedgerError('equity without external-flow units')
            return self._unit_nav
        nav = Decimal(str(equity)) / self._units
        if not nav.is_finite() or not math.isfinite(float(nav)):
            raise LedgerError('non-finite unit NAV')
        return nav

    def external_cash_flow(self, day, amount, quotes, *, event_id, price_field = 'open'):
        """Apply a same-day external deposit/withdrawal using pre-flow unit NAV.

        Amount/equity are limited to 1e12 to retain cent precision in the cash ledger.
        The caller supplies the observed open/close quotes, never a portfolio total.
        """
        self._flow_active(day)
        if not isinstance(event_id, str) or not event_id.strip() or event_id in self._flow_ids:
            raise LedgerError('external event_id must be non-empty and unique')
        if price_field not in ('open', 'close'): raise LedgerError('external valuation must use open or close')
        amount = self._flow_amount(amount)
        market, receivable, equity = self._flow_valuation(quotes, price_field)
        nav = self._flow_nav(equity)
        if nav <= 0 or float(nav) <= 0: raise LedgerError('external flow requires positive unit NAV')
        cash_after, equity_after = round(self.cash + amount, 2), round(equity + amount, 2)
        if cash_after < 0: raise LedgerError('external withdrawal exceeds available cash')
        if not 0 <= equity_after <= 1e12 or not 0 <= cash_after <= 1e12:
            raise LedgerError('external flow exceeds supported equity/cash range')
        units = self._units + Decimal(str(amount)) / nav
        # Exact full redemption must not leave decimal division residue.
        if equity_after == 0: units = Decimal(0)
        if not units.is_finite() or units < 0 or not math.isfinite(float(units)) or (equity_after > 0 and float(units) <= 0):
            raise LedgerError('invalid external-flow units')
        event = {'date': day, 'event_id': event_id, 'amount': amount, 'price_field': price_field,
            'market_value': round(market, 2), 'receivable': receivable, 'equity_before': equity,
            'equity_after': equity_after, 'unit_nav': float(nav), 'units_before': float(self._units), 'units_after': float(units)}
        self._cash(day, 'external_cash_flow', amount, event_id = event_id)
        self._units, self._unit_nav = units, nav
        self._external_total = round(self._external_total + amount, 2)
        self._external_day_total = round(self._external_day_total + amount, 2)
        self._flow_ids.add(event_id)
        self.external_flow_events.append(event)
        return dict(event)

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
        if self.external_flows:
            if type(day) is not date or self._flow_day is not None or (self._flow_last_day is not None and day <= self._flow_last_day):
                raise LedgerError('external-flow days must start after the previous close')
        actions = list(actions)
        if self._record_due: raise LedgerError('previous dividend record day has not closed')
        due, seen = [], set()
        # Validate explicit rights before touching cash, positions or settlement queues.
        for a in actions:
            terms = cash_entitlement_spec(a)
            if terms is None: continue
            key = (terms['instrument'], terms['ex_date'])
            if key in seen: raise LedgerError('duplicate cash dividend entitlement action')
            seen.add(key)
            if terms['record_date'] == day:
                if key in self.dividend_entitlements: raise LedgerError('duplicate dividend record close')
                due.append((key, terms))
            if terms['ex_date'] == day:
                right = self.dividend_entitlements.get(key)
                if right is None: raise LedgerError('dividend record-close evidence missing')
                if right['terms'] != terms: raise LedgerError('cash dividend entitlement terms changed')
                if right['applied']: raise LedgerError('cash dividend entitlement already applied')
        self._record_due, self._record_day = due, day if due else None
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
        if self.external_flows:
            self._flow_day = day
            self._external_day_total = 0.0

    def _ex_date(self, day, a, quote):
        i = a['instrument']; p = self.pos(i)
        cash, bonus = a.get('cash_per_share', 0.0), a.get('bonus_ratio', 0.0); rights_price, rights_ratio = a.get('rights_price', 0.0), a.get('rights_ratio', 0.0)
        right = self.dividend_entitlements.get((i, day)) if a.get('cash_entitlement') == 'record_close' else None
        entitled_qty = right['qty'] if right is not None else p.qty
        if not p.qty and right is None: return
        if cash:
            pay = a.get('pay_date')
            if pay is None:
                pay = day if day >= self.PAY_SPLIT else self._after(day, self.PAY_LAG)
                self.assumptions.append({'date': day, 'instrument': i, 'field': 'pay_date', 'value': pay})
            amount = round(entitled_qty * cash, 2)
            if amount or right is None: self.receivable[pay] += amount
        if right is not None: right['applied'] = True
        if not p.qty: return
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
        """Convert holdings in place or into another code, retaining costs and locked shares."""
        old = self.pos(a['instrument'])
        if not old.qty: return
        rounding = a.get('convert_rounding', 'floor')
        if rounding not in ('floor', 'ceil'): raise LedgerError('unknown conversion rounding')
        try: ratio = Decimal(str(a['convert_ratio']))
        except (KeyError, InvalidOperation, ValueError) as exc: raise LedgerError('invalid conversion ratio') from exc
        if not ratio.is_finite() or ratio <= 0 or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            raise LedgerError('invalid conversion ratio')
        round_mode = ROUND_CEILING if rounding == 'ceil' else ROUND_FLOOR
        quantity = lambda value: int((Decimal(value) * ratio).to_integral_value(rounding = round_mode))
        exact = Decimal(old.qty) * ratio; q = quantity(old.qty)
        if q <= 0: raise LedgerError('conversion would remove the entire holding as a fraction')
        queued = [(d, k, count) for d in sorted(self.listing) for k, (i, count) in enumerate(self.listing[d]) if i == a['instrument']]
        if sum(count for _, _, count in queued) != old.pending:
            raise LedgerError('conversion pending shares lack listing evidence')
        pending = quantity(old.pending); today_buy = quantity(old.today_buy)
        if pending + today_buy > q: raise LedgerError('conversion locks exceed converted holding')
        counts = [int((Decimal(count) * ratio).to_integral_value(rounding = ROUND_FLOOR)) for _, _, count in queued]
        # Allocate rounding residue to the last listing date, keeping it locked.
        if counts: counts[-1] += pending - sum(counts)
        new = old if a['convert_to'] == a['instrument'] else self.positions.get(a['convert_to'], Position())
        # Calculate all float-valued results before committing positions or queues.
        try:
            total_cost = old.cost * old.qty
            new_qty = q if new is old else new.qty + q
            new_cost = (total_cost if new is old else new.cost * new.qty + total_cost) / new_qty
            new_price = new.last_price
            if new is old or new_price is None:
                new_price = old.last_price / float(ratio) if old.last_price is not None else None
            fraction = float(exact - q)
        except (OverflowError, ZeroDivisionError) as exc:
            raise LedgerError('conversion results cannot be represented') from exc
        if not math.isfinite(new_cost) or not math.isfinite(fraction) or (new_price is not None and (not math.isfinite(new_price) or new_price <= 0)):
            raise LedgerError('conversion results must be finite with positive valuation price')
        if new is old:
            new.qty, new.pending, new.today_buy = q, pending, today_buy
        else:
            self.positions[a['convert_to']] = new
            new.qty = new_qty
            new.pending += pending; new.today_buy += today_buy
            old.qty = old.today_buy = old.pending = 0
        new.cost, new.last_price = new_cost, new_price
        for (d, k, _), count in zip(queued, counts, strict = True): self.listing[d][k] = (a['convert_to'], count)
        if abs(fraction) > 1e-9:
            entry = {'date': day, 'instrument': a['instrument'], 'field': 'conversion_fraction', 'value': fraction}
            if 'convert_rounding' in a: entry['rounding'] = rounding
            self.assumptions.append(entry)
        self.issues.append({'date': day, 'instrument': a['instrument'], 'kind': 'converted', 'to': a['convert_to'], 'qty': q})

    def settle_delisted(self, day, i):
        """退市持仓按最后估值价折成现金并记为假设。只由策略回测显式调用；研究回放仍按退市持仓阻断（status = blocked）"""
        p = self.pos(i); px = p.last_price or 0.0; value = round(p.qty * px, 2)
        self._cash(day, 'delisted_settlement', value, instrument = i)
        self.assumptions.append({'date': day, 'instrument': i, 'field': 'delisted_settlement', 'qty': p.qty, 'price': px, 'value': value})
        p.qty = p.today_buy = p.pending = 0

    # 开盘成交 -------------------------------------------------------------
    def execute(self, order, quote, day, rules, slippage = 0.0, budget = None):
        """order: {'instrument','side','amount'(买)|'qty':'all'(卖)}；budget 为 preopen_cash_only 下本事件可用现金"""
        if self.external_flows: self._flow_active(day)
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
        if self.external_flows:
            self._flow_active(day)
            _, _, flow_equity = self._flow_valuation(quotes, 'close')
            unit_nav = self._flow_nav(flow_equity)
        if self._record_due and day != self._record_day: raise LedgerError('dividend record-close date differs from start day')
        rights = {}
        for key, terms in self._record_due:
            p = self.positions.get(terms['instrument']); qty = p.qty if p is not None else 0
            try: valid_qty = type(qty) is int and qty >= 0 and math.isfinite(qty * terms['cash_per_share'])
            except OverflowError: valid_qty = False
            if not valid_qty:
                raise LedgerError('invalid dividend record-close quantity or amount')
            rights[key] = {'terms': terms, 'qty': qty, 'applied': False}
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
        if self.external_flows:
            prev_nav = self.equity_rows[-1]['unit_nav'] if self.equity_rows else 1.0
            row.update(units = float(self._units), unit_nav = float(unit_nav),
                external_flow = self._external_day_total,
                external_flow_total = self._external_total, daily_return = float(unit_nav) / prev_nav - 1 if prev_nav else None)
            self._unit_nav = unit_nav
            self._flow_day, self._flow_last_day = None, day
        self.equity_rows.append(row)
        self.dividend_entitlements.update(rights)
        self._record_due, self._record_day = [], None
        return row

    @property
    def status(self): return 'blocked' if any(x['kind'] == 'delisted_holding' for x in self.issues) else 'success'

    def equity_curve(self):
        if self.external_flows: return [1.0] + [r['unit_nav'] for r in self.equity_rows]
        return [self.initial] + [r['equity'] for r in self.equity_rows]
