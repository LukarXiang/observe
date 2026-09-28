from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class Position:
    qty: int = 0
    today_buy: int = 0
    pending: int = 0
    cost: float = 0.0
    last_price: float | None = None
    stale_price: bool = False
    @property
    def sellable(self):
        return self.qty - self.today_buy - self.pending

@dataclass
class Book:
    cash: float
    positions: dict = field(default_factory=dict)
    receivable: dict = field(default_factory=lambda: defaultdict(float))
    fees_paid: float = 0.0
    fills: list = field(default_factory=list)
    equity_rows: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    listing_dates: dict = field(default_factory=dict)

    @staticmethod
    def infer_missing_date(ex_date, trading_dates, before_2014_days = 10):
        dates = sorted(trading_dates or [])
        if ex_date >= date(2014, 1, 1): return ex_date
        following = [item for item in dates if item > ex_date]
        return following[before_2014_days - 1] if len(following) >= before_2014_days else ex_date + timedelta(days = before_2014_days)

    def position(self, instrument):
        return self.positions.setdefault(instrument, Position())

    def start_day(self, on, actions = None, quotes = None):
        quotes = quotes or {}
        for p in self.positions.values():
            p.today_buy = 0
        for instrument, listing_date in list(self.listing_dates.items()):
            if listing_date == on:
                self.position(instrument).pending = 0; del self.listing_dates[instrument]
        for a in actions or []:
            p = self.position(a["instrument"])
            if a.get("ex_date") == on:
                dividend = p.qty * a.get("cash_per_share", 0.0)
                pay_date = a.get("pay_date")
                if pay_date is None:
                    pay_date = a.get("assumed_pay_date") or self.infer_missing_date(on, a.get("trading_dates", [])); self.assumptions.append({"instrument": a["instrument"], "date": on, "field": "pay_date", "assumed": True, "value": pay_date})
                if dividend: self.receivable[pay_date] += dividend
                ratio = a.get("bonus_ratio", 0.0)
                if ratio:
                    added = int(p.qty * ratio); p.qty += added; p.pending += added
                    listing_date = a.get("bonus_list_date")
                    if listing_date is None:
                        following = [item for item in sorted(a.get("trading_dates", [])) if item > on]; listing_date = following[0] if following else on + timedelta(days = 1)
                        self.assumptions.append({"instrument": a["instrument"], "date": on, "field": "bonus_list_date", "assumed": True, "value": listing_date})
                    self.listing_dates[a["instrument"]] = listing_date
                q = quotes.get(a["instrument"], {})
                if q.get("suspended") and p.last_price is not None:
                    cash = a.get("cash_per_share", 0.0); rights = a.get("rights_price", 0.0) * a.get("rights_ratio", 0.0)
                    p.last_price = (p.last_price - cash + rights) / (1 + a.get("bonus_ratio", 0.0) + a.get("rights_ratio", 0.0)); p.stale_price = True
            if a.get("bonus_list_date") == on:
                p.today_buy = max(0, p.today_buy - p.pending); p.pending = 0
        amount = self.receivable.pop(on, 0.0)
        self.cash = round(self.cash + amount, 2)
        return amount

    def execute(self, order, quote, on, rules, slippage = 0.0, open_cash = None):
        side, instrument = order["side"], order["instrument"]
        p = self.position(instrument); raw = quote.get("open")
        if quote.get("suspended") or raw is None or raw <= 0: return {**order, "qty_filled": 0, "status": "rejected", "reject_reason": "suspended" if quote.get("suspended") else "no_open_price"}
        preclose = quote.get("preclose")
        if preclose is not None:
            down, up = rules.limit_prices(preclose, on)
            if side == "buy" and raw >= up: return {**order, "qty_filled": 0, "status": "rejected", "reject_reason": "limit_up"}
            if side == "sell" and raw <= down: return {**order, "qty_filled": 0, "status": "rejected", "reject_reason": "limit_down"}
        if side == "sell":
            qty = p.sellable if order.get("qty") in (None, "all") else min(p.sellable, int(order["qty"]))
        else:
            target = order.get("amount", 0.0); r = rules.for_date(on); qty = int(target / rules.fill_price(raw, "buy", slippage, on) // r.buy_unit * r.buy_unit)
            available = self.cash if open_cash is None else open_cash
            price = rules.fill_price(raw, "buy", slippage, on)
            participation = order.get("participation_limit", quote.get("participation_limit"))
            if participation is not None and quote.get("avg_amount") is not None:
                qty = min(qty, int(quote["avg_amount"] * participation / price // r.buy_unit * r.buy_unit))
            while qty and available < price * qty + rules.fees(price * qty, "buy", on)["total"]: qty -= r.buy_unit
        if not qty: return {**order, "qty_filled": 0, "status": "rejected", "reject_reason": "cash" if side == "buy" else "not_sellable"}
        price = rules.fill_price(raw, side, slippage, on); amount = round(price * qty, 2); fees = rules.fees(amount, side, on)
        if side == "buy": self.cash = round(self.cash - amount - fees["total"], 2); p.qty += qty; p.today_buy += qty
        else: self.cash = round(self.cash + amount - fees["total"], 2); p.qty -= qty
        p.cost = price; self.fees_paid += fees["total"]
        planned = order.get("qty") if side == "sell" else order.get("amount")
        status = "partially_filled" if side == "sell" and planned not in (None, "all") and qty < int(planned) else "filled"
        fill = {**order, "qty_filled": qty, "fill_price": price, "amount": amount, **fees, "status": status}; self.fills.append(fill)
        return fill

    def mark_to_market(self, on, quotes):
        market = 0.0; stale = False
        for instrument, p in self.positions.items():
            q = quotes.get(instrument, {}); close = q.get("close")
            if close is None:
                close = p.last_price; p.stale_price = True
            else: p.last_price = close; p.stale_price = bool(q.get("stale_price", False))
            if close is not None: market += p.qty * close
            stale = stale or p.stale_price
        receivable = sum(self.receivable.values()); equity = round(self.cash + market + receivable, 2)
        previous = self.equity_rows[-1]["equity"] if self.equity_rows else None
        row = {"date": on, "cash": self.cash, "market_value": round(market, 2), "receivable": round(receivable, 2), "equity": equity, "daily_return": None if previous is None else equity / previous - 1, "stale_price": stale}
        self.equity_rows.append(row); return row
