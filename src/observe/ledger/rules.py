from dataclasses import dataclass
from datetime import date
from math import ceil, floor


@dataclass(frozen=True)
class Rule:
    start: date = date.min
    buy_unit: int = 100
    price_tick: float = 0.01
    limit_pct: float = 0.10
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    commission_rate: float = 0.00025
    min_commission: float = 5.0

class RuleSet:
    def __init__(self, rules = None):
        self.rules = sorted(rules or [Rule()], key = lambda r: r.start)

    def for_date(self, on):
        matches = [r for r in self.rules if r.start <= on]
        if not matches:
            raise ValueError(f"no execution rule for {on}")
        return matches[-1]

    def fill_price(self, raw, side, slippage, on):
        r = self.for_date(on)
        value = raw * (1 + slippage if side == "buy" else 1 - slippage)
        return (ceil(value / r.price_tick) if side == "buy" else floor(value / r.price_tick)) * r.price_tick

    def limit_prices(self, preclose, on):
        r = self.for_date(on)
        return (round(preclose * (1 - r.limit_pct), 2), round(preclose * (1 + r.limit_pct), 2))

    def fees(self, amount, side, on):
        r = self.for_date(on)
        commission = max(amount * r.commission_rate, r.min_commission)
        stamp = amount * r.stamp_tax if side == "sell" else 0.0
        transfer = amount * r.transfer_fee
        return {"commission": round(commission, 2), "stamp_tax": round(stamp, 2), "transfer_fee": round(transfer, 2), "total": round(commission + stamp + transfer, 2)}
