"""执行规则集：按生效日期查买入单位、涨跌幅、报价单位与费率。查不到就报错，不套用默认值。"""
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path


@dataclass(frozen = True)
class Rule:
    start: date
    buy_unit: int = 100
    price_tick: float = 0.01
    limit_pct: float = 0.10
    stamp_tax: float = 0.0005          # 仅卖出
    transfer_fee: float = 0.00001
    commission_rate: float = 0.00025
    min_commission: float = 5.0


def _round(x, tick = 0.01, mode = ROUND_HALF_UP):
    t = Decimal(str(tick)); return float((Decimal(str(round(x, 8))) / t).to_integral_value(mode) * t)   # 先抹掉浮点噪声：10×1.01 不应向上取成 10.11


class RuleSet:
    def __init__(self, rules):
        self.rules = sorted(rules, key = lambda r: r.start)

    @classmethod
    def from_yaml(cls, path):
        import yaml
        items = yaml.safe_load(Path(path).read_text(encoding = 'utf-8'))['rules']
        return cls([Rule(**{**x, 'start': x['start'] if isinstance(x['start'], date) else date.fromisoformat(x['start'])}) for x in items])

    def scaled(self, **mult):  # 成本情景：按倍数调整费率，例如 commission_rate = 2
        return RuleSet([Rule(**{**r.__dict__, **{k: getattr(r, k) * v for k, v in mult.items()}}) for r in self.rules])

    def on(self, day):
        hit = [r for r in self.rules if r.start <= day]
        if not hit: raise ValueError(f'no execution rule for {day}')
        return hit[-1]

    def fill_price(self, raw, side, slippage, day):
        tick = self.on(day).price_tick; x = raw * (1 + slippage) if side == 'buy' else raw * (1 - slippage)
        return _round(x, tick, ROUND_CEILING if side == 'buy' else ROUND_FLOOR)

    def limit_prices(self, preclose, day):
        r = self.on(day); return _round(preclose * (1 - r.limit_pct), r.price_tick), _round(preclose * (1 + r.limit_pct), r.price_tick)

    def fees(self, amount, side, day):
        r = self.on(day)
        commission, stamp, transfer = _round(max(amount * r.commission_rate, r.min_commission)), _round(amount * r.stamp_tax) if side == 'sell' else 0.0, _round(amount * r.transfer_fee)
        return {'commission': commission, 'stamp_tax': stamp, 'transfer_fee': transfer, 'fee': round(commission + stamp + transfer, 2)}
