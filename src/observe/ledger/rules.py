"""执行规则集：费用按日期查，交易规则（涨跌幅、买入单位、报价单位）按「板块 × 是否 ST × 日期」查。查不到就报错，不套用默认值。"""
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path


@dataclass(frozen = True)
class Rule:
    """费用规则（全市场），也携带默认交易参数；没有配置交易规则表时直接使用这里的交易参数"""
    start: date
    buy_unit: int = 100
    price_tick: float = 0.01
    limit_pct: float = 0.10
    stamp_tax: float = 0.0005
    stamp_both_sides: bool = False     # 2008-09-19 之前印花税买卖双边征收
    transfer_fee: float = 0.00001
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    verified: bool = True
    source: str = ''


@dataclass(frozen = True)
class TradingRule:
    start: date
    board: str                          # main / gem / star / bse
    st: bool
    limit_pct: float
    buy_unit: int = 100
    price_tick: float = 0.01
    verified: bool = True
    source: str = ''


def _round(x, tick = 0.01, mode = ROUND_HALF_UP):
    t = Decimal(str(tick)); return float((Decimal(str(round(x, 8))) / t).to_integral_value(mode) * t)   # 先抹掉浮点噪声：10×1.01 不应向上取成 10.11


def _d(x): return x if isinstance(x, date) else date.fromisoformat(str(x))


class RuleSet:
    def __init__(self, rules, trading = None):
        self.rules = sorted(rules, key = lambda r: r.start); self.trading = sorted(trading or [], key = lambda r: r.start); self.used_unverified = set()

    @classmethod
    def from_yaml(cls, path):
        import yaml
        y = yaml.safe_load(Path(path).read_text(encoding = 'utf-8'))
        fees = [Rule(**{**x, 'start': _d(x['start'])}) for x in y.get('fees', y.get('rules', []))]
        return cls(fees, [TradingRule(**{**x, 'start': _d(x['start'])}) for x in y.get('trading', [])])

    def scaled(self, **mult):   # 成本情景：按倍数调整费率，例如 commission_rate = 2
        return RuleSet([replace(r, **{k: getattr(r, k) * v for k, v in mult.items()}) for r in self.rules], self.trading)

    def fee_rule(self, day):
        hit = [r for r in self.rules if r.start <= day]
        if not hit: raise ValueError(f'no fee rule for {day}')
        if not hit[-1].verified: self.used_unverified.add(('fees', hit[-1].start))
        return hit[-1]

    def on(self, day, board = 'main', st = False):
        r = self.fee_rule(day)
        if self.trading:
            t = [x for x in self.trading if x.start <= day and x.board == board and x.st == bool(st)]
            if not t: raise ValueError(f'no trading rule for {day} board={board} st={st}')
            t = t[-1]; r = replace(r, limit_pct = t.limit_pct, buy_unit = t.buy_unit, price_tick = t.price_tick, verified = r.verified and t.verified)
            if not t.verified: self.used_unverified.add(('trading', t.board, t.st, t.start))
        return r

    def fill_price(self, raw, side, slippage, day, board = 'main', st = False):
        tick = self.on(day, board, st).price_tick; x = raw * (1 + slippage) if side == 'buy' else raw * (1 - slippage)
        return _round(x, tick, ROUND_CEILING if side == 'buy' else ROUND_FLOOR)

    def limit_prices(self, preclose, day, board = 'main', st = False):
        r = self.on(day, board, st); return _round(preclose * (1 - r.limit_pct), r.price_tick), _round(preclose * (1 + r.limit_pct), r.price_tick)

    def fees(self, amount, side, day):
        r = self.fee_rule(day)
        commission = _round(max(amount * r.commission_rate, r.min_commission))
        stamp = _round(amount * r.stamp_tax) if side == 'sell' or r.stamp_both_sides else 0.0
        transfer = _round(amount * r.transfer_fee)
        return {'commission': commission, 'stamp_tax': stamp, 'transfer_fee': transfer, 'fee': round(commission + stamp + transfer, 2)}
