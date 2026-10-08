"""策略规格（声明式）：一个 YAML 描述一个策略的股票池、选股、调度、择时、退出与执行，不含任意代码。

规格只描述「做什么」；数据从快照读取，成交与费用由账本按执行规则集模拟。
"""
from datetime import date
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from ..factors.expr import parse
from .fields import FIELDS, INDEX_FIELDS


class _Strict(BaseModel):
    model_config = ConfigDict(extra = 'forbid', allow_inf_nan = False)


def _expr(text, fields = FIELDS):
    parse(text, fields); return text


class Source(_Strict):
    file: str                                       # repo/量化策略源代码 下的相对路径
    url: str = ''
    author: str = ''


class Universe(_Strict):
    boards: list[Literal['main', 'gem', 'star']] = Field(default_factory = lambda: ['main'], min_length = 1)
    exclude_st: bool = True                         # 决策日为 ST / *ST 的不入选
    exclude_paused: bool = True                     # 决策日停牌的不入选
    min_listed_days: int = Field(0, ge = 0)         # 决策日已上市的自然日天数（聚宽常见写法是 days）
    filters: list[str] = Field(default_factory = list)   # 表达式，结果 > 0 保留；多个条件同时满足

    @model_validator(mode = 'after')
    def _check(self):
        for f in self.filters: _expr(f)
        return self


class Step(_Strict):
    """选股管线的一步：filter 保留表达式 > 0 的；sort 按表达式排序后取前 take 个（缺失值排在最后、不入选）"""
    filter: str | None = None
    sort: str | None = None
    asc: bool = True
    take: int | None = Field(None, ge = 1)

    @model_validator(mode = 'after')
    def _check(self):
        if (self.filter is None) == (self.sort is None): raise ValueError('每一步必须且只能是 filter 或 sort 之一')
        if self.sort is not None and self.take is None: raise ValueError('sort 必须给出 take')
        if self.filter is not None and self.take is not None: raise ValueError('filter 不接受 take')
        _expr(self.filter or self.sort); return self


class Select(_Strict):
    """多条管线按顺序取并集（去重、保留先后），再截取前 n 个作为目标名单"""
    pipelines: list[list[Step]] = Field(min_length = 1)
    n: int = Field(ge = 1)


class Schedule(_Strict):
    """调仓日按执行日（成交日）定义：决策用前一交易日收盘后的数据，在调仓日开盘成交，对应聚宽 9:30 用 previous_date 数据的写法"""
    freq: Literal['daily', 'weekly', 'monthly', 'every_n'] = 'monthly'
    day: int = Field(1, ge = -23, le = 23)          # weekly：当周第几个交易日；monthly：当月第几个交易日；负数从末尾数
    n: int = Field(1, ge = 1)                       # every_n：每 n 个交易日

    @model_validator(mode = 'after')
    def _check(self):
        if self.day == 0: raise ValueError('day 不能为 0')
        if self.freq == 'weekly' and not -5 <= self.day <= 5: raise ValueError('weekly 的 day 在 ±1..5')
        return self


class Exposure(_Strict):
    """择时：在指数日线上按表达式决定仓位比例；以及固定空仓月份"""
    index: str | None = None                        # 指数代码，如 000300.SH；需要快照含 index_1d
    expr: str | None = None                         # 指数面板上的表达式，> 0 为看多
    on: float = Field(1.0, ge = 0, le = 1)
    off: float = Field(0.0, ge = 0, le = 1)
    empty_months: list[int] = Field(default_factory = list)   # 这些月份的调仓日清仓且不买入（如小市值常见的 1、4 月空仓）

    @model_validator(mode = 'after')
    def _check(self):
        if (self.index is None) != (self.expr is None): raise ValueError('index 与 expr 必须同时给出')
        if self.expr is not None: _expr(self.expr, INDEX_FIELDS)
        if any(not 1 <= m <= 12 for m in self.empty_months): raise ValueError('empty_months 取 1..12')
        return self


class Exits(_Strict):
    """持仓的逐日退出检查（收盘后判断，下一交易日开盘卖出）"""
    stop_loss: float | None = Field(None, gt = 0, lt = 1)       # 收盘价相对持仓成本跌幅达到即卖出
    take_profit: float | None = Field(None, gt = 0)            # 收盘价相对持仓成本涨幅达到即卖出
    limit_up_break: bool = False                               # 前一日收盘涨停、当日收盘未涨停即卖出（聚宽多在 14:00 判断，这里用收盘近似）
    drop_from_target: bool = False                             # 非调仓日也卖出已不满足股票池过滤的持仓（如变成 ST）


class Rebalance(_Strict):
    mode: Literal['keep', 'reweight'] = 'keep'      # keep：仍在目标中的持仓不动，现金等分买入新进名单；reweight：全部调回目标权重
    band: float = Field(0.2, ge = 0)                # reweight 下偏离目标市值超过该比例才调整


class Execution(_Strict):
    slippage: float = Field(0.001, ge = 0, lt = 0.1)
    initial_cash: float = Field(1_000_000, gt = 0)


class StrategySpec(_Strict):
    id: str = Field(pattern = r'^[a-z0-9][a-z0-9-]*$')
    title: str
    source: Source
    archetype: str
    idea: str                                       # 实现思路：原策略在做什么、这里如何表达
    deviations: list[str] = Field(default_factory = list)   # 与原文的差异（数据替代、时点近似等），报告逐条披露
    universe: Universe = Field(default_factory = Universe)
    select: Select
    schedule: Schedule = Field(default_factory = Schedule)
    rebalance: Rebalance = Field(default_factory = Rebalance)
    exposure: Exposure = Field(default_factory = Exposure)
    exits: Exits = Field(default_factory = Exits)
    execution: Execution = Field(default_factory = Execution)
    start: date | None = None                       # 留空用统一回测区间
    end: date | None = None
    valuation_policy: Literal['provider_final'] | None = Field(None, exclude_if=lambda v: v is None)

    def expressions(self):
        out = list(self.universe.filters)
        for p in self.select.pipelines: out += [s.filter or s.sort for s in p]
        return out


def load_spec(path, check_filename = True):
    path = Path(path); text = path.read_text(encoding = 'utf-8')
    spec = StrategySpec.model_validate(yaml.safe_load(text))
    if check_filename and path.stem != spec.id: raise ValueError(f'规格文件名 {path.name} 与 id {spec.id} 不一致')
    return spec, {'path': str(path), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text}
