"""无训练的规则策略：冻结来源/参数 → 历史候选/分数/目标 → 唯一账本 → 成本与离线复现。"""
import hashlib, math
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from .cache import StageCache, code_version, digest
from .data.prices import with_adjusted
from .data.constituents import POLICY as INDEX_POLICY, SIZES as INDEX_SIZES, membership_records
from .data.store import Store
from .evaluation.benchmarks import benchmark_comparison, price_levels, stock_price_levels
from .evaluation.portfolio import metrics
from .execution import InputBlocked, covered_history_warmup, load, sessions, validate_scope
from .experiments import reference, seal, verify_manifest
from .features import panel
from .factors.expr import compute
from .replay import RULES, ExecutionConfig, PortfolioConfig, ReproduceRefused, RunConfig, _Strict, _read, _run, check_snapshot, read_core
from .research import UniverseConfig
from .runs import RunStatus, canonical, compare_frames, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha, resolve_run, write_json
from .schedule import rebalance_dates
from .strategy_catalog import BOLL_SOURCE, BP_SOURCE, MA_SOURCE, MA5_SOURCE, MULTI_MA_SOURCE, SVM_SOURCE, RSI_SOURCE, ROTATION_SOURCE, PAIR_YILI_SOURCE, PAIR_HAITIAN_SOURCE, REVIEWS, read_source, strategy_id
from .universe import build_universe

IMPLEMENTATIONS = {'bp_component_v1': BP_SOURCE, 'ep_component_v1': BP_SOURCE, 'ma10_ma20_v1': MA_SOURCE,
                   'ema_slots_talib_approx_v1': '2022年度精选策略/55.【策略研发】三进兵策略（变形版）.txt',
                   'bp_csi800_weekly_v1': BP_SOURCE, 'ep_csi800_weekly_v1': BP_SOURCE, 'bollinger_breakout_corrected_v1': BOLL_SOURCE,
                   'ma5_ma10_price_v1': MA5_SOURCE, 'multi_ma_fixed_value_v1': MULTI_MA_SOURCE, 'svm_lagged_shape_corrected_v1': SVM_SOURCE, 'rsi_slots_corrected_v1': RSI_SOURCE, 'pair_zscore_rotation_corrected_v1': ROTATION_SOURCE,
                   'pair_yili_cmb_anchored_v1': PAIR_YILI_SOURCE, 'pair_haitian_cmb_anchored_v1': PAIR_HAITIAN_SOURCE}
VALUATION_FIELDS = {'bp_component_v1': 'pb_mrq', 'ep_component_v1': 'pe_ttm', 'bp_csi800_weekly_v1': 'pb_mrq', 'ep_csi800_weekly_v1': 'pe_ttm'}
INDEX_IMPLEMENTATIONS = {'bp_csi800_weekly_v1', 'ep_csi800_weekly_v1'}
SINGLE_IMPLEMENTATIONS = {'ma10_ma20_v1', 'bollinger_breakout_corrected_v1', 'ma5_ma10_price_v1', 'multi_ma_fixed_value_v1', 'svm_lagged_shape_corrected_v1'}
PAIR_IMPLEMENTATIONS = {'pair_yili_cmb_anchored_v1', 'pair_haitian_cmb_anchored_v1'}
FRAMES = {'universe': ['decision_date', 'instrument'], 'factors': ['date', 'instrument'], 'scores': ['decision_date', 'instrument'], 'targets': ['decision_date', 'instrument']}


class RuleParameters(_Strict):
    top_fraction: float = Field(.1, gt = 0, le = 1)
    positive_only: bool = True
    short: int = Field(10, ge = 1, le = 500)
    long: int = Field(20, ge = 2, le = 500)
    buy_multiplier: float = Field(1.01, gt = 0)
    mean_algorithm: Literal['window_fsum_v1'] | None = Field(None, exclude_if = lambda v: v is None)


class BollingerParameters(_Strict):
    boll_window: int = Field(ge = 2, le = 500)
    boll_std_multiplier: float = Field(gt = 0, le = 10)


class PairParameters(_Strict):
    instrument1: str
    instrument2: str
    test_days: Literal[120]
    regression_ratio: Literal[1.0]
    threshold: Literal[1.0]
    price_basis: Literal['decision_close_anchor_v1']


class MultiMAParameters(_Strict):
    windows: tuple[Literal[5], Literal[10], Literal[20], Literal[30]]
    target_value: Literal[20000.0]
    struggle10_20: Literal[0.003]
    struggle20_30: Literal[0.002]
    mean_algorithm: Literal['window_fsum_v1'] | None = Field(None, exclude_if = lambda v: v is None)


class SVMParameters(_Strict):
    history: Literal[252]
    feature_window: Literal[22]
    label_horizon: Literal[5]
    prediction_shape: Literal['one_row_v1']
    price_basis: Literal['frozen_back_adjusted_v1']
    volume_basis: Literal['raw_shares_v1']
    svc_parameters: dict

    @model_validator(mode = 'after')
    def _defaults(self):
        from .strategy_svm import SVC_DEFAULTS
        if self.svc_parameters != SVC_DEFAULTS: raise ValueError('SVM98只修预测形状，不更改当前默认SVC参数')
        return self


class IndexUniverseConfig(_Strict):
    index: Literal['000300.SH', '000905.SH', '000906.SH']
    policy: Literal['provider_weekly_asof']
    max_age_days: int = Field(7, ge = 0, le = 31)


class RSIParameters(_Strict):
    pool: list[str]
    history: Literal[61]
    period: Literal[6]
    max_positions: Literal[9]
    price_basis: Literal['frozen_back_adjusted_v1']
    window_policy: Literal['skip_known_pauses_v1']

    @model_validator(mode = 'after')
    def _pool(self):
        from .strategy_rsi import POOL
        if self.pool != list(POOL): raise ValueError('RSI须保留原42项股票池顺序和重复项')
        return self


class RotationParameters(_Strict):
    instrument1: Literal['002415.SZ']
    instrument2: Literal['000651.SZ']
    history: Literal[60]
    ddof: Literal[1]
    round_digits: Literal[4]
    threshold: Literal[2.0]
    price_basis: Literal['decision_close_anchor_v1']


class EMAParameters(_Strict):
    pool: list[str]
    periods: tuple[Literal[2], Literal[25], Literal[60]]
    max_positions: Literal[5]
    cash_divisor: Literal[1.5]
    history_anchor: Literal['2005-01-05']
    price_basis: Literal['frozen_back_adjusted_v1']
    window_policy: Literal['continuous_skip_known_pauses_v1']
    indicator_policy: Literal['talib_mean_seed_v1']

    @model_validator(mode = 'after')
    def _pool(self):
        from .strategy_ema import POOL
        if self.pool != list(POOL): raise ValueError('EMA须保留原八股顺序')
        return self


class StrategyConfig(_Strict):
    name: str = ''
    implementation: Literal['bp_component_v1', 'ep_component_v1', 'ma10_ma20_v1', 'bp_csi800_weekly_v1', 'ep_csi800_weekly_v1', 'bollinger_breakout_corrected_v1', 'ma5_ma10_price_v1', 'pair_yili_cmb_anchored_v1', 'pair_haitian_cmb_anchored_v1', 'multi_ma_fixed_value_v1', 'svm_lagged_shape_corrected_v1', 'rsi_slots_corrected_v1', 'pair_zscore_rotation_corrected_v1', 'ema_slots_talib_approx_v1']
    source_path: str
    snapshot: str
    start: date
    end: date
    universe: UniverseConfig                    # 策略必须显式声明股票池，不能静默继承现有默认规则
    index_universe: IndexUniverseConfig | None = None
    portfolio: PortfolioConfig                  # 调仓、缓冲、仓位与补单参数同样显式冻结
    parameters: RuleParameters | BollingerParameters | PairParameters | MultiMAParameters | SVMParameters | RSIParameters | RotationParameters | EMAParameters = Field(default_factory = RuleParameters)
    instrument: str = '000333.SZ'
    initial_cash: float = Field(1_000_000, gt = 0)
    rules: str = RULES
    execution: ExecutionConfig = Field(default_factory = lambda: ExecutionConfig(slippage = .001))
    cost_scenarios: list[Literal['base', 'fees_x2', 'slippage_x2']] = Field(default_factory = lambda: ['base', 'fees_x2', 'slippage_x2'], min_length = 1)
    benchmark: str = '000300.SH'
    cache: bool = True
    execution_instruments: list[str] | None = Field(None, exclude_if = lambda v: v is None)
    benchmark_kind: Literal['index', 'stock'] = Field('index', exclude_if = lambda v: v == 'index')

    @model_validator(mode = 'before')
    @classmethod
    def _explicit_universe(cls, value):
        if isinstance(value, dict) and value.get('implementation') in ('bollinger_breakout_corrected_v1', 'ma5_ma10_price_v1', 'multi_ma_fixed_value_v1', 'svm_lagged_shape_corrected_v1') and 'instrument' not in value:
            raise ValueError('新单股变体必须显式声明 instrument，不继承美的策略的默认标的')
        if isinstance(value, dict) and isinstance(value.get('universe'), dict):
            required = {'boards', 'min_listed_sessions', 'exclude_st', 'suspend_window', 'max_suspended', 'liquidity_window', 'min_avg_amount'}
            if required - set(value['universe']): raise ValueError(f'规则策略股票池规则须全部明确：{sorted(required - set(value["universe"]))}')
        return value

    @model_validator(mode = 'after')
    def _validate(self):
        validate_scope(self.execution_instruments)
        if getattr(self.parameters, 'mean_algorithm', None) is not None and self.implementation not in {'ma5_ma10_price_v1', 'multi_ma_fixed_value_v1'}:
            raise ValueError('数值算法修正仅用于已批准的国航与复星变体')
        if self.implementation == 'ema_slots_talib_approx_v1':
            from .strategy_ema import POOL
            if not isinstance(self.parameters, EMAParameters) or self.execution_instruments != list(POOL): raise ValueError('EMA执行范围须为原八股顺序')
            u, p = self.universe, self.portfolio
            if u.model_dump() != {'boards': ['main', 'gem'], 'min_listed_sessions': 1, 'exclude_st': False, 'suspend_window': 1, 'max_suspended': 1, 'liquidity_window': 1, 'min_avg_amount': 0.0}:
                raise ValueError('EMA固定池不得附加筛选')
            if (p.n, p.max_weight, p.rebalance_frequency, p.rebalance_every, p.rebalance_session, p.refill_between_rebalance, p.open_cash_policy, p.participation) != (5, 1, 'daily', 1, 1, False, 'sell_then_buy', .25):
                raise ValueError('EMA须5槽位、daily、先卖后买、不补单、25%参与')
            if self.execution.slippage != .00246 or self.execution.fee_multiplier != 1: raise ValueError('EMA须原基础滑点与成本倍率')
            if self.execution.liquidity_window != 20 or self.execution.liquidity_override is not None: raise ValueError('EMA须20日真实成交额约束')
            if self.benchmark_kind != 'index' or self.benchmark != '000300.SH': raise ValueError('EMA使用沪深300价格基准')
        elif self.implementation == 'pair_zscore_rotation_corrected_v1':
            if not isinstance(self.parameters, RotationParameters) or self.instrument != '002415.SZ' or self.execution_instruments != ['002415.SZ', '000651.SZ']:
                raise ValueError('66号须冻结海康/格力原序标的')
            u, p = self.universe, self.portfolio
            if u.model_dump() != {'boards': ['main'], 'min_listed_sessions': 1, 'exclude_st': False, 'suspend_window': 1, 'max_suspended': 1, 'liquidity_window': 1, 'min_avg_amount': 0.0}:
                raise ValueError('66号固定配对不得附加筛选')
            if (p.n, p.max_weight, p.rebalance_frequency, p.rebalance_every, p.rebalance_session, p.refill_between_rebalance, p.open_cash_policy) != (2, 1, 'daily', 1, 1, False, 'sell_then_buy'):
                raise ValueError('66号须daily、先卖后买、不补单')
            if self.benchmark_kind != 'index' or self.benchmark != '000300.SH': raise ValueError('66号使用沪深300价格基准')
        elif self.implementation == 'rsi_slots_corrected_v1':
            from .strategy_rsi import UNIQUE_POOL
            if not isinstance(self.parameters, RSIParameters) or self.execution_instruments != list(UNIQUE_POOL): raise ValueError('RSI执行范围须为原41只不同股票')
            if self.benchmark_kind != 'index' or self.benchmark != '000300.SH': raise ValueError('RSI使用沪深300价格基准')
            u, p = self.universe, self.portfolio
            if u.model_dump() != {'boards': ['main', 'gem'], 'min_listed_sessions': 1, 'exclude_st': False, 'suspend_window': 1, 'max_suspended': 1, 'liquidity_window': 1, 'min_avg_amount': 0.0}:
                raise ValueError('RSI固定股票池不得附加筛选条件')
            if (p.n, p.max_weight, p.rebalance_frequency, p.rebalance_every, p.rebalance_session, p.refill_between_rebalance, p.open_cash_policy) != (9, 1, 'daily', 1, 1, False, 'sell_then_buy'):
                raise ValueError('RSI须9槽位、daily、先卖后买、不补单')
        elif self.implementation in PAIR_IMPLEMENTATIONS:
            if not isinstance(self.parameters, PairParameters): raise ValueError('配对变体必须显式声明配对参数与价格基准')
            pair = [self.parameters.instrument1, self.parameters.instrument2]; validate_scope(pair)
            original = REVIEWS[IMPLEMENTATIONS[self.implementation]]['rules']['instruments']
            if pair != original or self.instrument != pair[0] or self.execution_instruments != pair:
                raise ValueError('配对标的、instrument与execution_instruments须与原来源顺序完全一致')
            if self.benchmark_kind != 'stock' or self.benchmark != pair[1]: raise ValueError('配对基准须为招行股票价格')
            if self.portfolio.n != 2 or self.portfolio.max_weight != 1 or self.portfolio.rebalance_frequency != 'daily' or self.portfolio.rebalance_every != 1:
                raise ValueError('配对日频近似须设置 n=2、max_weight=1、daily、rebalance_every=1')
        elif self.execution_instruments is not None:
            if self.implementation not in SINGLE_IMPLEMENTATIONS or self.execution_instruments != [self.instrument]:
                raise ValueError('单股策略 execution_instruments 必须恰为 [instrument]；全市场策略不得缩减范围')
        if self.benchmark_kind == 'stock' and self.implementation not in PAIR_IMPLEMENTATIONS | {'multi_ma_fixed_value_v1', 'svm_lagged_shape_corrected_v1'}:
            raise ValueError('股票价格基准仅用于已明确声明的实现')
        if self.start > self.end: raise ValueError('start 晚于 end')
        expected_construction = {'multi_ma_fixed_value_v1': 'conditional_values', 'rsi_slots_corrected_v1': 'signal_slots', 'ema_slots_talib_approx_v1': 'signal_slots', 'pair_zscore_rotation_corrected_v1': 'signal_cash_rotation'}.get(self.implementation, 'target_weights')
        if self.portfolio.construction != expected_construction: raise ValueError(f'规则策略必须显式选择 {expected_construction}')
        if self.portfolio.buffer or self.portfolio.max_sell is not None: raise ValueError('目标权重策略不接受排名缓冲或换出限制；设置 buffer=0、max_sell=null')
        if len(set(self.cost_scenarios)) != len(self.cost_scenarios) or 'base' not in self.cost_scenarios: raise ValueError('成本情景须不重复且包含 base')
        if 'slippage_x2' in self.cost_scenarios and self.execution.slippage >= .05: raise ValueError('加倍滑点超出执行范围')
        if self.implementation in INDEX_IMPLEMENTATIONS:
            if self.index_universe is None or self.index_universe.index != '000906.SH': raise ValueError('CSI800 实现必须显式声明中证800及周频历史成分口径')
            if 'star' in self.universe.boards and self.universe.min_listed_sessions < 6: raise ValueError('科创板日频版本须排除上市前5个交易日；当前未模拟IPO无涨跌幅期')
        elif self.index_universe is not None: raise ValueError('请为历史成分股票池使用单独登记的 CSI800 实现')
        if self.implementation == 'bollinger_breakout_corrected_v1':
            if not isinstance(self.parameters, BollingerParameters): raise ValueError('布林变体必须显式声明 boll_window、boll_std_multiplier')
            if self.portfolio.n != 1 or self.portfolio.rebalance_frequency != 'daily' or self.portfolio.rebalance_every != 1:
                raise ValueError('布林日频修正变体须显式设置 n=1、daily、rebalance_every=1')
        elif self.implementation == 'multi_ma_fixed_value_v1':
            if not isinstance(self.parameters, MultiMAParameters): raise ValueError('多均线须显式声明原固定参数')
            if self.instrument != '600196.SH' or self.execution_instruments != [self.instrument] or self.benchmark != self.instrument or self.benchmark_kind != 'stock':
                raise ValueError('多均线标的、范围与股票基准须为复星医药')
            if self.portfolio.n != 1 or self.portfolio.max_weight != 1 or self.portfolio.rebalance_frequency != 'daily' or self.portfolio.rebalance_every != 1 or self.portfolio.refill_between_rebalance:
                raise ValueError('多均线须设置单股、daily、不补单')
        elif self.implementation == 'svm_lagged_shape_corrected_v1':
            if not isinstance(self.parameters, SVMParameters): raise ValueError('SVM须显式冻结原参数和形状修正')
            if self.instrument != '600085.SH' or self.execution_instruments != [self.instrument] or self.benchmark != self.instrument or self.benchmark_kind != 'stock':
                raise ValueError('SVM标的、范围与股票基准须为同仁堂')
            if self.portfolio.n != 1 or self.portfolio.max_weight != 1 or self.portfolio.rebalance_frequency != 'weekly' or self.portfolio.rebalance_session != 3 or self.portfolio.rebalance_every != 1 or self.portfolio.refill_between_rebalance:
                raise ValueError('SVM须单股满仓、周内第三交易日、不补单')
        elif self.implementation not in PAIR_IMPLEMENTATIONS | {'rsi_slots_corrected_v1', 'pair_zscore_rotation_corrected_v1', 'ema_slots_talib_approx_v1'} and not isinstance(self.parameters, RuleParameters): raise ValueError('此实现不接受布林或配对参数，也不接受其他规则参数')
        if self.implementation == 'ma5_ma10_price_v1':
            if (self.parameters.short, self.parameters.long, self.parameters.buy_multiplier) != (5, 10, 1):
                raise ValueError('MA5/MA10来源参数须显式设置 short=5、long=10、buy_multiplier=1')
            if self.portfolio.n != 1 or self.portfolio.rebalance_frequency != 'daily' or self.portfolio.rebalance_every != 1:
                raise ValueError('MA5/MA10日频近似须设置 n=1、daily、rebalance_every=1')
        return self


def _signal_window(cfg):
    if isinstance(cfg.parameters, BollingerParameters): return cfg.parameters.boll_window
    if isinstance(cfg.parameters, PairParameters): return cfg.parameters.test_days
    if isinstance(cfg.parameters, MultiMAParameters): return 32
    if isinstance(cfg.parameters, SVMParameters): return cfg.parameters.history
    if isinstance(cfg.parameters, RSIParameters): return cfg.parameters.history
    if isinstance(cfg.parameters, EMAParameters): return 61
    if isinstance(cfg.parameters, RotationParameters): return cfg.parameters.history
    return cfg.parameters.long


def _strategy_code(cfg, *modules):
    extra = {'svm_lagged_shape_corrected_v1': ['strategy_svm.py'], 'rsi_slots_corrected_v1': ['strategy_rsi.py', 'execution.py'], 'ema_slots_talib_approx_v1': ['strategy_ema.py', 'execution.py', 'portfolio.py'], 'pair_zscore_rotation_corrected_v1': ['strategy_rotation.py']}.get(cfg.implementation, [])
    if getattr(cfg.parameters, 'mean_algorithm', None) == 'window_fsum_v1': extra = [*extra, 'strategy_means.py']
    return code_version(*modules, *extra)


def _mean(close, window, params):
    if getattr(params, 'mean_algorithm', None) == 'window_fsum_v1':
        from .strategy_means import window_fsum
        return window_fsum(close, window)
    return close.rolling(window, min_periods = window).mean()


def _pair_transition(state, z):
    if z > 1: return 'buy1'
    if z < -1: return 'buy2'
    if (state == 'buy1' and z < 0) or (state == 'buy2' and z >= 0): return 'even'
    return state


def _freeze_inputs(store, state, cfg, out):
    signal_window = _signal_window(cfg)
    warmup = max(signal_window, cfg.universe.min_listed_sessions, cfg.universe.suspend_window, cfg.universe.liquidity_window, cfg.execution.liquidity_window) + 1
    if cfg.implementation == 'rsi_slots_corrected_v1':
        warmup = covered_history_warmup(state, sessions(store.load_state(state, 'calendar')), cfg.start, cfg.end, cfg.execution_instruments)
    elif cfg.implementation == 'ema_slots_talib_approx_v1':
        warmup = len(sessions(store.load_state(state, 'calendar')))
    data = load(store, state, cfg.start, cfg.end, warmup, cfg.execution_instruments)
    for table in ('adj_factors', 'adj_coverage', 'index_1d'):
        filters = [('instrument', 'in', cfg.execution_instruments)] if cfg.execution_instruments is not None and table != 'index_1d' else None
        data[table] = store.load_state(state, table, filters = filters); data['partitions'][table] = state['tables'].get(table, {})
    if cfg.index_universe is not None:
        table = 'index_constituents'; data[table] = store.load_state(state, table); data['partitions'][table] = state['tables'].get(table, {})
    used = {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in ps.items()} for t, ps in data['partitions'].items()}
    write_json(out / 'data_manifest.json', {'snapshot_id': cfg.snapshot, 'used': used, 'tables': state['tables'], 'offline': True,
                                          'valuation_mapping': {'pe_ttm': 'BaoStock peTTM（未证明与聚宽 pe_ratio 完全等价）', 'pb_mrq': 'BaoStock pbMRQ（未证明与聚宽 pb_ratio 完全等价）'},
                                          'availability': '日线收盘数据确认可用后；不读取外部原始财务文件', 'mapping_version': 'daily_valuation_v1',
                                          **({'execution_instruments': cfg.execution_instruments} if cfg.execution_instruments is not None else {})})
    if cfg.index_universe is not None:
        manifest = _read(out / 'data_manifest.json')
        manifest['index_universe'] = {**cfg.index_universe.model_dump(), 'strict_evidence': False, 'note': '供应商周频历史成分，非调整公告时刻证明'}
        write_json(out / 'data_manifest.json', manifest)
    return data, used


def _apply_index_universe(cfg, data, universe, days):
    spec = cfg.index_universe
    try: selected = membership_records(data.get('index_constituents', pd.DataFrame()), spec.index, days, spec.max_age_days)
    except ValueError as exc: raise InputBlocked([{'kind': 'index_constituents_unavailable', 'detail': str(exc)}]) from exc
    if set(selected.instrument) - set(data['instruments'].query("kind == 'stock'").instrument):
        raise InputBlocked([{'kind': 'index_constituents_unknown_security', 'detail': '冻结成分中有证券主数据未确认的代码'}])
    pairs = set(zip(selected.date, selected.instrument))
    result = universe.copy(); result['in_index'] = [(d, i) in pairs for d, i in zip(result.decision_date, result.instrument)]
    outside = result.eligible & ~result.in_index
    result.loc[outside, 'reason'] = 'outside_index'
    result['eligible'] &= result.in_index
    result['universe_version'] += ':' + digest(spec.model_dump())[:12]
    return result


def _generate_signals(cfg, data, out):
    bars = data['bars_1d'].copy(); bars['date'] = pd.to_datetime(bars.date).dt.date
    cal = sessions(data['calendar']); days = [d for d in cal if cfg.start <= d <= cfg.end]
    if not cal or cfg.start < cal[0] or cfg.end > cal[-1]: raise InputBlocked([{'kind': 'calendar_does_not_cover', 'detail': '固定日历不覆盖规则策略请求区间'}])
    bar_days = set(bars.date)
    if not days or any(d not in bar_days for d in days): raise InputBlocked([{'kind': 'missing_session', 'detail': '规则策略请求区间日线/日历不完整'}])
    first_bar, last_bar = min(bar_days), max(bar_days)
    available_days = [d for d in cal if first_bar <= d <= last_bar]
    signal_window = _signal_window(cfg)
    warmup = max(signal_window, cfg.universe.min_listed_sessions, cfg.universe.suspend_window, cfg.universe.liquidity_window)
    if sum(d < days[0] for d in available_days) < warmup: raise InputBlocked([{'kind': 'insufficient_history', 'detail': f'规则策略需要 {warmup} 个交易日预热'}])
    uni = build_universe(bars, data['instruments'], available_days, cfg.universe.model_dump(), days[0], days[-1])
    if cfg.index_universe is not None: uni = _apply_index_universe(cfg, data, uni, days)
    if cfg.implementation in SINGLE_IMPLEMENTATIONS: uni = uni[uni.instrument.eq(cfg.instrument)].copy()
    if cfg.implementation in PAIR_IMPLEMENTATIONS:
        uni = uni[uni.instrument.isin(cfg.execution_instruments)].copy()
        if set(cfg.execution_instruments) - set(uni.instrument):
            raise InputBlocked([{'kind': 'instrument_missing', 'detail': '配对日线缺少必要标的'}])
    eligible = uni[uni.eligible]; ever = sorted(set(uni.instrument))
    if not ever: raise InputBlocked([{'kind': 'universe_empty', 'detail': '请求区间没有对应证券日线，不以空仓冒充策略复现'}])
    if cfg.implementation in VALUATION_FIELDS:
        field = VALUATION_FIELDS[cfg.implementation]
        if field not in bars: raise InputBlocked([{'kind': 'valuation_missing', 'detail': f'日线缺少 {field}，不伪造数据'}])
        view = bars.loc[bars.instrument.isin(set(ever)), ['date', 'instrument', 'is_trading', field]]
        wide = panel(view, available_days, ever, fields = (field,))
    else:
        view = with_adjusted(bars[bars.instrument.isin(set(ever))], data['adj_factors'], data['adj_coverage'])
        wide = panel(view, available_days, ever)
    mask = eligible.assign(value = True).pivot(index = 'decision_date', columns = 'instrument', values = 'value').reindex(index = available_days, columns = ever).fillna(False).astype(bool)
    params = cfg.parameters; fac, score_rows, targets, coverage = [], [], [], []
    p = cfg.portfolio; decision_days = rebalance_dates(days, cal, p.rebalance_frequency, p.rebalance_every, p.rebalance_session)
    state, held_weights = 0.0, {}
    if cfg.implementation in VALUATION_FIELDS:
        field = VALUATION_FIELDS[cfg.implementation]
        values = wide[field].where(np.isfinite(wide[field])); wide[field] = values
        factors = compute('1/' + field, wide, eligible = mask)
        members_by_day = eligible.groupby('decision_date').instrument.agg(list).to_dict()
        for day in days:
            members = set(members_by_day.get(day, ()))
            row = factors.loc[day].reindex(sorted(members)); valid = row[np.isfinite(row)]
            # Column blocks avoid millions of Python dictionaries in long stock panels.
            if len(row):
                fac.append(pd.DataFrame({'date': day, 'instrument': row.index, 'value': row.to_numpy(float),
                                         'input_value': values.loc[day, row.index].to_numpy(float)}))
            if len(valid):
                score_rows.append(pd.DataFrame({'decision_date': day, 'instrument': valid.index, 'score': valid.to_numpy(float)}))
            if day in decision_days:
                pickable = valid[valid > 0] if params.positive_only else valid
                count = math.floor(len(pickable) * params.top_fraction)
                chosen = sorted(pickable.index, key = lambda i: (-pickable[i], i))[:count]
                held_weights = {i: min(1 / count, p.max_weight) for i in chosen} if count else {}
            cash = 1 - sum(held_weights.values())
            targets += [{'decision_date': day, 'instrument': i, 'weight': float(w), 'cash_weight': cash, 'rebalance': day in decision_days} for i, w in sorted(held_weights.items())]
            targets.append({'decision_date': day, 'instrument': 'CASH', 'weight': cash, 'cash_weight': cash, 'rebalance': day in decision_days})
            coverage.append({'date': day, 'candidates': len(members), 'finite_factors': len(valid), 'positive_factors': int((valid > 0).sum()), 'selected': len(held_weights), 'cash_weight': cash})
            if cfg.index_universe is not None:
                coverage[-1].update(index_members = INDEX_SIZES[cfg.index_universe.index], index_members_with_bars = int(uni[uni.decision_date.eq(day)].in_index.sum()), index_policy = INDEX_POLICY)
        fac = pd.concat(fac, ignore_index = True) if fac else pd.DataFrame()
        score_rows = pd.concat(score_rows, ignore_index = True) if score_rows else pd.DataFrame(columns = ['decision_date', 'instrument', 'score'])
    elif cfg.implementation in PAIR_IMPLEMENTATIONS:
        i1, i2 = params.instrument1, params.instrument2
        factors = view.pivot(index = 'date', columns = 'instrument', values = 'back_factor').reindex(available_days)
        pair_state = 'empty'
        weights = {'empty': (0., 0.), 'buy1': (1., 0.), 'buy2': (0., 1.), 'even': (.5, .5)}
        for day in days:
            k = wide['close_adj'].index.get_loc(day)
            basis1, basis2 = factors.at[day, i1], factors.at[day, i2]
            # Each window ends at the decision day and uses that day's known price scale.
            first = wide['close_adj'][i1].iloc[k - params.test_days + 1:k + 1].to_numpy(float) / basis1
            second = wide['close_adj'][i2].iloc[k - params.test_days + 1:k + 1].to_numpy(float) / basis2
            spread = second - params.regression_ratio * first
            mean, sigma = float(np.mean(spread)), float(np.std(spread, ddof = 0))
            known = bool(np.isfinite(spread).all() and np.isfinite([basis1, basis2, mean, sigma]).all() and sigma > 0
                         and mask.at[day, i1] and mask.at[day, i2])
            z = float((spread[-1] - mean) / sigma) if known else np.nan
            previous = pair_state
            if known: pair_state = _pair_transition(pair_state, z)
            w1, w2 = weights[pair_state]; cash = 1 - w1 - w2
            fac.append({'date': day, 'instrument': i1, 'close1': first[-1], 'close2': second[-1], 'basis1': basis1, 'basis2': basis2,
                        'spread': spread[-1], 'middle': mean, 'std': sigma, 'z': z, 'valid': known,
                        'previous_state': previous, 'state': pair_state, 'weight1': w1, 'weight2': w2})
            if known: score_rows += [{'decision_date': day, 'instrument': i1, 'score': z}, {'decision_date': day, 'instrument': i2, 'score': -z}]
            targets += [{'decision_date': day, 'instrument': i, 'weight': w, 'cash_weight': cash, 'rebalance': day in decision_days}
                        for i, w in ((i1, w1), (i2, w2), ('CASH', cash))]
            coverage.append({'date': day, 'candidates': int(mask.loc[day, [i1, i2]].sum()), 'finite_factors': int(known),
                             'selected': int(w1 > 0) + int(w2 > 0), 'cash_weight': cash})
    elif cfg.implementation == 'pair_zscore_rotation_corrected_v1':
        from .strategy_rotation import generate
        bases = view.pivot(index = 'date', columns = 'instrument', values = 'back_factor').reindex(available_days)
        fac, score_rows, targets, coverage = generate(wide, bases, days, cfg.execution_instruments)
    elif cfg.implementation == 'rsi_slots_corrected_v1':
        from .strategy_rsi import generate
        fac, score_rows, targets, coverage, info = generate(view, available_days, days)
        write_json(out / 'rsi_backend.json', info)
    elif cfg.implementation == 'ema_slots_talib_approx_v1':
        from .strategy_ema import generate
        listings = data['instruments'].set_index('instrument').list_date.map(lambda d: pd.to_datetime(d).date() if pd.notna(d) else None).to_dict()
        fac, score_rows, targets, coverage, info = generate(view, cal, days, listings)
        write_json(out / 'ema_backend.json', info)
    elif cfg.implementation == 'svm_lagged_shape_corrected_v1':
        from .strategy_svm import generate
        fac, score_rows, targets, coverage, models, samples = generate(wide, mask, days, decision_days, cfg.instrument, p.max_weight)
        write_json(out / 'svm_models.json', models)
        pd.DataFrame(samples).to_parquet(out / 'svm_samples.parquet', index = False)
    elif cfg.implementation == 'multi_ma_fixed_value_v1':
        close = wide['close_adj'][cfg.instrument]
        averages = pd.DataFrame({f'ma{n}': _mean(close, n, params) for n in params.windows})
        bull = (averages.ma5 > averages.ma10) & (averages.ma10 > averages.ma20) & (averages.ma20 > averages.ma30)
        bear = (averages.ma5 < averages.ma10) & (averages.ma10 < averages.ma20)
        struggle = ((averages.ma10 / averages.ma20 - 1).abs() < params.struggle10_20) | ((averages.ma20 / averages.ma30 - 1).abs() < params.struggle20_30)
        crossdown = bull.shift(1, fill_value = False) & bull.shift(2, fill_value = False) & (averages.ma5.shift(1) > averages.ma10.shift(1)) & (averages.ma5 < averages.ma10)
        crossup = bear.shift(1, fill_value = False) & bear.shift(2, fill_value = False) & (averages.ma10.shift(1) < averages.ma20.shift(1)) & (averages.ma10 > averages.ma20)
        valid = np.isfinite(averages).all(axis = 1).rolling(3, min_periods = 3).sum().eq(3)
        for day in days:
            known = bool(valid.loc[day] and mask.at[day, cfg.instrument])
            flags = {'bull': bool(bull.loc[day]), 'bear': bool(bear.loc[day]), 'struggle': bool(struggle.loc[day]),
                     'crossdown': bool(crossdown.loc[day]), 'crossup': bool(crossup.loc[day])}
            fac.append({'date': day, 'instrument': cfg.instrument, 'close_adj': close.loc[day], **averages.loc[day].to_dict(), 'valid': known, **flags})
            enter = known and ((flags['bull'] and not flags['struggle']) or flags['crossup'])
            exit_signal = known and (flags['bear'] or flags['crossdown'])
            skip = known and flags['bull'] and flags['struggle']
            targets.append({'decision_date': day, 'instrument': cfg.instrument, 'target_value': params.target_value,
                            'enter_when_empty': enter, 'exit': exit_signal, 'skip_when_empty': skip, 'rebalance': day in decision_days})
            if known: score_rows.append({'decision_date': day, 'instrument': cfg.instrument, 'score': float(averages.at[day, 'ma5'] / averages.at[day, 'ma10'] - 1)})
            coverage.append({'date': day, 'candidates': int(mask.at[day, cfg.instrument]), 'finite_factors': int(known),
                             'enter_when_empty': enter, 'exit': exit_signal, 'skip_when_empty': skip, 'target_value': params.target_value})
    elif cfg.implementation == 'bollinger_breakout_corrected_v1':
        if cfg.instrument not in ever: raise InputBlocked([{'kind': 'instrument_missing', 'detail': cfg.instrument}])
        close = wide['close_adj'][cfg.instrument]
        rolling = close.rolling(params.boll_window, min_periods = params.boll_window)
        middle, std = rolling.mean(), rolling.std(ddof = 0)
        upper, lower = middle + params.boll_std_multiplier * std, middle - params.boll_std_multiplier * std
        for day in days:
            price, mid, sigma, top, bottom = (s.loc[day] for s in (close, middle, std, upper, lower))
            known = np.isfinite([price, mid, sigma, top, bottom]).all() and bool(mask.at[day, cfg.instrument])
            action = 'hold'
            if known:
                if price > top: state, action = p.max_weight, 'buy'
                elif price < bottom: state, action = 0.0, 'sell'
            fac.append({'date': day, 'instrument': cfg.instrument, 'close_adj': price, 'middle': mid, 'std': sigma,
                        'upper': top, 'lower': bottom, 'valid': bool(known), 'action': action, 'state': state})
            if known: score_rows.append({'decision_date': day, 'instrument': cfg.instrument, 'score': float(price / mid - 1)})
            targets += [{'decision_date': day, 'instrument': cfg.instrument, 'weight': state, 'cash_weight': 1 - state, 'rebalance': day in decision_days},
                        {'decision_date': day, 'instrument': 'CASH', 'weight': 1 - state, 'cash_weight': 1 - state, 'rebalance': day in decision_days}]
            coverage.append({'date': day, 'candidates': int(mask.at[day, cfg.instrument]), 'finite_factors': int(known), 'selected': int(state > 0), 'cash_weight': 1 - state})
    else:
        if cfg.instrument not in ever: raise InputBlocked([{'kind': 'instrument_missing', 'detail': cfg.instrument}])
        close = wide['close_adj'][cfg.instrument]; short = _mean(close, params.short, params); long = _mean(close, params.long, params)
        for day in days:
            price, a, b = close.loc[day], short.loc[day], long.loc[day]
            known = np.isfinite([price, a, b]).all() and bool(mask.at[day, cfg.instrument])
            if known:
                if cfg.implementation == 'ma5_ma10_price_v1':
                    if a > b and price > a: state = p.max_weight
                    if price < a: state = 0.0
                else:
                    if price > params.buy_multiplier * a: state = p.max_weight
                    elif price < b: state = 0.0
            fac.append({'date': day, 'instrument': cfg.instrument, 'close_adj': price, 'ma_short': a, 'ma_long': b, 'valid': bool(known), 'state': state})
            if known: score_rows.append({'decision_date': day, 'instrument': cfg.instrument, 'score': float(price / a - 1)})
            targets += [{'decision_date': day, 'instrument': cfg.instrument, 'weight': state, 'cash_weight': 1 - state, 'rebalance': day in decision_days},
                        {'decision_date': day, 'instrument': 'CASH', 'weight': 1 - state, 'cash_weight': 1 - state, 'rebalance': day in decision_days}]
            coverage.append({'date': day, 'candidates': int(mask.at[day, cfg.instrument]), 'finite_factors': int(known), 'selected': int(state > 0), 'cash_weight': 1 - state})
    for name, frame in {'universe': uni, 'factors': pd.DataFrame(fac), 'scores': pd.DataFrame(score_rows, columns = ['decision_date', 'instrument', 'score']),
                        'targets': pd.DataFrame(targets), 'signal_coverage': pd.DataFrame(coverage)}.items():
        if name in FRAMES and frame.duplicated(FRAMES[name]).any(): raise ValueError(f'{name}: 规则产物主键重复')
        frame.to_parquet(out / f'{name}.parquet', index = False)


def rule_source(root, cfg, snapshot):
    if not cfg.scores.run or cfg.portfolio.construction not in ('target_weights', 'conditional_values', 'signal_slots', 'signal_cash_rotation'): raise ValueError('rules 来源需要规则策略目录和规则组合')
    run = resolve_run(root, cfg.scores.run); doc = _read(run / 'config.json'); freeze = _read(run / 'signals_manifest.json')
    if doc.get('kind') != 'strategy' or doc['snapshot_id'] != snapshot: raise ValueError('规则策略与账本快照不匹配')
    if file_sha(run / 'source.original') != doc['source']['bytes_sha256']: raise ReproduceRefused('规则策略原始源码与登记指纹不符')
    if cfg.scores.model != doc['config']['implementation']: raise ValueError('规则策略实现编号不匹配')
    if cfg.portfolio.construction != doc['config']['portfolio']['construction']: raise ValueError('组合构建方式与冻结规则策略不匹配')
    if cfg.portfolio.construction in ('signal_slots', 'signal_cash_rotation') and cfg.portfolio.model_dump() != doc['config']['portfolio']:
        raise ValueError('槽位组合参数与冻结策略不匹配')
    if cfg.execution_instruments != doc['config'].get('execution_instruments'):
        raise ValueError('账本 execution_instruments 与冻结规则策略范围不匹配')
    if cfg.execution_instruments is not None:
        frame = pd.read_parquet(run / 'targets.parquet', columns = ['instrument'])
        if set(frame.instrument) - {'CASH', *cfg.execution_instruments}:
            raise ValueError('冻结目标超出 execution_instruments 范围')
    if any(file_sha(run / name) != sha for name, sha in freeze['files'].items()): raise ReproduceRefused('规则信号冻结产物指纹不同')
    if (cfg.start and cfg.start < date.fromisoformat(doc['config']['start'])) or (cfg.end and cfg.end > date.fromisoformat(doc['config']['end'])): raise ValueError('账本区间超出规则信号定义范围')
    meta = {'source': 'rules', 'run': str(run), 'strategy_id': doc['source']['strategy_id'], 'implementation': doc['config']['implementation'],
            'signals_manifest_sha256': file_sha(run / 'signals_manifest.json'), 'source_sha256': doc['source']['bytes_sha256'], 'evidence': 'exploratory_rule_strategy'}
    return meta, cfg.model_copy(update = {'scores': cfg.scores.model_copy(update = {'run': str(run)}), 'start': cfg.start or date.fromisoformat(doc['config']['start']), 'end': cfg.end or date.fromisoformat(doc['config']['end'])})


def rule_scores(meta, dates):
    run = Path(meta['run']); allowed = set(dates)
    scores = pd.read_parquet(run / 'scores.parquet'); scores['decision_date'] = pd.to_datetime(scores.decision_date).dt.date; scores = scores[scores.decision_date.isin(allowed)]
    universe = pd.read_parquet(run / 'universe.parquet'); universe['decision_date'] = pd.to_datetime(universe.decision_date).dt.date
    eligible = {day: set() for day in dates}
    for day, rows in universe[universe.eligible & universe.decision_date.isin(allowed)].groupby('decision_date'):
        eligible[day] = set(rows.instrument)
    return scores.to_dict('records'), eligible, rule_targets(meta, dates)


def rule_targets(meta, dates):
    run = Path(meta['run']); allowed = set(dates)
    frame = pd.read_parquet(run / 'targets.parquet'); frame['decision_date'] = pd.to_datetime(frame.decision_date).dt.date
    config = _read(run / 'config.json')['config']; construction = config['portfolio']['construction']
    targets = {}
    for day, data in frame[frame.decision_date.isin(allowed)].groupby('decision_date'):
        if construction == 'signal_cash_rotation':
            from .strategy_rotation import rotation_batch
            targets[day] = rotation_batch(data[['instrument', 'buy', 'sell']].to_dict('records'), config['execution_instruments'])
            continue
        if construction == 'signal_slots':
            if config['implementation'] == 'ema_slots_talib_approx_v1':
                from .strategy_ema import slot_batch
            else:
                from .strategy_rsi import slot_batch
            targets[day] = slot_batch(data[['instrument', 'buy', 'sell', 'priority']].to_dict('records'), config['parameters']['pool'])
            continue
        if construction == 'conditional_values':
            from .portfolio import conditional_value_orders
            if data.instrument.duplicated().any(): raise ValueError(f'{day}: 条件金额目标重复')
            fields = ['target_value', 'enter_when_empty', 'exit', 'skip_when_empty']
            signals = data.set_index('instrument')[fields].to_dict('index')
            conditional_value_orders(signals, {})
            targets[day] = signals
            continue
        if data.instrument.duplicated().any() or not np.isfinite(data.weight).all() or (data.weight < 0).any() or abs(data.weight.sum() - 1) > 1e-9:
            raise ValueError(f'{day}: 目标权重无效')
        targets[day] = {r.instrument: r.weight for r in data.itertuples() if r.instrument != 'CASH' and r.weight > 0}
    if set(targets) != allowed: raise ValueError('规则策略缺决策日目标，不当作现金状态')
    return targets


def run_strategy(root, output = None, **params): return _strategy(root, StrategyConfig.model_validate(params), output)


def _implementation_review(cfg):
    review = REVIEWS[IMPLEMENTATIONS[cfg.implementation]]
    if cfg.implementation == 'ema_slots_talib_approx_v1':
        review = {**review, 'scope': '用户批准的三进兵55 TA-Lib连续EMA近似变体', 'fidelity': '日频近似，原平台不等价',
            'gaps': [g for g in review['gaps'] if not g.startswith('八股早期连续行情待补')] + ['固定池事后偏差；平台即时成交/持仓遍历顺序等价性未证明'],
            'differences': ['TA-Lib EMA均值种子，从2005-01-05或上市起连续递推、跳过明确停牌；原平台内部实现未证明',
                '日线收盘决策/次日开盘，唯一Book的费用和成交约束；原平台即时成交与遍历顺序等价性未证明',
                '原固定八股用于来源日期之前，存在事后选择偏差；不作为历史可投资股票池',
                '保留EMA2/25/60严格交叉、5槽位、现金/(剩余槽位*1.5)、25%参与和0.246%基础滑点']}
    if getattr(cfg.parameters, 'mean_algorithm', None) == 'window_fsum_v1':
        review = {**review, 'scope': review['scope'] + '（用户批准的逐窗口稳定求和数值修正）',
                  'gaps': [*review['gaps'], '原平台均线浮点算法与本变体等价性未证明'],
                  'differences': [*review['differences'], '均线逐窗口math.fsum后除以窗口长度；保留严格比较，不引入容差；原滚动计算实验独立保留']}
    if cfg.index_universe is None: return review
    return {**review, 'scope': 'BP/EP 组件与中证800周频历史成分近似版本', 'implementations': sorted(INDEX_IMPLEMENTATIONS),
            'gaps': [g for g in review['gaps'] if g != '缺中证800历史成分'] + ['供应商成分按周更新，调整公告和临时调整的日精度未证明'],
            'differences': [f'使用对应日期的沪深300与中证500周频归档并集合成中证800；启用板块{cfg.universe.boards}，上市至少{cfg.universe.min_listed_sessions}交易日',
                            '科创板数量约束通过单独规则集声明；日线开盘成交不模拟盘中订单簿'] + review['differences'][1:]}


def _strategy(root, cfg, output = None, source_file = None, rules_file = None, reproduce_of = None, compare = None):
    store = Store(root); state = store.state(cfg.snapshot); env = environment(); source = Path(source_file or cfg.source_path)
    if cfg.implementation == 'rsi_slots_corrected_v1':
        from .strategy_rsi import backend
        env['rsi_backend'] = backend()[1]
    elif cfg.implementation == 'ema_slots_talib_approx_v1':
        from .strategy_ema import backend
        env['ema_backend'] = backend()[1]
    text, encoding = read_source(source); relative = IMPLEMENTATIONS[cfg.implementation]; review = _implementation_review(cfg)
    source_meta = {'strategy_id': strategy_id(relative), 'path': cfg.source_path, 'canonical_source': relative, 'bytes_sha256': file_sha(source),
                   'content_sha256': hashlib.sha256(text.encode()).hexdigest(), 'encoding': encoding, 'review': review}
    config_hash = digest({'config': cfg.model_dump(mode = 'json'), 'source': source_meta, 'implementation_code': _strategy_code(cfg, 'strategies.py', 'schedule.py', 'data/constituents.py')})
    out = create_run_dir(store.root / 'runs', output, f'{config_hash[:6]}-strategy')
    status = RunStatus(out, out.name, kind = 'strategy', registry = store.root / 'runs', config_hash = config_hash, evidence = 'exploratory', snapshot_id = cfg.snapshot, reproduce_of = reproduce_of)
    (out / 'source.original').write_bytes(source.read_bytes())
    (out / 'rules.yaml').write_bytes(Path(rules_file or cfg.rules).read_bytes())
    doc = {'kind': 'strategy', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': cfg.snapshot, 'batch_id': state['batch_id'], 'source': source_meta, 'environment': env,
           'implementation_code': _strategy_code(cfg, 'strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'data/constituents.py'), 'reproduce_of': reproduce_of}
    write_json(out / 'config.json', doc)
    limitations = [{'kind': 'strategy_port', 'detail': review['scope'], 'fidelity': review['fidelity'], 'gaps': review['gaps'], 'differences': review['differences']},
                   {'kind': 'evaluation_scope', 'detail': '固定参数工程验证；此区间已查看，不作为独立最终留出，不据此判断策略有效'}]
    subruns = {'backtests': []}; cache = None; final = 'blocked'
    try:
        data, used = _freeze_inputs(store, state, cfg, out); status.stage('load')
        runtime = {k: env[k] for k in ('python', 'packages', 'lock_sha256')}
        if 'rsi_backend' in env: runtime['rsi_backend'] = env['rsi_backend']
        if 'ema_backend' in env: runtime['ema_backend'] = env['ema_backend']
        cache = StageCache(root, {'snapshot': cfg.snapshot, 'used': used, 'runtime': runtime}, cfg.cache and reproduce_of is None)
        payload = {'config': cfg.model_dump(mode = 'json', exclude = {'cache', 'initial_cash', 'execution', 'cost_scenarios', 'rules', 'benchmark', 'name'}),
                   'source': source_meta, 'code': _strategy_code(cfg, 'strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'universe.py', 'data/prices.py', 'data/constituents.py')}
        cache.materialize('rule_signals', payload, out, lambda dest: _generate_signals(cfg, data, dest))
        signal_files = ['source.original', 'config.json', 'universe.parquet', 'factors.parquet', 'scores.parquet', 'targets.parquet', 'signal_coverage.parquet']
        if cfg.implementation == 'svm_lagged_shape_corrected_v1': signal_files += ['svm_models.json', 'svm_samples.parquet']
        if cfg.implementation == 'rsi_slots_corrected_v1': signal_files += ['rsi_backend.json']
        if cfg.implementation == 'ema_slots_talib_approx_v1': signal_files += ['ema_backend.json']
        write_json(out / 'signals_manifest.json', {'files': {name: file_sha(out / name) for name in signal_files}})
        status.stage('signals')
        report_rows, comparisons, active_rows = [], [], []
        for scenario in cfg.cost_scenarios:
            execution = cfg.execution.model_dump()
            if scenario == 'fees_x2': execution['fee_multiplier'] *= 2
            if scenario == 'slippage_x2': execution['slippage'] *= 2
            bc = RunConfig(snapshot = cfg.snapshot, start = cfg.start, end = cfg.end, initial_cash = cfg.initial_cash, boards = cfg.universe.boards,
                           rules = cfg.rules, portfolio = cfg.portfolio, execution = execution, scores = {'source': 'rules', 'run': str(out), 'model': cfg.implementation}, cache = cfg.cache,
                           execution_instruments = cfg.execution_instruments)
            result = _run(root, bc, runs_root = out / 'variants', rules_file = out / 'rules.yaml', tag = scenario, parent_run_id = out.name, force_recompute = reproduce_of is not None)
            subruns['backtests'].append({'model': cfg.implementation, 'scenario': scenario, **reference(result)})
            limitations += result['limitations']; child = Path(result['output']); ok = result['status'] in ('success', 'success_limited')
            row = {'scenario': scenario, 'status': result['status'], 'metrics': _read(child / 'metrics.json') if ok else None,
                   'trading': _read(child / 'trading.json') if ok else None, 'yearly': [], 'blocked': _read(child / 'status.json').get('blocked')}
            if ok:
                equity = _read(child / 'equity.json'); dates = [date.fromisoformat(r['date']) for r in equity]
                if cfg.benchmark_kind == 'stock':
                    levels, index_info = stock_price_levels(data['bars_1d'], data['adj_factors'], data['adj_coverage'], sessions(data['calendar']), dates, cfg.benchmark)
                else:
                    levels, index_info = price_levels(data['index_1d'], sessions(data['calendar']), dates, cfg.benchmark)
                daily, active = benchmark_comparison(equity, cfg.initial_cash, levels, cfg.benchmark, cfg.implementation, scenario)
                active_rows.append(daily); comparisons.append({'scenario': scenario, 'index': index_info, 'metrics': active})
                for year in sorted({r['date'][:4] for r in equity}):
                    indices = [k for k, r in enumerate(equity) if r['date'].startswith(year)]; previous = equity[indices[0] - 1]['equity'] if indices[0] else cfg.initial_cash
                    row['yearly'].append({'year': int(year), 'sessions': len(indices), 'metrics': canonical(metrics([previous, *[equity[k]['equity'] for k in indices]]))})
            report_rows.append(row); status.stage(scenario, **reference(result))
        if active_rows: pd.concat(active_rows, ignore_index = True).to_parquet(out / 'benchmark_daily.parquet', index = False)
        write_json(out / 'benchmark_eval.json', comparisons)
        if cfg.implementation == 'bollinger_breakout_corrected_v1':
            formula = f'close > MA{cfg.parameters.boll_window}+{cfg.parameters.boll_std_multiplier}*STD(ddof=0) 买入；close < 下轨退出；中间保持'
        elif cfg.implementation == 'ma5_ma10_price_v1':
            formula = 'MA5 > MA10 且 close > MA5 买入；close < MA5 退出；其余保持'
        elif cfg.implementation == 'multi_ma_fixed_value_v1':
            formula = 'MA5>MA10>MA20>MA30且不纠缠、或连续两日空头后MA10上穿MA20，实际空仓买固定20000元；空头或连续两日多头后MA5下穿MA10清仓；多头空仓纠缠跳过后续分支；持仓不补买'
        elif cfg.implementation == 'pair_zscore_rotation_corrected_v1':
            formula = '60行动态锚定价差price1-price2，STD(ddof=1)、z四位小数；z<=-2切1、z>=2切2，中间保持信号状态；仅切换时先清另股后用实际现金买入，不随拒单回退状态或补单'
        elif cfg.implementation == 'rsi_slots_corrected_v1':
            formula = '61交易行内重启RSI6取int；15<RSI<25稳定升序买、RSI>85或<10卖；9槽位，开盘先卖后买，每笔按实际现金/剩余槽位分配，已有持仓不调整'
        elif cfg.implementation == 'ema_slots_talib_approx_v1':
            formula = '连续TA-Lib EMA2/25/60；短线上穿中线且中线>长线买、短线下穿中线且中线>长线卖；严格交叉、原八股顺序、5槽位；开盘先卖后买，实际现金/(剩余槽位*1.5)'
        elif cfg.implementation == 'svm_lagged_shape_corrected_v1':
            formula = '252日历史，22日7特征、5日涨跌标签；224样本默认SVC训练，最后滞后5日特征二维predict；周内第三交易日满仓或清仓，不做标准化或特征更新'
        elif cfg.implementation in PAIR_IMPLEMENTATIONS:
            formula = '120日决策日真实价尺度锚定价差 price2-price1；总体标准差z>1买1、z<-1买2；仅反侧跨均值恢复各半仓，其余保持'
        else:
            formula = '1/' + VALUATION_FIELDS[cfg.implementation] if cfg.implementation in VALUATION_FIELDS else f'close > {cfg.parameters.buy_multiplier}*MA{cfg.parameters.short} 买入；close < MA{cfg.parameters.long} 退出'
        write_json(out / 'report.json', {'strategy': source_meta, 'implementation': cfg.implementation, 'parameters': cfg.parameters.model_dump(),
                    'formula': formula,
                    'evidence': 'exploratory', 'period': {'start': str(cfg.start), 'end': str(cfg.end)}, 'universe': cfg.universe.model_dump(), 'portfolio': cfg.portfolio.model_dump(),
                    'decision_time': '收盘数据确认可用后', 'execution_time': '下一交易日开盘', 'results': report_rows, 'benchmark': comparisons,
                    'coverage': canonical(pd.read_parquet(out / 'signal_coverage.parquet').to_dict('records')), 'limitations': limitations})
        final = 'blocked' if any(r['status'] == 'blocked' for r in subruns['backtests']) else 'success_limited'
    except InputBlocked as exc:
        limitations.append({'kind': 'strategy_blocked', 'detail': str(exc), 'issues': exc.issues}); status.stage('signals', 'blocked')
    except Exception as exc:
        write_json(out / 'subruns.json', subruns); status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    if cache is not None: write_json(out / 'cache.json', cache.report())
    write_json(out / 'limitations.json', limitations); write_json(out / 'subruns.json', subruns)
    extra = compare(out, subruns) if compare is not None else {}
    if extra.get('reproduction', {}).get('result') == 'mismatch': extra['execution_status'], final = final, 'mismatch'
    status.finish(final, limitations = limitations, **extra); seal(out, status, env)
    return {'run_id': out.name, 'output': str(out), 'status': final, 'subruns': subruns, **extra}


def reproduce_strategy(root, run, output = None, abs_tol = 1e-9, rel_tol = 0):
    source = resolve_run(root, run); ensure_outside(source, output); verify_manifest(source)
    doc = _read(source / 'config.json'); cfg = StrategyConfig.model_validate(doc['config'])
    check_snapshot(Store(root), cfg.snapshot, _read(source / 'data_manifest.json'))
    if file_sha(source / 'source.original') != doc['source']['bytes_sha256']: raise ReproduceRefused('冻结策略源码指纹不同')
    expected_subs = _read(source / 'subruns.json'); changed = drift(doc['environment'], environment())
    def compare(out, subruns):
        differences, tables = [], {}
        frames = {**FRAMES, **({'svm_samples': ['decision_date', 'feature_date']} if cfg.implementation == 'svm_lagged_shape_corrected_v1' else {})}
        for name, keys in frames.items():
            expected = pd.read_parquet(source / f'{name}.parquet'); actual = pd.read_parquet(out / f'{name}.parquet')
            diff, summary = compare_frames(expected, actual, keys, abs_tol, rel_tol); tables[name] = summary; differences += diff
        # 产物比较包含成本、基准和分年报告；运行编号/父目录不属于经济结果。
        extras = {'svm_lagged_shape_corrected_v1': ['svm_models'], 'rsi_slots_corrected_v1': ['rsi_backend'], 'ema_slots_talib_approx_v1': ['ema_backend']}.get(cfg.implementation, [])
        for key in ('report', 'benchmark_eval', *extras):
            diff, summary = compare_tables({key: _read(source / f'{key}.json')}, {key: _read(out / f'{key}.json')}, abs_tol, rel_tol); tables[key] = summary; differences += diff
        for expected, actual in zip(expected_subs['backtests'], subruns['backtests'], strict = True):
            diff, summary = compare_tables(read_core(resolve_run(root, expected['output'])), read_core(actual['output']), abs_tol, rel_tol)
            tables[expected['scenario']] = summary; differences += [{**d, 'scenario': expected['scenario']} for d in diff]
        comparison = {'result': 'mismatch' if differences else 'match', 'differences': differences, 'tables': tables, 'code_drift': changed,
                      'implementation_drift': doc['implementation_code'] != _strategy_code(cfg, 'strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'data/constituents.py'), 'cache': 'bypassed'}
        write_json(out / 'comparison.json', comparison)
        return {'reproduction': {'of': str(source), 'result': comparison['result'], 'differences': len(differences)}}
    return _strategy(root, cfg, output, source / 'source.original', source / 'rules.yaml', str(source), compare)
