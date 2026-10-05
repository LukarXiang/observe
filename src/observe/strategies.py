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
from .evaluation.benchmarks import benchmark_comparison, price_levels
from .evaluation.portfolio import metrics
from .execution import InputBlocked, load, sessions
from .experiments import reference, seal, verify_manifest
from .features import panel
from .factors.expr import compute
from .replay import RULES, ExecutionConfig, PortfolioConfig, ReproduceRefused, RunConfig, _Strict, _read, _run, check_snapshot, read_core
from .research import UniverseConfig
from .runs import RunStatus, canonical, compare_frames, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha, resolve_run, write_json
from .schedule import rebalance_dates
from .strategy_catalog import BP_SOURCE, MA_SOURCE, REVIEWS, read_source, strategy_id
from .universe import build_universe

IMPLEMENTATIONS = {'bp_component_v1': BP_SOURCE, 'ep_component_v1': BP_SOURCE, 'ma10_ma20_v1': MA_SOURCE,
                   'bp_csi800_weekly_v1': BP_SOURCE, 'ep_csi800_weekly_v1': BP_SOURCE}
VALUATION_FIELDS = {'bp_component_v1': 'pb_mrq', 'ep_component_v1': 'pe_ttm', 'bp_csi800_weekly_v1': 'pb_mrq', 'ep_csi800_weekly_v1': 'pe_ttm'}
INDEX_IMPLEMENTATIONS = {'bp_csi800_weekly_v1', 'ep_csi800_weekly_v1'}
FRAMES = {'universe': ['decision_date', 'instrument'], 'factors': ['date', 'instrument'], 'scores': ['decision_date', 'instrument'], 'targets': ['decision_date', 'instrument']}


class RuleParameters(_Strict):
    top_fraction: float = Field(.1, gt = 0, le = 1)
    positive_only: bool = True
    short: int = Field(10, ge = 1, le = 500)
    long: int = Field(20, ge = 2, le = 500)
    buy_multiplier: float = Field(1.01, gt = 0)


class IndexUniverseConfig(_Strict):
    index: Literal['000300.SH', '000905.SH', '000906.SH']
    policy: Literal['provider_weekly_asof']
    max_age_days: int = Field(7, ge = 0, le = 31)


class StrategyConfig(_Strict):
    name: str = ''
    implementation: Literal['bp_component_v1', 'ep_component_v1', 'ma10_ma20_v1', 'bp_csi800_weekly_v1', 'ep_csi800_weekly_v1']
    source_path: str
    snapshot: str
    start: date
    end: date
    universe: UniverseConfig                    # 策略必须显式声明股票池，不能静默继承现有默认规则
    index_universe: IndexUniverseConfig | None = None
    portfolio: PortfolioConfig                  # 调仓、缓冲、仓位与补单参数同样显式冻结
    parameters: RuleParameters = Field(default_factory = RuleParameters)
    instrument: str = '000333.SZ'
    initial_cash: float = Field(1_000_000, gt = 0)
    rules: str = RULES
    execution: ExecutionConfig = Field(default_factory = lambda: ExecutionConfig(slippage = .001))
    cost_scenarios: list[Literal['base', 'fees_x2', 'slippage_x2']] = Field(default_factory = lambda: ['base', 'fees_x2', 'slippage_x2'], min_length = 1)
    benchmark: str = '000300.SH'
    cache: bool = True

    @model_validator(mode = 'before')
    @classmethod
    def _explicit_universe(cls, value):
        if isinstance(value, dict) and isinstance(value.get('universe'), dict):
            required = {'boards', 'min_listed_sessions', 'exclude_st', 'suspend_window', 'max_suspended', 'liquidity_window', 'min_avg_amount'}
            if required - set(value['universe']): raise ValueError(f'规则策略股票池规则须全部明确：{sorted(required - set(value["universe"]))}')
        return value

    @model_validator(mode = 'after')
    def _validate(self):
        if self.start > self.end: raise ValueError('start 晚于 end')
        if self.portfolio.construction != 'target_weights': raise ValueError('规则策略必须显式选择 target_weights')
        if self.portfolio.buffer or self.portfolio.max_sell is not None: raise ValueError('目标权重策略不接受排名缓冲或换出限制；设置 buffer=0、max_sell=null')
        if len(set(self.cost_scenarios)) != len(self.cost_scenarios) or 'base' not in self.cost_scenarios: raise ValueError('成本情景须不重复且包含 base')
        if 'slippage_x2' in self.cost_scenarios and self.execution.slippage >= .05: raise ValueError('加倍滑点超出执行范围')
        if self.implementation in INDEX_IMPLEMENTATIONS:
            if self.index_universe is None or self.index_universe.index != '000906.SH': raise ValueError('CSI800 实现必须显式声明中证800及周频历史成分口径')
            if 'star' in self.universe.boards and self.universe.min_listed_sessions < 6: raise ValueError('科创板日频版本须排除上市前5个交易日；当前未模拟IPO无涨跌幅期')
        elif self.index_universe is not None: raise ValueError('请为历史成分股票池使用单独登记的 CSI800 实现')
        return self


def _freeze_inputs(store, state, cfg, out):
    warmup = max(cfg.parameters.long, cfg.universe.min_listed_sessions, cfg.universe.suspend_window, cfg.universe.liquidity_window, cfg.execution.liquidity_window) + 1
    data = load(store, state, cfg.start, cfg.end, warmup)
    for table in ('adj_factors', 'adj_coverage', 'index_1d'):
        data[table] = store.load_state(state, table); data['partitions'][table] = state['tables'].get(table, {})
    if cfg.index_universe is not None:
        table = 'index_constituents'; data[table] = store.load_state(state, table); data['partitions'][table] = state['tables'].get(table, {})
    used = {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in ps.items()} for t, ps in data['partitions'].items()}
    write_json(out / 'data_manifest.json', {'snapshot_id': cfg.snapshot, 'used': used, 'tables': state['tables'], 'offline': True,
                                          'valuation_mapping': {'pe_ttm': 'BaoStock peTTM（未证明与聚宽 pe_ratio 完全等价）', 'pb_mrq': 'BaoStock pbMRQ（未证明与聚宽 pb_ratio 完全等价）'},
                                          'availability': '日线收盘数据确认可用后；不读取外部原始财务文件', 'mapping_version': 'daily_valuation_v1'})
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
    if not days or any(d not in set(bars.date) for d in days): raise InputBlocked([{'kind': 'missing_session', 'detail': '规则策略请求区间日线/日历不完整'}])
    available_days = [d for d in cal if bars.date.min() <= d <= bars.date.max()]
    warmup = max(cfg.parameters.long, cfg.universe.min_listed_sessions, cfg.universe.suspend_window, cfg.universe.liquidity_window)
    if sum(d < days[0] for d in available_days) < warmup: raise InputBlocked([{'kind': 'insufficient_history', 'detail': f'规则策略需要 {warmup} 个交易日预热'}])
    uni = build_universe(bars, data['instruments'], available_days, cfg.universe.model_dump(), days[0], days[-1])
    if cfg.index_universe is not None: uni = _apply_index_universe(cfg, data, uni, days)
    if cfg.implementation == 'ma10_ma20_v1': uni = uni[uni.instrument.eq(cfg.instrument)].copy()
    eligible = uni[uni.eligible]; ever = sorted(set(uni.instrument)); view = with_adjusted(bars[bars.instrument.isin(set(ever))], data['adj_factors'], data['adj_coverage'])
    if not ever: raise InputBlocked([{'kind': 'universe_empty', 'detail': '请求区间没有对应证券日线，不以空仓冒充策略复现'}])
    wide = panel(view, available_days, ever)
    mask = eligible.assign(value = True).pivot(index = 'decision_date', columns = 'instrument', values = 'value').reindex(index = available_days, columns = ever).fillna(False).astype(bool)
    params = cfg.parameters; fac, score_rows, targets, coverage = [], [], [], []
    p = cfg.portfolio; decision_days = rebalance_dates(days, cal, p.rebalance_frequency, p.rebalance_every, p.rebalance_session)
    state, held_weights = 0.0, {}
    if cfg.implementation in VALUATION_FIELDS:
        field = VALUATION_FIELDS[cfg.implementation]
        if field not in bars: raise InputBlocked([{'kind': 'valuation_missing', 'detail': f'日线缺少 {field}，不伪造数据'}])
        values = wide[field].where(np.isfinite(wide[field])); wide[field] = values
        factors = compute('1/' + field, wide, eligible = mask)
        for day in days:
            members = set(eligible[eligible.decision_date.eq(day)].instrument)
            row = factors.loc[day].reindex(sorted(members)); valid = row[np.isfinite(row)]
            fac += [{'date': day, 'instrument': i, 'value': None if pd.isna(v) else float(v), 'input_value': None if pd.isna(values.at[day, i]) else float(values.at[day, i])} for i, v in row.items()]
            score_rows += [{'decision_date': day, 'instrument': i, 'score': float(v)} for i, v in valid.items()]
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
    else:
        if cfg.instrument not in ever: raise InputBlocked([{'kind': 'instrument_missing', 'detail': cfg.instrument}])
        close = wide['close_adj'][cfg.instrument]; short = close.rolling(params.short, min_periods = params.short).mean(); long = close.rolling(params.long, min_periods = params.long).mean()
        for day in days:
            price, a, b = close.loc[day], short.loc[day], long.loc[day]
            known = np.isfinite([price, a, b]).all() and bool(mask.at[day, cfg.instrument])
            if known:
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
    if not cfg.scores.run or cfg.portfolio.construction != 'target_weights': raise ValueError('rules 来源需要规则策略目录和 target_weights')
    run = resolve_run(root, cfg.scores.run); doc = _read(run / 'config.json'); freeze = _read(run / 'signals_manifest.json')
    if doc.get('kind') != 'strategy' or doc['snapshot_id'] != snapshot: raise ValueError('规则策略与账本快照不匹配')
    if file_sha(run / 'source.original') != doc['source']['bytes_sha256']: raise ReproduceRefused('规则策略原始源码与登记指纹不符')
    if cfg.scores.model != doc['config']['implementation']: raise ValueError('规则策略实现编号不匹配')
    if any(file_sha(run / name) != sha for name, sha in freeze['files'].items()): raise ReproduceRefused('规则信号冻结产物指纹不同')
    if (cfg.start and cfg.start < date.fromisoformat(doc['config']['start'])) or (cfg.end and cfg.end > date.fromisoformat(doc['config']['end'])): raise ValueError('账本区间超出规则信号定义范围')
    meta = {'source': 'rules', 'run': str(run), 'strategy_id': doc['source']['strategy_id'], 'implementation': doc['config']['implementation'],
            'signals_manifest_sha256': file_sha(run / 'signals_manifest.json'), 'source_sha256': doc['source']['bytes_sha256'], 'evidence': 'exploratory_rule_strategy'}
    return meta, cfg.model_copy(update = {'scores': cfg.scores.model_copy(update = {'run': str(run)}), 'start': cfg.start or date.fromisoformat(doc['config']['start']), 'end': cfg.end or date.fromisoformat(doc['config']['end'])})


def rule_scores(meta, dates):
    run = Path(meta['run']); allowed = set(dates)
    scores = pd.read_parquet(run / 'scores.parquet'); scores['decision_date'] = pd.to_datetime(scores.decision_date).dt.date; scores = scores[scores.decision_date.isin(allowed)]
    universe = pd.read_parquet(run / 'universe.parquet'); universe['decision_date'] = pd.to_datetime(universe.decision_date).dt.date
    eligible = {day: set(universe[universe.eligible & universe.decision_date.eq(day)].instrument) for day in dates}
    frame = pd.read_parquet(run / 'targets.parquet'); frame['decision_date'] = pd.to_datetime(frame.decision_date).dt.date
    targets = {}
    for day, data in frame[frame.decision_date.isin(allowed)].groupby('decision_date'):
        if data.instrument.duplicated().any() or not np.isfinite(data.weight).all() or (data.weight < 0).any() or abs(data.weight.sum() - 1) > 1e-9:
            raise ValueError(f'{day}: 目标权重无效')
        targets[day] = {r.instrument: r.weight for r in data.itertuples() if r.instrument != 'CASH' and r.weight > 0}
    if set(targets) != allowed: raise ValueError('规则策略缺决策日目标，不当作现金状态')
    return scores.to_dict('records'), eligible, targets


def run_strategy(root, output = None, **params): return _strategy(root, StrategyConfig.model_validate(params), output)


def _implementation_review(cfg):
    review = REVIEWS[IMPLEMENTATIONS[cfg.implementation]]
    if cfg.index_universe is None: return review
    return {**review, 'scope': 'BP/EP 组件与中证800周频历史成分近似版本', 'implementations': sorted(INDEX_IMPLEMENTATIONS),
            'gaps': [g for g in review['gaps'] if g != '缺中证800历史成分'] + ['供应商成分按周更新，调整公告和临时调整的日精度未证明'],
            'differences': [f'使用对应日期的沪深300与中证500周频归档并集合成中证800；启用板块{cfg.universe.boards}，上市至少{cfg.universe.min_listed_sessions}交易日',
                            '科创板数量约束通过单独规则集声明；日线开盘成交不模拟盘中订单簿'] + review['differences'][1:]}


def _strategy(root, cfg, output = None, source_file = None, rules_file = None, reproduce_of = None, compare = None):
    store = Store(root); state = store.state(cfg.snapshot); env = environment(); source = Path(source_file or cfg.source_path)
    text, encoding = read_source(source); relative = IMPLEMENTATIONS[cfg.implementation]; review = _implementation_review(cfg)
    source_meta = {'strategy_id': strategy_id(relative), 'path': cfg.source_path, 'canonical_source': relative, 'bytes_sha256': file_sha(source),
                   'content_sha256': hashlib.sha256(text.encode()).hexdigest(), 'encoding': encoding, 'review': review}
    config_hash = digest({'config': cfg.model_dump(mode = 'json'), 'source': source_meta, 'implementation_code': code_version('strategies.py', 'schedule.py', 'data/constituents.py')})
    out = create_run_dir(store.root / 'runs', output, f'{config_hash[:6]}-strategy')
    status = RunStatus(out, out.name, kind = 'strategy', registry = store.root / 'runs', config_hash = config_hash, evidence = 'exploratory', snapshot_id = cfg.snapshot, reproduce_of = reproduce_of)
    (out / 'source.original').write_bytes(source.read_bytes())
    (out / 'rules.yaml').write_bytes(Path(rules_file or cfg.rules).read_bytes())
    doc = {'kind': 'strategy', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': cfg.snapshot, 'batch_id': state['batch_id'], 'source': source_meta, 'environment': env,
           'implementation_code': code_version('strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'data/constituents.py'), 'reproduce_of': reproduce_of}
    write_json(out / 'config.json', doc)
    limitations = [{'kind': 'strategy_port', 'detail': review['scope'], 'fidelity': review['fidelity'], 'gaps': review['gaps'], 'differences': review['differences']},
                   {'kind': 'evaluation_scope', 'detail': '固定参数工程验证；此区间已查看，不作为独立最终留出，不据此判断策略有效'}]
    subruns = {'backtests': []}; cache = None; final = 'blocked'
    try:
        data, used = _freeze_inputs(store, state, cfg, out); status.stage('load')
        cache = StageCache(root, {'snapshot': cfg.snapshot, 'used': used, 'runtime': {k: env[k] for k in ('python', 'packages', 'lock_sha256')}}, cfg.cache and reproduce_of is None)
        payload = {'config': cfg.model_dump(mode = 'json', exclude = {'cache', 'initial_cash', 'execution', 'cost_scenarios', 'rules', 'benchmark', 'name'}),
                   'source': source_meta, 'code': code_version('strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'universe.py', 'data/prices.py', 'data/constituents.py')}
        cache.materialize('rule_signals', payload, out, lambda dest: _generate_signals(cfg, data, dest))
        write_json(out / 'signals_manifest.json', {'files': {name: file_sha(out / name) for name in ['source.original', 'config.json', 'universe.parquet', 'factors.parquet', 'scores.parquet', 'targets.parquet', 'signal_coverage.parquet']}})
        status.stage('signals')
        report_rows, comparisons, active_rows = [], [], []
        for scenario in cfg.cost_scenarios:
            execution = cfg.execution.model_dump()
            if scenario == 'fees_x2': execution['fee_multiplier'] *= 2
            if scenario == 'slippage_x2': execution['slippage'] *= 2
            bc = RunConfig(snapshot = cfg.snapshot, start = cfg.start, end = cfg.end, initial_cash = cfg.initial_cash, boards = cfg.universe.boards,
                           rules = cfg.rules, portfolio = cfg.portfolio, execution = execution, scores = {'source': 'rules', 'run': str(out), 'model': cfg.implementation}, cache = cfg.cache)
            result = _run(root, bc, runs_root = out / 'variants', rules_file = out / 'rules.yaml', tag = scenario, parent_run_id = out.name, force_recompute = reproduce_of is not None)
            subruns['backtests'].append({'model': cfg.implementation, 'scenario': scenario, **reference(result)})
            limitations += result['limitations']; child = Path(result['output']); ok = result['status'] in ('success', 'success_limited')
            row = {'scenario': scenario, 'status': result['status'], 'metrics': _read(child / 'metrics.json') if ok else None,
                   'trading': _read(child / 'trading.json') if ok else None, 'yearly': [], 'blocked': _read(child / 'status.json').get('blocked')}
            if ok:
                equity = _read(child / 'equity.json'); levels, index_info = price_levels(data['index_1d'], sessions(data['calendar']), [date.fromisoformat(r['date']) for r in equity], cfg.benchmark)
                daily, active = benchmark_comparison(equity, cfg.initial_cash, levels, cfg.benchmark, cfg.implementation, scenario)
                active_rows.append(daily); comparisons.append({'scenario': scenario, 'index': index_info, 'metrics': active})
                for year in sorted({r['date'][:4] for r in equity}):
                    indices = [k for k, r in enumerate(equity) if r['date'].startswith(year)]; previous = equity[indices[0] - 1]['equity'] if indices[0] else cfg.initial_cash
                    row['yearly'].append({'year': int(year), 'sessions': len(indices), 'metrics': canonical(metrics([previous, *[equity[k]['equity'] for k in indices]]))})
            report_rows.append(row); status.stage(scenario, **reference(result))
        if active_rows: pd.concat(active_rows, ignore_index = True).to_parquet(out / 'benchmark_daily.parquet', index = False)
        write_json(out / 'benchmark_eval.json', comparisons)
        write_json(out / 'report.json', {'strategy': source_meta, 'implementation': cfg.implementation, 'parameters': cfg.parameters.model_dump(),
                    'formula': '1/' + VALUATION_FIELDS[cfg.implementation] if cfg.implementation in VALUATION_FIELDS else f'close > {cfg.parameters.buy_multiplier}*MA{cfg.parameters.short} 买入；close < MA{cfg.parameters.long} 退出',
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
        for name, keys in FRAMES.items():
            expected = pd.read_parquet(source / f'{name}.parquet'); actual = pd.read_parquet(out / f'{name}.parquet')
            diff, summary = compare_frames(expected, actual, keys, abs_tol, rel_tol); tables[name] = summary; differences += diff
        # 产物比较包含成本、基准和分年报告；运行编号/父目录不属于经济结果。
        for key in ('report', 'benchmark_eval'):
            diff, summary = compare_tables({key: _read(source / f'{key}.json')}, {key: _read(out / f'{key}.json')}, abs_tol, rel_tol); tables[key] = summary; differences += diff
        for expected, actual in zip(expected_subs['backtests'], subruns['backtests'], strict = True):
            diff, summary = compare_tables(read_core(resolve_run(root, expected['output'])), read_core(actual['output']), abs_tol, rel_tol)
            tables[expected['scenario']] = summary; differences += [{**d, 'scenario': expected['scenario']} for d in diff]
        comparison = {'result': 'mismatch' if differences else 'match', 'differences': differences, 'tables': tables, 'code_drift': changed,
                      'implementation_drift': doc['implementation_code'] != code_version('strategies.py', 'schedule.py', 'features.py', 'factors/expr.py', 'data/constituents.py'), 'cache': 'bypassed'}
        write_json(out / 'comparison.json', comparison)
        return {'reproduction': {'of': str(source), 'result': comparison['result'], 'differences': len(differences)}}
    return _strategy(root, cfg, output, source / 'source.original', source / 'rules.yaml', str(source), compare)
