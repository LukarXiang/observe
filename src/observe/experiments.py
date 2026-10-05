"""完整研究实验与组合变体（模块 18 / 19）。
研究只拟合一次；每个模型及成本情景复用冻结预测，并调用唯一的组合与账本循环。
CLI、任务队列和 API 共用这里的函数；评价只读取已保存产物。"""
import hashlib, json, shutil
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, ValidationError, model_validator

from .data.store import Store
from .cache import StageCache, code_version
from .evaluation.research import factor_diagnostics, holdout_start, model_comparisons
from .evaluation.benchmarks import benchmark_comparison, price_levels
from .execution import sessions
from .features import load_factor_set
from .replay import RULES, ExecutionConfig, PortfolioConfig, ReproduceRefused, RunConfig, _Strict, _read, _run, check_snapshot, read_core
from .research import CORE as RESEARCH_CORE, CORE_JSON, ResearchConfig, _research
from .runs import RunStatus, canonical, compare_frames, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha, resolve_run, write_json

OK = ('success', 'success_limited')
MODELS = Literal['single_factor', 'equal_blend', 'ridge', 'lgbm']
SCENARIOS = Literal['base', 'fees_x2', 'slippage_x2']


class BenchmarkConfig(_Strict):
    enabled: bool = True
    index: str = Field('000300.SH', pattern = r'^\d{6}\.(SH|SZ|BJ)$')


class ExperimentConfig(ResearchConfig):
    name: str = Field('', max_length = 120)
    initial_cash: float = Field(1_000_000, gt = 0)
    rules: str = RULES
    portfolio: PortfolioConfig = Field(default_factory = lambda: PortfolioConfig(n = 20, max_weight = 0.1, rebalance_every = 5, buffer = 10, max_sell = 5))
    execution: ExecutionConfig = Field(default_factory = lambda: ExecutionConfig(slippage = 0.001))
    backtest_models: list[MODELS] = Field(default_factory = list)
    cost_scenarios: list[SCENARIOS] = Field(default_factory = lambda: ['base', 'fees_x2', 'slippage_x2'], min_length = 1)
    n_boot: int = Field(1000, ge = 100, le = 20000)
    benchmarks: BenchmarkConfig = Field(default_factory = BenchmarkConfig)

    @model_validator(mode = 'after')
    def _models(self):
        available = ['single_factor', 'equal_blend', 'ridge'] + (['lgbm'] if self.models.lgbm else [])
        if not self.backtest_models: self.backtest_models = available
        if set(self.backtest_models) - set(available): raise ValueError(f'回测模型不在本次研究模型集合内：{sorted(set(self.backtest_models) - set(available))}')
        if len(set(self.backtest_models)) != len(self.backtest_models) or len(set(self.cost_scenarios)) != len(self.cost_scenarios): raise ValueError('模型或成本情景重复')
        if 'base' not in self.cost_scenarios: raise ValueError('成本情景必须包含 base，以便统一比较')
        if 'slippage_x2' in self.cost_scenarios and self.execution.slippage >= 0.05: raise ValueError('滑点加倍后必须仍小于 0.1')
        return self


class VariantConfig(_Strict):
    model: MODELS = 'ridge'
    initial_cash: float | None = Field(None, gt = 0)
    portfolio: PortfolioConfig | None = None
    execution: ExecutionConfig | None = None
    start: date | None = None
    end: date | None = None


class FactorEvalConfig(_Strict):
    run: str = Field(min_length = 1)
    min_names: int | None = Field(None, ge = 2)
    n_boot: int = Field(1000, ge = 100, le = 20000)
    seed: int = Field(20261001, ge = 0, le = 2147483647)
    rebalance_every: int = Field(5, ge = 1)


def experiment_params(file_cfg = None, **cli):
    raw = {**(file_cfg or {}), **{k: v for k, v in cli.items() if v is not None}}; output = raw.pop('output', None)
    try: return {'config': ExperimentConfig.model_validate(raw), 'output': output}
    except ValidationError as exc: raise ValueError(f'experiment 配置不合法：{exc}') from None


def verify_manifest(path):
    path = Path(path); manifest = _read(path / 'manifest.json')
    bad = [n for n, h in manifest['files'].items() if not (path / n).is_file() or file_sha(path / n) != h]
    if bad: raise ReproduceRefused(f'实验冻结产物已改动或缺失：{path}：{bad[:20]}')
    return manifest


def reference(result):
    return {k: result[k] for k in ('run_id', 'output', 'status')} | {'manifest_sha256': file_sha(Path(result['output']) / 'manifest.json')}


def seal(out, status, env):
    write_json(Path(out) / 'manifest.json', {'run_id': out.name, 'kind': status.data['kind'], 'status': status.data['status'], 'environment': env,
               'files': {p.relative_to(out).as_posix(): file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p != out / 'manifest.json'}})


def research_tables(path):
    path = Path(path)
    return {n: pd.read_parquet(path / f'{n}.parquet') for n in ('factors', 'labels', 'split_plan', 'predictions')}


def detailed_evaluation(path, cfg, out):
    t = research_tables(path); fset = load_factor_set(Path(path) / 'factor_set.yaml'); hold = holdout_start(t['split_plan'])
    days = sorted(d for d in set(t['factors'].date) if hold is None or d < hold)
    names = [f['name'] for f in fset['factors']]
    daily, groups, factors = factor_diagnostics(t['factors'], t['labels'], names, days, cfg.label_h, cfg.models.min_names, cfg.n_boot, cfg.models.seed, hold, cfg.portfolio.rebalance_every)
    daily.to_parquet(out / 'factor_daily.parquet', index = False); groups.to_parquet(out / 'factor_groups.parquet', index = False)
    models = model_comparisons(t['predictions'], t['labels'], t['split_plan'], cfg.label_h, cfg.models.min_names, cfg.models.top_n, cfg.n_boot, cfg.models.seed)
    write_json(out / 'factor_diagnostics.json', factors); write_json(out / 'model_comparison.json', models)
    return factors, models


def _portfolio_report(subruns):
    rows = []
    for item in subruns:
        out = Path(item['output']); row = {k: item[k] for k in ('model', 'scenario', 'run_id', 'status')}
        row['valid_performance'] = item['status'] in OK
        row['metrics'] = _read(out / 'metrics.json') if row['valid_performance'] else None
        row['trading'] = _read(out / 'trading.json') if row['valid_performance'] else None
        row['blocked'] = _read(out / 'status.json').get('blocked') or _read(out / 'status.json').get('issues', [])
        rows.append(row)
    return rows


def scenario_execution(cfg, scenario):
    execution = cfg.execution.model_dump()
    if scenario == 'fees_x2': execution['fee_multiplier'] *= 2
    if scenario == 'slippage_x2': execution['slippage'] *= 2
    return execution


def evaluate_benchmarks(root, state, cfg, research, out, subruns, force_recompute = False):
    """每个成本情景各跑一份完整股票池等权账本；评价不改子实验指标。"""
    store = Store(root); calendar = sessions(store.load_state(state, 'calendar')); limitations, comparisons, tables = [], [], []
    used = {n: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in state['tables'].get(n, {}).items()} for n in ('calendar', 'index_1d')}
    write_json(out / 'benchmark_data_manifest.json', {'snapshot_id': cfg.snapshot, 'tables': state['tables'], 'used': used})
    index = store.load_state(state, 'index_1d')
    for scenario in cfg.cost_scenarios:
        portfolio = {**cfg.portfolio.model_dump(), 'construction': 'universe_equal'}
        bc = RunConfig(snapshot = cfg.snapshot, initial_cash = cfg.initial_cash, boards = cfg.universe.boards, rules = cfg.rules, portfolio = portfolio,
                       execution = scenario_execution(cfg, scenario), scores = {'source': 'universe', 'run': research['output']}, cache = cfg.cache)
        r = _run(root, bc, runs_root = out / 'benchmarks', rules_file = out / 'rules.yaml', tag = f'universe-equal-{scenario}', parent_run_id = out.name, force_recompute = force_recompute)
        subruns['benchmarks'].append({'model': 'universe_equal', 'scenario': scenario, **reference(r)}); limitations += r['limitations']
        if r['status'] not in OK: limitations.append({'kind': 'benchmark_blocked', 'scenario': scenario, 'detail': '同股票池等权账本被阻断，该基准的完整绩效与主动比较不可用'})
    equal = {x['scenario']: x for x in subruns['benchmarks']}; metadata = None
    for item in subruns['backtests']:
        if item['status'] not in OK: continue
        rows = _read(Path(item['output']) / 'equity.json'); dates = [pd.Timestamp(r['date']).date() for r in rows]
        levels, meta = price_levels(index, calendar, dates, cfg.benchmarks.index); metadata = meta
        daily, price = benchmark_comparison(rows, cfg.initial_cash, levels, cfg.benchmarks.index, item['model'], item['scenario']); tables.append(daily)
        pool = equal[item['scenario']]; pool_metric = {'available': False, 'reason': '同股票池等权账本被阻断'}
        if pool['status'] in OK:
            eq = _read(Path(pool['output']) / 'equity.json'); by_date = {pd.Timestamp(r['date']).date(): r['equity'] for r in eq}
            # 两个账本都在首日开盘前从相同资金起步；缺失首日时不能把较晚净值当成期初。
            initial = cfg.initial_cash if eq and pd.Timestamp(eq[0]['date']).date() == dates[0] else float('nan')
            levels = [initial, *[by_date.get(d, float('nan')) for d in dates]]
            daily, pool_metric = benchmark_comparison(rows, cfg.initial_cash, levels, 'universe_equal', item['model'], item['scenario']); tables.append(daily)
        comparisons.append({'model': item['model'], 'scenario': item['scenario'], 'benchmarks': {cfg.benchmarks.index: price, 'universe_equal': pool_metric}})
    price_ok = bool(comparisons) and all(x['benchmarks'][cfg.benchmarks.index]['benchmark_missing_days'] == 0 for x in comparisons)
    if not price_ok: limitations.append({'kind': 'price_benchmark_missing', 'detail': '指定快照中的价格指数缺失或不完整；保留全部策略日收益，主动指标只使用相邻两日指数均有效的日期', 'index': cfg.benchmarks.index})
    result = {'available': any(x['benchmarks'][k]['available'] for x in comparisons for k in x['benchmarks']), 'price_index': metadata,
              'universe_equal': {'description': '同一历史研究候选完整等权、同调仓节奏、同成本情景，使用唯一账本；整手、现金、成交限制导致实际权重偏离目标',
                                 'ignored_topn_fields': ['n', 'buffer', 'max_sell']}, 'comparisons': comparisons}
    write_json(out / 'benchmark_eval.json', result)
    daily = pd.concat(tables, ignore_index = True) if tables else pd.DataFrame(columns = ['date', 'model_id', 'scenario', 'benchmark_id', 'strategy_return', 'benchmark_return', 'active_return', 'relative_nav'])
    daily['date'] = pd.to_datetime(daily.date).dt.date; daily.to_parquet(out / 'benchmark_daily.parquet', index = False)
    return result, limitations


def run_experiment(root, output = None, runs_root = None, **params):
    p = experiment_params(params); return _experiment(root, p['config'], output or p['output'], runs_root)


def _experiment(root, cfg, output = None, runs_root = None, factor_file = None, rules_file = None, reproduce_of = None, compare = None):
    state = Store(root).state(cfg.snapshot); env = environment(); frozen_rules = Path(rules_file or cfg.rules)
    if not frozen_rules.is_file(): raise FileNotFoundError(f'执行规则不存在：{frozen_rules}')
    doc = {'kind': 'experiment', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': cfg.snapshot, 'batch_id': state['batch_id'], 'environment': env,
           'factor_set_sha256': file_sha(factor_file or cfg.factor_set), 'rules_sha256': file_sha(frozen_rules), 'reproduce_of': reproduce_of}
    fingerprint = hashlib.sha256(json.dumps(canonical(doc['config']), sort_keys = True).encode()).hexdigest()
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, fingerprint[:6] + '-experiment')
    status = RunStatus(out, out.name, kind = 'experiment', registry = Path(root) / 'runs', config_hash = fingerprint, snapshot_id = cfg.snapshot,
                       evidence = 'exploratory' if cfg.minute_pool else 'development_oos')
    write_json(out / 'config.json', doc); shutil.copyfile(frozen_rules, out / 'rules.yaml'); status.stage('config')
    subruns, limitations, report, cache = {'research': None, 'backtests': [], 'benchmarks': []}, [], None, None
    try:
        rc = ResearchConfig(**{k: getattr(cfg, k) for k in ResearchConfig.model_fields})
        research = _research(root, rc, runs_root = out / 'research', factor_file = factor_file, parent_run_id = out.name, force_recompute = reproduce_of is not None)
        subruns['research'] = reference(research); status.stage('research', **subruns['research'])
        limitations += research['limitations']
        final = research['status']
        if final in OK:
            rp = Path(research['output']); used = _read(rp / 'data_manifest.json')['used']
            cache = StageCache(root, {'snapshot': cfg.snapshot, 'used': used, 'code': code_version('cache.py'),
                                     'runtime': {k: env[k] for k in ('python', 'packages', 'lock_sha256')}}, cfg.cache and reproduce_of is None)
            payload = {'upstream': {n: file_sha(rp / f'{n}.parquet') for n in ('factors', 'labels', 'split_plan', 'predictions')},
                       'config': {'h': cfg.label_h, 'models': cfg.models.model_dump(), 'n_boot': cfg.n_boot, 'rebalance_every': cfg.portfolio.rebalance_every},
                       'code': code_version('evaluation/research.py', 'evaluation/paired.py', 'evaluation/ranking.py', 'dataset.py', 'experiments.py')}
            cache.materialize('diagnostics', payload, out, lambda dest: detailed_evaluation(research['output'], cfg, dest))
            factors, models = _read(out / 'factor_diagnostics.json'), _read(out / 'model_comparison.json'); status.stage('diagnostics')
            for model in cfg.backtest_models:
                for scenario in cfg.cost_scenarios:
                    execution = scenario_execution(cfg, scenario)
                    if scenario == 'slippage_x2' and cfg.execution.slippage == 0: limitations.append({'kind': 'zero_slippage_sensitivity', 'detail': '基准滑点为 0，滑点加倍仍为 0，此情景没有增加执行压力'})
                    bc = RunConfig(snapshot = cfg.snapshot, initial_cash = cfg.initial_cash, boards = cfg.universe.boards, rules = cfg.rules, portfolio = cfg.portfolio,
                                   execution = execution, scores = {'source': 'predictions', 'run': research['output'], 'model': model}, cache = cfg.cache)
                    r = _run(root, bc, runs_root = out / 'variants', rules_file = out / 'rules.yaml', tag = f'{model}-{scenario}', parent_run_id = out.name, force_recompute = reproduce_of is not None)
                    subruns['backtests'].append({'model': model, 'scenario': scenario, **reference(r)}); limitations += r['limitations']
                    status.stage(f'{model}_{scenario}', **reference(r))
            blocked = [x for x in subruns['backtests'] if x['status'] not in OK]
            if blocked: limitations.append({'kind': 'portfolio_blocked', 'detail': '部分账本无法处理持仓或输入，相关组合绩效不进入正式比较', 'runs': [{k: x[k] for k in ('model', 'scenario', 'status')} for x in blocked]})
            benchmark = {'available': False, 'reason': '本实验配置禁用基准'}
            if cfg.benchmarks.enabled:
                benchmark, extra_limits = evaluate_benchmarks(root, state, cfg, research, out, subruns, reproduce_of is not None); limitations += extra_limits
                status.stage('benchmarks')
            final = 'blocked' if blocked else ('success_limited' if limitations else 'success')
            summary = research['summary']
            report = {'header': {'snapshot_id': cfg.snapshot, 'batch_id': state['batch_id'], 'period': summary, 'universe': cfg.universe.model_dump(),
                       'label': {'name': 'adj_open_to_open_h', 'h': cfg.label_h}, 'decision_time': '收盘数据确认可用后', 'execution_time': '下一交易日开盘，先卖后买',
                       'execution': cfg.execution.model_dump(), 'portfolio': cfg.portfolio.model_dump(), 'evidence': 'exploratory' if cfg.minute_pool or limitations else 'development_oos',
                       'approximation': '日频开盘参考价近似；收益已扣交易费用，未扣个人股息红利所得税', 'holdout': '最终留出不生成预测、不用于因子挑选、参数与组合选择'},
                      'factor': {'evaluation': _read(Path(research['output']) / 'factor_eval.json'), 'diagnostics': factors},
                      'model': {'evaluation': _read(Path(research['output']) / 'model_eval.json'), **models}, 'portfolio': _portfolio_report(subruns['backtests']),
                      'limitations': limitations, 'benchmark': benchmark}
            write_json(out / 'report.json', report); status.stage('report')
        else: status.stage('backtests', 'not_run', reason = '研究阶段未完成，不用部分预测进入账本')
    except Exception as exc:
        if cache is not None: write_json(out / 'cache.json', cache.report())
        write_json(out / 'subruns.json', subruns); status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    if cache is not None: write_json(out / 'cache.json', cache.report())
    write_json(out / 'subruns.json', subruns); write_json(out / 'limitations.json', limitations)
    extra = compare(out, subruns) if compare is not None else {}
    if extra.get('reproduction', {}).get('result') == 'mismatch': extra['execution_status'], final = final, 'mismatch'
    status.finish(final, subruns = subruns, limitations = limitations, evidence = report['header']['evidence'] if report else status.data['evidence'], **extra); seal(out, status, env)
    return {'run_id': out.name, 'output': str(out), 'status': final, 'subruns': subruns, 'limitations': limitations, **extra}


def run_variant(root, parent, config = None, output = None):
    """只允许组合 / 执行参数变化，继承固定快照、候选、预测和规则；子目录不改父实验任何已有文件。"""
    source = resolve_run(root, parent); doc, st = _read(source / 'config.json'), _read(source / 'status.json')
    if st['status'] not in OK + ('blocked',): raise ValueError(f'父实验状态 {st["status"]} 不能派生组合变体')
    verify_manifest(source)
    if doc.get('kind') == 'experiment':
        research = resolve_run(root, _read(source / 'subruns.json')['research']['output']); defaults = doc['config']; rules_file = source / 'rules.yaml'
    elif doc.get('kind') == 'research': research, defaults, rules_file = source, {}, Path(RULES)
    else: raise ValueError('组合变体的父实验必须是 research 或 experiment')
    if _read(research / 'status.json')['status'] not in OK: raise ValueError('父实验的研究阶段没有完整预测')
    patch = VariantConfig.model_validate(config or {}); rc = _read(research / 'config.json')['config']
    portfolio = {**defaults.get('portfolio', {}), **(patch.portfolio.model_dump(exclude_unset = True) if patch.portfolio else {})}
    execution = {**defaults.get('execution', {}), **(patch.execution.model_dump(exclude_unset = True) if patch.execution else {})}
    cfg = RunConfig(snapshot = doc['snapshot_id'], initial_cash = patch.initial_cash or defaults.get('initial_cash', 1_000_000), boards = rc['universe']['boards'],
                    rules = str(rules_file), portfolio = portfolio, execution = execution, start = patch.start, end = patch.end,
                    scores = {'source': 'predictions', 'run': str(research), 'model': patch.model})
    result = _run(root, cfg, output, source / 'variants', rules_file = rules_file, tag = 'variant', parent_run_id = source.name)
    return {**result, 'parent_run_id': source.name, 'training_reused': True}


def run_factor_eval(root, output = None, **params):
    cfg = FactorEvalConfig.model_validate(params); source = resolve_run(root, cfg.run); verify_manifest(source)
    if _read(source / 'config.json').get('kind') == 'experiment': source = resolve_run(root, _read(source / 'subruns.json')['research']['output'])
    doc = _read(source / 'config.json')
    if doc.get('kind') != 'research' or _read(source / 'status.json')['status'] not in OK: raise ValueError('因子评价需要已完成 research 实验中的因子、标签与切分')
    verify_manifest(source); rc = ResearchConfig.model_validate(doc['config']); fs = load_factor_set(source / 'factor_set.yaml')
    plan = pd.read_parquet(source / 'split_plan.parquet'); hold = holdout_start(plan); env = environment()
    out = create_run_dir(Path(root) / 'runs', output, 'factor-eval'); status = RunStatus(out, out.name, kind = 'factor_eval', registry = Path(root) / 'runs', parent_run_id = source.name)
    write_json(out / 'config.json', {'kind': 'factor_eval', 'config': cfg.model_dump(mode = 'json'), 'snapshot_id': doc['snapshot_id'], 'environment': env,
                                    'source': {'run_id': source.name, 'manifest_sha256': file_sha(source / 'manifest.json')}})
    try:
        fac, lab = (pd.read_parquet(source / f'{n}.parquet') for n in ('factors', 'labels')); days = sorted(d for d in set(fac.date) if hold is None or d < hold)
        daily, groups, result = factor_diagnostics(fac, lab, [f['name'] for f in fs['factors']], days, rc.label_h, cfg.min_names or rc.models.min_names,
                                                   cfg.n_boot, cfg.seed, hold, cfg.rebalance_every)
        daily.to_parquet(out / 'factor_daily.parquet', index = False); groups.to_parquet(out / 'factor_groups.parquet', index = False); write_json(out / 'factor_diagnostics.json', result)
        limitations = _read(source / 'limitations.json'); final = 'success_limited' if limitations else 'success'
        write_json(out / 'limitations.json', limitations); status.finish(final, limitations = limitations); seal(out, status, env)
    except Exception as exc: status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    return {'run_id': out.name, 'output': str(out), 'status': final, 'parent_run_id': source.name, 'factors': len(fs['factors'])}


def reproduce_experiment(root, run, output = None, abs_tol = 1e-9, rel_tol = 0.0):
    source = resolve_run(root, run); ensure_outside(source, output); verify_manifest(source)
    doc, subs = _read(source / 'config.json'), _read(source / 'subruns.json'); raw = dict(doc['config']); raw.setdefault('benchmarks', {'enabled': False}); cfg = ExperimentConfig.model_validate(raw)
    research = resolve_run(root, subs['research']['output']); verify_manifest(research)
    check_snapshot(Store(root), cfg.snapshot, _read(research / 'data_manifest.json'))
    if (source / 'benchmark_data_manifest.json').exists(): check_snapshot(Store(root), cfg.snapshot, _read(source / 'benchmark_data_manifest.json'))
    frozen = research / 'factor_set.yaml'
    if file_sha(frozen) != doc['factor_set_sha256'] or file_sha(source / 'rules.yaml') != doc['rules_sha256']: raise ReproduceRefused('冻结因子集或执行规则不一致')

    def compare(out, now):
        summaries, diffs = {}, []
        a, b = research, Path(now['research']['output'])
        for n, keys in RESEARCH_CORE.items():
            af, bf = a / f'{n}.parquet', b / f'{n}.parquet'
            if af.exists() and bf.exists():
                d, s = compare_frames(pd.read_parquet(af), pd.read_parquet(bf), keys, abs_tol, rel_tol); diffs += [{'table': n, **x} for x in d]; summaries[n] = s
            else: summaries[n] = {'differences': int(af.exists() != bf.exists())}
        expected_json = {n: _read(a / f'{n}.json') if (a / f'{n}.json').exists() else None for n in CORE_JSON}
        actual_json = {n: _read(b / f'{n}.json') if (b / f'{n}.json').exists() else None for n in CORE_JSON}
        d, s = compare_tables(expected_json, actual_json, abs_tol, rel_tol); diffs += d; summaries.update(s)
        old = {(x['model'], x['scenario']): x for x in subs['backtests'] + subs.get('benchmarks', [])}; new = {(x['model'], x['scenario']): x for x in now['backtests'] + now.get('benchmarks', [])}
        summaries['backtest_set'] = {'differences': int(old.keys() != new.keys())}
        for key in old.keys() & new.keys():
            d, s = compare_tables(read_core(resolve_run(root, old[key]['output'])), read_core(new[key]['output']), abs_tol, rel_tol)
            label = '/'.join(key); diffs += [{'variant': label, **x} for x in d]; summaries[label] = {'differences': sum(v['differences'] for v in s.values()), 'tables': s}
            if old[key]['status'] != new[key]['status']: summaries[label]['differences'] += 1
        for n in ('factor_diagnostics', 'model_comparison', 'benchmark_eval'):
            old_file, new_file = source / f'{n}.json', out / f'{n}.json'
            d, s = compare_tables({n: _read(old_file) if old_file.exists() else None}, {n: _read(new_file) if new_file.exists() else None}, abs_tol, rel_tol)
            diffs += d; summaries.update(s)
        for n, keys in (('factor_daily', ('factor', 'date')), ('factor_groups', ('factor', 'date', 'quantile')), ('benchmark_daily', ('model_id', 'scenario', 'benchmark_id', 'date'))):
            af, bf = source / f'{n}.parquet', out / f'{n}.parquet'
            if af.exists() and bf.exists():
                d, s = compare_frames(pd.read_parquet(af), pd.read_parquet(bf), keys, abs_tol, rel_tol); diffs += [{'table': n, **x} for x in d]; summaries[n] = s
            else: summaries[n] = {'differences': int(af.exists() != bf.exists())}
        result = 'match' if not any(s['differences'] for s in summaries.values()) else 'mismatch'
        write_json(out / 'comparison.json', {'source': str(source), 'result': result, 'tolerance': {'abs': abs_tol, 'rel': rel_tol}, 'tables': summaries, 'differences': diffs[:200],
                                            'code_drift': drift(doc['environment'], environment())})
        return {'reproduction': {'of': str(source), 'result': result, 'differences': sum(s['differences'] for s in summaries.values())}}

    return _experiment(root, cfg, output, source.parent, factor_file = frozen, rules_file = source / 'rules.yaml', reproduce_of = str(source), compare = compare)
