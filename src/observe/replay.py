"""离线回放：固定快照 → 执行输入适配器 → 逐日循环（组合 + 账本）→ 评价。
产物写入新建的实验目录；目录一建立就写 running 状态，结束时提交终态，manifest.json 最后写入。

冻结输入：config.json 保存展开默认值后的完整配置、快照与分区清单、规则内容指纹、代码与依赖版本；rules.yaml 是实际使用的规则原文。
复现只用冻结输入：数据或规则对不上就拒绝，代码版本不同照常比较并报告差异；源目录只读。"""
import hashlib, json
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .data.audit import audit_status
from .data.store import Store
from .evaluation import metrics as evaluate, trading_stats
from .execution import InputBlocked, build, load
from .ledger import LedgerError, RuleSet
from .loop import POLICIES, run_loop
from .portfolio import DEFAULT_PARTICIPATION
from .cache import StageCache, code_version, digest
from .runs import (KEYS, STATUS_FIELDS, RunStatus, canonical, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha,
                   write_json, write_table, resolve_run)

RULES = 'configs/rule_profiles/main_board.yaml'
BASELINE = {'source': 'baseline', 'name': 'lexicographic_engineering_baseline_v1', 'evidence': 'engineering_baseline_not_a_prediction_model',
            'detail': '按证券代码字典序给研究候选打分，只用于验证执行链路，不是预测模型'}
CORE = ('scores', 'orders', 'fills', 'cash_events', 'equity', 'positions_daily', 'receivables', 'metrics', 'trading', 'limitations')


# 配置 -------------------------------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra = 'forbid', allow_inf_nan = False)


class PortfolioConfig(_Strict):
    construction: Literal['topn', 'universe_equal', 'target_weights'] = 'topn'
    n: int = Field(1, ge = 1)
    max_weight: float = Field(1.0, gt = 0, le = 1)
    rebalance_every: int = Field(1, ge = 1)
    rebalance_frequency: Literal['sessions', 'daily', 'weekly', 'monthly'] = 'sessions'
    rebalance_session: int = Field(1, ge = 1)
    buffer: int = Field(10, ge = 0)
    max_sell: int | None = Field(5, ge = 0)
    participation: float = Field(DEFAULT_PARTICIPATION, gt = 0, le = 1)
    open_cash_policy: Literal[POLICIES] = 'sell_then_buy'
    refill_between_rebalance: bool = True


class ExecutionConfig(_Strict):
    slippage: float = Field(0.0, ge = 0, lt = 0.1)
    liquidity_window: int = Field(20, ge = 1)
    liquidity_override: float | None = Field(None, gt = 0)       # 只给隔离的合成测试用；设定后运行记为 success_limited
    fee_multiplier: float = Field(1.0, gt = 0)                   # 成本情景：佣金、印花税、过户费与最低佣金同乘；冻结的规则原文不变


class ScoresConfig(_Strict):
    source: Literal['baseline', 'predictions', 'universe', 'rules'] = 'baseline'
    run: str | None = None                                       # predictions：研究实验目录（observe research 的产物）
    model: str = 'ridge'
    allow_cross_snapshot: bool = False                           # 研究实验的快照与回放快照不同时默认拒绝；确需用新快照评价旧预测，显式设为 true，记为对照情景并记录双方版本


class RunConfig(_Strict):
    snapshot: str = Field(min_length = 1)
    start: date | None = None
    end: date | None = None
    initial_cash: float = Field(100000.0, gt = 0)
    boards: list[Literal['main', 'gem', 'star', 'bse']] = Field(default_factory = lambda: ['main'], min_length = 1)
    rules: str = RULES
    portfolio: PortfolioConfig = Field(default_factory = PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory = ExecutionConfig)
    scores: ScoresConfig = Field(default_factory = ScoresConfig)
    cache: bool = True


ALIASES = {'snapshot_id': 'snapshot', 'cash': 'initial_cash'}


def run_params(file_cfg = None, **cli):
    """显式命令行参数 > YAML / 队列参数 > 默认值。别名规范化；同一字段的两种写法同时出现、未知字段、非法数值都报错。
    返回 {'output': ..., 'config': RunConfig}"""
    raw = dict(file_cfg or {})
    if not isinstance(raw, dict): raise ValueError('run 配置必须是对象')
    for alias, name in ALIASES.items():
        if alias in raw:
            if name in raw: raise ValueError(f'配置同时给出 {alias} 与 {name}，只能保留一个')
            raw[name] = raw.pop(alias)
    raw.update({k: v for k, v in cli.items() if v is not None})
    output = raw.pop('output', None)
    try: cfg = RunConfig.model_validate(raw)
    except ValidationError as exc: raise ValueError(f'run 配置不合法：{exc}') from None
    if cfg.start and cfg.end and cfg.start > cfg.end: raise ValueError(f'start {cfg.start} 晚于 end {cfg.end}')
    return {'output': output, 'config': cfg}


# 运行 -------------------------------------------------------------------------------------------------
def baseline_scores(candidates):
    """确定性工程基线：研究候选按代码字典序，越靠前分数越高。长表 decision_date / instrument / score"""
    return [{'decision_date': d, 'instrument': i, 'score': float(-k)} for d in sorted(candidates) for k, i in enumerate(sorted(candidates[d]))]


class _Recorder:
    """逐日收盘后记录持仓与应收分红；下一交易日持仓遇到未记录的除权除息时中止（账本无法正确处理）"""
    def __init__(self, dates, unexplained):
        self.next = dict(zip(dates, dates[1:])); self.unexplained = unexplained; self.book = None; self.positions, self.receivables = [], []

    def __call__(self, day, book, row):
        self.book = book
        for i, p in sorted(book.positions.items()):
            if p.qty: self.positions.append({'date': day, 'instrument': i, 'qty': p.qty, 'sellable': p.sellable, 'pending': p.pending, 'cost': p.cost, 'last_price': p.last_price, 'stale': p.stale})
        self.receivables += [{'date': day, 'pay_date': pay, 'amount': round(v, 2)} for pay, v in sorted(book.receivable.items())]
        nxt = self.next.get(day)
        hit = sorted(i for i, p in book.positions.items() if p.qty and (nxt, i) in self.unexplained)
        if hit: raise LedgerError(f'unrecorded corporate action on {nxt}: {hit} 前收盘被重设，但快照没有对应的公司行动')

    def tables(self):
        b = self.book
        return {'fills': b.fills if b else [], 'cash_events': b.cash_events if b else [], 'equity': b.equity_rows if b else [],
                'positions_daily': self.positions, 'receivables': self.receivables}


def _hash(x): return hashlib.sha256(json.dumps(canonical(x), sort_keys = True).encode()).hexdigest()


def score_source(cfg, snapshot_id = None, root = None):
    """分数来源：工程基线，或研究实验的样本外预测表（同时带来研究候选）。预测表与股票池按文件哈希冻结。
    predictions 时先检查身份：研究实验与回放的快照一致（否则须显式声明对照情景）、模型存在、预测都落在该实验的研究候选上、fit_asof 早于决策日、请求区间不超出预测的定义范围"""
    sc = cfg.scores
    if cfg.portfolio.construction == 'target_weights' and sc.source != 'rules': raise ValueError('target_weights 必须显式使用 rules 来源')
    if sc.source == 'rules':
        from .strategies import rule_source
        return rule_source(root, cfg, snapshot_id)
    if cfg.portfolio.construction == 'universe_equal' and sc.source != 'universe': raise ValueError('universe_equal 必须显式使用 universe 分数来源')
    if sc.source == 'baseline': return BASELINE, cfg
    if sc.source == 'universe': return universe_source(cfg, snapshot_id, root)
    if not sc.run: raise ValueError('scores.source = predictions 需要 scores.run（研究实验目录）')
    run = resolve_run(root, sc.run) if root is not None else Path(sc.run); st = json.loads((run / 'status.json').read_text(encoding = 'utf-8')); rdoc = json.loads((run / 'config.json').read_text(encoding = 'utf-8'))
    if st.get('kind') != 'research' or st.get('status') not in ('success', 'success_limited'): raise ValueError(f'{run} 不是已完成的研究实验（状态 {st.get("status")}）')
    pred = pd.read_parquet(run / 'predictions.parquet', columns = ['model_id', 'decision_date', 'instrument', 'fit_asof', 'score', 'split_id']); pred = pred[pred.model_id == sc.model]
    if not len(pred): raise ValueError(f'研究实验 {run.name} 没有模型 {sc.model} 的预测')
    if pred.duplicated(['decision_date', 'instrument']).any(): raise ValueError('预测主键重复，不能用于账本')
    if not np.isfinite(pred.score.to_numpy(float)).all(): raise ValueError('预测包含非有限分数，不能用于账本')
    day = pd.to_datetime(pred.decision_date)
    if day.isna().any() or pd.to_datetime(pred.fit_asof).isna().any(): raise ValueError('预测 decision_date / fit_asof 缺失')
    if (pd.to_datetime(pred.fit_asof) >= day).any(): raise ValueError(f'研究实验 {run.name} 的模型 {sc.model} 有 fit_asof 不早于决策日的预测，拒绝使用')
    plan = pd.read_parquet(run / 'split_plan.parquet'); joined = pred.merge(plan[['split_id', 'fit_asof', 'test_start', 'test_end']], on = 'split_id', how = 'left', suffixes = ('', '_plan'), validate = 'many_to_one')
    d, fit, planned = pd.to_datetime(joined.decision_date), pd.to_datetime(joined.fit_asof), pd.to_datetime(joined.fit_asof_plan)
    if planned.isna().any() or (fit != planned).any() or ((d < pd.to_datetime(joined.test_start)) | (d > pd.to_datetime(joined.test_end))).any():
        raise ValueError('预测 split_id / fit_asof / 决策日与冻结测试窗不一致')
    uni = pd.read_parquet(run / 'universe.parquet', columns = ['decision_date', 'instrument', 'eligible']); uni = uni[uni.eligible]
    off = pred[['decision_date', 'instrument']].merge(uni[['decision_date', 'instrument']], how = 'left', indicator = True)._merge.eq('left_only').sum()
    if off: raise ValueError(f'研究实验 {run.name} 有 {int(off)} 条预测不在该实验的研究候选里，预测与候选不对应')
    first, last = day.min().date(), day.max().date()
    if (cfg.start and cfg.start < first) or (cfg.end and cfg.end > last): raise ValueError(f'请求区间 {cfg.start}..{cfg.end} 超出该模型预测的定义范围 {first}..{last}')
    meta = {'source': 'predictions', 'run': str(run.resolve()), 'research_run_id': run.name, 'model': sc.model, 'evidence': f"{st.get('evidence') or 'development_oos'}_prediction",
            'predictions_sha256': file_sha(run / 'predictions.parquet'), 'universe_sha256': file_sha(run / 'universe.parquet'), 'research_snapshot': rdoc.get('snapshot_id')}
    if snapshot_id is not None and rdoc.get('snapshot_id') != snapshot_id:
        if not sc.allow_cross_snapshot: raise ValueError(f"研究实验的快照 {rdoc.get('snapshot_id')} 与回放快照 {snapshot_id} 不一致；确需用新快照评价旧预测，请显式设置 scores.allow_cross_snapshot 并按对照情景解读")
        meta['cross_snapshot_scenario'] = {'research_snapshot': rdoc.get('snapshot_id'), 'replay_snapshot': snapshot_id}
    return meta, cfg.model_copy(update = {'start': cfg.start or first, 'end': cfg.end or last})


def universe_source(cfg, snapshot_id, root):
    """完整测试区间的历史研究候选；不读取模型分数或标签，不按预测缺失过滤。"""
    if not cfg.scores.run or cfg.portfolio.construction != 'universe_equal': raise ValueError('universe 来源需要研究目录和 universe_equal 组合')
    run = resolve_run(root, cfg.scores.run); st, doc = _read(run / 'status.json'), _read(run / 'config.json')
    if st.get('kind') != 'research' or st.get('status') not in ('success', 'success_limited'): raise ValueError('等权基准需要已完成的研究实验')
    if doc['snapshot_id'] != snapshot_id: raise ValueError('等权基准必须使用研究实验的同一快照')
    uni = pd.read_parquet(run / 'universe.parquet'); plan = pd.read_parquet(run / 'split_plan.parquet')
    if uni.duplicated(['decision_date', 'instrument']).any() or not len(plan): raise ValueError('等权基准的候选主键重复或切分为空')
    first, last = pd.Timestamp(plan.test_start.min()).date(), pd.Timestamp(plan.test_end.max()).date()
    if (cfg.start and cfg.start < first) or (cfg.end and cfg.end > last): raise ValueError('等权基准不能超出冻结测试区间')
    meta = {'source': 'universe', 'run': str(run.resolve()), 'research_run_id': run.name, 'evidence': f"{st.get('evidence') or 'development_oos'}_universe",
            'universe_sha256': file_sha(run / 'universe.parquet'), 'split_plan_sha256': file_sha(run / 'split_plan.parquet'), 'research_snapshot': doc['snapshot_id']}
    return meta, cfg.model_copy(update = {'start': cfg.start or first, 'end': cfg.end or last})


def _universe_scores(meta, dates):
    u = pd.read_parquet(Path(meta['run']) / 'universe.parquet'); u['decision_date'] = pd.to_datetime(u.decision_date).dt.date
    eligible = {d: set() for d in dates}
    for row in u[u.eligible & u.decision_date.isin(set(dates))].itertuples(): eligible[row.decision_date].add(row.instrument)
    return [{'decision_date': d, 'instrument': i, 'score': 0.0} for d in dates for i in sorted(eligible[d])], eligible


def _prediction_scores(meta, dates):
    run, keep = Path(meta['run']), set(dates)
    p = pd.read_parquet(run / 'predictions.parquet'); p = p[(p.model_id == meta['model'])]; p['decision_date'] = pd.to_datetime(p.decision_date).dt.date
    u = pd.read_parquet(run / 'universe.parquet', columns = ['decision_date', 'instrument', 'eligible']); u['decision_date'] = pd.to_datetime(u.decision_date).dt.date
    p = p[p.decision_date.isin(keep)].sort_values(['decision_date', 'instrument'])
    scores = [{'decision_date': d, 'instrument': i, 'score': float(v)} for d, i, v in zip(p.decision_date, p.instrument, p.score)]
    eligible = {d: set() for d in dates}
    for d, i in zip(*u.loc[u.eligible & u.decision_date.isin(keep), ['decision_date', 'instrument']].to_numpy().T): eligible[d].add(i)
    return scores, eligible


def run_offline(root, output = None, runs_root = None, **params):
    """公开入口：params 与 YAML 同构（snapshot、start、end、initial_cash、boards、rules、portfolio、execution）"""
    p = run_params(params); return _run(root, p['config'], output or p['output'], runs_root)


def _run(root, cfg, output = None, runs_root = None, rules_file = None, tag = None, reproduce_of = None, compare = None, parent_run_id = None, force_recompute = False):
    store = Store(root); state = store.state(cfg.snapshot)                       # 快照不存在时在建目录之前就报错
    source, cfg = score_source(cfg, state.get('snapshot_id') or cfg.snapshot, root)
    rules_file = Path(rules_file or cfg.rules); rules_text = rules_file.read_text(encoding = 'utf-8'); rules = RuleSet.from_yaml(rules_file)
    doc = {'config': cfg.model_dump(mode = 'json'), 'snapshot_id': state.get('snapshot_id') or cfg.snapshot, 'batch_id': state.get('batch_id'),
           'scores': source, 'rules': {'source_path': str(cfg.rules), 'sha256': hashlib.sha256(rules_text.encode()).hexdigest(), 'fingerprint': rules.config_fingerprint()},
           'environment': environment(), 'reproduce_of': reproduce_of}
    h = _hash({k: doc[k] for k in ('config', 'snapshot_id', 'scores', 'rules')})
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, '-'.join(x for x in (h[:6], tag) if x))
    status = RunStatus(out, out.name, registry = Path(root) / 'runs', evidence = source['evidence'], config_hash = h, reproduce_of = reproduce_of, parent_run_id = parent_run_id)
    (out / 'rules.yaml').write_text(rules_text, encoding = 'utf-8'); write_json(out / 'config.json', doc); status.stage('config')
    cache, limitations, x = None, [], cfg.execution
    if 'cross_snapshot_scenario' in source: limitations.append({'kind': 'cross_snapshot_scenario', 'detail': '对照情景：用与预测不同的快照做回放评价', **source['cross_snapshot_scenario']})
    audit_rules = rules
    if x.fee_multiplier != 1: m = x.fee_multiplier; rules = rules.scaled(commission_rate = m, stamp_tax = m, transfer_fee = m, min_commission = m)
    try:
        tables = load(store, state, cfg.start, cfg.end, x.liquidity_window)
        audit = {k: v for k, v in audit_status(root, doc['batch_id'], audit_rules).items() if k != 'rows'}   # 成本情景不改变市场数据审计所依据的原始规则
        if audit['status'] != 'passed':
            limitations.append({'kind': 'data_audit', 'detail': f"快照批次审计状态为 {audit['status']}（范围 {audit['scope']}），不是全快照审计通过", 'audit': audit})
        write_json(out / 'data_manifest.json', {'snapshot_id': doc['snapshot_id'], 'batch_id': doc['batch_id'], 'offline': True, 'audit': audit, 'tables': state.get('tables', {}),
                                                'used': {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in parts.items()} for t, parts in tables['partitions'].items()}})
        status.stage('load')
        inputs = build(tables, cfg.start, cfg.end, cfg.boards, x.liquidity_window, x.liquidity_override); limitations += inputs.limitations
        status.stage('inputs', **inputs.info)
        if source['source'] == 'baseline': scores, eligible = baseline_scores(inputs.candidates), inputs.candidates
        elif source['source'] == 'universe': scores, eligible = _universe_scores(source, inputs.dates)
        elif source['source'] == 'rules':
            from .strategies import rule_scores
            scores, eligible, _ = rule_scores(source, inputs.dates)
        else: scores, eligible = _prediction_scores(source, inputs.dates)
        write_table(out, 'scores', scores); status.stage('scores', rows = len(scores), source = source['source'])
        data_manifest = _read(out / 'data_manifest.json'); env = doc['environment']
        cache = StageCache(root, {'snapshot': doc['snapshot_id'], 'used': data_manifest['used'], 'code': code_version('cache.py'),
                                 'runtime': {k: env[k] for k in ('python', 'packages', 'lock_sha256')}}, cfg.cache and reproduce_of is None and not force_recompute)
        payload = {'config': {k: v for k, v in cfg.model_dump(mode = 'json').items() if k not in ('scores', 'rules', 'cache')},
                   'scores': digest(scores), 'eligible': digest([{'date': d, 'instruments': sorted(v)} for d, v in sorted(eligible.items())]),
                   'rules': rules.config_fingerprint(), 'code': code_version('loop.py', 'portfolio.py', 'ledger/book.py', 'ledger/rules.py', 'execution.py', 'evaluation/portfolio.py', 'replay.py', 'schedule.py'),
                   **({'rule_inputs': source} if source['source'] == 'rules' else {})}
        cache.materialize('ledger', payload, out, lambda dest: _simulate(inputs, scores, eligible, cfg, rules, dest, status))
        sim = _read(out / 'loop_result.json'); info = sim['info']
        status.stage('loop', 'blocked' if 'blocked' in info else 'done'); status.stage('labels', 'not_run', reason = '回放不使用标签；标签由研究阶段 labels.build_labels 生成')
        if (out / 'metrics.json').exists(): status.stage('evaluation')
        final = 'blocked' if sim['status'] == 'blocked' else ('success_limited' if limitations else 'success')
    except InputBlocked as exc:
        final, info = 'blocked', {'blocked': exc.issues}; status.stage('inputs', 'blocked')
    except LedgerError as exc:
        final, info = 'blocked', {'blocked': [{'kind': 'ledger', 'detail': str(exc)}]}; status.stage('loop', 'blocked')
    except Exception as exc:
        if cache is not None: write_json(out / 'cache.json', cache.report())
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    try:
        if cache is not None: write_json(out / 'cache.json', cache.report())
        write_table(out, 'limitations', limitations)
        extra = {}
        if compare is not None:
            extra = compare(out, {'status': final, 'limitations': limitations, **info})
            if extra['reproduction']['result'] == 'mismatch': extra['execution_status'], final = final, 'mismatch'
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    status.finish(final, limitations = limitations, **info, **extra)
    _manifest(out, status.data, doc['environment'])
    return {'run_id': out.name, 'output': str(out), 'status': final, 'limitations': limitations, **info.get('summary', {}),
            **({'blocked': info['blocked']} if 'blocked' in info else {}), **({'reproduction': extra['reproduction']} if extra else {})}


def _simulate(inputs, scores, eligible, cfg, rules, out, status = None):
    rec, orders = None, None
    try:
        by_date = {}
        for s in scores: by_date.setdefault(s['decision_date'], {})[s['instrument']] = s['score']
        rec = _Recorder(inputs.dates, inputs.unexplained); p = cfg.portfolio; x = cfg.execution
        target_options = {}
        if cfg.scores.source == 'rules':
            from .strategies import rule_scores
            _, _, targets = rule_scores({'run': cfg.scores.run}, inputs.dates)
            target_options['targets_by_date'] = targets
        book, orders = run_loop(inputs.dates, inputs.market, by_date, cfg.initial_cash, rules, eligible_by_date = eligible, actions = inputs.actions,
                                rebalance_every = p.rebalance_every, open_cash_policy = p.open_cash_policy, slippage = x.slippage, n = p.n, buffer = p.buffer,
                                max_sell = p.max_sell, max_weight = p.max_weight, refill_between_rebalance = p.refill_between_rebalance, calendar = inputs.calendar,
                                participation = p.participation, on_close = rec, construction = p.construction,
                                rebalance_frequency = p.rebalance_frequency, rebalance_session = p.rebalance_session, **target_options)
        if status is not None: status.stage('loop', orders = len(orders))
        metrics = evaluate(book.equity_curve()); write_json(out / 'metrics.json', metrics)
        write_json(out / 'trading.json', canonical(trading_stats(book.equity_rows, book.fills, orders, rec.positions, cfg.initial_cash)))
        final = book.status
        info = {'assumptions': book.assumptions, 'issues': book.issues, 'rules_used_unverified': sorted(map(str, rules.used_unverified)),
                'summary': {'final_equity': book.equity_curve()[-1], 'days': len(inputs.dates), 'start': inputs.info['start'], 'end': inputs.info['end']}}
    except LedgerError as exc:
        final, info = 'blocked', {'blocked': [{'kind': 'ledger', 'detail': str(exc)}]}
    if rec is not None:
        for name, rows in rec.tables().items(): write_table(out, name, rows)
    if orders is not None: write_table(out, 'orders', orders)
    write_json(out / 'loop_result.json', canonical({'status': final, 'info': info}))


def read_core(out):
    """核心产物：缺失的表记为 None（例如 blocked 运行没有订单）"""
    out = Path(out)
    return {n: json.loads((out / f'{n}.json').read_text(encoding = 'utf-8')) if (out / f'{n}.json').exists() else None for n in CORE}


def core_hash(out): return hashlib.sha256(json.dumps(read_core(out), sort_keys = True, ensure_ascii = False).encode()).hexdigest()


def _manifest(out, status, env):
    files = {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': status['run_id'], 'status': status['status'], 'environment': env, 'files': files, 'core_tables': list(CORE),
                                       'core_hash': core_hash(out)})


# 复现 -------------------------------------------------------------------------------------------------
class ReproduceRefused(ValueError):
    """冻结输入对不上，不能称作原实验复现"""


def _read(path):
    try: return json.loads(Path(path).read_text(encoding = 'utf-8'))
    except (OSError, ValueError) as exc: raise ReproduceRefused(f'无法读取 {path}: {exc}') from None


def check_snapshot(store, snapshot, data_manifest):
    """快照存在，且实际读取过的每个分区文件与源实验记录的字节哈希一致；否则拒绝称作原实验复现"""
    try: state = store.state(snapshot)
    except FileNotFoundError: raise ReproduceRefused(f'快照 {snapshot} 不存在') from None
    for table, parts in data_manifest.get('used', {}).items():
        for part, v in parts.items():
            cur = state['tables'].get(table, {}).get(part, {}); path = store.root / v['file']
            if cur.get('file') != v['file'] or not path.exists() or file_sha(path) != v['file_sha256']:
                raise ReproduceRefused(f'快照数据与源实验记录不一致：{table}/{part}')
    return state


def reproduce(root, run, output = None, abs_tol = 1e-9, rel_tol = 0.0):
    """先读取并校验源实验的全部冻结输入与核心产物，再在新目录重跑并逐表比较；源目录只读。研究实验转给 research.reproduce_research"""
    source = resolve_run(root, run); run = str(source); ensure_outside(source, output)
    kind = _read(source / 'config.json').get('kind')
    if kind == 'research':
        from .research import reproduce_research
        return reproduce_research(root, run, output, abs_tol, rel_tol)
    if kind == 'paired':
        from .paired import reproduce_paired
        return reproduce_paired(root, run, output, abs_tol, rel_tol)
    if kind == 'experiment':
        from .experiments import reproduce_experiment
        return reproduce_experiment(root, run, output, abs_tol, rel_tol)
    if kind == 'strategy':
        if 'spec' in _read(source / 'config.json'):
            from .strategy.run import reproduce_strategy
        else:
            from .strategies import reproduce_strategy
        return reproduce_strategy(root, run, output, abs_tol, rel_tol)
    manifest = _read(source / 'manifest.json'); doc = _read(source / 'config.json'); src_status = _read(source / 'status.json')
    if src_status.get('status') not in ('success', 'success_limited', 'blocked', 'mismatch'): raise ReproduceRefused(f"源实验状态为 {src_status.get('status')}，不是已完成的实验")
    integrity = sorted(n for n, sha in manifest.get('files', {}).items() if not (source / n).exists() or file_sha(source / n) != sha)
    try: cfg = RunConfig.model_validate(doc['config'])
    except (KeyError, ValidationError) as exc: raise ReproduceRefused(f'源实验配置不合法：{exc}') from None
    frozen = source / 'rules.yaml'
    if not frozen.exists(): raise ReproduceRefused('源实验没有冻结的 rules.yaml')
    if RuleSet.from_yaml(frozen).config_fingerprint() != doc['rules']['fingerprint']: raise ReproduceRefused('冻结的 rules.yaml 与记录的规则指纹不一致')
    sc = doc.get('scores') or {}
    if sc.get('source') == 'rules':
        parent = resolve_run(root, sc['run'])
        if file_sha(parent / 'signals_manifest.json') != sc['signals_manifest_sha256']: raise ReproduceRefused('规则策略信号清单与源账本记录不同')
    if sc.get('source') in ('predictions', 'universe'):
        names = (('predictions_sha256', 'predictions.parquet'), ('universe_sha256', 'universe.parquet')) if sc['source'] == 'predictions' else (('universe_sha256', 'universe.parquet'), ('split_plan_sha256', 'split_plan.parquet'))
        for key, name in names:
            f = resolve_run(root, sc['run']) / name
            if not f.exists() or file_sha(f) != sc[key]: raise ReproduceRefused(f'研究实验的 {name} 与源实验记录不一致')
    check_snapshot(Store(root), cfg.snapshot, _read(source / 'data_manifest.json'))
    expected = read_core(source); expected_status = {k: src_status.get('execution_status', src_status.get('status')) if k == 'status' else src_status.get(k) for k in STATUS_FIELDS}
    code = drift(doc.get('environment') or {}, environment())
    workspace = Path(cfg.rules); workspace_differs = not workspace.exists() or RuleSet.from_yaml(workspace).config_fingerprint() != doc['rules']['fingerprint']
    summary = src_status.get('summary') or {}
    if summary.get('start'): cfg = cfg.model_copy(update = {'start': date.fromisoformat(summary['start']), 'end': date.fromisoformat(summary['end'])})

    def compare(out, now):
        actual = read_core(out); diffs, tables = compare_tables({**expected, 'status': expected_status}, {**actual, 'status': {k: canonical(now.get(k)) for k in STATUS_FIELDS}}, abs_tol, rel_tol)
        result = 'match' if not diffs and not integrity else 'mismatch'
        comparison = {'source': str(source), 'reproduced': str(out), 'result': result, 'tolerance': {'abs': abs_tol, 'rel': rel_tol},
                      'tables': tables, 'differences': diffs, 'source_integrity': {'modified_files': integrity}, 'code_drift': code,
                      'rules': {'used': 'frozen', 'workspace_differs': workspace_differs}, 'keys': {k: list(v) if v else ['row'] for k, v in KEYS.items()}}
        write_json(Path(out) / 'comparison.json', comparison)
        return {'reproduction': {'of': str(source), 'result': result, 'differences': len(diffs), 'modified_source_files': integrity, 'code_drift': sorted(code)}}

    return _run(root, cfg, output, source.parent, rules_file = frozen, tag = 'repro', reproduce_of = str(source.resolve()), compare = compare)
