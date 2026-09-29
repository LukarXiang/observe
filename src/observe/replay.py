"""离线回放：固定快照 → 执行输入适配器 → 逐日循环（组合 + 账本）→ 评价。
产物写入新建的实验目录；目录一建立就写 running 状态，结束时提交终态，manifest.json 最后写入。

冻结输入：config.json 保存展开默认值后的完整配置、快照与分区清单、规则内容指纹、代码与依赖版本；rules.yaml 是实际使用的规则原文。
复现只用冻结输入：数据或规则对不上就拒绝，代码版本不同照常比较并报告差异；源目录只读。"""
import hashlib, json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .data.audit import audit_status
from .data.store import Store
from .evaluation import metrics as evaluate, trading_stats
from .execution import InputBlocked, build, load
from .ledger import LedgerError, RuleSet
from .loop import POLICIES, run_loop
from .portfolio import DEFAULT_PARTICIPATION
from .runs import (KEYS, STATUS_FIELDS, RunStatus, canonical, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha,
                   write_json, write_table)

RULES = 'configs/rule_profiles/main_board.yaml'
BASELINE = {'source': 'baseline', 'name': 'lexicographic_engineering_baseline_v1', 'evidence': 'engineering_baseline_not_a_prediction_model',
            'detail': '按证券代码字典序给研究候选打分，只用于验证执行链路，不是预测模型'}
CORE = ('scores', 'orders', 'fills', 'cash_events', 'equity', 'positions_daily', 'receivables', 'metrics', 'trading', 'limitations')


# 配置 -------------------------------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra = 'forbid', allow_inf_nan = False)


class PortfolioConfig(_Strict):
    n: int = Field(1, ge = 1)
    max_weight: float = Field(1.0, gt = 0, le = 1)
    rebalance_every: int = Field(1, ge = 1)
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
    source: Literal['baseline', 'predictions'] = 'baseline'
    run: str | None = None                                       # predictions：研究实验目录（observe research 的产物）
    model: str = 'ridge'


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


def score_source(cfg):
    """分数来源：工程基线，或研究实验的样本外预测表（同时带来研究候选）。预测表与股票池按文件哈希冻结"""
    sc = cfg.scores
    if sc.source == 'baseline': return BASELINE, cfg
    if not sc.run: raise ValueError('scores.source = predictions 需要 scores.run（研究实验目录）')
    run = Path(sc.run); st = json.loads((run / 'status.json').read_text(encoding = 'utf-8'))
    if st.get('kind') != 'research' or st.get('status') not in ('success', 'success_limited'): raise ValueError(f'{run} 不是已完成的研究实验（状态 {st.get("status")}）')
    pred = pd.read_parquet(run / 'predictions.parquet', columns = ['model_id', 'decision_date'])
    days = pred.loc[pred.model_id == sc.model, 'decision_date']
    if not len(days): raise ValueError(f'研究实验 {run.name} 没有模型 {sc.model} 的预测')
    meta = {'source': 'predictions', 'run': str(run.resolve()), 'research_run_id': run.name, 'model': sc.model, 'evidence': 'development_oos_prediction',
            'predictions_sha256': file_sha(run / 'predictions.parquet'), 'universe_sha256': file_sha(run / 'universe.parquet')}
    first, last = pd.Timestamp(days.min()).date(), pd.Timestamp(days.max()).date()
    return meta, cfg.model_copy(update = {'start': cfg.start or first, 'end': cfg.end or last})


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


def _run(root, cfg, output = None, runs_root = None, rules_file = None, tag = None, reproduce_of = None, compare = None):
    store = Store(root); state = store.state(cfg.snapshot)                       # 快照不存在时在建目录之前就报错
    source, cfg = score_source(cfg)
    rules_file = Path(rules_file or cfg.rules); rules_text = rules_file.read_text(encoding = 'utf-8'); rules = RuleSet.from_yaml(rules_file)
    doc = {'config': cfg.model_dump(mode = 'json'), 'snapshot_id': state.get('snapshot_id') or cfg.snapshot, 'batch_id': state.get('batch_id'),
           'scores': source, 'rules': {'source_path': str(cfg.rules), 'sha256': hashlib.sha256(rules_text.encode()).hexdigest(), 'fingerprint': rules.config_fingerprint()},
           'environment': environment(), 'reproduce_of': reproduce_of}
    h = _hash({k: doc[k] for k in ('config', 'snapshot_id', 'scores', 'rules')})
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, '-'.join(x for x in (h[:6], tag) if x))
    status = RunStatus(out, out.name, evidence = source['evidence'], config_hash = h, reproduce_of = reproduce_of)
    (out / 'rules.yaml').write_text(rules_text, encoding = 'utf-8'); write_json(out / 'config.json', doc); status.stage('config')
    rec, orders, limitations, x = None, None, [], cfg.execution
    if x.fee_multiplier != 1: m = x.fee_multiplier; rules = rules.scaled(commission_rate = m, stamp_tax = m, transfer_fee = m, min_commission = m)
    try:
        tables = load(store, state, cfg.start, cfg.end, x.liquidity_window)
        audit = {k: v for k, v in audit_status(root, doc['batch_id'], rules).items() if k != 'rows'}   # 运行前检查：与数据中心页面同一判定
        if audit['status'] != 'passed':
            limitations.append({'kind': 'data_audit', 'detail': f"快照批次审计状态为 {audit['status']}（范围 {audit['scope']}），不是全快照审计通过", 'audit': audit})
        write_json(out / 'data_manifest.json', {'snapshot_id': doc['snapshot_id'], 'batch_id': doc['batch_id'], 'offline': True, 'audit': audit, 'tables': state.get('tables', {}),
                                                'used': {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in parts.items()} for t, parts in tables['partitions'].items()}})
        status.stage('load')
        inputs = build(tables, cfg.start, cfg.end, cfg.boards, x.liquidity_window, x.liquidity_override); limitations += inputs.limitations
        status.stage('inputs', **inputs.info)
        if source['source'] == 'baseline': scores, eligible = baseline_scores(inputs.candidates), inputs.candidates
        else: scores, eligible = _prediction_scores(source, inputs.dates)
        write_table(out, 'scores', scores); status.stage('scores', rows = len(scores), source = source['source'])
        by_date = {}
        for s in scores: by_date.setdefault(s['decision_date'], {})[s['instrument']] = s['score']
        rec = _Recorder(inputs.dates, inputs.unexplained); p = cfg.portfolio
        book, orders = run_loop(inputs.dates, inputs.market, by_date, cfg.initial_cash, rules, eligible_by_date = eligible, actions = inputs.actions,
                                rebalance_every = p.rebalance_every, open_cash_policy = p.open_cash_policy, slippage = x.slippage, n = p.n, buffer = p.buffer,
                                max_sell = p.max_sell, max_weight = p.max_weight, refill_between_rebalance = p.refill_between_rebalance, calendar = inputs.calendar,
                                participation = p.participation, on_close = rec)
        status.stage('loop', orders = len(orders))
        status.stage('labels', 'not_run', reason = '回放不使用标签；标签由研究阶段 labels.build_labels 生成')
        metrics = evaluate(book.equity_curve()); write_json(out / 'metrics.json', metrics)
        write_json(out / 'trading.json', canonical(trading_stats(book.equity_rows, book.fills, orders, rec.positions, cfg.initial_cash))); status.stage('evaluation')
        final = 'blocked' if book.status == 'blocked' else ('success_limited' if limitations else 'success')
        info = {'assumptions': book.assumptions, 'issues': book.issues, 'rules_used_unverified': sorted(map(str, rules.used_unverified)),
                'summary': {'final_equity': book.equity_curve()[-1], 'days': len(inputs.dates), 'start': inputs.info['start'], 'end': inputs.info['end']}}
    except InputBlocked as exc:
        final, info = 'blocked', {'blocked': exc.issues}; status.stage('inputs', 'blocked')
    except LedgerError as exc:
        final, info = 'blocked', {'blocked': [{'kind': 'ledger', 'detail': str(exc)}]}; status.stage('loop', 'blocked')
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    try:
        if rec is not None:
            for name, rows in rec.tables().items(): write_table(out, name, rows)
        if orders is not None: write_table(out, 'orders', orders)
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
    source = Path(run); ensure_outside(source, output)
    if _read(source / 'config.json').get('kind') == 'research':
        from .research import reproduce_research
        return reproduce_research(root, run, output, abs_tol, rel_tol)
    manifest = _read(source / 'manifest.json'); doc = _read(source / 'config.json'); src_status = _read(source / 'status.json')
    if src_status.get('status') not in ('success', 'success_limited', 'blocked', 'mismatch'): raise ReproduceRefused(f"源实验状态为 {src_status.get('status')}，不是已完成的实验")
    integrity = sorted(n for n, sha in manifest.get('files', {}).items() if not (source / n).exists() or file_sha(source / n) != sha)
    try: cfg = RunConfig.model_validate(doc['config'])
    except (KeyError, ValidationError) as exc: raise ReproduceRefused(f'源实验配置不合法：{exc}') from None
    frozen = source / 'rules.yaml'
    if not frozen.exists(): raise ReproduceRefused('源实验没有冻结的 rules.yaml')
    if RuleSet.from_yaml(frozen).config_fingerprint() != doc['rules']['fingerprint']: raise ReproduceRefused('冻结的 rules.yaml 与记录的规则指纹不一致')
    sc = doc.get('scores') or {}
    if sc.get('source') == 'predictions':
        for key, name in (('predictions_sha256', 'predictions.parquet'), ('universe_sha256', 'universe.parquet')):
            f = Path(sc['run']) / name
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
