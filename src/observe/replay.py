"""离线回放：固定快照 → 执行输入适配器 → 逐日循环（组合 + 账本）→ 评价。
产物写入新建的实验目录；目录一建立就写 running 状态，结束时提交终态，manifest.json 最后写入。

冻结输入：config.json 保存展开默认值后的完整配置、快照与分区清单、规则内容指纹、代码与依赖版本；rules.yaml 是实际使用的规则原文。
复现只用冻结输入：数据或规则对不上就拒绝，代码版本不同照常比较并报告差异；源目录只读。"""
import hashlib, json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .data.store import Store
from .evaluation import metrics as evaluate
from .execution import InputBlocked, build, load
from .ledger import LedgerError, RuleSet
from .loop import POLICIES, run_loop
from .portfolio import DEFAULT_PARTICIPATION
from .runs import (KEYS, STATUS_FIELDS, RunStatus, canonical, compare_tables, create_run_dir, drift, ensure_outside, environment, file_sha,
                   write_json, write_table)

RULES = 'configs/rule_profiles/main_board.yaml'
BASELINE = {'source': 'baseline', 'name': 'lexicographic_engineering_baseline_v1', 'evidence': 'engineering_baseline_not_a_prediction_model',
            'detail': '按证券代码字典序给研究候选打分，只用于验证执行链路，不是预测模型'}
CORE = ('scores', 'orders', 'fills', 'cash_events', 'equity', 'positions_daily', 'receivables', 'metrics', 'limitations')


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


class RunConfig(_Strict):
    snapshot: str = Field(min_length = 1)
    start: date | None = None
    end: date | None = None
    initial_cash: float = Field(100000.0, gt = 0)
    boards: list[Literal['main', 'gem', 'star', 'bse']] = Field(default_factory = lambda: ['main'], min_length = 1)
    rules: str = RULES
    portfolio: PortfolioConfig = Field(default_factory = PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory = ExecutionConfig)


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


def run_offline(root, output = None, runs_root = None, **params):
    """公开入口：params 与 YAML 同构（snapshot、start、end、initial_cash、boards、rules、portfolio、execution）"""
    p = run_params(params); return _run(root, p['config'], output or p['output'], runs_root)


def _run(root, cfg, output = None, runs_root = None, rules_file = None, tag = None, reproduce_of = None, compare = None):
    store = Store(root); state = store.state(cfg.snapshot)                       # 快照不存在时在建目录之前就报错
    rules_file = Path(rules_file or cfg.rules); rules_text = rules_file.read_text(encoding = 'utf-8'); rules = RuleSet.from_yaml(rules_file)
    doc = {'config': cfg.model_dump(mode = 'json'), 'snapshot_id': state.get('snapshot_id') or cfg.snapshot, 'batch_id': state.get('batch_id'),
           'scores': BASELINE, 'rules': {'source_path': str(cfg.rules), 'sha256': hashlib.sha256(rules_text.encode()).hexdigest(), 'fingerprint': rules.config_fingerprint()},
           'environment': environment(), 'reproduce_of': reproduce_of}
    h = _hash({k: doc[k] for k in ('config', 'snapshot_id', 'scores', 'rules')})
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, '-'.join(x for x in (h[:6], tag) if x))
    status = RunStatus(out, out.name, evidence = BASELINE['evidence'], config_hash = h, reproduce_of = reproduce_of)
    (out / 'rules.yaml').write_text(rules_text, encoding = 'utf-8'); write_json(out / 'config.json', doc); status.stage('config')
    rec, orders, limitations, x = None, None, [], cfg.execution
    try:
        tables = load(store, state, cfg.start, cfg.end, x.liquidity_window)
        write_json(out / 'data_manifest.json', {'snapshot_id': doc['snapshot_id'], 'batch_id': doc['batch_id'], 'offline': True, 'tables': state.get('tables', {}),
                                                'used': {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in parts.items()} for t, parts in tables['partitions'].items()}})
        status.stage('load')
        inputs = build(tables, cfg.start, cfg.end, cfg.boards, x.liquidity_window, x.liquidity_override); limitations = list(inputs.limitations)
        status.stage('inputs', **inputs.info)
        scores = baseline_scores(inputs.candidates); write_table(out, 'scores', scores); status.stage('scores', rows = len(scores))
        by_date = {}
        for s in scores: by_date.setdefault(s['decision_date'], {})[s['instrument']] = s['score']
        rec = _Recorder(inputs.dates, inputs.unexplained); p = cfg.portfolio
        book, orders = run_loop(inputs.dates, inputs.market, by_date, cfg.initial_cash, rules, eligible_by_date = inputs.candidates, actions = inputs.actions,
                                rebalance_every = p.rebalance_every, open_cash_policy = p.open_cash_policy, slippage = x.slippage, n = p.n, buffer = p.buffer,
                                max_sell = p.max_sell, max_weight = p.max_weight, refill_between_rebalance = p.refill_between_rebalance, calendar = inputs.calendar,
                                participation = p.participation, on_close = rec)
        status.stage('loop', orders = len(orders))
        status.stage('labels', 'not_run', reason = '工程基线不使用标签；标签由 labels.build_labels 在研究阶段生成')
        metrics = evaluate(book.equity_curve()); write_json(out / 'metrics.json', metrics); status.stage('evaluation')
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


def reproduce(root, run, output = None, abs_tol = 1e-9, rel_tol = 0.0):
    """先读取并校验源实验的全部冻结输入与核心产物，再在新目录重跑并逐表比较；源目录只读"""
    source = Path(run); ensure_outside(source, output)
    manifest = _read(source / 'manifest.json'); doc = _read(source / 'config.json'); src_status = _read(source / 'status.json')
    if src_status.get('status') not in ('success', 'success_limited', 'blocked', 'mismatch'): raise ReproduceRefused(f"源实验状态为 {src_status.get('status')}，不是已完成的实验")
    integrity = sorted(n for n, sha in manifest.get('files', {}).items() if not (source / n).exists() or file_sha(source / n) != sha)
    try: cfg = RunConfig.model_validate(doc['config'])
    except (KeyError, ValidationError) as exc: raise ReproduceRefused(f'源实验配置不合法：{exc}') from None
    frozen = source / 'rules.yaml'
    if not frozen.exists(): raise ReproduceRefused('源实验没有冻结的 rules.yaml')
    if RuleSet.from_yaml(frozen).config_fingerprint() != doc['rules']['fingerprint']: raise ReproduceRefused('冻结的 rules.yaml 与记录的规则指纹不一致')
    store = Store(root)
    try: state = store.state(cfg.snapshot)
    except FileNotFoundError: raise ReproduceRefused(f'快照 {cfg.snapshot} 不存在') from None
    for table, parts in _read(source / 'data_manifest.json').get('used', {}).items():
        for part, v in parts.items():
            cur = state['tables'].get(table, {}).get(part, {}); path = store.root / v['file']
            if cur.get('file') != v['file'] or not path.exists() or file_sha(path) != v['file_sha256']:
                raise ReproduceRefused(f'快照数据与源实验记录不一致：{table}/{part}')
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
