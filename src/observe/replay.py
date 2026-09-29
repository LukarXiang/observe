"""离线回放：固定快照 → 执行输入适配器 → 逐日循环（组合 + 账本）→ 评价。
产物写入新建的实验目录；目录一建立就写 running 状态，结束时提交终态，manifest.json 最后写入。复现的源目录只读。"""
import hashlib, json
from pathlib import Path

from .data.store import Store
from .evaluation import metrics as evaluate
from .execution import InputBlocked, build, load
from .ledger import LedgerError, RuleSet
from .loop import run_loop
from .portfolio import DEFAULT_PARTICIPATION
from .runs import RunDirError, RunStatus, canonical, create_run_dir, ensure_outside, file_sha, write_json, write_table

RULES = 'configs/rule_profiles/main_board.yaml'
BASELINE = {'name': 'lexicographic_engineering_baseline_v1', 'evidence': 'engineering_baseline_not_a_prediction_model',
            'detail': '按证券代码字典序给研究候选打分，只用于验证执行链路，不是预测模型'}
CORE = ('scores', 'orders', 'fills', 'cash_events', 'equity', 'positions_daily', 'receivables', 'metrics', 'limitations')
BOOK_TABLES = ('fills', 'cash_events', 'equity', 'positions_daily', 'receivables')


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


def config_hash(cfg): return hashlib.sha256(json.dumps(canonical(cfg), sort_keys = True).encode()).hexdigest()


def run_offline(root, output = None, snapshot = None, initial_cash = 100000.0, start = None, end = None, boards = ('main',), liquidity_window = 20,
                liquidity_override = None, runs_root = None, tag = None, **extra):
    if snapshot is None: raise ValueError('run 必须指定已固定的 snapshot')
    store = Store(root); state = store.state(snapshot)                           # 快照不存在时在建目录之前就报错
    rules = RuleSet.from_yaml(RULES)
    cfg = {'snapshot_id': state.get('snapshot_id') or snapshot, 'batch_id': state.get('batch_id'), 'initial_cash': float(initial_cash),
           'start': None if start is None else str(start), 'end': None if end is None else str(end), 'boards': sorted(boards),
           'liquidity_window': int(liquidity_window), 'liquidity_override': liquidity_override, 'baseline': BASELINE,
           'portfolio': {'n': 1, 'max_weight': 1.0, 'rebalance_every': 1, 'participation': DEFAULT_PARTICIPATION},
           'rules': {'path': RULES, 'fingerprint': rules.config_fingerprint()}, **extra}
    h = config_hash(cfg)
    out = create_run_dir(Path(runs_root) if runs_root else Path(root) / 'runs', output, '-'.join(x for x in (h[:6], tag) if x))
    status = RunStatus(out, out.name, evidence = BASELINE['evidence'], config_hash = h)
    write_json(out / 'config.json', cfg); status.stage('config')
    rec, orders, limitations = None, None, []
    try:
        tables = load(store, state, start, end, cfg['liquidity_window'])
        write_json(out / 'data_manifest.json', {'snapshot_id': cfg['snapshot_id'], 'batch_id': cfg['batch_id'], 'tables': state.get('tables', {}), 'offline': True})
        status.stage('load')
        inputs = build(tables, start, end, boards, cfg['liquidity_window'], liquidity_override); limitations = list(inputs.limitations)
        status.stage('inputs', **inputs.info)
        scores = baseline_scores(inputs.candidates); write_table(out, 'scores', scores); status.stage('scores', rows = len(scores))
        by_date = {}
        for x in scores: by_date.setdefault(x['decision_date'], {})[x['instrument']] = x['score']
        rec = _Recorder(inputs.dates, inputs.unexplained); p = cfg['portfolio']
        book, orders = run_loop(inputs.dates, inputs.market, by_date, cfg['initial_cash'], rules, eligible_by_date = inputs.candidates, actions = inputs.actions,
                                n = p['n'], max_weight = p['max_weight'], rebalance_every = p['rebalance_every'], participation = p['participation'],
                                calendar = inputs.calendar, on_close = rec)
        status.stage('loop', orders = len(orders))
        status.stage('labels', 'not_run', reason = '工程基线不使用标签；标签由 labels.build_labels 在研究阶段生成')
        metrics = evaluate(book.equity_curve()); write_json(out / 'metrics.json', metrics); status.stage('evaluation')
        final = 'blocked' if book.status == 'blocked' else ('success_limited' if limitations else 'success')
        summary = {'final_equity': book.equity_curve()[-1], 'days': len(inputs.dates), 'total_return': metrics['total_return']}
        info = {'assumptions': book.assumptions, 'issues': book.issues, 'rules_used_unverified': sorted(map(str, rules.used_unverified)), 'summary': summary}
    except InputBlocked as exc:
        final, info = 'blocked', {'blocked': exc.issues}; status.stage('inputs', 'blocked')
    except LedgerError as exc:
        final, info = 'blocked', {'blocked': [{'kind': 'ledger', 'detail': str(exc)}]}; status.stage('loop', 'blocked')
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    if rec is not None:
        for name, rows in rec.tables().items(): write_table(out, name, rows)
    if orders is not None: write_table(out, 'orders', orders)
    write_table(out, 'limitations', limitations)
    status.finish(final, limitations = limitations, **info)
    _manifest(out, status.data)
    return {'run_id': out.name, 'output': str(out), 'status': final, 'limitations': limitations, **info.get('summary', {}),
            **({'blocked': info['blocked']} if 'blocked' in info else {})}


def core_hash(out):
    body = {n: json.loads((out / f'{n}.json').read_text(encoding = 'utf-8')) if (out / f'{n}.json').exists() else None for n in CORE}
    return hashlib.sha256(json.dumps(body, sort_keys = True, ensure_ascii = False).encode()).hexdigest()


def _manifest(out, status):
    files = {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': status['run_id'], 'status': status['status'], 'files': files, 'core_tables': list(CORE), 'core_hash': core_hash(out)})


def reproduce(root, run, output = None):
    """先读取并校验源实验，再在新目录重跑；源目录只读"""
    source = Path(run); ensure_outside(source, output)
    if not (source / 'manifest.json').exists(): raise RunDirError(f'{source} 不是已完成的实验目录（缺 manifest.json）')
    config = json.loads((source / 'config.json').read_text(encoding = 'utf-8')); expected = json.loads((source / 'metrics.json').read_text(encoding = 'utf-8'))
    inputs = json.loads((source / 'status.json').read_text(encoding = 'utf-8')).get('stages', {}).get('inputs', {})
    start, end = (inputs.get('start'), inputs.get('end')) if isinstance(inputs, dict) else (config.get('start'), config.get('end'))
    result = run_offline(root, output = output, snapshot = config['snapshot_id'], initial_cash = config['initial_cash'], start = start, end = end,
                         boards = config.get('boards', ('main',)), liquidity_window = config.get('liquidity_window', 20),
                         liquidity_override = config.get('liquidity_override'), runs_root = source.parent, tag = 'repro', reproduce_of = str(source.resolve()))
    target = Path(result['output']); actual = json.loads((target / 'metrics.json').read_text(encoding = 'utf-8')) if (target / 'metrics.json').exists() else {}
    keys = ('total_return', 'annual_return', 'max_drawdown')
    comparison = {'source': str(source), 'reproduced': str(target), 'keys': list(keys), 'matches': all(expected.get(k) == actual.get(k) for k in keys),
                  'expected': {k: expected.get(k) for k in keys}, 'actual': {k: actual.get(k) for k in keys}}
    write_json(target / 'comparison.json', comparison); result['matches'] = comparison['matches']
    return result
