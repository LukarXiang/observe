"""策略回测运行：规格 + 快照 → 新的实验目录（不覆盖），登记为 kind = strategy。

产物：config.json（规格原文与指纹）、data_manifest.json、equity / fills / orders / positions_daily / decisions、
metrics.json（含逐年收益）、trading.json、status.json、manifest.json。统一汇总见 strategy/report.py。
"""
from bisect import bisect_left
from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from ..data.audit import audit_status
from ..data.store import Store
from ..evaluation import metrics as evaluate, trading_stats
from ..execution import InputBlocked, build, load, sessions
from ..factors.expr import parse
from ..ledger import LedgerError, RuleSet
from ..replay import RULES, _Recorder
from ..runs import RunStatus, canonical, create_run_dir, environment, file_sha, write_json, write_table
from .engine import Selector, run_strategy_loop, summarize_targets
from .fields import FIELDS, IndexPanel, StockPanel
from .spec import load_spec

DEFAULT_START = date(2022, 1, 1)          # 统一回测区间起点：本地日线自 2021-01 起，留出一年预热给回看较长的表达式
LIQUIDITY_WINDOW = 20


def _lookback(spec):
    lb = [parse(e, FIELDS).lookback for e in spec.expressions()]
    return max(lb + [1])


def yearly_returns(equity_rows, initial):
    if not equity_rows: return {}
    e = pd.DataFrame(equity_rows)[['date', 'equity']]; e['year'] = pd.to_datetime(e.date).dt.year
    out, prev = {}, initial
    for y, g in e.groupby('year'):
        out[str(y)] = float(g.equity.iloc[-1] / prev - 1); prev = g.equity.iloc[-1]
    return out


def run_strategy(root, spec_path, snapshot = None, start = None, end = None, output = None, rules_file = None):
    root = Path(root); store = Store(root); state = store.state(snapshot)
    spec, meta = load_spec(spec_path)
    rules_path = Path(rules_file or RULES); rules_text = rules_path.read_text(encoding = 'utf-8'); rules = RuleSet.from_yaml(rules_path)
    cal = sessions(store.load_state(state, 'calendar'))
    s = pd.Timestamp(start or spec.start or DEFAULT_START).date(); e = pd.Timestamp(end or spec.end or cal[-1]).date()
    s = cal[bisect_left(cal, s)] if bisect_left(cal, s) < len(cal) else s
    warm = cal[max(bisect_left(cal, s) - _lookback(spec) - LIQUIDITY_WINDOW - 5, 0)]
    doc = {'kind': 'strategy', 'config': {'name': spec.id, 'start': str(s), 'end': str(e)}, 'spec': {k: v for k, v in meta.items() if k != 'text'},
           'snapshot_id': state.get('snapshot_id') or snapshot, 'batch_id': state.get('batch_id'),
           'rules': {'source_path': str(rules_path), 'sha256': hashlib.sha256(rules_text.encode()).hexdigest(), 'fingerprint': rules.config_fingerprint()},
           'environment': environment()}
    h = hashlib.sha256(json.dumps(canonical({k: doc[k] for k in ('config', 'spec', 'snapshot_id', 'rules')}), sort_keys = True).encode()).hexdigest()
    out = create_run_dir(root / 'runs', output, f'strategy-{spec.id}')
    status = RunStatus(out, out.name, kind = 'strategy', registry = root / 'runs', evidence = 'exploration', config_hash = h, strategy = spec.id)
    (out / 'spec.yaml').write_text(meta['text'], encoding = 'utf-8'); (out / 'rules.yaml').write_text(rules_text, encoding = 'utf-8')
    write_json(out / 'config.json', doc); status.stage('config')
    limitations = [{'kind': 'deviation', 'detail': d} for d in spec.deviations]
    limitations.append({'kind': 'strategy_port', 'detail': '聚宽策略的声明式改写，非原代码逐行执行；证据级别为探索'})
    try:
        tables = load(store, state, warm, e, LIQUIDITY_WINDOW)
        for t in ('adj_factors', 'adj_coverage'):
            tables[t] = store.load_state(state, t) if t in state.get('tables', {}) else None
        index_bars = None
        if spec.exposure.index:
            if 'index_1d' not in state.get('tables', {}):
                raise InputBlocked([{'kind': 'index_missing', 'detail': f'择时需要指数日线 index_1d（{spec.exposure.index}），快照里没有；先运行 observe data index'}])
            index_bars = store.load_state(state, 'index_1d')
        audit = {k: v for k, v in audit_status(root, doc['batch_id'], rules).items() if k != 'rows'}
        if audit['status'] != 'passed': limitations.append({'kind': 'data_audit', 'detail': f"快照批次审计状态为 {audit['status']}", 'audit': audit})
        used = {t: {p: v for p, v in state['tables'].get(t, {}).items()} for t in ('adj_factors', 'adj_coverage', 'index_1d') if t in state.get('tables', {})}
        used = {**tables['partitions'], **used}
        write_json(out / 'data_manifest.json', {'snapshot_id': doc['snapshot_id'], 'batch_id': doc['batch_id'], 'offline': True, 'audit': audit,
                                                'used': {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in parts.items()} for t, parts in used.items()}})
        status.stage('load')
        inputs = build(tables, s, e, spec.universe.boards, LIQUIDITY_WINDOW); limitations += inputs.limitations
        status.stage('inputs', **inputs.info)
        inst = tables['instruments'].drop_duplicates('instrument', keep = 'last').set_index('instrument')
        stocks = set(inst.index[inst.kind == 'stock'])
        bars = tables['bars_1d']; bars = bars[bars.instrument.isin(stocks) & bars.board.isin(spec.universe.boards)]
        days = [d for d in cal if warm <= d <= e]
        panel = StockPanel(bars, tables['adj_factors'], tables['adj_coverage'], inst, days, rules)
        selector = Selector(spec, panel, inputs.candidates, IndexPanel(index_bars, spec.exposure.index, days) if index_bars is not None else None)
        status.stage('signals', expressions = len(selector.values), lookback = _lookback(spec), warmup_start = str(warm))
        rec = _Recorder(inputs.dates, inputs.unexplained)
        book, orders, decisions = run_strategy_loop(spec, inputs, panel, selector, rules, on_close = rec)
        for name, rows in rec.tables().items(): write_table(out, name, rows)
        write_table(out, 'orders', orders); write_table(out, 'decisions', summarize_targets(decisions))
        m = evaluate(book.equity_curve()); m['yearly_returns'] = yearly_returns(book.equity_rows, book.initial)
        write_json(out / 'metrics.json', canonical(m))
        write_json(out / 'trading.json', canonical(trading_stats(book.equity_rows, book.fills, orders, rec.positions, book.initial)))
        settled = [a for a in book.assumptions if a.get('field') == 'delisted_settlement']
        if settled: limitations.append({'kind': 'delisted_settlement', 'detail': f'{len(settled)} 笔退市持仓按最后估值价折现', 'rows': settled[:20]})
        info = {'assumptions': len(book.assumptions), 'issues': book.issues, 'rules_used_unverified': sorted(map(str, rules.used_unverified)),
                'summary': {'final_equity': book.equity_curve()[-1], 'days': len(inputs.dates), 'start': inputs.info['start'], 'end': inputs.info['end'],
                            'rebalances': len(decisions), 'empty_target_rebalances': sum(1 for d in decisions if not d['targets'])}}
        final = 'blocked' if book.status == 'blocked' else 'success_limited'
        status.stage('loop', orders = len(orders))
    except InputBlocked as exc:
        final, info = 'blocked', {'blocked': exc.issues}; status.stage('inputs', 'blocked')
    except LedgerError as exc:
        final, info = 'blocked', {'blocked': [{'kind': 'ledger', 'detail': str(exc)}]}; status.stage('loop', 'blocked')
    except Exception as exc:
        status.finish('failed', error = f'{type(exc).__name__}: {exc}'); raise
    write_table(out, 'limitations', limitations)
    status.finish(final, limitations = limitations, **info)
    files = {p.name: file_sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write_json(out / 'manifest.json', {'run_id': out.name, 'kind': 'strategy', 'strategy': spec.id, 'status': final, 'environment': doc['environment'], 'files': files})
    return {'run_id': out.name, 'output': str(out), 'strategy': spec.id, 'status': final, **info.get('summary', {}),
            **({'blocked': info['blocked']} if 'blocked' in info else {})}
