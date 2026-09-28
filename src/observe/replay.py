"""Small deterministic offline execution and reproduction flow."""
import hashlib, json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .data.store import Store
from .data.standardize import market_input
from .ledger.rules import RuleSet
from .loop import run_loop

def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def run_offline(root, output=None, snapshot=None, initial_cash=100000.0, start=None, end=None, reproduce_from=None):
    store = Store(root)
    if reproduce_from:
        source = Path(reproduce_from); config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        snapshot, initial_cash, start, end = config['snapshot_id'], config['initial_cash'], config.get('start'), config.get('end'); output = output or str(source.parent / f'{source.name}-reproduced')
    if snapshot is None:
        raise ValueError('run 必须指定已固定的 snapshot')
    state = store.state(snapshot); bars = store.load_state(state, 'bars_1d')
    if start: bars = bars[bars.date >= pd.to_datetime(start).date()]
    if end: bars = bars[bars.date <= pd.to_datetime(end).date()]
    bars = bars.sort_values(['date', 'instrument']); dates = sorted(set(bars.date))
    if not dates: raise ValueError('快照没有可执行日线')
    adj = store.load_state(state, 'adj_factors')
    coverage = store.load_state(state, 'adj_coverage')
    view = market_input(bars, adj, coverage)
    rules = RuleSet.from_yaml('configs/rule_profiles/main_board.yaml'); market, scores = {}, {}
    for day, group in view.groupby('date'):
        market[day] = {r.instrument: {'open': r.adj_open if pd.notna(r.adj_open) else r.open, 'close': r.adj_close if pd.notna(r.adj_close) else r.close, 'preclose': r.preclose, 'suspended': not bool(r.is_trading), 'board': r.board, 'is_st': bool(r.is_st), 'avg_amount_20d': 1e12} for r in group.itertuples() if bool(r.is_trading) and pd.notna(r.open) and pd.notna(r.close)}
        scores[day] = {i: float(-n) for n, i in enumerate(sorted(market[day]))}
    book, orders = run_loop(dates, market, scores, initial_cash, rules, eligible_by_date={d: set(scores[d]) for d in dates}, n=1, max_weight=1.0, rebalance_every=1)
    out = Path(output or (Path(root) / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')))
    config = {'snapshot_id': state.get('snapshot_id') or snapshot, 'batch_id': state.get('batch_id'), 'initial_cash': initial_cash, 'start': str(dates[0]), 'end': str(dates[-1]), 'baseline': 'lexicographic_first_v1'}
    _json(out / 'config.json', config); _json(out / 'data_manifest.json', {'snapshot_id': config['snapshot_id'], 'batch_id': config['batch_id'], 'tables': state.get('tables', {}), 'offline': True})
    for name, value in (('scores', scores), ('orders', orders), ('fills', book.fills), ('cash_events', book.cash_events), ('equity', book.equity_rows)):
        pd.DataFrame(value.items() if name == 'scores' else value).to_json(out / f'{name}.json', orient='records', force_ascii=False, date_format='iso')
    curve = book.equity_curve(); metrics = {'total_return': curve[-1] / curve[0] - 1, 'final_equity': curve[-1], 'days': len(dates)}
    positions = [{'instrument': i, 'qty': p.qty, 'sellable': p.sellable, 'cost': p.cost, 'last_price': p.last_price, 'stale': p.stale} for i, p in book.positions.items() if p.qty]
    limitations = []
    if len(coverage) == 0: limitations.append({'kind': 'adjustment_coverage', 'detail': 'snapshot has no adjustment coverage'})
    if len(view) and view.close_adj.isna().any(): limitations.append({'kind': 'adjusted_price_missing', 'rows': int(view.close_adj.isna().sum())})
    close_field = view['close_adj'].where(view['close_adj'].notna(), view['close'])
    label_view = view.assign(_close=close_field).sort_values(['instrument', 'date'])
    label_view['_value'] = label_view.groupby('instrument')._close.shift(-1) / label_view._close - 1
    _json(out / 'labels.json', [{'date': str(r[0]), 'instrument': r[1], 'value': None if pd.isna(r[2]) else float(r[2])} for r in label_view[['date', 'instrument', '_value']].itertuples(index=False, name=None)])
    _json(out / 'positions.json', positions); _json(out / 'limitations.json', limitations)
    _json(out / 'metrics.json', metrics); _json(out / 'status.json', {'status': book.status if not limitations else 'success_limited', 'assumptions': book.assumptions, 'issues': book.issues, 'rules_used_unverified': sorted(map(str, rules.used_unverified)), 'evidence': 'offline_snapshot_replay', 'limitations': limitations})
    _json(out / 'manifest.json', {'files': sorted(p.name for p in out.iterdir()), 'core_hash': hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()})
    return {'run_id': out.name, 'output': str(out), **metrics, 'status': book.status}

def reproduce(root, run, output=None): return run_offline(root, output=output, reproduce_from=run)
