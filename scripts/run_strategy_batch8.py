"""Use python -m scripts.run_strategy_batch8; source66 independent checks and archives."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import build, load, sessions
from observe.ledger import Book, RuleSet
from observe.runs import canonical, compare_tables, file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.verify_strategy_batch3 import require


def signal_check(root, output):
    cfg = read(output / 'config.json')['config']; store = Store(root); state = store.state(cfg['snapshot']); pair = cfg['execution_instruments']
    filters = [('instrument', 'in', pair)]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters = filters), store.load_state(state, 'adj_factors', filters = filters),
                         store.load_state(state, 'adj_coverage', filters = filters))
    view['date'] = pd.to_datetime(view.date).dt.date; calendar = sessions(store.load_state(state, 'calendar'))
    adj = view.pivot(index = 'date', columns = 'instrument', values = 'close_adj').reindex(calendar)
    raw = view.pivot(index = 'date', columns = 'instrument', values = 'close').reindex(calendar)
    factors = pd.read_parquet(output / 'factors.parquet'); targets = pd.read_parquet(output / 'targets.parquet').set_index(['decision_date', 'instrument'])
    expected = {}; held = 0; switches = {1: 0, 2: 0}
    for row in factors.itertuples():
        k = calendar.index(row.date); bases = [adj.at[row.date, i] / raw.at[row.date, i] for i in pair]
        spread = [a / bases[0] - b / bases[1] for a, b in zip(adj[pair[0]].iloc[k - 59:k + 1], adj[pair[1]].iloc[k - 59:k + 1], strict = True)]
        avg = math.fsum(spread) / 60; std = math.sqrt(math.fsum((value - avg) ** 2 for value in spread) / 59)
        zraw = (spread[-1] - avg) / std; z = float(round(np.float64(zraw), 4)); previous = held; action = 0
        if held != 1 and z <= -2: held = action = 1
        elif held != 2 and z >= 2: held = action = 2
        require(math.isclose(row.mean, avg, rel_tol = 1e-10, abs_tol = 1e-9) and math.isclose(row.std, std, rel_tol = 1e-10, abs_tol = 1e-9), 'Independent sample statistics differ')
        require(math.isclose(row.z_raw, zraw, rel_tol = 1e-10, abs_tol = 1e-9) and row.z_rounded == z, 'Four-decimal zscore differs')
        require(row.window_start == calendar[k - 59] and row.window_end == row.date and row.previous_state == previous and row.state == held and row.action == action, 'Source timing/state differs')
        for n, instrument in enumerate(pair, 1):
            target = targets.loc[(row.date, instrument)]
            require(target['buy'] == (action == n) and target['sell'] == (action != 0 and action != n), 'Switch intent differs')
        if action: switches[action] += 1
        expected[row.date] = action
    order_checks = []
    for child in read(output / 'subruns.json')['backtests']:
        folder = Path(child['output']); doc = read(folder / 'config.json')['config']; x = doc['execution']; p = doc['portfolio']
        inputs = build(load(store, state, doc['start'], doc['end'], x['liquidity_window'], pair), doc['start'], doc['end'], doc['boards'], x['liquidity_window'], x['liquidity_override'])
        rules = RuleSet.from_yaml(folder / 'rules.yaml'); m = x['fee_multiplier']
        if m != 1: rules = rules.scaled(commission_rate = m, stamp_tax = m, transfer_fee = m, min_commission = m)
        book = Book(doc['initial_cash'], calendar = inputs.calendar); orders = []
        for k, day in enumerate(inputs.dates):
            quotes = inputs.market[day]; book.start_day(day, inputs.actions.get(day, ()), quotes)
            action = expected[inputs.dates[k - 1]] if k else 0
            if action:
                decision = inputs.dates[k - 1]; buy, sell = pair[action - 1], pair[2 - action]
                if sell in book.positions and book.positions[sell].qty:
                    intent = {'instrument': sell, 'side': 'sell', 'qty': 'all', 'reason': 'rotation_exit', 'decision_date': decision, 'participation': p['participation']}
                    orders.append({**book.execute(intent, quotes.get(sell, {}), day, rules, x['slippage']), 'exec_date': day})
                intent = {'instrument': buy, 'side': 'buy', 'amount': book.cash, 'reason': 'rotation_enter', 'decision_date': decision, 'participation': p['participation']}
                orders.append({**book.execute(intent, quotes.get(buy, {}), day, rules, x['slippage']), 'exec_date': day})
            book.close_day(day, quotes)
        diff, _ = compare_tables({'orders': read(folder / 'orders.json'), 'equity': read(folder / 'equity.json'), 'cash_events': read(folder / 'cash_events.json')},
                                 canonical({'orders': orders, 'equity': book.equity_rows, 'cash_events': book.cash_events}), 1e-9, 0)
        require(not diff, f'Independent source rotation differs: {diff[:2]}')
        order_checks.append({'scenario': child['scenario'], 'orders_checked': len(orders), 'sessions_checked': len(inputs.dates)})
    return {'sessions_checked': len(factors), 'switches': switches, 'orders': order_checks,
            'method': 'Independent fsum sample std/60-session dynamically anchored spread/NumPy scalar round/source signal state; independent intents via sole Book ledger'}


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    path = directory / 'pair_zscore_rotation_batch8-verification.json'; doc = read(path)
    fees = read(directory / 'fee-hand-checks.json'); protection = read(directory / 'protection.json')
    require(doc['status'] == fees['status'] == protection['status'] == 'ok', 'Incomplete verification')
    report = doc['report']
    result = {'status': 'ok', 'run_id': doc['run']['run_id'], 'reproduction': doc['reproduction'], 'source_sha256': doc['source_sha256'],
              'freeze': read(directory / 'pair_zscore_rotation_batch8-freeze.json'), 'hand_check': doc['hand_check'],
              'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark'][0],
              'validation_file': str(path), 'validation_sha256': file_sha(path), 'protection': protection, 'fees': fees,
              'original_strategy_complete': False, 'progress': archive(root, '第八批66号zscore修正轮换验收完成：三成本、独立窗口/状态/现金、禁网复现及旧资产保护通过'),
              'limitations': report['strategy']['review']['differences'] + report['strategy']['review']['gaps']}
    write_json(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['run', 'verify', 'fees', 'protect', 'summary'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--directory', default = 'data/staging/strategies-batch8/20261005-zscore-rotation')
    parser.add_argument('--config', default = 'configs/strategies/pair_zscore_rotation_batch8.yaml')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch8-verification.json')
    args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'run': result = run_one(args.root, directory, args.config, batch_label = '第八批')
    elif args.action == 'verify': result = verify_one(args.root, directory, 'pair_zscore_rotation_batch8', checker = signal_check, batch_label = '第八批')
    elif args.action == 'fees': result = fee_checks(directory)
    elif args.action == 'protect': result = protect(args.root, directory)
    else: result = summarize(args.root, directory, args.output)
    print(json.dumps(result, ensure_ascii = False))
