"""Use python -m scripts.run_strategy_batch7; source40 stage archives and checks."""
import argparse
import json
import math
from pathlib import Path

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


def wilder(prices):
    changes = [b - a for a, b in zip(prices, prices[1:])]
    gain = math.fsum(max(c, 0) for c in changes[:6]) / 6
    loss = math.fsum(max(-c, 0) for c in changes[:6]) / 6
    for change in changes[6:]:
        gain = (gain * 5 + max(change, 0)) / 6
        loss = (loss * 5 + max(-change, 0)) / 6
    return 100 * gain / (gain + loss) if gain + loss > 1e-14 else 0.


def signal_check(root, output):
    cfg = read(output / 'config.json')['config']; store = Store(root); state = store.state(cfg['snapshot'])
    pool = cfg['parameters']['pool']; filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters = filters), store.load_state(state, 'adj_factors', filters = filters),
                         store.load_state(state, 'adj_coverage', filters = filters))
    view['date'] = pd.to_datetime(view.date).dt.date; calendar = sessions(store.load_state(state, 'calendar'))
    grouped = {i: rows.sort_values('date') for i, rows in view.groupby('instrument')}
    factors = pd.read_parquet(output / 'factors.parquet'); targets = pd.read_parquet(output / 'targets.parquet').set_index(['decision_date', 'instrument'])
    expected = {}; checked = 0; paused_windows = 0
    for row in factors.itertuples():
        prices = grouped[row.instrument]; window = prices[prices.date.le(row.date) & prices.is_trading].tail(61)
        start, end = window.date.iloc[0], window.date.iloc[-1]
        require(list(prices[prices.date.between(start, row.date)].date) == [d for d in calendar if start <= d <= row.date], 'Missing historical row skipped')
        require(len(window) == 61 and start == row.window_start and end == row.window_end and row.window_rows == 61, 'RSI traded-row window differs')
        raw = wilder(window.close_adj.tolist()); value = int(raw); buy, sell = 15 < value < 25, value > 85 or value < 10
        require(math.isclose(raw, row.rsi_raw, rel_tol = 1e-11, abs_tol = 1e-9) and value == row.rsi_int, 'Independent Wilder RSI differs')
        require(row.buy == buy and row.sell == sell and row.first_pool_index == pool.index(row.instrument) and row.pool_occurrences == pool.count(row.instrument), 'Source flags or multiplicity differs')
        target = targets.loc[(row.date, row.instrument)]
        require(target['buy'] == buy and target['sell'] == sell and target.priority == value, 'Frozen slot target differs')
        expected.setdefault(row.date, {})[row.instrument] = (value, buy, sell); checked += 1
        paused_windows += int((~prices[prices.date.between(row.window_start, row.date)].is_trading).any())
    order_checks = []
    for child in read(output / 'subruns.json')['backtests']:
        folder = Path(child['output']); doc = read(folder / 'config.json')['config']; x = doc['execution']; p = doc['portfolio']
        inputs = build(load(store, state, doc['start'], doc['end'], x['liquidity_window'], doc['execution_instruments']),
                       doc['start'], doc['end'], doc['boards'], x['liquidity_window'], x['liquidity_override'])
        rules = RuleSet.from_yaml(folder / 'rules.yaml'); m = x['fee_multiplier']
        if m != 1: rules = rules.scaled(commission_rate = m, stamp_tax = m, transfer_fee = m, min_commission = m)
        book = Book(doc['initial_cash'], calendar = inputs.calendar); orders = []; max_held = 0; partial = rejected = 0
        for k, day in enumerate(inputs.dates):
            quotes = inputs.market[day]; book.start_day(day, inputs.actions.get(day, ()), quotes)
            if k:
                decision = inputs.dates[k - 1]; signals = expected[decision]
                # Independent source branches; Book remains the sole cash/fee/NAV implementation.
                def limit(i, upper):
                    q = quotes.get(i, {}); price, pre = q.get('open'), q.get('preclose')
                    if price is None or pre is None: return False
                    low, high = rules.limit_prices(pre, day, q.get('board', 'main'), bool(q.get('is_st', False)))
                    return price >= high if upper else price <= low
                def execute(intent):
                    nonlocal partial, rejected
                    result = book.execute({**intent, 'decision_date': decision, 'participation': p['participation']}, quotes.get(intent['instrument'], {}), day, rules, x['slippage'])
                    orders.append({**result, 'exec_date': day}); partial += result['status'] == 'partial'; rejected += result['status'] == 'rejected'
                for instrument, position in list(book.positions.items()):
                    if not position.qty: continue
                    if instrument not in pool:
                        execute({'instrument': instrument, 'side': 'sell', 'qty': 'all', 'reason': 'slot_exit_universe'})
                    elif signals[instrument][2] and not limit(instrument, True) and position.sellable > 0:
                        execute({'instrument': instrument, 'side': 'sell', 'qty': 'all', 'reason': 'slot_exit'})
                candidates = sorted((i for i in pool if signals[i][1]), key = lambda i: signals[i][0])
                for instrument in candidates:
                    if instrument in book.positions and book.positions[instrument].qty: continue
                    free = 9 - sum(position.qty > 0 for position in book.positions.values())
                    if free > 0 and not limit(instrument, False):
                        execute({'instrument': instrument, 'side': 'buy', 'amount': book.cash / free, 'reason': 'slot_enter'})
            book.close_day(day, quotes); max_held = max(max_held, sum(position.qty > 0 for position in book.positions.values()))
            require(max_held <= 9, 'Exceeded original nine positions')
        diff, _ = compare_tables({'orders': read(folder / 'orders.json'), 'equity': read(folder / 'equity.json'), 'cash_events': read(folder / 'cash_events.json')},
                                 canonical({'orders': orders, 'equity': book.equity_rows, 'cash_events': book.cash_events}), 1e-9, 0)
        require(not diff, f'Independent source allocation differs: {diff[:2]}')
        order_checks.append({'scenario': child['scenario'], 'sessions_checked': len(inputs.dates), 'orders_checked': len(orders),
                             'max_positions': max_held, 'partial_orders': partial, 'rejected_orders': rejected})
    return {'sessions_checked': len(expected), 'rsi_windows_checked': checked, 'windows_containing_known_pauses': paused_windows, 'orders': order_checks,
            'method': 'Independent math Wilder RSI restarted per 61 traded rows, source thresholds/order/duplicates, independent intent replay using the same sole Book ledger'}


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    path = directory / 'rsi_slots_corrected_batch7-verification.json'; doc = read(path)
    fees = read(directory / 'fee-hand-checks.json'); protection = read(directory / 'protection.json')
    require(doc['status'] == fees['status'] == protection['status'] == 'ok', 'Incomplete verification')
    report = doc['report']
    result = {'status': 'ok', 'run_id': doc['run']['run_id'], 'reproduction': doc['reproduction'],
              'source_sha256': doc['source_sha256'], 'freeze': read(directory / 'rsi_slots_corrected_batch7-freeze.json'),
              'hand_check': doc['hand_check'], 'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark'][0],
              'validation_file': str(path), 'validation_sha256': file_sha(path), 'protection': protection, 'fees': fees,
              'original_strategy_complete': False, 'progress': archive(root, '第七批RSI修正近似变体验收完成：三成本、独立窗口/槽位、禁网复现及旧资产保护通过'),
              'limitations': report['strategy']['review']['differences'] + report['strategy']['review']['gaps']}
    write_json(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('action', choices = ['run', 'verify', 'fees', 'protect', 'summary'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--directory', default = 'data/staging/strategies-batch7/20261005-rsi-slots')
    parser.add_argument('--config', default = 'configs/strategies/rsi_slots_corrected_batch7.yaml')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch7-verification.json')
    args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'run': result = run_one(args.root, directory, args.config, batch_label = '第七批')
    elif args.action == 'verify': result = verify_one(args.root, directory, 'rsi_slots_corrected_batch7', checker = signal_check, batch_label = '第七批')
    elif args.action == 'fees': result = fee_checks(directory)
    elif args.action == 'protect': result = protect(args.root, directory)
    else: result = summarize(args.root, directory, args.output)
    print(json.dumps(result, ensure_ascii = False))
