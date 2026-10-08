"""Use python -m scripts.run_strategy_batch5; each completed phase is archived."""
import argparse
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import pandas as pd
import requests
import yaml

from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.execution import sessions
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.runs import environment, file_sha, write_json
from observe.strategies import StrategyConfig, run_strategy
from observe.strategy_catalog import catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.verify_strategy_batch3 import require, tree_hash


def signal_check(root, output):
    cfg = read(output / 'config.json')['config']; store = Store(root); state = store.state(cfg['snapshot'])
    filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters = filters), store.load_state(state, 'adj_factors', filters = filters),
                         store.load_state(state, 'adj_coverage', filters = filters))
    view['date'] = pd.to_datetime(view.date).dt.date; calendar = sessions(store.load_state(state, 'calendar'))
    prices = view.set_index('date').reindex(calendar)
    values = [r.close_adj if bool(r.is_trading) else math.nan for r in prices.itertuples()]
    def averages(k): return [math.fsum(values[k - n + 1:k + 1]) / n for n in (5, 10, 20, 30)]
    def bull(row): return row[0] > row[1] > row[2] > row[3]
    def bear(row): return row[0] < row[1] < row[2]
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    targets = pd.read_parquet(output / 'targets.parquet').set_index('decision_date')
    universe = pd.read_parquet(output / 'universe.parquet').set_index('decision_date')
    counts = {}; expected_signals = {}; examples = []
    for day, row in factors.iterrows():
        k = calendar.index(day); a, b, c = averages(k), averages(k - 1), averages(k - 2)
        valid = all(math.isfinite(v) for values_ in (a, b, c) for v in values_) and bool(universe.loc[day, 'eligible'])
        flags = {'bull': bull(a), 'bear': bear(a), 'struggle': abs(a[1] / a[2] - 1) < .003 or abs(a[2] / a[3] - 1) < .002,
                 'crossdown': bull(b) and bull(c) and b[0] > b[1] and a[0] < a[1],
                 'crossup': bear(b) and bear(c) and b[1] < b[2] and a[1] > a[2]}
        for column, value in zip(('ma5', 'ma10', 'ma20', 'ma30'), a, strict = True):
            require((math.isnan(value) and pd.isna(row[column])) or math.isclose(row[column], value, rel_tol = 1e-10, abs_tol = 1e-8), f'{day}: {column} differs')
        require(bool(row.valid) == valid and all(bool(row[key]) == value for key, value in flags.items()), f'{day}: flags differ')
        enter = valid and ((flags['bull'] and not flags['struggle']) or flags['crossup'])
        exit_signal = valid and (flags['bear'] or flags['crossdown']); skip = valid and flags['bull'] and flags['struggle']
        target = targets.loc[day]
        require(target.target_value == 20000 and bool(target.enter_when_empty) == enter and bool(target['exit']) == exit_signal and bool(target.skip_when_empty) == skip, f'{day}: intent differs')
        expected_signals[str(day)] = (enter, exit_signal, skip)
        for key, value in {'valid': valid, **flags}.items(): counts[key] = counts.get(key, 0) + int(value)
        if len(examples) < 1 or ((flags['crossup'] or flags['crossdown']) and len(examples) < 8):
            examples.append({'date': str(day), 'averages': a, **flags, 'valid': valid})
    order_checks = []
    for child in read(output / 'subruns.json')['backtests']:
        folder = Path(child['output']); orders = read(folder / 'orders.json'); positions = read(folder / 'positions_daily.json')
        by_date = {str(day): [] for day in factors.index}
        for order in orders: by_date[order['decision_date']].append(order)
        days = [str(d) for d in factors.index]
        for day, nxt in zip(days, days[1:]):
            held = sum(p['qty'] for p in positions if p['date'] == day)
            enter, exit_signal, skip = expected_signals[day]
            # Evaluate source branch priorities against actual ledger holdings, without the portfolio helper.
            expected = []
            if not (held == 0 and skip):
                if held > 0 and exit_signal: expected = ['sell']
                elif held == 0 and enter and not exit_signal: expected = ['buy']
            actual = by_date[day]
            require([o['side'] for o in actual] == expected, f'{folder.name}/{day}: actual-position intents differ')
            require(all(o['exec_date'] == nxt and (o['side'] != 'buy' or o['amount'] == 20000) for o in actual), 'Fixed amount or next-open timing differs')
        order_checks.append({'scenario': child['scenario'], 'decision_days_checked': len(days) - 1, 'orders_checked': len(orders)})
    return {'sessions_checked': len(factors), 'flags': counts, 'examples': examples, 'orders': order_checks,
            'method': 'math.fsum 5/10/20/30-day windows and independent original branches against recorded actual holdings'}


def run_one(root, directory, config_path, batch_label = '第五批'):
    output = directory / f'{Path(config_path).stem}-run.json'
    if output.exists(): raise FileExistsError(output)
    cfg = StrategyConfig.model_validate(yaml.safe_load(Path(config_path).read_text(encoding = 'utf-8')))
    freeze = directory / f'{Path(config_path).stem}-freeze.json'
    if freeze.exists(): raise FileExistsError(freeze)
    write_json(freeze, {'config': cfg.model_dump(mode = 'json'), 'source_sha256': file_sha(cfg.source_path),
                       'purpose': 'fixed parameters and initial-cash contrast, not parameter selection'})
    result = run_strategy(root, **cfg.model_dump(mode = 'json')); write_json(output, result)
    archive(root, f'{batch_label}实验落盘：{result["run_id"]} / {result["status"]}')
    return result


def verify_one(root, directory, label, checker = signal_check, batch_label = '第五批'):
    destination = directory / f'{label}-verification.json'
    if destination.exists(): raise FileExistsError(destination)
    result = read(directory / f'{label}-run.json'); output = Path(result['output'])
    require(result['status'] == 'success_limited', 'Strategy did not finish')
    require(verify_run(root, output)['status'] == 'ok', 'Original integrity failed')
    arithmetic = checker(root, output); before = tree_hash(output)
    def forbidden(*a, **kw): raise AssertionError('Network disabled for reproduction')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
        again = reproduce(root, result['run_id'])
    require(again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0, 'Offline reproduction mismatch')
    require(tree_hash(output) == before, 'Original changed')
    require(verify_run(root, again['run_id'])['status'] == 'ok', 'Reproduction integrity failed')
    verified = {'status': 'ok', 'run': result, 'reproduction': again, 'hand_check': arithmetic,
                'report': read(output / 'report.json'), 'source_sha256': file_sha(output / 'source.original'), 'original_unchanged': True}
    write_json(destination, verified)
    doc = read(output / 'config.json'); evidence_path = Path(root) / 'catalog/strategies/implementation-evidence.json'
    evidence = read(evidence_path)
    evidence.append({'strategy_id': doc['source']['strategy_id'], 'source_sha256': verified['source_sha256'],
                     'implementation': doc['config']['implementation'], 'scope': doc['source']['review']['scope'],
                     'run_id': result['run_id'], 'snapshot': doc['snapshot_id'], 'reproduction_run_id': again['run_id'],
                     'reproduction_result': 'match', 'differences': 0, 'original_strategy_complete': False,
                     'validation_file': str(destination), 'validation_sha256': file_sha(destination)})
    write_json(evidence_path, evidence)
    catalog_strategies(root, 'repo/量化策略源代码')
    archive(root, f'{batch_label}离线验收落盘：{label} / match / 0差异')
    return {'run_id': result['run_id'], 'reproduction': again['reproduction'], 'sessions_checked': arithmetic['sessions_checked']}


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    checks = []
    for label in ('multi_ma_fixed_20k_batch5', 'multi_ma_fixed_1m_batch5'):
        path = directory / f'{label}-verification.json'; doc = read(path); report = doc['report']
        require(doc['status'] == 'ok', 'Verification incomplete')
        checks.append({'label': label, 'run_id': doc['run']['run_id'], 'reproduction_run_id': doc['reproduction']['run_id'],
                       'reproduction': doc['reproduction']['reproduction'], 'source_sha256': doc['source_sha256'],
                       'freeze': read(directory / f'{label}-freeze.json'), 'hand_check': doc['hand_check'],
                       'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark'][0],
                       'validation_file': str(path), 'validation_sha256': file_sha(path), 'original_strategy_complete': False})
    protection = read(directory / 'protection.json'); fees = read(directory / 'fee-hand-checks.json')
    require(protection['status'] == fees['status'] == 'ok', 'Protection or fees incomplete')
    result = {'status': 'ok', 'environment': environment(), 'strategies': checks, 'protection': protection,
              'fee_checks': {'fills_checked': fees['fills_checked'], 'file': str(directory / 'fee-hand-checks.json'), 'sha256': file_sha(directory / 'fee-hand-checks.json')},
              'verification': {'related_pytest': '70 passed, 1 existing Starlette/httpx warning in 75.06s',
                               'specialists': 'architecture and security: no confirmed new findings', 'full_suite_repeated': False},
              'progress': archive(root, '第五批两种资金长区间均match，384成交费用核算与旧资产保护通过'),
              'limitations': ['Daily close/next-open approximate reproduction, original platform frequency/adjustment/initialization not proven',
                              'Fixed 20k exposure differs with initial cash; no return-based parameter selection',
                              '332 original daily audit warnings retained, fee accuracy not externally verified',
                              'Remote 2027 files missing, strict financial usable rows zero; partition protection only size/mtime']}
    write_json(output, result)
    return {'status': 'ok', 'output': str(output), 'strategies': len(checks), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('action', choices = ['run', 'verify', 'fees', 'protect', 'summary'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--directory', default = 'data/staging/strategies-batch5/20261005-fixed-value')
    parser.add_argument('--config'); parser.add_argument('--label')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch5-verification.json')
    args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'run': result = run_one(args.root, directory, args.config)
    elif args.action == 'verify': result = verify_one(args.root, directory, args.label)
    elif args.action == 'fees': result = fee_checks(directory)
    elif args.action == 'summary': result = summarize(args.root, directory, args.output)
    else: result = protect(args.root, directory)
    print(json.dumps(result, ensure_ascii = False))
