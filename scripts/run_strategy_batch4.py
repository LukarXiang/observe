"""Use python -m scripts.run_strategy_batch4 to run this batch workflow."""
import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import socket
from unittest.mock import patch

import pandas as pd
import requests
import yaml

from observe.data.sources.baostock import BaoStock
from observe.data import raw, standardize
from observe.data.prices import with_adjusted
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.runs import environment, file_sha, write_json
from observe.strategies import PAIR_IMPLEMENTATIONS, run_strategy
from observe.execution import sessions
from scripts.archive_strategy_progress import archive
from scripts.verify_strategy_batch3 import require, signal_check, tree_hash


def read(path): return json.loads(Path(path).read_text(encoding = 'utf-8'))


def mutable_registry(name):
    path = Path(name)
    return path.parent == Path('runs') and path.name.startswith('registry.sqlite')


def checkpoint(root, directory, note = '第四批保护基线已建立，开始补齐基准'):
    directory.mkdir(parents = True, exist_ok = False); store = Store(root); state = store.published()
    files = [p for folder in ('snapshots', 'runs') for p in (store.root / folder).rglob('*')
             if p.is_file() and not mutable_registry(p.relative_to(store.root))]
    controls = {p.relative_to(store.root).as_posix(): file_sha(p) for p in sorted(files)}
    partitions = {}
    for entries in state['tables'].values():
        for entry in entries.values():
            path = store.root / entry['file']; stat = path.stat()
            partitions[entry['file']] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    write_json(directory / 'baseline.json', {'published': state, 'controls': controls, 'partitions': partitions,
               'environment': environment(), 'partition_check': 'size/mtime only; not a fresh full partition hash'})
    source = store.root / 'catalog/strategies/implementation-evidence.json'
    (directory / 'implementation-evidence.before.json').write_bytes(source.read_bytes())
    write_json(directory / 'progress-start.json', archive(root, note))


def supplement_calendar(root, directory, source = None):
    output = directory / 'calendar-supplement.json'
    if output.exists(): raise FileExistsError(output)
    store = Store(root)
    with operation_lock(root, DATA_WRITER):
        state = store.published(); old = store.load_state(state, 'calendar')
        old['date'] = pd.to_datetime(old.date).dt.date
        required = set(pd.date_range('2021-01-04', '2026-09-29').date)
        missing = required - set(old.date)
        require(missing == set(pd.date_range('2021-01-09', '2021-01-10').date), 'Calendar gap differs from reviewed two-date supplement')
        try:
            with (source or BaoStock(root)).session() as src:
                response = src.calendar(min(missing), max(missing))
            original = raw.save(root, 'baostock', 'calendar', 'batch4_2021-01-09_2021-01-10', response)
            incoming = standardize.calendar(response)
            require(set(incoming.date) == missing and not incoming.date.duplicated().any(), 'Incomplete calendar response')
            require(not incoming.is_open.any(), 'Unexpected trading session requires further market-data review')
            merged = pd.concat([old, incoming], ignore_index = True).sort_values('date').reset_index(drop = True)
            entry = store.write_partition('calendar', 'all', merged)
            batch = store.write_batch({'calendar': {'all': entry}}, note = 'batch4: two provider-confirmed missing closed dates')
            store.publish(batch)
            result = {'status': 'published', 'batch_id': batch, 'added_dates': sorted(str(d) for d in missing),
                      'raw_file': str(original), 'raw_sha256': file_sha(original), 'old_rows_retained': len(old)}
        except Exception as exc:
            write_json(output, {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'base_batch': state['batch_id']})
            raise
        write_json(output, result); return result


def run_one(root, directory, config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding = 'utf-8'))
    output = directory / f'{config["implementation"]}-run.json'
    if output.exists(): raise FileExistsError(output)
    result = run_strategy(root, **config); write_json(output, result)
    archive(root, f'第四批实验落盘：{result["run_id"]} / {result["status"]}')
    return result


def pair_signal_check(root, output):
    doc = read(output / 'config.json'); cfg = doc['config']; p = cfg['parameters']
    store = Store(root); snapshot = store.state(cfg['snapshot'])
    filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(snapshot, 'bars_1d', filters = filters),
                         store.load_state(snapshot, 'adj_factors', filters = filters), store.load_state(snapshot, 'adj_coverage', filters = filters))
    view['date'] = pd.to_datetime(view.date).dt.date; calendar = sessions(store.load_state(snapshot, 'calendar'))
    series = {i: view[view.instrument.eq(i)].set_index('date').reindex(calendar) for i in cfg['execution_instruments']}
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    targets = pd.read_parquet(output / 'targets.parquet'); universe = pd.read_parquet(output / 'universe.parquet')
    state = 'empty'; examples = []; counts = {}; i1, i2 = cfg['execution_instruments']
    for day, row in factors.iterrows():
        k = calendar.index(day); inputs = []
        for instrument in (i1, i2):
            frame = series[instrument]; basis = frame.loc[day, 'back_factor']
            window = frame.iloc[k - p['test_days'] + 1:k + 1]
            inputs.append([r.close_adj / basis if bool(r.is_trading) else math.nan for r in window.itertuples()])
        spread = [b - p['regression_ratio'] * a for a, b in zip(*inputs, strict = True)]
        mean = math.fsum(spread) / p['test_days']; sigma = math.sqrt(math.fsum((x - mean) ** 2 for x in spread) / p['test_days'])
        eligible = universe[universe.decision_date.eq(day)].eligible.all()
        known = bool(all(math.isfinite(x) for x in spread) and sigma > 0 and eligible)
        z = (spread[-1] - mean) / sigma if known else math.nan
        require(row.valid == known, f'{day}: pair validity differs')
        previous = state
        if known:
            if z > 1: state = 'buy1'
            elif z < -1: state = 'buy2'
            elif state == 'buy1' and z < 0: state = 'even'
            elif state == 'buy2' and z >= 0: state = 'even'
        expected = {'middle': mean, 'std': sigma, 'spread': spread[-1], 'z': z, 'close1': inputs[0][-1], 'close2': inputs[1][-1]}
        for column, value in expected.items():
            require((math.isnan(value) and pd.isna(row[column])) or math.isclose(row[column], value, rel_tol = 1e-10, abs_tol = 1e-7), f'{day}: {column} differs')
        require(row.previous_state == previous and row.state == state, f'{day}: pair state differs')
        weights = {'empty': [0, 0], 'buy1': [1, 0], 'buy2': [0, 1], 'even': [.5, .5]}[state]
        actual = targets[targets.decision_date.eq(day)].set_index('instrument').weight
        require(list(actual.loc[[i1, i2, 'CASH']]) == [*weights, 1 - sum(weights)], f'{day}: pair targets differ')
        counts[state] = counts.get(state, 0) + 1
        if len(examples) < 1 or (state != previous and len(examples) < 8):
            examples.append({'date': str(day), **expected, 'previous': previous, 'state': state})
    return {'sessions_checked': len(factors), 'state_counts': counts, 'examples': examples,
            'arithmetic': 'math.fsum over separately anchored 120-day windows; independent transition conditions'}


def verify_one(root, directory, implementation):
    destination = directory / f'{implementation}-verification.json'
    if destination.exists(): raise FileExistsError(destination)
    result = read(directory / f'{implementation}-run.json'); path = Path(result['output'])
    require(result['status'] == 'success_limited', 'Strategy did not finish')
    require(verify_run(root, path)['status'] == 'ok', 'Original integrity failed')
    arithmetic = pair_signal_check(root, path) if implementation in PAIR_IMPLEMENTATIONS else signal_check(root, path)
    before = tree_hash(path)
    def forbidden(*a, **kw): raise AssertionError('Network disabled for reproduction')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
        again = reproduce(root, result['run_id'])
    require(again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0, 'Offline reproduction mismatch')
    require(tree_hash(path) == before, 'Original run changed')
    require(verify_run(root, again['run_id'])['status'] == 'ok', 'Reproduction integrity failed')
    verified = {'status': 'ok', 'run': result, 'reproduction': again, 'hand_check': arithmetic, 'report': read(path / 'report.json'),
                'source_sha256': file_sha(path / 'source.original'), 'original_unchanged': True}
    write_json(destination, verified)
    evidence_path = Path(root) / 'catalog/strategies/implementation-evidence.json'; evidence = read(evidence_path)
    doc = read(path / 'config.json')
    evidence.append({'strategy_id': doc['source']['strategy_id'], 'source_sha256': verified['source_sha256'],
                     'implementation': implementation, 'scope': doc['source']['review']['scope'],
                     'run_id': result['run_id'], 'snapshot': doc['snapshot_id'], 'reproduction_run_id': again['run_id'],
                     'reproduction_result': 'match', 'differences': 0, 'original_strategy_complete': False,
                     'validation_file': str(destination), 'validation_sha256': file_sha(destination)})
    write_json(evidence_path, evidence)
    archive(root, f'第四批离线验收落盘：{implementation} / match / 0差异')
    return {'run_id': result['run_id'], 'reproduction': again['reproduction'], 'sessions_checked': arithmetic['sessions_checked']}


def protect(root, directory):
    output = directory / 'protection.json'
    if output.exists(): raise FileExistsError(output)
    baseline = read(directory / 'baseline.json'); store = Store(root)
    controls = {name: sha for name, sha in baseline['controls'].items() if not mutable_registry(name)}
    require(all(file_sha(store.root / name) == sha for name, sha in controls.items()), 'Old snapshot or experiment changed')
    require(all((store.root / name).stat().st_size == item['size'] and (store.root / name).stat().st_mtime_ns == item['mtime_ns'] for name, item in baseline['partitions'].items()), 'Old partition changed')
    require(all(store.published()['tables'].get(t) == ps for t, ps in baseline['published']['tables'].items() if t not in ('index_1d', 'calendar')), 'Old table references changed')
    old_cal = store.load_state(baseline['published'], 'calendar'); current_cal = store.load('calendar')
    retained_cal = current_cal.merge(old_cal[['date']], on = 'date', how = 'inner').sort_values('date').reset_index(drop = True)
    pd.testing.assert_frame_equal(retained_cal, old_cal.sort_values('date').reset_index(drop = True))
    old = store.load_state(baseline['published'], 'index_1d'); current = store.load('index_1d')
    retained = current.merge(old[['date', 'index']], on = ['date', 'index'], how = 'inner').sort_values(['date', 'index']).reset_index(drop = True)
    pd.testing.assert_frame_equal(retained, old.sort_values(['date', 'index']).reset_index(drop = True))
    result = {'status': 'ok', 'old_controls_unchanged': len(controls), 'mutable_registry_files_excluded': len(baseline['controls']) - len(controls),
              'old_partition_metadata_unchanged': len(baseline['partitions']), 'old_calendar_rows_unchanged': len(old_cal),
              'old_index_rows_unchanged': len(old), 'partition_check': 'size/mtime only; no fresh full hash', 'baseline_sha256': file_sha(directory / 'baseline.json')}
    write_json(output, result); return result


def fee_checks(directory):
    output = directory / 'fee-hand-checks.json'
    if output.exists(): raise FileExistsError(output)
    checks = []
    cents = lambda value: float(Decimal(str(value)).quantize(Decimal('.01'), rounding = ROUND_HALF_UP))
    for path in sorted(directory.glob('*-run.json')):
        parent = read(path)
        for child in parent['subruns']['backtests']:
            folder = Path(child['output']); config = read(folder / 'config.json')['config']
            fees = yaml.safe_load((folder / 'rules.yaml').read_text(encoding = 'utf-8'))['fees']
            multiplier = config['execution']['fee_multiplier']; dates = {}; fills = read(folder / 'fills.json')
            for fill in fills:
                rule = sorted((r for r in fees if str(r['start']) <= fill['date']), key = lambda r: r['start'])[-1]
                value = fill['value']
                expected = {'commission': cents(max(value * rule['commission_rate'] * multiplier, rule['min_commission'] * multiplier)),
                            'stamp_tax': cents(value * rule['stamp_tax'] * multiplier) if fill['side'] == 'sell' or rule['stamp_both_sides'] else 0.,
                            'transfer_fee': cents(value * rule['transfer_fee'] * multiplier)}
                expected['fee'] = cents(sum(expected.values()))
                require(cents(fill['qty_filled'] * fill['fill_price']) == value, f'{folder.name}/{fill["fill_id"]}: value differs')
                require(all(fill[key] == amount for key, amount in expected.items()), f'{folder.name}/{fill["fill_id"]}: fees differ')
                dates[str(rule['start'])] = dates.get(str(rule['start']), 0) + 1
            checks.append({'run_id': parent['run_id'], 'child': child['run_id'], 'scenario': child['scenario'],
                           'fills_checked': len(fills), 'fee_regime_counts': dates, 'fee_total': cents(sum(f['fee'] for f in fills))})
    result = {'status': 'ok', 'fills_checked': sum(c['fills_checked'] for c in checks), 'checks': checks,
              'method': 'Independent multiplication from frozen YAML date ranges and Decimal half-up cents; verifies arithmetic, not external fee accuracy'}
    write_json(output, result); return result


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    expected = ['ma10_ma20_v1', 'bollinger_breakout_corrected_v1', 'ma5_ma10_price_v1',
                'pair_yili_cmb_anchored_v1', 'pair_haitian_cmb_anchored_v1']
    checks = []
    for implementation in expected:
        path = directory / f'{implementation}-verification.json'; validation = read(path)
        require(validation['status'] == 'ok', f'{implementation}: verification incomplete')
        report = validation['report']
        checks.append({'implementation': implementation, 'run_id': validation['run']['run_id'],
                       'source_sha256': validation['source_sha256'], 'period': report['period'],
                       'sessions_checked': validation['hand_check']['sessions_checked'],
                       'reproduction_run_id': validation['reproduction']['run_id'], 'reproduction': validation['reproduction']['reproduction'],
                       'results': [{'scenario': row['scenario'], 'metrics': row['metrics'], 'trading': row['trading']} for row in report['results']],
                       'yearly_base': report['results'][0]['yearly'], 'benchmark': report['benchmark'][0],
                       'validation_file': str(path), 'validation_sha256': file_sha(path), 'original_strategy_complete': False})
    protection = read(directory / 'protection.json'); fees = read(directory / 'fee-hand-checks.json')
    require(protection['status'] == fees['status'] == 'ok', 'Protection or fee checks incomplete')
    result = {'status': 'ok', 'environment': environment(), 'freeze': read(directory / 'freeze.json'),
              'strategies': checks, 'protection': protection,
              'fee_hand_checks': {'fills_checked': fees['fills_checked'], 'file': str(directory / 'fee-hand-checks.json'),
                                  'sha256': file_sha(directory / 'fee-hand-checks.json')},
              'progress': archive(root, '第四批五项长区间实验均已验收、费用与旧资产保护通过'),
              'limitations': ['All are approximate/component research; not complete original-platform reproduction or independent final holdout',
                              'Full daily audit retains 306 beyond_limit and 26 vwap_outside warnings; fee profiles remain unverified',
                              'Remote handoff still missing 2027 files; strict historical financial usable rows remain zero',
                              'Old partition protection in this batch checks size/mtime, not fresh complete byte hashes']}
    write_json(output, result); return {'status': 'ok', 'output': str(output), 'strategies': len(checks), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('action', choices = ['checkpoint', 'calendar', 'run', 'verify', 'protect', 'fees', 'summary']); parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', default = 'data/staging/strategies-batch4/20261005-long-history')
    parser.add_argument('--config'); parser.add_argument('--implementation')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch4-verification.json')
    args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'checkpoint': result = checkpoint(args.root, directory)
    elif args.action == 'calendar': result = supplement_calendar(args.root, directory)
    elif args.action == 'run': result = run_one(args.root, directory, args.config)
    elif args.action == 'verify': result = verify_one(args.root, directory, args.implementation)
    elif args.action == 'fees': result = fee_checks(directory)
    elif args.action == 'summary': result = summarize(args.root, directory, args.output)
    else: result = protect(args.root, directory)
    print(json.dumps(result, ensure_ascii = False))
