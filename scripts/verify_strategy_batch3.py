"""Read-only batch checks, independent signal arithmetic and offline reproduction."""
import argparse
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.execution import sessions
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.runs import environment, file_sha, write_json


def read(path): return json.loads(Path(path).read_text(encoding = 'utf-8'))


def require(condition, detail):
    if not condition: raise ValueError(detail)


def tree_hash(path): return {p.relative_to(path).as_posix(): file_sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def signal_check(root, output):
    store = Store(root); doc = read(output / 'config.json'); cfg = doc['config']; state = store.state(cfg['snapshot'])
    instrument = cfg['instrument']
    bars = pd.concat([pd.read_parquet(store.root / v['file'], filters = [('instrument', '==', instrument)]) for v in state['tables']['bars_1d'].values()], ignore_index = True)
    view = with_adjusted(bars, store.load_state(state, 'adj_factors'), store.load_state(state, 'adj_coverage'))
    view['date'] = pd.to_datetime(view.date).dt.date
    close = view.set_index('date').close_adj.where(view.set_index('date').is_trading).reindex(sessions(store.load_state(state, 'calendar')))
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    targets = pd.read_parquet(output / 'targets.parquet'); uni = pd.read_parquet(output / 'universe.parquet').set_index('decision_date')
    state_weight, examples = 0.0, []
    for day, row in factors.iterrows():
        k = close.index.get_loc(day); values = close.iloc[:k + 1].to_numpy(float); price = values[-1]
        p = cfg['parameters']; expected = {}; action = 'hold'
        if cfg['implementation'] == 'bollinger_breakout_corrected_v1':
            window = values[-p['boll_window']:]
            mean = math.fsum(window) / len(window)
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in window) / len(window))
            top, bottom = mean + p['boll_std_multiplier'] * std, mean - p['boll_std_multiplier'] * std
            expected = {'middle': mean, 'std': std, 'upper': top, 'lower': bottom}
            if price > top: next_state, action = cfg['portfolio']['max_weight'], 'buy'
            elif price < bottom: next_state, action = 0.0, 'sell'
            else: next_state = state_weight
        else:
            short = math.fsum(values[-p['short']:]) / p['short']; long = math.fsum(values[-p['long']:]) / p['long']
            expected = {'ma_short': short, 'ma_long': long}
            if cfg['implementation'] == 'ma5_ma10_price_v1':
                next_state = cfg['portfolio']['max_weight'] if short > long and price > short else state_weight
                if price < short: next_state = 0.0
            else:
                next_state = cfg['portfolio']['max_weight'] if price > p['buy_multiplier'] * short else 0.0 if price < long else state_weight
        known = np.isfinite([price, *expected.values()]).all() and bool(uni.loc[day, 'eligible'])
        require(bool(row.valid) == bool(known), f'{day}: validity differs')
        for column, value in expected.items():
            require(np.isclose(row[column], value, rtol = 1e-10, atol = 1e-7, equal_nan = True), f'{day}: {column} differs')
        if known: state_weight = next_state
        require(row.state == state_weight, f'{day}: state differs')
        selected = targets[targets.decision_date.eq(day)].set_index('instrument')
        require(selected.loc[instrument, 'weight'] == state_weight and selected.loc['CASH', 'weight'] == 1 - state_weight, f'{day}: targets differ')
        if len(examples) < 1 or (known and action != 'hold' and len(examples) < 4):
            examples.append({'date': str(day), 'close_adj': price, **expected, 'state': state_weight, 'action': action})
    return {'sessions_checked': len(factors), 'arithmetic': 'math.fsum and population variance; separate from pandas rolling', 'examples': examples}


def dependency_check(probe_directory):
    probes = read(probe_directory / 'probes.json')['probes']; by_name = {p['endpoint']: p for p in probes}
    cninfo = read(probe_directory / 'cninfo_shares/response-000.bin')
    cninfo_rows = cninfo['records']
    names = pd.read_parquet(by_name['sz_names']['raw_file']); universe = pd.read_parquet(by_name['bs_universe']['raw_file'])
    changes = names[names['变更日期'].between('2024-04-01', '2024-06-28')]
    joined = changes.merge(universe.assign(证券代码 = universe.code.str[3:]), on = '证券代码')
    examples = joined[joined['证券代码'].isin(['300420', '000755'])]
    return {'probe_report_sha256': file_sha(probe_directory / 'probes.json'),
            'cninfo_rows': len(cninfo_rows), 'cninfo_count_matches': len(cninfo_rows) == cninfo['count'] == cninfo['total'],
            'cninfo_missing_announcement': sum(not r.get('DECLAREDATE') for r in cninfo_rows),
            'name_asof_counterexamples': examples.to_dict('records'),
            'conclusions': {'shares': 'Sample event dates found, but units differ across providers and announcement/version/completeness evidence is insufficient; no shares publication',
                            'pcf': '60 daily pcfNcfTTM sample rows archived; net-cash-flow definition and platform equivalence still require evidence; no daily field substitution',
                            'names': 'SZ exchange change events have dates, but coverage and availability still need review; BaoStock historical query returns future names',
                            'micro400_and_small_value100': 'remain blocked by data; no strategy implementation claimed'}}


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--root', default = 'data'); parser.add_argument('--runs', required = True)
    parser.add_argument('--probes', required = True); parser.add_argument('--output', required = True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    store = Store(args.root); probe_directory = Path(args.probes); baseline = read(probe_directory / 'baseline.json')
    validations = []
    def forbidden(*a, **kw): raise AssertionError('Network disabled during offline reproduction')
    for run in read(args.runs):
        require(run['status'] == 'success_limited', f'{run["run_id"]}: run did not finish')
        path = Path(run['output']); before = tree_hash(path); verified = verify_run(args.root, path)
        require(verified['status'] == 'ok', f'{run["run_id"]}: integrity failed')
        arithmetic = signal_check(args.root, path)
        with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
            again = reproduce(args.root, run['run_id'])
        require(again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0, f'{run["run_id"]}: reproduction failed')
        require(tree_hash(path) == before, 'Original run was modified')
        require(verify_run(args.root, again['run_id'])['status'] == 'ok', 'Reproduction integrity failed')
        validations.append({'run_id': run['run_id'], 'source_sha256': file_sha(path / 'source.original'), 'integrity': 'ok', 'original_unchanged': True,
                            'hand_check': arithmetic, 'reproduction': again, 'report': read(path / 'report.json')})
        print(json.dumps({'run_id': run['run_id'], 'reproduction': again['reproduction'], 'checked_sessions': arithmetic['sessions_checked']}, ensure_ascii = False), flush = True)
    require(all(store.published()['tables'].get(t) == ps for t, ps in baseline['published']['tables'].items()), 'Old partition references changed')
    require(all(file_sha(store.root / f) == sha for f, sha in baseline['controls'].items() if f != 'PUBLISHED.json'), 'Old control or run bytes changed')
    require(all(file_sha(store.root / f) == item['sha256'] and (store.root / f).stat().st_size == item['size'] and (store.root / f).stat().st_mtime_ns == item['mtime_ns'] for f, item in baseline['partitions'].items()), 'Old partition bytes or metadata changed')
    write_json(output, {'status': 'ok', 'environment': environment(), 'snapshot': read(Path(validations[0]['reproduction']['output']) / 'config.json')['snapshot_id'],
                        'published_batch': store.published()['batch_id'], 'baseline_sha256': file_sha(probe_directory / 'baseline.json'),
                        'protection': {'old_controls_checked': len(baseline['controls']) - 1, 'old_partitions_checked': len(baseline['partitions']), 'all_unchanged': True},
                        'dependencies': dependency_check(probe_directory), 'strategies': validations})
    print(json.dumps({'status': 'ok', 'output': str(output), 'strategies': len(validations)}, ensure_ascii = False))


if __name__ == '__main__': main()
