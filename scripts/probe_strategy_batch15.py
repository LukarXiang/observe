"""Bounded RSI-history probes; append new evidence without replacing frozen inputs."""
import argparse
from datetime import date
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pandas as pd

from observe.data import raw, standardize as std
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.data.prices import with_adjusted
from observe.execution import sessions
from observe.runs import file_sha, write_json
from observe.strategy_rsi import UNIQUE_POOL
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import FIELDS, INSTRUMENTS, checked_artifacts, equal_cells
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-191431-aa71'
START, END = '2019-01-01', '2026-09-29'
MISSING = tuple(i for i in UNIQUE_POOL if i not in INSTRUMENTS)
CONFIG = Path('configs/strategies/rsi_slots_corrected_batch7.yaml')


def initialize(root, directory):
    checkpoint(root, directory, note='Batch15 RSI history checkpoint: preserve batch14 and all earlier experiments')
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['tables'] == state['tables'], 'Published inputs differ from batch14 snapshot')
    cfg = read_config()
    master = store.load_state(state, 'instruments').set_index('instrument')
    actions = store.load_state(state, 'corp_actions', filters=[('instrument', 'in', list(UNIQUE_POOL))])
    write_json(directory / 'precheck.json', {'snapshot': SNAPSHOT, 'start': START, 'end': END,
        'config_file': str(CONFIG), 'config_sha256': file_sha(CONFIG), 'source_file': cfg['source_path'],
        'source_sha256': file_sha(cfg['source_path']), 'missing_history_instruments': list(MISSING),
        'reused_history_instruments': sorted(set(UNIQUE_POOL) & set(INSTRUMENTS)),
        'latest_listing': str(max(master.loc[list(UNIQUE_POOL), 'list_date'])),
        'rights_since_2019': actions[actions.rights_ratio.gt(0) & actions.ex_date.ge(date(2019, 1, 1))].astype(str).to_dict('records'),
        'plan': 'Query missing 39 stocks through existing SDK; require exact frozen overlap/factors; retain old rows in new yearly partitions; keep approved RSI/Book/cost rules',
        'first_decision': 'Determined from complete 61 traded rows after publication, not returns',
        'partial_market_only': True, 'published': False})
    return {'status': 'initialized', 'output': str(directory)}


def read_config():
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def custody(directory):
    doc = read(directory / 'precheck.json')
    require(file_sha(CONFIG) == doc['config_sha256'], 'Original RSI config changed')
    require(file_sha(doc['source_file']) == doc['source_sha256'], 'Original RSI source changed')
    return doc


def worker(root, directory, instrument):
    require(instrument in MISSING, 'Unknown probe instrument')
    folder = directory / 'probes' / instrument; folder.mkdir(parents=True, exist_ok=False)
    socket.setdefaulttimeout(15)
    result = {'instrument': instrument, 'status': 'failed', 'files': [], 'start': START, 'end': END, 'published': False}
    def save(frame, dataset, parameters):
        path = raw.save(root, 'rsi_history_probe_batch15', dataset, directory.name, frame)
        result['files'].append({'dataset': dataset, 'file': str(path), 'sha256': file_sha(path),
            'rows': len(frame), 'columns': list(frame), 'parameters': parameters})
    try:
        with BaoStock(root).session() as source:
            code = std.to_baostock(instrument)
            params = {'code': code, 'fields': FIELDS, 'start': START, 'end': END, 'frequency': 'd', 'adjustflag': '3'}
            frame = source._rows('query_history_k_data_plus', params,
                lambda: source.bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END, frequency='d', adjustflag='3'))
            save(frame, f'{instrument}_bars', params)
            save(source.adjust_factor(code, start='1990-01-01', end=END), f'{instrument}_factors', {'code': code, 'start': '1990-01-01', 'end': END})
        result['status'] = 'success' if all(item['rows'] for item in result['files']) else 'empty_component'
    except Exception as exc:
        result.update(status='partial' if result['files'] else 'failed', error=f'{type(exc).__name__}: {exc}'[:1200])
    finally: write_json(folder / 'result.json', result)
    return result


def bounded(root, directory, instruments):
    results = []
    for instrument in instruments:
        cmd = [sys.executable, '-m', 'scripts.probe_strategy_batch15', 'worker', '--root', str(root), '--directory', str(directory), '--instrument', instrument]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            record = {'instrument': instrument, 'returncode': process.returncode, 'stdout': process.stdout[-1200:], 'stderr': process.stderr[-1200:]}
        except subprocess.TimeoutExpired:
            record = {'instrument': instrument, 'status': 'timeout', 'timeout_seconds': 120}
        terminal = directory / 'probes' / instrument / 'result.json'
        record['result'] = read(terminal) if terminal.exists() else {'status': record.get('status', 'failed'), 'files': [], 'error': 'No terminal result; raw request evidence retained'}
        results.append(record)
        print(json.dumps({'instrument': instrument, 'status': record['result']['status'], 'rows': [i['rows'] for i in record['result']['files']]}), flush=True)
        archive(root, f'Batch15 RSI {instrument} historical probe {record["result"]["status"]}; new raw evidence archived')
    return results


def probe(root, directory):
    custody(directory); output = directory / 'probe-results.json'
    for path in (output, directory / 'probes', directory / 'existing-api.json'):
        if path.exists(): raise FileExistsError(path)
    import baostock as bs
    api = directory / 'existing-api.json'
    write_json(api, {'version': bs.__version__, 'signature': str(inspect.signature(bs.query_history_k_data_plus)),
        'source': inspect.getsource(bs.query_history_k_data_plus), 'fields': FIELDS, 'start': START, 'end': END})
    results = bounded(root, directory, MISSING)
    write_json(output, {'results': results, 'api_file': str(api), 'api_sha256': file_sha(api), 'snapshot': SNAPSHOT, 'published': False})
    return {'status': 'archived', 'output': str(output)}


def retry(root, directory):
    custody(directory); checked_artifacts(directory)
    output = directory / 'retry-results.json'; require(not output.exists(), 'Retry already archived')
    original = directory / 'probe-results.json'; first = read(original)
    target = directory / 'retry-pass'; target.mkdir(exist_ok=False)
    instruments = [r['instrument'] for r in first['results'] if r['result']['status'] != 'success']
    results = bounded(root, target, instruments)
    write_json(output, {'results': results, 'original_file': str(original), 'original_sha256': file_sha(original),
        'published': False, 'policy': 'One bounded independent retry; original failures retained'})
    return {'status': 'archived', 'output': str(output)}


def analyzed_inputs(root, directory):
    custody(directory); artifacts, evidence = checked_artifacts(directory)
    store = Store(root); state = store.state(SNAPSHOT); days = sessions(store.load_state(state, 'calendar'))
    master = store.load_state(state, 'instruments').set_index('instrument')
    filters = [('instrument', 'in', list(UNIQUE_POOL))]
    frozen = store.load_state(state, 'bars_1d', filters=filters)
    factors = store.load_state(state, 'adj_factors', filters=filters)
    coverage = store.load_state(state, 'adj_coverage', filters=filters)
    profiles, prefixes = [], []
    for instrument in MISSING:
        names = [f'{instrument}_bars', f'{instrument}_factors']
        if any(name not in artifacts for name in names):
            profiles.append({'instrument': instrument, 'status': 'missing_component', 'missing': [name for name in names if name not in artifacts]})
            continue
        raw_meta, raw_bars = artifacts[names[0]]; _, raw_factors = artifacts[names[1]]
        bars = std.daily(raw_bars); old = frozen[frozen.instrument.eq(instrument)]
        expected = {d for d in days if max(date.fromisoformat(START), master.loc[instrument, 'list_date']) <= d <= date.fromisoformat(END)}
        require(set(bars.instrument) == {instrument} and not bars.duplicated(['date', 'instrument']).any(), 'Unexpected or duplicate bars')
        joined = old.merge(bars, on=['date', 'instrument'], how='inner', validate='one_to_one', suffixes=('_old', '_new'))
        columns = list(old.columns.difference(['date', 'instrument']))
        left = joined[[f'{c}_old' for c in columns]].set_axis(columns, axis=1)
        right = joined[[f'{c}_new' for c in columns]].set_axis(columns, axis=1)
        incoming = std.adj_factors(raw_factors); original = factors[factors.instrument.eq(instrument)]
        factor_rows = original.merge(incoming, on=['instrument', 'ex_date'], how='outer', validate='one_to_one', suffixes=('_old', '_new'), indicator=True)
        shared = factor_rows[factor_rows._merge.eq('both')]
        price = bars.loc[bars.is_trading, ['open', 'high', 'low', 'close', 'preclose']].to_numpy(float)
        amounts = bars[['volume', 'amount']].to_numpy(float)
        adjusted = with_adjusted(bars, original, coverage[coverage.instrument.eq(instrument)])
        profile = {'instrument': instrument, 'status': 'profiled', 'raw_file': raw_meta['file'], 'raw_sha256': raw_meta['sha256'],
            'rows': len(bars), 'prefix_rows': int((bars.date < date(2021, 1, 1)).sum()),
            'missing_sessions': sorted(str(d) for d in expected - set(bars.date)), 'extra_sessions': sorted(str(d) for d in set(bars.date) - expected),
            'old_rows': len(old), 'overlap_rows': len(joined), 'exact_overlap_differences': equal_cells(left, right, columns),
            'factor_unmatched_events': int(factor_rows._merge.ne('both').sum()),
            'factor_exact_differences': int(shared.back_factor_old.ne(shared.back_factor_new).sum()),
            'invalid_price_cells': int((~np.isfinite(price) | (price <= 0)).sum()),
            'invalid_amount_cells': int((~np.isfinite(amounts) | (amounts < 0)).sum()),
            'invalid_trade_flags': sorted(set(raw_bars.tradestatus.astype(str)) - {'0', '1'}),
            'invalid_st_flags': sorted(set(raw_bars.isST.astype(str)) - {'0', '1'}),
            'adjusted_trading_prices_missing': int(adjusted.loc[adjusted.is_trading, 'close_adj'].isna().sum()),
            'paused_rows': int((~bars.is_trading).sum()),
            'coverage': coverage[coverage.instrument.eq(instrument)].astype(str).to_dict('records')}
        valid = not any(profile[k] for k in ('missing_sessions', 'extra_sessions', 'factor_unmatched_events', 'factor_exact_differences',
            'invalid_price_cells', 'invalid_amount_cells', 'invalid_trade_flags', 'invalid_st_flags', 'adjusted_trading_prices_missing'))
        valid = valid and profile['old_rows'] == profile['overlap_rows'] and not any(profile['exact_overlap_differences'].values())
        profile['status'] = 'ready' if valid else 'blocked'
        profiles.append(profile)
        if valid: prefixes.append(bars[bars.date < date(2021, 1, 1)].copy())
    prefix = pd.concat(prefixes, ignore_index=True) if prefixes else frozen.iloc[:0].copy()
    return prefix, profiles, evidence


def analyze(root, directory):
    output = directory / 'input-analysis.json'; require(not output.exists(), 'Analysis already archived')
    prefix, profiles, evidence = analyzed_inputs(root, directory)
    result = {'status': 'ready' if len(profiles) == len(MISSING) and all(p['status'] == 'ready' for p in profiles) else 'blocked',
        'snapshot': SNAPSHOT, 'probe_evidence': evidence, 'stocks': profiles, 'prefix_rows_ready': len(prefix),
        'original_adjustments_reused': True, 'old_overlap_retained': True, 'published': False,
        'limits': ['Selected pool history only; source-platform and corporate-action-date equality remains approximate',
                   'New retrievals are independently archived; old rows, factors and experiments remain unchanged']}
    write_json(output, result); archive(root, f'Batch15 RSI historical input analysis {result["status"]}, {len(prefix)} candidate prefix rows')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'probe', 'retry', 'worker', 'analyze'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch15/20261005-rsi-history'))
    parser.add_argument('--instrument', choices=MISSING)
    args = parser.parse_args()
    if args.action == 'worker':
        if not args.instrument: parser.error('--instrument required')
        result = worker(args.root, args.directory, args.instrument)
    else: result = {'initialize': initialize, 'probe': probe, 'retry': retry, 'analyze': analyze}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False), flush=True)
