"""Probe longer history for approved variants; preserve frozen overlap unchanged."""
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
from observe.runs import file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-152153-eb05'
START, END = '2005-01-05', '2026-09-29'
INSTRUMENTS = ('600519.SH', '000333.SZ', '601111.SH', '600036.SH', '600887.SH',
               '603288.SH', '600196.SH', '600085.SH', '002415.SZ', '000651.SZ')
ENDPOINTS = (*INSTRUMENTS, 'calendar', 'index_000300')
FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST,peTTM,pbMRQ'


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    socket.setdefaulttimeout(15)
    result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'start': START, 'end': END, 'published': False}

    def save(frame, dataset, parameters):
        path = raw.save(root, 'long_history_probe_batch14', dataset, directory.name, frame)
        result['files'].append({'dataset': dataset, 'file': str(path), 'sha256': file_sha(path),
                                'rows': len(frame), 'columns': list(frame), 'parameters': parameters})

    try:
        with BaoStock(root).session() as source:
            if endpoint == 'calendar': save(source.calendar(START, END), endpoint, {'start': START, 'end': END})
            elif endpoint == 'index_000300':
                save(source.index_daily('sh.000300', START, END), endpoint, {'code': 'sh.000300', 'start': START, 'end': END, 'adjustflag': '3'})
            else:
                code = std.to_baostock(endpoint)
                params = {'code': code, 'fields': FIELDS, 'start': START, 'end': END, 'frequency': 'd', 'adjustflag': '3'}
                frame = source._rows('query_history_k_data_plus', params,
                    lambda: source.bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END, frequency='d', adjustflag='3'))
                save(frame, f'{endpoint}_bars', params)
                factors = source.adjust_factor(code, start='1990-01-01', end=END)
                save(factors, f'{endpoint}_factors', {'code': code, 'start': '1990-01-01', 'end': END})
        result['status'] = 'success' if all(item['rows'] for item in result['files']) else 'empty_component'
    except Exception as exc:
        result.update(status='partial' if result['files'] else 'failed', error=f'{type(exc).__name__}: {exc}'[:1200])
    finally: write_json(folder / 'result.json', result)
    return result


def probe(root, directory):
    require((directory / 'baseline.json').is_file(), 'Batch baseline missing')
    output = directory / 'probe-results.json'
    if any(path.exists() for path in (output, directory / 'probes', directory / 'existing-api.json')): raise FileExistsError(output)
    import baostock as bs
    write_json(directory / 'existing-api.json', {'version': bs.__version__,
        'signature': str(inspect.signature(bs.query_history_k_data_plus)), 'source': inspect.getsource(bs.query_history_k_data_plus),
        'fields': FIELDS, 'stock_query': 'Existing SDK query_history_k_data_plus through BaoStock._rows/session, no source execution'})
    records = bounded_requests(root, directory, ENDPOINTS)
    write_json(output, {'results': records, 'snapshot': SNAPSHOT, 'published': False,
        'api_file': str(directory / 'existing-api.json'), 'api_sha256': file_sha(directory / 'existing-api.json')})
    return {'status': 'archived', 'output': str(output)}


def bounded_requests(root, directory, endpoints):
    records = []
    for endpoint in endpoints:
        command = [sys.executable, '-m', 'scripts.probe_strategy_batch14', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            record = {'endpoint': endpoint, 'returncode': process.returncode, 'stdout': process.stdout[-1500:], 'stderr': process.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            record = {'endpoint': endpoint, 'status': 'timeout', 'timeout_seconds': 120}
        terminal = directory / 'probes' / endpoint / 'result.json'
        record['result'] = read(terminal) if terminal.exists() else {'status': record.get('status', 'failed'), 'files': [], 'error': 'No terminal worker response; raw request evidence retained'}
        records.append(record)
        print(json.dumps({'endpoint': endpoint, 'status': record['result']['status'], 'rows': [i['rows'] for i in record['result']['files']]}), flush=True)
        archive(root, f'第十四批{endpoint}长历史探测{record["result"]["status"]}，新原始输入独立归档')
    return records


def retry(root, directory):
    original = directory / 'probe-results.json'; first = read(original)
    require(file_sha(Path(first['api_file'])) == first['api_sha256'], 'API evidence changed')
    target = directory / 'retry-pass'; target.mkdir(exist_ok=False)
    endpoints = [r['endpoint'] for r in first['results'] if r['result']['status'] != 'success']
    records = bounded_requests(root, target, endpoints)
    output = directory / 'retry-results.json'
    require(not output.exists(), 'Retry output already exists')
    write_json(output, {'results': records, 'original_file': str(original), 'original_sha256': file_sha(original),
        'published': False, 'policy': 'One independent bounded retry; first-pass failures remain immutable'})
    return {'status': 'archived', 'output': str(output)}


def checked_artifacts(directory):
    original = directory / 'probe-results.json'; probes = read(original)
    require(file_sha(Path(probes['api_file'])) == probes['api_sha256'], 'API evidence changed')
    passes = [(original, probes)]
    repeated = directory / 'retry-results.json'
    if repeated.exists():
        retry_doc = read(repeated)
        require(Path(retry_doc['original_file']).resolve() == original.resolve() and
                retry_doc['original_sha256'] == file_sha(original), 'First-pass evidence changed')
        passes.append((repeated, retry_doc))
    artifacts = {}; evidence = []
    for path, doc in passes:
        evidence.append({'file': str(path), 'sha256': file_sha(path)})
        for record in doc['results']:
            for item in record['result']['files']:
                require(file_sha(Path(item['file'])) == item['sha256'], 'Raw probe changed')
                frame = pd.read_parquet(item['file'])
                require(len(frame) == item['rows'] and list(frame) == item['columns'], 'Raw probe schema changed')
                if len(frame): artifacts[item['dataset']] = ({**item, 'probe_file': str(path)}, frame)
    return artifacts, evidence


def equal_cells(left, right, columns):
    differences = {}
    for column in columns:
        a, b = left[column], right[column]
        equal = a.eq(b) | (a.isna() & b.isna())
        differences[column] = int((~equal).sum())
    return differences


def analyze(root, directory):
    output = directory / 'input-analysis.json'
    if output.exists(): raise FileExistsError(output)
    artifacts, evidence = checked_artifacts(directory)
    require('calendar' in artifacts, 'Calendar unavailable; historical coverage cannot be inferred')
    store = Store(root); state = store.state(SNAPSHOT)
    calendar = std.calendar(artifacts['calendar'][1]); days = sorted(calendar.loc[calendar.is_open, 'date'])
    old_calendar = store.load_state(state, 'calendar')
    common = old_calendar.merge(calendar, on='date', how='inner', validate='one_to_one', suffixes=('_old', '_new'))
    calendar_info = {'rows': len(calendar), 'sessions': len(days), 'common_dates': len(common), 'old_dates': len(old_calendar),
        'overlap_differences': int(common.is_open_old.ne(common.is_open_new).sum()),
        'first': str(calendar.date.min()), 'last': str(calendar.date.max())}
    profiles = []; master = store.load_state(state, 'instruments').set_index('instrument')
    actions = store.load_state(state, 'corp_actions', filters=[('instrument', 'in', list(INSTRUMENTS))])
    factors = store.load_state(state, 'adj_factors', filters=[('instrument', 'in', list(INSTRUMENTS))])
    coverage = store.load_state(state, 'adj_coverage', filters=[('instrument', 'in', list(INSTRUMENTS))])
    for instrument in INSTRUMENTS:
        if f'{instrument}_bars' not in artifacts:
            profiles.append({'instrument': instrument, 'status': 'missing_bars'}); continue
        source, frame = artifacts[f'{instrument}_bars']; bars = std.daily(frame)
        require(not bars.duplicated(['date', 'instrument']).any() and set(bars.instrument) == {instrument}, 'Unexpected instrument or duplicate dates')
        listing = pd.to_datetime(master.loc[instrument, 'list_date']).date()
        expected = {d for d in days if d >= listing}
        old = store.load_state(state, 'bars_1d', filters=[('instrument', '==', instrument)])
        joined = old.merge(bars, on=['date', 'instrument'], how='inner', validate='one_to_one', suffixes=('_old', '_new'))
        columns = list(old.columns.difference(['date', 'instrument']))
        left = joined[[f'{c}_old' for c in columns]].set_axis(columns, axis=1)
        right = joined[[f'{c}_new' for c in columns]].set_axis(columns, axis=1)
        dates = set(bars.date); finite = bars.loc[bars.is_trading, ['open', 'high', 'low', 'close', 'preclose']].to_numpy(float)
        events = actions[actions.instrument.eq(instrument) & (actions.ex_date >= date.fromisoformat(START))]
        rights = events[events.rights_ratio.gt(0)]
        profile = {'instrument': instrument, 'status': 'profiled', 'raw_file': source['file'], 'raw_sha256': source['sha256'],
            'rows': len(bars), 'first': str(bars.date.min()), 'last': str(bars.date.max()), 'listing': str(listing),
            'pre2021_rows': int((bars.date < date(2021, 1, 1)).sum()), 'paused_rows': int((~bars.is_trading).sum()),
            'missing_sessions': sorted(str(d) for d in expected - dates), 'extra_sessions': sorted(str(d) for d in dates - expected),
            'raw_trade_flags': sorted(frame.tradestatus.unique()), 'raw_st_flags': sorted(frame.isST.unique()),
            'nonfinite_trading_price_cells': int((~np.isfinite(finite)).sum()), 'nonpositive_trading_price_cells': int((finite <= 0).sum()),
            'old_rows': len(old), 'overlap_rows': len(joined), 'exact_overlap_differences': equal_cells(left, right, columns),
            'coverage': coverage[coverage.instrument.eq(instrument)].astype(str).to_dict('records'),
            'rights_events': rights.astype(str).to_dict('records'),
            'execution_limit': 'Book has no rights subscription model; do not run across rights events without an explicit policy'}
        if f'{instrument}_factors' in artifacts:
            _, incoming = artifacts[f'{instrument}_factors']; incoming = std.adj_factors(incoming)
            old_factors = factors[factors.instrument.eq(instrument)].copy()
            compared = old_factors.merge(incoming, on=['instrument', 'ex_date'], how='outer', validate='one_to_one', suffixes=('_old', '_new'), indicator=True)
            overlap = compared[compared._merge.eq('both')]
            profile['factor_comparison'] = {'old_events': len(old_factors), 'new_events': len(incoming),
                'unmatched': compared[~compared._merge.eq('both')].astype(str).to_dict('records'),
                'exact_factor_differences': int(overlap.back_factor_old.ne(overlap.back_factor_new).sum())}
        profiles.append(profile)
    if 'index_000300' in artifacts:
        _, frame = artifacts['index_000300']; frame['date'] = pd.to_datetime(frame.date).dt.date
        old = store.load_state(state, 'index_1d'); old = old[old['index'].eq('000300.SH')]
        joined = old.merge(frame, on='date', validate='one_to_one', suffixes=('_old', '_new'))
        columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
        a = joined[[f'{c}_old' for c in columns]].astype(float).set_axis(columns, axis=1)
        b = joined[[f'{c}_new' for c in columns]].astype(float).set_axis(columns, axis=1)
        index_info = {'rows': len(frame), 'first': str(frame.date.min()), 'last': str(frame.date.max()),
            'missing_sessions': sorted(str(d) for d in set(days) - set(frame.date)),
            'overlap_days': len(joined), 'old_days': len(old), 'exact_overlap_differences': equal_cells(a, b, columns)}
    else: index_info = {'status': 'missing'}
    result = {'snapshot': SNAPSHOT, 'probe_evidence': evidence, 'calendar': calendar_info, 'index': index_info, 'stocks': profiles,
        'published': False, 'limits': ['Current retrievals are new evidence; old partitions and overlapping rows remain read-only',
            'This probe does not establish historical names, financial versions or a complete historical market universe',
            'Corporate-action dates and pre-2015 fee assumptions retain existing explicit approximation limits']}
    write_json(output, result)
    archive(root, '第十四批长历史覆盖、重叠字段、复权因子及早期配股边界核查已存档，发布与回测尚未执行')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe', 'worker', 'retry', 'analyze'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch14/20261005-long-history'))
    parser.add_argument('--endpoint', choices=ENDPOINTS)
    args = parser.parse_args()
    if args.action == 'worker':
        if not args.endpoint: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = {'probe': probe, 'retry': retry, 'analyze': analyze}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False), flush=True)
