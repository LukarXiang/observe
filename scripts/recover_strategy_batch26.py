"""Recover accepted PS/PCF raw fields and archive component evidence, without trades."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import secrets
from unittest.mock import patch

import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.data.valuations import FIELDS, import_valuations, load_valuations, query_valuations
from observe.features import factor_frame, panel
from observe.runs import canonical, file_sha
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

UPSTREAM = Path('docs/handoff/2026-10-06-batch24-verification.json')
ACCEPTED = Path('data/staging/strategies-batch24/20261006-volatility-value-raw')
MANIFEST = ACCEPTED / 'raw-value-audit.json'
FILES = ('src/observe/data/valuations.py', 'src/observe/data/store.py', 'src/observe/features.py', 'src/observe/factors/expr.py',
    'src/observe/research.py', 'src/observe/strategy/fields.py', 'src/observe/strategy/spec.py', 'src/observe/strategy/run.py',
    'src/observe/cli.py', 'scripts/recover_strategy_batch26.py', 'scripts/verify_offline_tests.py', 'tests/unit/test_valuations.py',
    'tests/integration/test_valuation_recovery.py', 'tests/unit/test_batch26_archive.py', 'tests/unit/test_offline_verification.py')
CORE = {'baseline.json', 'input-binding.json', 'raw-manifest.json', 'recovery.json', 'component-research.json',
    'component-offline.json', 'offline-verification.json', 'sandbox-test-diagnostic.json'}


def offline_names(implementation):
    key = hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest()[:20]
    return f'component-offline-{key}.json', f'offline-verification-{key}.json'


def start(root, directory):
    require((directory / 'baseline.json').is_file() and not (directory / 'input-binding.json').exists(), 'Baseline missing/already bound')
    receipt = read(UPSTREAM); require(receipt['status'] == 'ok', 'Upstream not accepted')
    for name in ('raw-value-audit.json', 'value-input.parquet', 'source-reviews/review.json'):
        require(file_sha(ACCEPTED / name) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
    reference = read(ACCEPTED / 'source-reviews/review.json'); copies = []
    folder = directory / 'references'; folder.mkdir()
    for k, row in enumerate(reference['references']):
        path = Path(row['copy']); require(file_sha(path) == row['sha256'], 'Reference changed')
        copied = folder / f'{k}-{path.name}'; shutil.copyfile(path, copied)
        copies.append({'file': str(copied), 'sha256': file_sha(copied), 'original': row,
            'use': 'BaoStock separate psTTM/pcfNcfTTM and Qlib sample cross-section; core uses existing observe cs_zscore'})
    source = reference['sources'][1]; require(file_sha(source['source_copy']) == source['source_sha256'], 'Original90 source changed')
    copied = folder / 'original90.source'; shutil.copyfile(source['source_copy'], copied)
    manifest = directory / 'raw-manifest.json'; shutil.copyfile(MANIFEST, manifest)
    binding = {'upstream': str(UPSTREAM), 'upstream_sha256': file_sha(UPSTREAM), 'baseline_sha256': file_sha(directory / 'baseline.json'),
        'accepted_manifest': str(MANIFEST), 'manifest_file': str(manifest), 'manifest_sha256': file_sha(manifest),
        'ten_stock_input': str(ACCEPTED / 'value-input.parquet'), 'ten_stock_sha256': file_sha(ACCEPTED / 'value-input.parquet'),
        'references': copies, 'original90': {'file': str(copied), 'sha256': file_sha(copied), 'source_path': source['source_path']},
        'strict_usable': False, 'original_pool_missing': True, 'not_a_backtest': True}
    save(directory / 'input-binding.json', binding)
    return archive(root, 'Batch26 accepted full-market raw manifest and reference implementations bound; PS core recovery in development')


def binding(directory):
    doc = read(directory / 'input-binding.json')
    require(file_sha(doc['upstream']) == doc['upstream_sha256'] and file_sha(doc['manifest_file']) == doc['manifest_sha256'] and
        file_sha(doc['accepted_manifest']) == doc['manifest_sha256'] and file_sha(doc['ten_stock_input']) == doc['ten_stock_sha256'] and
        file_sha(directory / 'baseline.json') == doc['baseline_sha256'], 'Batch26 input binding differs')
    for row in [*doc['references'], doc['original90']]: require(file_sha(row['file']) == row['sha256'], 'Frozen reference/source changed')
    return doc


def recover(root, directory):
    doc = binding(directory); require(not (directory / 'recovery.json').exists(), 'Recovery already archived')
    result = import_valuations(root, doc['manifest_file'], doc['manifest_sha256'], log=lambda message: print(message, file=sys.stderr, flush=True))
    with operation_lock(root, DATA_WRITER): sid = Store(root).snapshot('Batch26 vendor-final PS/PCF recovery; strict unusable')
    save(directory / 'recovery.json', {**result, 'snapshot': sid})
    save(directory / 'progress-recovery.json', archive(root, 'Batch26 PS and net-cash-flow PCF immutable tables recovered; no original90 trading claim'))
    return result


def compute(root, directory):
    doc = binding(directory); recovery = read(directory / 'recovery.json'); store = Store(root); state = store.state(recovery['snapshot'])
    values, used = load_valuations(store, state, recovery['first'], recovery['last'], FIELDS, 'provider_final')
    require(len(values) == recovery['rows'] and not values.strict_usable.any(), 'Recovered rows/evidence differ')
    frozen = pd.read_parquet(doc['ten_stock_input']); frozen['date'] = pd.to_datetime(frozen.date).dt.date
    sample = values.merge(frozen[['date', 'instrument']], on=['date', 'instrument'], validate='one_to_one')[['date', 'instrument', *FIELDS]]
    pd.testing.assert_frame_equal(sample.sort_values(['date', 'instrument']).reset_index(drop=True),
        frozen[['date', 'instrument', *FIELDS]].sort_values(['date', 'instrument']).reset_index(drop=True))
    dates = sorted(set(values.date)); names = sorted(set(values.instrument))
    view = values[['date', 'instrument', 'is_trading', *FIELDS]]
    wide = panel(view, dates, names, fields=FIELDS)
    eligible = view.pivot(index='date', columns='instrument', values='is_trading').reindex(index=dates, columns=names).fillna(False).astype(bool)
    fset = {'min_obs_ratio': 1., 'factors': [{'name': f, 'expr': f'cs_zscore({f})'} for f in FIELDS]}
    actual = factor_frame(fset, wide, eligible).sort_values(['date', 'instrument']).reset_index(drop=True)
    daily = []; max_difference = {f: 0. for f in FIELDS}
    by_day = {day: group for day, group in actual.groupby('date', sort=False)}
    for day, frame in view.groupby('date', sort=True):
        frame = frame[frame.is_trading].sort_values('instrument'); packet = {'date': str(day), 'trading_rows': len(frame), 'fields': {}}
        actual_day = by_day[day].set_index('instrument')
        for field in FIELDS:
            x = frame[field].to_numpy(dtype=float); valid = np.isfinite(x); finite = x[valid]
            mean = math.fsum(map(float, finite)) / len(finite) if len(finite) else math.nan
            sd = math.sqrt(math.fsum((float(v) - mean) ** 2 for v in finite) / (len(finite) - 1)) if len(finite) > 1 else math.nan
            reference = (x - mean) / sd if sd > 0 else np.full(len(x), np.nan)
            output = actual_day.loc[frame.instrument, field].to_numpy(dtype=float)
            require(np.array_equal(np.isfinite(reference), np.isfinite(output)), 'Zscore valid mask differs')
            error = float(np.max(np.abs(reference[valid] - output[valid]))) if valid.any() and sd > 0 else 0.
            require(error < 1e-10, 'Sample zscore differs'); max_difference[field] = max(max_difference[field], error)
            packet['fields'][field] = {'finite': int(np.isfinite(output).sum()), 'blank': int(np.isnan(output).sum()),
                'max_abs_difference': error, 'values_sha256': hashlib.sha256(output.astype('<f8').tobytes()).hexdigest()}
        daily.append(packet)
    strict = query_valuations(root, recovery['snapshot'], recovery['first'], recovery['first'], ['600519.SH'])['coverage']
    require(strict['returned_rows'] == 0 and strict['strict_excluded'] == strict['source_rows'], 'Strict query leaked values')
    return {'snapshot': recovery['snapshot'], 'source_manifest_sha256': doc['manifest_sha256'], 'input_rows': len(values),
        'ten_stock_component_rows_match': len(sample), 'daily_cross_sections': len(daily), 'trading_factor_rows': len(actual),
        'max_abs_difference': max_difference, 'daily': daily, 'strict_query': strict,
        'used': {t: {p: {**v, 'file_sha256': file_sha(store.root / v['file'])} for p, v in mapping.items()} for t, mapping in used.items()},
        'not_a_backtest': True, 'original_strategy_complete': False,
        'limits': ['Vendor-final values have no historical disclosure/revision vintage; platform definitions unproved',
            'Cross-sections cover raw market trading rows, not original90 index universe or targets',
            'No pre2021 filling, original90 trades, cost or NAV claim; sole Book untouched']}


def study(root, directory):
    save(directory / 'component-research.json', canonical(compute(root, directory)))
    return archive(root, 'Batch26 full-market sample-z components independently verified; old ten-stock input exact; original90 remains blocked')


def worker(root, directory, offline_output='component-offline.json'):
    require(offline_output in ('component-offline.json', 'component-offline-final.json', 'component-offline-verified.json') or
        re.fullmatch(r'component-offline-[0-9a-f]{20}\.json', offline_output), 'Invalid offline output')
    def blocked(*args, **kwargs): raise AssertionError('Offline batch26 attempted network')
    with patch.object(socket, 'socket', blocked), patch.object(socket, 'create_connection', blocked):
        result = canonical(compute(root, directory))
    save(directory / offline_output, result)


def offline(root, directory, final=False):
    before = {name: file_sha(name) for name in FILES}
    output_name, receipt_name = offline_names(before) if final else ('component-offline.json', 'offline-verification.json')
    prefix = Path(receipt_name).stem if final else 'offline'
    command = [sys.executable, '-m', 'scripts.recover_strategy_batch26', 'worker', '--root', str(root), '--directory', str(directory)]
    if final: command += ['--offline-output', output_name]
    with (directory / f'{prefix}.stdout').open('x') as out, (directory / f'{prefix}.stderr').open('x') as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    require(before == {name: file_sha(name) for name in FILES}, 'Offline implementation changed during worker')
    require((directory / 'component-research.json').read_bytes() == (directory / output_name).read_bytes(), 'Offline component bytes differ')
    result = {'result': 'match', 'differences': 0, 'sha256': file_sha(directory / 'component-research.json'), 'socket_network_disabled': True,
        'command': command, 'not_a_backtest': True, 'output_file': str(directory / output_name), 'implementation_sha256': before}
    save(directory / receipt_name, result); return result


def recheck(root, directory): return offline(root, directory, final=True)


def required_commands():
    return [
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/recover_strategy_batch26.py',
            'scripts/verify_offline_tests.py', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(directory); require(not (directory / 'checked-state.json').exists(), 'Checks already frozen')
    require(all((directory / name).is_file() for name in CORE | set(offline_names({name: file_sha(name) for name in FILES}))), 'Required core evidence missing')
    folder = directory / f'checks-{secrets.token_hex(8)}'; folder.mkdir()
    commands = required_commands()
    rows = []
    before = {name: file_sha(name) for name in FILES}
    for k, command in enumerate(commands):
        path = folder / f'{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row = {'command': command, 'returncode': result.returncode, 'log': str(path), 'log_sha256': file_sha(path)}
        save(folder / f'{k}.json', row); rows.append(row)
        require(result.returncode == 0, f'Check failed: {command}; see {path}')
    require(before == {name: file_sha(name) for name in FILES}, 'Implementation changed during checks')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked implementation archive exists'); shutil.copyfile(name, copied)
        require(file_sha(copied) == before[name], 'Checked implementation copy differs')
    evidence = {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}
    checked = {'status': 'passed', 'commands': rows, 'implementation_sha256': before, 'evidence_sha256': evidence,
        'test_scope': 'Full pytest via process-local pipe wakeup; ordinary pytest socket IPC denied in sandbox'}
    save(directory / 'checked-state.json', checked)
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json')
    require(checked['status'] == 'passed' and len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']), 'Required checks did not pass')
    require([r['command'] for r in checked['commands']] == required_commands(), 'Required check commands differ')
    require(CORE <= checked['evidence_sha256'].keys(), 'Required core evidence not checked')
    require(checked['implementation_sha256'] == {name: file_sha(name) for name in FILES}, 'Checked implementation drift')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked implementation copy drift')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, f'Checked evidence drift: {name}')
    for row in checked['commands']: require(file_sha(row['log']) == row['log_sha256'], 'Check log changed')
    output_name, receipt_name = offline_names(checked['implementation_sha256'])
    require({output_name, receipt_name} <= checked['evidence_sha256'].keys(), 'Final offline evidence not checked')
    doc = binding(directory); component = read(directory / 'component-research.json'); offline_result = read(directory / receipt_name)
    require(offline_result['result'] == 'match' and offline_result['differences'] == 0 and offline_result['socket_network_disabled'] is True, 'Offline verification not accepted')
    require(offline_result['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / output_name), 'Offline evidence changed')
    require(offline_result['implementation_sha256'] == checked['implementation_sha256'], 'Offline implementation differs from checked code')
    require(canonical(compute(root, directory)) == component, 'Recovered inputs/component changed after checks')
    protection = protect(root, directory)
    recovery = read(directory / 'recovery.json'); receipt = read(recovery['receipt']); audit = read(recovery['audit'])
    require(audit['status'] == 'passed' and receipt['audit_sha256'] == file_sha(recovery['audit']), 'Recovery audit changed')
    require({k: v for k, v in receipt['result'].items() if k != 'status'} == {k: v for k, v in recovery.items() if k not in ('status', 'snapshot')}, 'Recovery receipt differs')
    progress = archive(root, 'Batch26 PS/PCF core recovery, independent components and offline match archived; original90 pool still unavailable')
    evidence_files = [p for p in directory.rglob('*') if p.is_file() and p.name not in ('offline.stdout', 'offline.stderr')]
    result = {'status': 'ok', 'reviews': {'manually_reviewed': progress['manually_reviewed'], 'sources': progress['sources'], 'original90_complete': False},
        'progress': progress, 'recovery': recovery, 'component': {k: v for k, v in component.items() if k not in ('daily', 'used')},
        'offline': offline_result, 'protection': protection, 'input_binding': doc,
        'checks': {'implementation_sha256': {name: file_sha(name) for name in FILES},
            'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in evidence_files},
            'verification': checked}, 'limitations': component['limits']}
    save(Path('docs/handoff/2026-10-06-batch26-verification.json'), result)
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'snapshot': recovery['snapshot'], 'batch_id': recovery['batch_id'], 'rows': recovery['rows']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'recover', 'study', 'offline', 'recheck', 'worker', 'checks', 'finish'])
    parser.add_argument('--offline-output', default='component-offline.json')
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch26/20261006-valuation-recovery')
    args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.offline_output) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
