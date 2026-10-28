"""Execute the five approved, preflighted short islands without stitching NAVs."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from observe.data.store import Store
from observe.execution import build, load
from observe.runs import file_sha
from observe.strategies import StrategyConfig
from scripts.archive_strategy_progress import archive
from scripts.preflight_pair66_short_windows import EXPECTED, PREPARED, PREPARED_SHA
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.run_strategy_batch8 import signal_check
from scripts.supplement_pair66_history import CONFIG, CONFIG_SHA, SNAPSHOT, SOURCE_SHA
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-history-supplement/20261011-pair66-short-islands')
PREFLIGHT = Path('data/staging/strategies-history-supplement/20261011-pair66-short-islands-preflight')
PREFLIGHT_SHA = '77f31778c26dbe2cb5f9e3aea29f3b590ae369ccaad31da4973b8db7d69c75ce'
PARENT = Path('docs/handoff/2026-10-11-batch96-final-review.json')
PARENT_SHA = 'b0c022f54cf69dc0b8d75000b1a9ff6398df49f5050ff8e0fe67dfe1afe42f10'
BASELINE = Path('data/staging/strategies-batch96/20261011-formal-source-acceptance-v2/baseline.json')
BASELINE_SHA = '6793e056b0cf0ba9423cad26ef95855269e8a5ca9b8a654ac15ee945a396d88d'
# This archived metadata is checksum evidence only, never a Store input.
ASSETS = Path('data/metadata/states/11e52610c1c5fb1cf3dbae62/assets/index.json')
ASSETS_SHA = '9814f72be87a1613ebed657770c22b0148c7c6aefbec95c4d138cfaf48f456cf'
RECEIPT = Path('docs/handoff/2026-10-11-pair66-short-windows-verification.json')
HELPERS = (
    'scripts/backtest_pair66_short_windows.py', 'tests/unit/test_pair66_short_windows.py',
    'scripts/archive_strategy_progress.py', 'scripts/preflight_pair66_short_windows.py',
    'scripts/recover_strategy_batch94_reference.py', 'scripts/reference_strategy_batch94.py',
    'scripts/review_strategy_batch18.py', 'scripts/run_strategy_batch4.py',
    'scripts/run_strategy_batch5.py', 'scripts/run_strategy_batch8.py',
    'scripts/supplement_pair66_history.py', 'scripts/verify_strategy_batch3.py',
    'configs/rule_profiles/main_board.yaml', 'pyproject.toml', 'uv.lock',
)


def bound(path, sha):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and file_sha(path) == sha, 'Bound file changed: '+str(path))
    return path


def original():
    bound(CONFIG, CONFIG_SHA)
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    bound(cfg['source_path'], SOURCE_SHA)
    return cfg


def checked_config(value, row):
    expected = {**original(), 'snapshot': SNAPSHOT, 'start': row['start'], 'end': row['end'], 'cache': False}
    cfg = StrategyConfig.model_validate(value).model_dump(mode='json')
    require(cfg == StrategyConfig.model_validate(expected).model_dump(mode='json'), 'Frozen economic or window parameters changed')
    return cfg


def implementation():
    names = sorted(set(HELPERS) | {p.as_posix() for p in Path('src/observe').rglob('*.py')})
    return {n: file_sha(n) for n in names}


def antecedents():
    bound(PARENT, PARENT_SHA); bound(BASELINE, BASELINE_SHA); bound(ASSETS, ASSETS_SHA)
    bound(PREPARED, PREPARED_SHA); bound(PREFLIGHT/'preflight.json', PREFLIGHT_SHA)
    doc = read(PREFLIGHT/'preflight.json')
    bound(PREFLIGHT/'factors.parquet', doc['factors_sha256'])
    bound(Path('data/snapshots')/(SNAPSHOT+'.json'), doc['snapshot_sha256'])
    for n, sha in doc['code_sha256'].items(): bound(n, sha)
    require(doc['status'] == 'ok' and doc['source_sha256'] == SOURCE_SHA and doc['config_sha256'] == CONFIG_SHA
        and doc['decision_days'] == 320 and doc['new_trade_backtests'] == 0, 'Preflight scope differs')
    require(tuple((r['start'], r['end'], r['decision_days']) for r in doc['windows']) == EXPECTED, 'Preflight inventory differs')
    require([r['executable_signal_actions'] for r in doc['windows']] == [1, 1, 0, 1, 3, 1], 'Preflight actions differ')
    policy = read('configs/research_metadata.json')
    require(next(d for d in policy['decisions'] if d['id'] == 'D-rotation66')['status'] == 'approved', 'Approval missing')
    return doc


def prepare():
    require(not DIRECTORY.exists(), 'Immutable directory exists; do not overwrite or rerun prepare')
    preflight = antecedents(); config = original(); store = Store('data'); state = store.state(SNAPSHOT)
    require(store.published() == read(BASELINE)['published'] and state['tables'] == store.published()['tables'], 'Publication drift')
    checksums = {r['path']: r['file_sha256'] for r in read(ASSETS)['assets']}
    # Check all signal and execution partitions before any order, including warmup.
    tables = ('bars_1d', 'calendar', 'instruments', 'corp_actions', 'adj_factors', 'adj_coverage', 'index_1d')
    payload = {}
    for table in tables:
        for entry in state['tables'][table].values():
            name = 'data/'+entry['file']
            require(name in checksums, 'Payload checksum evidence missing: '+name)
            bound(name, checksums[name]); payload[name] = checksums[name]
    rows = []; omitted = []
    with network_blocked():
        for row in preflight['windows']:
            if not row['executable_signal_actions']:
                omitted.append({'start': row['start'], 'end': row['end'], 'sessions': row['decision_days'],
                    'reason': 'zero_signal_window', 'new_trading_backtests': 0})
                continue
            cfg = checked_config({**config, 'snapshot': SNAPSHOT, 'start': row['start'], 'end': row['end'], 'cache': False}, row)
            inputs = build(load(store, state, row['start'], row['end'], 61, cfg['execution_instruments']),
                row['start'], row['end'], cfg['universe']['boards'], cfg['execution']['liquidity_window'], None)
            require(len(inputs.dates) == row['decision_days'] and not inputs.unexplained, 'Incomplete or unexplained execution inputs')
            for quotes in inputs.market.values():
                for instrument in cfg['execution_instruments']:
                    q = quotes.get(instrument, {})
                    require(not q.get('suspended', True) and q.get('open') and q.get('close')
                        and q.get('avg_amount_20d') and q['avg_amount_20d'] > 0, 'Open/liquidity input missing')
            rows.append({**row, 'label': 'pair66-short-'+str(len(rows)+1), 'execution_input': inputs.info,
                'execution_limitations': inputs.limitations})
    require(len(rows) == 5 and sum(r['decision_days'] for r in rows) == 291, 'Five-island coverage differs')
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASELINE, DIRECTORY/'baseline.json')
    hashes = implementation()
    for name in hashes:
        destination = DIRECTORY/'implementation'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    for row in rows:
        path = DIRECTORY/(row['label']+'.yaml')
        cfg = checked_config({**config, 'snapshot': SNAPSHOT, 'start': row['start'], 'end': row['end'], 'cache': False}, row)
        with path.open('x', encoding='utf-8') as stream: yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
        row.update(config_file=str(path), config_sha256=file_sha(path))
    save(DIRECTORY/'prepared.json', {'selected': rows, 'omitted': omitted, 'snapshot': SNAPSHOT,
        'implementation_sha256': hashes, 'payload_sha256': payload, 'parent_sha256': PARENT_SHA,
        'preflight_sha256': PREFLIGHT_SHA, 'parameters_changed': False, 'initial_cash': 1000000,
        'selection': 'All remaining short islands from previously frozen input eligibility inventory; no outcome selection',
        'no_stitched_nav': True, 'data_downloaded': False})
    return {'status': 'prepared', 'directory': str(DIRECTORY), 'sessions': 291, 'partitions_hashed': len(payload)}


def validate():
    antecedents(); doc = read(DIRECTORY/'prepared.json'); bound(DIRECTORY/'baseline.json', BASELINE_SHA)
    require(doc['implementation_sha256'] == implementation(), 'Runtime implementation drift')
    for n, sha in doc['implementation_sha256'].items(): bound(DIRECTORY/'implementation'/n, sha)
    require([(r['start'], r['end'], r['decision_days']) for r in doc['selected']] == [r for r in EXPECTED if r[2] != 29], 'Frozen selection drift')
    for row in doc['selected']:
        path = DIRECTORY/(row['label']+'.yaml')
        require(str(path) == row['config_file'] and row['label'] in {f'pair66-short-{n}' for n in range(1, 6)}, 'Config path drift')
        bound(path, row['config_sha256']); checked_config(yaml.safe_load(path.read_text(encoding='utf-8')), row)
    require(Store('data').published() == read(BASELINE)['published'], 'Publication changed')
    return doc


def selected(label):
    rows = [r for r in validate()['selected'] if r['label'] == label]
    require(len(rows) == 1, 'Unknown frozen label')
    return rows[0]


def compare_preflight(output, row):
    expected = pd.read_parquet(PREFLIGHT/'factors.parquet')
    expected = expected[expected.window_start_date.eq(row['start'])].drop(columns='window_start_date').reset_index(drop=True)
    actual = pd.read_parquet(output/'factors.parquet').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    return len(actual)


def run(label):
    row = selected(label)
    with network_blocked(): return run_one('data', DIRECTORY, row['config_file'], batch_label='Pair66 approved short islands')


def verify(label):
    row = selected(label); output = Path(read(DIRECTORY/(label+'-run.json'))['output'])
    compare_preflight(output, row)
    with network_blocked(): return verify_one('data', DIRECTORY, label, checker=signal_check, batch_label='Pair66 approved short islands')


def finish():
    require(not RECEIPT.exists(), 'Final receipt exists')
    doc = validate(); rows = []
    for row in doc['selected']:
        name = DIRECTORY/(row['label']+'-verification.json'); verified = read(name)
        require(verified['status'] == 'ok' and verified['original_unchanged'] and
            verified['reproduction']['reproduction']['result'] == 'match' and
            verified['reproduction']['reproduction']['differences'] == 0, 'Window reproduction incomplete')
        require(verified['hand_check']['sessions_checked'] == row['decision_days'], 'Independent coverage incomplete')
        report = verified['report']; require(len(report['results']) == 3, 'Missing costs')
        for result in report['results']:
            require(result['status'] == 'success_limited' and result['trading']['filled_orders'] > 0, 'No real trades in a selected window')
            require(sum(r['sessions'] for r in result['yearly']) == row['decision_days'], 'Annual coverage incomplete')
        output = Path(verified['run']['output']); compare_preflight(output, row)
        rows.append({**row, 'run_id': verified['run']['run_id'], 'reproduction_run_id': verified['reproduction']['run_id'],
            'validation_file': str(name), 'validation_sha256': file_sha(name), 'hand_check': verified['hand_check'],
            'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark']})
    for n, sha in doc['payload_sha256'].items(): bound(n, sha)
    fees = fee_checks(DIRECTORY); protection = protect('data', DIRECTORY)
    progress = archive('data', 'Approved pair66 five short eligibility islands291days: sole Book three costs/independent signals and fees/offline match0; one29day zero-signal island omitted')
    require(progress['sources'] == 695 and progress['manually_reviewed'] == 278 and progress['local_strategy_runs'] == 87, 'Coverage counts differ')
    save(RECEIPT, {'status': 'ok', 'nature': 'supplement_existing_approved_approximate_backtests',
        'formal_trading_acceptance': True, 'original_strategy_complete': False, 'library_goal_complete': False,
        'source_sha256': SOURCE_SHA, 'snapshot': SNAPSHOT, 'progress': progress, 'windows': rows, 'omitted': doc['omitted'],
        'prepared_file': str(DIRECTORY/'prepared.json'), 'prepared_sha256': file_sha(DIRECTORY/'prepared.json'),
        'implementation_sha256': doc['implementation_sha256'], 'fees': fees, 'protection': protection,
        'payload_partitions_byte_hashed': len(doc['payload_sha256']), 'fresh_process_reproduction': True,
        'network_forbidden': True, 'new_original_parent_runs': 5, 'new_reproduction_parent_runs': 5,
        'new_strategy_sources': 0, 'new_source_backtest_coverage': 0, 'decision_days': 291, 'three_cost_ledger_days': 873,
        'limits': ['Approved dynamic-anchor daily approximation; original intraday/platform equivalence not proven',
            'Five independent empty-state1million-cash islands; never stitch NAV; short annualized estimates unstable',
            'Fixed pair and already viewed history are not independent holdout; no parameter selection',
            'Fees/trading rules retained unverified historical approximations; independent checks validate arithmetic only',
            'Old full controls rehashed; all used table partitions byte-hashed; other partitions size/mtime only']})
    return {'status': 'ok', 'receipt': str(RECEIPT), 'progress': progress, 'fills_checked': fees['fills_checked']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'finish'))
    parser.add_argument('--label', choices=tuple(f'pair66-short-{n}' for n in range(1, 6)))
    args = parser.parse_args()
    if (args.action in ('run', 'verify')) != (args.label is not None): parser.error('Exactly run/verify require --label')
    print(json.dumps(globals()[args.action](args.label) if args.label else globals()[args.action](), ensure_ascii=False, default=str))
