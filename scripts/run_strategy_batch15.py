"""Publish checked RSI history and archive extended, sole-ledger experiments."""
import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import requests
import yaml

from observe.data.audit import audit_daily
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.execution import sessions
from observe.integrity import verify_run
from observe.ledger import RuleSet
from observe.replay import reproduce
from observe.runs import environment, file_sha, write_json
from observe.strategies import StrategyConfig
from observe.strategy_rsi import UNIQUE_POOL
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch15 import CONFIG, END, MISSING, SNAPSHOT, START, analyzed_inputs, custody
from scripts.run_strategy_batch4 import fee_checks, mutable_registry, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.run_strategy_batch7 import signal_check
from scripts.run_strategy_batch14 import retain_old_rows
from scripts.verify_strategy_batch3 import require, tree_hash

LABEL = 'rsi_slots_corrected_batch7-extended'
YEARS = {'2019', '2020'}
FILES = {'scripts/probe_strategy_batch15.py', 'scripts/run_strategy_batch15.py', 'src/observe/execution.py',
    'src/observe/strategies.py', 'tests/unit/test_history_scope.py', 'tests/integration/test_rsi_strategy.py',
    'tests/unit/test_batch15_gates.py'}
CORE = {'baseline.json', 'precheck.json', 'existing-api.json', 'probe-results.json', 'retry-results.json', 'input-analysis.json',
    'publication.json', 'historical-daily-audit.csv', 'prepared-config.json', 'old-rsi-reproduction.json', 'fee-hand-checks.json',
    f'{LABEL}-run.json', f'{LABEL}-freeze.json', f'{LABEL}-verification.json'}


def merge_year(old, incoming):
    require(not incoming.duplicated(['date', 'instrument']).any(), 'Incoming prefix duplicates')
    require(not old[['date', 'instrument']].merge(incoming[['date', 'instrument']], on=['date', 'instrument']).shape[0], 'Incoming prefix overlaps frozen rows')
    result = pd.concat([old, incoming], ignore_index=True).sort_values(['date', 'instrument']).reset_index(drop=True)
    retain_old_rows(old, result, ['date', 'instrument'])
    return result


def publication_references(old, new):
    require(set(old['tables']) == set(new['tables']), 'Table set changed')
    changed = set()
    for table, entries in old['tables'].items():
        now = new['tables'][table]
        if table != 'bars_1d': require(now == entries, f'Unexpected {table} change')
        else:
            require(set(now) == set(entries), 'Year partition keys changed')
            changed = {year for year in entries if entries[year] != now[year]}
            require(changed == YEARS, 'Only 2019/2020 references may expand')
    return changed


def publish(root, directory):
    output = directory / 'publication.json'; require(not output.exists(), 'Publication already archived')
    analysis = read(directory / 'input-analysis.json'); require(analysis['status'] == 'ready', 'RSI inputs are not ready')
    prefix, profiles, evidence = analyzed_inputs(root, directory)
    require(analysis['snapshot'] == SNAPSHOT and analysis['stocks'] == profiles and analysis['probe_evidence'] == evidence, 'Analysis changed or differs from inputs')
    store = Store(root); baseline = read(directory / 'baseline.json')
    with operation_lock(root, DATA_WRITER):
        state = store.published(); require(state == baseline['published'] and state['tables'] == store.state(SNAPSHOT)['tables'], 'Publication moved since baseline')
        require(set(prefix.instrument) == set(MISSING) and set(prefix.date.map(lambda d: str(d.year))) == YEARS, 'Prefix stock/year set incomplete')
        days = [d for d in sessions(store.load_state(state, 'calendar')) if date(2019, 1, 1) <= d < date(2021, 1, 1)]
        issues = audit_daily(prefix, days, store.load_state(state, 'instruments'), RuleSet.from_yaml('configs/rule_profiles/main_board.yaml'))
        require(not issues.level.eq('block').any(), 'RSI historical daily audit blockers')
        audit_path = directory / 'historical-daily-audit.csv'; require(not audit_path.exists(), 'Audit already exists'); issues.to_csv(audit_path, index=False)
        parts, scope_records = {}, []
        for year in sorted(YEARS):
            old = store.load_state(state, 'bars_1d', parts=[year]); incoming = prefix[prefix.date.map(lambda d: str(d.year) == year)]
            merged = merge_year(old, incoming); existing = state['tables']['bars_1d'][year]['history_scope']
            require(existing['policy'] == 'selected_instruments_v1', 'Old history scope unknown')
            instruments = sorted(set(existing['instruments']) | set(MISSING))
            require(set(merged.instrument) == set(instruments), 'Expanded history scope differs from actual rows')
            entry = store.write_partition('bars_1d', year, merged)
            entry['history_scope'] = {'policy': 'selected_instruments_v1', 'instruments': instruments,
                'start': str(merged.date.min()), 'end': str(merged.date.max()), 'full_market': False,
                'probe_evidence': evidence, 'retained_entry': state['tables']['bars_1d'][year]}
            parts[year] = entry; scope_records.append({'year': year, 'old_rows': len(old), 'added_rows': len(incoming),
                'rows': len(merged), 'scope_instruments': len(instruments), 'old_rows_exactly_retained': True})
        prospective = {'tables': {**state['tables'], 'bars_1d': {**state['tables']['bars_1d'], **parts}}}
        publication_references(state, prospective)
        bid = store.write_batch({'bars_1d': parts}, note='batch15: append RSI39 stock histories to immutable 2019/2020 partitions; retain prior ten stocks exactly')
        audit = store.commit_audit(bid, issues, RuleSet.from_yaml('configs/rule_profiles/main_board.yaml').config_fingerprint(),
            [START, '2020-12-31'], scope='selected_history', input_state=prospective)
        store.publish(bid); sid = store.snapshot('batch15 RSI fixed pool historical extension; only 49-stock 2019/2020 coverage')
        result = {'status': 'published', 'batch_id': bid, 'snapshot': sid, 'old_snapshot': SNAPSHOT,
            'prefix_rows_added': len(prefix), 'partitions': scope_records, 'probe_evidence': evidence,
            'analysis_sha256': file_sha(directory / 'input-analysis.json'), 'audit_id': audit,
            'audit_sha256': file_sha(audit_path), 'new_warning_counts': issues.groupby('rule').size().to_dict(),
            'strict_financial_usable_rows': 0, 'old_adjustments_actions_calendar_index_unchanged': True, 'full_market': False}
        write_json(output, result)
    archive(root, f'Batch15 RSI39 history published {bid}; {len(prefix)} new rows, retained prior ten-stock rows')
    return result


def prepare(root, directory):
    custody(directory); output = directory / 'prepared-config.json'; require(not output.exists(), 'Prepared config already exists')
    publication = read(directory / 'publication.json'); store = Store(root); state = store.state(publication['snapshot'])
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); bars = store.load_state(state, 'bars_1d', filters=[('instrument', 'in', list(UNIQUE_POOL))])
    bars = bars[bars.date.ge(date(2019, 1, 1))]; ready = {}
    for instrument in UNIQUE_POOL:
        traded = bars[bars.instrument.eq(instrument) & bars.is_trading].sort_values('date')
        require(len(traded) >= 61, f'{instrument}: no full RSI history')
        ready[instrument] = str(traded.iloc[60].date)
    start = max(date.fromisoformat(d) for d in ready.values()); days = sessions(store.load_state(state, 'calendar'))
    actions = store.load_state(state, 'corp_actions', filters=[('instrument', 'in', list(UNIQUE_POOL))])
    rights = actions[actions.rights_ratio.gt(0) & actions.ex_date.ge(start) & actions.ex_date.le(date.fromisoformat(END))]
    if len(rights): start = next(d for d in days if d > rights.ex_date.max())
    require(start <= date.fromisoformat(END), 'No supported interval')
    cfg.update(snapshot=publication['snapshot'], start=str(start), cache=False)
    cfg['name'] += '（历史延伸）'; spec = StrategyConfig.model_validate(cfg)
    path = directory / f'{LABEL}.yaml'; require(not path.exists(), 'Config already exists')
    path.write_text(yaml.safe_dump(spec.model_dump(mode='json'), allow_unicode=True, sort_keys=False), encoding='utf-8')
    original = custody(directory)
    result = {'label': LABEL, 'config_file': str(path), 'config_sha256': file_sha(path),
        'original_config_sha256': original['config_sha256'], 'source_sha256': original['source_sha256'],
        'snapshot': publication['snapshot'], 'start': str(start), 'end': END, 'sessions': sum(start <= d <= date.fromisoformat(END) for d in days),
        'first_61_traded_rows': ready, 'rights_boundary_after_ready': rights.astype(str).to_dict('records'),
        'parameters_changed': False, 'selection': 'Earliest jointly complete traded windows and supported rights boundary; no performance selection'}
    write_json(output, result); archive(root, f'Batch15 RSI extended config frozen from {start}; original pool, parameters and cash retained')
    return result


def config_record(directory):
    original = custody(directory); record = read(directory / 'prepared-config.json')
    require(file_sha(record['config_file']) == record['config_sha256'], 'Prepared RSI config changed')
    require(record['source_sha256'] == original['source_sha256'] and record['original_config_sha256'] == original['config_sha256'], 'Prepared source/config binding differs')
    actual = StrategyConfig.model_validate(yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))).model_dump(mode='json')
    approved = StrategyConfig.model_validate(yaml.safe_load(CONFIG.read_text(encoding='utf-8'))).model_dump(mode='json')
    for key in ('snapshot', 'start', 'cache', 'name'): approved[key] = actual[key]
    require(actual == approved, 'Extended RSI economic config differs from approved source')
    require((actual['snapshot'], actual['start'], actual['end']) == (record['snapshot'], record['start'], record['end']), 'Prepared interval differs')
    return record


def check_prepared_interval(record, publication, days):
    require(record['snapshot'] == publication['snapshot'], 'Prepared publication snapshot differs')
    start, end = date.fromisoformat(record['start']), date.fromisoformat(record['end'])
    require(start <= end and record['sessions'] == sum(start <= d <= end for d in days), 'Prepared session count differs')


def bound_config(root, directory):
    record = config_record(directory)
    publication = read(directory / 'publication.json')
    store = Store(root)
    check_prepared_interval(record, publication, sessions(store.load_state(store.state(publication['snapshot']), 'calendar')))
    return record


def reproduce_old(root, directory):
    output = directory / 'old-rsi-reproduction.json'; require(not output.exists(), 'Old reproduction already archived')
    old = read('docs/handoff/2026-10-05-batch7-verification.json'); integrity = verify_run(root, old['run_id'])
    require(integrity['status'] == 'ok', 'Old RSI integrity failed'); path = Path(integrity['output']); before = tree_hash(path)
    def forbidden(*a, **kw): raise AssertionError('Offline reproduction attempted network')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
        result = reproduce(root, old['run_id'])
    require(result['reproduction']['result'] == 'match' and result['reproduction']['differences'] == 0, 'Old RSI output changed')
    require(tree_hash(path) == before and verify_run(root, result['run_id'])['status'] == 'ok', 'Old RSI or reproduction integrity failed')
    write_json(output, {'status': 'ok', 'original_run_id': old['run_id'], 'original_manifest_sha256': file_sha(path / 'manifest.json'),
        'reproduction': result, 'old_original_unchanged': True})
    archive(root, 'Batch15 loading change: old RSI1329-day three-cost frozen experiment reproduced offline match/0')
    return result


def protect(root, directory):
    output = directory / 'protection.json'; require(not output.exists(), 'Protection already archived')
    baseline = read(directory / 'baseline.json'); store = Store(root); current = store.published()
    controls = {p: sha for p, sha in baseline['controls'].items() if not mutable_registry(p)}
    require(all(file_sha(store.root / p) == sha for p, sha in controls.items()), 'Old controls changed')
    require(all((store.root / p).stat().st_size == row['size'] and (store.root / p).stat().st_mtime_ns == row['mtime_ns'] for p, row in baseline['partitions'].items()), 'Old partition metadata changed')
    publication_references(baseline['published'], current)
    prefix, _, _ = analyzed_inputs(root, directory)
    for year in YEARS:
        old = store.load_state(baseline['published'], 'bars_1d', parts=[year]); new = store.load_state(current, 'bars_1d', parts=[year])
        retain_old_rows(old, new, ['date', 'instrument'])
        incoming = prefix[prefix.date.map(lambda d: str(d.year) == year)]
        retain_old_rows(incoming, new, ['date', 'instrument'])
        require(len(new) == len(old) + len(incoming), 'New yearly rows differ from checked old plus incoming')
        scope = current['tables']['bars_1d'][year]['history_scope']
        require(scope['instruments'] == sorted(set(new.instrument)) and scope['policy'] == 'selected_instruments_v1' and
            not scope['full_market'] and scope['start'] == str(new.date.min()) and scope['end'] == str(new.date.max()), 'Expanded scope metadata differs')
    result = {'status': 'ok', 'old_controls_unchanged': len(controls), 'old_partition_metadata_unchanged': len(baseline['partitions']),
        'old_2019_2020_rows_exactly_retained': True, 'all_other_table_and_year_references_unchanged': True,
        'partition_check': 'size/mtime only, not a fresh full old partition hash', 'baseline_sha256': file_sha(directory / 'baseline.json')}
    write_json(output, result); return result


def checks(root, directory):
    output = directory / 'checks.json'; require(not output.exists(), 'Checks already archived')
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch15_gates.py', 'tests/unit/test_history_scope.py',
        'tests/integration/test_rsi_strategy.py', 'tests/integration/test_strategy_scope.py', 'tests/integration/test_stable_strategy_means.py'],
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
         'scripts/verify_financial_import.py', 'scripts/probe_strategy_batch15.py', 'scripts/run_strategy_batch15.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True); require(result.returncode == 0, 'Verification failed')
    evidence = {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()}
    require(CORE <= set(evidence), 'Required evidence missing')
    write_json(output, {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)}, 'evidence_sha256': evidence})
    return {'status': 'ok', 'output': str(output)}


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch15-verification.json'); require(not output.exists(), 'Final receipt already exists')
    checked = read(directory / 'checks.json')
    require(checked['commands'] and all(row['returncode'] == 0 for row in checked['commands']), 'Commands did not pass')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked file sets incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Checked code changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    _, profiles, evidence = analyzed_inputs(root, directory); analysis = read(directory / 'input-analysis.json')
    require(analysis['status'] == 'ready' and analysis['stocks'] == profiles and analysis['probe_evidence'] == evidence, 'Final raw input evidence differs')
    record = config_record(directory); publication = read(directory / 'publication.json'); store = Store(root)
    require(store.published()['batch_id'] == publication['batch_id'] and store.published()['tables'] == store.state(publication['snapshot'])['tables'], 'Published state changed')
    check_prepared_interval(record, publication, sessions(store.load_state(store.state(publication['snapshot']), 'calendar')))
    verified = read(directory / f'{LABEL}-verification.json'); run = read(directory / f'{LABEL}-run.json')
    require(verified['status'] == 'ok' and verified['run'] == run and verified['source_sha256'] == record['source_sha256'] and verified['original_unchanged'], 'Extended verification incomplete')
    require(verified['reproduction']['reproduction']['result'] == 'match' and verified['reproduction']['reproduction']['differences'] == 0, 'Extended reproduction differs')
    expected = StrategyConfig.model_validate(yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))).model_dump(mode='json')
    for result in (run, verified['reproduction']):
        integrity = verify_run(root, result['run_id']); require(integrity['status'] == 'ok', 'Extended run integrity failed')
        require(read(Path(integrity['output']) / 'config.json')['config'] == expected, 'Actual extended config differs')
    require(read(Path(verify_run(root, run['run_id'])['output']) / 'report.json') == verified['report'], 'Verified RSI report differs from actual')
    require(verified['hand_check']['sessions_checked'] == record['sessions'] and verified['hand_check']['rsi_windows_checked'] == record['sessions'] * len(UNIQUE_POOL), 'RSI window coverage incomplete')
    original = read(directory / 'old-rsi-reproduction.json')
    require(original['status'] == 'ok' and original['old_original_unchanged'] and original['reproduction']['reproduction']['result'] == 'match' and original['reproduction']['reproduction']['differences'] == 0, 'Old reproduction incomplete')
    require(verify_run(root, original['original_run_id'])['status'] == verify_run(root, original['reproduction']['run_id'])['status'] == 'ok', 'Old reproduction integrity failed')
    require(original['original_run_id'] == read('docs/handoff/2026-10-05-batch7-verification.json')['run_id'] and
        original['original_manifest_sha256'] == file_sha(Path(verify_run(root, original['original_run_id'])['output']) / 'manifest.json'), 'Old RSI identity differs')
    fees = read(directory / 'fee-hand-checks.json')
    require(fees['status'] == 'ok' and len(fees['checks']) == 3 and {(r['run_id'], r['scenario']) for r in fees['checks']} == {(run['run_id'], s) for s in ('base', 'fees_x2', 'slippage_x2')}, 'Fee cost coverage incomplete')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for name, sha in checked['implementation_sha256'].items():
        target = frozen / Path(name).name; require(not target.exists(), 'Frozen name collision'); shutil.copyfile(name, target)
        require(file_sha(target) == sha, 'Frozen implementation differs')
    result = {'status': 'ok', 'environment': environment(), 'publication': publication, 'strategy': record, 'verification': verified,
        'old_rsi_reproduction': original, 'fees': fees, 'protection': protection, 'checks': checked,
        'original_strategy_complete': False, 'progress': archive(root, 'Batch15 RSI history extension archived: independent windows/slots, three costs, offline match and old-asset protection'),
        'limitations': ['Fixed post-selected 2024 source pool used before source date; not a historical investable selection rule',
            'Only49-stock2019/2020 history; earlier years retain ten-stock scope; no global historical market claim',
            'Known pauses skipped, missing observations still block; action/fee/platform equivalence remains approximate',
            'Existing warnings retained, newly appended selected-history audit is not a full snapshot audit; strict financial usable zero']}
    write_json(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'prepare', 'run', 'verify', 'reproduce-old', 'fees', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch15/20261005-rsi-history'))
    args = parser.parse_args()
    if args.action == 'run': result = run_one(args.root, args.directory, bound_config(args.root, args.directory)['config_file'], batch_label='Batch15')
    elif args.action == 'verify':
        bound_config(args.root, args.directory); result = verify_one(args.root, args.directory, LABEL, checker=signal_check, batch_label='Batch15')
    elif args.action == 'fees': result = fee_checks(args.directory)
    else: result = {'publish': publish, 'prepare': prepare, 'reproduce-old': reproduce_old, 'checks': checks, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
