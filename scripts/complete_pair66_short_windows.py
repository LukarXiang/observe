"""New immutable guards for continuing the frozen pair66 short-island driver."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from observe.integrity import verify_run
from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts import backtest_pair66_short_windows as frozen
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.run_strategy_batch8 import signal_check
from scripts.verify_strategy_batch3 import require

DIRECTORY = frozen.DIRECTORY
PREPARED_SHA = '56bb297e13f5235b70fae72655b561285866fc4a862de494369578a844030d1c'
FIRST_SHA = '3e983632d8b5d10ba84a6d5c987bc05ad178af55204c4f004cf7bd0ba3493ef4'
RECOVERY = DIRECTORY/'completion-guards'
FILES = ('scripts/complete_pair66_short_windows.py', 'tests/unit/test_pair66_short_completion.py')
REVIEWS = Path('data/staging/strategies-history-supplement/20261011-pair66-short-review-v2/review-completion.json')
FINAL = Path('docs/handoff/2026-10-11-pair66-short-windows-final-review.json')
SCENARIOS = ('base', 'fees_x2', 'slippage_x2')


def validate_payload(doc):
    expected = {'data/'+v['file'] for table in ('bars_1d', 'calendar', 'instruments', 'corp_actions',
        'adj_factors', 'adj_coverage', 'index_1d') for v in frozen.Store('data').state(frozen.SNAPSHOT)['tables'][table].values()}
    require(set(doc['payload_sha256']) == expected and len(expected) == 28, 'Exact payload inventory differs')
    for name, sha in doc['payload_sha256'].items(): frozen.bound(name, sha)


def inputs():
    frozen.bound(DIRECTORY/'prepared.json', PREPARED_SHA)
    frozen.bound(DIRECTORY/'pair66-short-1-run.json', FIRST_SHA)
    doc = frozen.validate(); validate_payload(doc)
    return doc


def prepare():
    inputs(); require(not RECOVERY.exists(), 'Guard directory exists')
    RECOVERY.mkdir()
    hashes = {n: file_sha(n) for n in FILES}
    for name, sha in hashes.items():
        out = RECOVERY/'implementation'/name; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, out); frozen.bound(out, sha)
    save(RECOVERY/'implementation.json', hashes)
    return {'status': 'prepared', 'prepared_sha256': PREPARED_SHA, 'files': hashes}


def validate():
    doc = inputs(); hashes = read(RECOVERY/'implementation.json')
    require(set(hashes) == set(FILES), 'Completion guard inventory differs')
    for name, sha in hashes.items():
        frozen.bound(name, sha); frozen.bound(RECOVERY/'implementation'/name, sha)
    return doc


def selected(label):
    rows = [r for r in validate()['selected'] if r['label'] == label]
    require(len(rows) == 1, 'Unknown frozen label')
    return rows[0]


def actual_run(result, row):
    run_id = result['run_id']; output = Path('data/runs')/run_id
    require(output.is_dir() and output.resolve() == Path(result['output']).resolve(), 'Run output identity differs')
    require(result['status'] == 'success_limited', 'Run not successful')
    require(verify_run('data', output)['status'] == 'ok', 'Actual run integrity failed')
    doc = read(output/'config.json')
    require(doc['kind'] == 'strategy' and doc['snapshot_id'] == frozen.SNAPSHOT
        and doc['source']['bytes_sha256'] == frozen.SOURCE_SHA, 'Run source/snapshot identity differs')
    frozen.checked_config(doc['config'], row)
    require(read(output/'subruns.json') == result['subruns'], 'Run child identity differs')
    report = read(output/'report.json')
    require(tuple(r['scenario'] for r in report['results']) == SCENARIOS, 'Three-cost report inventory differs')
    children = read(output/'subruns.json')['backtests']
    require(tuple(c['scenario'] for c in children) == SCENARIOS, 'Three-cost ledger inventory differs')
    count = 0
    for child, summary in zip(children, report['results'], strict=True):
        folder = output/'variants'/child['run_id']
        require(folder.resolve() == Path(child['output']).resolve(), 'Child output identity differs')
        require(summary['status'] == 'success_limited', 'Incomplete child report')
        fills = read(folder/'fills.json'); equity = read(folder/'equity.json')
        require(len(fills) > 0 and len(fills) == summary['trading']['filled_orders'], 'Nonempty actual fills differ')
        require(len(equity) == row['decision_days'] and sum(y['sessions'] for y in summary['yearly']) == row['decision_days'], 'Ledger/yearly coverage differs')
        count += len(fills)
    frozen.compare_preflight(output, row)
    return output, report, count


def registered_verification(path, result, verified):
    require(verified['status'] == 'ok' and verified['original_unchanged'] and verified['run'] == result, 'Verification run identity differs')
    records = read('data/catalog/strategies/implementation-evidence.json')
    matched = [r for r in records if r['run_id'] == result['run_id'] and r.get('validation_file') == str(path)]
    require(len(matched) == 1 and matched[0]['validation_sha256'] == file_sha(path), 'Verification registration or SHA differs')
    entry = matched[0]
    require(entry['source_sha256'] == frozen.SOURCE_SHA and entry['snapshot'] == frozen.SNAPSHOT
        and entry['reproduction_result'] == 'match' and entry['differences'] == 0
        and entry['reproduction_run_id'] == verified['reproduction']['run_id'], 'Verification provenance differs')
    latest = read('data/catalog/strategies/latest.json')
    catalog = Path('data/catalog/strategies')/latest['catalog_id']/'catalog.json'
    frozen.bound(catalog, latest['sha256'])
    sources = [r for r in read(catalog) if r['bytes_sha256'] == frozen.SOURCE_SHA]
    require(len(sources) == 1 and entry in sources[0]['implementation_evidence'], 'Active catalog registration incomplete')


def checked_verification(row):
    result = read(DIRECTORY/(row['label']+'-run.json'))
    output, report, fills = actual_run(result, row)
    path = DIRECTORY/(row['label']+'-verification.json'); verified = read(path)
    registered_verification(path, result, verified)
    require(verified['report'] == report and verified['source_sha256'] == frozen.SOURCE_SHA, 'Verification summary differs from actual run')
    reproduction = verified['reproduction']
    require(reproduction['reproduction']['result'] == 'match' and reproduction['reproduction']['differences'] == 0, 'Reproduction not match0')
    again, again_report, again_fills = actual_run(reproduction, row)
    require(again != output and Path(read(again/'config.json')['reproduce_of']).resolve() == output.resolve(), 'Reproduction ancestry differs')
    require(again_report['results'] == report['results'] and again_fills == fills, 'Reproduction report differs')
    require(verified['hand_check']['sessions_checked'] == row['decision_days'], 'Independent signal coverage differs')
    require(tuple(c['scenario'] for c in verified['hand_check']['orders']) == SCENARIOS and
        all(c['sessions_checked'] == row['decision_days'] and c['orders_checked'] > 0 for c in verified['hand_check']['orders']), 'Independent order coverage incomplete')
    return {**row, 'run_id': result['run_id'], 'reproduction_run_id': reproduction['run_id'],
        'validation_file': str(path), 'validation_sha256': file_sha(path), 'hand_check': verified['hand_check'],
        'period': report['period'], 'results': report['results'], 'benchmark': report['benchmark'], 'fills_checked': fills}


def run(label):
    row = selected(label)
    path = DIRECTORY/(label+'-run.json')
    if path.exists():
        actual_run(read(path), row)
        return {'status': 'already_complete', 'run_id': read(path)['run_id']}
    require(not (DIRECTORY/(label+'-freeze.json')).exists(), 'Incomplete prior attempt retained; use a new reviewed attempt directory')
    with network_blocked(): result = run_one('data', DIRECTORY, row['config_file'], batch_label='Pair66 guarded approved short islands')
    actual_run(result, row)
    return result


def verify_direct(label):
    row = selected(label); actual_run(read(DIRECTORY/(label+'-run.json')), row)
    path = DIRECTORY/(label+'-verification.json')
    if not path.exists():
        with network_blocked(): verify_one('data', DIRECTORY, label, checker=signal_check, batch_label='Pair66 guarded approved short islands')
    # A partial registry update is rejected, never silently repaired or accepted.
    return checked_verification(row)


def fresh_command(label):
    return [sys.executable, '-B', '-m', 'scripts.complete_pair66_short_windows', 'worker', '--label', label]


def fresh_evidence(label):
    name = RECOVERY/(label+'-fresh-worker.json'); doc = read(name)
    require(doc['command'] == fresh_command(label) and doc['returncode'] == 0 and
        doc['worker_pid'] != doc['manager_pid'] and doc['worker_parent_pid'] == doc['manager_pid'] and
        doc['reproduction_created'] is True and doc['implementation_sha256'] == read(RECOVERY/'implementation.json'),
        'Fresh-process provenance differs')
    for key in ('stdout', 'stderr'): frozen.bound(doc[key+'_file'], doc[key+'_sha256'])
    captured = json.loads(Path(doc['stdout_file']).read_text(encoding='utf-8').splitlines()[-1])
    fields = ('run_id', 'reproduction_run_id', 'worker_pid', 'worker_parent_pid', 'reproduction_created', 'validation_sha256')
    require(set(captured) == set(fields) and captured == {k: doc[k] for k in fields}, 'Fresh-worker stdout identity differs')
    path = DIRECTORY/(label+'-verification.json')
    frozen.bound(path, doc['validation_sha256'])
    verified = read(path)
    require(doc['run_id'] == verified['run']['run_id'] and doc['reproduction_run_id'] == verified['reproduction']['run_id'],
        'Fresh-worker run identity differs')
    return {'file': str(name), 'sha256': file_sha(name), **doc}


def worker(label):
    require(not (DIRECTORY/(label+'-verification.json')).exists(), 'Worker requires a new reproduction, not a cached verification')
    row = verify_direct(label)
    return {'run_id': row['run_id'], 'reproduction_run_id': row['reproduction_run_id'],
        'worker_pid': os.getpid(), 'worker_parent_pid': os.getppid(), 'reproduction_created': True,
        'validation_sha256': row['validation_sha256']}


def verify(label):
    selected(label)
    if (RECOVERY/(label+'-fresh-worker.json')).exists():
        fresh_evidence(label)
        return checked_verification(selected(label))
    require(not (DIRECTORY/(label+'-verification.json')).exists(), 'Verification without fresh-worker provenance requires new reviewed recovery')
    stdout = RECOVERY/(label+'-worker.stdout'); stderr = RECOVERY/(label+'-worker.stderr')
    started = datetime.now().isoformat(); command = fresh_command(label)
    with stdout.open('xb') as out, stderr.open('xb') as err:
        result = subprocess.run(command, stdout=out, stderr=err, cwd=Path.cwd(), check=False)
    evidence = {'command': command, 'returncode': result.returncode, 'manager_pid': os.getpid(), 'started_at': started,
        'finished_at': datetime.now().isoformat(), 'implementation_sha256': read(RECOVERY/'implementation.json'),
        'stdout_file': str(stdout), 'stdout_sha256': file_sha(stdout), 'stderr_file': str(stderr), 'stderr_sha256': file_sha(stderr)}
    if result.returncode == 0:
        evidence.update(json.loads(stdout.read_text(encoding='utf-8').splitlines()[-1]))
    save(RECOVERY/(label+'-fresh-worker.json'), evidence)
    require(result.returncode == 0, 'Fresh verification worker failed; preserve logs and use new reviewed recovery')
    fresh_evidence(label)
    return checked_verification(selected(label))


def review_binding(hashes):
    doc = read(REVIEWS)
    require(doc['status'] == 'ok' and doc['implementation_sha256'] == hashes and len(doc['reviews']) == 6, 'Review coverage differs')
    require({r['role'] for r in doc['reviews']} == {'security', 'abuse', 'architecture', 'composition', 'assumptions', 'cascade'}, 'Six review roles missing')
    for row in doc['reviews']:
        frozen.bound(row['file'], row['sha256'])
        body = read(row['file'])
        require(row['status'] == body['status'] == 'ok' and body['role'] == row['role'] and
            body['scope'] == 'pair66_short_completion_v2' and body['implementation_sha256'] == hashes,
            'Review body identity/status/implementation differs')
        require(body['findings'] == [], 'Open review finding')


def finish_locked():
    require(not frozen.RECEIPT.exists() and not FINAL.exists(), 'Final acceptance exists')
    doc = validate(); hashes = read(RECOVERY/'implementation.json'); review_binding(hashes)
    fresh = [fresh_evidence(r['label']) for r in doc['selected']]
    rows = [checked_verification(r) for r in doc['selected']]
    # New attempt for each finalization; never rewrite failed fee/protection stages.
    attempt = Path(tempfile.mkdtemp(prefix='acceptance-attempt-', dir=RECOVERY))
    shutil.copyfile(frozen.BASELINE, attempt/'baseline.json')
    for row in doc['selected']:
        name = row['label']+'-run.json'; shutil.copyfile(DIRECTORY/name, attempt/name)
    fees = fee_checks(attempt)
    require(len(fees['checks']) == 15 and fees['fills_checked'] == sum(r['fills_checked'] for r in rows) > 0, 'Actual fee coverage differs')
    protection = protect('data', attempt)
    progress = archive('data', 'Pair66 five short islands291days/sole Book three costs/independent signal and fees/forbidden-network fresh-process match0; no new source')
    require(progress['sources'] == 695 and progress['manually_reviewed'] == 278 and
        progress['locally_backtested_sources'] == progress['locally_reproduced_sources'] == 12, 'Source coverage differs')
    navigation = read('data/catalog/strategies/progress/latest.json'); destination = Path(progress['output'])
    summary = read(destination/'summary.json')
    require(navigation == {'checkpoint': progress['checkpoint'], 'directory': str(destination),
        'summary_sha256': file_sha(destination/'summary.json')} and
        summary == {k: v for k, v in progress.items() if k != 'output'}, 'Progress navigation/readback differs')
    latest = read('data/catalog/strategies/latest.json')
    require(latest['catalog_id'] == progress['catalog_id'] and latest['sha256'] == progress['catalog_sha256'], 'Catalog/progress identity differs')
    evidence = {str(DIRECTORY/'prepared.json'): PREPARED_SHA, str(REVIEWS): file_sha(REVIEWS)}
    for row in fresh: evidence[row['file']] = row['sha256']
    for row in rows:
        for suffix in ('-run.json', '-verification.json'):
            name = DIRECTORY/(row['label']+suffix); evidence[str(name)] = file_sha(name)
    for name in (attempt/'fee-hand-checks.json', attempt/'protection.json', RECOVERY/'implementation.json'):
        evidence[str(name)] = file_sha(name)
    for name in (destination/'summary.json', destination/'sources.json', destination/'runs.json', destination/'progress.md'):
        evidence[str(name)] = file_sha(name)
    save(frozen.RECEIPT, {'status': 'ok', 'nature': 'supplement_existing_approved_approximate_backtests',
        'formal_trading_acceptance': True, 'original_strategy_complete': False, 'library_goal_complete': False,
        'source_sha256': frozen.SOURCE_SHA, 'snapshot': frozen.SNAPSHOT, 'progress': progress, 'windows': rows, 'omitted': doc['omitted'],
        'prepared_file': str(DIRECTORY/'prepared.json'), 'prepared_sha256': PREPARED_SHA,
        'implementation_sha256': {**doc['implementation_sha256'], **hashes}, 'fees': fees, 'protection': protection,
        'payload_partitions_byte_hashed': 28, 'network_forbidden': True, 'fresh_process_reproduction': True,
        'fresh_worker_evidence': fresh,
        'new_original_parent_runs': 5, 'new_reproduction_parent_runs': 5, 'new_strategy_sources': 0,
        'new_source_backtest_coverage': 0, 'decision_days': 291, 'three_cost_ledger_days': 873,
        'evidence_sha256': evidence, 'limits': [
            'Approved dynamic-anchor daily approximation; intraday/platform equivalence not proven',
            'Five independent1million-cash empty-state islands; NAV not stitched; short annualization unstable',
            'Fixed pair and already viewed history are not independent holdout; no parameter selection',
            'Historical fee/rule assumptions remain unverified; hand checks prove arithmetic only',
            'Incomplete freeze/verification attempts must use new reviewed recovery; never overwrite old evidence']})
    evidence[str(frozen.RECEIPT)] = file_sha(frozen.RECEIPT)
    for name, sha in evidence.items(): frozen.bound(name, sha)
    save(FINAL, {'status': 'ok', 'progress': progress, 'formal_trading_acceptance': True,
        'original_strategy_complete': False, 'library_goal_complete': False,
        'receipt_file': str(frozen.RECEIPT), 'receipt_sha256': file_sha(frozen.RECEIPT),
        'readback_bound_files': len(evidence), 'evidence_sha256': evidence,
        'implementation_sha256': {**doc['implementation_sha256'], **hashes}})
    return {'status': 'ok', 'receipt': str(FINAL), 'progress': progress, 'fills_checked': fees['fills_checked']}


def finish():
    with operation_lock('data', DATA_WRITER): return finish_locked()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'worker', 'finish'))
    parser.add_argument('--label', choices=tuple(f'pair66-short-{n}' for n in range(1, 6)))
    args = parser.parse_args()
    if (args.action in ('run', 'verify', 'worker')) != (args.label is not None): parser.error('Exactly run/verify/worker require --label')
    print(json.dumps(globals()[args.action](args.label) if args.label else globals()[args.action](), ensure_ascii=False, default=str))
