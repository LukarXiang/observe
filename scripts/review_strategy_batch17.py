"""Archive scheduled-time dependency corrections without running strategies."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

from observe.data.store import Store
from observe.runs import environment, file_sha, write_json
from observe.strategy_catalog import catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

FILES = {'src/observe/strategy_catalog.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch17_archive.py',
         'scripts/review_strategy_batch17.py'}
CORE = {'baseline.json', 'catalog-before.json', 'red-regression.json', 'scan.json', 'offline-scan.json', 'code-before/strategy_catalog.py'}
GAP = '盘中事件/撮合与对应分钟或Tick股票池'


def save(path, value):
    require(not path.exists(), f'Archive exists: {path}')
    write_json(path, value)


def start(root, directory):
    checkpoint(root, directory, 'Batch17 schedule dependency regression and full-library scan started')
    latest = read(Path(root) / 'catalog/strategies/latest.json')
    old = Path(latest['directory'])
    save(directory / 'catalog-before.json', {'catalog_id': latest['catalog_id'],
        'files_sha256': {str(old / name): file_sha(old / name) for name in ('catalog.json', 'summary.json', 'catalog.parquet')}})
    copied = directory / 'code-before'; copied.mkdir()
    shutil.copyfile('src/observe/strategy_catalog.py', copied / 'strategy_catalog.py')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_schedules.py', '--tb=no']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'code_sha256': file_sha(copied / 'strategy_catalog.py'),
        'test_sha256': file_sha('tests/unit/test_catalog_schedules.py')})
    require(result.returncode == 1 and '28 failed, 5 passed' in result.stdout, 'Expected schedule regression not reproduced')
    return {'status': 'ok', 'regression': '28 failed, 5 passed on original code'}


def scan(root, directory):
    require(not (directory / 'scan.json').exists(), 'Scan exists')
    before = read(directory / 'catalog-before.json')
    require(all(file_sha(p) == sha for p, sha in before['files_sha256'].items()), 'Old catalog changed')
    old_path = next(p for p in before['files_sha256'] if Path(p).name == 'catalog.json')
    old = {r['path']: r for r in read(old_path)}
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    current_path = Path(catalog['output']) / 'catalog.json'; current = read(current_path)
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old), 'Source set differs')
    changes = []
    for row in current:
        previous = old[row['path']]
        require(row['bytes_sha256'] == previous['bytes_sha256'], 'Source changed')
        require(all(row[k] == previous[k] for k in ('rules', 'fidelity', 'review_status', 'implementations')), 'Manual rules changed')
        keys = ('scheduled_times', 'gaps', 'status')
        difference = {k: {'before': previous[k], 'after': row[k]} for k in keys if row[k] != previous[k]}
        if difference: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': difference})
    result = {'status': 'ok', 'catalog': catalog, 'catalog_file': str(current_path), 'catalog_sha256': file_sha(current_path),
        'changes': changes, 'schedule_detection_counts': dict(Counter(r['schedule_detection'] for r in current)),
        'intraday_gap_before': sum(GAP in r['gaps'] for r in old.values()),
        'intraday_gap_after': sum(GAP in r['gaps'] for r in current), 'strategy_results': [],
        'limits': ['Static call candidates; not proof of active control flow or platform runtime frequency',
            'Dynamic times, aliases and tokenization failures still require manual review',
            'Manual rules and original bytes unchanged; no new backtest or data publication']}
    save(directory / 'scan.json', result)
    archive(root, 'Batch17 schedule literal detection corrected; full-library changes archived, no trading rules changed')
    return {'status': 'ok', 'changed_sources': len(changes), 'catalog': catalog['catalog_id']}


def checks(root, directory):
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Evidence missing')
    require(not (directory / 'checks.json').exists(), 'Checks exist')
    verify_offline_scan(directory, read(directory / 'scan.json'), read(directory / 'offline-scan.json'))
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_catalog_dependencies.py',
        'tests/unit/test_batch17_archive.py',
        'tests/integration/test_strategy_merge.py'],
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py',
            'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py', 'scripts/review_strategy_batch17.py'],
        ['git', 'diff', '--check']]
    results = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        print(results[-1], flush=True); require(result.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def verify(root, directory):
    require(not (directory / 'offline-scan.json').exists(), 'Offline receipt exists')
    scanned = read(directory / 'scan.json')
    offline = directory / 'catalog-offline'; offline.mkdir(exist_ok=False)
    proof = offline / 'catalog/strategies/implementation-evidence.json'; proof.parent.mkdir(parents=True)
    original_proof = Path(root) / 'catalog/strategies/implementation-evidence.json'
    shutil.copyfile(original_proof, proof)
    def forbidden(*args, **kwargs): raise AssertionError('Catalog recheck attempted network')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden):
        repeated = catalog_strategies(offline, 'repo/量化策略源代码')
    require(repeated['catalog_id'] == scanned['catalog']['catalog_id'], 'Offline catalog identity differs')
    pairs = []
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        old = Path(scanned['catalog']['output']) / name; new = Path(repeated['output']) / name
        require(file_sha(old) == file_sha(new), f'Offline {name} differs')
        pairs.append({'original_file': str(old), 'recomputed_file': str(new), 'sha256': file_sha(old)})
    save(directory / 'offline-scan.json', {'status': 'match', 'differences': 0, 'files': pairs,
        'implementation_evidence_sha256': file_sha(proof), 'original_evidence_file': str(original_proof),
        'scan_sha256': file_sha(directory / 'scan.json'), 'network': 'socket.connect and connect_ex forbidden',
        'not_a_strategy_reproduction': True})
    return {'status': 'match', 'differences': 0}


def verify_offline_scan(directory, scanned, offline):
    require(offline['status'] == 'match' and offline['differences'] == 0
        and offline['scan_sha256'] == file_sha(directory / 'scan.json'), 'Offline verification differs')
    names = {'catalog.json', 'summary.json', 'catalog.parquet'}
    pairs = offline['files']
    require(len(pairs) == len(names) and {Path(p['original_file']).name for p in pairs} == names, 'Offline file set incomplete')
    for pair in pairs:
        name = Path(pair['original_file']).name
        expected = Path(scanned['catalog']['output']) / name
        repeated = directory / 'catalog-offline/catalog/strategies' / scanned['catalog']['catalog_id'] / name
        require(Path(pair['original_file']) == expected and Path(pair['recomputed_file']) == repeated, 'Offline file path differs')
        require(file_sha(expected) == pair['sha256'] == file_sha(repeated), 'Offline catalog file changed')


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch17-verification.json'); require(not output.exists(), 'Receipt exists')
    checked = read(directory / 'checks.json')
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']), 'Checks incomplete')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked sets incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    before = read(directory / 'catalog-before.json'); scanned = read(directory / 'scan.json')
    offline = read(directory / 'offline-scan.json')
    verify_offline_scan(directory, scanned, offline)
    require(file_sha(offline['original_evidence_file']) == offline['implementation_evidence_sha256'], 'Implementation evidence changed')
    require(all(file_sha(p) == sha for p, sha in before['files_sha256'].items()), 'Old catalog changed')
    require(file_sha(scanned['catalog_file']) == scanned['catalog_sha256'], 'New catalog changed')
    require(read(Path(root) / 'catalog/strategies/latest.json')['catalog_id'] == scanned['catalog']['catalog_id'], 'Latest catalog differs')
    for row in read(scanned['catalog_file']):
        require(file_sha(Path('repo/量化策略源代码') / row['path']) == row['bytes_sha256'], 'Original source changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / Path(path).name; require(not target.exists(), 'Filename collision')
        shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    result = {'status': 'ok', 'environment': environment(), 'scan': scanned, 'offline_scan': offline, 'checks': checked, 'protection': protection,
        'progress': archive(root, 'Batch17 schedule dependency correction verified; originals and experiments protected'), 'strategy_results': []}
    save(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'scan', 'verify', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch17/20261005-schedule-dependencies'))
    args = parser.parse_args()
    print({'start': start, 'scan': scan, 'verify': verify, 'checks': checks, 'finish': finish}[args.action](args.root, args.directory), flush=True)
