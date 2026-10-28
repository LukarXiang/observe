"""New-directory recovery of the frozen Batch94 reference's network call contract."""
import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

import requests

from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts import reference_strategy_batch94 as frozen
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch94/20261010-kalman-fork-reference-recovery')
FAILED = frozen.DIRECTORY
FILES = ('scripts/recover_strategy_batch94_reference.py', 'tests/unit/test_batch94_reference_recovery.py')


@contextmanager
def network_blocked():
    def forbidden(*args, **kwargs): raise AssertionError('Batch94 reference network forbidden')
    with ExitStack() as stack:
        for obj, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'), (socket.socket, 'connect'),
                (socket.socket, 'connect_ex'), (requests.sessions.Session, 'request')):
            stack.enter_context(patch.object(obj, name, forbidden))
        yield


def prepare():
    require(not DIRECTORY.exists(), 'Recovery directory already exists')
    frozen.historical_stage(); frozen.validate_runtime(FAILED); frozen.validate_index('data')
    require(read(FAILED/'smoke-command.json')['returncode'] == 1 and
        file_sha(FAILED/'smoke.log') == read(FAILED/'smoke-command.json')['log_sha256'] and
        "network_disabled() missing 1 required positional argument: 'function'" in (FAILED/'smoke.log').read_text(),
        'Recovery antecedent differs')
    failed_hashes = {str(p): file_sha(p) for p in sorted(FAILED.rglob('*')) if p.is_file()}
    with operation_lock('data', DATA_WRITER):
        shutil.copytree(FAILED, DIRECTORY)
        save(DIRECTORY/'recovery-antecedent.json', {'failure': 'network_disabled callback mistaken for context manager',
            'failed_directory': str(FAILED), 'failed_evidence_sha256': failed_hashes,
            'model_or_economic_rules_changed': False, 'old_files_modified': False,
            'protection_baseline': 'Copied original pre-compute baseline; completion rehashes all its controls',
            'new_trading_backtests': 0})
        hashes = {name: file_sha(name) for name in FILES}
        for name in FILES:
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        save(DIRECTORY/'recovery-implementation.json', hashes)
    return {'status': 'prepared', 'directory': str(DIRECTORY)}


def validate_recovery():
    antecedent = read(DIRECTORY/'recovery-antecedent.json')
    require(all(file_sha(name) == sha for name, sha in antecedent['failed_evidence_sha256'].items()),
        'Failed original evidence changed')
    hashes = read(DIRECTORY/'recovery-implementation.json')
    require(set(hashes) == set(FILES) and all(file_sha(name) == sha for name, sha in hashes.items()),
        'Recovery implementation changed')
    return hashes


def worker(mode):
    validate_recovery()
    with patch.object(frozen, 'DIRECTORY', DIRECTORY), patch.object(frozen, 'network_disabled', network_blocked):
        return frozen.worker(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'compute', 'replay', 'worker', 'protect'))
    parser.add_argument('--mode', choices=('smoke', 'compute', 'replay'))
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare()
    elif args.action == 'worker': result = worker(args.mode)
    elif args.action == 'protect':
        validate_recovery(); frozen.historical_stage(); frozen.validate_runtime(DIRECTORY); frozen.validate_index('data')
        result = protect('data', DIRECTORY)
    else:
        env = os.environ.copy(); env['OBSERVE_BATCH94_WORKER'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-B', '-m', 'scripts.recover_strategy_batch94_reference',
            'worker', '--mode', args.action], env=env, check=True)
        result = {'status': 'worker_completed', 'mode': args.action}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
