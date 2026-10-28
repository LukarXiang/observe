"""New-directory repair of Arrow-array JSON representation; original rules stay frozen."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pandas as pd

from observe.runs import file_sha
from scripts import research_strategy_batch98 as frozen
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch98/20261011-original-components-json-recovery')
FILES = ('scripts/recover_strategy_batch98_diagnostics.py', 'tests/unit/test_batch98_json_recovery.py')
ORIGINAL_PREPARED_SHA = '5640a4b2204d010df5da8d14faaeadc21b3e26a200e4ba77bb2fcb7203a023b2'
canonical_before = frozen.canonical


def canonical(value):
    if isinstance(value, pd.api.extensions.ExtensionArray): return canonical_before(value.tolist())
    return canonical_before(value)


def prepare():
    require(not DIRECTORY.exists(), 'Immutable recovery directory exists')
    require(file_sha(frozen.DIRECTORY/'prepared.json') == ORIGINAL_PREPARED_SHA, 'Original prepared differs')
    original = read(frozen.DIRECTORY/'prepared.json')
    off = read(frozen.DIRECTORY/'offline-verification.json')
    for mode in ('study', 'replay'):
        for name, digest in off['outputs_sha256'].items():
            require(file_sha(frozen.DIRECTORY/mode/name) == digest, 'Original artifact differs')
    DIRECTORY.mkdir(parents=True)
    for name in ('index.parquet', 'dependencies.json'):
        shutil.copyfile(frozen.DIRECTORY/name, DIRECTORY/name)
    for name in ('sources', 'implementation'):
        shutil.copytree(frozen.DIRECTORY/name, DIRECTORY/name)
    for name in FILES:
        dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
    prepared = {**original, 'implementation_sha256': {
        **original['implementation_sha256'], **{name: file_sha(name) for name in FILES}},
        'recovery_antecedent': {'original_prepared': str(frozen.DIRECTORY/'prepared.json'),
            'original_prepared_sha256': ORIGINAL_PREPARED_SHA, 'original_outputs_sha256': off['outputs_sha256'],
            'change': 'Only pandas ExtensionArray JSON serialization; no arithmetic/source rules/inputs changed'}}
    save(DIRECTORY/'prepared.json', prepared)
    return {'status': 'prepared', 'source_rules_changed': False}


def unchanged_semantics():
    original = read(frozen.DIRECTORY/'study/diagnostics.json')
    current = read(DIRECTORY/'study/diagnostics.json')
    require(original['case_count'] == current['case_count'] == 27, 'Scope differs')
    for before, after in zip(original['cases'], current['cases'], strict=True):
        if before['case'] == 'rank_excludes_top3_selects17':
            require(before['error'] == after['error'] is None and len(after['value']) == 17 and
                    before['value'] == str(pd.array(after['value'], dtype='str')),
                    'Ranking content or diagnosed representation differs')
        else: require(before == after, 'Non-ranking diagnostic semantics changed')
    for name in ('index-components.parquet', 'summary.json'):
        require(file_sha(frozen.DIRECTORY/'study'/name) == file_sha(DIRECTORY/'study'/name),
                'Price/OLS arithmetic changed')
    result = {'status': 'ok', 'diagnostics': 27, 'representation_changes': 1,
        'nonranking_diagnostics_unchanged': 26, 'ranking_elements': 17,
        'economic_rules_changed': False, 'arithmetic_outputs_byte_identical': 2,
        'original_and_recovery_sha256': {str(p): file_sha(p) for p in (
            frozen.DIRECTORY/'study/diagnostics.json', DIRECTORY/'study/diagnostics.json')}}
    save(DIRECTORY/'representation-verification.json', result); return result


def worker(mode):
    with patch.object(frozen, 'DIRECTORY', DIRECTORY), patch.object(frozen, 'canonical', canonical):
        return frozen.worker(mode)


def offline():
    with patch.object(frozen, 'DIRECTORY', DIRECTORY): result = frozen.offline()
    unchanged_semantics(); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'study', 'replay', 'worker', 'offline'])
    parser.add_argument('--mode', choices=['study', 'replay']); args = parser.parse_args()
    if args.action == 'prepare': result = prepare()
    elif args.action == 'worker': result = worker(args.mode)
    elif args.action == 'offline': result = offline()
    else:
        # frozen.run intentionally launches the original module; the repaired worker is explicit here.
        import subprocess
        import sys
        folder = DIRECTORY/(args.action+'-command'); folder.mkdir(exist_ok=False)
        command = [sys.executable, '-B', '-m', 'scripts.recover_strategy_batch98_diagnostics', 'worker', '--mode', args.action]
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (folder/'stdout.log').write_text(process.stdout, encoding='utf-8')
        (folder/'stderr.log').write_text(process.stderr, encoding='utf-8')
        save(folder/'command.json', {'command': command, 'returncode': process.returncode,
            'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
        require(process.returncode == 0, 'Recovery worker failed; logs retained')
        result = json.loads(process.stdout.splitlines()[-1])
    print(json.dumps(result, ensure_ascii=False))
