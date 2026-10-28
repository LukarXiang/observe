"""Recover contaminated band reference in new directories; original outputs stay frozen."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from observe.runs import file_sha
from scripts import research_strategy_batch99 as stage
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-fresh-namespace-recovery-v2')
FILES = (*stage.FILES, 'scripts/recover_strategy_batch99_namespace.py',
         'tests/unit/test_batch99_namespace_recovery.py')


def compute(frame, loader=None):
    loader = loader or stage.source_namespace
    diagnostics = stage.diagnostics([loader(0), loader(1)])
    # Paper callbacks rebind Price_Range/get_price in their own namespace.
    # Reload source definitions before using the real candidate history.
    components, summary = stage.components(loader(1), frame)
    summary['synthetic_cases'] = diagnostics['case_count']
    return diagnostics, components, summary


def prepare():
    require(not DIRECTORY.exists(), 'Immutable recovery exists')
    require(read(stage.DIRECTORY/'prepared.json')['source_inventory'][1]['sha256'] == stage.SOURCE_SHA[1],
            'Original source identity differs')
    frame = pd.read_parquet(stage.DIRECTORY/'study/ETF-range-components.parquet')
    require(len(frame) == 5249 and frame.original_up.eq(3.).all() and frame.original_down.eq(2.).all(),
            'Contamination antecedent differs')
    DIRECTORY.mkdir(parents=True)
    shutil.copyfile(stage.DIRECTORY/'ETF-input.parquet', DIRECTORY/'ETF-input.parquet')
    for name in FILES:
        target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    save(DIRECTORY/'prepared.json', {'original_prepared': str(stage.DIRECTORY/'prepared.json'),
        'original_prepared_sha256': file_sha(stage.DIRECTORY/'prepared.json'),
        'original_study_sha256': {n: file_sha(stage.DIRECTORY/'study'/n) for n in stage.OUTPUTS},
        'operand_sha256': file_sha(DIRECTORY/'ETF-input.parquet'),
        'implementation_sha256': {n: file_sha(n) for n in FILES},
        'source_sha256': list(stage.SOURCE_SHA), 'economic_rule_changed': False,
        'reason': 'Paper diagnostics rebound Price_Range to constant3/2; fresh source namespace restores original bands',
        'old_program_outputs_preserved': True, 'formal_acceptance': False})
    return {'status': 'prepared', 'directory': str(DIRECTORY)}


def worker(mode):
    require(mode in ('study', 'replay'), 'Unreviewed mode'); before = read(DIRECTORY/'prepared.json')
    require(file_sha(before['original_prepared']) == before['original_prepared_sha256'] and
            all(file_sha(stage.DIRECTORY/'study'/n) == sha for n, sha in before['original_study_sha256'].items()),
            'Original evidence changed')
    require(file_sha(DIRECTORY/'ETF-input.parquet') == before['operand_sha256'], 'Recovery input differs')
    for name, sha in before['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == sha, 'Recovery code differs: '+name)
    target = DIRECTORY/mode; require(not target.exists(), 'Immutable recovery output exists')
    with network_blocked():
        diag, frame, summary = compute(pd.read_parquet(DIRECTORY/'ETF-input.parquet'))
        require(not frame.original_up.eq(3.).all() and not frame.original_down.eq(2.).all(),
                'Constant paper bands leaked into historical reference')
        example = frame.loc[frame.date.eq('2018-07-20')].iloc[0]
        require(example.original_up == 2.487 and example.original_down == 2.461,
                'Original source historical example bands differ')
        target.mkdir(); save(target/'diagnostics.json', diag)
        require(read(target/'diagnostics.json') == diag and
                file_sha(target/'diagnostics.json') == before['original_study_sha256']['diagnostics.json'],
                'Recovery changed diagnostics or container shapes')
        frame.to_parquet(target/'ETF-range-components.parquet', index=False)
        save(target/'summary.json', summary)
    return {'status': 'ok', 'mode': mode, 'output_sha256': {n: file_sha(target/n) for n in stage.OUTPUTS}}


def run(mode):
    folder = DIRECTORY/(mode+'-command'); folder.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.recover_strategy_batch99_namespace', 'worker', '--mode', mode]
    process = subprocess.run(command, capture_output=True, text=True, timeout=240)
    for name, content in (('stdout.log', process.stdout), ('stderr.log', process.stderr)):
        (folder/name).write_text(content, encoding='utf-8')
    save(folder/'command.json', {'command': command, 'returncode': process.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Recovery worker failed; logs retained')
    return json.loads(process.stdout.splitlines()[-1])


def offline():
    require(all(read(DIRECTORY/(mode+'-command')/'command.json')['returncode'] == 0
                for mode in ('study', 'replay')), 'Recovery command failed')
    require(all(file_sha(DIRECTORY/'study'/n) == file_sha(DIRECTORY/'replay'/n) for n in stage.OUTPUTS),
            'Recovery replay differs')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study/ETF-range-components.parquet'),
        pd.read_parquet(DIRECTORY/'replay/ETF-range-components.parquet'), check_exact=True)
    result = {'status': 'match', 'differences': 0, 'fresh_process': True, 'network_blocked': True,
              'diagnostics_unchanged': True, 'economic_rule_changed': False, 'new_trading_backtests': 0,
              'output_sha256': {n: file_sha(DIRECTORY/'study'/n) for n in stage.OUTPUTS},
              'prepared_sha256': file_sha(DIRECTORY/'prepared.json')}
    save(DIRECTORY/'offline-verification.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'replay', 'worker', 'offline'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    if args.action == 'worker': result = worker(args.mode)
    elif args.action in ('study', 'replay'): result = run(args.action)
    else: result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
