"""Persist local and isolated Batch99 checks; no formal source or trading acceptance."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from observe.runs import file_sha
from scripts import research_strategy_batch99 as stage
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-stage-validation')


def test_command(paths):
    program = ('import sys; from scripts.verify_offline_tests import main; '
               'from scripts.recover_strategy_batch94_reference import network_blocked\n'
               'with network_blocked(): sys.exit(main('+repr(['-q', *paths])+'))')
    return [sys.executable, '-B', '-c', program]


def record(command, tag, cwd=None, env=None):
    process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=240)
    logs = {}
    for kind, content in (('stdout', process.stdout), ('stderr', process.stderr)):
        path = DIRECTORY/(tag+'.'+kind+'.log'); path.write_text(content, encoding='utf-8')
        logs[kind] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'command': command, 'cwd': str(cwd or Path.cwd()), 'returncode': process.returncode, 'logs': logs}
    save(DIRECTORY/(tag+'-command.json'), result)
    require(process.returncode == 0, 'Actual validation failed; logs retained: '+tag)
    return result, process.stdout


def checks(recovery=False):
    global DIRECTORY
    if recovery: DIRECTORY = DIRECTORY.with_name('20261011-stage-validation-recovery')
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    tests = ['tests/unit/test_batch99_research.py']
    extra = ['scripts/recover_strategy_batch99_namespace.py'] if recovery else []
    if recovery: tests.append('tests/unit/test_batch99_namespace_recovery.py')
    expected = 21 if recovery else 18
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
        'scripts/research_strategy_batch99.py', 'scripts/probe_strategy_batch99.py', 'scripts/verify_strategy_batch99.py', *extra],
        test_command([*tests, 'tests/unit/test_research_metadata.py']),
        ['git', 'diff', '--check']]
    rows = []; count = None
    for i, command in enumerate(commands):
        row, stdout = record(command, 'local'+str(i)); rows.append(row)
        if i == 1:
            match = re.search(r'\b(\d+) passed\b', stdout)
            require(match is not None and not re.search(r'\d+ skipped', stdout), 'Test scope incomplete')
            count = int(match.group(1))
    isolated = DIRECTORY/'isolated-workspace'; isolated.mkdir()
    paths = [str(p) for folder in ('src', 'scripts') for p in Path(folder).rglob('*.py')
             if '__pycache__' not in p.parts]
    paths += [*tests, 'pyproject.toml', 'uv.lock']
    paths += [str(Path('repo/量化策略源代码')/p) for p in stage.SOURCE_PATHS]
    copied = {}
    for name in sorted(set(paths)):
        target = isolated/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        digest = file_sha(name); require(file_sha(target) == digest, 'Isolated copy differs')
        copied[name] = digest
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated.resolve())+os.pathsep+str((isolated/'src').resolve())
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    imports = ('import importlib,json,pathlib; root=pathlib.Path.cwd().resolve(); '
        'names=["scripts.research_strategy_batch99","scripts.verify_strategy_batch3",'
        '"scripts.recover_strategy_batch94_reference","observe.data.store","observe.strategy_catalog"]; '
        'files={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; '
        'assert all(pathlib.Path(p).is_relative_to(root) for p in files.values()),files; '
        'print(json.dumps(files))')
    _, origins = record([sys.executable, '-B', '-c', imports], 'isolated-imports', isolated, env)
    _, stdout = record(test_command(tests), 'isolated-tests', isolated, env)
    require(re.search(r'\b'+str(expected)+r' passed\b', stdout) is not None and not re.search(r'\d+ skipped', stdout),
            'Isolated test scope incomplete')
    require(all(file_sha(isolated/name) == digest for name, digest in copied.items()), 'Checked isolated files differ')
    save(DIRECTORY/'isolated.json', {'status': 'ok', 'copied_sha256': copied,
        'copied_files': len(copied), 'actual_imports': json.loads(origins), 'tests_passed': expected, 'skips': 0})
    checked = (*stage.FILES, 'scripts/probe_strategy_batch99.py', 'scripts/verify_strategy_batch99.py', *extra,
               *[n for n in tests if n not in stage.FILES])
    result = {'status': 'ok', 'commands': rows, 'local_passed': count, 'isolated_passed': expected,
              'skips': 0, 'copied_files': len(copied), 'new_trading_backtests': 0,
              'scope': 'Related offline tests and required Ruff/diff only; not full pytest or formal acceptance',
              'implementation_sha256': {n: file_sha(n) for n in checked},
              'evidence_sha256': {str(p): file_sha(p) for p in DIRECTORY.iterdir() if p.is_file()}}
    save(DIRECTORY/'checked.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--recovery', action='store_true')
    args = parser.parse_args(); print(json.dumps(checks(args.recovery), ensure_ascii=False))
