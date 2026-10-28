"""Archive exact local commands and isolated source tests before Batch98 acceptance."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from observe.runs import file_sha
from scripts import accept_strategy_batch98 as acceptance
from scripts import research_strategy_batch98 as stage
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = acceptance.GUARD.parent
NEW_FILES = ('scripts/accept_strategy_batch98.py', 'scripts/probe_strategy_batch98.py',
    'scripts/diagnose_strategy_batch98.py', 'scripts/verify_strategy_batch98.py',
    'scripts/recover_strategy_batch98_diagnostics.py', 'tests/unit/test_batch98_json_recovery.py',
    'tests/unit/test_batch98_acceptance.py', 'tests/unit/test_batch98_rsrs_boundary.py')


def record(command, tag, cwd=None, env=None):
    process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    stdout = DIRECTORY/(tag+'.stdout.log'); stderr = DIRECTORY/(tag+'.stderr.log')
    stdout.write_text(process.stdout, encoding='utf-8'); stderr.write_text(process.stderr, encoding='utf-8')
    row = {'command': command, 'returncode': process.returncode, 'cwd': str(cwd or Path.cwd()),
           'stdout': str(stdout), 'stderr': str(stderr), 'stdout_sha256': file_sha(stdout), 'stderr_sha256': file_sha(stderr)}
    save(DIRECTORY/(tag+'-command.json'), row)
    require(process.returncode == 0, 'Validation command failed; actual logs preserved: '+tag)
    return row, process.stdout


def checks():
    require(not DIRECTORY.exists(), 'Immutable validation directory exists'); DIRECTORY.mkdir(parents=True)
    commands = []
    for i, command in enumerate(acceptance.validation_commands()):
        row, stdout = record(command, 'local'+str(i)); commands.append(row)
        if i == 1:
            require(re.search(r'\b53 passed\b', stdout) is not None and not re.search(r'\d+ skipped', stdout),
                    'Expected local test scope incomplete')
    isolated = DIRECTORY/'isolated-workspace'; isolated.mkdir()
    paths = [str(p) for directory in ('src', 'scripts') for p in Path(directory).rglob('*.py')
             if '__pycache__' not in p.parts]
    paths += ['tests/unit/test_batch98_research.py', 'pyproject.toml', 'uv.lock']
    paths += [str(Path('repo/量化策略源代码')/p) for p in stage.SOURCE_PATHS]
    copied = {}
    for name in sorted(set(paths)):
        dest = isolated/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
        require(file_sha(dest) == file_sha(name), 'Isolated copy changed')
        copied[name] = file_sha(dest)
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated.resolve())+os.pathsep+str((isolated/'src').resolve())
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    import_check = ('import importlib,json,pathlib; root=pathlib.Path.cwd().resolve(); '
        'names=["scripts.research_strategy_batch98","scripts.review_strategy_batch59",'
        '"scripts.verify_strategy_batch3","observe.data.store","observe.strategy_catalog"]; '
        'files={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; '
        'assert all(pathlib.Path(p).is_relative_to(root) for p in files.values()),files; '
        'print(json.dumps(files))')
    _, origins = record([sys.executable, '-B', '-c', import_check], 'isolated-imports', isolated, env)
    _, stdout = record([sys.executable, '-B', '-m', 'pytest', '-q', 'tests/unit/test_batch98_research.py'],
                       'isolated-tests', isolated, env)
    require(re.search(r'\b13 passed\b', stdout) is not None and not re.search(r'\d+ skipped', stdout),
            'Isolated test scope incomplete')
    for name, digest in copied.items(): require(file_sha(isolated/name) == digest, 'Isolated checked source changed')
    save(DIRECTORY/'isolated.json', {'status': 'ok', 'cwd': str(isolated.resolve()), 'copied_sha256': copied,
        'copied_files': len(copied), 'actual_imports': json.loads(origins), 'tests_passed': 13, 'skips': 0,
        'synthetic_tests_not_trading': True})
    checked_paths = [n for n in stage.FILES if n != 'src/observe/strategy_catalog.py']+list(NEW_FILES)
    evidence = {str(p): file_sha(p) for p in DIRECTORY.iterdir() if p.is_file()}
    result = {'status': 'ok', 'commands': commands, 'local_passed': 53, 'local_skipped': 0,
        'isolated_passed': 13, 'isolated_skipped': 0, 'copied_files': len(copied),
        'evidence_sha256': evidence, 'implementation_sha256': {n: file_sha(n) for n in checked_paths},
        'scope': 'Required ruff/diff plus 53 related tests and 13 isolated source tests; not whole pytest suite'}
    save(acceptance.GUARD, result)
    return {'status': 'ok', 'local_passed': 53, 'isolated_passed': 13, 'copied_files': len(copied)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['checks'])
    parser.parse_args(); print(json.dumps(checks(), ensure_ascii=False))
