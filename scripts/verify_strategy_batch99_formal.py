"""Actual formal tests, isolated imports and a further fresh C-VIX/option replay."""
import json
import os
from pathlib import Path
import re
import shutil
import sys

from observe.runs import file_sha
from scripts import accept_strategy_batch99 as acceptance
from scripts import recover_strategy_batch99_namespace as recovery
from scripts import verify_strategy_batch99 as helpers
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = acceptance.GUARD.parent
TESTS = ('tests/unit/test_batch99_research.py', 'tests/unit/test_batch99_namespace_recovery.py',
         'tests/unit/test_batch99_timing.py', 'tests/unit/test_batch99_acceptance.py')
FILES = ('scripts/accept_strategy_batch99.py', 'scripts/diagnose_strategy_batch99_timing.py',
         'scripts/verify_strategy_batch99_formal.py', *TESTS)


def checks():
    DIRECTORY.mkdir(parents=True, exist_ok=False); helpers.DIRECTORY = DIRECTORY
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
        *[n for n in FILES if n.endswith('.py')], 'scripts/research_strategy_batch99.py',
        'scripts/probe_strategy_batch99.py', 'scripts/recover_strategy_batch99_namespace.py',
        'scripts/archive_strategy_batch99_stage.py', 'scripts/verify_strategy_batch99.py'],
        helpers.test_command([*TESTS, 'tests/unit/test_research_metadata.py']), ['git', 'diff', '--check']]
    rows = []
    for i, command in enumerate(commands):
        row, output = helpers.record(command, 'formal-local'+str(i)); rows.append(row)
        if i == 1: require(re.search(r'\b66 passed\b', output) is not None and not re.search(r'\d+ skipped', output), 'Local scope incomplete')
    isolated = DIRECTORY/'isolated-workspace'; isolated.mkdir()
    paths = [str(p) for folder in ('src', 'scripts') for p in Path(folder).rglob('*.py') if '__pycache__' not in p.parts]
    paths += [*TESTS, 'pyproject.toml', 'uv.lock']
    paths += [str(Path('repo/量化策略源代码')/p) for p in acceptance.stage.SOURCE_PATHS]
    paths += [str(recovery.DIRECTORY/n) for n in ('study/ETF-range-components.parquet', 'ETF-input.parquet', 'study/summary.json')]
    copied = {}
    for name in sorted(set(paths)):
        target = isolated/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        require(file_sha(target) == file_sha(name), 'Isolated bytes changed'); copied[name] = file_sha(target)
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated.resolve())+os.pathsep+str((isolated/'src').resolve())
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    program = ('import importlib,json,pathlib; root=pathlib.Path.cwd().resolve(); '
        'names=["scripts.accept_strategy_batch99","scripts.research_strategy_batch99",'
        '"scripts.recover_strategy_batch99_namespace","scripts.diagnose_strategy_batch99_timing",'
        '"scripts.verify_strategy_batch3","observe.data.store","observe.strategy_catalog"]; '
        'files={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; '
        'assert all(pathlib.Path(p).is_relative_to(root) for p in files.values()),files; print(json.dumps(files))')
    _, origins = helpers.record([sys.executable, '-B', '-c', program], 'formal-isolated-imports', isolated, env)
    _, stdout = helpers.record(helpers.test_command(TESTS), 'formal-isolated-tests', isolated, env)
    require(re.search(r'\b43 passed\b', stdout) is not None and not re.search(r'\d+ skipped', stdout), 'Isolated scope incomplete')
    require(all(file_sha(isolated/n) == sha for n,sha in copied.items()), 'Checked isolated source changed')
    save(DIRECTORY/'isolated.json', {'status': 'ok', 'cwd': str(isolated.resolve()), 'copied_sha256': copied,
        'actual_imports': json.loads(origins), 'tests_passed': 43, 'skips': 0})
    replay = DIRECTORY/'fresh-replay'; replay.mkdir()
    for name in ('prepared.json', 'ETF-input.parquet'):
        shutil.copyfile(recovery.DIRECTORY/name, replay/name)
        require(file_sha(replay/name) == file_sha(recovery.DIRECTORY/name), 'Replay input changed')
    shutil.copytree(recovery.DIRECTORY/'implementation', replay/'implementation')
    program = ('import json; from pathlib import Path; from scripts import recover_strategy_batch99_namespace as worker; '
               'worker.DIRECTORY=Path('+repr(str(replay))+'); print(json.dumps(worker.worker("replay")))')
    command, stdout = helpers.record([sys.executable, '-B', '-c', program], 'formal-fresh-replay')
    result = json.loads(stdout.splitlines()[-1])
    require(result['status'] == 'ok' and all(file_sha(replay/'replay'/name) ==
        file_sha(recovery.DIRECTORY/'study'/name) == sha for name,sha in result['output_sha256'].items()),
        'Further actual recovery replay differs')
    save(DIRECTORY/'fresh-replay.json', {'status': 'match', 'differences': 0, 'fresh_process': True,
        'network_blocked': True, 'command': command, 'output_sha256': result['output_sha256']})
    checked = sorted(set((*acceptance.stage.FILES, *FILES, 'scripts/recover_strategy_batch99_namespace.py',
        'scripts/verify_strategy_batch99.py', 'scripts/probe_strategy_batch99.py')))
    result = {'status': 'ok', 'commands': rows, 'local_passed': 66, 'isolated_passed': 43, 'skips': 0,
        'copied_files': len(copied), 'actual_imports': 7, 'new_trading_backtests': 0,
        'evidence_sha256': {str(p): file_sha(p) for p in DIRECTORY.iterdir() if p.is_file()},
        'implementation_sha256': {n: file_sha(n) for n in checked},
        'scope': '66 related and43 isolated tests plus required Ruff/diff and actual further fresh replay; not full pytest'}
    save(acceptance.GUARD, result)
    return {k:result[k] for k in ('status', 'local_passed', 'isolated_passed', 'copied_files', 'actual_imports')}


if __name__ == '__main__': print(json.dumps(checks(), ensure_ascii=False))
