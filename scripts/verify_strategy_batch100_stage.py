"""Capture local and isolated checks for Batch100; arithmetic acceptance is separate."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from observe.runs import file_sha
from scripts import research_strategy_batch100 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch100/20261011-source-stage-validation')


def tests(cwd, folder, env=None):
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '--disable-warnings',
               'tests/unit/test_batch100_research.py', '--junitxml='+str(folder/'tests.xml')]
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    (folder/'stdout.log').write_text(result.stdout); (folder/'stderr.log').write_text(result.stderr)
    save(folder/'command.json', {'command': command, 'cwd': str(cwd), 'returncode': result.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(result.returncode == 0, 'Stage tests failed; logs retained')
    cases = list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
    require(len(cases) == 11 and all(len(c) == 0 for c in cases), 'Test coverage/error/skip differs')
    return len(cases)


def run():
    require(not DIRECTORY.exists(), 'Immutable validation exists'); study.validate()
    DIRECTORY.mkdir(parents=True); local = DIRECTORY/'local'; local.mkdir()
    local_passed = tests(Path.cwd(), local.resolve())
    isolated = Path(tempfile.mkdtemp(prefix='observe-batch100-source-check-'))
    paths = [p for name in ('src', 'scripts') for p in Path(name).rglob('*.py')]
    paths += [Path('tests/unit/test_batch100_research.py'), Path('pyproject.toml'), Path('uv.lock')]
    paths += [Path('repo/量化策略源代码')/n for n in study.SOURCE_PATHS]
    paths += [study.DIRECTORY/'sources/0.source', study.DIRECTORY/'sources/1.source', study.REGISTRY]
    row = read(study.DIRECTORY/'dependencies.json')['candidate']
    paths += [Path(row['file']), *(Path(w['file']) for w in row['wire_responses'])]
    copied = {}
    for path in sorted(set(paths)):
        target = isolated/path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); copied[str(path)] = file_sha(target)
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated/'src')+os.pathsep+str(isolated)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    folder = DIRECTORY/'isolated'; folder.mkdir()
    code = ('import json;import scripts.research_strategy_batch100 as s;import observe.strategy_catalog as c;'
            'import observe.data.store as d;import observe.runs as r;'
            'print(json.dumps({m.__name__:m.__file__ for m in (s,c,d,r)}))')
    imported = subprocess.run([sys.executable, '-B', '-c', code], cwd=isolated, env=env,
        capture_output=True, text=True, timeout=30)
    require(imported.returncode == 0, 'Isolated import failed: '+imported.stderr)
    actual = json.loads(imported.stdout)
    require(len(actual) == 4 and all(Path(p).is_relative_to(isolated) for p in actual.values()), 'Editable import leaked')
    isolated_passed = tests(isolated, folder.resolve(), env)
    save(folder/'isolation.json', {'cwd': str(isolated), 'copied_sha256': copied, 'actual_imports': actual,
        'tests_passed': isolated_passed, 'skips': 0,
        'scope': 'Fresh code imports and 11 source/operand boundary tests; no full history computation or trading'})
    commands = []
    for name, command in (
        ('ruff', [str(Path(sys.executable).parent/'ruff'), 'check', 'src', 'tests',
            'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
            'scripts/research_strategy_batch100.py', 'scripts/verify_strategy_batch100_stage.py']),
        ('diff', ['git', 'diff', '--check'])):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (DIRECTORY/(name+'.log')).write_text(result.stdout+result.stderr)
        commands.append({'command': command, 'returncode': result.returncode, 'log_sha256': file_sha(DIRECTORY/(name+'.log'))})
        require(result.returncode == 0, 'Required check failed: '+name)
    record = {'status': 'ok_source_stage', 'local_passed': local_passed, 'isolated_passed': isolated_passed,
        'copied_files': len(copied), 'skips': 0, 'commands': commands, 'formal_acceptance': False,
        'scope': 'Source loading, original callback/compatibility boundaries, 400-row prefix first window, independent scalar corruption rejection; longest run still separate',
        'implementation_sha256': {n: file_sha(n) for n in (*study.FILES, 'scripts/verify_strategy_batch100_stage.py')},
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(DIRECTORY.rglob('*')) if p.is_file()},
        'new_trading_backtests': 0}
    save(DIRECTORY/'checked.json', record)
    return {'status': record['status'], 'local': local_passed, 'isolated': isolated_passed, 'copied': len(copied)}


if __name__ == '__main__': print(json.dumps(run(), ensure_ascii=False))
