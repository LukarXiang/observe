"""Capture actual complete-history readback tests in local and copied workspaces."""
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
from scripts import recover_strategy_batch100 as recovery
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch100/20261011-formal-validation')
TESTS = ('tests/unit/test_batch100_research.py', 'tests/unit/test_batch100_acceptance.py',
         'tests/unit/test_batch100_recovery.py')
DRIVERS = ('scripts/accept_strategy_batch100.py', 'scripts/verify_strategy_batch100_formal.py')


def tests(cwd, folder, env=None):
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '--disable-warnings', *TESTS,
               '--junitxml='+str(folder/'tests.xml')]
    process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    (folder/'stdout.log').write_text(process.stdout); (folder/'stderr.log').write_text(process.stderr)
    save(folder/'command.json', {'command': command, 'cwd': str(cwd), 'returncode': process.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Formal tests failed; logs retained')
    cases = list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
    require(len(cases) == 42 and all(len(c) == 0 for c in cases), 'Formal test/skip/failure scope differs')
    return len(cases)


def run():
    require(not DIRECTORY.exists(), 'Immutable formal validation exists'); study.validate()
    recovery.validate(); offline = read(recovery.DIRECTORY/'offline-verification.json')
    require(offline['status'] == 'match' and offline['differences'] == 0, 'Longest replay not ready')
    for mode in ('study', 'replay'):
        for name, digest in offline['output_sha256'].items():
            require(file_sha(recovery.DIRECTORY/mode/'assembled'/name) == digest, 'Replay output changed')
    DIRECTORY.mkdir(parents=True); local = DIRECTORY/'local'; local.mkdir()
    local_passed = tests(Path.cwd(), local.resolve())
    isolated = Path(tempfile.mkdtemp(prefix='observe-batch100-formal-check-'))
    paths = [p for name in ('src', 'scripts') for p in Path(name).rglob('*.py')]
    paths += [*(Path(n) for n in TESTS), Path('pyproject.toml'), Path('uv.lock')]
    paths += [Path('repo/量化策略源代码')/n for n in study.SOURCE_PATHS]
    paths += [study.DIRECTORY/'sources/0.source', study.DIRECTORY/'sources/1.source', study.REGISTRY,
        study.DIRECTORY/'ETF-input.parquet', recovery.DIRECTORY/'study/assembled/feature-prefix.parquet',
        recovery.DIRECTORY/'study/assembled/summary.json']
    row = read(study.DIRECTORY/'dependencies.json')['candidate']
    paths += [Path(row['file']), *(Path(w['file']) for w in row['wire_responses'])]
    copied = {}
    for path in sorted(set(paths)):
        target = isolated/path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); copied[str(path)] = file_sha(target)
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated/'src')+os.pathsep+str(isolated)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    folder = DIRECTORY/'isolated'; folder.mkdir()
    code = ('import json;import scripts.research_strategy_batch100 as s;import scripts.accept_strategy_batch100 as a;'
        'import scripts.recover_strategy_batch100 as z;'
        'import observe.strategy_catalog as c;import observe.data.store as d;import observe.runs as r;'
        'print(json.dumps({m.__name__:m.__file__ for m in (s,a,z,c,d,r)}))')
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=isolated, env=env,
        capture_output=True, text=True, timeout=30)
    require(process.returncode == 0, 'Copied imports failed: '+process.stderr)
    actual = json.loads(process.stdout)
    require(len(actual) == 6 and all(Path(p).is_relative_to(isolated) for p in actual.values()), 'Editable import leaked')
    isolated_passed = tests(isolated, folder.resolve(), env)
    save(folder/'isolation.json', {'cwd': str(isolated), 'copied_sha256': copied, 'actual_imports': actual,
        'tests_passed': isolated_passed, 'skips': 0,
        'scope': '42 copied tests with actual complete feature table; no additional full-history worker or model inference'})
    commands = []
    for name, command in (
        ('ruff', [str(Path(sys.executable).parent/'ruff'), 'check', 'src', 'tests',
            'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
            'scripts/research_strategy_batch100.py', 'scripts/verify_strategy_batch100_stage.py',
            'scripts/archive_strategy_batch100_stage.py', *recovery.FILES, *DRIVERS]),
        ('diff', ['git', 'diff', '--check'])):
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (DIRECTORY/(name+'.log')).write_text(process.stdout+process.stderr)
        commands.append({'command': command, 'returncode': process.returncode, 'log_sha256': file_sha(DIRECTORY/(name+'.log'))})
        require(process.returncode == 0, 'Required validation failed: '+name)
    checked = {'status': 'ok', 'local_passed': local_passed, 'isolated_passed': isolated_passed,
        'copied_files': len(copied), 'skips': 0, 'commands': commands,
        'scope': 'Original source/rule/JSON and actual longest dates/price/flags/derived tail rejection; copied imports; not full pytest or trading',
        'implementation_sha256': {n: file_sha(n) for n in (*study.FILES, *recovery.FILES, *DRIVERS, *TESTS)},
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(DIRECTORY.rglob('*')) if p.is_file()}}
    save(DIRECTORY/'checked.json', checked)
    return {'status': 'ok', 'local': local_passed, 'isolated': isolated_passed, 'copied_files': len(copied)}


if __name__ == '__main__': print(json.dumps(run(), ensure_ascii=False))
