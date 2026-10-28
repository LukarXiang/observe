"""Fresh source gates and saved-reference readback before Batch102 acceptance."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from observe.runs import file_sha
from scripts import accept_strategy_batch102 as acceptance
from scripts import diagnose_strategy_batch102_transitions as transitions
from scripts import verify_strategy_batch102_stage as reference_check
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = acceptance.CHECK
TESTS = ('tests/unit/test_batch102_research.py', 'tests/unit/test_batch102_reference.py',
    'tests/unit/test_batch102_transitions.py', 'tests/unit/test_batch102_acceptance.py')
FILES = ('scripts/accept_strategy_batch102.py', 'scripts/diagnose_strategy_batch102_transitions.py',
    'scripts/verify_strategy_batch102_formal.py', *TESTS, *acceptance.SPECS)


def worker():
    import pytest
    import observe.runs as runs
    from observe.ledger import book
    root = Path.cwd().resolve()
    modules = (acceptance, transitions, acceptance.study, acceptance.study.original, book, runs)
    imports = {m.__name__: str(Path(inspect.getfile(m)).resolve()) for m in modules}
    require(all(Path(name).is_relative_to(root) for name in imports.values()), 'Actual imports escaped isolated tree')
    with network_blocked():
        rc = pytest.main(['-q', *TESTS, '--junitxml='+str(DIRECTORY/'tests.xml')]); require(rc == 0, 'Isolated formal tests failed')
        save(DIRECTORY/'diagnostics.json', transitions.diagnostics())
        save(DIRECTORY/'saved-readback.json', reference_check.read_saved())
    save(DIRECTORY/'actual-imports.json', imports)
    return {'status': 'ok', 'actual_imports': len(imports)}


def count_tests(path):
    suites = ET.parse(path).getroot().findall('testsuite')
    require(suites and sum(int(s.get('failures', 0))+int(s.get('errors', 0))+int(s.get('skipped', 0)) for s in suites) == 0, 'JUnit failures/skips')
    return sum(int(s.get('tests', 0)) for s in suites)


def verify():
    require(not DIRECTORY.exists(), 'Immutable formal validation exists'); DIRECTORY.mkdir(parents=True)
    phase = read(acceptance.PHASE)
    for name, sha in phase['evidence_sha256'].items(): require(file_sha(name) == sha, 'Historical stage changed')
    for name, sha in read(reference_check.DIRECTORY/'checked.json')['copied_file_sha256'].items():
        require(file_sha(reference_check.DIRECTORY/'isolated'/name) == sha, 'Old copied byte changed')
    save(DIRECTORY/'implementation.json', {n: file_sha(n) for n in FILES})
    for name in FILES:
        dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
    # A fresh Python process re-evaluates the supplemental unchanged callbacks.
    command = [sys.executable, '-c', 'from scripts.diagnose_strategy_batch102_transitions import diagnostics; '
        'import json; print(json.dumps(diagnostics(),ensure_ascii=False,sort_keys=True,allow_nan=False))']
    result = subprocess.run(command, capture_output=True, text=True)
    (DIRECTORY/'diagnostic-worker.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'diagnostic-command.json', {'returncode': result.returncode,
        'log_sha256': file_sha(DIRECTORY/'diagnostic-worker.log')}); require(result.returncode == 0, 'Fresh callback worker failed')
    save(DIRECTORY/'diagnostics.json', json.loads(result.stdout)); save(DIRECTORY/'saved-readback.json', reference_check.read_saved())
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(DIRECTORY/'local-tests.xml')],
        capture_output=True, text=True)
    (DIRECTORY/'local-tests.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'local-command.json', {'returncode': result.returncode,
        'log_sha256': file_sha(DIRECTORY/'local-tests.log')}); require(result.returncode == 0, 'Local formal tests failed')
    isolated = DIRECTORY/'isolated'; isolated.mkdir()
    prepared = read(acceptance.study.DIRECTORY/'prepared.json'); proof = read(acceptance.study.DIRECTORY/'offline-verification.json')
    names = {str(p) for folder in ('src', 'scripts') for p in Path(folder).rglob('*.py') if '__pycache__' not in p.parts}
    names.update(FILES); names.update(prepared['evidence_sha256']); names.update(proof['outputs_sha256'])
    names.update(str(acceptance.study.DIRECTORY/p) for p in ('prepared.json', 'offline-verification.json'))
    names.update(str(acceptance.study.DIRECTORY/'implementation'/p) for p in prepared['implementation_sha256'])
    copied = {}
    for name in sorted(names):
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'Unsafe copy path')
        dest = isolated/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        sha = file_sha(path); require(file_sha(dest) == sha, 'Copied SHA differs'); copied[name] = sha
    (isolated/DIRECTORY).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join((str((isolated/'src').resolve()), str(isolated.resolve())))
    result = subprocess.run([sys.executable, '-m', 'scripts.verify_strategy_batch102_formal', 'worker'], cwd=isolated,
        env=env, capture_output=True, text=True)
    (DIRECTORY/'isolated.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'isolated-command.json', {'returncode': result.returncode,
        'log_sha256': file_sha(DIRECTORY/'isolated.log')}); require(result.returncode == 0, 'Isolated formal checks failed')
    for name in ('diagnostics.json', 'saved-readback.json', 'actual-imports.json', 'tests.xml'):
        shutil.copyfile(isolated/DIRECTORY/name, DIRECTORY/('isolated-'+name))
    require(read(DIRECTORY/'diagnostics.json') == read(DIRECTORY/'isolated-diagnostics.json') and
        read(DIRECTORY/'saved-readback.json') == read(DIRECTORY/'isolated-saved-readback.json'), 'Fresh saved values/callbacks differ')
    tests = {'local': count_tests(DIRECTORY/'local-tests.xml'), 'isolated': count_tests(DIRECTORY/'isolated-tests.xml')}
    require(tests == {'local': 34, 'isolated': 34} and read(DIRECTORY/'diagnostics.json')['case_count'] == 11, 'Actual counts differ')
    evidence = {n: file_sha(n) for n in FILES}
    evidence.update({str(p): file_sha(p) for p in DIRECTORY.glob('*') if p.is_file()})
    save(DIRECTORY/'checked.json', {'status': 'ok_formal_ready', 'formal_trading_acceptance': False,
        'tests': tests, 'supplemental_diagnostics': 11, 'diagnostic_replay_differences': 0,
        'saved_readback': read(DIRECTORY/'saved-readback.json'), 'evidence_sha256': evidence,
        'copied_file_sha256': copied, 'actual_imports': read(DIRECTORY/'isolated-actual-imports.json'),
        'new_trading_backtests': 0, 'independent_agent_review': False})
    return {'status': 'ok_formal_ready', 'tests': tests, 'copied_files': len(copied)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('verify', 'worker'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](), ensure_ascii=False))
