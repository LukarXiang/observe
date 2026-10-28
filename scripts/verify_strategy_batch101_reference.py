"""Read saved longest ES rows and capture local/copied validation, not trading acceptance."""
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch101 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch101/20261011-longest-reference-validation')
TESTS = ('tests/unit/test_batch101_research.py', 'tests/unit/test_batch101_inputs.py',
         'tests/unit/test_batch101_reference.py', 'tests/unit/test_batch101_reference_readback.py',
         'tests/unit/test_batch101_acceptance.py')


def readback(frame, summary, frames, open_days):
    require(summary['strict_usable'] is False and summary['original_daily_return_pipeline_runnable'] is False and
            summary['new_trading_backtests'] == summary['new_strategy_parent_runs'] ==
            summary['full_original_portfolio_weights'] == 0, 'Saved reference promoted to trades')
    require(not frame.duplicated(['instrument', 'asof_native_close', 'policy']).any() and
            set(frame.instrument) == set(frames) == set(summary['profiles']), 'Saved identities differ')
    require(len(frame) == summary['reference_rows'] == 2*sum(len(f)-119 for f in frames.values()) and
            summary['native_windows_per_policy'] == len(frame)//2, 'Saved longest scope differs')
    require(frame.native_rows.eq(120).all() and not frame.original_return_input_proven.any(), 'Saved original input claim differs')
    calendar = set(open_days); cells = 0; largest = 0.; crossing = 0
    for symbol, native in sorted(frames.items()):
        selected = frame.loc[frame.instrument.eq(symbol)]
        profile = summary['profiles'][symbol]
        observed = set(native.date)
        missing = sorted(d for d in calendar if native.date.iloc[0] <= d <= native.date.iloc[-1] and d not in observed)
        require(profile['candidate_rows'] == len(native) and profile['windows_per_policy'] == len(native)-119 and
                profile['first_candidate'] == native.date.iloc[0] and profile['first_reference_asof'] == native.date.iloc[119] and
                profile['last_reference_asof'] == native.date.iloc[-1] and
                profile['missing_dates_within_native_first_last'] == missing and
                profile['year_counts_per_policy'] == native.date.iloc[119:].str[:4].value_counts().sort_index().to_dict(),
                'Saved profile/window years differ')
        gap_windows = 0
        for policy in study.POLICIES:
            saved = selected.loc[selected.policy.eq(policy)].reset_index(drop=True)
            require(saved.asof_native_close.tolist() == native.date.iloc[119:].tolist() and
                    saved.window_first.tolist() == native.date.iloc[:-119].tolist(), 'Saved native120 boundaries differ')
            gaps = 0
            for number, row in enumerate(saved.itertuples(index=False)):
                operand = native.iloc[number:number+120]
                crossed = [d for d in missing if row.window_first <= d <= row.asof_native_close]
                require(json.loads(row.missing_open_dates) == crossed and row.missing_open_count == len(crossed) and
                        row.natural_day_rows == (dt.date.fromisoformat(row.asof_native_close)-
                                               dt.date.fromisoformat(row.window_first)).days+1, 'Saved gap/day shape differs')
                values = study.independent_returns(operand, policy)
                manual = -math.fsum(sorted(map(float, values))[:2])
                require(math.isfinite(row.ES) and math.isclose(row.ES, manual, rel_tol=1e-12, abs_tol=1e-15) and
                        math.isclose(row.independent_ES, manual, rel_tol=1e-12, abs_tol=1e-15) and
                        bool(row.inverse_ES_finite) == (math.isfinite(manual) and manual != 0), 'Saved ES value differs')
                largest = max(largest, abs(row.ES-manual)); cells += 1; gaps += bool(crossed)
            require(profile['zero_ES_by_policy'][policy] == int(saved.ES.eq(0).sum()) and
                    profile['negative_ES_by_policy'][policy] == int(saved.ES.lt(0).sum()), 'Saved ES boundary statistics differ')
            gap_windows = gaps
        require(profile['gap_crossing_windows_per_policy'] == gap_windows, 'Saved gap count differs')
        crossing += gap_windows
        comparison = selected.pivot(index='asof_native_close', columns='policy', values='ES')
        difference = comparison[study.POLICIES[0]]-comparison[study.POLICIES[1]]
        require(profile['ES_policy_differences'] == int(difference.ne(0).sum()) and
                profile['max_ES_policy_absolute_difference'] == float(difference.abs().max()), 'Saved policy comparison differs')
    require(cells == len(frame) and np.isfinite(largest), 'Saved ES readback incomplete')
    return {'status': 'ok', 'all_saved_ES_values_checked': cells, 'max_abs_difference': largest,
        'native_gap_crossing_windows_per_policy': crossing, 'return_check': 'Explicit natural-day scalar reference from native operands',
        'original_platform_input_proven': False, 'new_trading_backtests': 0}


def full_readback():
    study.validate()
    proof = read(study.DIRECTORY/'offline-verification.json')
    require(proof['status'] == 'match' and proof['differences'] == 0 and proof['network_blocked'] is True and
            proof['reference_rows_per_mode'] == 30924, 'Longest replay incomplete')
    for mode in ('study', 'replay'):
        for name, digest in proof['output_sha256'].items():
            require(file_sha(study.DIRECTORY/mode/name) == digest, 'Longest saved bytes changed')
    frame = pd.read_parquet(study.DIRECTORY/'study/ES-reference.parquet')
    base = read(study.inputs.DIRECTORY/'prepared.json')
    frames = {Path(name).stem: pd.read_parquet(name) for name in base['input_sha256']}
    summary = read(study.DIRECTORY/'study/summary.json')
    return readback(frame, summary, frames, read(study.DIRECTORY/'open-calendar.json'))


def tests(cwd, folder, env=None):
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '--disable-warnings', *TESTS,
               '--junitxml='+str(folder/'tests.xml')]
    process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    (folder/'stdout.log').write_text(process.stdout); (folder/'stderr.log').write_text(process.stderr)
    save(folder/'command.json', {'command': command, 'cwd': str(cwd), 'returncode': process.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Reference tests failed; logs retained')
    cases = list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
    require(len(cases) == 50 and all(len(case) == 0 for case in cases), 'Reference test/skip/failure scope differs')
    return len(cases)


def run():
    require(not DIRECTORY.exists(), 'Immutable reference validation exists')
    result = full_readback(); DIRECTORY.mkdir(parents=True)
    local = DIRECTORY/'local'; local.mkdir(); local_passed = tests(Path.cwd(), local.resolve())
    isolated = Path(tempfile.mkdtemp(prefix='observe-batch101-reference-check-'))
    paths = [path for name in ('src', 'scripts') for path in Path(name).rglob('*.py')]
    paths += [*(Path(name) for name in TESTS), Path('pyproject.toml'), Path('uv.lock')]
    paths += [Path('repo/量化策略源代码')/name for name in study.original.SOURCES]
    paths += [Path('docs/handoff/batch101-y2024a-072-source-review.yaml'), Path('docs/handoff/batch101-y2024b-036-source-review.yaml')]
    paths += [study.original.PREREAD/f'{number}.source' for number in (0, 1)]
    prepared = read(study.DIRECTORY/'prepared.json')
    paths += [Path(name) for name in prepared['evidence_sha256']]
    paths += [path for path in study.DIRECTORY.rglob('*') if path.is_file()]
    copied = {}
    for path in sorted(set(paths)):
        target = isolated/path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); copied[str(path)] = file_sha(target)
    env = os.environ.copy(); env['PYTHONPATH'] = str(isolated/'src')+os.pathsep+str(isolated)
    env['PYTHONDONTWRITEBYTECODE'] = '1'; folder = DIRECTORY/'isolated'; folder.mkdir()
    code = ('import json;import scripts.reference_strategy_batch101 as s;import scripts.verify_strategy_batch101_reference as v;'
            'import scripts.research_strategy_batch101 as o;import scripts.prepare_strategy_batch101 as p;'
            'import observe.runs as r;import observe.strategy_catalog as c;'
            'print(json.dumps({m.__name__:m.__file__ for m in (s,v,o,p,r,c)}))')
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=isolated, env=env, capture_output=True, text=True, timeout=30)
    require(process.returncode == 0, 'Copied imports failed: '+process.stderr)
    actual = json.loads(process.stdout)
    require(len(actual) == 6 and all(Path(name).is_relative_to(isolated) for name in actual.values()), 'Editable import leaked')
    isolated_passed = tests(isolated, folder.resolve(), env)
    process = subprocess.run([sys.executable, '-B', '-m', 'scripts.verify_strategy_batch101_reference', 'readback'],
                             cwd=isolated, env=env, capture_output=True, text=True, timeout=180)
    (folder/'readback.log').write_text(process.stdout+process.stderr)
    require(process.returncode == 0 and json.loads(process.stdout) == result, 'Copied full ES readback differs')
    save(folder/'isolation.json', {'cwd': str(isolated), 'copied_sha256': copied, 'actual_imports': actual,
        'tests_passed': isolated_passed, 'skips': 0, 'copied_full_readback': result})
    commands = []
    for name, command in (
        ('ruff', [str(Path(sys.executable).parent/'ruff'), 'check', 'src', 'tests',
            'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py',
            *study.FILES, 'scripts/verify_strategy_batch101_reference.py', 'scripts/accept_strategy_batch101.py']), ('diff', ['git', 'diff', '--check'])):
        process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (DIRECTORY/(name+'.log')).write_text(process.stdout+process.stderr)
        commands.append({'command': command, 'returncode': process.returncode, 'log_sha256': file_sha(DIRECTORY/(name+'.log'))})
        require(process.returncode == 0, 'Required check failed: '+name)
    save(DIRECTORY/'checked.json', {'status': 'ok_reference_phase', 'local_passed': local_passed,
        'isolated_passed': isolated_passed, 'copied_files': len(copied), 'skips': 0, 'full_readback': result,
        'implementation_sha256': {name: file_sha(name) for name in (*study.FILES, *TESTS, 'scripts/verify_strategy_batch101_reference.py', 'scripts/accept_strategy_batch101.py')},
        'evidence_sha256': {str(path): file_sha(path) for path in sorted(DIRECTORY.rglob('*')) if path.is_file()},
        'commands': commands, 'new_trading_backtests': 0, 'formal_source_research_acceptance': False})
    return {'status': 'ok_reference_phase', 'local': local_passed, 'isolated': isolated_passed,
        'saved_ES_readback_per_workspace': result['all_saved_ES_values_checked']}


if __name__ == '__main__':
    print(json.dumps(full_readback() if sys.argv[1:] == ['readback'] else run(), ensure_ascii=False))
