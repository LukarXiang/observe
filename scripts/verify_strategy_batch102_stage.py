"""Read all saved Batch102 reference values locally and in a copied isolated tree."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import numpy as np

from observe.runs import file_sha
from scripts import reference_strategy_batch102 as reference
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch102/20261011-reference-stage-validation')
TESTS = ('tests/unit/test_batch102_research.py', 'tests/unit/test_batch102_reference.py')


def saved_etf_expected(frames, calendar):
    """Direct saved-column reconstruction, without compiling original callbacks per row."""
    rows = []; profiles = {}; days = set(calendar)
    for symbol, frame in sorted(frames.items()):
        observed = set(frame.date)
        missing = sorted(d for d in days if frame.date.iloc[0] <= d <= frame.date.iloc[-1] and d not in observed)
        prices = frame.close.to_numpy(float)
        for end in range(19, len(frame)):
            values = prices[end-19:end+1]; inc = float(values[-1]/values[0]-1)
            long = float(np.mean(values)); short = float(np.mean(values[-5:]))
            independent_long = math.fsum(values)/20; independent_short = math.fsum(values[-5:])/5
            crossed = sum(frame.date.iloc[end-19] <= d <= frame.date.iloc[end] for d in missing)
            rows.append({'instrument': symbol, 'window_first': frame.date.iloc[end-19], 'asof_native_close': frame.date.iloc[end],
                'inc19_intervals': inc, 'MA20': long, 'MA5': short,
                'independent_MA20': independent_long, 'independent_MA5': independent_short,
                'eligible_metric_reference': bool(short > long and .05 < inc < .2),
                'independent_eligible': bool(independent_short > independent_long and .05 < inc < .2),
                'missing_open_count': crossed, 'original_attribute_history_proven': False})
        profiles[symbol] = {'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
            'windows': len(frame)-19, 'missing_open_dates': missing,
            'year_windows': frame.date.iloc[19:].str[:4].value_counts().sort_index().to_dict()}
    return pd.DataFrame(rows), profiles


def read_saved():
    with network_blocked():
        reference.validate(); proof = read(reference.DIRECTORY/'offline-verification.json')
        require(proof['status'] == 'ok' and proof['differences'] == [], 'Frozen offline reference failed')
        for name, sha in proof['outputs_sha256'].items(): require(file_sha(name) == sha, 'Saved output changed')
        inputs = read(reference.DIRECTORY/'dependencies.json')
        pmi = reference.pmi_reference(pd.read_parquet(reference.DIRECTORY/'inputs/PMI.parquet'))
        index, annual = reference.index_reference(pd.read_parquet(reference.DIRECTORY/'inputs/index.parquet'))
        frames = {s: pd.read_parquet(reference.DIRECTORY/'inputs'/(s+'.parquet')) for s in inputs['ETF_candidates_in_selected_registries']}
        etf, profiles = saved_etf_expected(frames, inputs['calendar']); checked = {}
        for name, frame in (('PMI', pmi), ('index', index), ('annual', annual), ('ETF', etf)):
            actual = pd.read_parquet(reference.DIRECTORY/'compute'/(name+'-reference.parquet'))
            pd.testing.assert_frame_equal(actual, frame, check_exact=True)
            checked[name] = {'rows': len(frame), 'columns': len(frame.columns), 'all_values_checked': True}
        require(read(reference.DIRECTORY/'compute/diagnostics.json') == reference.original.diagnostics(), 'Saved original callbacks differ')
        require(read(reference.DIRECTORY/'compute/summary.json')['ETF_profiles'] == profiles, 'Saved profile/year counts differ')
        return {'status': 'ok', 'saved_tables': checked, 'all_saved_cells_checked': True,
            'network_blocked': True, 'new_trading_backtests': 0}


def worker():
    import pytest
    from observe.ledger import book
    import observe.runs as runs
    root = Path.cwd().resolve()
    modules = (reference, reference.original, book, runs)
    imports = {m.__name__: str(Path(inspect.getfile(m)).resolve()) for m in modules}
    require(all(Path(p).is_relative_to(root) for p in imports.values()), 'Isolated imports escaped copied tree')
    save(DIRECTORY/'actual-imports.json', imports)
    with network_blocked():
        result = pytest.main(['-q', *TESTS, '--junitxml='+str(DIRECTORY/'isolated-tests.xml')])
        require(result == 0, 'Copied actual tests failed')
        save(DIRECTORY/'saved-readback.json', read_saved())
    return {'status': 'ok', 'actual_imports': len(imports)}


def verify():
    require(not DIRECTORY.exists(), 'Immutable stage validation exists'); DIRECTORY.mkdir(parents=True)
    save(DIRECTORY/'implementation.json', {__file__: file_sha(__file__)})
    save(DIRECTORY/'saved-readback.json', read_saved())
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS,
        '--junitxml='+str(DIRECTORY/'local-tests.xml')], capture_output=True, text=True)
    (DIRECTORY/'local-tests.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'local-test-command.json', {'returncode': result.returncode,
        'log_sha256': file_sha(DIRECTORY/'local-tests.log')}); require(result.returncode == 0, 'Local tests failed')
    isolated = DIRECTORY/'isolated'; isolated.mkdir()
    prepared = read(reference.DIRECTORY/'prepared.json')
    names = {str(p) for folder in ('src', 'scripts') for p in Path(folder).rglob('*.py') if '__pycache__' not in p.parts}
    names.update(TESTS); names.update(prepared['evidence_sha256']); names.update(read(reference.DIRECTORY/'offline-verification.json')['outputs_sha256'])
    names.update(str(reference.DIRECTORY/p) for p in ('prepared.json', 'offline-verification.json'))
    names.update(str(reference.DIRECTORY/'implementation'/p) for p in prepared['implementation_sha256'])
    copied = {}
    for name in sorted(names):
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'Unexpected copy path')
        target = isolated/path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        sha = file_sha(path); require(file_sha(target) == sha, 'Copied byte mismatch'); copied[name] = sha
    dest = isolated/DIRECTORY; dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join((str((isolated/'src').resolve()), str(isolated.resolve())))
    result = subprocess.run([sys.executable, '-m', 'scripts.verify_strategy_batch102_stage', 'worker'],
        cwd=isolated, env=env, capture_output=True, text=True)
    (DIRECTORY/'isolated-tests.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'isolated-command.json', {'returncode': result.returncode,
        'log_sha256': file_sha(DIRECTORY/'isolated-tests.log')}); require(result.returncode == 0, 'Isolated tests/readback failed')
    for name in ('actual-imports.json', 'isolated-tests.xml', 'saved-readback.json'):
        shutil.copyfile(dest/name, DIRECTORY/('isolated-'+name))
    require(read(DIRECTORY/'saved-readback.json') == read(DIRECTORY/'isolated-saved-readback.json'), 'Readbacks differ')
    import xml.etree.ElementTree as ET
    counts = {}
    for tag, name in (('local', 'local-tests.xml'), ('isolated', 'isolated-isolated-tests.xml')):
        suites = ET.parse(DIRECTORY/name).getroot().findall('testsuite')
        require(suites and sum(int(s.get('failures', 0))+int(s.get('errors', 0))+int(s.get('skipped', 0)) for s in suites) == 0, 'JUnit tests incomplete')
        counts[tag] = sum(int(s.get('tests', 0)) for s in suites)
    doc = {'status': 'ok_stage_only', 'formal_source_research_acceptance': False, 'formal_trading_acceptance': False,
        'tests': counts, 'copied_file_sha256': copied, 'actual_imports': read(DIRECTORY/'isolated-actual-imports.json'),
        'readback': read(DIRECTORY/'saved-readback.json'), 'new_trading_backtests': 0, 'original_strategy_complete': False}
    save(DIRECTORY/'checked.json', doc); return {'status': doc['status'], 'tests': counts, 'copied_files': len(copied)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('verify', 'worker'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](), ensure_ascii=False))
