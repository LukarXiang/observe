"""Full saved price/heat readback and new local/copied gates before source-only acceptance."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import accept_strategy_batch103 as acceptance
from scripts import reference_strategy_batch103_emotion as volume
from scripts import verify_strategy_batch103_stage as price_check
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = acceptance.CHECK
TESTS = (*acceptance.price.TESTS, volume.TEST, 'tests/unit/test_batch103_acceptance.py')
FILES = ('scripts/accept_strategy_batch103.py', 'scripts/verify_strategy_batch103_formal.py',
         'scripts/verify_strategy_batch103_stage.py',
         'scripts/accept_strategy_batch100.py', 'scripts/recover_strategy_batch100.py',
         *TESTS, *acceptance.SPECS)


def close(a, b):
    return (pd.isna(a) and (b is None or pd.isna(b))) or (b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10))


def volume_readback():
    """Audit saved scope via global offsets and independently classify the last30 MA7 ratios."""
    volume.validate(); root = volume.DIRECTORY
    acceptance.offline_checks(root, 3)
    frame = pd.read_parquet(root/'compute/emotion-reference.parquet'); inputs = read(root/'dependencies.json')
    natives = {s: pd.read_parquet(root/'inputs'/(s+'.parquet')) for s in inputs['bindings']}
    expected = {(s, d) for s, f in natives.items() for d in f.date.iloc[99:]}
    require(len(frame) == len(expected) == 48437 and not frame.duplicated(['instrument', 'asof_history_last']).any() and
        set(zip(frame.instrument, frame.asof_history_last)) == expected, 'Saved heat windows differ')
    calendar = set(inputs['calendar']); positions = {s: dict(zip(f.date, range(len(f)))) for s, f in natives.items()}
    arrays = {s: f.volume.to_numpy() for s, f in natives.items()}
    floating_arrays = {s: a.astype('float64') for s, a in arrays.items()}
    missing = {s: sorted(d for d in calendar-set(f.date) if f.date.iloc[0] <= d <= f.date.iloc[-1]) for s, f in natives.items()}
    ns = volume.source.literal(0); holder = {}
    ns.update(attribute_history=lambda *a, **k: pd.DataFrame({'volume': holder['operand']}))
    ns['g'].lag0 = 7; ns['g'].lag = 5
    for row in frame.itertuples(index=False):
        values = arrays[row.instrument]; end = positions[row.instrument][row.asof_history_last]
        first = natives[row.instrument].date.iloc[end-99]
        require(row.window_first == first and row.native_volume_dtype == str(values.dtype) and
            row.calendar_complete_for_window == (first >= inputs['calendar'][0]) and
            not row.platform_history_dtype_proven and row.missing_open_count ==
            sum(first <= d <= row.asof_history_last for d in missing[row.instrument]), 'Saved volume identity/gap differs')
        operand = values[end-99:end+1]; ns['g'].target_market = row.instrument
        actuals = []
        for history in (operand, operand.astype('float64')):
            holder['operand'] = history; ns['g'].emotion_rate = 0.
            result = ns['EmotionMonitor'](SimpleNamespace()); rate = float(ns['g'].emotion_rate)
            actuals.append((volume.result_label(result), rate if math.isfinite(rate) else None))
        require(row.native_result == actuals[0][0] and close(row.native_rate, actuals[0][1]) and
            row.float_reference_result == actuals[1][0] and close(row.float_reference_rate, actuals[1][1]),
            'Saved original callback/dtype results differ')
        floating = floating_arrays[row.instrument]; ratios = []
        for point in range(end-29, end+1):
            average = math.fsum(floating[point-6:point+1])/7
            ratios.append(floating[point]/average-1 if average else np.nan)
        positive = ratios[-1] >= 0
        crossing = next((i for i, r in enumerate(reversed(ratios)) if (r < 0 if positive else r >= 0)), None)
        result = None if crossing is None else ((1 if crossing >= 3 else 0) if positive else (-1 if crossing >= 5 else 0))
        rate = float(round(ratios[-1]*100, 2)) if math.isfinite(ratios[-1]) else None
        label = volume.result_label(result)
        require(row.independent_result == label and close(row.independent_rate, rate) and
            close(row.independent_last_ratio, ratios[-1]) and close(row.independent_crossing_distance, crossing),
            'Saved independent volume arithmetic differs')
        require(row.float_state_disagreement == (row.float_reference_result != label) and
            row.native_vs_float_state_disagreement == (row.native_result != row.float_reference_result), 'Saved disagreement flags differ')
    summary = read(root/'compute/summary.json')
    for key, actual in {'windows': len(frame), 'native_results': frame.native_result.value_counts().sort_index().to_dict(),
        'float_reference_results': frame.float_reference_result.value_counts().sort_index().to_dict(),
        'float_state_disagreements': int(frame.float_state_disagreement.sum()),
        'native_vs_float_state_disagreements': int(frame.native_vs_float_state_disagreement.sum()),
        'float_rate_disagreements': int(frame.float_reference_rate.fillna(1e100).ne(frame.independent_rate.fillna(1e100)).sum()),
        'gap_crossing_windows': int(frame.missing_open_count.gt(0).sum()),
        'windows_before_complete_calendar_coverage': int((~frame.calendar_complete_for_window).sum())}.items():
        require(summary[key] == actual, 'Saved volume aggregate differs: '+key)
    for symbol, profile in summary['profiles'].items():
        native = natives[symbol]
        require(profile['rows'] == len(native) and profile['windows'] == len(native)-99 and
            profile['first'] == native.date.iloc[0] and profile['last'] == native.date.iloc[-1] and
            profile['dtype'] == str(native.volume.dtype) and profile['missing_known_open_dates'] == missing[symbol] and
            profile['year_windows'] == native.date.iloc[99:].str[:4].value_counts().sort_index().to_dict(), 'Saved volume profile/year differs')
    require(read(root/'compute/dtype-diagnostics.json') == volume.dtype_diagnostics(), 'Saved dtype diagnostic differs')
    return {'windows': len(frame), 'saved_cells_checked': frame.size,
        'native_vs_float_state_disagreements': summary['native_vs_float_state_disagreements'],
        'float_state_disagreements': summary['float_state_disagreements'],
        'float_rate_disagreements': summary['float_rate_disagreements'],
        'gap_crossing_windows': summary['gap_crossing_windows'], 'year_profiles_checked': len(natives),
        'original_dynamic_pool_signals': 0, 'new_trading_backtests': 0}


def saved_readback(): return {'price': price_check.saved_readback(), 'volume': volume_readback()}


def count_tests(path):
    suites = ET.parse(path).getroot().findall('testsuite')
    require(suites and sum(int(s.get('failures', 0))+int(s.get('errors', 0))+int(s.get('skipped', 0)) for s in suites) == 0, 'Tests failed or skipped')
    count = sum(int(s.get('tests', 0)) for s in suites); require(count == 48, 'Formal test scope differs'); return count


def worker():
    import pytest
    from observe.ledger import book
    modules = (acceptance, acceptance.price, volume, volume.source, book)
    root = Path.cwd().resolve(); imports = {m.__name__: str(Path(inspect.getfile(m)).resolve()) for m in modules}
    require(all(Path(name).is_relative_to(root) for name in imports.values()), 'Actual imports escaped copy')
    with network_blocked():
        result = saved_readback(); rc = pytest.main(['-q', *TESTS, '--junitxml='+str(DIRECTORY/'tests.xml')])
        require(rc == 0, 'Copied formal tests failed')
    save(DIRECTORY/'saved-readback.json', result); save(DIRECTORY/'actual-imports.json', imports)
    return {'status': 'ok', 'actual_imports': len(imports)}


def verify():
    require(not DIRECTORY.exists(), 'Immutable formal validation exists')
    acceptance.price.validate(); volume.validate(); DIRECTORY.mkdir(parents=True)
    save(DIRECTORY/'implementation.json', {name: file_sha(name) for name in FILES})
    for name in FILES:
        dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
    with network_blocked(): readback = saved_readback()
    save(DIRECTORY/'saved-readback.json', readback)
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(DIRECTORY/'local-tests.xml')],
        capture_output=True, text=True)
    (DIRECTORY/'local.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'local-command.json', {'returncode': result.returncode, 'log_sha256': file_sha(DIRECTORY/'local.log')})
    require(result.returncode == 0, 'Local formal tests failed')
    names = set(FILES)
    for root in (acceptance.price.DIRECTORY, volume.DIRECTORY):
        prepared = read(root/'prepared.json'); proof = read(root/'offline-verification.json')
        names |= set(prepared['evidence_sha256']) | set(prepared['implementation_sha256']) | set(proof['outputs_sha256'])
        names |= {str(root/'implementation'/name) for name in prepared['implementation_sha256']}
        names |= {str(root/'prepared.json'), str(root/'offline-verification.json')}
        for cmd in proof['commands']: names.add(cmd['log'])
        names |= {str(p) for p in (root/'replay').iterdir() if p.is_file()}
    # Newly imported helpers can be absent from both older frozen graphs.
    names |= set(acceptance.price.implementation_files())
    isolated = DIRECTORY/'isolated'; copied = {}
    for name in sorted(names):
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'Unsafe copy path')
        dest = isolated/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        sha = file_sha(path); require(file_sha(dest) == sha, 'Copied SHA differs'); copied[name] = sha
    (isolated/DIRECTORY).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join((str((isolated/'src').resolve()), str(isolated.resolve())))
    env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONHASHSEED'] = '103'
    result = subprocess.run([sys.executable, '-B', '-m', 'scripts.verify_strategy_batch103_formal', 'worker'],
        cwd=isolated, env=env, capture_output=True, text=True)
    (DIRECTORY/'isolated.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'isolated-command.json', {'returncode': result.returncode, 'log_sha256': file_sha(DIRECTORY/'isolated.log')})
    require(result.returncode == 0, 'Isolated formal checks failed; preserved')
    for name in ('saved-readback.json', 'actual-imports.json', 'tests.xml'):
        shutil.copyfile(isolated/DIRECTORY/name, DIRECTORY/('isolated-'+name))
    require(readback == read(DIRECTORY/'isolated-saved-readback.json'), 'Saved local/copied readbacks differ')
    tests = {'local': count_tests(DIRECTORY/'local-tests.xml'), 'isolated': count_tests(DIRECTORY/'isolated-tests.xml')}
    require(all(file_sha(name) == sha == file_sha(isolated/name) for name, sha in copied.items()), 'Files changed during verification')
    evidence = {name: file_sha(name) for name in FILES}
    evidence.update({str(p): file_sha(p) for p in DIRECTORY.iterdir() if p.is_file()})
    save(DIRECTORY/'checked.json', {'status': 'ok_formal_ready', 'tests': tests, 'saved_readback': readback,
        'copied_file_sha256': copied, 'actual_imports': read(DIRECTORY/'isolated-actual-imports.json'),
        'evidence_sha256': evidence, 'formal_trading_acceptance': False, 'new_trading_backtests': 0,
        'independent_agent_review': False})
    return {'status': 'ok_formal_ready', 'tests': tests, 'copied_files': len(copied), 'saved_readback': readback}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('verify', 'worker'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](), ensure_ascii=False))
