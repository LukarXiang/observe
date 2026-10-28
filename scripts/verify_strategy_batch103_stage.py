"""Read every saved native price reference and run bound local/copied tests; stage only."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch103 as reference
from scripts import research_strategy_batch103 as study
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch103/20261011-price-stage-validation')


def saved_readback():
    """Global offsets independently verify saved values, flags, gaps and complete scope."""
    reference.validate(); root = reference.DIRECTORY
    offline = read(root/'offline-verification.json')
    require(offline['status'] == 'ok' and offline['differences'] == [] and offline['fresh_network_blocked_processes'] == 2,
            'Actual two-worker evidence differs')
    for name, sha in offline['outputs_sha256'].items(): require(file_sha(name) == sha, 'Saved output changed')
    inputs = read(root/'dependencies.json'); calendar = set(inputs['calendar'])
    frames = {s: pd.read_parquet(root/'inputs'/(s+'.parquet')) for s in inputs['candidate_bindings']}
    counts = {}; cells = 0; gaps = {}; disagreements = 0
    for mode, pool, count in (('real', study.REAL_POOL, 13), ('momentum', study.MOMENTUM_POOL, 30)):
        frame = pd.read_parquet(root/'compute'/(mode+'-price-reference.parquet'))
        expected = {(s, d) for s in pool if s in frames for d in frames[s].date.iloc[count:]}
        require(len(frame) == len(expected) and not frame.duplicated(['instrument', 'asof_native_close']).any() and
            set(zip(frame.instrument, frame.asof_native_close)) == expected, 'Saved complete window scope differs')
        arrays = {s: f.close.to_numpy(float) for s, f in frames.items()}
        positions = {s: dict(zip(f.date, range(len(f)))) for s, f in frames.items()}
        missing = {s: sorted(d for d in calendar-set(f.date) if f.date.iloc[0] <= d <= f.date.iloc[-1]) for s, f in frames.items()}
        for row in frame.itertuples(index=False):
            symbol = row.instrument; values = arrays[symbol]; end = positions[symbol][row.asof_native_close]
            first = frames[symbol].date.iloc[end-count]; current = values[end]
            require(row.window_first == first and row.history_rows == count and row.current_is_daily_close_proxy,
                    'Saved date/operand semantics changed')
            gap = sum(first <= d <= row.asof_native_close for d in missing[symbol])
            require(row.missing_open_count == gap, 'Saved gap count differs')
            if mode == 'real':
                old = values[end-13:end]; cp = (current/values[end-13]-1)*100
                mean = math.fsum(old)/13; deviation = (current/mean-1)*100
                original_mean = old.mean(); original_deviation = (current/original_mean-1)*100
                checks = [(row.cp_increase, cp), (row.MA13, original_mean), (row.deviation, original_deviation),
                    (row.independent_cp, cp), (row.independent_MA13, mean), (row.independent_deviation, deviation)]
                require(row.price_eligible_reference == (cp >= .1 and original_deviation >= 0) and
                    row.independent_eligible == (cp >= .1 and deviation >= 0), 'Saved eligibility flags differ')
                disagreements += row.price_eligible_reference != row.independent_eligible
            else:
                old = math.fsum(values[end-23:end-20]); recent = math.fsum(values[end-2:end+1]); base = values[end-21]
                score = (recent/old-1)*100
                checks = [(row.old_mean, old/3), (row.score, score), (row.base_price, base),
                    (row.independent_old_mean, old/3), (row.independent_score, score), (row.independent_base_price, base)]
                require(row.entry_reference == (current > base*1.001) and row.exit_reference == (current < base*.999),
                        'Saved entry/exit flags differ')
            require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10) for a, b in checks), 'Saved independent arithmetic differs')
        counts[mode] = len(frame); cells += frame.size; gaps[mode] = int(frame.missing_open_count.gt(0).sum())
    doc = read(root/'compute/diagnostics.json')
    require(doc == study.diagnostics() and doc['case_count'] == 45 and doc['new_trading_backtests'] == 0,
            'Actual diagnostic readback differs')
    summary = read(root/'compute/summary.json')
    require(summary['real14_row_price_windows'] == counts['real'] and summary['momentum31_row_price_windows'] == counts['momentum'] and
        summary['real_gap_crossing_windows'] == gaps['real'] and summary['momentum_gap_crossing_windows'] == gaps['momentum'] and
        summary['real_price_eligibility_numeric_disagreements'] == disagreements, 'Saved summary totals differ')
    for symbol, profile in summary['candidate_profiles'].items():
        require(profile['rows'] == len(frames[symbol]) and profile['first'] == frames[symbol].date.iloc[0] and
            profile['last'] == frames[symbol].date.iloc[-1], 'Saved candidate range differs')
        for mode, by_year in profile['year_windows'].items():
            count = 13 if mode == 'real' else 30
            require(by_year == frames[symbol].date.iloc[count:].str[:4].value_counts().sort_index().to_dict(), 'Saved per-year scope differs')
    return {'windows': counts, 'saved_cells_checked': cells, 'gap_crossing_windows': gaps,
            'independent_eligibility_disagreements': disagreements, 'diagnostics': 45,
            'full_original_pool_signals': 0, 'new_trading_backtests': 0}


def tests(output):
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *reference.TESTS,
                            '--junitxml='+str(output/'tests.xml')], capture_output=True, text=True)
    (output/'tests.log').write_text(result.stdout+result.stderr)
    save(output/'test-command.json', {'returncode': result.returncode, 'log_sha256': file_sha(output/'tests.log')})
    require(result.returncode == 0, 'Tests failed; actual failure retained')
    suites = ET.parse(output/'tests.xml').getroot().findall('testsuite')
    require(suites and sum(int(s.get('failures', 0))+int(s.get('errors', 0))+int(s.get('skipped', 0)) for s in suites) == 0,
            'JUnit incomplete')
    count = sum(int(s.get('tests', 0)) for s in suites); require(count == 26, 'Expected test scope differs')
    return count


def worker():
    root = Path.cwd().resolve()
    imports = {}
    for name in ('scripts.reference_strategy_batch103', 'scripts.research_strategy_batch103', 'observe.runs', 'observe.data.store'):
        module = importlib.import_module(name); path = Path(module.__file__).resolve()
        require(path.is_relative_to(root), 'Copied worker imports original machine code')
        imports[name] = {'file': path.relative_to(root).as_posix(), 'sha256': file_sha(path)}
    dest = DIRECTORY/'worker'; dest.mkdir(parents=True, exist_ok=False)
    with network_blocked():
        readback = saved_readback(); count = tests(dest)
    save(dest/'result.json', {'readback': readback, 'tests': count, 'actual_imports': imports})
    return {'tests': count, 'readback': readback}


def verify():
    require(not DIRECTORY.exists(), 'Immutable verification directory exists'); reference.validate()
    DIRECTORY.mkdir(parents=True)
    with network_blocked(): readback = saved_readback(); local_count = tests(DIRECTORY)
    doc = read(reference.DIRECTORY/'prepared.json')
    paths = set(doc['evidence_sha256']) | set(doc['implementation_sha256']) | set(reference.TESTS)
    paths |= {str(reference.DIRECTORY/'implementation'/name) for name in doc['implementation_sha256']}
    paths |= {str(p) for p in (reference.DIRECTORY/'compute').iterdir() if p.is_file()}
    paths |= {str(reference.DIRECTORY/'offline-verification.json'), str(reference.DIRECTORY/'prepared.json'),
              'scripts/verify_strategy_batch103_stage.py'}
    isolated = DIRECTORY/'isolated'; copied = {}
    for name in sorted(paths):
        path = Path(name); require(not path.is_absolute() and '..' not in path.parts, 'Unconfined copy path')
        dest = isolated/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
        require(file_sha(dest) == file_sha(path), 'Copied SHA differs'); copied[name] = file_sha(path)
    save(DIRECTORY/'copied-files.json', copied)
    env = os.environ.copy(); env['PYTHONPATH'] = os.pathsep.join((str((isolated/'src').resolve()), str(isolated.resolve())))
    env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONHASHSEED'] = '103'
    result = subprocess.run([sys.executable, '-B', '-m', 'scripts.verify_strategy_batch103_stage', 'worker'],
        cwd=isolated, env=env, capture_output=True, text=True)
    (DIRECTORY/'isolated.log').write_text(result.stdout+result.stderr)
    save(DIRECTORY/'isolated-command.json', {'returncode': result.returncode, 'log_sha256': file_sha(DIRECTORY/'isolated.log')})
    require(result.returncode == 0, 'Copied worker failed; retained')
    copied_result = read(isolated/DIRECTORY/'worker/result.json')
    require(copied_result['readback'] == readback and copied_result['tests'] == local_count, 'Copied scope differs')
    require(all(file_sha(name) == sha for name, sha in copied.items()), 'Original files changed during verification')
    save(DIRECTORY/'isolated-result.json', copied_result)
    checked = {'status': 'ok_stage_only', 'local_tests': local_count, 'isolated_tests': copied_result['tests'],
        'copied_files': len(copied), 'copied_manifest_sha256': file_sha(DIRECTORY/'copied-files.json'), 'readback': readback,
        'formal_source_research_acceptance': False, 'formal_trading_acceptance': False, 'new_trading_backtests': 0}
    save(DIRECTORY/'checked.json', checked); return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('verify', 'worker'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](), ensure_ascii=False))
