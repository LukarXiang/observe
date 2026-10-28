"""Isolated cached-fork Kalman reference, rooted operands and literal RSRS capture.

This is a numerical research stage, never a trading or formal acceptance gate.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts import review_strategy_batch94 as previous
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch59 import index_operand
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch94/20261010-kalman-fork-reference')
CACHE_READY = Path('docs/handoff/2026-10-10-batch94-cache-readiness.json')
CACHE_READY_SHA = 'c91c6f29368371b1aecbccf2fb40b282b60274fe435f75e0938b7f09573b0fad'
STAGE_RECEIPT = Path('docs/handoff/2026-10-10-batch94-numerical-stage.json')
STAGE_RECEIPT_SHA = '08516e45f755db0b1be5cfa5177309440d2545bcc80d65948886f57932b3498e'
DEPENDENCY_CACHES = tuple(Path('/home/zuozhe/.cache/uv/archive-v0')/name for name in (
    'y-KD1DmrKGAFR40o7AzLm', 'x1fmFc8cEQ4nLXpwsXF7y'))
FILES = ('scripts/reference_strategy_batch94.py', 'tests/unit/test_batch94_reference.py')


def historical_stage():
    require(file_sha(STAGE_RECEIPT) == STAGE_RECEIPT_SHA, 'Historical stage receipt changed')
    receipt = read(STAGE_RECEIPT)
    require(file_sha(receipt['stage_binding']) == receipt['stage_binding_sha256'], 'Historical binding changed')
    binding = read(receipt['stage_binding'])
    for name, sha in binding['evidence_sha256'].items():
        require(file_sha(previous.DIRECTORY/name) == sha, 'Historical evidence changed: '+name)
    for name, sha in binding['implementation_sha256'].items():
        require(file_sha(name) == sha, 'Historical stage code changed: '+name)
    require(file_sha(CACHE_READY) == CACHE_READY_SHA, 'Cached fork receipt changed')
    cached = read(CACHE_READY)
    for row in cached['files']:
        require(file_sha(row['copy']) == row['sha256'], 'Cached fork changed')
    require(file_sha(cached['distribution_metadata_copy']) == cached['distribution_metadata_sha256'],
        'Fork distribution metadata changed')
    return receipt, cached


def validate_index(root, directory=previous.DIRECTORY):
    """Compare the complete frozen frame AND partition set against the actual Store."""
    previous.canonical_root(root)
    frame, operand = index_operand(root)
    declared = read(directory/'index-binding.json')
    require(bool(operand['partitions']) and
        {k: v for k, v in declared.items() if k != 'frozen_sha256'} == operand,
        'Rooted index partition identity differs')
    require(file_sha(directory/'index-input.parquet') == declared['frozen_sha256'], 'Frozen index bytes differ')
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'index-input.parquet'), frame, check_exact=True)
    return operand


def prepare(root):
    previous.canonical_root(root)
    require(not DIRECTORY.exists(), 'Reference directory already exists')
    _, cached = historical_stage()
    operand = validate_index(root)
    require(previous.candidate_inputs() == read(previous.DIRECTORY/'candidate-inputs.json'), 'Candidate chain changed')
    cache_files = []
    base, duplicate = DEPENDENCY_CACHES
    names = sorted(p.relative_to(base).as_posix() for p in (base/'skbase').rglob('*.py'))
    require(names and names == sorted(p.relative_to(duplicate).as_posix()
        for p in (duplicate/'skbase').rglob('*.py')), 'Dependency cache inventory differs')
    names += ['scikit_base-0.12.5.dist-info/METADATA', 'scikit_base-0.12.5.dist-info/licenses/LICENSE']
    for name in names:
        require(file_sha(base/name) == file_sha(duplicate/name), 'Dependency cache bytes differ')
        cache_files.append({'source': str(base/name), 'duplicate_source': str(duplicate/name),
            'copy': 'dependency/'+name, 'sha256': file_sha(base/name)})
    with operation_lock(root, DATA_WRITER):
        checkpoint(root, DIRECTORY, 'Batch94 isolated cached-fork numerical reference starts; no trades')
        save(DIRECTORY/'rooted-index-binding.json', operand)
        for row in cache_files:
            target = DIRECTORY/row['copy']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['source'], target)
        for row in cached['files']:
            target = DIRECTORY/'fork'/row['module']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['copy'], target)
            cache_files.append({'source': row['copy'], 'copy': 'fork/'+row['module'], 'sha256': row['sha256']})
        shutil.copyfile(cached['distribution_metadata_copy'], DIRECTORY/'sktime.METADATA')
        cache_files.append({'source': cached['distribution_metadata_copy'], 'copy': 'sktime.METADATA',
            'sha256': cached['distribution_metadata_sha256']})
        save(DIRECTORY/'runtime-binding.json', {'files': cache_files, 'fork': 'sktime 0.38.5 unofficial pykalman fork',
            'dependency': 'scikit-base 0.12.5', 'installation_performed': False,
            'cache_duplicate_bytes_match': True, 'original_platform_library_equivalence_proven': False})
        hashes = {name: file_sha(name) for name in FILES}
        for name in FILES:
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        save(DIRECTORY/'implementation-binding.json', hashes)
    return {'status': 'prepared', 'runtime_files': len(cache_files), 'new_trading_backtests': 0}


def validate_runtime(directory):
    doc = read(directory/'runtime-binding.json')
    require(doc['fork'] == 'sktime 0.38.5 unofficial pykalman fork' and
        doc['dependency'] == 'scikit-base 0.12.5' and not doc['installation_performed'] and
        not doc['original_platform_library_equivalence_proven'], 'Runtime scope changed')
    for row in doc['files']:
        path = directory/row['copy']
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()) and
            file_sha(path) == row['sha256'], 'Frozen runtime differs')
    require(not list(directory.rglob('*.pyc')), 'Bytecode written into frozen runtime')
    return doc


def load_fork(directory):
    require(os.environ.get('OBSERVE_BATCH94_WORKER') == '1' and sys.dont_write_bytecode,
        'Cached runtime requires a fresh bytecode-disabled worker')
    require('skbase' not in sys.modules and '_observe_batch94_cached_kalman' not in sys.modules,
        'Cached runtime already loaded')
    validate_runtime(directory)
    sys.path.insert(0, str((directory/'dependency').resolve()))
    name = '_observe_batch94_cached_kalman'
    spec = importlib.util.spec_from_file_location(name, directory/'fork/__init__.py',
        submodule_search_locations=[str((directory/'fork').resolve())])
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    require(sys.modules['skbase'].__version__ == '0.12.5', 'Dependency version differs')
    runtime = {str((directory/r['copy']).resolve()): r['sha256'] for r in read(directory/'runtime-binding.json')['files']}
    loaded = []
    for key, value in sorted(sys.modules.items()):
        if key == 'skbase' or key.startswith('skbase.') or key == name or key.startswith(name+'.'):
            path = str(Path(value.__file__).resolve())
            require(path in runtime and file_sha(path) == runtime[path], 'Loaded module outside frozen runtime')
            loaded.append({'module': key, 'file': path, 'sha256': runtime[path]})
    return module.KalmanFilter, loaded


def precision_system(observations):
    """Gaussian posterior, prior y0/P0=.15, measurement R=.15, increment Q=.1."""
    y = [float(v) for v in observations]
    require(len(y) >= 2 and all(math.isfinite(v) for v in y), 'Posterior observations invalid')
    diagonal = [1/.15+1/.1] + [1/.15+2/.1]*(len(y)-2) + [1/.15+1/.1]
    diagonal[0] += 1/.15
    rhs = [v/.15 for v in y]; rhs[0] += y[0]/.15
    return diagonal, [-1/.1]*(len(y)-1), rhs


def tridiagonal_solve(diagonal, off_diagonal, rhs):
    d = list(diagonal); b = list(rhs); n = len(d)
    require(n == len(b) and len(off_diagonal) == n-1 and n >= 2, 'Tridiagonal shape differs')
    for i in range(1, n):
        multiplier = off_diagonal[i-1]/d[i-1]
        d[i] -= multiplier*off_diagonal[i-1]; b[i] -= multiplier*b[i-1]
    result = [0.]*n; result[-1] = b[-1]/d[-1]
    for i in range(n-2, -1, -1): result[i] = (b[i]-off_diagonal[i]*result[i+1])/d[i]
    return np.asarray(result)


def posterior(observations):
    return tridiagonal_solve(*precision_system(observations))


def posterior_variances(size):
    d, off, _ = precision_system(np.ones(size))
    return np.asarray([tridiagonal_solve(d, off, [float(i == j) for j in range(size)])[i]
        for i in range(size)])


def independent_slope(values):
    midpoint = (len(values)-1)/2
    return math.fsum((i-midpoint)*float(v) for i, v in enumerate(values))/math.fsum(
        (i-midpoint)**2 for i in range(len(values)))


def kalman_components(kalman):
    ns = previous.literal(previous.DIRECTORY, 1, {'np': np, 'KalmanFilter': kalman})
    fn = ns['kalman_filter']; expected_cov = posterior_variances(20); rows = []; profiles = []
    inputs = read(previous.DIRECTORY/'candidate-inputs.json')['funds']
    for symbol in previous.POOL:
        if symbol not in inputs: continue
        frame, _ = previous.native_frame(inputs[symbol]); closes = frame.close.to_numpy(float); start = len(rows)
        for end in range(20, len(frame)+1):
            window = closes[end-20:end]; observations = window/window[0]
            # Exact SHA-bound original helper with the cached fork. No ranking or date-index adapter.
            predicted = fn(observations, .15)
            model = kalman(initial_state_mean=observations[0], initial_state_covariance=.15,
                transition_matrices=1, observation_covariance=.15, transition_covariance=.1, transition_offsets=None)
            means, covariance = model.smooth(observations)
            require(predicted.shape == means.shape == (20, 1) and covariance.shape == (20, 1, 1), 'Fork output shape differs')
            np.testing.assert_array_equal(predicted, means)
            scalar_means = posterior(observations)
            np.testing.assert_allclose(means[:, 0], scalar_means, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(covariance[:, 0, 0], expected_cov, rtol=2e-12, atol=2e-12)
            score = np.polyfit(np.arange(20), predicted[-20:], 1)[0]
            scalar_score = independent_slope(scalar_means)
            require(score.shape == (1,) and math.isclose(float(score[0]), scalar_score, rel_tol=2e-10, abs_tol=2e-12),
                'Independent slope differs')
            rows.append({'symbol': symbol, 'window_first': frame.date.iloc[end-20],
                'last_input_date': frame.date.iloc[end-1], 'score': float(score[0]), 'independent_score': scalar_score,
                'maximum_state_error': float(np.max(np.abs(means[:, 0]-scalar_means))),
                'maximum_covariance_error': float(np.max(np.abs(covariance[:, 0, 0]-expected_cov))),
                'score_shape': '[1]', 'native_unadjusted_reference': True})
        profiles.append({'symbol': symbol, 'windows': len(rows)-start,
            'first_last_input': frame.date.iloc[19], 'last_last_input': frame.date.iloc[-1]})
        print(json.dumps({'completed_fund': symbol, 'windows': len(rows)-start}), flush=True)
    table = pd.DataFrame(rows)
    return table, {'windows': len(table), 'profiles': profiles, 'state_cells_compared': 20*len(table),
        'covariance_cells_compared': 20*len(table), 'maximum_state_error': float(table.maximum_state_error.max()),
        'maximum_covariance_error': float(table.maximum_covariance_error.max()),
        'maximum_score_error': float((table.score-table.independent_score).abs().max()),
        'original_helper_executed': True, 'full_original_pool_ranking_rows': 0, 'score_shape': [1],
        'missing_pool': ['159928.XSHE'], 'not_a_trading_backtest': True,
        'original_platform_library_equivalence_proven': False}


def literal_rsrs_capture():
    frame = pd.read_parquet(previous.DIRECTORY/'index-input.parquet'); pointer = [618]
    g = SimpleNamespace(N=18, M=600, ref_stock='000300.XSHG', score_threshold=.7)
    def history(stock, count, unit, fields):
        require(stock == g.ref_stock and unit == '1d' and count in (18, 618), 'Literal query differs')
        return frame.iloc[pointer[0]-count:pointer[0]].reset_index(drop=True)[fields]
    ns = previous.literal(previous.DIRECTORY, 1, {'np': np, 'g': g, 'attribute_history': history,
        'log': SimpleNamespace(info=lambda *args: None)})
    original = ns['get_ols']; actual = []
    def captured(x, y):
        result = original(x, y); actual.append(result); return result
    ns['get_ols'] = captured
    g.slope_series = ns['initial_slope_series']()[:-1]; actual.clear()
    prior = pd.read_parquet(previous.DIRECTORY/'rsrs-reference.parquet'); rows = []
    for end in range(618, len(frame)):
        pointer[0] = end; signal = ns['get_timing_signal'](None, 'ignored')
        require(len(actual) == 1, 'Literal callback helper count differs')
        intercept, slope, r2 = actual.pop(); z = ns['get_zscore'](g.slope_series[-600:]); score = z*r2
        old = prior.iloc[end-618]
        require(signal == old.signal, 'Actual callback classification differs from old stage')
        previous.close_enough(score, old.scalar_score, 'Actual Series callback independent score')
        rows.append({'decision_date': frame.date.iloc[end], 'intercept': intercept, 'slope': slope,
            'r2': r2, 'zscore': z, 'score': score, 'signal': signal, 'prior_score': old.score,
            'independent_score': old.scalar_score, 'actual_Series_helper': True})
    table = pd.DataFrame(rows)
    return table, {'rows': len(table), 'classification_differences': 0,
        'maximum_prior_score_difference': float((table.score-table.prior_score).abs().max()),
        'maximum_independent_score_difference': float((table.score-table.independent_score).abs().max()),
        'not_a_trading_backtest': True}


def worker(mode):
    historical_stage(); validate_runtime(DIRECTORY)
    hashes = read(DIRECTORY/'implementation-binding.json')
    require(all(file_sha(name) == sha for name, sha in hashes.items()), 'Reference implementation changed')
    with network_disabled():
        kalman, modules = load_fork(DIRECTORY)
        if mode == 'smoke':
            ns = previous.literal(previous.DIRECTORY, 1, {'np': np, 'KalmanFilter': kalman})
            observed = np.linspace(1., 2., 20); means = ns['kalman_filter'](observed, .15)
            np.testing.assert_allclose(means[:, 0], posterior(observed), rtol=2e-12, atol=2e-12)
            model = kalman(initial_state_mean=observed[0], initial_state_covariance=.15,
                transition_matrices=1, observation_covariance=.15, transition_covariance=.1)
            filtered = model.filter(observed)[0]
            require(abs(float(means[0, 0]-filtered[0, 0])) > .01, 'Smooth/filter distinction missing')
            return {'status': 'ok', 'loaded_modules': modules, 'shape': list(means.shape),
                'smooth_filter_first_mean_difference': float(means[0, 0]-filtered[0, 0]), 'network_forbidden': True}
        table, report = kalman_components(kalman); rsrs, capture = literal_rsrs_capture()
        result = {'Kalman': report, 'actual_RSRS': capture, 'loaded_modules': modules,
            'network_forbidden': True, 'new_requests': 0, 'new_trading_backtests': 0,
            'formal_source_research_acceptance': False, 'formal_trading_acceptance': False}
        if mode == 'compute':
            with (DIRECTORY/'kalman-native-reference.parquet').open('xb') as out: table.to_parquet(out, index=False)
            with (DIRECTORY/'literal-rsrs-capture.parquet').open('xb') as out: rsrs.to_parquet(out, index=False)
            save(DIRECTORY/'component-research.json', result)
        else:
            require(mode == 'replay', 'Unknown worker mode')
            pd.testing.assert_frame_equal(table, pd.read_parquet(DIRECTORY/'kalman-native-reference.parquet'), check_exact=True)
            pd.testing.assert_frame_equal(rsrs, pd.read_parquet(DIRECTORY/'literal-rsrs-capture.parquet'), check_exact=True)
            require(result == read(DIRECTORY/'component-research.json'), 'Numerical summary differs')
            save(DIRECTORY/'offline-verification.json', {'status': 'match', 'differences': 0,
                'Kalman_windows': len(table), 'actual_RSRS_rows': len(rsrs), 'network_forbidden': True,
                'source_sha256': previous.SOURCE_SHA[1], 'duplicate_is_independent_economic_sample': False,
                'implementation_sha256': hashes, 'table_sha256': {name: file_sha(DIRECTORY/name) for name in
                    ('kalman-native-reference.parquet', 'literal-rsrs-capture.parquet')}, 'not_a_trading_backtest': True})
    return result


def run_worker(mode):
    env = os.environ.copy(); env['OBSERVE_BATCH94_WORKER'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('PYTHONPATH', None)
    return subprocess.run([sys.executable, '-B', '-m', 'scripts.reference_strategy_batch94', 'worker', '--mode', mode],
        env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'compute', 'replay', 'smoke', 'worker', 'protect'))
    parser.add_argument('--mode', choices=('compute', 'replay', 'smoke'))
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare('data')
    elif args.action == 'worker': result = worker(args.mode)
    elif args.action == 'protect':
        historical_stage(); validate_index('data'); validate_runtime(DIRECTORY)
        result = protect('data', DIRECTORY); save(DIRECTORY/'protection.json', result)
    else:
        run_worker(args.action); result = {'status': 'worker_completed', 'mode': args.action}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
