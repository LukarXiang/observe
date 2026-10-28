"""Batch94 numerical stage: literal FFT/RSRS, native candidates, no trading claims."""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch59 import index_operand
from scripts.review_strategy_batch73 import safe_function
from scripts.review_strategy_batch79 import reused_inputs
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch94/20261010-etf-filters-numerical-stage')
READINESS = Path('data/staging/strategies-batch94/20261010-etf-filters-readiness')
PARENT = Path('docs/handoff/2026-10-10-batch93-final-review.json')
POOL = ('510050.XSHG', '159928.XSHE', '510300.XSHG', '159949.XSHE')
SOURCE_SHA = ('344b7d6ce450c51142b2070edef0a4eacb4ce9ce6f41779b0353207ee5ea46cb',
    '3bfda3a9fe80661ed94e1a19ed1b214b6576d12b8cf1bdcdaaf58a5d4a2d2805',
    '3bfda3a9fe80661ed94e1a19ed1b214b6576d12b8cf1bdcdaaf58a5d4a2d2805')
CUTOFF = '2026-09-29'
FILES = ('scripts/review_strategy_batch94.py', 'tests/unit/test_batch94_stage.py')
TOLERANCE = {'relative': 2e-9, 'absolute': 2e-10}


def canonical_root(root):
    require(Path(root).resolve() == Path('data').resolve(), 'Batch94 requires the actual data root')


def parent_binding():
    doc = read(PARENT)
    require(doc['status'] == 'ok' and doc['formal_source_research_acceptance'] and
        not doc['formal_trading_acceptance'] and doc['manually_reviewed'] == 270, 'Parent acceptance differs')
    paths = {str(PARENT): file_sha(PARENT)}
    for key in ('receipt', 'checked_state', 'final_binding', 'formal_report', 'review_completion'):
        name = doc[key+'_file'] if key != 'formal_report' else doc[key]
        require(file_sha(name) == doc[key+'_sha256'], 'Parent evidence changed: '+name)
        paths[name] = file_sha(name)
    for name, sha in doc['reviewed_implementation_sha256'].items():
        require(file_sha(name) == sha, 'Accepted implementation changed: '+name)
        paths[name] = sha
    return paths


def literal(directory, number, namespace):
    path = directory/'sources'/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number], 'Source identity differs')
    tree = ast.parse(read_source(path)[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == 14 and all(safe_function(n) for n in nodes), 'Definition contract differs')
    # Imports and all platform top-level expressions are excluded. Only reviewed SHA-bound definitions run.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def native_frame(row):
    require(file_sha(row['file']) == row['sha256'], 'Candidate byte SHA differs')
    original = pd.read_parquet(row['file'])
    require(len(original) == row['rows'], 'Candidate rows differ')
    frame = original.copy()
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    require(not frame.date.duplicated().any() and frame.date.is_monotonic_increasing, 'Native dates ambiguous')
    # Some old parsers preserve numeric strings. Explicitly parse a working copy; raw bytes stay immutable.
    for name in ('open', 'high', 'low', 'close'):
        frame[name] = pd.to_numeric(frame[name], errors='raise').astype(float)
        require(np.isfinite(frame[name]).all() and (frame[name] > 0).all(), 'Native OHLC not finite/positive')
    return frame.loc[frame.date <= CUTOFF].reset_index(drop=True), original


def candidate_inputs():
    inputs = reused_inputs()['inputs']
    selected = {s: inputs[s] for s in POOL if s in inputs}
    require(set(selected) == {'510050.XSHG', '510300.XSHG', '159949.XSHE'}, 'Known candidate pool changed')
    profiles = []
    for symbol, row in selected.items():
        frame, original = native_frame(row)
        for wire in row.get('wire_responses', []):
            require(file_sha(wire['file']) == wire['sha256'], 'Candidate wire changed')
        changes = frame.close.pct_change()
        jumps = frame.loc[changes.abs() >= .2, ['date', 'close']].copy()
        jumps['native_close_change'] = changes.loc[jumps.index]
        profiles.append({'symbol': symbol, 'candidate': row, 'raw_rows': len(original),
            'numeric_source_dtypes': {s: str(original[s].dtype) for s in ('open', 'high', 'low', 'close')},
            'rows_to_cutoff': len(frame), 'excluded_after_cutoff': len(original)-len(frame),
            'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'price_jump_examples': jumps.to_dict('records'),
            'strict_usable': False, 'native_prices_only': True, 'event_explanation_proven': False})
    return {'funds': selected, 'profiles': profiles, 'missing_from_known_candidates': ['159928.XSHE'],
        'new_requests': 0, 'original_adjusted_price_equivalence_proven': False}


def benchmark_candidates():
    folder = Path('data/staging/strategies-batch74/20261008-FOF-two-eight')
    result_file = folder/'sina-fallback/index-sz399006/result.json'
    registry = folder/'sina-fallback/results.json'
    require(file_sha(registry) == read(folder/'checked-state.json')['evidence_sha256']['sina-fallback/results.json'],
        'Benchmark registry changed')
    row = read(result_file)
    require(row in read(registry)['results'] and row['case'] == 'index-sz399006', 'Benchmark identity differs')
    frame, _ = native_frame(row)
    for wire in row['wire_responses']:
        require(file_sha(wire['file']) == wire['sha256'], 'Benchmark wire changed')
    # Independent earlier BaoStock candidate is audited, not substituted or silently merged.
    old_file = Path('data/staging/strategies-batch12/20261005-daily-candidates/probes/bs_399006/result.json')
    old = read(old_file)
    receipt = read('docs/handoff/2026-10-05-batch12-verification.json')
    require(old['raw_sha256'] == 'b17838d931f8c8a0ec30e51f1a2557cf0c379374514bbebc3a39103427bf64ed' and
        file_sha(old['raw_file']) == old['raw_sha256'] and old['rows'] == 3967 and
        old in [r['result'] for r in receipt['probes']['results'] if r.get('endpoint') == 'bs_399006'],
        'Earlier benchmark evidence changed')
    older = pd.read_parquet(old['raw_file'])
    older['date'] = pd.to_datetime(older.date).dt.strftime('%Y-%m-%d')
    overlap = frame.merge(older, on='date', suffixes=('_sina', '_bs'))
    errors = {name: float(np.max(np.abs(pd.to_numeric(overlap[name+'_sina']).to_numpy(float)-
        pd.to_numeric(overlap[name+'_bs']).to_numpy(float)))) for name in ('open', 'high', 'low', 'close')}
    return {'sina': row, 'sina_registry': str(registry), 'sina_registry_sha256': file_sha(registry),
        'earlier_baostock': old, 'earlier_result_sha256': file_sha(old_file), 'rows_to_cutoff': len(frame),
        'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'overlap_rows': len(overlap),
        'maximum_OHLC_provider_difference': errors, 'published_benchmark_rows': 0,
        'new_requests': 0, 'strict_usable': False, 'platform_benchmark_equivalence_proven': False}


def supplement_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_sina
    source = __import__('inspect').getsource(fn)
    registry = Path('data/staging/strategies-batch84/20261009-industry-ETF-T0/probe-results.json')
    state = registry.parent/'checked-state.json'
    require(file_sha(registry) == read(state)['evidence_sha256']['probe-results.json'], 'Previous supplement changed')
    row = next(r for r in read(registry)['results'] if r['case'] == 'daily-161725')
    require(row['status'] == 'failed' and 'NameResolutionError' in row['error'] and not row['wire_responses'],
        'Previous DNS diagnosis changed')
    return {'missing_symbol': '159928.XSHE', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz159928'},
        'installed_function_source': source, 'installed_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'prior_failure_registry': str(registry), 'prior_failure_registry_sha256': file_sha(registry),
        'prior_failure': row, 'new_requests': 0, 'new_samples': 0,
        'decision': 'Same finance.sina.com.cn host previously failed DNS; no blind symbol-only retry',
        'missing': ['159928 native history', 'all four fund historical identity/events/reference prices/adjustment/state',
            'fund fees and platform open execution/position feedback', 'pykalman version and exact smooth semantics']}


def prepare(root, directory):
    canonical_root(root)
    require(not directory.exists(), 'Stage directory already exists')
    bindings = parent_binding(); ready = read(READINESS/'source-readiness.json')
    require(file_sha(READINESS/'source-readiness.json') == read('docs/handoff/2026-10-10-batch94-readiness.json')['readiness_sha256'],
        'Readiness changed')
    funds = candidate_inputs(); benchmarks = benchmark_candidates(); supplements = supplement_evidence()
    frame, operand = index_operand(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Published batch changed')
    with operation_lock(root, DATA_WRITER):
        checkpoint(root, directory, 'Batch94 numerical stage baseline; three original native ETF candidates restored')
        save(directory/'parent-binding.json', bindings)
        save(directory/'candidate-inputs.json', funds); save(directory/'benchmark-candidates.json', benchmarks)
        save(directory/'supplement-evidence.json', supplements)
        (directory/'sources').mkdir()
        for i, row in enumerate(ready['sources']):
            source = Path('repo/量化策略源代码')/row['path']
            require(file_sha(source) == file_sha(READINESS/f'{i}.source') == SOURCE_SHA[i], 'Original source differs')
            shutil.copyfile(source, directory/'sources'/f'{i}.source')
        save(directory/'source-map.json', {'sources': ready['sources'], 'source_sha256': SOURCE_SHA,
            'duplicate_relation': {'2': 1}, 'formal_source_research_acceptance': False})
        with (directory/'index-input.parquet').open('xb') as out: frame.to_parquet(out, index=False)
        save(directory/'index-binding.json', {**operand, 'frozen_sha256': file_sha(directory/'index-input.parquet')})
    return {'status': 'prepared', 'native_funds': len(funds['funds']), 'index_rows': len(frame),
        'benchmark_candidate_rows': benchmarks['rows_to_cutoff'], 'new_trading_backtests': 0}


def scalar_ols(x, y):
    x = [float(v) for v in x]; y = [float(v) for v in y]
    mx = math.fsum(x)/len(x); my = math.fsum(y)/len(y)
    dx = [v-mx for v in x]; dy = [v-my for v in y]
    slope = math.fsum(a*b for a, b in zip(dx, dy, strict=True))/math.fsum(a*a for a in dx)
    intercept = my-slope*mx
    r2 = 1-math.fsum((b-(slope*a+intercept))**2 for a, b in zip(x, y, strict=True))/math.fsum(b*b for b in dy)
    return intercept, slope, r2


def scalar_z(values):
    mean = math.fsum(float(v) for v in values)/len(values)
    std = math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/len(values))
    return (float(values[-1])-mean)/std


def close_enough(a, b, label):
    require(math.isclose(float(a), float(b), rel_tol=TOLERANCE['relative'], abs_tol=TOLERANCE['absolute']),
        f'{label} differs: {a!r} != {b!r}')


def timing(score):
    return 'BUY' if score > .7 else 'SELL' if score < -.7 else 'KEEP'


def rsrs_components(directory, number=0):
    frame = pd.read_parquet(directory/'index-input.parquet')
    require(file_sha(directory/'index-input.parquet') == read(directory/'index-binding.json')['frozen_sha256'], 'Index bytes changed')
    frame['date'] = frame.date.astype(str); values = frame[['low', 'high']].to_numpy(float)
    log = SimpleNamespace(info=lambda *args: None)
    position = [618]; calls = []
    def history(stock, count, unit, fields):
        require(stock == '000300.XSHG' and unit == '1d' and count in (18, 618), 'RSRS history query differs')
        calls.append((position[0], count, list(fields)))
        return frame.iloc[position[0]-count:position[0]].reset_index(drop=True)[fields]
    g = SimpleNamespace(N=18, M=600, ref_stock='000300.XSHG', score_threshold=.7)
    ns = literal(directory, number, {'np': np, 'g': g, 'attribute_history': history, 'log': log})
    initial = ns['initial_slope_series'](); independent = [scalar_ols(*values[i:i+18].T)[1] for i in range(600)]
    require(len(initial) == len(independent) == 600, 'Original seed length differs')
    for a, b in zip(initial, independent, strict=True): close_enough(a, b, 'seed')
    g.slope_series = initial[:-1]; independent = independent[:-1]; rows = []
    max_errors = dict.fromkeys(('intercept', 'slope', 'r2', 'zscore', 'score'), 0.)
    for end in range(618, len(frame)):
        position[0] = end
        a_intercept, a_slope, a_r2 = ns['get_ols'](*values[end-18:end].T)
        b_intercept, b_slope, b_r2 = scalar_ols(*values[end-18:end].T)
        signal = ns['get_timing_signal'](None, 'argument_is_ignored')
        independent.append(b_slope)
        a_z = ns['get_zscore'](g.slope_series[-600:]); b_z = scalar_z(independent[-600:])
        a_score = a_z*a_r2; b_score = b_z*b_r2
        for name, a, b in zip(max_errors, (a_intercept, a_slope, a_r2, a_z, a_score),
                (b_intercept, b_slope, b_r2, b_z, b_score), strict=True):
            close_enough(a, b, name); max_errors[name] = max(max_errors[name], abs(float(a)-float(b)))
        require(signal == timing(b_score), 'Strict signal differs from independent formula')
        rows.append({'decision_date': frame.date.iloc[end], 'last_input_date': frame.date.iloc[end-1],
            'intercept': a_intercept, 'slope': a_slope, 'r2': a_r2, 'zscore': a_z, 'score': a_score,
            'scalar_intercept': b_intercept, 'scalar_slope': b_slope, 'scalar_r2': b_r2,
            'scalar_zscore': b_z, 'scalar_score': b_score, 'signal': signal})
    return pd.DataFrame(rows), {'rows': len(rows), 'first': rows[0]['decision_date'], 'last': rows[-1]['decision_date'],
        'last_input_date': rows[-1]['last_input_date'], 'seed_rows': 618, 'seed_slopes_before_discard': 600,
        'seed_slopes_after_discard': 599, 'discarded_window_start': 599, 'first_appended_window_start': 600,
        'ignored_seed_last_row': frame.date.iloc[617], 'original_initial_gap_preserved': True,
        'maximum_independent_errors': max_errors, 'signals': dict(Counter(r['signal'] for r in rows)),
        'literal_history_calls': len(calls), 'not_a_trading_backtest': True}


def fft_slope_kernel(size=250, cutoff=60, momentum=20):
    """Independent DFT cosine projection; no FFT or polyfit used."""
    require(size >= momentum >= 2 and 0 <= cutoff < size/2, 'DFT projection parameters invalid')
    kept = [k for k in range(size) if k <= cutoff or k >= size-cutoff]
    midpoint = (momentum-1)/2; norm = math.fsum((i-midpoint)**2 for i in range(momentum))
    weights = [(i-midpoint)/norm for i in range(momentum)]
    return [math.fsum(w*math.cos(2*math.pi*k*(size-momentum+t-j)/size)
        for k in kept for t, w in enumerate(weights))/size for j in range(size)]


def fft_components(directory):
    inputs = read(directory/'candidate-inputs.json')['funds']; kernel = fft_slope_kernel(); rows = []
    original = literal(directory, 0, {'np': np})['fftFilter']; profiles = []
    for symbol in POOL:
        if symbol not in inputs: continue
        frame, _ = native_frame(inputs[symbol]); closes = frame.close.to_numpy(float); start = len(rows)
        maximum = 0.
        for end in range(250, len(frame)+1):
            window = closes[end-250:end]; logs = np.log(window/window[0])
            filtered = original(60, 250, logs)
            score = float(np.polyfit(np.arange(20), filtered[-20:], 1)[0].real)
            scalar_score = math.fsum(w*math.log(float(v)/float(window[0])) for w, v in zip(kernel, window, strict=True))
            close_enough(score, scalar_score, 'FFT slope'); maximum = max(maximum, abs(score-scalar_score))
            rows.append({'symbol': symbol, 'last_input_date': frame.date.iloc[end-1], 'score': score,
                'independent_DFT_score': scalar_score, 'imaginary_maximum': float(np.max(np.abs(filtered.imag))),
                'window_first': frame.date.iloc[end-250], 'input_rows': 250, 'native_unadjusted_reference': True})
        profiles.append({'symbol': symbol, 'windows': len(rows)-start, 'first_last_input': frame.date.iloc[249],
            'last_last_input': frame.date.iloc[-1], 'maximum_independent_error': maximum})
    return pd.DataFrame(rows), {'windows': len(rows), 'profiles': profiles, 'kept_bins': 121, 'removed_bins': 129,
        'boundary_bins_60_190_kept': True, 'full_original_pool_ranking_rows': 0, 'missing_pool': ['159928.XSHE'],
        'original_adjustment_equivalence_proven': False, 'last_observation_has_no_supported_next_execution_day': True,
        'not_a_trading_backtest': True}


def diagnostics(directory):
    rows = []; log = SimpleNamespace(info=lambda *args: None, debug=lambda *args: None)
    g = SimpleNamespace(stock_pool=list(POOL), momentum_day=20, stock_num=1, M=600, N=18,
        score_threshold=.7, ref_stock='000300.XSHG', slope_series=[1.]*599)
    dates = pd.date_range('2020-01-01', periods=250)
    ns = literal(directory, 0, {'np': np, 'g': g, 'log': log,
        'attribute_history': lambda stock, n, unit, fields: pd.DataFrame({'close': np.linspace(1, 2, n)}, index=dates[-n:])})
    try: ns['get_rank'](None, ['unused'])
    except KeyError as exc: rows.append({'case': 'FFT_datetime_index_zero_label', 'error_type': type(exc).__name__, 'error': str(exc)})
    else: raise AssertionError('Expected original pandas label error')
    ns['attribute_history'] = lambda stock, n, unit, fields: pd.DataFrame({'close': np.linspace(1, 2, n)})
    rank = ns['get_rank'](None, ['unused'])
    require(len(rank) == 2 and rank[0] == POOL[0], 'Original rank pair/stable tie differs')
    rows.append({'case': 'FFT_RangeIndex_shape_assumption', 'result_security': rank[0], 'result_length': len(rank),
        'score_type': type(rank[1]).__name__, 'stock_pool_argument_ignored': True, 'synthetic': True})
    calls = []
    class SmoothSpy:
        def __init__(self, **kwargs): calls.append({'constructor': kwargs})
        def smooth(self, observations):
            calls.append({'method': 'smooth', 'shape': list(observations.shape)})
            return np.asarray(observations).reshape(-1, 1), np.zeros((len(observations), 1, 1))
    kns = literal(directory, 1, {'np': np, 'g': g, 'log': log, 'KalmanFilter': SmoothSpy,
        'attribute_history': ns['attribute_history']})
    krank = kns['get_rank'](None, ['unused'])
    require(len(calls) == 8 and all(calls[i]['method'] == 'smooth' for i in (1, 3, 5, 7)) and
        krank[1].shape == (1,) and calls[0]['constructor']['observation_covariance'] == .15 and
        calls[0]['constructor']['transition_covariance'] == .1, 'Kalman invocation shape differs')
    rows.append({'case': 'Kalman_constructor_and_smooth_call_spy', 'calls': calls, 'score_shape': list(krank[1].shape),
        'synthetic': True, 'Kalman_numerics_reproduced': False, 'pykalman_installed': importlib.util.find_spec('pykalman') is not None})
    kns['attribute_history'] = lambda stock, n, unit, fields: pd.DataFrame({'close': np.linspace(1, 2, n)}, index=dates[-n:])
    try: kns['get_rank'](None, [])
    except KeyError as exc: rows.append({'case': 'Kalman_datetime_index_zero_label', 'error_type': type(exc).__name__})
    else: raise AssertionError('Expected original Kalman pandas label error')
    for score in (-.7000000001, -.7, .7, .7000000001):
        ns['attribute_history'] = lambda *args: SimpleNamespace(low=np.arange(18.), high=np.arange(18.)+1)
        ns['get_ols'] = lambda *args: (0., 1., 1.)
        ns['get_zscore'] = lambda *args, score=score: score
        require(ns['get_timing_signal'](None, None) == timing(score), 'Strict threshold differs')
        rows.append({'case': 'RSRS_strict_boundary', 'score': score, 'signal': timing(score), 'synthetic': True})
    output = StringIO()
    with redirect_stdout(output):
        targets = []
        ns['order_target_value'] = lambda security, value: targets.append([security, value])
        position = SimpleNamespace(security=POOL[0], avg_cost=10., price=1., value=100., total_amount=100)
        context = SimpleNamespace(portfolio=SimpleNamespace(positions={POOL[0]: position}))
        ns['check_lose'](context)
    require(targets == [[POOL[0], 0]], 'Original negative90 percent threshold differs')
    rows.append({'case': 'negative90_percent_stoploss', 'orders': targets, 'synthetic': True})
    return {'cases': rows, 'case_count': len(rows), 'platform_top_level_executed': False,
        'numerical_proxy_approved': False, 'new_trading_backtests': 0}


def validate_inputs(directory):
    require(parent_binding() == read(directory/'parent-binding.json'), 'Parent binding changed')
    require(candidate_inputs() == read(directory/'candidate-inputs.json'), 'Fund input evidence changed')
    require(benchmark_candidates() == read(directory/'benchmark-candidates.json'), 'Benchmark evidence changed')
    require(supplement_evidence() == read(directory/'supplement-evidence.json'), 'Supplement evidence changed')
    bound = read(directory/'index-binding.json')
    for row in bound['partitions']: require(file_sha(row['file']) == row['sha256'], 'Index partition changed')
    for number in range(3): literal(directory, number, {})


def compute(root, directory):
    canonical_root(root); validate_inputs(directory)
    require(not (directory/'component-research.json').exists(), 'Numerical stage already computed')
    index, index_summary = rsrs_components(directory); fft, fft_summary = fft_components(directory)
    cases = diagnostics(directory)
    for name, frame in (('rsrs-reference.parquet', index), ('fft-native-reference.parquet', fft)):
        with (directory/name).open('xb') as out: frame.to_parquet(out, index=False)
    save(directory/'diagnostics.json', cases)
    result = {'status': 'numerical_stage_computed_not_accepted', 'RSRS': index_summary, 'FFT': fft_summary,
        'pykalman_numeric_windows': 0, 'new_trading_backtests': 0, 'formal_source_research_acceptance': False,
        'formal_trading_acceptance': False, 'tolerance': TOLERANCE}
    save(directory/'component-research.json', result)
    return result


def blocked(*args, **kwargs): raise AssertionError('Network forbidden during numerical replay')


def offline(root, directory):
    canonical_root(root); validate_inputs(directory)
    with ExitStack() as stack:
        for name in ('connect', 'connect_ex'):
            stack.enter_context(patch.object(socket.socket, name, blocked))
        stack.enter_context(patch.object(socket, 'create_connection', blocked))
        stack.enter_context(patch.object(socket, 'getaddrinfo', blocked))
        stack.enter_context(patch.object(requests.sessions.Session, 'request', blocked))
        index, index_summary = rsrs_components(directory); fft, fft_summary = fft_components(directory)
        pd.testing.assert_frame_equal(index, pd.read_parquet(directory/'rsrs-reference.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(fft, pd.read_parquet(directory/'fft-native-reference.parquet'), check_exact=True)
        for number in (1, 2):
            other, summary = rsrs_components(directory, number)
            pd.testing.assert_frame_equal(index, other, check_exact=True)
            require(summary == index_summary, 'Other literal RSRS source differs')
        require(diagnostics(directory) == read(directory/'diagnostics.json'), 'Literal diagnostics changed')
        previous = read(directory/'component-research.json')
        require(index_summary == previous['RSRS'] and fft_summary == previous['FFT'], 'Component summaries changed')
    result = {'status': 'match', 'differences': 0, 'RSRS_rows': len(index), 'FFT_rows': len(fft),
        'diagnostics': len(read(directory/'diagnostics.json')['cases']), 'network_forbidden': True,
        'literal_RSRS_source_numbers_compared': [0, 1, 2], 'same_SHA_duplicate_is_independent_economic_sample': False,
        'original_full_strategy_reproduction': False, 'not_a_trading_backtest': True}
    save(directory/'offline-verification.json', result)
    return result


def finish_stage(root, directory):
    canonical_root(root); validate_inputs(directory)
    require(read(directory/'offline-verification.json')['status'] == 'match', 'Offline stage incomplete')
    commands = read(directory/'command-results.json')
    require(commands['implementation_sha256'] == {name: file_sha(name) for name in FILES} and
        all(c['returncode'] == 0 and file_sha(c['log_file']) == c['log_sha256'] for c in commands['commands']),
        'Stage command checks missing, stale, or failed')
    with operation_lock(root, DATA_WRITER):
        result = protect(root, directory)
        progress = archive(root, 'Batch94 numerical stage: long literal RSRS and3 native FFT references; no formal acceptance/trading')
        save(directory/'progress-checkpoint.json', progress)
        (directory/'implementation').mkdir()
        hashes = {name: file_sha(name) for name in FILES}
        for name in FILES:
            dest = directory/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        evidence = {p.relative_to(directory).as_posix(): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        save(directory/'stage-binding.json', {'status': 'numerical_stage_verified_not_formally_accepted',
            'implementation_sha256': hashes, 'evidence_sha256': evidence, 'protection': result,
            'checkpoint': progress['checkpoint'], 'source_count_formally_accepted': 270,
            'new_trading_backtests': 0, 'full_library_complete': False})
    return {'status': 'stage_archived_not_accepted', 'checkpoint': progress['checkpoint'], 'protection': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'compute', 'offline', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    action = {'prepare': prepare, 'compute': compute, 'offline': offline, 'finish': finish_stage}[args.action]
    print(json.dumps(action(args.root, args.directory), ensure_ascii=False, indent=2))
