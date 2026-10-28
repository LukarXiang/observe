"""Longest candidate ES operands, with two explicit references and no portfolio trades."""
import argparse
import ast
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from scripts import prepare_strategy_batch101 as inputs
from scripts import research_strategy_batch101 as original
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch101/20261011-longest-ES-reference')
POLICIES = ('explicit_forward_fill', 'no_forward_fill')
FILES = (*inputs.FILES, 'scripts/reference_strategy_batch101.py', 'tests/unit/test_batch101_reference.py')
OUTPUTS = ('ES-reference.parquet', 'summary.json', 'diagnostics.json')


def prepare():
    require(not DIRECTORY.exists(), 'Immutable ES reference output exists')
    base = read(inputs.DIRECTORY/'prepared.json')
    parent = read(base['parent'])
    require(file_sha(base['parent']) == base['parent_sha256'] and parent['status'] == 'ok' and
            parent['progress']['manually_reviewed'] == 286, 'Accepted parent identity differs')
    for name, digest in base['input_sha256'].items(): require(file_sha(name) == digest, 'Native input changed')
    for name, digest in base['implementation_sha256'].items():
        require(file_sha(name) == file_sha(inputs.DIRECTORY/'implementation'/name) == digest, 'Prepared code changed')
    require(file_sha(inputs.DIRECTORY/'dependencies.json') == base['dependencies_sha256'], 'Dependency inventory changed')
    state = Store('data').state(inputs.parent_study.SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(inputs.parent_study.SNAPSHOT+'.json')) ==
            inputs.parent_study.SNAPSHOT_SHA, 'Actual frozen snapshot changed')
    calendar = Store('data').load_state(state, 'calendar')
    days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d')
    require(days.is_monotonic_increasing and not days.duplicated().any(), 'Open calendar invalid')
    DIRECTORY.mkdir(parents=True)
    save(DIRECTORY/'open-calendar.json', days.loc[days.le('2026-09-29')].tolist())
    for number in (0, 1):
        require(file_sha(inputs.DIRECTORY/'sources'/f'{number}.source') == original.SOURCE_SHA[number], 'Source copy changed')
    for name in FILES:
        target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    bindings = {str(inputs.DIRECTORY/'prepared.json'): file_sha(inputs.DIRECTORY/'prepared.json'),
        str(inputs.DIRECTORY/'dependencies.json'): base['dependencies_sha256'], base['parent']: base['parent_sha256'],
        **base['input_sha256'], str(DIRECTORY/'open-calendar.json'): file_sha(DIRECTORY/'open-calendar.json')}
    bindings.update({str(inputs.DIRECTORY/'sources'/f'{n}.source'): sha for n, sha in enumerate(original.SOURCE_SHA)})
    save(DIRECTORY/'prepared.json', {'evidence_sha256': bindings,
        'implementation_sha256': {name: file_sha(name) for name in FILES},
        'source_sha256': list(original.SOURCE_SHA), 'snapshot': inputs.parent_study.SNAPSHOT,
        'policies': list(POLICIES), 'all_native120_windows_including_gap_crossings_are_reference_only': True,
        'asof_semantics': 'Last included native observation; not a decision time or original signal',
        'gap_policy': 'Calendar missing dates annotated; ffill is an explicit reference only, no source input repair',
        'full_original_portfolio_admitted': False, 'new_trading_backtests': 0, 'strict_usable': False})
    return {'status': 'prepared_reference_only', 'funds': len(base['input_sha256'])}


def validate():
    doc = read(DIRECTORY/'prepared.json')
    require(doc['source_sha256'] == list(original.SOURCE_SHA) and doc['policies'] == list(POLICIES) and
            doc['full_original_portfolio_admitted'] is False and doc['new_trading_backtests'] == 0 and
            doc['strict_usable'] is False, 'Reference scope differs')
    for name, digest in doc['evidence_sha256'].items(): require(file_sha(name) == digest, 'Frozen operand changed: '+name)
    for name, digest in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == digest, 'Frozen implementation changed: '+name)
    original.tree(0); original.tree(1)
    return doc


def reference_returns(window, policy):
    require(policy in POLICIES, 'Unknown return reference')
    series = pd.Series(window.close.to_numpy(float), index=pd.to_datetime(window.date))
    daily = series.resample('D').last()
    if policy == 'explicit_forward_fill': daily = daily.ffill()
    return daily.pct_change(fill_method=None).fillna(0).to_numpy(float)


def independent_returns(window, policy):
    """Construct each natural day directly, without resample, pct_change or pandas fill."""
    require(policy in POLICIES, 'Unknown return reference')
    observations = {dt.date.fromisoformat(d): float(v) for d, v in zip(window.date, window.close, strict=True)}
    start = dt.date.fromisoformat(window.date.iloc[0]); stop = dt.date.fromisoformat(window.date.iloc[-1])
    prior = None; last_observed = None; values = []
    for offset in range((stop-start).days+1):
        current = observations.get(start+dt.timedelta(days=offset))
        if current is not None: last_observed = current
        if policy == 'explicit_forward_fill': current = last_observed
        values.append(0. if prior is None or current is None else current/prior-1.)
        prior = current
    return np.asarray(values, dtype=float)


def original_ES_function():
    nodes = [node for node in ast.walk(original.tree(0))
             if isinstance(node, ast.FunctionDef) and node.name == 'get_stock_ES']
    require(len(nodes) == 1, 'Original ES definition missing')
    values = np.empty((0, 1)); namespace = {'getdailyreturn': lambda *a: values.copy()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<unchanged72-ES-return-injection>', 'exec'), namespace)
    def evaluate(returns):
        nonlocal values
        values = np.asarray(returns, dtype=float).reshape(-1, 1)
        return namespace['get_stock_ES']('reference_only', '1d', 120, 2.58)
    return evaluate


def components(frames, open_days):
    kernel = original_ES_function(); rows = []; profiles = {}; max_return_error = 0.; max_ES_error = 0.
    calendar = set(open_days)
    for symbol, frame in sorted(frames.items()):
        normalized, excluded = inputs.normalize(frame)
        require(excluded == 0 and normalized.equals(frame) and len(frame) >= 120, 'Frozen native window invalid')
        observed = set(frame.date)
        require(observed <= calendar, 'Native observations outside frozen open calendar')
        missing = sorted(d for d in calendar if frame.date.iloc[0] <= d <= frame.date.iloc[-1] and d not in observed)
        symbol_rows = []
        for end in range(119, len(frame)):
            window = frame.iloc[end-119:end+1]; start, stop = window.date.iloc[0], window.date.iloc[-1]
            crossed = [d for d in missing if start <= d <= stop]
            for policy in POLICIES:
                daily = reference_returns(window, policy); manual_daily = independent_returns(window, policy)
                require(daily.shape == manual_daily.shape and np.isfinite(daily).all() and
                        np.array_equal(daily, manual_daily), 'Independent natural-day returns differ')
                error = float(np.max(np.abs(daily-manual_daily))); max_return_error = max(max_return_error, error)
                result = kernel(daily)
                # confidence2.58: inclusive i<=1, divided by1, regardless of natural-day length.
                manual_ES = -math.fsum(sorted(map(float, manual_daily))[:2])
                require(isinstance(result, np.ndarray) and result.shape == (1,) and
                        math.isclose(float(result[0]), manual_ES, rel_tol=1e-12, abs_tol=1e-15), 'Original inclusive ES tail differs')
                ES = float(result[0]); max_ES_error = max(max_ES_error, abs(ES-manual_ES))
                row = {'instrument': symbol, 'window_first': start, 'asof_native_close': stop,
                    'policy': policy, 'native_rows': 120, 'natural_day_rows': len(daily),
                    'missing_open_dates': json.dumps(crossed), 'missing_open_count': len(crossed),
                    'ES': ES, 'independent_ES': manual_ES,
                    'inverse_ES_finite': math.isfinite(ES) and ES != 0,
                    'original_return_input_proven': False}
                rows.append(row); symbol_rows.append(row)
        local = pd.DataFrame(symbol_rows)
        comparison = local.pivot(index='asof_native_close', columns='policy', values='ES')
        profiles[symbol] = {'candidate_rows': len(frame), 'first_candidate': frame.date.iloc[0],
            'first_reference_asof': frame.date.iloc[119], 'last_reference_asof': frame.date.iloc[-1],
            'windows_per_policy': len(frame)-119, 'year_counts_per_policy':
                frame.date.iloc[119:].str[:4].value_counts().sort_index().to_dict(),
            'missing_dates_within_native_first_last': missing,
            'gap_crossing_windows_per_policy': int(local.loc[local.policy.eq(POLICIES[0]), 'missing_open_count'].gt(0).sum()),
            'ES_policy_differences': int(comparison[POLICIES[0]].ne(comparison[POLICIES[1]]).sum()),
            'max_ES_policy_absolute_difference': float((comparison[POLICIES[0]]-comparison[POLICIES[1]]).abs().max()),
            'zero_ES_by_policy': {p: int(local.loc[local.policy.eq(p), 'ES'].eq(0).sum()) for p in POLICIES},
            'negative_ES_by_policy': {p: int(local.loc[local.policy.eq(p), 'ES'].lt(0).sum()) for p in POLICIES}}
        print(json.dumps({'fund_complete': symbol, 'reference_windows_per_policy': len(frame)-119}), flush=True)
    output = pd.DataFrame(rows)
    return output, {'profiles': profiles, 'native_windows_per_policy': len(output)//2, 'reference_rows': len(output),
        'independent_return_max_abs_error': max_return_error, 'independent_ES_max_abs_error': max_ES_error,
        'policies': list(POLICIES), 'strict_usable': False, 'original_daily_return_pipeline_runnable': False,
        'full_original_portfolio_weights': 0, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
        'limits': ['Unchanged nested ES accepts injected reference returns; original resample/how still fails',
            'Five candidates are not original seven-fund allocation',
            '120 native observations per window, natural days can exceed120; fixed original denominator1',
            'Gap-crossing windows retained and marked for sensitivity only',
            'As-of close is not a trading decision; adjustment/pause/events and platform equivalence unproven']}


def worker(mode):
    validate(); folder = DIRECTORY/mode
    require(mode in ('study', 'replay') and folder.is_dir() and not (folder/'completed.json').exists(), 'Worker identity/output differs')
    require(not any((folder/name).exists() for name in OUTPUTS), 'Immutable worker artifacts exist')
    base = read(inputs.DIRECTORY/'prepared.json')
    with network_blocked():
        frames = {Path(name).stem: pd.read_parquet(name) for name in base['input_sha256']}
        frame, summary = components(frames, read(DIRECTORY/'open-calendar.json'))
        diagnostics = original.diagnostics()
    frame.to_parquet(folder/OUTPUTS[0], index=False); save(folder/OUTPUTS[1], summary); save(folder/OUTPUTS[2], diagnostics)
    save(folder/'completed.json', {'status': 'ok', 'mode': mode, 'network_blocked': True,
        'prepared_sha256': file_sha(DIRECTORY/'prepared.json'),
        'output_sha256': {name: file_sha(folder/name) for name in OUTPUTS}, 'reference_rows': len(frame)})
    return {'status': 'ok', 'mode': mode, 'reference_rows': len(frame)}


def run(mode):
    validate(); folder = DIRECTORY/mode; folder.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.reference_strategy_batch101', 'worker', '--mode', mode]
    with (folder/'stdout.log').open('x') as out, (folder/'stderr.log').open('x') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        timed_out = False
        try: returncode = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.kill(); returncode = process.wait(); timed_out = True
    save(folder/'command.json', {'command': command, 'cwd': str(Path.cwd()), 'returncode': returncode,
        'timeout1800': timed_out, 'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(returncode == 0 and not timed_out, 'Reference worker failed; keep terminal logs, never overwrite')
    return read(folder/'completed.json')


def offline():
    validate(); target = DIRECTORY/'offline-verification.json'
    require(not target.exists(), 'Immutable verification exists')
    run('replay')
    proofs = [read(DIRECTORY/mode/'completed.json') for mode in ('study', 'replay')]
    for mode, proof in zip(('study', 'replay'), proofs, strict=True):
        require(proof['status'] == 'ok' and proof['mode'] == mode and proof['network_blocked'] is True and
                proof['prepared_sha256'] == file_sha(DIRECTORY/'prepared.json'), 'Worker completion identity differs')
        for name, digest in proof['output_sha256'].items(): require(file_sha(DIRECTORY/mode/name) == digest, 'Worker output changed')
        require(read(DIRECTORY/mode/'command.json')['returncode'] == 0, 'Worker terminal status differs')
    require(proofs[0]['reference_rows'] == proofs[1]['reference_rows'] == 30924 and
            proofs[0]['output_sha256'] == proofs[1]['output_sha256'], 'Longest component byte replay differs')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study'/OUTPUTS[0]),
                                  pd.read_parquet(DIRECTORY/'replay'/OUTPUTS[0]), check_exact=True)
    doc = {'status': 'match', 'differences': 0, 'fresh_processes': 2, 'network_blocked': True,
        'reference_rows_per_mode': 30924, 'native_windows_per_policy': 15462,
        'prepared_sha256': file_sha(DIRECTORY/'prepared.json'), 'output_sha256': proofs[0]['output_sha256'],
        'new_trading_backtests': 0, 'formal_source_research_acceptance': False}
    save(target, doc); return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'worker', 'offline'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    result = worker(args.mode) if args.action == 'worker' else run('study') if args.action == 'study' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
