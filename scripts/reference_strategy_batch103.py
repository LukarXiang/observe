"""Freeze ETF6/81 inputs and longest native price expressions; no original signals/trades."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from scripts import research_strategy_batch103 as study
from scripts.prepare_strategy_batch101 import normalize
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.research_strategy_batch100 import SNAPSHOT, SNAPSHOT_SHA
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch103/20261011-ETF-original-price-stage')
PARENT = Path('docs/handoff/2026-10-11-batch102-final-review.json')
REGISTRIES = (
    'data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/results.json',
    'data/staging/strategies-batch75/20261008-ETF-info-rotation/probe-results.json',
    'data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight/probe-results.json',
    'data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probe-results.json',
    'data/staging/strategies-batch81/20261009-stock-bond-drawdown/probe-results.json')
OUTPUTS = ('diagnostics.json', 'real-price-reference.parquet', 'momentum-price-reference.parquet', 'summary.json')
TESTS = ('tests/unit/test_batch103_research.py', 'tests/unit/test_batch103_reference.py')


def candidates():
    """Bounded original registries only, excluding copied verification trees."""
    symbols = set(study.REAL_POOL) | set(study.MOMENTUM_POOL) | {'511880.XSHG'}
    frames = {}; bindings = {}; evidence = {}; rows_seen = []
    for registry in REGISTRIES:
        doc = read(registry); evidence[registry] = file_sha(registry)
        for row in doc['results']:
            if row.get('status') != 'success': continue
            query = row.get('query', {})
            native_symbol = query.get('parameters', {}).get('symbol', row.get('symbol'))
            code = str(native_symbol)[-6:]
            matching = [s for s in symbols if s[:6] == code]
            if not matching: continue
            is_fund = (query.get('function') == 'fund_etf_hist_sina' or
                row['case'].startswith('fund-') or 'ETF-info-rotation' in registry)
            if not is_fund: continue
            require(len(matching) == 1 and native_symbol in ('sh'+code, 'sz'+code), 'Candidate identity differs')
            if query: require(query.get('function') == 'fund_etf_hist_sina', 'Candidate API differs')
            symbol = matching[0]
            require(native_symbol == ('sz' if symbol.endswith('XSHE') else 'sh')+code,
                    'Candidate exchange differs')
            require(symbol not in frames and file_sha(row['file']) == row['sha256'], 'Duplicate or changed candidate')
            evidence[row['file']] = row['sha256']
            for wire in row['wire_responses']:
                require(wire['status_code'] == 200 and file_sha(wire['file']) == wire['sha256'], 'Wire differs')
                evidence[wire['file']] = wire['sha256']
            raw = pd.read_parquet(row['file']); frame, excluded = normalize(raw)
            require(len(raw) == row['rows'], 'Native candidate row count differs')
            frames[symbol] = frame
            bindings[symbol] = {'registry': registry, 'case': row['case'], 'file': row['file'],
                'sha256': row['sha256'], 'native_rows': len(raw), 'future_excluded': excluded,
                'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'retained_rows': len(frame),
                'native_symbol': native_symbol, 'query': query or None, 'strict_usable': False}
            rows_seen.append({'registry': registry, 'case': row['case'], 'instrument': symbol})
    return frames, bindings, evidence, rows_seen


def inventory():
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA and
        state['tables'] == store.published()['tables'], 'Actual snapshot/publication changed')
    symbols = sorted({s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for s in
        (*study.REAL_POOL, *study.MOMENTUM_POOL, '511880.XSHG', '000300.XSHG',
         '399632.XSHE', '000016.XSHG', '000010.XSHG', '000015.XSHG', '399324.XSHE',
         '399006.XSHE', '000905.XSHG', '399673.XSHE', '399001.XSHE')})
    counts = {}
    with duckdb.connect() as con:
        for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
            paths = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
            column = 'index' if table == 'index_1d' else 'instrument'; rows = dict.fromkeys(symbols, 0)
            if paths:
                marks = ','.join('?' for _ in symbols)
                rows.update(dict(con.execute(f'SELECT "{column}", count(*) FROM read_parquet(?, union_by_name=true) '
                    f'WHERE "{column}" IN ({marks}) GROUP BY "{column}"', [paths, *symbols]).fetchall()))
            counts[table] = rows
    calendars = {}
    for entry in state['tables']['calendar'].values():
        name = str(Path('data')/entry['file'])
        require(fingerprint(pd.read_parquet(name)) == entry['sha'], 'Calendar logical SHA differs')
        calendars[name] = file_sha(name)
        if entry.get('file_sha256'): require(calendars[name] == entry['file_sha256'], 'Calendar bytes differ')
    cal = store.load_state(state, 'calendar')
    days = pd.to_datetime(cal.loc[cal.is_open, 'date']).dt.strftime('%Y-%m-%d').tolist()
    require(days == sorted(set(days)), 'Open calendar invalid')
    frames, bindings, evidence, matches = candidates()
    return {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
        'published_batch': store.published()['batch_id'], 'active_table_counts': counts,
        'calendar': [d for d in days if d <= '2026-09-29'], 'calendar_byte_sha256': calendars,
        'candidate_bindings': bindings, 'scanned_registries': list(REGISTRIES), 'matched_registry_rows': matches,
        'real_pool_candidates': sorted(set(frames)&set(study.REAL_POOL)),
        'real_pool_missing_selected_candidates': sorted(set(study.REAL_POOL)-set(frames)),
        'momentum_pool_candidates': sorted(set(frames)&set(study.MOMENTUM_POOL)),
        'momentum_pool_missing_selected_candidates': sorted(set(study.MOMENTUM_POOL)-set(frames)),
        'evidence_sha256': {**evidence, **calendars}, 'strict_usable': False,
        'absence_scope': 'Actual active snapshot tables and five selected original registries; not whole-machine absence',
        'new_requests': 0, 'new_responses': 0, 'published': False,
        'no_repeat_network_evidence': ['docs/handoff/2026-10-11-batch102-final-review.json',
            'docs/handoff/2026-10-11-batch99-final-review.json'],
        'missing_dependencies': ['full original fund pools and historical identity/events/adjustment/status',
            '11:30 and 14:50 actual current prices, dynamic heat reference volume with original history semantics',
            '14:40/14:50 fund execution, fees, cash/position/order feedback',
            'ETF81 pandas compatibility and nondeterministic set/tie semantics unresolved']}, frames


def implementation_files():
    paths = set(TESTS) | {'scripts/reference_strategy_batch103.py', 'pyproject.toml', 'uv.lock'}
    root = Path.cwd().resolve()
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__name__', '')
        file = getattr(module, '__file__', None)
        if file and (name.startswith('observe') or name.startswith('scripts')):
            path = Path(file).resolve()
            require(path.is_relative_to(root) and path.suffix == '.py', 'Actual import outside project')
            paths.add(path.relative_to(root).as_posix())
    return sorted(paths)


def prepare():
    require(not DIRECTORY.exists(), 'Immutable Batch103 directory exists')
    parent = read(PARENT); policy = read('configs/research_metadata.json')
    require(parent['status'] == 'ok' and parent['formal_source_research_acceptance'] and
        parent['progress']['manually_reviewed'] == 290 and policy['accepted_receipt'] == str(PARENT) and
        policy['accepted_checkpoint'] == parent['progress']['checkpoint'], 'Accepted parent changed')
    for name, sha in parent['evidence_sha256'].items(): require(file_sha(name) == sha, 'Parent evidence changed: '+name)
    for number in (0, 1): study.tree(number)
    inputs, frames = inventory(); files = implementation_files()
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch103 full source/input stage started;formal290/trading12/parents87 unchanged')
        for name in files:
            target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        for number in (0, 1):
            target = DIRECTORY/'sources'/f'{number}.source'; target.parent.mkdir(exist_ok=True); shutil.copyfile(study.PREREAD/f'{number}.source', target)
        for symbol, frame in frames.items():
            target = DIRECTORY/'inputs'/(symbol+'.parquet'); target.parent.mkdir(exist_ok=True); frame.to_parquet(target, index=False)
        save(DIRECTORY/'dependencies.json', inputs)
        evidence = {**inputs['evidence_sha256'], str(PARENT): file_sha(PARENT),
            str(DIRECTORY/'dependencies.json'): file_sha(DIRECTORY/'dependencies.json')}
        for number in (0, 1):
            for path in (study.PREREAD/f'{number}.source', Path('repo/量化策略源代码')/study.SOURCES[number], DIRECTORY/'sources'/f'{number}.source'):
                evidence[str(path)] = study.SOURCE_SHA[number]
        evidence.update({str(p): file_sha(p) for p in (DIRECTORY/'inputs').glob('*.parquet')})
        save(DIRECTORY/'prepared.json', {'parent': str(PARENT), 'source_sha256': list(study.SOURCE_SHA),
            'evidence_sha256': evidence, 'implementation_sha256': {n: file_sha(n) for n in files},
            'source_research_formally_accepted': False, 'new_trading_backtests': 0, 'strict_usable': False})
    return {'status': 'prepared', 'candidate_funds': len(frames), 'frozen_implementation_files': len(files)}


def validate():
    doc = read(DIRECTORY/'prepared.json')
    require(doc['source_sha256'] == list(study.SOURCE_SHA) and doc['new_trading_backtests'] == 0 and not doc['strict_usable'], 'Scope differs')
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Frozen input changed: '+name)
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == sha, 'Implementation changed: '+name)
    return doc


def expressions(number, function, names):
    fn = next(n for n in study.tree(number).body if isinstance(n, ast.FunctionDef) and n.name == function)
    selected = {}
    for name in names:
        nodes = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and
                 len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        if number == 1 and name == 'R':
            require(len(nodes) == 2 and sum(ast.unparse(n) == 'R + random.random() / 1000' for n in nodes) == 1,
                    'Original score/tie assignments changed')
            nodes = [n for n in nodes if ast.unparse(n) != 'R + random.random() / 1000']
        require(len(nodes) == 1, 'Original expression ambiguous: '+name)
        selected[name] = compile(ast.Expression(nodes[0]), '<unchanged-source-expression>', 'eval')
    return selected


def price_references(frames, calendar):
    real_code = expressions(0, 'get_signal', ('cp_increase', 'ma_n1', 'pre_price'))
    momentum_code = expressions(1, 'my_trade', ('his_n_price', 'R', 'base_price'))
    rows = {'real': [], 'momentum': []}; profiles = {}
    for symbol, frame in sorted(frames.items()):
        values = frame.close.to_numpy(float); observed = set(frame.date)
        missing = sorted(d for d in calendar if frame.date.iloc[0] <= d <= frame.date.iloc[-1] and d not in observed)
        profiles[symbol] = {'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
            'missing_open_dates': missing, 'calendar_extra': sorted(observed-set(calendar)), 'year_windows': {}}
        modes = [('real', 13)] if symbol in study.REAL_POOL else []
        if symbol in study.MOMENTUM_POOL: modes.append(('momentum', 30))
        for mode, history_count in modes:
            profiles[symbol]['year_windows'][mode] = frame.date.iloc[history_count:].str[:4].value_counts().sort_index().to_dict()
            for end in range(history_count, len(frame)):
                previous = values[end-history_count:end]; current = values[end]
                result = {'instrument': symbol, 'window_first': frame.date.iloc[end-history_count],
                    'asof_native_close': frame.date.iloc[end], 'history_rows': history_count,
                    'current_is_daily_close_proxy': True,
                    'missing_open_count': sum(frame.date.iloc[end-history_count] <= d <= frame.date.iloc[end] for d in missing)}
                if mode == 'real':
                    ns = {'current_price': current, 'close_data': {'close': previous}, 'g': SimpleNamespace(lag1=13, lag2=13)}
                    for name, code in real_code.items(): ns[name] = eval(code, ns)
                    mean = math.fsum(previous)/13; increase = (current/previous[0]-1)*100
                    deviation = (current/mean-1)*100
                    result.update(cp_increase=float(ns['cp_increase']), MA13=float(ns['ma_n1']), deviation=float(ns['pre_price']),
                        independent_cp=increase, independent_MA13=mean, independent_deviation=deviation,
                        price_eligible_reference=bool(ns['cp_increase'] >= .1 and ns['pre_price'] >= 0),
                        independent_eligible=bool(increase >= .1 and deviation >= 0))
                    require(math.isclose(ns['ma_n1'], mean, rel_tol=1e-12, abs_tol=1e-12) and
                        math.isclose(ns['pre_price'], deviation, rel_tol=1e-12, abs_tol=1e-10) and ns['cp_increase'] == increase,
                        'Independent real price arithmetic differs')
                else:
                    ns = {'current_price': current, 'curr_price': current, 'close_data1': previous, 'N': 21}
                    for name, code in momentum_code.items(): ns[name] = eval(code, ns)
                    old = math.fsum(previous[7:10]); new = math.fsum([previous[28], previous[29], current])
                    rank = (new/old-1)*100; base = previous[9]
                    result.update(old_mean=float(ns['his_n_price']), score=float(ns['R']), base_price=float(ns['base_price']),
                        independent_old_mean=old/3, independent_score=rank, independent_base_price=float(base),
                        entry_reference=bool(current > ns['base_price']*1.001), exit_reference=bool(current < ns['base_price']*.999))
                    require(math.isclose(ns['R'], rank, rel_tol=1e-12, abs_tol=1e-10) and
                        math.isclose(ns['his_n_price'], old/3, rel_tol=1e-12, abs_tol=1e-12) and ns['base_price'] == base,
                        'Independent momentum arithmetic differs')
                rows[mode].append(result)
    return pd.DataFrame(rows['real']), pd.DataFrame(rows['momentum']), profiles


def worker(mode):
    require(mode in ('compute', 'replay'), 'Unknown mode'); validate()
    target = DIRECTORY/mode; require(not target.exists(), 'Immutable worker directory exists'); target.mkdir()
    with network_blocked():
        inputs = read(DIRECTORY/'dependencies.json')
        frames = {s: pd.read_parquet(DIRECTORY/'inputs'/(s+'.parquet')) for s in inputs['candidate_bindings']}
        real, momentum, profiles = price_references(frames, inputs['calendar']); diagnostics = study.diagnostics()
        real.to_parquet(target/OUTPUTS[1], index=False); momentum.to_parquet(target/OUTPUTS[2], index=False)
        save(target/OUTPUTS[0], diagnostics)
        summary = {'diagnostics': diagnostics['case_count'], 'real14_row_price_windows': len(real),
            'momentum31_row_price_windows': len(momentum), 'candidate_profiles': profiles,
            'real_gap_crossing_windows': int(real.missing_open_count.gt(0).sum()),
            'momentum_gap_crossing_windows': int(momentum.missing_open_count.gt(0).sum()),
            'real_price_eligibility_numeric_disagreements': int(real.price_eligible_reference.ne(real.independent_eligible).sum()),
            'momentum_max_independent_score_error': float((momentum.score-momentum.independent_score).abs().max()),
            'network_blocked': True, 'full_original_pool_signals': 0, 'new_trading_backtests': 0,
            'formal_source_research_acceptance': False, 'original_strategy_complete': False, 'strict_usable': False}
        save(target/OUTPUTS[3], summary)
    return summary


def run():
    validate(); require(not (DIRECTORY/'offline-verification.json').exists(), 'Immutable verification exists')
    commands = []
    for mode in ('compute', 'replay'):
        env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONHASHSEED'] = '103'
        result = subprocess.run([sys.executable, '-B', '-m', 'scripts.reference_strategy_batch103', 'worker', '--mode', mode],
            capture_output=True, text=True, env=env)
        log = DIRECTORY/(mode+'.log'); log.write_text(result.stdout+result.stderr)
        entry = {'mode': mode, 'returncode': result.returncode, 'log': str(log), 'log_sha256': file_sha(log)}
        commands.append(entry); save(DIRECTORY/(mode+'-command.json'), entry)
        require(result.returncode == 0, 'Reference failed; failed evidence retained')
    differences = [n for n in OUTPUTS if file_sha(DIRECTORY/'compute'/n) != file_sha(DIRECTORY/'replay'/n)]
    doc = {'status': 'ok' if not differences else 'error', 'differences': differences, 'commands': commands,
        'outputs_sha256': {str(DIRECTORY/'compute'/n): file_sha(DIRECTORY/'compute'/n) for n in OUTPUTS},
        'fresh_network_blocked_processes': 2, 'hash_seed_for_diagnostics_only': 103,
        'full_original_pool_signals': 0, 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', doc); require(not differences, 'Replay differs'); return doc


def set_trials():
    validate(); target = DIRECTORY/'actual-set-trials'; require(not target.exists(), 'Immutable set trials exist'); target.mkdir()
    outputs = []
    for seed in ('1', '2', '3', '4'):
        env = os.environ.copy(); env['PYTHONHASHSEED'] = seed; env['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run([sys.executable, '-B', '-c',
            'import json; from scripts.recover_strategy_batch94_reference import network_blocked; '
            'from scripts.research_strategy_batch103 import actual_set_trial; '
            '\nwith network_blocked(): print(json.dumps(actual_set_trial(),allow_nan=False))'],
            env=env, capture_output=True, text=True)
        (target/(seed+'.log')).write_text(result.stdout+result.stderr)
        save(target/(seed+'-command.json'), {'hash_seed': seed, 'returncode': result.returncode,
            'log_sha256': file_sha(target/(seed+'.log'))})
        require(result.returncode == 0, 'Actual source set trial failed; retained')
        outputs.append({'hash_seed': seed, **json.loads(result.stdout)})
    doc = {'trials': outputs, 'unique_iteration_orders': len({tuple(x['iteration']) for x in outputs}),
        'unique_winners': sorted({x['position']['ETF_HOLD'] for x in outputs}),
        'source_set_retained': True, 'controlled_equal_prices_and_draw_only': True,
        'production_seed_or_sort_fix_approved': False, 'original_signal_or_backtest': False}
    save(target/'results.json', doc); return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'set_trials', 'protect'))
    parser.add_argument('--mode', choices=('compute', 'replay')); args = parser.parse_args()
    if args.action == 'worker': result = worker(args.mode)
    elif args.action == 'protect': validate(); result = protect('data', DIRECTORY)
    else: result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
