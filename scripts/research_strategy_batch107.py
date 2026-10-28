"""ETF27/ETF61 literal diagnostics and longest native operands; no ETF trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import warnings

import duckdb
import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch106 import implementation_files, network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch107/20261011-ETF-regression-means-v1')
PARENT = Path('docs/handoff/2026-10-11-batch106-final-review.json')
PARENT_SHA = 'c3f6e8e4269b9f0b63005be819a4c183c41f526f50e283517ad2ad80fa933c76'
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
SOURCES = ('2023年度精选策略/27.从相关系数角度探讨etf轮动的可行性.txt',
    '2024年度精选策略2/61.简单ETF策略，年化97%.txt')
SOURCE_SHA = ('0e60cde5dfd8f828dc872be2d44aa950c80b22ebc82b395522906ac291eff3fc',
    '9e47ff904080216ab711f70c59b5c514c54eab2585b07e18e4d03c7f658ef63c')
POOL = ('510180.XSHG', '159915.XSHE', '513100.XSHG')
REGISTRIES = (
    ('data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight/probe-results.json', 'fund-510180'),
    ('data/staging/strategies-batch75/20261008-ETF-info-rotation/probe-results.json', '159915'),
    ('data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probe-results.json', 'fund-513100'))
PROBE = Path('data/staging/strategies-batch107/20261011-original-pool-probe-v1/checked.json')
FILES = ('scripts/research_strategy_batch107.py', 'scripts/probe_strategy_batch107_pool.py',
    'tests/unit/test_batch107_research.py')
OUTPUTS = ('diagnostics.json', 'regression-operands.parquet', 'ranking-operands.parquet', 'summary.json')


def source_path(number, directory=DIRECTORY):
    require(number in (0, 1), 'Unknown source')
    target = directory/'sources'/f'{number}.source'
    path = target if target.exists() else Path('repo/量化策略源代码')/SOURCES[number]
    require(file_sha(path) == SOURCE_SHA[number], 'Original source bytes changed')
    return path


def literal(number, namespace=None, directory=DIRECTORY):
    path = source_path(number, directory)
    nodes = [n for n in ast.parse(read_source(path)[0]).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (13 if number == 0 else 12), 'Full definition count differs')
    for n in nodes:
        require(not n.decorator_list and not n.returns and not n.type_params and
            all(a.annotation is None for a in (*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs)) and
            all(isinstance(d, ast.Constant) for d in n.args.defaults) and
            all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults), 'Unsafe function definition')
    ns = {'g': SimpleNamespace(), 'np': np,
        'log': SimpleNamespace(info=lambda *a: None, debug=lambda *a: None, set_level=lambda *a: None)}
    ns.update(namespace or {})
    # Imports, SMTP and original top-level code are not executed.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def initialize(number, directory=DIRECTORY):
    calls = []
    ns = literal(number, {'set_benchmark': lambda s: calls.append(['benchmark', s]),
        'set_option': lambda *a: calls.append(['option', list(a)]),
        'set_slippage': lambda s: calls.append(['slippage', s]), 'FixedSlippage': lambda s: s,
        'set_order_cost': lambda v, **k: calls.append(['cost', v, k]), 'OrderCost': lambda **k: k,
        'run_daily': lambda f, **k: calls.append(['daily', f.__name__, k])}, directory)
    context = SimpleNamespace(current_dt=dt.datetime(2022, 1, 4),
        portfolio=SimpleNamespace(starting_cash=100000., cash=100000., positions={}))
    with redirect_stdout(StringIO()):
        ns['initialize'](context)
    return ns, context, calls


def score_function(directory=DIRECTORY):
    """Compile the original regression assignments with data injected as an operand."""
    path = source_path(0, directory)
    tree = ast.parse(read_source(path)[0])
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_rank')
    loop = next(n for n in definition.body if isinstance(n, ast.For))
    assignments = loop.body[1:-1]
    require(len(assignments) == 6 and all(isinstance(n, ast.Assign) for n in assignments), 'Regression statements differ')
    function = ast.FunctionDef(name='operand', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='data')],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), decorator_list=[],
        body=assignments+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=s, ctx=ast.Load()) for s in
            ('slope', 'intercept', 'annualized_returns', 'r_squared', 'score')], ctx=ast.Load()))], type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    ns = {'np': np, 'math': math}
    exec(compile(module, str(path), 'exec'), ns)
    return ns['operand']


def independent_score(values):
    require(len(values) == 29 and all(math.isfinite(float(v)) and v > 0 for v in values), 'Invalid29-row operand')
    y = [math.log(float(v)) for v in values]
    mean = math.fsum(y)/29
    variance = math.fsum((v-mean)**2 for v in y)
    if variance == 0:
        return None
    slope = math.fsum((k-14)*(v-mean) for k, v in enumerate(y))/math.fsum((k-14)**2 for k in range(29))
    intercept = mean-14*slope
    r2 = 1-math.fsum((v-(slope*k+intercept))**2 for k, v in enumerate(y))/variance
    return slope, intercept, math.expm1(slope*250), r2, math.expm1(slope*250)*r2


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return {'nonfinite': repr(value)}
    return value


def diagnostics(directory=DIRECTORY):
    rows = []
    def case(name, callback):
        with redirect_stdout(StringIO()), warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter('always')
            try: value, error = clean(callback()), None
            except Exception as exc: value, error = None, {'type': type(exc).__name__, 'message': str(exc)}
        rows.append({'case': name, 'value': value, 'error': error, 'warnings': [str(n.message) for n in notes]})
    case('27_initial_pool_fees_schedule', lambda: initialize(0, directory)[2])
    case('61_initial_pool_budget_schedule', lambda: {'pool': initialize(1, directory)[0]['g'].all_stks,
        'budget': initialize(1, directory)[0]['g'].money, 'calls': initialize(1, directory)[2]})
    def rank_case(slopes, missing_math=False):
        ns, _, _ = initialize(0, directory)
        ns['attribute_history'] = lambda stock, *a: pd.DataFrame({'close': np.exp(np.arange(29)*slopes[POOL.index(stock)])})
        if not missing_math: ns['math'] = math
        return ns['get_rank'](['IGNORED'])
    case('27_global_pool_ignores_argument', lambda: rank_case([.002, .001, -.001]))
    case('27_all_negative_still_selects', lambda: rank_case([-.003, -.002, -.001]))
    case('27_tied_scores_original_pool_order', lambda: rank_case([.001]*3))
    case('27_math_implicit_failure', lambda: rank_case([.001]*3, True))
    case('27_constant_window_original_nonfinite', lambda: score_function(directory)(pd.DataFrame({'close': np.ones(29)*100.})))
    ns, context, _ = initialize(0, directory)
    current = {s: SimpleNamespace(paused=False, is_st=False, name='ETF', high_limit=11., low_limit=9.) for s in POOL}
    ns['get_current_data'] = lambda: current
    def limit(up, held=False, series=False):
        context.portfolio.positions = {POOL[0]: SimpleNamespace()} if held else {}
        value = 11. if up else 9.
        ns['history'] = lambda *a, **k: {s: pd.Series([value], index=pd.to_datetime(['2022-01-03'])) if series else np.array([value]) for s in POOL}
        return ns['filter_limitup_stock' if up else 'filter_limitdown_stock'](context, [POOL[0]])
    for up, held, name in [(True, False, 'new_up_equal'), (True, True, 'held_up_equal'),
            (False, False, 'new_down_equal'), (False, True, 'held_down_equal')]:
        case('27_'+name, lambda up=up, held=held: limit(up, held))
    case('27_current_series_negative_index', lambda: limit(True, series=True))
    def filtered_top():
        calls = []
        context.portfolio.positions = {}
        ns['get_rank'] = lambda *a: [POOL[0]]
        current[POOL[0]].paused = True
        ns['history'] = lambda *a, **k: {s: np.array([10.]) for s in POOL}
        ns['adjust_position'] = lambda c, selected: calls.append(selected)
        ns['my_trade'](context)
        current[POOL[0]].paused = False
        return calls
    case('27_filtered_top_no_second_choice', filtered_top)
    def missing_position():
        ns, ctx, _ = initialize(0, directory)
        return ns['adjust_position'](ctx, [POOL[0]])
    case('27_missing_position_key', missing_position)
    def opening(filled):
        ns = literal(0, {'order_target_value': lambda *a: SimpleNamespace(filled=filled)}, directory)
        return ns['open_position'](POOL[0], 10000.)
    for filled in (0, 1): case('27_open_filled_'+str(filled), lambda filled=filled: opening(filled))
    def closing(status, filled, amount):
        ns = literal(0, {'OrderStatus': SimpleNamespace(held='held'),
            'order_target_value': lambda *a: SimpleNamespace(status=status, filled=filled, amount=amount)}, directory)
        return ns['close_position'](SimpleNamespace(security=POOL[0]))
    for status, filled, amount, name in [('held', 100, 100, 'complete'), ('held', 50, 100, 'partial'), ('cancelled', 100, 100, 'cancelled')]:
        case('27_close_'+name, lambda status=status, filled=filled, amount=amount: closing(status, filled, amount))
    def stop(price):
        orders = []
        ns, ctx, _ = initialize(0, directory)
        ctx.portfolio.positions = {POOL[0]: SimpleNamespace(security=POOL[0], avg_cost=10., price=price, value=100., total_amount=100)}
        ns['order_target_value'] = lambda *a: orders.append(list(a))
        ns['check_lose'](ctx)
        return orders
    case('27_stop_at_minus80', lambda: stop(2.))
    case('27_stop_above_minus80', lambda: stop(2.001))
    def mean(n, ref, returned=None):
        ns, _, _ = initialize(1, directory)
        ns['g'].security = '159608.XSHE'
        calls = []
        def history(**k):
            calls.append(k)
            return pd.DataFrame({'close': np.arange(returned if returned is not None else k['count'])+1.})
        ns['attribute_history'] = history
        return {'mean': ns['ma'](n=n, ref=ref), 'requests': calls}
    case('61_ma60_ref2_uses_first60_of62', lambda: mean(60, 2))
    case('61_ma60_missing_history_not_blocked', lambda: mean(60, 2, 3))
    def crossing(current_x, current_y, old_x, old_y):
        ns = literal(1, directory=directory)
        ns['ma'] = lambda typ, n, ref, unit: {(20, 0): current_x, (10, 0): current_y, (20, 1): old_x, (10, 1): old_y}[n, ref]
        return ns['cross'](20, 10)
    case('61_sell_current_equality_previous_less', lambda: crossing(10., 10., 9., 10.))
    case('61_sell_previous_equality_no_cross', lambda: crossing(11., 10., 10., 10.))
    def buy_case(paused=False, held=False, full=False):
        ns, ctx, _ = initialize(1, directory)
        symbol = ns['g'].all_stks[1]
        ns['g'].stks_buy = [symbol]
        ns['get_current_data'] = lambda: {symbol: SimpleNamespace(paused=paused)}
        ns['get_name'] = lambda s: s
        ns['order_value'] = lambda *a: None
        if held: ctx.portfolio.positions = {symbol: SimpleNamespace(total_amount=100)}
        elif full: ctx.portfolio.positions = {'other': SimpleNamespace(total_amount=100)}
        ns['buy'](ctx)
        return 'returned_without_order'
    case('61_dict_has_key_buy_failure', buy_case)
    case('61_paused_skip_before_has_key', lambda: buy_case(paused=True))
    case('61_held_skip_before_has_key', lambda: buy_case(held=True))
    case('61_full_slot_skip_before_has_key', lambda: buy_case(full=True))
    def sell_case():
        ns, ctx, _ = initialize(1, directory)
        symbol = ns['g'].all_stks[1]
        ns['g'].stks_sell = [symbol]
        ctx.portfolio.positions = {symbol: SimpleNamespace(total_amount=100)}
        ns['order_target'] = lambda *a: None
        ns['get_name'] = lambda s: s
        return ns['sell'](ctx)
    case('61_dict_has_key_sell_failure', sell_case)
    return {'case_count': len(rows), 'cases': rows, 'orders_are_intents_only': True,
        'economic_rules_changed': False, 'new_trading_backtests': 0, 'formal_source_research_acceptance': False}


def selection(catalog):
    rows = []
    major = {'get_fundamentals', 'get_fundamentals_continuously', 'run_query', 'get_factor_values',
        'get_index_stocks', 'get_all_securities', 'get_industry_stocks', 'get_concept_stocks',
        'get_money_flow', 'get_dominant_future', 'read_file', 'get_valuation'}
    for r in catalog:
        if r['review_status'] == '人工审查完成': continue
        path = Path('repo/量化策略源代码')/r['path']
        require(file_sha(path) == r['bytes_sha256'], 'Unreviewed source changed')
        text = read_source(path)[0]
        error = None
        calls = set()
        reachable = set()
        classes = []
        try:
            start = r.get('code_start_line') or 1
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tree = ast.parse('\n'*(start-1)+'\n'.join(text.splitlines()[start-1:]))
            defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
            def names(node):
                return {c.func.id if isinstance(c.func, ast.Name) else c.func.attr for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, (ast.Name, ast.Attribute))}
            calls = names(tree)
            reachable = set(defs)&{'initialize', 'handle_data', 'before_trading_start', 'after_trading_end'}
            for c in ast.walk(tree):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in ('run_daily', 'run_weekly', 'run_monthly') and c.args and isinstance(c.args[0], ast.Name):
                    reachable.add(c.args[0].id)
            previous = set()
            while previous != reachable:
                previous = set(reachable)
                reachable |= {v for k in previous if k in defs for v in names(defs[k]) if v in defs}
            direct = {v for k in reachable if k in defs for v in names(defs[k])}
        except SyntaxError as exc:
            error = {'line': exc.lineno, 'message': exc.msg}
            direct = set()
        rows.append({'path': r['path'], 'bytes_sha256': r['bytes_sha256'], 'syntax_error': error,
            'module_dependency_calls': sorted(calls&major), 'reachable_free_function_candidates': sorted(reachable),
            'direct_free_function_dependency_calls': sorted(direct&major), 'classes_not_resolved': classes,
            'selected_for_full_review': r['path'] in SOURCES})
    require(len(rows) == 397, 'Remaining scope differs')
    return {'remaining_sources': len(rows), 'rows': rows, 'full_source_reviews': list(SOURCES),
        'scope': 'Free-function candidate graph only; dynamic dispatch/classes/imports and branch reachability are unresolved',
        'parse_failure_policy': 'Keep for manual reading; never infer dependency-free or tradeable'}


def runtime_binding(directory):
    doc = read(directory/'prepared.json')
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(directory/'implementation'/name) == sha, 'Frozen runtime differs: '+name)
    for name, sha in doc['input_sha256'].items(): require(file_sha(name) == sha, 'Frozen input changed: '+name)
    require(file_sha(PARENT) == PARENT_SHA and file_sha('data/snapshots/'+SNAPSHOT+'.json') == SNAPSHOT_SHA, 'Parent/snapshot changed')
    return doc


def prepare(directory=DIRECTORY):
    require(not directory.exists() and file_sha(PARENT) == PARENT_SHA, 'Stage exists or parent changed')
    parent = read(PARENT)
    policy = read('configs/research_metadata.json')
    require(parent['formal_source_research_acceptance'] and policy['accepted_receipt'] == str(PARENT) and
        policy['accepted_checkpoint'] == parent['progress']['checkpoint'], 'Formal parent differs')
    for name, sha in parent['evidence_sha256'].items(): require(file_sha(name) == sha, 'Parent evidence changed: '+name)
    require(file_sha('data/snapshots/'+SNAPSHOT+'.json') == SNAPSHOT_SHA, 'Snapshot changed')
    catalog = Path('data/catalog/strategies')/parent['progress']['catalog_id']/'catalog.json'
    require(file_sha(catalog) == parent['progress']['catalog_sha256'], 'Catalog changed')
    selected = selection(read(catalog))
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', directory, 'Batch107 original27/61 full review and397-source free-function screening;formal298/trading12/parents87 unchanged')
        save(directory/'selection.json', selected)
        (directory/'sources').mkdir()
        review = []
        for number, name in enumerate(SOURCES):
            path = source_path(number, directory)
            target = directory/'sources'/f'{number}.source'
            shutil.copyfile(path, target)
            text, encoding = read_source(target)
            tree = ast.parse(text)
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            review.append({'path': name, 'sha256': file_sha(target), 'lines': len(text.splitlines()),
                'encoding': encoding, 'definitions': [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno,
                    'ast': ast.dump(n, include_attributes=False)} for n in nodes]})
        save(directory/'source-review.json', {'sources': review, 'complete_originals_copied': True,
            'top_level_imports_not_executed': True, 'formal_source_research_acceptance': False})
        (directory/'inputs').mkdir()
        bindings = {}
        for symbol, (name, case) in zip(POOL, REGISTRIES, strict=True):
            doc = read(name)
            matching = [r for r in doc['results'] if r['case'] == case]
            require(len(matching) == 1 and matching[0]['status'] == 'success', 'Candidate missing')
            row = matching[0]
            require(file_sha(row['file']) == row['sha256'], 'Native candidate changed')
            for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Original response changed')
            raw = pd.read_parquet(row['file'])
            require(len(raw) == row['rows'], 'Candidate row count changed')
            frame = pd.DataFrame({'date': pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d'), 'close': pd.to_numeric(raw.close, errors='raise').astype(float)})
            require(not frame.date.duplicated().any() and frame.date.is_monotonic_increasing and
                frame.date.notna().all() and np.isfinite(frame.close).all() and (frame.close > 0).all(), 'Invalid native candidate')
            frame = frame.loc[frame.date.le('2026-09-29')].reset_index(drop=True)
            target = directory/'inputs'/(symbol+'.parquet')
            frame.to_parquet(target, index=False)
            bindings[symbol] = {'registry': name, 'registry_sha256': file_sha(name), 'original_candidate': row,
                'input': str(target), 'input_sha256': file_sha(target), 'rows': len(frame),
                'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'future_excluded': len(raw)-len(frame), 'strict_usable': False}
        store = Store('data')
        state = store.state(SNAPSHOT)
        require(state['tables'] == store.published()['tables'], 'Frozen/current publication differs')
        calendar = store.load_state(state, 'calendar')
        calendar.to_parquet(directory/'inputs/calendar.parquet', index=False)
        pool61 = initialize(1, directory)[0]['g'].all_stks
        symbols = tuple(s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for s in (*POOL, *pool61, '510300.XSHG'))
        counts = {}
        with duckdb.connect() as con:
            for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
                paths = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
                col = 'index' if table == 'index_1d' else 'instrument'
                counts[table] = dict.fromkeys(symbols, 0)
                if paths:
                    counts[table].update({s: int(n) for s, n in con.execute(f'SELECT "{col}",count(*) FROM read_parquet(?,union_by_name=true) WHERE "{col}" IN ({",".join("?" for _ in symbols)}) GROUP BY "{col}"', [paths, *symbols]).fetchall()})
        save(directory/'dependencies.json', {'candidate_bindings': bindings, 'active_table_counts': counts,
            'pool61': pool61, 'pool61_existing_native_candidates': [], 'probe': str(PROBE), 'probe_sha256': file_sha(PROBE),
            'probe_results': read(PROBE), 'original61_pool_reduced': False, 'index399306_tradeability_proven': False,
            'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA, 'published_batch': state['batch_id'],
            'calendar_source_partitions': list(state['tables']['calendar'].values()), 'original_fund_execution_admitted': False})
        files = implementation_files()
        files.update({name: file_sha(name) for name in FILES})
        for name in files:
            target = directory/'implementation'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        paths = [directory/'dependencies.json', directory/'source-review.json', directory/'selection.json',
            *sorted((directory/'inputs').glob('*')), *sorted((directory/'sources').glob('*'))]
        save(directory/'prepared.json', {'parent': str(PARENT), 'parent_sha256': PARENT_SHA,
            'catalog': str(catalog), 'catalog_sha256': file_sha(catalog), 'implementation_sha256': files,
            'input_sha256': {str(p): file_sha(p) for p in paths}, 'formal_source_research_acceptance': False,
            'formal_trading_acceptance': False})
    return {'status': 'prepared', 'native_candidates': 3, 'implementation_copies': len(files)}


def operands(inputs, calendar, original):
    days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d').tolist()
    positions = {d: k for k, d in enumerate(days)}
    rows = []
    profiles = {}
    maximum = dict.fromkeys(('slope', 'intercept', 'annual_return', 'r_squared', 'score'), 0.)
    for symbol in POOL:
        frame = inputs[symbol]
        missing = sorted(set(d for d in days if frame.date.iloc[0] <= d <= frame.date.iloc[-1])-set(frame.date))
        invalid = 0
        for k in range(28, len(frame)):
            window = frame.iloc[k-28:k+1]
            values = window.close.to_numpy(float)
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                result = [float(v) for v in original(pd.DataFrame({'close': values}))]
            alternative = independent_score(values)
            finite = all(math.isfinite(v) for v in result) and alternative is not None
            if not finite: invalid += 1
            end = window.date.iloc[-1]
            require(end in positions and window.date.iloc[0] in positions, 'Native date outside calendar')
            row = {'instrument': symbol, 'history_start': window.date.iloc[0], 'history_end': end,
                'next_calendar_open': days[positions[end]+1] if positions[end]+1 < len(days) else None,
                'year': int(end[:4]), 'crosses_calendar_gap': positions[end]-positions[window.date.iloc[0]]+1 != 29,
                'finite': finite}
            for j, column in enumerate(maximum):
                row[column] = result[j]
                row['independent_'+column] = alternative[j] if alternative is not None else float('nan')
                if finite:
                    difference = abs(result[j]-alternative[j])
                    maximum[column] = max(maximum[column], difference)
                    require(math.isclose(result[j], alternative[j], rel_tol=1e-9, abs_tol=1e-10), 'Independent regression mismatch: '+column)
            rows.append(row)
        profiles[symbol] = {'rows': len(frame), 'windows': len(frame)-28, 'first': frame.date.iloc[0],
            'last': frame.date.iloc[-1], 'calendar_missing_dates': missing, 'nonfinite_windows': invalid}
    component = pd.DataFrame(rows)
    ranks = []
    for day, group in component.groupby('history_end', sort=True):
        if len(group) != 3 or not group.finite.all(): continue
        by_symbol = group.set_index('instrument')
        # Python's stable sort matches the literal original, including ties and negative scores.
        ranking = sorted(POOL, key=lambda s: by_symbol.loc[s, 'score'], reverse=True)
        other = sorted(POOL, key=lambda s: by_symbol.loc[s, 'independent_score'], reverse=True)
        ranks.append({'history_end': day, 'next_calendar_open': by_symbol.loc[POOL[0], 'next_calendar_open'],
            'year': int(day[:4]), 'top': ranking[0], 'second': ranking[1], 'third': ranking[2],
            'top_score': float(by_symbol.loc[ranking[0], 'score']), 'all_negative': bool((group.score < 0).all()),
            'crosses_calendar_gap': bool(group.crosses_calendar_gap.any()),
            'independent_top': other[0], 'independent_order_difference': ranking != other})
    return component, pd.DataFrame(ranks), profiles, maximum


def compute(directory, label):
    runtime_binding(directory)
    output = directory/label
    output.mkdir(exist_ok=False)
    with network_blocked():
        diag = diagnostics(directory)
        save(output/'diagnostics.json', diag)
        inputs = {s: pd.read_parquet(directory/'inputs'/(s+'.parquet')) for s in POOL}
        component, ranks, profiles, maximum = operands(inputs, pd.read_parquet(directory/'inputs/calendar.parquet'), score_function(directory))
        component.to_parquet(output/'regression-operands.parquet', index=False)
        ranks.to_parquet(output/'ranking-operands.parquet', index=False)
        summary = {'status': 'literal_native_operands_not_original_platform_signals_or_trades',
            'diagnostic_cases': diag['case_count'], 'native_windows': len(component), 'profiles': profiles,
            'common_ranking_windows': len(ranks), 'ranking_first': ranks.history_end.iloc[0], 'ranking_last': ranks.history_end.iloc[-1],
            'all_negative_ranking_windows': int(ranks.all_negative.sum()),
            'cross_gap_native_windows': int(component.crosses_calendar_gap.sum()),
            'cross_gap_ranking_windows': int(ranks.crosses_calendar_gap.sum()),
            'independent_regression_max_abs_difference': maximum,
            'independent_order_differences': int(ranks.independent_order_difference.sum()),
            'by_year': ranks.groupby('year').agg(rows=('year', 'size'), all_negative=('all_negative', 'sum'), cross_gap=('crosses_calendar_gap', 'sum')).reset_index().to_dict('records'),
            'original61_actual_pool_windows': 0, 'original61_pool_reduced': False,
            'complete_original_signals': 0, 'new_trading_backtests': 0, 'original_strategy_complete': False,
            'formal_source_research_acceptance': False, 'library_goal_complete': False}
        save(output/'summary.json', summary)
    return summary


def run(directory=DIRECTORY):
    for label in ('compute', 'replay'):
        command = [sys.executable, '-m', 'scripts.research_strategy_batch107', 'compute', '--directory', str(directory), '--label', label]
        child = subprocess.run(command, capture_output=True, text=True, timeout=180)
        save(directory/(label+'-command.json'), {'command': command, 'returncode': child.returncode,
            'stdout': child.stdout, 'stderr': child.stderr, 'network_blocked': True})
        require(child.returncode == 0, 'Offline process failed; retain directory')
    require(all(file_sha(directory/'compute'/n) == file_sha(directory/'replay'/n) for n in OUTPUTS), 'Fresh offline outputs differ')
    return {'status': 'two_fresh_offline_processes_match', 'files': len(OUTPUTS)}


def verify(directory=DIRECTORY):
    runtime_binding(directory)
    require(all(file_sha(directory/'compute'/n) == file_sha(directory/'replay'/n) for n in OUTPUTS), 'Offline bytes differ')
    with network_blocked():
        inputs = {s: pd.read_parquet(directory/'inputs'/(s+'.parquet')) for s in POOL}
        component, ranks, _, _ = operands(inputs, pd.read_parquet(directory/'inputs/calendar.parquet'), score_function(directory))
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'compute/regression-operands.parquet'), component, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'compute/ranking-operands.parquet'), ranks, check_exact=True)
    require(read('data/catalog/strategies/latest.json')['catalog_id'] == Path(read(directory/'prepared.json')['catalog']).parent.name, 'Stage published an unaccepted catalog')
    tests = read(directory/'tests.json')
    require(tests['returncode'] == 0 and tests['network_blocked'] is True, 'Actual tests missing')
    proof = {'status': 'ok_component_stage', 'summary': read(directory/'compute/summary.json'),
        'fresh_network_blocked_processes': 2, 'offline_differences': 0, 'saved_exact_readback': True,
        'protection': protect('data', directory), 'source_sha256': list(SOURCE_SHA), 'tests': tests,
        'formal_source_research_acceptance': False, 'formal_trading_acceptance': False,
        'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}}
    save(directory/'checked.json', proof)
    return {k: v for k, v in proof.items() if k not in ('evidence_sha256', 'tests')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'compute', 'run', 'verify'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--label', default='compute')
    args = parser.parse_args()
    result = compute(args.directory, args.label) if args.action == 'compute' else globals()[args.action](args.directory)
    print(json.dumps(clean(result), ensure_ascii=False, allow_nan=False))
