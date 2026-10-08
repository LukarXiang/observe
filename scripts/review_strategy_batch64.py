"""Freeze price-shape arithmetic and TA-Lib pattern outputs without inventing trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import importlib.util
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import talib
from talib import abstract

from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch52 import dependency_state
from scripts.review_strategy_batch58 import definitions
from scripts import review_strategy_batch62 as operands
from scripts import review_strategy_batch63 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch63/20261007-SVR-value-financial-recovery')
RECEIPT = Path('docs/handoff/2026-10-07-batch63-verification.json')
SOURCES = ('2022年度精选策略/16.选股因子系列研究(十八) — 价格形态选股因子.txt',
    '2022年度精选策略/50.K线形态识别与验证-升级.txt')
SOURCE_SHA = ('c6a8ae85090ec6362b1b94ac9b533d6c1e6f7742a58f55a5f705ccb322ccc46b',
    '2cb9f6ae64a5eae05cc24719a4482b924643b85170c9adb294571e8cf59a683d')
REFERENCES = (('repo/backtrader', 'backtrader/talib.py'),
    ('repo/polars_ta', 'polars_ta/talib/__init__.py'),
    ('repo/skfolio', 'src/skfolio/preprocessing/_transformer/_cross_sectional/_cs_standard_scaler.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch64.py', 'tests/unit/test_batch64_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'default-failure-binding.json', 'query-fixture-failure-binding.json', 'source-assumption-failure-binding.json',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json', 'supplement-decision.json',
    'price-input.parquet', 'calendar-input.parquet', 'component-research.json', 'shape-values.parquet',
    'pattern-values.parquet', 'diagnostics.json', 'component-offline.json', 'diagnostics-offline.json',
    'offline-catalog.json', 'offline-verification.json'}
OHLC = ('open', 'high', 'low', 'close')
SHAPES = ('HighOpen', 'CloseLow')


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch63.py') == proof['final_script_sha256'], 'Accepted batch63 changed')
    bound = operands.binding(root)
    return_value = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'), 'upstream': previous.binding(root, ACCEPTED),
        'price_file': bound['price_file'], 'price_sha256': bound['price_sha256'],
        'calendar_file': bound['calendar_file'], 'calendar_sha256': bound['calendar_sha256'], 'not_a_backtest': True}
    if directory is not None: require(return_value == read(directory / 'input-binding.json'), 'Input binding changed')
    return return_value


def preflight(root, directory):
    bound = binding(root); require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch64 price-shape and candle research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def parsed_source(text, number):
    edits = []
    if number == 0:
        lines = text.splitlines()
        require(lines[562] == '    print "调整 r2 值: ", np.nanmean(result_r2)', 'Legacy print changed')
        edits = [{'line': 563, 'original': lines[562], 'normalized': '    print("调整 r2 值: ", np.nanmean(result_r2))',
            'purpose': 'AST parsing only; full notebook and LRData are not executed'}]
        lines[562] = edits[0]['normalized']; text = '\n'.join(lines)
    return ast.parse(text), {'edits': edits, 'parsed_code_sha256': hashlib.sha256(text.encode()).hexdigest()}


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    tree, normalization = parsed_source(read_source(Path(row['source_copy']))[0], number)
    require(row['normalization'] == normalization, 'Source normalization changed')
    return tree


def pattern_names(tree):
    calls = [n.value for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
        isinstance(n.value.func, ast.Name) and n.value.func.id == 'check_bar']
    require(len(calls) == 61 and all(len(n.args) == 2 and not n.keywords and isinstance(n.args[0], ast.Attribute) and
        isinstance(n.args[0].value, ast.Name) and n.args[0].value.id == 'tb' and isinstance(n.args[1], ast.Constant) for n in calls), 'Pattern calls changed')
    names = [n.args[0].attr for n in calls]
    require(len(set(names)) == 61 and set(names) == set(talib.get_function_groups()['Pattern Recognition']), 'Pattern library groups changed')
    return names


def api_evidence():
    import baostock as bs
    names = talib.get_function_groups()['Pattern Recognition']; library = Path(talib._ta_lib.__file__)
    shared = sorted(Path(talib.__file__).parent.parent.glob('ta_lib.libs/*'))
    return {'pandas': pd.__version__, 'numpy': np.__version__, 'wrapper': talib.__version__,
        'c_version': talib.__ta_version__.decode(), 'compatibility': talib.get_compatibility(),
        'binary': str(library), 'binary_sha256': file_sha(library),
        'shared_libraries': {str(p): file_sha(p) for p in shared if p.is_file()},
        'pattern_functions': {n: {'info': abstract.Function(n).info, 'lookback': abstract.Function(n).lookback,
            'direct_doc': getattr(talib, n).__doc__} for n in names},
        'query_history_signature': str(inspect.signature(bs.query_history_k_data_plus)),
        'query_history_source': inspect.getsource(bs.query_history_k_data_plus),
        'constituents_source': inspect.getsource(BaoStock.index_constituents),
        'modules_installed': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqfactor', 'statsmodels')},
        'limits': ['Direct and abstract pattern APIs share the same C implementation; not independent algorithm validation',
            'Original platform TA-Lib version, pre-adjustment, average-price and universe semantics are unproved']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    price = pd.read_parquet(operands.OPERAND)
    tables = {}
    for name in ('index_constituents', 'industry_members', 'instrument_status'):
        frame = store.load_state(state, name)
        tables[name] = {'rows': len(frame), 'columns': list(frame)}
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']), 'required_tables': tables,
        'numeric_stocks': sorted(price.instrument.unique()), 'numeric_rows': len(price), 'numeric_columns': list(price),
        'avg_column_available': 'avg' in price, 'historical_equity_dependencies': dependency_state(root),
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'limits': ['Ten numeric stocks are not original 000002/399107 historical members or 2014 random whole pool',
            'Known is_trading=false rows can be skipped; missing observations are not inferred suspensions',
            'No original avg, historical style factors, industry or jqfactor transformation proof']}


def supplements():
    accepted = previous.supplements()
    return {**accepted, 'decision': 'Reuse accepted BaoStock login/DNS failures and installed interface evidence; no original jqdata/jqfactor credentials or archived avg/style-factor versions',
        'limits': ['BaoStock constituent adapter supports only 000300/000905; cannot substitute for original 000002/399107',
            'Vendor amount/volume is not silently substituted for platform avg; arithmetic outputs exclude VwapClose',
            'Supplier failure is not proof that historical data do not exist']}


def start(root, directory):
    bound = binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    failure = Path('data/staging/strategies-batch64/20261007-default-expression-failure')
    save(directory / 'default-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    failure = Path('data/staging/strategies-batch64/20261007-query-panel-fixture-failure')
    save(directory / 'query-fixture-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    failure = Path('data/staging/strategies-batch64/20261007-longshort-source-assumption-failure')
    save(directory / 'source-assumption-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for i, (name, sha) in enumerate(zip(SOURCES, SOURCE_SHA, strict=True)):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        tree, normalization = parsed_source(text, i); copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'normalization': normalization, 'definition_ast': definitions(tree),
            'review': REVIEWS[name], 'newly_reviewed': True})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Direct TA-Lib OHLC/lookback/parameters and missing-aware cross-section comparison; not platform replacements'})
    for name in ('price', 'calendar'): shutil.copyfile(bound[f'{name}_file'], directory / f'{name}-input.parquet')
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'snapshot': SNAPSHOT, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch64 two full price-shape/candle sources and original data limitations archived')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source scope changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and file_sha(row['source_path']) == file_sha(row['source_copy']) == sha and
            row['review'] == REVIEWS[name] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Source/rule/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
        require(subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip() == row['commit'], 'Reference HEAD changed')
    bound = read(directory / 'input-binding.json')
    for name in ('price', 'calendar'): require(file_sha(directory / f'{name}-input.parquet') == bound[f'{name}_sha256'], 'Operand changed')
    failure = read(directory / 'default-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 5 passed' in red['stdout'] and 'Unsafe or missing functions' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red default binding changed')
    failure = read(directory / 'query-fixture-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Query failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Query failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 6 passed' in red['stdout'] and "has no attribute 'empty'" in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red query binding changed')
    failure = read(directory / 'source-assumption-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Source assumption failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Source assumption failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 6 passed' in red['stdout'] and 'Original longshort defects changed' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red source assumption binding changed')


def safe_default(node):
    return isinstance(node, ast.Constant) or (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and
        all(isinstance(n, ast.Constant) and type(n.value) in (int, float) for n in (node.left, node.right)))


def selected(directory, number, names, ns):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and all(safe_default(d) for d in n.args.defaults) and
        all(d is None or safe_default(d) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-price-shape-functions>', 'exec'), ns)
    return ns


def run_nodes(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-price-shape-arithmetic>', 'exec'), ns)
    return ns


def shape_values(price, dates, tree):
    dated = price.assign(date=pd.to_datetime(price.date))
    panels = {n: dated.pivot(index='date', columns='instrument', values=n).reindex(dates).sort_index(axis=1) for n in OHLC}
    loops = [n for n in tree.body if isinstance(n, ast.For) and any(isinstance(x, ast.Assign) and
        isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'temp1' for x in n.body)]
    require(len(loops) == 2, 'Original shape loops changed')
    assignments = [n for n in loops[0].body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('temp1', 'temp2')]
    require(len(assignments) == 2 and all(ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
        for a, b in zip(assignments, [n for n in loops[1].body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and
            n.targets[0].id in ('temp1', 'temp2')], strict=True)), 'Original shape arithmetic differs')
    records = []; max_error = 0.
    for width in (10, 20):
        for end, day in enumerate(dates):
            windows = {n: f.iloc[max(0, end-width+1):end+1] for n, f in panels.items()}
            local = run_nodes(assignments, {'df_data': windows, 'log': np.log})
            for stock in panels['open'].columns:
                row = {'window': width, 'date': day.strftime('%Y-%m-%d'), 'instrument': stock,
                    'window_rows': len(windows['open'])}
                for name, output, numerator, denominator in zip(SHAPES, ('temp1', 'temp2'), ('high', 'close'), ('open', 'low'), strict=True):
                    values = [math.log(a/b) for a, b in zip(windows[numerator][stock], windows[denominator][stock], strict=True)
                        if pd.notna(a) and pd.notna(b) and a > 0 and b > 0]
                    expected = math.fsum(values)/len(values) if values else math.nan; actual = float(local[output][stock])
                    require((math.isnan(actual) and math.isnan(expected)) or math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'Independent shape arithmetic differs')
                    if values: max_error = max(max_error, abs(actual-expected))
                    row[name] = actual; row[f'{name}_finite_rows'] = len(values)
                records.append(row)
    return pd.DataFrame(records), {'max_abs_error': max_error,
        'arithmetic_ast': [ast.dump(n, include_attributes=False) for n in assignments], 'rtol': 1e-12, 'atol': 1e-12}


def pattern_values(price, dates, names):
    pieces = []; counts = {'insufficient_100_rows': 0, 'unknown_internal_calendar_day': 0, 'invalid_OHLC': 0, 'eligible': 0, 'known_paused_skipped': 0}
    calls = 0; functions = {n: getattr(talib, n) for n in names}; alternate = {n: abstract.Function(n) for n in names}
    for stock, frame in price.groupby('instrument', sort=True):
        frame = frame.sort_values('date').copy(); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.is_unique and frame.is_trading.notna().all() and frame.date.isin(dates).all(), 'Unproved price identities/status')
        counts['known_paused_skipped'] += int((~frame.is_trading).sum())
        absent = (~dates.isin(frame.date)).astype(int).cumsum()
        active = frame[frame.is_trading].reset_index(drop=True); calendar_positions = dates.get_indexer(active.date)
        arrays = {n: active[f'{n}_adj'].to_numpy(dtype=float) for n in OHLC}
        signals = np.full((len(active), len(names)), np.nan); flags = []
        for end in range(len(active)):
            if end < 99: reason = 'insufficient_100_rows'
            elif absent[calendar_positions[end]] - absent[calendar_positions[end-99]]: reason = 'unknown_internal_calendar_day'
            else:
                window = {n: a[end-99:end+1] for n, a in arrays.items()}
                if not all(np.isfinite(a).all() and (a > 0).all() for a in window.values()): reason = 'invalid_OHLC'
                else:
                    reason = 'eligible'
                    for k, name in enumerate(names):
                        value = functions[name](*[window[n] for n in OHLC])[-1]
                        reference = alternate[name](window)[-1]
                        require(value == reference, 'Direct/abstract TA-Lib output differs')
                        signals[end, k] = value; calls += 1
            flags.append(reason); counts[reason] += 1
        output = pd.DataFrame(signals, columns=names).astype('Int16')
        output.insert(0, 'window_status', flags); output.insert(0, 'instrument', stock)
        output.insert(0, 'date', active.date.dt.strftime('%Y-%m-%d').to_numpy()); pieces.append(output)
    values = pd.concat(pieces, ignore_index=True)
    return values, {'rows': len(values), 'statuses': counts, 'pattern_names': names, 'numeric_window_rows': 100,
        'direct_abstract_matched_cells': calls, 'nonzero': {n: int((values[n].fillna(0) != 0).sum()) for n in names},
        'comparison': 'Both interfaces use the same C library; this validates binding/defaults only, not independent pattern logic',
        'input': 'Frozen back-adjusted OHLC; explicit paused rows skipped; unknown internal days block all 61 outputs',
        'limits': ['Default pre-adjustment/platform library version is not proved equivalent to frozen back-adjusted inputs',
            'No random whole-universe replay or repaired discern_pattern success claimed']}


def component(directory):
    validate_sources(directory); require(api_evidence() == read(directory / 'existing-apis.json'), 'Numerical library/API changed')
    price = pd.read_parquet(directory / 'price-input.parquet'); calendar = pd.read_parquet(directory / 'calendar-input.parquet')
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date'])).sort_values()
    require(len(dates) == 5280 and dates.is_unique and not price.duplicated(['date', 'instrument']).any(), 'Operand calendar changed')
    shapes, arithmetic = shape_values(price, dates, source_tree(directory, 0))
    patterns, pattern = pattern_values(price, dates, pattern_names(source_tree(directory, 1)))
    scopes = []
    for label, start, end in (('original_price_research', '2013-01-01', '2018-01-01'), ('extended_numeric_operands', '2005-01-05', '2026-09-29')):
        part = shapes[shapes.date.between(start, end)]
        scopes.append({'scope': label, 'rows': len(part), 'first': part.date.min(), 'last': part.date.max(),
            'finite': {n: int(part[n].notna().sum()) for n in SHAPES},
            'complete_windows': {n: int((part[f'{n}_finite_rows'] == part.window).sum()) for n in SHAPES},
            'fingerprint': hashlib.sha256(part.to_json(orient='split', double_precision=15).encode()).hexdigest()})
    result = {'snapshot': SNAPSHOT, 'numeric_price_rows': len(price), 'shape_rows': len(shapes), 'scopes': scopes,
        'shape_arithmetic': arithmetic, 'patterns': pattern, 'not_a_backtest': True, 'platform_equivalent': False,
        'original_strategy_complete': False, 'cost_backtests': [], 'real_model_fits': 0,
        'limits': ['Overlapping scopes and daily arithmetic are not original monthly/2W scheduled pool selection or independent market samples',
            'Original mean skips NaNs; finite input counts retained. Raw same-day ratios, no platform default fill inferred',
            'No avg/VwapClose, standardization, neutralization, style factors, IC or original model results invented',
            'Patterns use installed TA-Lib defaults, not comment penetration=0 or short display bar counts',
            'No financial strict records admitted; unique Book unchanged; source statistical fees are diagnostics only']}
    return result, shapes, patterns


def diagnostics(directory):
    validate_sources(directory); trees = [source_tree(directory, i) for i in range(2)]; rows = []
    raw = read_source(Path(read(directory / 'source-reviews/review.json')['sources'][0]['source_copy']))[0]
    try: ast.parse(raw)
    except SyntaxError as exc: rows.append({'case': 'legacy_print_syntax', 'line': exc.lineno, 'error': type(exc).__name__})
    else: raise ValueError('Expected legacy print syntax')
    import time
    try: time.clock()
    except AttributeError as exc: rows.append({'case': 'time_clock_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected removed clock')
    dates = pd.bdate_range('2020-01-01', periods=120)
    frame = pd.DataFrame({n: np.arange(120, dtype=float)+100+offset for n, offset in zip(OHLC, (0, 2, -2, 1), strict=True)}, index=dates)
    ns = selected(directory, 0, ['get_period_date', 'GetPchg', 'GetPchg_100', 'factor_IC_analysis', 'delect_stop', 'get_stock_A'],
        {'pd': pd, 'np': np, 'datetime': dt, 'get_price': lambda *a, **kw: frame.copy()})
    try: ns['get_period_date']('M', '2020-01-01', '2020-06-30')
    except TypeError as exc: rows.append({'case': 'resample_how_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected legacy resample failure')
    selection = pd.DataFrame({'HighOpen': [3., 2., 1.]}, index=list('ABC')); ns['get_period_date'] = lambda *a: ['D0', 'D1']
    try: ns['GetPchg'](1, {'D0': selection}, 'HighOpen', 'D0', 'D1', 'M')
    except AttributeError as exc: rows.append({'case': 'dataframe_sort_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected removed sort')
    try: range(1, 11) + ['longshort']
    except TypeError as exc: rows.append({'case': 'range_plus_list_python3', 'error': type(exc).__name__})
    try: [1., 2., 3.][0*3/5:1*3/5]
    except TypeError as exc: rows.append({'case': 'year_slices_float_python3', 'error': type(exc).__name__})
    seen = []
    def correlation(a, b): seen.append(list(a.index)); return (0., 1.)
    class QueryPanel(dict):
        empty = False
    ns.update(st=SimpleNamespace(pearsonr=correlation), standardlize=lambda v, **kw: v,
        get_period_date=lambda *a: ['D0', 'D1', 'D2'],
        get_price=lambda *a, **kw: QueryPanel(close=pd.DataFrame([[1., 1., 1.], [2., 3., 4.]], columns=list('ABC') if a[1] == 'D0' else list('BCD'))))
    ns['factor_IC_analysis']({'D0': selection, 'D1': selection.set_axis(list('BCD'))}, 'HighOpen', 'D0', 'D2', 'M')
    require(seen == [list('ABC'), list('ABC'), list('BC'), list('BC')], 'Original IC persistent index differs')
    rows.append({'case': 'IC_persistent_frame_drops_new_universe_assets', 'correlation_indices': seen, 'fixture_standardization': 'identity, not platform equivalence'})
    ns.update(get_security_info=lambda s: SimpleNamespace(start_date=dt.date(2020, 1, 2) if s == 'A' else dt.date(2020, 1, 1)))
    selected_stocks = ns['delect_stop'](['A', 'B'], '2020-04-01', 90)
    require(selected_stocks == ['B'], 'Original strict listing-day boundary changed')
    rows.append({'case': 'IPO_filter_strict_90_calendar_days', 'selected': selected_stocks})
    code = next(n for n in trees[0].body if isinstance(n, ast.For) and any(isinstance(x, ast.Assign) and
        any(isinstance(t, ast.Subscript) and ast.unparse(t).startswith("result.loc['longshort'") for t in x.targets) for x in ast.walk(n)))
    sums = [n for n in ast.walk(code) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and 'longshort' in ast.unparse(t) for t in n.targets)]
    require(len(sums) == 2 and all(isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add) for n in sums) and
        'pchg_2W' in ast.unparse(sums[0].value) and 'pchg_M' in ast.unparse(sums[1].value), 'Original longshort arithmetic changed')
    rows.append({'case': 'longshort_adds_both_long_groups_with_correct_frequency_dict', 'assignment_ast': [ast.dump(n, include_attributes=False) for n in sums]})
    ns = selected(directory, 1, ['discern_pattern', 'find_pattern', 'get_k_value'], {'pd': pd, 'np': np, 'tb': talib,
        'dt': SimpleNamespace(datetime=SimpleNamespace(now=lambda: dt.datetime(2020, 12, 31))), 'get_price': lambda *a, **kw: frame.tail(100).copy()})
    try: list(ns['discern_pattern']('A', '2020-12-31'))
    except KeyError as exc: rows.append({'case': 'discern_pattern_datetime_series_minus1_is_label', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected datetime Series label failure')
    ns.update(get_all_securities=lambda *a, **kw: pd.DataFrame(index=['A']), get_trade_days=lambda **kw: [d.date() for d in dates[-15:]])
    signal = pd.Series([0, -100, 0, 100], index=dates[:4])
    result = ns['find_pattern'](lambda *a: signal)
    require(result == ('A', dates[1]), 'Original first signal selection changed')
    rows.append({'case': 'find_pattern_first_nonzero_not_latest_and_keeps_negative', 'stock': result[0], 'date': str(result[1].date())})
    result = ns['find_pattern'](lambda *a: (_ for _ in ()).throw(ValueError('fixture failure')))
    require(result is None, 'Original broad exception behavior changed')
    rows.append({'case': 'find_pattern_swallow_all_exceptions_returns_none', 'result': result})
    ns['get_trade_days'] = lambda **kw: [d.date() for d in dates[:5]]
    try: ns['get_k_value'](('A', dates[0].date()), 3)
    except IndexError as exc: rows.append({'case': 'display_requires_tenth_future_trade_day', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected short future calendar failure')
    ns['get_trade_days'] = lambda **kw: [d.date() for d in dates[:15]]
    try: ns['get_k_value'](('A', dates[0].date()), 3)
    except KeyError as exc: rows.append({'case': 'display_OHLC_integer_series_index_is_label', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected display integer-label failure')
    names = pattern_names(trees[1]); metadata = api_evidence()['pattern_functions']
    rows.append({'case': 'comment_penetration_zero_is_not_executed_default', 'parameters': {n: metadata[n]['info']['parameters'] for n in names if metadata[n]['info']['parameters']}})
    return {'cases': rows, 'synthetic_fixture': True, 'not_a_backtest': True, 'real_model_fits': 0,
        'real_historical_trade_windows': 0, 'platform_equivalent': False,
        'limits': ['Frozen original functions/AST with deterministic queries and time; original defects retained',
            'No patched full notebook, random whole-universe replay, IC result or price-statistic fee ledger claimed']}


def study(root, directory):
    binding(root, directory); before = implementation(); comp, shapes, patterns = component(directory); diag = diagnostics(directory)
    require(before == implementation() and not any((directory / n).exists() for n in ('shape-values.parquet', 'pattern-values.parquet')), 'Study changed or exists')
    shapes.to_parquet(directory / 'shape-values.parquet', index=False); patterns.to_parquet(directory / 'pattern-values.parquet', index=False)
    save(directory / 'component-research.json', comp); save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch64 real price-shape arithmetic and 61 TA-Lib pattern outputs archived; no trades')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch64 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        comp, shapes, patterns = component(directory); diag = diagnostics(directory); catalog = offline_catalog(root, directory)
        dep = inventory(root)
    for name, frame in (('shape', shapes), ('pattern', patterns)):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / f'{name}-values.parquet'), check_exact=True)
    require(before == implementation() and comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json'), 'Offline evidence differs')
    save(directory / 'component-offline.json', comp); save(directory / 'diagnostics-offline.json', diag)
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'values_sha256': {n: file_sha(directory / f'{n}-values.parquet') for n in ('shape', 'pattern')},
        'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch64 forbidden-network real components/diagnostics and three catalogs match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch64.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch64_research.py', 'tests/unit/test_batch63_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json') and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') == file_sha(directory / 'diagnostics-offline.json'), 'Offline proof changed')
    for name, sha in off['values_sha256'].items(): require(file_sha(directory / f'{name}-values.parquet') == sha, 'Offline values changed')
    comp, shapes, patterns = component(directory); diag = diagnostics(directory)
    for name, frame in (('shape', shapes), ('pattern', patterns)):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / f'{name}-values.parquet'), check_exact=True)
    require(comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        inventory(root) == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json'), 'Recomputed evidence differs')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch64 price-shape/candle research accepted; original whole-pool research remains gated')
    receipt = Path('docs/handoff/2026-10-07-batch64-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'offline_catalog': read(directory / 'offline-catalog.json'), 'component': comp, 'diagnostics': diag, 'protection': protection, 'progress': progress,
        'supplements': read(directory / 'supplement-decision.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt), 'checked_state_sha256': file_sha(directory / 'checked-state.json'),
        'final_script_sha256': file_sha(__file__), 'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'start', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch64/20261007-price-shape-candles')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
