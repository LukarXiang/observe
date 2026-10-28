"""Freeze external timing and GP research; missing original CSVs block trades."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import datetime as dt
import importlib.util
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch82 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch61 import OPERAND, OPERAND_SHA, CALENDAR, CALENDAR_SHA
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch83/20261009-external-timing-GP')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略2/91.【复现】特征分布建模择时.txt',
    '2021年度精选策略/40.基于遗传算法挖掘因子2.txt')
SOURCE_SHA = ('95039a7c5a6a1fc4a9c4b37124ed80c691fe77bfac0a4c22f7c38a1008ecd3ed',
    '77c87a1fb7c854fbb2e6acc9b0a6caa1ab72212990382d05a83d1ca823d128a0')
REFERENCES = (('repo/gplearn', 'gplearn/functions.py'), ('repo/gplearn', 'gplearn/genetic.py'),
    ('repo/qlib', 'qlib/data/ops.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch83.py', 'tests/unit/test_batch83_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'dependency-inventory.json', 'supplement-decision.json',
    'price-input.parquet', 'calendar-input.parquet', 'diagnostics.json', 'component-research.json',
    'alpha2-values.parquet', 'offline-verification.json', 'offline-catalog.json',
    'development-specialized-attempt01/result.json', 'development-specialized-diagnosis.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch82-verification.json')
    review = Path('docs/handoff/2026-10-09-batch82-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted82 binding changed')
    for name in ('scripts/review_strategy_batch82.py', 'tests/unit/test_batch82_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted82 entrypoint changed')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(previous.DIRECTORY/name) == sha for name, sha in checked['evidence_sha256'].items()),
        'Accepted82 frozen evidence changed')
    operands = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']
    require(file_sha(OPERAND) == OPERAND_SHA == operands['price-input.parquet'] and
        file_sha(CALENDAR) == CALENDAR_SHA == operands['calendar-input.parquet'], 'Accepted long inputs changed')
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'price_file': str(OPERAND),
        'price_sha256': OPERAND_SHA, 'calendar_file': str(CALENDAR), 'calendar_sha256': CALENDAR_SHA,
        'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def inventory(root):
    command = ['rg', '--files', '--hidden', '--no-ignore', 'repo', str(root),
        '-g', '*[tT][rR][aA][dD][eE]_[lL][iI][sS][tT]*',
        '-g', '*[sS][tT][oO][cC][kK]_[dD][aA][tT][aA]*', '-g', '!**/.git/**']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    require(result.returncode in (0, 1), 'CSV filename search failed')
    manifests = []
    def matches(value):
        if isinstance(value, str): return [value] if re.search(r'(?:trade_list|stock_data)\.csv', value, re.I) else []
        if isinstance(value, (list, dict)):
            return [m for child in (value.values() if isinstance(value, dict) else value) for m in matches(child)]
        return []
    for path in sorted((Path(root)/'handoff').glob('*.manifest.json')):
        manifests.append({'file': str(path), 'sha256': file_sha(path), 'matches': matches(read(path))})
    frame = pd.read_parquet(OPERAND)
    state = Store(root).state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'search': {'command': command, 'returncode': result.returncode, 'matches': result.stdout.splitlines(),
            'matched_file_sha256': {n: file_sha(n) for n in result.stdout.splitlines() if Path(n).is_file()},
            'handoff_manifests': manifests},
        'installed': {n: importlib.util.find_spec(n) is not None for n in
            ('talib', 'gplearn', 'jqdatasdk', 'jqfactor_analyzer', 'graphviz')},
        'reference_rows': len(frame), 'reference_stocks': sorted(frame.instrument.unique()),
        'reference_dates': [frame.date.min(), frame.date.max()], 'reference_columns': list(frame),
        'original_training_rows_admitted': 0, 'original_trade_dates_admitted': 0, 'not_a_backtest': True,
        'limits': ['Search covers actual filenames and two present handoff manifests, not unknown archive contents/platform files',
            'No avg/pre_close/pct10 label definitions, original training pool/order/versions or historical index membership proved',
            'Ten declared reference stocks and adjusted highs are arithmetic inputs, not original GP training or CSI800',
            'ETF identity/events/reference price/status/fees and open execution remain unproved']}


def preflight(root, directory):
    bound = binding(root); accepted = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in accepted['implementation_sha256'].items()), 'Accepted implementation changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in accepted['evidence_sha256'].items()), 'Accepted evidence changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    for repo, name in REFERENCES: require((Path(repo)/name).is_file(), 'Reference missing')
    dep = inventory(root)
    checkpoint(root, directory, 'Batch83 parent82 and long arithmetic inputs verified; original CSVs not substituted')
    save(directory/'input-binding.json', bound); save(directory/'dependency-inventory.json', dep)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    for name in ('price', 'calendar'): shutil.copyfile(bound[name+'_file'], directory/(name+'-input.parquet'))
    return {'status': 'ok', 'reference_rows': dep['reference_rows'], 'not_a_backtest': True}


def definitions(tree):
    return [{'name': node.name, 'line': node.lineno, 'ast': ast.dump(node, include_attributes=False)}
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    sources = []; refs = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source reviewed/changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        sources.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'original_line_numbers_retained': True, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory/'references').mkdir(exist_ok=False)
    for repo, name in REFERENCES:
        path = Path(repo)/name; copied = directory/'references'/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Native GP vector/closure contracts and per-instrument rolling; no custom GP engine or substituted training'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(old) == len(current) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder/'review.json', {'snapshot': SNAPSHOT, 'sources': sources, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory/'supplement-decision.json', {'new_requests': 0, 'new_rows': 0, 'published': False,
        'decision': 'Original user CSVs have no specified reconstructible API; search local files/manifests first. Installed market APIs cannot recover precomputed dates or unknown labels.',
        'restore_required': ['trade_list.csv with generating code/vintage/available_at',
            'stock_data.csv with seven field definitions/pct10 horizon/price basis/order/pool and label maturity'],
        'model_policy': 'Use native gplearn if original training data are restored; no dependency install or training on reference stocks',
        'economic_variants_require_confirmation': True, 'not_a_backtest': True})
    return archive(root, 'Batch83 two complete sources and20 GP definitions including duplicate alpha2 frozen; original CSV gates retained')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True):
        text, encoding = read_source(Path(row['source_copy']))
        require(file_sha(row['source_path']) == file_sha(row['source_copy']) == sha == row['source_sha256'] and
            row['review'] == REVIEWS[name] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference count changed')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    for name in ('price', 'calendar'):
        require(file_sha(directory/(name+'-input.parquet')) == read(directory/'input-binding.json')[name+'_sha256'], 'Reference input changed')


def selected_nodes(tree, names, namespace, occurrences=None):
    occurrences = occurrences or {}; nodes = []
    for name in names:
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(bool(found) and (len(found) == 1 or name in occurrences), 'Missing/ambiguous selected definition')
        node = copy.deepcopy(found[occurrences.get(name, 0)])
        # Annotations are inert for this diagnostic loader; never evaluate them.
        for nested in ast.walk(node):
            if isinstance(nested, ast.FunctionDef):
                annotations = [nested.returns, *(a.annotation for a in
                    (*nested.args.posonlyargs, *nested.args.args, *nested.args.kwonlyargs)),
                    *(a.annotation for a in (nested.args.vararg, nested.args.kwarg) if a)]
                require(all(a is None or isinstance(a, ast.Name) or
                    (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name)) for a in annotations), 'Unsafe annotation')
                nested.returns = None
                for a in (*nested.args.posonlyargs, *nested.args.args, *nested.args.kwonlyargs, nested.args.vararg, nested.args.kwarg):
                    if a is not None: a.annotation = None
                require(safe_function(nested), 'Unsafe function definition')
        nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch83-selected>', 'exec'), namespace)
    return namespace


def selected(directory, number, names, namespace, occurrences=None):
    row = read(directory/'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source changed')
    return selected_nodes(ast.parse(read_source(Path(row['source_copy']))[0]), names, namespace, occurrences)


def plain(value):
    if isinstance(value, np.ndarray): return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value): return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)): return value.isoformat()
    return value


def notebook_boundaries(directory, stock_list, target):
    row = read(directory/'source-reviews/review.json')['sources'][1]
    require(file_sha(row['source_copy']) == SOURCE_SHA[1], 'Notebook source changed')
    tree = ast.parse(read_source(Path(row['source_copy']))[0])
    loops = [n for n in tree.body if isinstance(n, ast.For) and
        (ast.unparse(n.iter) == 'range(len(stock_list) - 1)' or
         (isinstance(n.iter, ast.Name) and n.iter.id == 'length'))]
    require(len(loops) == 2, 'Notebook boundary loop scope changed')
    ns = {'np': np, 'stock_list': stock_list, 'length': [], 'num': 1,
        'num_set_y_nan': 0, 'y_train': np.array(target, dtype=float)}
    exec(compile(ast.Module(body=loops, type_ignores=[]), '<frozen-batch83-boundary-loops>', 'exec'), ns)
    return {'length': ns['length'], 'labels_after_mask': plain(ns['y_train']),
        'labels_after_nan_to_num': plain(np.nan_to_num(ns['y_train']))}


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; schedules = []; costs = []; slips = []
    def record(case, **values): rows.append(plain({'case': case, **values}))
    def failure(case, kind, fn):
        try: fn()
        except kind as exc: record(case, error=type(exc).__name__)
        else: raise ValueError('Expected original failure absent: '+case)
    g = SimpleNamespace(); context = SimpleNamespace(current_dt=dt.datetime(2020, 3, 20, 9),
        portfolio=SimpleNamespace(long_positions={}, total_value=100000.))
    ns = {'g': g, 'pd': pd, 'np': np, 'dt': dt, 'BytesIO': BytesIO,
        'log': SimpleNamespace(set_level=lambda *a: None), 'set_option': lambda *a: None, 'set_benchmark': lambda *a: None,
        'run_daily': lambda fn, *a, **kw: schedules.append([fn.__name__, list(a), kw]),
        'set_slippage': slips.append, 'FixedSlippage': lambda x: x,
        'set_commission': costs.append, 'PerTrade': lambda **kw: kw,
        'order_target_value': lambda *a: orders.append(['value', *a]), 'order_target': lambda *a: orders.append(['quantity', *a])}
    selected(directory, 0, ['initialize', 'set_backtest', 'set_params', 'set_variables', 'before_trading_start', 'get_signal', 'trade', 'HMA', 'calc_netbuy'], ns)
    csv = b'row,datein,dateout\n0,2020-03-20,2020-03-23\n1,2020-03-20,2020-03-20\n'
    ns['read_file'] = lambda name: csv
    ns['initialize'](context)
    record('91_defaults', schedules=schedules.copy(), costs=costs.copy(), slips=slips.copy(), to_buy=g.to_buy,
        fast=g.fast_period, slow=g.slow_period, csv_dates=[str(x) for x in g.orders_frame.datein])
    ns['get_signal'](context); record('91_same_day_exit_priority', to_buy=g.to_buy)
    g.orders_frame = pd.DataFrame({'datein': [dt.date(2020, 3, 20)], 'dateout': [dt.date(2020, 3, 23)]})
    ns['get_signal'](context); record('91_entry', to_buy=g.to_buy)
    context.current_dt = dt.datetime(2020, 3, 21, 9); ns['get_signal'](context); record('91_no_date_keeps_true', to_buy=g.to_buy)
    context.current_dt = dt.datetime(2020, 3, 23, 9); ns['get_signal'](context); record('91_exit', to_buy=g.to_buy)
    context.current_dt = dt.datetime(2020, 3, 24, 9); ns['get_signal'](context); record('91_no_date_keeps_false', to_buy=g.to_buy)
    for flag, holdings in [(True, {}), (True, {'OTHER': 1}), (False, {}), (False, {'OTHER': 1})]:
        orders.clear(); g.to_buy = flag; context.portfolio.long_positions = holdings
        with redirect_stdout(StringIO()): ns['trade'](context)
        record(f'91_order_{flag}_{bool(holdings)}', orders=orders.copy())
    g.to_buy = True; context.portfolio.long_positions = {}; orders.clear()
    ns['order_target_value'] = lambda *a: (_ for _ in ()).throw(RuntimeError('controlled rejected order'))
    failure('91_failed_order', RuntimeError, lambda: ns['trade'](context)); record('91_failed_order_state', to_buy=g.to_buy)
    for year in (2009, 2011, 2013):
        for label, moment in [('exact', dt.datetime(year, 1, 1)), ('later', dt.datetime(year, 1, 1, 0, 0, 1))]:
            context.current_dt = moment; ns['before_trading_start'](context)
            record(f'91_fee_{year}_{label}', fee=costs[-1], slippage=slips[-1])
    ns['read_file'] = lambda *a: (_ for _ in ()).throw(FileNotFoundError('controlled missing CSV'))
    failure('91_missing_csv', FileNotFoundError, ns['set_variables'])
    ns['read_file'] = lambda *a: b'row,datein,dateout\n0,invalid,2020-03-20\n'
    failure('91_bad_csv_dates', AttributeError, ns['set_variables'])
    failure('91_hma_wrong_type', ValueError, lambda: ns['HMA']([1., 2.], 30))
    failure('91_hma_missing_talib', NameError, lambda: ns['HMA'](pd.Series(np.arange(40.)), 30))
    billboard = pd.DataFrame({'day': ['2020-03-20'], 'sales_depart_name': ['机构专用']})
    failure('91_netbuy_recursive', RecursionError, lambda: ns['calc_netbuy'](billboard))
    names = ['_my_metric', '_rolling_rank', '_rolling_prod', '_cube', '_square', '_delta', '_delay',
        '_corr', '_ts_sum', '_sma', '_stddev', '_ts_rank', '_ts_argmin', '_ts_argmax', '_ts_min', '_ts_max', '_ts_argmaxmin']
    gp = selected(directory, 1, names, {'np': np, 'pd': pd, 'rankdata': rankdata})
    data = np.array([1., 2., 3., 10., 20., 30.]); integers = np.full(6, 3, dtype=int); floats = np.full(6, 3.)
    for name in ('_delta', '_delay', '_square', '_cube'):
        record('40'+name, result=gp[name](data), boundary_index=3)
    for name in ('_corr', '_ts_sum', '_sma', '_stddev', '_ts_rank', '_ts_argmin', '_ts_argmax', '_ts_min', '_ts_max', '_ts_argmaxmin'):
        for label, windows in [('int', integers), ('float', floats), ('nonconstant', np.array([3, 3, 4, 3, 3, 3]))]:
            args = (data, data[::-1], windows) if name == '_corr' else (data, windows)
            output = gp[name](*args)
            record('40'+name+'_'+label, result=output, dtype=str(output.dtype))
    for case, target, pred in [('small', np.ones(5), np.arange(5.)), ('zero_low', np.r_[0., np.ones(9)], np.arange(10.))]:
        with np.errstate(divide='ignore', invalid='ignore'): value = gp['_my_metric'](target, pred, np.ones(len(target)))
        record('40_metric_'+case, value=value)
    y = np.arange(1., 21.); predicted = np.arange(20.)
    record('40_metric_weights_ignored', ones=gp['_my_metric'](y, predicted, np.ones(20)), zeros=gp['_my_metric'](y, predicted, np.zeros(20)))
    record('40_nan_to_num_label', result=np.nan_to_num(np.array([np.nan, np.inf, -np.inf])))
    for label, stocks in [('singleton_tail', ['A', 'A', 'B']), ('repeated_tail', ['A', 'A', 'B', 'B'])]:
        record('40_boundary_'+label, **notebook_boundaries(directory, stocks, range(1, len(stocks)+1)))
    record('40_stddev_uses_all_window_values', result=gp['_stddev'](data, np.array([3., 3., 3., 9., 9., 9.])))
    failure('40_square_missing_arg', TypeError, lambda: np.square())
    for occurrence in (0, 1):
        local = selected(directory, 1, ['alpha2'], {'np': np, 'pd': pd,
            'jq': SimpleNamespace(get_price=lambda *a, **kw: {'high': pd.DataFrame({'A': [2., 4., 6.]}),
                'open': pd.DataFrame({'A': [1., 2., 3.]}), 'volume': pd.DataFrame({'A': [1., 2., 3.]})})}, {'alpha2': occurrence})
        if occurrence == 0: record('40_alpha2_first', result=local['alpha2'](['A'], '2020-03-20', 3).values)
        else: failure('40_alpha2_last_missing_sigmoid', NameError, lambda: local['alpha2'](['A'], '2020-03-20', 3))
    return {'cases': rows, 'not_a_backtest': True, 'original_training': False,
        'loader_policy': 'Only selected definitions; inert simple annotations removed; original bodies/defaults retained; both alpha2 occurrences separately selected'}


def compute(directory):
    validate_sources(directory); price = pd.read_parquet(directory/'price-input.parquet')
    calendar = pd.read_parquet(directory/'calendar-input.parquet')
    dates = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d').sort_values().tolist()
    dates = [d for d in dates if '2005-01-05' <= d <= '2026-09-29']
    require(len(dates) == len(set(dates)) == 5280 and not price.duplicated(['date', 'instrument']).any(), 'Reference calendar/keys changed')
    high = price.assign(reference_high=price.high_adj.where(price.is_trading)).pivot(index='date', columns='instrument', values='reference_high').reindex(dates)
    values = []; scopes = []
    for label, subset in [('declared500', high.loc[:'2020-03-20'].tail(500)), ('extended', high)]:
        ns = selected(directory, 1, ['alpha2'], {'np': np, 'pd': pd,
            'jq': SimpleNamespace(get_price=lambda *a, **kw: {'high': subset})}, {'alpha2': 0})
        actual = ns['alpha2'](list(high.columns), subset.index[-1], len(subset)); finite = 0; blocked = 0
        for k in range(1, len(subset)):
            for stock in subset.columns:
                a, b = float(subset.iloc[k-1][stock]), float(subset.iloc[k][stock]); got = float(actual.iloc[k-1][stock])
                valid = math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0
                expected = (12./b - 12./a)**2 if valid else math.nan
                require((math.isnan(got) and not valid) or (valid and math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-14)), 'Independent alpha2 mismatch')
                finite += int(valid); blocked += int(not valid)
                values.append({'scope': label, 'date': subset.index[k], 'instrument': stock, 'valid': valid,
                    'previous_high': a, 'high': b, 'value': got, 'independent': expected})
        scopes.append({'scope': label, 'start': subset.index[0], 'end': subset.index[-1], 'rows': len(subset),
            'finite_endpoint_pairs': finite, 'blocked_endpoint_pairs': blocked, 'independent_differences': 0})
    return pd.DataFrame(values), {'scopes': scopes, 'reference_stocks': list(high.columns), 'not_a_backtest': True,
        'price_basis': 'Previously accepted back-adjusted high per stock, no inferred missing/suspended values; original platform adjustment equivalence unproved',
        'reference_formula': 'First alpha2 occurrence: square of adjacent difference in12/high; later occurrence overrides it in the full notebook',
        'original_training_runs': 0, 'original_timing_signals': 0, 'new_trading_backtests': 0, 'cost_backtests': [],
        'limits': ['Overlapping scopes and ten reference stocks are not independent GP validation or original CSI800',
            'First alpha2 is a hardcoded research expression; no claim it was reproduced by training',
            'Default91 uses missing CSV dates, not HMA; no substituted timing result',
            'Label maturity/version, cross-stock operators, current library compatibility and ETF execution gates remain']}


def study(root, directory):
    binding(root, directory); before = implementation(); diagnostic = diagnostics(directory); frame, result = compute(directory)
    require(before == implementation(), 'Implementation changed during study')
    path = directory/'alpha2-values.parquet'
    with path.open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diagnostic); save(directory/'component-research.json', {**result, 'implementation_sha256': before})
    return archive(root, 'Batch83 original CSV/state/GP defect diagnostics and longest hardcoded alpha2 arithmetic archived; training/trades0')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch83 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); frame, result = compute(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory/'alpha2-values.parquet'), check_exact=True)
    require(before == implementation() and diag == read(directory/'diagnostics.json') and
        {**result, 'implementation_sha256': before} == read(directory/'component-research.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'diagnostic_cases': len(diag['cases']), 'not_a_backtest': True})
    return archive(root, 'Batch83 forbidden-network exact diagnostics/alpha2/inventory/three catalogs match0; missing CSVs remain blocked')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch83.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch83_research.py', 'tests/unit/test_batch82_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(all((directory/n).is_file() for n in CORE), 'Core evidence missing')
    folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for number, command in enumerate(commands()):
        path = folder/f'check-{number}.log'
        with path.open('x') as stream: outcome = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': outcome.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(outcome.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        copied = directory/'implementation-checked'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and current == checked['implementation_sha256'] == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands(), 'Checks changed')
    for row in checked['commands']: require(row['returncode'] == 0 and file_sha(row['log']) == row['sha256'], 'Check log changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    result = read(directory/'component-research.json')
    require(result['original_training_runs'] == result['original_timing_signals'] == result['new_trading_backtests'] == 0 and
        result['cost_backtests'] == [] and result['not_a_backtest'] and off['result'] == 'match' and off['differences'] == 0 and
        off['socket_network_disabled'] and Store(root).published() == read(directory/'baseline.json')['published'], 'Scope/offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch83 external timing and GP source/arithmetic research accepted; original training/trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch83-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory/'diagnostics.json'), 'research': result,
        'inventory': read(directory/'dependency-inventory.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': progress['catalog_sha256']})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'newly_reviewed': 2, 'new_trading_backtests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'start', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
