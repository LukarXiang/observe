"""Audit the frozen Chan state machine; absent original 1m inputs gate trades."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import inspect
from io import StringIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch86 as previous
from scripts import review_strategy_batch82 as minute_audit
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch83 import plain, selected_nodes
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch87/20261009-chan-minute-state-machine')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2023年度精选策略/50.缠论交易策略.txt', '2024年度精选策略2/11.缠论交易策略.txt')
SOURCE_SHA = '454651949b62a7899aa0e35461a3bcca4bb77e27c012bc8b4d11e8fa7c1e0c77'
INSTRUMENT = '300354.SZ'
REFERENCES = (('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'),
    ('repo/backtrader', 'backtrader/indicators/ema.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch87.py', 'tests/unit/test_batch87_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'dependency-inventory.json', 'supplement-decision.json',
    'daily-reference.parquet', 'calendar-input.parquet', 'input-manifest.json', 'diagnostics.json',
    'component-research.json', 'state-values.parquet', 'indicator-values.parquet',
    'offline-verification.json', 'offline-catalog.json', 'minute-capabilities.json', 'tdx-probe.json',
    'development-date-failure/failure.json', 'development-date-failure/red-result.json',
    'development-capability-schema-failure/failure.json', 'development-capability-schema-failure/red-result.json',
    'diagnostics-corrected.json', 'diagnostic-label-correction.json'}
STATE_FIELDS = ('k_list', 'chan_k_list', 'fx_list', 'stroke_list', 'line_list', 'line_index',
    'pivot_list', 'trend_list', 'buy_list', 'sell_list', 'x_buy_list', 'x_sell_list', 'macd')


def implementation(): return {n: file_sha(n) for n in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch86-verification.json')
    review = Path('docs/handoff/2026-10-09-batch86-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted86 binding changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted86 evidence changed')
    for n in ('scripts/review_strategy_batch86.py', 'tests/unit/test_batch86_research.py'):
        require(file_sha(n) == accepted['implementation_sha256'][n], 'Parent entrypoint changed')
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review_file': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Input binding changed')
    return result


def definitions(tree):
    return [{'name': n.name, 'kind': type(n).__name__, 'line': n.lineno, 'end_line': n.end_lineno,
        'ast_sha256': hashlib.sha256(ast.dump(n).encode()).hexdigest()}
        for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef))]


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed before preflight')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch87 parent86 verified; original300354/1m state-machine audit begins')
    save(directory/'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(file_sha(path) == SOURCE_SHA and len(text.splitlines()) == 582, 'Source changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA,
            'encoding': encoding, 'complete_lines_read': 582, 'definitions': definitions(ast.parse(text)),
            'economic_changes': [], 'top_level_executed': False})
    save(directory/'sources-frozen.json', rows)
    store = Store(root); state = store.state(SNAPSHOT); filters = [('instrument', '=', INSTRUMENT)]
    frames = {t: store.load_state(state, t, filters=filters) for t in
        ('instruments', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions', 'minute_universe')}
    calendar = store.load_state(state, 'calendar'); manifest = {}
    for name, frame in {**frames, 'calendar': calendar}.items():
        path = directory/('daily-reference.parquet' if name == 'bars_1d' else name+'-input.parquet')
        with path.open('xb') as out: frame.to_parquet(out, index=False)
        manifest[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}
    save(directory/'input-manifest.json', manifest)
    inventory = {'snapshot': SNAPSHOT, 'instrument': INSTRUMENT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'instrument_tables': manifest, 'identity': plain(frames['instruments'].to_dict('records')),
        'daily_start': frames['bars_1d'].date.min(), 'daily_end': frames['bars_1d'].date.max(),
        'minute_original_rows_admitted': 0, 'not_a_backtest': True,
        'limits': ['No registered1m table; original instrument absent from5m sample and minute_universe',
            'Daily1392 rows start2021, not listing2012; daily reference does not recover1m or platform adjustment',
            'Current identity is not a complete historical name/status chain; corporate record/pay/list dates missing',
            'every_bar callback order/current completed-bar visibility and default slippage/participation unproved']}
    save(directory/'dependency-inventory.json', inventory)
    minute_receipt = read('docs/handoff/2026-10-09-batch82-verification.json')
    linked = {}
    for n in ('existing-apis.json', 'probe-results.json', 'dependency-inventory.json'):
        path = minute_audit.DIRECTORY/n
        require(file_sha(path) == minute_receipt['checks']['evidence_sha256'][n], 'Minute evidence changed')
        linked[n] = {'file': str(path), 'sha256': file_sha(path)}
    require(minute_audit.api_evidence() == read(minute_audit.DIRECTORY/'existing-apis.json'), 'Installed minute API changed')
    save(directory/'supplement-decision.json', {'prior_evidence': linked, 'new_requests': 0, 'new_rows': 0,
        'published': False, 'not_a_backtest': True,
        'decision': 'Installed Eastmoney1m API requests ndays=5; start/end filter response locally and cannot request listing history. Prior same-host DNS failure retained, changing symbol is not a historical capability.',
        'restore_required': ['Original300354 completed1m OHLC with timestamp/adjustment/coverage from explicit initial date',
            'Original every_bar callback order and get_bars include-now/paused/dynamic-price semantics',
            'Original signal duplicates/stroke-pop semantics accepted or concrete correction separately approved',
            'UniqueBook intraday execution: T+1, actual cash/position, events, prices/limits/fees and costs'],
        'economic_variants_require_confirmation': True})
    return {'status': 'ok', 'sources_frozen': 2, 'unique_contents': 1, 'not_a_backtest': True}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = read(directory/'sources-frozen.json'); refs = []
    for name, row in zip(SOURCES, rows, strict=True):
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == SOURCE_SHA, 'Source reviewed/changed')
        row['review'] = REVIEWS[name]; row['newly_reviewed'] = True
    (directory/'references').mkdir(exist_ok=False)
    for repo, name in REFERENCES:
        path = Path(repo)/name; copied = directory/'references'/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Native minute API limits, EMA seed contrast and incremental-order contract; original Chan class retained'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(old) == len(current) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(directory/'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'unique_contents': 1, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'snapshot': SNAPSHOT, 'not_a_backtest': True})
    return archive(root, 'Batch87 two582-line duplicate Chan sources frozen; original300354/1m trades blocked')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json'); frozen = read(directory/'sources-frozen.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(frozen) == 2 and doc['unique_contents'] == 1 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for name, row, original in zip(SOURCES, doc['sources'], frozen, strict=True):
        text, encoding = read_source(Path(row['source_copy']))
        require({k: row[k] for k in original} == original and file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA and
            row['review'] == REVIEWS[name] and encoding == row['encoding'] and len(text.splitlines()) == 582 and
            row['definitions'] == definitions(ast.parse(text)), 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference count changed')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    for row in read(directory/'input-manifest.json').values(): require(file_sha(row['file']) == row['sha256'], 'Input changed')


def namespace(directory):
    row = read(directory/'sources-frozen.json')[0]
    require(file_sha(row['source_copy']) == SOURCE_SHA, 'Selected source changed')
    tree = ast.parse(read_source(Path(row['source_copy']))[0]); nodes = []
    ns = {'np': np, 'copy': copy.copy, 'g': SimpleNamespace(security='300354.XSHE'),
        'log': SimpleNamespace(info=lambda *a: None), 'order': lambda *a: None}
    # Only the SHA-pinned, reviewed classes are evaluated; imports/top-level chan stay inert.
    for node in tree.body:
        if not isinstance(node, ast.ClassDef): continue
        require(node.name in ('BarData', 'Chan_Strategy') and not node.bases and not node.keywords and not node.decorator_list and not node.type_params, 'Unsafe class declaration')
        selected = []
        for member in node.body:
            if isinstance(member, ast.FunctionDef):
                temporary = {}; selected_nodes(ast.Module(body=[member], type_ignores=[]), [member.name], temporary)
                function = copy.deepcopy(member)
                function.returns = None
                for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs): arg.annotation = None
                selected.append(function)
            else:
                require(isinstance(member, ast.Assign) and all(isinstance(t, ast.Name) for t in member.targets) and
                    isinstance(member.value, (ast.Constant, ast.List)) and (not isinstance(member.value, ast.List) or not member.value.elts), 'Unsafe class field')
                selected.append(copy.deepcopy(member))
        new_class = copy.deepcopy(node); new_class.body = selected; nodes.append(new_class)
    require(len(nodes) == 2, 'Class count changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch87-classes>', 'exec'), ns)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    selected_nodes(tree, names, ns)
    ns['chan'] = ns['Chan_Strategy']()
    return ns


def state_value(value):
    if hasattr(value, 'high_price') and hasattr(value, 'low_price'): return plain(vars(value))
    if isinstance(value, dict): return {str(k): state_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [state_value(v) for v in value]
    return plain(value)


def state_digest(chan):
    state = {n: state_value(getattr(chan, n)) for n in STATE_FIELDS}
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bar(ns, k, high=12., low=10., close=11.):
    return ns['BarData'](dt.datetime(2021, 1, 4)+dt.timedelta(minutes=k), high, low, close_price=close)


def diagnostics(directory):
    ns = namespace(directory); rows = []
    def record(case, **values): rows.append({'case': case, **plain(values)})
    events = []
    ns.update(set_benchmark=lambda v: events.append(['benchmark', v]), set_option=lambda *v: events.append(['option', *v]),
        OrderCost=lambda **kw: kw, set_order_cost=lambda v, **kw: events.append(['cost', v, kw]),
        run_daily=lambda f, **kw: events.append(['schedule', f.__name__, kw]))
    ctx = SimpleNamespace(current_dt=dt.datetime(2021, 1, 4, 9, 31)); ns['initialize'](ctx); ns['before_market_open'](ctx)
    record('active_symbol_and_schedules', events=events, security=ns['g'].security)
    queries = []; sample = [(ctx.current_dt, 15., 16., 14., 15.5)]
    ns['get_bars'] = lambda *a, **kw: queries.append([a, kw]) or sample
    with redirect_stdout(StringIO()): ns['market_open'](ctx); ns['market_open'](ctx)
    record('market_open_duplicate_and_omitted_open', query=queries, bars=state_value(ns['chan'].k_list))
    ns['get_bars'] = lambda *a, **kw: None; ns['market_open'](ctx)
    record('none_history_no_reset', bars=len(ns['chan'].k_list))
    ns['initialize'](ctx); record('initialize_does_not_reset_chan', bars=len(ns['chan'].k_list))
    x = np.array([1., 2., 4., 8.]); record('indicator_seed_and_rounding', sma=ns['SMA'](x, 3), ema=ns['EMA'](x, 3), macd=ns['MACD'](x))
    for label, include, items in [('first_three_no_fx', True, [(10, 5), (12, 6), (11, 5)]),
        ('no_include_third_fx', False, [(10, 5), (12, 6), (11, 5)]),
        ('up_include', True, [(10, 5), (11, 6), (12, 7), (11.5, 7.5)]),
        ('down_include', True, [(12, 7), (11, 6), (10, 5), (9, 6)]),
        ('double_fractal_only_last_to_stroke', False, [(10, 5), (12, 3), (11, 4)])]:
        c = ns['Chan_Strategy'](include=include)
        with redirect_stdout(StringIO()):
            for k, (hi, lo) in enumerate(items): c.on_bar(bar(ns, k, hi, lo, (hi+lo)/2))
        record(label, chan=state_value(c.chan_k_list), fx=c.fx_list, strokes=c.stroke_list)
    c = ns['Chan_Strategy'](); c.k_list = [bar(ns, 10)]; calls = []; ns['order'] = lambda *a: calls.append(list(a))
    for price in (10., 1200., 2000., 6000., 10000.):
        c.k_list[-1].close_price = price
        with redirect_stdout(StringIO()):
            for kind in ('B1', 'S2'): c.on_buy_sell([bar(ns, 2).datetime, 11., kind, bar(ns, 10).datetime])
        record('increment_quantity_'+str(price), orders=calls[-2:], amount=abs(calls[-1][1])*price)
    signal = [bar(ns, 2).datetime, 11., 'B1', bar(ns, 10).datetime]
    c.k_list[-1].close_price = 10.
    with redirect_stdout(StringIO()): c.on_buy_sell(signal); c.on_buy_sell(signal); c.on_buy_sell(signal, False)
    record('duplicate_signal_and_invalid_no_reversal', last_orders=calls[-2:], buy=len(c.buy_list), invalid=c.x_buy_list)
    ns['order'] = lambda *a: None; c.k_list[-1].close_price = 0.
    with redirect_stdout(StringIO()):
        try: c.on_buy_sell(signal)
        except ZeroDivisionError: record('zero_price_appends_before_error', buy=len(c.buy_list), error='ZeroDivisionError')
        else: raise ValueError('Missing original zero-price failure')
    c = ns['Chan_Strategy'](); c.chan_k_list = [bar(ns, k, close=10.+k) for k in range(5)]
    record('macd_segment_seed_and_reverse_dates', value=c.cal_macd(bar(ns, 1).datetime, bar(ns, 4).datetime),
        reverse=c.cal_macd(bar(ns, 4).datetime, bar(ns, 1).datetime))
    c = ns['Chan_Strategy'](); c.stroke_list = [[20, 10, bar(ns, 0).datetime, 'up', 0],
        [10, 5, bar(ns, 4).datetime, 'down', 4]]
    middle = [22, 11, bar(ns, 5).datetime, 'up', 5]; extension = [9, 4, bar(ns, 8).datetime, 'down', 8]
    c.fx_list = [*copy.deepcopy(c.stroke_list), middle, extension]; c.cal_macd = lambda *a: 0
    with redirect_stdout(StringIO()): c.on_stroke(extension)
    record('stroke_correction_short_distance_pops', strokes=c.stroke_list)
    c = ns['Chan_Strategy'](); calls = []; ns['order'] = lambda *a: calls.append(list(a)); c.k_list = [bar(ns, 20)]
    data = [[10, 6, bar(ns, 0).datetime, 'up', 0], [8, 4, bar(ns, 4).datetime, 'down', 4],
        [12, 7, bar(ns, 8).datetime, 'up', 8], [9, 5, bar(ns, 12).datetime, 'down', 12]]
    pivot = [data[0][2], data[-1][2], 5., 10., data]; c.pivot_list=[pivot]
    c.macd = {data[0][2]: 2., data[2][2]: 1.}
    with redirect_stdout(StringIO()): c.on_trend(pivot, data)
    record('trend_signal_confirmation_not_pivot_date', signals=c.sell_list, orders=calls)
    c = ns['Chan_Strategy'](); c.stroke_list = [[30, 15, bar(ns, 0).datetime, 'down', 0],
        [20, 10, bar(ns, 4).datetime, 'up', 4], [10, 5, bar(ns, 8).datetime, 'down', 8]]
    middle = [22, 11, bar(ns, 9).datetime, 'up', 9]; extension = [9, 4, bar(ns, 12).datetime, 'down', 12]
    c.fx_list = [*copy.deepcopy(c.stroke_list), middle, extension]; c.cal_macd = lambda *a: 0
    with redirect_stdout(StringIO()): c.on_stroke(extension)
    record('stroke_correction_three_strokes_self_distance_pops', strokes=c.stroke_list)
    return {'cases': rows, 'not_a_backtest': True, 'messages_sent': 0, 'new_trading_backtests': 0,
        'policy': 'Only SHA-pinned class/function definitions with inert order/API stubs; annotations removed, bodies/defaults unchanged'}


def validate_diagnostic_correction(directory, corrected):
    original = read(directory/'diagnostics.json'); expected = copy.deepcopy(original)
    require(len(expected['cases']) == 20 and expected['cases'][18]['case'] == 'stroke_correction_zero_self_distance_pops', 'Unexpected original diagnostic')
    expected['cases'][18]['case'] = 'stroke_correction_short_distance_pops'
    expected['cases'].append(corrected['cases'][-1])
    require(corrected == expected and len(corrected['cases']) == 21 and
        corrected['cases'][-1]['case'] == 'stroke_correction_three_strokes_self_distance_pops' and
        len(corrected['cases'][-1]['strokes']) == 2 and corrected['cases'][-1]['strokes'][-1][4] == 9, 'Unexpected diagnostic correction')


def correct_diagnostics(root, directory):
    binding(root, directory); validate_sources(directory); corrected = diagnostics(directory)
    validate_diagnostic_correction(directory, corrected)
    save(directory/'diagnostics-corrected.json', corrected)
    save(directory/'diagnostic-label-correction.json', {'original_file': str(directory/'diagnostics.json'),
        'original_sha256': file_sha(directory/'diagnostics.json'), 'corrected_file': str(directory/'diagnostics-corrected.json'),
        'corrected_sha256': file_sha(directory/'diagnostics-corrected.json'), 'original_cases': 20, 'corrected_cases': 21,
        'reason': 'Initial two-stroke fixture actually proves index distance8-5=3, not self-distance0; label corrected and three-stroke fixture added. Original source,20-case record and state/indicator tables remain unchanged.',
        'not_a_backtest': True})
    return archive(root, 'Batch87 diagnostic label corrected with separate three-stroke self-distance proof; original20cases retained')


def minute_capabilities():
    from tdxpy.hq import TdxHq_API
    paths = ('docs/01-数据源实测报告.md', 'scripts/probe/probe_tdx.py', 'src/observe/data/sources/tdx.py',
        'repo/mootdx/mootdx/quotes.py', 'repo/mootdx/mootdx/consts.py', 'data/probe/tdx/result_scan.json')
    scan = read(paths[-1])['scan_hosts']
    return {'files': {n: file_sha(n) for n in paths},
        'native_methods': {n: inspect.getsource(getattr(TdxHq_API, n)) for n in
            ('connect', 'get_security_bars', 'get_history_minute_time_data', 'get_history_transaction_data')},
        'historical_host_scan': {'n_minutes_ok': scan['n_minutes_ok'], 'n_bars_ok': scan['n_bars_ok']},
        'limits': ['BaoStock does not expose1m; Eastmoney latest5days/Sina recent days cannot recover listing history',
            'TDX historical minutes contain minute price/volume, not original1m OHLC; transaction snapshots can only form a declared approximate series',
            'Earlier measured minute bar boundaries vary by provider; current original stock/date coverage requires direct audit',
            'Mootdx Quotes.factory writes global config; this probe uses its existing native tdxpy transport with no config changes',
            'Old all-server K-line empty evidence is historical, not a fresh claim all servers are still unavailable'],
        'not_a_backtest': True}


def probe(root, directory):
    binding(root, directory); validate_sources(directory)
    capabilities = minute_capabilities(); save(directory/'minute-capabilities.json', capabilities)
    copies = directory/'capability-source-copies'; copies.mkdir(exist_ok=False)
    for name in capabilities['files']:
        path = copies/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, path)
    from tdxpy.hq import TdxHq_API
    api = TdxHq_API(auto_retry=False, heartbeat=False, raise_exception=True)
    result = {'status': 'failed', 'host': ['123.125.108.14', 7709], 'connect_timeout_seconds': 5,
        'original_symbol': '300354', 'planned_queries': [
            {'method': 'get_security_bars', 'args': [8, 0, '300354', 0, 800]},
            {'method': 'get_history_transaction_data', 'args': [0, '300354', 0, 2000, 20120920]},
            {'method': 'get_history_transaction_data', 'args': [0, '300354', 0, 2000, 20260929]}],
        'tcp_attempts': 1, 'queries_started': [], 'responses': [], 'new_rows_admitted': 0,
        'published': False, 'not_a_backtest': True, 'capabilities_sha256': file_sha(directory/'minute-capabilities.json')}
    try:
        require(api.connect('123.125.108.14', 7709, time_out=5) is not None, 'Native TDX connection not established')
        for k, query in enumerate(result['planned_queries']):
            result['queries_started'].append(query)
            data = getattr(api, query['method'])(*query['args']); path = directory/f'tdx-response-{k}.json'
            save(path, data)
            result['responses'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(data or []), 'complete_day_proved': False})
        result['status'] = 'candidate' if any(r['rows'] for r in result['responses']) else 'empty'
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: api.disconnect(); save(directory/'tdx-probe.json', result)
    return archive(root, 'Batch87 bounded native TDX1m/historical snapshot capability probe archived; original coverage still gated')


def independent_indicators(values):
    def ema(x, period):
        result = []
        for value in x:
            result.append(float(value) if not result else (2.*float(value)+(period-1)*result[-1])/(period+1))
        return np.round(result, 4)
    slow = ema(values, 26); fast = ema(values, 12); diff = fast-slow; dea = ema(diff, 9)
    sma = np.round([sum(map(float, values[max(0, k-4):k+1]))/min(k+1, 5) for k in range(len(values))], 4)
    return sma, ema(values, 5), np.round(diff, 4), dea, np.round(2*(diff-dea), 4)


def reference_inclusion(merged, incoming, include):
    merged = copy.deepcopy(merged)
    run_fx = not include or len(merged) >= 3
    if include and len(merged) >= 3:
        prev, last = merged[-2:]; hi, lo = incoming[0:2]; last_hi, last_lo = last[0:2]
        contained = min(last_hi, hi) == hi and max(last_lo, lo) == lo
        contains = min(last_hi, hi) == last_hi and max(last_lo, lo) == last_lo
        if contained or contains:
            merged[-1] = [max(last_hi, hi), max(last_lo, lo), incoming[2]] if last_hi > prev[0] else [min(last_hi, hi), min(last_lo, lo), incoming[2]]
        else: merged.append(incoming)
    else: merged.append(incoming)
    fx = []
    if run_fx and len(merged) >= 3:
        left, mid, right = merged[-3:]
        if mid[0] > max(left[0], right[0]): fx.append([*mid[:2], mid[2], 'up', len(merged)-2])
        if mid[1] < min(left[1], right[1]): fx.append([*mid[:2], mid[2], 'down', len(merged)-2])
    return merged, fx


def component(directory):
    validate_sources(directory); frames = read(directory/'input-manifest.json'); price = pd.read_parquet(frames['bars_1d']['file']).sort_values('date')
    price = price.assign(date=pd.to_datetime(price.date).dt.strftime('%Y-%m-%d'))
    calendar = pd.read_parquet(frames['calendar']['file'])
    dates = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d')
    require(len(price) == 1392 and not price.duplicated(['date', 'instrument']).any() and set(price.instrument) == {INSTRUMENT}, 'Reference keys changed')
    require(price.date.tolist() == dates[(dates >= price.date.min()) & (dates <= price.date.max())].tolist(), 'Daily missing sessions')
    active = price[price.is_trading].copy(); require(np.isfinite(active[['high', 'low', 'close']]).all().all() and
        active.low.gt(0).all() and active.high.ge(active.low).all() and active.close.between(active.low, active.high).all(), 'Invalid reference OHLC')
    ns = namespace(directory); closes=active.close.to_numpy(dtype=float); expected = independent_indicators(closes)
    actual = (ns['SMA'](closes), ns['EMA'](closes), *ns['MACD'](closes))
    names = ('sma5', 'ema5', 'diff', 'dea', 'macd'); ind = active[['date', 'instrument', 'close']].reset_index(drop=True); maxima = {}
    for name, got, want in zip(names, actual, expected, strict=True):
        maxima[name] = float(np.max(np.abs(got-want))); require(maxima[name] <= 0.00010000001, 'Independent indicator differs')
        ind[name] = got; ind[name+'_independent'] = want
    states=[]; scopes=[]
    for include, build_pivot in ((True, False), (False, False), (True, True)):
        c=ns['Chan_Strategy'](include=include, build_pivot=build_pivot); orders=[]; merged=[]; fx=[]; checkpoint_hashes={}
        ns['order'] = lambda *a: orders.append(list(a)); label=f'include{include}-line{build_pivot}'
        with redirect_stdout(StringIO()):
            for k, row in enumerate(active.itertuples()):
                timestamp=dt.datetime.fromisoformat(row.date); before_order=len(orders)
                c.on_bar(ns['BarData'](timestamp, row.high, row.low, close_price=row.close))
                merged, added=reference_inclusion(merged, [row.high, row.low, timestamp], include); fx.extend(added)
                require(plain([[b.high_price, b.low_price, b.datetime] for b in c.chan_k_list]) == plain(merged) and
                    plain(c.fx_list) == plain(fx), 'Independent inclusion/fractal differs')
                require(all(s[3] <= timestamp and s[0] <= s[3] for s in c.buy_list+c.sell_list), 'Signal confirmation in future')
                new_orders=orders[before_order:]; new_signals=(c.buy_list+c.sell_list)
                require(len(orders) == len(new_signals) and all(o[0]=='300354.XSHE' for o in new_orders), 'Signal/order mapping differs')
                for o in new_orders: require(abs(o[1]) == round(3000/row.close)*100, 'Increment quantity differs')
                digest=state_digest(c)
                states.append({'scope': label, 'date': row.date, 'observed_rows': k+1, 'state_sha256': digest,
                    **{n+'_count': len(getattr(c,n)) for n in STATE_FIELDS},
                    'new_orders_json': json.dumps(new_orders), 'independent_inclusion_fractal_match': True})
                if k+1 in (3, 4, 100, 500, len(active)): checkpoint_hashes[k+1]=digest
            # Fresh prefix runs cannot see any suffix; compare complete mutable state, not just final chart.
            for count, digest in checkpoint_hashes.items():
                fresh=ns['Chan_Strategy'](include=include, build_pivot=build_pivot); ns['order']=lambda *a: None
                for row in active.iloc[:count].itertuples():
                    fresh.on_bar(ns['BarData'](dt.datetime.fromisoformat(row.date), row.high, row.low, close_price=row.close))
                require(state_digest(fresh) == digest, 'Prefix state changed with future suffix')
        scopes.append({'scope': label, 'rows': len(active), 'first': active.date.min(), 'last': active.date.max(),
            'reference_orders': len(orders), 'buy_signals': len(c.buy_list), 'sell_signals': len(c.sell_list),
            'prefix_counts_verified': sorted(checkpoint_hashes), 'final_state_sha256': state_digest(c),
            'independent_inclusion_fractal_differences': 0,
            'final_state': {n: state_value(getattr(c,n)) for n in STATE_FIELDS if n not in ('k_list','chan_k_list')}})
    summary={'instrument': INSTRUMENT, 'frequency': '1d_reference_only', 'scopes': scopes,
        'daily_rows':len(price), 'known_paused_rows_skipped':len(price)-len(active), 'indicator_maximum_absolute_differences':maxima,
        'state_rows':len(states), 'indicator_rows':len(ind), 'original_1m_rows':0, 'new_trading_backtests':0, 'cost_backtests':[],
        'not_a_backtest':True, 'price_basis':'Frozen raw daily OHLC, no future adjustment, no claim of platform dynamic real-price equivalence',
        'limits':['Reference1d confirms component behavior, never original1m signals or returns; only includeTrue/lineFalse is active default',
            'Other two settings are declared diagnostic branches, overlapping input is not independent validation',
            'Independent coverage is indicators/inclusion/fractals/quantity and confirmation bounds; stroke/line/pivot/trend are original full-class replay, not independently equivalent platform engine',
            'No same-bar fills, cash/position/PnL calculated; uniqueBook unchanged; missing1m and execution gates retained']}
    return plain(summary), pd.DataFrame(states), ind


def study(root, directory):
    binding(root, directory); before=implementation(); diag=diagnostics(directory); summary, states, indicators=component(directory)
    require(before==implementation(), 'Implementation changed during study')
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', summary)
    for name, frame in (('state-values',states),('indicator-values',indicators)):
        with (directory/(name+'.parquet')).open('xb') as out: frame.to_parquet(out,index=False)
    return archive(root,'Batch87 full original Chan class daily reference and controlled boundaries archived; original1m trades blocked')


def offline(root, directory):
    binding(root,directory); before=implementation()
    def forbidden(*a,**kw): raise AssertionError('Network forbidden')
    with patch.object(socket.socket,'connect',forbidden), patch.object(socket,'create_connection',forbidden), patch.object(requests.sessions.Session,'request',forbidden):
        diag=diagnostics(directory); summary,states,indicators=component(directory); catalog=offline_catalog(root,directory)
        require(minute_capabilities() == read(directory/'minute-capabilities.json'), 'Minute capabilities changed')
        for name, sha in read(directory/'minute-capabilities.json')['files'].items():
            require(file_sha(directory/'capability-source-copies'/name) == sha, 'Capability copy changed')
    validate_diagnostic_correction(directory, diag)
    correction = read(directory/'diagnostic-label-correction.json')
    require(file_sha(directory/'diagnostics.json') == correction['original_sha256'] and
        file_sha(directory/'diagnostics-corrected.json') == correction['corrected_sha256'], 'Diagnostic correction binding changed')
    require(diag==read(directory/'diagnostics-corrected.json') and summary==read(directory/'component-research.json'), 'Offline JSON differs')
    for name,frame in (('state-values',states),('indicator-values',indicators)):
        pd.testing.assert_frame_equal(frame,pd.read_parquet(directory/(name+'.parquet')),check_exact=True)
    require(before==implementation(),'Implementation changed during offline')
    save(directory/'offline-catalog.json',catalog); validate_catalog(root,directory)
    save(directory/'offline-verification.json',{'result':'match','differences':0,'socket_network_disabled':True,
        'implementation_sha256':before,'catalog':catalog,'state_rows':len(states),'indicator_rows':len(indicators),'not_a_backtest':True})
    return archive(root,'Batch87 forbidden-network original class reference/diagnostics and three catalogs match0; original trades blocked')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests','scripts/review_strategy_batch87.py'],
        [sys.executable,'-m','scripts.verify_offline_tests','-q','tests/unit/test_batch87_research.py',
            'tests/unit/test_batch86_research.py','tests/unit/test_catalog_dependencies.py','tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py','tests/integration/test_strategy_merge.py'],['git','diff','--check']]


def checks(root,directory):
    binding(root,directory); validate_sources(directory); validate_catalog(root,directory); before=implementation(); rows=[]
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()},'Core missing')
    folder=directory/'checks'; folder.mkdir(exist_ok=False)
    for k,command in enumerate(commands()):
        path=folder/f'check-{k}.log'
        with path.open('x') as out: process=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
        rows.append({'command':command,'returncode':process.returncode,'log':str(path),'sha256':file_sha(path)})
        require(process.returncode==0,'Check failed: '+str(path))
    require(before==implementation(),'Implementation changed during checks')
    for name in FILES:
        path=directory/'implementation-checked'/name; path.parent.mkdir(parents=True,exist_ok=True)
        require(not path.exists(),'Checked copy exists'); shutil.copyfile(name,path)
    save(directory/'checked-state.json',{'status':'passed','commands':rows,'implementation_sha256':before,
        'evidence_sha256':{p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status':'passed'}


def finish(root,directory):
    binding(root,directory); validate_sources(directory); validate_catalog(root,directory)
    checked=read(directory/'checked-state.json'); off=read(directory/'offline-verification.json'); current=implementation()
    require(checked['status']=='passed' and current==checked['implementation_sha256']==off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']]==commands(),'Checks changed')
    for row in checked['commands']: require(row['returncode']==0 and file_sha(row['log'])==row['sha256'],'Log changed')
    for name,sha in checked['evidence_sha256'].items(): require(file_sha(directory/name)==sha,'Evidence changed')
    for name,sha in current.items(): require(file_sha(directory/'implementation-checked'/name)==sha,'Checked copy changed')
    require(off['result']=='match' and off['differences']==0 and off['socket_network_disabled'] and
        Store(root).published()==read(directory/'baseline.json')['published'],'Offline/publication changed')
    protection=protect(root,directory); progress=archive(root,'Batch87 two duplicate Chan sources and original class reference accepted; original1m trades blocked')
    receipt=Path('docs/handoff/2026-10-09-batch87-verification.json')
    save(receipt,{'status':'ok','snapshot':SNAPSHOT,'reviews':read(directory/'source-reviews/review.json'),
        'checks':checked,'offline':off,'diagnostics':read(directory/'diagnostics-corrected.json'),
        'diagnostic_correction':read(directory/'diagnostic-label-correction.json'),
        'research':read(directory/'component-research.json'),'inventory':read(directory/'dependency-inventory.json'),
        'minute_capabilities':read(directory/'minute-capabilities.json'),'tdx_probe':read(directory/'tdx-probe.json'),
        'supplement':read(directory/'supplement-decision.json'),'protection':protection,'progress':progress,
        'newly_reviewed':2,'unique_contents':1,'not_a_backtest':True,'new_trading_backtests':0,'original_strategy_complete':False})
    save(directory/'final-binding-verification.json',{'status':'ok','receipt_sha256':file_sha(receipt),
        'checked_state_sha256':file_sha(directory/'checked-state.json'),'final_script_sha256':file_sha(__file__),
        'catalog_sha256':progress['catalog_sha256']})
    return {'status':'ok','checkpoint':progress['checkpoint'],'newly_reviewed':2,'new_trading_backtests':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','start','probe','study','correct_diagnostics','offline','checks','finish'])
    parser.add_argument('--root',default='data'); parser.add_argument('--directory',type=Path,default=DIRECTORY)
    args=parser.parse_args(); print(json.dumps(globals()[args.action](args.root,args.directory),ensure_ascii=False,default=str))
