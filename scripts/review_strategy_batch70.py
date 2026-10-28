"""Research cross-sectional futures momentum and sugar turtle without invented fills."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
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
import requests
import talib

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts import review_strategy_batch69 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch70/20261008-cross-momentum-sugar-turtle')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY
RECEIPT = Path('docs/handoff/2026-10-08-batch69-verification.json')
SOURCES = ('2020年度精选策略/17 商品期货截面动量模型，真的靠谱吗？.txt',
    '2020年度精选策略/49 海龟克隆优化.txt')
SOURCE_SHA = ('7b0c7ee966f0fa2c5cbc7ad497c173f40361488caca648e5b0395eb7bd123f33',
    '3b1b81d92397c967b3d3e8b0b264f488183017f330bebefe749cfb4bf127640d')
POOL = ('AL', 'NI', 'CU', 'PB', 'AG', 'RU', 'MA', 'PP', 'TA', 'L', 'V', 'M', 'P', 'Y',
    'OI', 'C', 'CS', 'JD', 'SR', 'HC', 'J', 'I', 'SF', 'RB', 'ZC', 'FG')
QUERIES = {ins: {'symbol': ins+'0'} for ins in POOL if ins not in ('CU', 'RB')}
QUERIES['SR2001'] = {'symbol': 'SR2001'}
REFERENCES = (('repo/rqalpha', 'rqalpha/examples/turtle.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_future.py'),
    ('repo/backtrader', 'backtrader/indicators/atr.py'),
    ('repo/akshare', 'akshare/futures/futures_zh_sina.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch70.py',
    'tests/unit/test_batch70_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'diagnostics.json', 'probe-results.json', 'probe-audit.json', 'component-values.parquet',
    'cross-section-values.parquet', 'component-research.json', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch69 changed')
    reused = {}
    for ins, case in (('RB', 'rb_daily'), ('CU', 'cu_daily')):
        row = read(ACCEPTED / previous.PROBE_FOLDER / case / 'result.json')
        require(row['status'] == 'success' and len(row['files']) == 1, 'Accepted continuous sample missing')
        item = row['files'][0]; require(file_sha(item['file']) == item['sha256'], 'Accepted sample changed')
        reused[ins] = row
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'reused_supplier_samples': reused, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Binding changed')
    return result


def parsed_source(text):
    lines = text.splitlines(); require(lines[6] == '原文策略源码如下：', 'Article prefix changed')
    lines[:8] = ['']*8
    return ast.parse('\n'.join(lines))


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return parsed_source(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch70-functions>', 'exec'), namespace)
    return namespace


def api_evidence():
    return {'supplier_api': previous.api_evidence(), 'queries': QUERIES,
        'reused_continuous_symbols': ['RB0', 'CU0'],
        'network_policy': '25 new cases, max one GET each, sequential, timeout8/10, parent45; no proxies/redirects/retries',
        'limits': ['26 supplier continuous symbols do not prove original 8888 index construction or historical dominant mapping',
            'SR2001 requested symbol needs expiry/identity/units/availability evidence; a response alone is not a complete tradable history']}


def inventory(root):
    return {**previous.inventory(root), 'source_pool': list(POOL),
        'additional_gates': ['Original historical 26-product daily main/index availability and stable tied ranks unproved',
            'Sugar original main-contract 21-bar/expiry/position.price and rollover semantics unproved',
            'No approved economic changes to channel19, simple ATR20, risk units, stale state or close/rollover branches']}


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch70 cross-momentum/sugar-turtle protection baseline started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    folder = directory / 'source-reviews'; folder.mkdir(); frozen = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; require(file_sha(path) == sha, 'Source SHA changed')
        shutil.copyfile(path, folder / f'{strategy_id(name)}.source'); frozen.append({'path': name, 'sha256': sha})
    save(directory / 'sources-frozen.json', frozen)
    return {'status': 'ok'}


def start(root, directory):
    binding(root, directory)
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = []; refs = []
    for i, name in enumerate(SOURCES):
        original = Path('repo/量化策略源代码') / name; copied = directory / 'source-reviews' / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and
            file_sha(original) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        text, encoding = read_source(copied)
        rows.append({'source_path': str(original), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(parsed_source(text)),
            'original_line_numbers_retained': True, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only risk/quantity/ATR/API comparison; not replacement ledger or platform equivalence'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch70 two complete commodity sources reviewed; original pool/main execution remains gated')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            row['source_sha256'] == file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(parsed_source(text)) and
            row['original_line_numbers_retained'], 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and
            file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; schedules = []; declarations = []
    log = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 2, 9, 0),
        portfolio=SimpleNamespace(starting_cash=1000000., total_value=1000000., positions={}, long_positions={}, short_positions={}))
    def namespace(number):
        ns = {'g': SimpleNamespace(), 'np': np, 'talib': talib, 'isnan': math.isnan, 'log': log,
            'order_target': lambda *a, **kw: orders.append({'kind': 'target', 'args': list(a), 'kwargs': kw}),
            'order': lambda *a, **kw: orders.append({'kind': 'increment', 'args': list(a), 'kwargs': kw}),
            'get_dominant_future': lambda ins: ins+'_OLD', 'send_message': lambda *a: None,
            'set_benchmark': lambda x: declarations.append(['benchmark', x]),
            'set_option': lambda *a: declarations.append(['option', *a]),
            'set_subportfolios': lambda x: declarations.append(['subportfolios', x]),
            'set_order_cost': lambda x, **kw: declarations.append(['cost', x, kw]),
            'set_slippage': lambda x, **kw: declarations.append(['slippage', x, kw]),
            'SubPortfolioConfig': lambda **kw: kw, 'OrderCost': lambda **kw: kw, 'PriceRelatedSlippage': lambda x: x,
            'run_daily': lambda fn, **kw: schedules.append(['daily', fn.__name__, kw]),
            'run_weekly': lambda fn, day, **kw: schedules.append(['weekly', fn.__name__, day, kw])}
        names = [n.name for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef)]
        return selected(directory, number, names, ns)
    def error(case, fn, kind):
        try: fn()
        except kind as exc: rows.append({'case': case, 'error': type(exc).__name__, 'orders': orders.copy()})
        else: raise ValueError('Expected original failure: '+case)
    cross = namespace(0); turtle = namespace(1); cross['initialize'](context)
    g = cross['g']; require(tuple(g.instruments) == POOL and g.BackWindow == 15 and g.Range == .3, 'Cross defaults differ')
    rows.append({'case': 'cross_defaults_26_weekly7_margin_cost', 'instruments': g.instruments.copy(),
        'schedules': schedules.copy(), 'declarations': declarations.copy(), 'rank_slots_each_side': 7})
    orders.clear(); g.Momentum = {}; error('cross_no_momentum_fixed_pool_rank_IndexError', lambda: cross['Trade'](context), IndexError)
    g.Momentum = {ins: float(i) for i, ins in enumerate(POOL[:6])}
    error('cross_six_candidates_still_requires_seven', lambda: cross['Trade'](context), IndexError)
    def trade_state(momentum):
        g.instruments = list(POOL); g.MappingReal = {ins: ins+'_OLD' for ins in POOL}
        g.MappingIndex = {ins: cross['get_future_code'](ins) for ins in POOL}; g.Momentum = momentum
        g.LastRealPrice = {ins+'_OLD': 100. for ins in momentum}; g.TradeLots = {ins+'_OLD': 2.5 for ins in momentum}
        g.ClosePrice = {}; g.Reentry_long = g.Reentry_short = True
        context.portfolio.long_positions = {ins+'_OLD': SimpleNamespace(total_amount=0) for ins in POOL}
        context.portfolio.short_positions = {ins+'_OLD': SimpleNamespace(total_amount=0) for ins in POOL}
        orders.clear()
    trade_state({ins: float(i) for i, ins in enumerate(POOL[:7])}); cross['Trade'](context)
    openings = [r for r in orders if r['args'][1] != 0]
    require(len(openings) == 7 and all(r['kwargs']['side'] == 'long' for r in openings), 'Overlapping lists buy priority differs')
    rows.append({'case': 'cross_seven_candidates_overlap_all_buy_priority', 'open_intents': openings.copy(),
        'global_reentry_true_did_not_block': g.Reentry_long and g.Reentry_short})
    trade_state({ins: 0. for ins in POOL}); cross['Trade'](context)
    openings = [r for r in orders if r['args'][1] != 0]
    require([r['args'][0] for r in openings if r['kwargs']['side'] == 'long'] == [ins+'_OLD' for ins in POOL[:7]], 'Stable tied buy order differs')
    require([r['args'][0] for r in openings if r['kwargs']['side'] == 'short'] == [ins+'_OLD' for ins in POOL[-7:]], 'Stable tied short membership differs')
    rows.append({'case': 'cross_equal_momentum_stable_insertion_and_original_trade_order', 'open_intents': openings.copy()})
    g.instruments = ['CU']; g.MappingReal = {'CU': 'CU_OLD'}; g.MappingIndex = {'CU': 'CU8888.XSGE'}
    g.ATR = {}; g.Momentum = {}; requested = []
    def history(code, count, *a):
        requested.append([code, count]); close = np.arange(count, dtype=float)+100
        return pd.DataFrame({'open': close, 'close': close, 'high': close+2, 'low': close-1}, index=range(-count, 0))
    cross['get_CCFX_end_date'] = lambda *a: dt.date(2099, 1, 1); cross['attribute_history'] = history
    cross['market_open'](context)
    require(g.CurrentPrice['CU'] == 119. and g.Price_DaysAgo['CU'] == 105. and
        math.isclose(g.Momentum['CU'],119./105.-1), 'Cross14interval differs')
    rows.append({'case': 'cross_20rows_14interval_return_ATR15', 'momentum': float(g.Momentum['CU']), 'ATR': float(g.ATR['CU8888.XSGE'])})
    g.Times = {'CU_OLD': 0}; g.HighPrice = {'CU_OLD': 100.}; g.LowPrice = {'CU_OLD': False}; g.LastRealPrice = {'CU_OLD': 96.}
    context.portfolio.long_positions = {'CU_OLD': SimpleNamespace(total_amount=1)}; context.portfolio.short_positions = {}
    orders.clear(); cross['TrailingStop'](context,'CU_OLD','CU8888.XSGE')
    require(len(orders) == 1 and g.Times['CU_OLD'] == 0, '1ATR stop or unchanged counter differs')
    rows.append({'case': 'cross_1ATR_stop_is_not_percent_and_counter_stays_zero', 'orders': orders.copy()})
    requested.clear(); cross['get_CCFX_end_date'] = lambda *a: context.current_dt.date(); old = g.Momentum.copy()
    cross['market_open'](context); require(not requested and g.Momentum == old, 'Stale momentum expiry differs')
    rows.append({'case': 'cross_expiry_retains_previous_momentum', 'momentum': g.Momentum.copy()})
    g.MappingReal = {'CU': 'CU_OLD'}; context.portfolio.long_positions = {}; context.portfolio.short_positions = {'CU_OLD': SimpleNamespace(total_amount=2)}
    orders.clear(); error('cross_short_roll_reads_new_key', lambda: cross['replace_old_futures'](context,'CU','CU_NEW'), KeyError)
    g.ATR = {}; cross['attribute_history'] = history
    require(cross['get_lots'](1000000./26,'CU') == (1000000./26)*.0001/5, 'Initial fractional lot differs')
    rows.append({'case': 'cross_initial_fractional_lot_and_unused_open', 'lots': cross['get_lots'](1000000./26,'CU')})
    schedules.clear(); declarations.clear(); turtle['initialize'](context); tg = turtle['g']
    require((tg.future_index,tg.window,tg.limit_unit) == ('SR',20,6), 'Turtle defaults differ')
    rows.append({'case': 'turtle_SR20_limit6_schedule_and_no_explicit_slippage', 'schedules': schedules.copy(), 'declarations': declarations.copy()})
    frame = history('SR',21); frame.iloc[19, frame.columns.get_loc('high')] = 10000.
    require(turtle['check_break'](frame,121.,20) == 1, 'Channel must exclude penultimate extreme')
    rows.append({'case': 'turtle_channel_19rows_excludes_last_two', 'upper': float(frame.high.iloc[:-2].max()), 'penultimate_high': 10000., 'break': 1})
    require(turtle['check_break'](frame,float(frame.high.iloc[:-2].max()),20) == 0, 'Strict upper equality differs')
    rows.append({'case': 'turtle_channel_equal_upper_no_break'})
    frame = history('SR',21); atr = turtle['get_ATR'](frame,20)
    require(atr == 3., 'Simple mean ATR differs')
    rows.append({'case': 'turtle_ATR20_mean_no_first_row_wrap', 'ATR': float(atr)})
    error('turtle_nonempty_short_history_not_guarded', lambda: turtle['get_ATR'](frame.iloc[:5],20), IndexError)
    decisions = [turtle['get_next_signal'](price,100.,3.,side) for price,side in ((101.5,1),(94.,1),(98.5,-1),(106.,-1))]
    require(decisions == [1,-1,1,-1], 'Inclusive add/stop thresholds differ')
    rows.append({'case': 'turtle_half_ATR_add_and_two_ATR_stop_inclusive', 'signals': decisions})
    quantity = turtle['get_unit'](1000000.,3.,'SR'); require(not float(quantity).is_integer(), 'Quantity must stay fractional')
    rows.append({'case': 'turtle_equity_1percent_SR10_unrounded', 'quantity': float(quantity)})
    context.portfolio.positions = {'SR_OLD': SimpleNamespace(total_amount=1)}
    context.portfolio.long_positions = {'SR_OLD': SimpleNamespace(total_amount=1,price=100.)}; context.portfolio.short_positions = {}
    require(turtle['get_position'](context) == 0, 'dict_keys failure should be swallowed')
    rows.append({'case': 'turtle_unused_get_position_dict_keys_returns_zero'})
    tg.position = 1; tg.price_mark = 100.; context.portfolio.long_positions['SR_OLD'].price = 95.
    require(turtle['get_risk_signal'](context,'SR_OLD') is True, 'Inclusive95% risk differs')
    rows.append({'case': 'turtle_position_price95percent_inclusive', 'price_field_semantics_proved': False})
    tg.position = 1; tg.add_time = 3; tg.last_price = 100.; tg.price_mark = 100.; tg.last_future = 'SR_OLD'; tg.future = 'SR_NEW'
    turtle['get_CCFX_end_date'] = lambda *a: context.current_dt.date(); orders.clear(); turtle['while_open'](context)
    require(len(orders) == 1 and tg.position == 0 and tg.add_time == 0 and tg.last_price == 100. and tg.price_mark == 100., 'Roll reset scope differs')
    rows.append({'case': 'turtle_roll_equality_typo_followed_by_real_reset_before_expiry', 'orders': orders.copy(),
        'position': tg.position, 'last_price_retained': tg.last_price, 'mark_retained': tg.price_mark, 'last_future': tg.last_future})
    tg.future = tg.last_future = 'SR_OLD'; tg.position = 1; tg.add_time = 1; tg.last_price = 100.; tg.price_mark = 100.
    turtle['get_CCFX_end_date'] = lambda *a: dt.date(2099,1,1); turtle['attribute_history'] = lambda *a: history('SR',21)
    turtle['check_break'] = lambda *a: 0; context.portfolio.long_positions['SR_OLD'].price = 100.
    orders.clear(); turtle['while_open'](context)
    require(len(orders) == 1 and tg.add_time == 2 and tg.last_price == 120., 'None addition state differs')
    rows.append({'case': 'turtle_None_add_intent_still_increments_counter', 'orders': orders.copy(), 'add_time': tg.add_time, 'last_price': tg.last_price})
    tg.position = 1; tg.add_time = 1; tg.last_price = 130.; orders.clear(); turtle['while_open'](context)
    require(tg.position == 0 and tg.add_time == 0 and orders[0]['args'][1] == 0, 'None stop state differs')
    rows.append({'case': 'turtle_None_ATR_stop_still_resets_internal_position', 'orders': orders.copy(), 'position': tg.position})
    require(len(rows) == 22, 'Diagnostic count differs')
    return {'cases': rows, 'not_a_backtest': True, 'synthetic_fixture': True, 'platform_equivalent': False,
        'real_historical_trade_windows': 0,
        'limits': ['Explicit numpy/platform mocks and negative integer labels are diagnostic scaffolding only',
            'Orders return None and never fill, debit costs or generate NAV',
            'position.price semantics, source platform versions and real historical main contracts remain unproved']}


def study(root, directory):
    binding(root,directory)
    with redirect_stdout(StringIO()): result = diagnostics(directory)
    save(directory / 'diagnostics.json',result)
    return archive(root,'Batch70 original rank/ATR/channel/rollover intention diagnostics archived; no trades')


def worker(directory, case):
    folder = directory / 'probes' / case; folder.mkdir(parents=True,exist_ok=False)
    row = {'case': case, 'function': 'futures_zh_daily_sina', 'parameters': QUERIES[case], 'status': 'failed',
        'attempts': [], 'responses': [], 'files': [], 'published': False, 'strict_usable': False,
        'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session,method,url,**kw):
        require(not row['attempts'] and method.upper() == 'GET' and url == previous.SINA_URL and
            kw.get('params') == {'symbol':QUERIES[case]['symbol'],'type':'2021_04_12'}, 'Unexpected or repeated request')
        row['attempts'].append({'method':method,'url':url,'parameters':kw.get('params')})
        session.trust_env = False; kw['timeout'] = (8,10); kw['allow_redirects'] = False
        response = original(session,method,url,**kw); path = folder / 'response.bin'
        with path.open('xb') as out: out.write(response.content)
        row['responses'].append({'file':str(path),'sha256':file_sha(path),'status_code':response.status_code,
            'url':response.url,'encoding':response.encoding}); response.raise_for_status(); return response
    try:
        import akshare as ak
        require(api_evidence() == read(directory / 'existing-apis.json'),'Installed API/settings changed')
        with patch.object(requests.sessions.Session,'request',request),redirect_stdout(sys.stderr): frame = ak.futures_zh_daily_sina(**QUERIES[case])
        path = folder / 'parsed.parquet'; frame.to_parquet(path,index=False)
        row['files'].append({'file':str(path),'sha256':file_sha(path),'rows':len(frame),'columns':list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    finally: save(folder / 'result.json',row)
    return row


def probe(root,directory):
    binding(root,directory); rows = []
    for case in QUERIES:
        require(not (directory / 'probes' / case).exists(),'Case already attempted')
        command = [sys.executable,'-m','scripts.review_strategy_batch70','worker','--directory',str(directory),'--case',case]
        try:
            result = subprocess.run(command,capture_output=True,text=True,timeout=45)
            save(directory / f'worker-{case}.json',{'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode == 0,'Worker failed before archive')
        except subprocess.TimeoutExpired as exc:
            save(directory / f'worker-{case}.json',{'command':command,'status':'timeout','stdout':str(exc.stdout),
                'stderr':str(exc.stderr),'timeout_seconds':45})
            folder = directory / 'probes' / case; folder.mkdir(parents=True,exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json',{'case':case,'function':'futures_zh_daily_sina','parameters':QUERIES[case],
                    'status':'timeout','error':'Worker45s deadline; request completion unproved','attempts':[], 'responses':[], 'files':[],
                    'partial_files':{str(p):file_sha(p) for p in folder.iterdir() if p.is_file()},
                    'published':False,'strict_usable':False,'api_sha256':file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / 'probes' / case / 'result.json'))
    save(directory / 'probe-results.json',{'results':rows,'published':False,'not_a_backtest':True})
    save(directory / 'probe-audit.json',probe_audit(directory))
    return archive(root,'Batch70 bounded 24 continuous/one historical sugar contract probes and reused RB/CU archived')


def replay_frame(row):
    import akshare as ak
    wire = row['responses']; require(len(wire) == 1 and len(row['files']) == 1,'Parsed data lacks wire evidence')
    calls = []
    def replay(session,method,url,**kw):
        require(not calls and method.upper() == 'GET' and url == previous.SINA_URL and
            kw.get('params') == row['attempts'][0]['parameters'],'Unexpected replay request')
        calls.append(url); response = requests.Response(); response.status_code = wire[0]['status_code']
        response._content = Path(wire[0]['file']).read_bytes(); response.encoding = wire[0]['encoding']; response.url = wire[0]['url']; return response
    with patch.object(requests.sessions.Session,'request',replay): return ak.futures_zh_daily_sina(**row['parameters'])


def probe_audit(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'),'API changed'); rows = read(directory / 'probe-results.json')['results']
    require([r['case'] for r in rows] == list(QUERIES),'Probe scope changed'); audit = []
    for row in rows:
        case = row['case']; require(row == read(directory / 'probes' / case / 'result.json') and row['parameters'] == QUERIES[case] and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and len(row['attempts']) <= 1 and
            not row['published'] and not row['strict_usable'],'Probe binding changed')
        for item in row['files']+row['responses']: require(file_sha(item['file']) == item['sha256'],'Probe bytes changed')
        for name,sha in row.get('partial_files',{}).items(): require(file_sha(name) == sha,'Partial timeout evidence changed')
        entry = {'case':case,'requested_symbol':row['parameters']['symbol'],'status':row['status'],'rows':0,
            'responses':len(row['responses']),'original_index_or_main_equivalence':False}
        if row['files']:
            require(row['status'] in ('success','empty') and len(row['files']) == 1,'Parsed status inconsistent')
            frame = pd.read_parquet(row['files'][0]['file']); pd.testing.assert_frame_equal(frame,replay_frame(row),check_exact=True)
            require(len(frame) == row['files'][0]['rows'] and list(frame) == row['files'][0]['columns'],'Parsed profile differs')
            dates = pd.to_datetime(frame['date'],errors='coerce')
            entry.update({'rows':len(frame),'date_min':str(dates.min()),'date_max':str(dates.max()),'invalid_dates':int(dates.isna().sum()),
                'rows_after_cutoff':int((dates>pd.Timestamp('2026-09-29')).sum()),'wire_parser_match':True})
        else: require(row['status'] in ('failed','timeout') and row.get('error'),'Failure not archived'); entry['error'] = row['error']
        audit.append(entry)
    return {'samples':audit,'new_probe_cases':len(QUERIES),'reused_symbols':['RB0','CU0'],
        'not_a_backtest':True,'published':False,'new_trading_backtests':0}


def numeric_windows(frame,case,directory):
    frame = frame.copy(); dates = pd.to_datetime(frame['date'],errors='raise')
    require(not dates.duplicated().any() and dates.is_monotonic_increasing,'Supplier date order/uniqueness invalid')
    frame['date'] = dates; beyond = int((dates>pd.Timestamp('2026-09-29')).sum())
    frame = frame[frame.date.le('2026-09-29')].reset_index(drop=True); data = frame[['open','high','low','close']].to_numpy(dtype=float)
    valid = (np.isfinite(data).all(axis=1)&(data>0).all(axis=1)&
        (data[:,1]>=np.maximum(data[:,0],data[:,3]))&(data[:,2]<=np.minimum(data[:,0],data[:,3])))
    momentum = selected(directory,0,['market_open'],{'g':SimpleNamespace(),'np':np,'talib':talib})
    mg = momentum['g']; mg.instruments = [case]; mg.MappingReal = {case:'REAL'}; mg.MappingIndex = {case:'INDEX'}
    mg.BackWindow = 15
    mg.PriceArray = {}; mg.LastRealPrice = {}; mg.ATR = {}; mg.CurrentPrice = {}; mg.Price_DaysAgo = {}; mg.ClosePrice = {}; mg.Momentum = {}
    momentum['get_CCFX_end_date'] = lambda *a: dt.date(2099,1,1)
    context = SimpleNamespace(current_dt=dt.datetime(2020,1,2)); rows = []; invalid = 0; comparisons = 0
    models = [('cross20',20)] if case != 'SR2001' else []
    if case in ('SR','SR2001'): models.append(('turtle21',21))
    turtle = selected(directory,1,['get_ATR','check_break','get_unit'],{'np':np})
    for model,count in models:
        for end in range(count-1,len(frame)):
            start = end-count+1
            if not valid[start:end+1].all(): invalid += 1; continue
            window = frame.iloc[start:end+1].copy(); window.index = range(-count,0)
            _,high,low,close = data[start:end+1].T
            if model == 'cross20':
                momentum['attribute_history'] = lambda code,n,*a: window.iloc[-n:]
                momentum['market_open'](context); value = float(mg.Momentum[case]); atr = float(mg.ATR['INDEX'])
                expected = float(close[-1]/close[-15]-1)
                require(value == expected and math.isclose(atr,previous.independent_atr(high,low,close,15),rel_tol=1e-12,abs_tol=1e-9),'Independent cross arithmetic differs')
                rows.append({'case':case,'model':model,'date':str(frame.date.iloc[end].date()),'close':float(close[-1]),
                    'ATR':atr,'momentum':value,'channel_upper':math.nan,'channel_lower':math.nan,'break':0,'risk_unit':math.nan})
                comparisons += 2
            else:
                atr = float(turtle['get_ATR'](window,20)); upper = float(max(high[:-2])); lower = float(min(low[:-2]))
                expected = math.fsum(max(high[i],close[i-1])-min(low[i],close[i-1]) for i in range(1,21))/20
                signal = turtle['check_break'](window,float(close[-1]),20)
                require(math.isclose(atr,expected,rel_tol=1e-12,abs_tol=1e-9) and
                    signal == (1 if close[-1]>upper else -1 if close[-1]<lower else 0),'Independent turtle arithmetic differs')
                quantity = float(turtle['get_unit'](1000000.,atr,'SR')) if atr>0 else math.nan
                rows.append({'case':case,'model':model,'date':str(frame.date.iloc[end].date()),'close':float(close[-1]),
                    'ATR':atr,'momentum':math.nan,'channel_upper':upper,'channel_lower':lower,'break':signal,'risk_unit':quantity})
                comparisons += 2
    columns = ['case','model','date','close','ATR','momentum','channel_upper','channel_lower','break','risk_unit']
    return pd.DataFrame(rows,columns=columns), {'case':case,'supplier_rows':len(dates),'retained_rows':len(frame),
        'rows_after_cutoff_excluded':beyond,'invalid_OHLC_rows':int((~valid).sum()),'invalid_windows_blocked':invalid,
        'component_rows':len(rows),'independent_numeric_comparisons':comparisons,
        'date_min':str(frame.date.min()),'date_max':str(frame.date.max())}


def cross_sections(values):
    rows = []; incomplete = 0; frame = values[values.model.eq('cross20')]
    for day,group in frame.groupby('date',sort=True):
        mapping = dict(zip(group['case'],group.momentum,strict=True))
        if set(mapping) != set(POOL): incomplete += 1; continue
        ranked = sorted(POOL,key=lambda ins:mapping[ins],reverse=True)
        for side,chosen in (('buy',ranked[:7]),('sell',ranked[-1:-8:-1])):
            rows.extend({'date':day,'side':side,'rank':i+1,'case':ins,'momentum':float(mapping[ins])} for i,ins in enumerate(chosen))
    return pd.DataFrame(rows,columns=['date','side','rank','case','momentum']), incomplete


def components(directory):
    validate_sources(directory); require(api_evidence() == read(directory / 'existing-apis.json'),'Indicator/API changed')
    samples = {**read(directory / 'input-binding.json')['reused_supplier_samples'],
        **{r['case']:r for r in read(directory / 'probe-results.json')['results']}}
    frames = []; coverage = []
    for case in (*POOL,'SR2001'):
        row = samples[case]
        if row['status'] != 'success': coverage.append({'case':case,'status':row['status'],'component_rows':0,'independent_numeric_comparisons':0}); continue
        item = row['files'][0]; require(file_sha(item['file']) == item['sha256'],'Supplier sample changed')
        values,profile = numeric_windows(pd.read_parquet(item['file']),case,directory); frames.append(values); coverage.append(profile)
    values = pd.concat(frames,ignore_index=True); ranks,incomplete = cross_sections(values)
    return values,ranks,{'coverage':coverage,'component_rows':len(values),'cross_section_rows':len(ranks),
        'complete_26_product_dates':int(ranks.date.nunique()),'incomplete_candidate_dates_skipped':incomplete,
        'independent_numeric_comparisons':sum(r['independent_numeric_comparisons'] for r in coverage),
        'not_a_backtest':True,'original_signal_reproduced':False,'trading_variant_approved':False,
        'limits':['Supplier continuous sensitivity and requested SR2001 sample only, not original 8888 index/main histories',
            'Original numeric functions execute with explicit synthetic mapping/expiry and negative-label scaffolding only',
            'All26 complete-date ranks are daily arithmetic, not original weekly trading/rollover/reentry/positions',
            'No interpolation, costs, fills, NAV or approved trade variant; invalid OHLC windows blocked']}


def prepare(root,directory):
    binding(root,directory); require(not (directory / 'component-values.parquet').exists(),'Component already archived')
    values,ranks,summary = components(directory)
    values.to_parquet(directory / 'component-values.parquet',index=False); ranks.to_parquet(directory / 'cross-section-values.parquet',index=False)
    save(directory / 'component-research.json',summary)
    return archive(root,'Batch70 source numeric functions and complete supplier26-product ranks archived; not original trades')


def offline(root,directory):
    binding(root,directory); before = implementation()
    def denied(*a,**kw): raise ValueError('Batch70 attempted network')
    with patch.object(socket,'create_connection',denied),patch.object(socket.socket,'connect',denied), \
            patch.object(requests.sessions.Session,'request',denied),redirect_stdout(StringIO()):
        diag = diagnostics(directory); cat = offline_catalog(root,directory); dep = inventory(root); audit = probe_audit(directory)
        values,ranks,summary = components(directory)
    pd.testing.assert_frame_equal(values,pd.read_parquet(directory / 'component-values.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(ranks,pd.read_parquet(directory / 'cross-section-values.parquet'),check_exact=True)
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and dep == read(directory / 'dependency-inventory.json') and
        audit == read(directory / 'probe-audit.json') and summary == read(directory / 'component-research.json'),'Offline evidence differs')
    save(directory / 'offline-catalog.json',cat); validate_catalog(root,directory)
    save(directory / 'offline-verification.json',{'result':'match','differences':0,'socket_network_disabled':True,
        'implementation_sha256':before,'evidence_sha256':{name:file_sha(directory/name) for name in
            ('diagnostics.json','probe-audit.json','component-values.parquet','cross-section-values.parquet','component-research.json')},'not_a_backtest':True})
    return archive(root,'Batch70 forbidden-network source arithmetic/ranks/diagnostics/parsers and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests','scripts/review_strategy_batch70.py'],
        [sys.executable,'-m','scripts.verify_offline_tests','-q','tests/unit/test_batch70_research.py','tests/unit/test_batch69_research.py',
            'tests/unit/test_catalog_futures.py','tests/unit/test_catalog_dependencies.py','tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py','tests/integration/test_strategy_merge.py'],['git','diff','--check']]


def checks(root,directory):
    binding(root,directory); validate_sources(directory); validate_catalog(root,directory); before = implementation(); rows = []
    for k,command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as output: result = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        rows.append({'command':command,'returncode':result.returncode,'log':str(path),'sha256':file_sha(path)})
        require(result.returncode == 0,f'Check failed: {path}')
    require(before == implementation(),'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True,exist_ok=True)
        require(not copied.exists(),'Checked copy exists'); shutil.copyfile(name,copied)
    save(directory / 'checked-state.json',{'status':'passed','commands':rows,'implementation_sha256':before,
        'evidence_sha256':{p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status':'passed'}


def finish(root,directory):
    binding(root,directory); validate_sources(directory); validate_catalog(root,directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']),'Checks changed')
    for name,sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha,'Evidence changed')
    for name,sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha,'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'],'Offline proof changed')
    with redirect_stdout(StringIO()): diag = diagnostics(directory); audit = probe_audit(directory)
    require(diag == read(directory / 'diagnostics.json') and audit == read(directory / 'probe-audit.json') and
        inventory(root) == read(directory / 'dependency-inventory.json'),'Recomputed evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'],'Publication changed')
    protection = protect(root,directory); progress = archive(root,'Batch70 cross-momentum/sugar-turtle source research accepted; original futures trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch70-verification.json')
    save(receipt,{'status':'ok','snapshot':SNAPSHOT,'reviews':read(directory / 'source-reviews/review.json'),'checks':checked,'offline':off,
        'diagnostics':diag,'probe_audit':audit,'component':read(directory / 'component-research.json'),'protection':protection,
        'progress':progress,'newly_reviewed':2,'not_a_backtest':True,'original_strategy_complete':False,'formal_strategy_acceptance':False})
    save(directory / 'final-binding-verification.json',{'status':'ok','receipt_sha256':file_sha(receipt),
        'checked_state_sha256':file_sha(directory / 'checked-state.json'),'final_script_sha256':file_sha(__file__),
        'catalog_sha256':file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status':'ok','checkpoint':progress['checkpoint'],'manually_reviewed':progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','start','study','worker','probe','prepare','offline','checks','finish'])
    parser.add_argument('--root',default='data'); parser.add_argument('--directory',type=Path,default=DIRECTORY)
    parser.add_argument('--case',choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory,args.case) if args.action == 'worker' else globals()[args.action](args.root,args.directory)
    print(json.dumps(result,ensure_ascii=False,default=str))
