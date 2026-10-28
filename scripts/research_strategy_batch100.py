"""Frozen original callbacks and 400-row feature prefixes, never trading results."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import datetime as dt
import importlib.util
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import talib

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch100/20261011-original-feature-stage')
PARENT = Path('docs/handoff/2026-10-11-batch99-final-review.json')
PARENT_SHA = '628c7b67ce4194eb577de00f4dd466b12725b674abcce2dbf4018834b9ba2215'
SOURCE_PATHS = ('2022年度精选策略/32.行业指数轮动.txt',
                '2022年度精选策略/26.Xgboost 50ETF 2.0 分钟级回测.txt')
SOURCE_SHA = ('eee496f1bc636b0372ba32153596d3bb72c9d4f73e2c3520a16c35da532f0e76',
              '4947832e5d01a5a15b4408de799c0c4e908d5e91eda996fbc33bb0f3c22c3a8a')
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
REGISTRY = Path('data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/results.json')
REGISTRY_SHA = 'd258e2fb847f2441c1f646e15aba2fa98360911e1e63dcefec25729b67b8ac68'
CANDIDATE_SHA = '17897aa46f5d66ba2878ad90f537ddd6c2d3d2d1baaf2b321cdc333d6b4237f2'
MODELS = ('col_50etf_20191109.dat', 'model_50etf_20191109.dat')
SYMBOLS = ('000932.SH', '000913.SH', '399967.SZ', '000993.SH', '000819.SH',
           '399975.SZ', '399971.SZ', '511880.SH', '510050.SH', '000300.SH')
REFERENCES = ('repo/qlib/qlib/contrib/model/xgboost.py',
    'repo/akshare/akshare/index/index_zh_em.py',
    'repo/rqalpha/rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py')
FILES = ('scripts/research_strategy_batch100.py', 'tests/unit/test_batch100_research.py',
    'scripts/review_strategy_batch73.py', 'scripts/recover_strategy_batch94_reference.py',
    'scripts/review_strategy_batch18.py', 'scripts/run_strategy_batch4.py',
    'scripts/archive_strategy_progress.py', 'scripts/verify_strategy_batch3.py',
    'src/observe/runs.py', 'src/observe/data/store.py', 'src/observe/data/locks.py',
    'src/observe/strategy_catalog.py', 'pyproject.toml', 'uv.lock')
OUTPUTS = ('diagnostics.json', 'feature-prefix.parquet', 'summary.json')


def literal(path, number, namespace=None):
    require(number in (0, 1) and file_sha(path) == SOURCE_SHA[number], 'Source SHA differs')
    tree = ast.parse(read_source(Path(path))[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (14 if number == 0 else 12) and all(safe_function(n) for n in nodes),
            'Original definition contract differs')
    ns = {'pd': pd, 'np': np, 'talib': talib, 'datetime': dt, 'BytesIO': BytesIO,
          'g': SimpleNamespace(), 'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def original(number, **kwargs):
    return literal(DIRECTORY/'sources'/f'{number}.source', number, kwargs)


def canonical(value):
    if isinstance(value, (np.ndarray, pd.Series)): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else 'inf' if value > 0 else '-inf'}
    if value is None or isinstance(value, (str, bool, int, float)): return value
    raise TypeError('Unsupported diagnostic archive type: '+type(value).__name__)


def parent_check():
    require(file_sha(PARENT) == PARENT_SHA, 'Parent receipt differs')
    parent = read(PARENT)
    require(parent['status'] == 'ok' and parent['progress']['manually_reviewed'] == 284,
            'Parent acceptance differs')
    # Runtime catalog may advance only after this stage; its old frozen bytes remain required.
    for name, digest in parent['evidence_sha256'].items():
        if name == 'src/observe/strategy_catalog.py':
            name = 'data/staging/strategies-batch99/20261011-formal-source-acceptance/strategy_catalog.proposed.py'
        require(file_sha(name) == digest, 'Parent evidence differs: '+name)
    return {'file': str(PARENT), 'sha256': PARENT_SHA, 'bound_files': len(parent['evidence_sha256'])}


def candidate():
    require(file_sha(REGISTRY) == REGISTRY_SHA, 'Candidate registry differs')
    row = next(r for r in read(REGISTRY)['results'] if r['case'] == 'fund-510050')
    require(row['sha256'] == file_sha(row['file']) == CANDIDATE_SHA, 'Candidate bytes differ')
    for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Wire bytes differ')
    raw = pd.read_parquet(row['file'])
    require(len(raw) == 5254, 'Candidate rows differ')
    frame = raw[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    frame = frame.loc[frame.date.le('2026-09-29')].reset_index(drop=True)
    require(len(frame) == 5252 and not frame.date.duplicated().any() and frame.date.is_monotonic_increasing,
            'Candidate calendar/shape differs')
    for col in ('open', 'high', 'low', 'close', 'volume'):
        frame[col] = pd.to_numeric(frame[col], errors='raise').astype(float)
        require(np.isfinite(frame[col]).all() and (frame[col] > 0).all(), 'Candidate field invalid: '+col)
    require((frame.high >= frame[['open', 'close', 'low']].max(axis=1)).all() and
            (frame.low <= frame[['open', 'close', 'high']].min(axis=1)).all(), 'OHLC ordering differs')
    return frame, row


def inventory():
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA and
            state['tables'] == store.published()['tables'], 'Snapshot/publication differs')
    tables = {}
    with duckdb.connect() as con:
        for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
            files = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
            column = 'index' if table == 'index_1d' else 'instrument'
            counts = dict.fromkeys(SYMBOLS, 0)
            if files:
                marks = ','.join('?' for _ in SYMBOLS)
                rows = con.execute(f'SELECT "{column}", count(*) FROM read_parquet(?, union_by_name=true) '
                    f'WHERE "{column}" IN ({marks}) GROUP BY "{column}"', [files, *SYMBOLS]).fetchall()
                counts.update({s: int(n) for s, n in rows})
            tables[table] = counts
    frame, row = candidate()
    calendar = store.load_state(state, 'calendar')
    days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d')
    days = set(days.loc[days.between(frame.date.iloc[0], frame.date.iloc[-1])])
    require(set(frame.date) == days, 'Candidate/open-calendar difference')
    hits = {name: [] for name in MODELS}
    # One filename scan; prune metadata instead of traversing repeated exported objects.
    command = ['rg', '--files', '--hidden', '--no-ignore', '-g', MODELS[0], '-g', MODELS[1],
               '-g', '!data/metadata/**', 'repo', 'data']
    found = subprocess.run(command, capture_output=True, text=True, timeout=120)
    require(found.returncode in (0, 1), 'Model inventory search failed: '+found.stderr)
    for name in found.stdout.splitlines():
        path = Path(name); require(path.is_file() and path.name in hits, 'Unexpected model search path')
        hits[path.name].append(name)
    return {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
        'published_batch': store.published()['batch_id'], 'tables': tables,
        'bars_1m_registered': 'bars_1m' in state['tables'], 'original_models_found': hits,
        'model_search': {'command': command, 'returncode': found.returncode, 'stderr': found.stderr},
        'xgboost_installed': importlib.util.find_spec('xgboost') is not None,
        'candidate': row, 'candidate_retained': len(frame), 'future_excluded': 2,
        'candidate_calendar_missing_extra': [0, 0], 'strict_usable': False,
        'volume_unit_and_platform_adjustment_equivalence': 'unproven; raw vendor volume, no rescaling',
        'reference_sha256': {n: file_sha(n) for n in REFERENCES},
        'minute_api_limit': 'AKShare index period1 trends2 asks ndays=5; slicing start/end cannot recover 4800 bars or old history',
        'carrier_limit': 'Original ETF dictionary maps every sector to the same index code; RQAlpha explicitly supports synthetic index orders, which does not prove JoinQuant execution or an exchange-traded ETF mapping',
        'new_trading_backtests': 0, 'new_requests': 0, 'published_rows': 0}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable stage directory exists')
    parent = parent_check(); inputs = inventory(); frame, _ = candidate()
    for n, name in enumerate(SOURCE_PATHS): literal(Path('repo/量化策略源代码')/name, n)
    hashes = {name: file_sha(name) for name in FILES}
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch100 full source/feature study started; formal284/trade12/parents87 unchanged')
        (DIRECTORY/'sources').mkdir()
        for n, name in enumerate(SOURCE_PATHS): shutil.copyfile(Path('repo/量化策略源代码')/name, DIRECTORY/'sources'/f'{n}.source')
        for name in (*FILES, *REFERENCES):
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        frame.to_parquet(DIRECTORY/'ETF-input.parquet', index=False)
        save(DIRECTORY/'dependencies.json', inputs)
        save(DIRECTORY/'prepared.json', {'parent': parent, 'implementation_sha256': hashes,
            'reference_sha256': inputs['reference_sha256'], 'operand_sha256': file_sha(DIRECTORY/'ETF-input.parquet'),
            'dependencies_sha256': file_sha(DIRECTORY/'dependencies.json'),
            'source_sha256': list(SOURCE_SHA), 'features_scope': 'Original create_fea prefix strictly before alpha005; full original failure preserved',
            'not_a_backtest': True})
    return {'status': 'prepared', 'rows': len(frame)}


def validate():
    doc = read(DIRECTORY/'prepared.json'); require(doc['parent'] == parent_check(), 'Parent binding differs')
    for name, digest in doc['implementation_sha256'].items():
        require(file_sha(DIRECTORY/'implementation'/name) == digest, 'Frozen implementation differs: '+name)
        if name != 'src/observe/strategy_catalog.py': require(file_sha(name) == digest, 'Runtime implementation differs: '+name)
    for name, digest in doc['reference_sha256'].items():
        require(file_sha(DIRECTORY/'implementation'/name) == digest, 'Frozen reference differs')
    require(file_sha(DIRECTORY/'ETF-input.parquet') == doc['operand_sha256'] and
            file_sha(DIRECTORY/'dependencies.json') == doc['dependencies_sha256'], 'Frozen input differs')
    for n in (0, 1): original(n)
    return doc


def diagnostics():
    rows = []
    def case(name, function):
        try:
            with redirect_stdout(StringIO()): value = function()
            rows.append({'case': name, 'value': canonical(value), 'error': None})
        except Exception as exc:
            rows.append({'case': name, 'value': None, 'error': type(exc).__name__, 'message': str(exc)})

    def rotation(signal='sell_the_stocks', holdings=(), time=(14, 15), immediate=False):
        calls = []; positions = {s: SimpleNamespace(total_amount=q) for s, q in holdings}
        class Positions(dict):
            def __missing__(self, key): return SimpleNamespace(total_amount=0)
        portfolio = SimpleNamespace(cash=1234., positions=Positions(positions))
        def target(s, value):
            calls.append(['target', s, value])
            if immediate: portfolio.cash += 100.
        ns = original(0, set_benchmark=lambda x: calls.append(['benchmark', x]),
            order_target_value=target, order_value=lambda s, value: calls.append(['buy', s, value]))
        ns['set_params'](); calls.clear()
        ns['get_signal'] = lambda c: signal
        ns['getCurrentPrice'] = lambda s: calls.append(['unused_price', s]) or 100.
        context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 2, *time), portfolio=portfolio)
        ns['handle_data'](context, None)
        return calls
    case('sector_identity_mapping', lambda: sector_mapping())
    for name, signal, holdings, time, immediate in (
        ('risk_off_empty', 'sell_the_stocks', (), (14, 15), False),
        ('risk_off_money_held', 'sell_the_stocks', (('511880.XSHG', 100), ('old', 100)), (14, 15), False),
        ('risk_off_two_positions', 'sell_the_stocks', (('old1', 100), ('old2', 100)), (14, 15), False),
        ('risk_on_same_held', '000932.XSHG', (('000932.XSHG', 100), ('old', 100)), (14, 15), False),
        ('risk_on_new_deferred_cash', '000932.XSHG', (('old1', 100), ('old2', 100)), (14, 15), False),
        ('risk_off_immediate_double_sell', 'sell_the_stocks', (('old1', 100), ('old2', 100)), (14, 15), True),
        ('wrong_minute_no_action', '000932.XSHG', (), (14, 14), False)):
        case(name, lambda s=signal, h=holdings, t=time, i=immediate: rotation(s, h, t, i))

    def sector_signal(equal=False, nan=False):
        calls = []; ns = original(0, set_benchmark=lambda *a: None)
        ns['set_params']()
        def history(symbol, count, **kw):
            calls.append([symbol, count, kw])
            if nan: values = np.full(count, np.nan)
            elif count == 2400: values = np.linspace(100., 110., count)
            else: values = np.full(count, 110. if equal else 100.)
            return pd.DataFrame({'close': values})
        ns['attribute_history'] = history
        return {'signal': ns['get_signal'](None), 'calls': calls}
    case('stable_tie_first_sector_and_minute_counts', sector_signal)
    case('strict_mean_equality_risk_off', lambda: sector_signal(equal=True))
    case('all_nan_risk_off', lambda: sector_signal(nan=True))
    case('empty_minute_price', lambda: original(0, attribute_history=lambda *a, **k: pd.DataFrame({'close': []}))['getStockPrice']('x', 10))
    for date in ('2009-01-01', '2009-01-02', '2011-01-01', '2011-01-02', '2013-01-01', '2013-01-02'):
        def fees(date=date):
            calls = []; ns = original(0, FixedSlippage=lambda v: v, PerTrade=lambda **k: k,
                set_slippage=lambda v: calls.append(['slippage', v]), set_commission=lambda v: calls.append(['commission', v]))
            ns['set_slip_fee'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(date)))
            return calls
        case('fees_'+date, fees)

    ns = original(1)
    case('rank_returns_series', lambda: ns['ts_rank'](np.array([1., 2., 2.])))
    case('rolling_rank_current_pandas_fails', lambda: pd.Series(np.arange(20.)).rolling(10).apply(ns['ts_rank']))
    case('rankeq10_counts_exact_tens', lambda: ns['ts_rankeq10'](np.array([10., 9.5, 10., np.nan])))
    def xgb_trade(predictions, missing=None):
        calls = []; frame = pd.DataFrame({'a': [0., 1.], 'b': [2., 3.]})
        def load(buf):
            name = buf.getvalue().decode()
            if missing == name: raise FileNotFoundError(name)
            if name == MODELS[0]: return ['b', 'a']
            return [SimpleNamespace(predict=lambda d, v=v: np.array([v])) for v in predictions]
        def matrix(f):
            calls.append(['matrix', list(f.columns), f.values.tolist()]); return f
        ns = original(1, pickle=SimpleNamespace(load=load), read_file=lambda name: name.encode(),
            xgb=SimpleNamespace(DMatrix=matrix), order_value=lambda s, v: calls.append(['buy', s, v]),
            order_target_value=lambda s, v: calls.append(['target', s, v]))
        ns['create_fea'] = lambda c: frame
        context = SimpleNamespace(etf='510050.XSHG', portfolio=SimpleNamespace(cash=1234.))
        ns['trade'](context); return calls
    for name, predictions in (('xgb_buy', [.003]), ('xgb_sell', [-.003]), ('xgb_upper_equal', [.002]),
            ('xgb_lower_equal', [-.002]), ('xgb_ensemble_mean_hold', [.004, -.002]),
            ('xgb_nan_hold', [float('nan')]), ('xgb_empty_ensemble_error', [])):
        case(name, lambda p=predictions: xgb_trade(p))
    for name in MODELS: case('xgb_missing_'+name, lambda n=name: xgb_trade([.003], n))
    def daily_calls():
        calls = []; ns = original(1, attribute_history=lambda *a, **k: calls.append(list(a)) or pd.DataFrame({a[3]: np.arange(400.)}))
        output = ns['get_price']('510050.XSHG')
        return {'calls': calls, 'return_order_lengths': [len(x) for x in output]}
    case('five_separate_daily_history_calls', daily_calls)
    def init():
        calls = []; context = SimpleNamespace()
        ns = original(1, run_daily=lambda f, *a, **kw: calls.append(['schedule', f.__name__, list(a), kw]),
            OrderCost=lambda **kw: kw, set_order_cost=lambda *a, **kw: calls.append(['cost', list(a), kw]))
        ns['initialize'](context); return {'etf': context.etf, 'calls': calls}
    case('xgb_daily_0950_and_stock_cost', init)
    return {'case_count': len(rows), 'cases': rows, 'synthetic_only': True,
            'model_predictions_are_stubs': True, 'real_pickle_executed': False,
            'new_trading_backtests': 0, 'actual_json_roundtrip': True}


def sector_mapping():
    ns = original(0, set_benchmark=lambda *a: None); ns['set_params']()
    g = ns['g']
    return {'stocks': g.stocks, 'ETFs': g.ETFs, 'money': g.ETFrili,
            'lag': g.lag, 'bars': g.bars, 'time': [g.hour, g.minute]}


def feature_prefix(ns):
    """Extract only reviewed statements before original alpha005; never fix ts_rank."""
    path = DIRECTORY/'sources/1.source'
    tree = ast.parse(read_source(path)[0])
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_fea'))
    def is_rank_assignment(n):
        return isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript) and \
            isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == 'result' and \
            isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value == 'alpha005'
    positions = [i for i, n in enumerate(node.body) if is_rank_assignment(n)]
    require(len(positions) == 1 and positions[0] == 13, 'Unreviewed feature prefix boundary')
    node.name = 'create_fea_prefix'
    node.body = node.body[:positions[0]]+[ast.Return(value=ast.Name(id='result', ctx=ast.Load()))]
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path)+':prefix_only', 'exec'), ns)
    return ns['create_fea_prefix']


def set_window(ns, window):
    # The original get_price definition still runs and makes all five separate 400-row calls.
    def history(stock, count, unit, field):
        require(stock == '510050.XSHG' and count == 400 and unit == '1d', 'Feature input request differs')
        return window[[field]].copy()
    ns['attribute_history'] = history


def scalar_checks(ns, window, tail):
    """Independent scalar tails for every rolling/diff/lag column, flags and raw arithmetic."""
    price = window[['close', 'open', 'high', 'low', 'volume']].copy()
    base = ns['devfea'](price, price.high, price.low, price.close, price.open, price.volume)
    expected = {}; max_relative = 0.; cells = 0
    for col in base:
        values = base[col].tolist(); x = values[-1]; expected[col] = x
        for n in (5, 10, 20, 30, 60):
            a = values[-n:]; mean = math.fsum(a)/n
            var = math.fsum((v-mean)**2 for v in a)/(n-1)
            order = sorted(a); mid = n//2
            median = order[mid] if n % 2 else (order[mid-1]+order[mid])/2
            for stat, value in zip(('mean', 'max', 'min', 'std', 'var', 'median'),
                    (mean, max(a), min(a), math.sqrt(var), var, median), strict=True):
                expected[f'{col}_{stat}_rl_{n}D'] = value
        for n in (1, 5, 10, 20, 30, 60): expected[f'{col}_diff_{n}D'] = x-values[-n-1]
        for n in (1, 2, 3, 5): expected[f'{col}_lag_{n}D'] = values[-n-1]
        expected[col+'_diff2'] = (x-values[-2])-(values[-2]-values[-3])
    for n in (5, 10, 20): expected['long_MA'+str(n)+'_flag'] = int(base.close.iloc[-1] > base['MA'+str(n)].iloc[-1])
    expected['long_MA5_MA10'] = int(base.MA5.iloc[-1] > base.MA10.iloc[-1])
    expected['long_MA5_MA20'] = int(base.MA5.iloc[-1] > base.MA20.iloc[-1])
    expected['alpha002'] = base.open.iloc[-1]/base.close.iloc[-1]
    expected['alpha003'] = -base.volume.iloc[-1]/(math.fsum(base.volume.iloc[-20:])/20)
    def corr(a, b):
        am, bm = math.fsum(a)/len(a), math.fsum(b)/len(b)
        cross = math.fsum((x-am)*(y-bm) for x, y in zip(a, b, strict=True))
        den = math.sqrt(math.fsum((x-am)**2 for x in a)*math.fsum((y-bm)**2 for y in b))
        return cross/den if den else float('nan')
    expected['aphla001'] = corr(base.close.iloc[-10:].tolist(), base.volume.iloc[-10:].tolist())
    expected['alpha004'] = corr((base.high.iloc[-10:]/base.close.iloc[-10:]).tolist(), base.volume.iloc[-10:].tolist())
    require(set(expected) == set(tail.index), 'Independent feature scope differs')
    for col, x in expected.items():
        y = float(tail[col]); cells += 1
        if not math.isfinite(x) or not math.isfinite(y):
            require(math.isnan(x) and math.isnan(y), 'Nonfinite feature differs: '+col)
        else:
            relative = abs(x-y)/max(1., abs(x), abs(y)); max_relative = max(max_relative, relative)
            require(math.isclose(x, y, rel_tol=2e-8, abs_tol=2e-8), 'Independent feature differs: '+col)
    return cells, max_relative


def components(frame):
    ns = original(1); prefix = feature_prefix(ns); context = SimpleNamespace(etf='510050.XSHG')
    rows = []; cells = 0; max_relative = 0.; original_errors = []
    for i in range(400, len(frame)):
        window = frame.iloc[i-400:i].set_index('date')
        set_window(ns, window); output = prefix(context); tail = output.iloc[-1]
        checked, error = scalar_checks(ns, window, tail); cells += checked; max_relative = max(max_relative, error)
        rows.append({'decision_date': frame.date.iloc[i], 'first_input_date': frame.date.iloc[i-400],
            'last_input_date': frame.date.iloc[i-1], **{n: float(v) for n, v in tail.items()}})
        if i == 400 or i == len(frame)-1:
            try: ns['create_fea'](context)
            except Exception as exc: original_errors.append({'decision_date': frame.date.iloc[i], 'error': type(exc).__name__, 'message': str(exc)})
            else: raise AssertionError('Original rolling rank unexpectedly succeeded')
        if (i-399) % 500 == 0: print(json.dumps({'feature_windows_completed': i-399}), flush=True)
    output = pd.DataFrame(rows)
    require(len(output) == len(frame)-400, 'Feature coverage differs')
    return output, {'candidate_rows': len(frame), 'windows': len(output),
        'first_decision': output.decision_date.iloc[0], 'last_decision': output.decision_date.iloc[-1],
        'window_rows': 400, 'feature_columns': len(output.columns)-3, 'independent_scalar_cells': cells,
        'max_scalar_relative_error': max_relative, 'original_full_pipeline_failures': original_errors,
        'year_counts': output.groupby(output.decision_date.str[:4]).size().to_dict(),
        'seed_policy': 'Original indicators recomputed inside each 400-row window; not continuous full-history seeds',
        'scope': 'Candidate-price feature prefix before alpha005; not original selected model inputs or predictions',
        'strict_usable': False, 'cost_scenarios': 'blocked; no trading input/model/execution admission',
        'new_trading_backtests': 0, 'new_strategy_parent_runs': 0}


def worker(mode):
    require(mode in ('study', 'replay'), 'Unknown worker mode'); validate()
    target = DIRECTORY/mode; require(not target.exists(), 'Immutable worker target exists')
    with network_blocked():
        diag = diagnostics()
        # Historical calculation creates a fresh namespace after all callback diagnostics.
        output, summary = components(pd.read_parquet(DIRECTORY/'ETF-input.parquet'))
        target.mkdir(); save(target/'diagnostics.json', diag)
        require(read(target/'diagnostics.json') == diag, 'Actual JSON shape differs')
        output.to_parquet(target/'feature-prefix.parquet', index=False); save(target/'summary.json', summary)
    return {'status': 'ok', 'mode': mode, 'output_sha256': {n: file_sha(target/n) for n in OUTPUTS}}


def run(mode):
    folder = DIRECTORY/(mode+'-command'); folder.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.research_strategy_batch100', 'worker', '--mode', mode]
    process = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    for name, content in (('stdout.log', process.stdout), ('stderr.log', process.stderr)):
        (folder/name).write_text(content, encoding='utf-8')
    save(folder/'command.json', {'command': command, 'returncode': process.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Worker failed; logs retained')
    return json.loads(process.stdout.splitlines()[-1])


def offline():
    validate()
    require(all(read(DIRECTORY/(m+'-command')/'command.json')['returncode'] == 0 for m in ('study', 'replay')),
            'Fresh worker failed')
    hashes = {n: file_sha(DIRECTORY/'study'/n) for n in OUTPUTS}
    require(all(file_sha(DIRECTORY/'replay'/n) == h for n, h in hashes.items()), 'Replay bytes differ')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study/feature-prefix.parquet'),
        pd.read_parquet(DIRECTORY/'replay/feature-prefix.parquet'), check_exact=True)
    proof = {'status': 'match', 'differences': 0, 'fresh_process': True, 'network_blocked': True,
        'output_sha256': hashes, 'prepared_sha256': file_sha(DIRECTORY/'prepared.json'), 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', proof); return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'replay', 'worker', 'offline', 'protect'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    result = worker(args.mode) if args.action == 'worker' else protect('data', DIRECTORY) if args.action == 'protect' \
        else run(args.action) if args.action in ('study', 'replay') else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
