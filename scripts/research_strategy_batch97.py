"""Frozen seven-day trend/crowd research. No portfolio, orders, costs, or NAV."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
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
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

PREREAD = Path('data/staging/strategies-batch97/20261011-source-preread')
DIRECTORY = Path('data/staging/strategies-batch97/20261011-components-v1')
SNAPSHOT = '20261006-145049-e06c'
PARENT = Path('docs/handoff/2026-10-11-pair66-short-windows-final-review.json')
PARENT_SHA = 'b283df3c47f6bbb3de408effd993504c8b3da23451f42ac0f692f91caf8dcdf9'
SOURCE_SHA = ('4a9e21d05fb014d129ea24928cec287311511cfadc6c5ff8c308e26b52277989',
              'd0350173eed29f6464d5a350efea957e876d718479b4ca1cd62dbd4694ef3e86')
# Metadata provides byte-checksum evidence only; Store inputs always come from the snapshot.
ASSETS = Path('data/metadata/states/11e52610c1c5fb1cf3dbae62/assets/index.json')
ASSETS_SHA = '9814f72be87a1613ebed657770c22b0148c7c6aefbec95c4d138cfaf48f456cf'
SYMBOLS = ('000333.SZ', '000651.SZ', '002415.SZ', '600036.SH', '600085.SH',
           '600196.SH', '600519.SH', '600887.SH', '601111.SH', '603288.SH')
FILES = ('scripts/research_strategy_batch97.py', 'tests/unit/test_batch97_research.py',
         'scripts/recover_strategy_batch94_reference.py', 'scripts/review_strategy_batch18.py',
         'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py',
         'src/observe/data/store.py', 'src/observe/data/locks.py', 'src/observe/runs.py',
         'src/observe/strategy_catalog.py', 'pyproject.toml', 'uv.lock')
OPERANDS = ('prices.parquet', 'calendar.parquet', 'amounts.parquet')
OUTPUTS = ('diagnostics.json', 'trend-components.parquet', 'crowd-components.parquet', 'summary.json')


def definitions(source, number, namespace):
    """Execute only reviewed functions; defaults/annotations cannot run arbitrary code."""
    require(number in (0, 1) and file_sha(source) == SOURCE_SHA[number], 'Source identity differs')
    tree = ast.parse(read_source(source)[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (19 if number == 0 else 5), 'Source definition inventory differs')
    # crowd_line has a clock-valued default: archive its AST, never evaluate that definition.
    nodes = [n for n in nodes if n.name != 'crowd_line']
    for node in nodes:
        require(not node.decorator_list and not node.returns and not node.type_params,
                'Definition has executable decorators or return annotation')
        annotations = [a.annotation for a in (*node.args.posonlyargs, *node.args.args,
                                               *node.args.kwonlyargs)]
        require(all(a is None or (isinstance(a, ast.Name) and a.id == 'int') for a in annotations),
                'Unreviewed annotation')
        require(all(isinstance(d, ast.Constant) for d in node.args.defaults) and
                all(d is None or isinstance(d, ast.Constant) for d in node.args.kw_defaults),
                'Executable default is forbidden')
        require(all(a is None or a.annotation is None for a in (node.args.vararg, node.args.kwarg)),
                'Executable variable annotation')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def namespace(number):
    return definitions(DIRECTORY/'sources'/f'{number}.source', number,
        {'np': np, 'pd': pd, 'talib': talib, 'g': SimpleNamespace(),
         'log': SimpleNamespace(info=lambda *args: None),
         'get_security_info': lambda code: SimpleNamespace(display_name=code)})


def verify_entries(root, tables):
    require(set(tables) == {'bars_1d', 'calendar', 'instruments'}, 'Input table scope differs')
    require(all(tables.values()), 'Empty table inventory')
    require(file_sha(ASSETS) == ASSETS_SHA, 'Archived byte-checksum inventory changed')
    checksums = {r['path']:r['file_sha256'] for r in read(ASSETS)['assets']}
    result = {}
    for table, entries in tables.items():
        require(bool(entries), 'Empty table inventory')
        for key, row in entries.items():
            path = Path(root)/row['file']
            expected = checksums.get(str(path))
            require(expected is not None and path.is_file() and not path.is_symlink() and
                    file_sha(path) == expected, 'Partition byte SHA differs: '+str(path))
            result[table+'/'+key] = {'file': str(path), 'file_sha256': expected,
                'logical_fingerprint': row['sha'], 'rows': row['rows']}
    return result


def prepare():
    require(not DIRECTORY.exists(), 'Immutable component directory exists')
    require(file_sha(PARENT) == PARENT_SHA and read(PARENT)['status'] == 'ok', 'Parent differs')
    preread = read(PREREAD/'source-preread.json')
    require(preread['parent_sha256'] == PARENT_SHA and
            len(preread['sources']) == 2, 'Preread parent or sources differ')
    store = Store('data'); state = store.state(SNAPSHOT)
    require(state['tables'] == store.published()['tables'], 'Snapshot and publication differ')
    tables = {name: state['tables'][name] for name in ('bars_1d', 'calendar', 'instruments')}
    print('Checking frozen partition bytes', flush=True)
    partitions = verify_entries('data', tables)
    price_files = [str(Path('data')/row['file']) for row in tables['bars_1d'].values()]
    with duckdb.connect() as con:
        prices = con.execute('SELECT date, instrument, open, high, low, close, volume, is_trading '
            'FROM read_parquet(?) WHERE instrument IN (SELECT unnest(?)) ORDER BY instrument,date',
            [price_files, list(SYMBOLS)]).fetchdf()
        amounts = con.execute("SELECT date, instrument, amount FROM read_parquet(?) "
            "WHERE date BETWEEN DATE '2021-01-04' AND DATE '2026-09-29' ORDER BY date,instrument",
            [price_files]).fetchdf()
    require(not prices.duplicated(['instrument', 'date']).any() and
            not amounts.duplicated(['instrument', 'date']).any(), 'Duplicate daily observations')
    require(set(prices.instrument) == set(SYMBOLS), 'Reference symbols differ')
    require(np.isfinite(amounts.amount).all() and amounts.amount.ge(0).all(), 'Invalid money observations')
    calendar = store.load_state(state, 'calendar')
    with operation_lock('data', DATA_WRITER):
        DIRECTORY.mkdir(parents=True, exist_ok=False)
        (DIRECTORY/'sources').mkdir()
        for number in range(2):
            source = PREREAD/'sources'/f'{number}.source'
            require(file_sha(source) == SOURCE_SHA[number], 'Preread source differs')
            shutil.copyfile(source, DIRECTORY/'sources'/source.name)
        for name, frame in zip(OPERANDS, (prices, calendar, amounts), strict=True):
            with (DIRECTORY/name).open('xb') as out: frame.to_parquet(out, index=False)
        # Freeze the complete inventory, including inherited baseline and source AST evidence.
        controls = {str(PREREAD/name): file_sha(PREREAD/name) for name in
            ('baseline.json', 'source-preread.json', 'catalog-before.json', 'strategy_catalog.before.py')}
        implementation = {name: file_sha(name) for name in FILES}
        for name in FILES:
            copied = DIRECTORY/'implementation'/name
            copied.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, copied)
        save(DIRECTORY/'prepared.json', {'schema': 1, 'snapshot': SNAPSHOT,
            'snapshot_sha256': file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')),
            'tables': tables, 'partitions': partitions, 'parent_sha256': PARENT_SHA,
            'controls_sha256': controls, 'implementation_sha256': implementation,
            'sources_sha256': list(SOURCE_SHA), 'operands_sha256': {n: file_sha(DIRECTORY/n) for n in OPERANDS},
            'symbols': list(SYMBOLS), 'limits': ['Raw prices and volume, not original dynamic pre-adjusted values',
                'Observed daily money population, not proven original end-date securities list',
                'No historical total market cap, historical display names, or original one-minute execution inputs',
                'No stock selection, live targets, Book, trades, costs, or NAV'], 'new_trading_backtests': 0})
    return {'status': 'prepared', 'partitions': len(partitions), 'price_rows': len(prices), 'amount_rows': len(amounts)}


def binding():
    doc = read(DIRECTORY/'prepared.json')
    require(doc['snapshot'] == SNAPSHOT and doc['parent_sha256'] == PARENT_SHA and
            doc['symbols'] == list(SYMBOLS) and doc['sources_sha256'] == list(SOURCE_SHA), 'Preparation scope differs')
    require(set(doc['implementation_sha256']) == set(FILES) and
            set(doc['operands_sha256']) == set(OPERANDS), 'Binding inventory differs')
    require(file_sha(PARENT) == PARENT_SHA and
            file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == doc['snapshot_sha256'], 'Parent/snapshot changed')
    require(set(doc['controls_sha256']) == {str(PREREAD/n) for n in
            ('baseline.json', 'source-preread.json', 'catalog-before.json', 'strategy_catalog.before.py')}, 'Control scope differs')
    for name, digest in doc['controls_sha256'].items(): require(file_sha(name) == digest, 'Historical control changed')
    for name, digest in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == digest, 'Implementation changed: '+name)
    for name, digest in doc['operands_sha256'].items(): require(file_sha(DIRECTORY/name) == digest, 'Operand changed')
    for number in range(2): namespace(number)
    state = Store('data').state(SNAPSHOT)
    require(doc['tables'] == {t: state['tables'][t] for t in ('bars_1d', 'calendar', 'instruments')}, 'Input identities changed')
    require(doc['partitions'] == verify_entries('data', doc['tables']), 'Partition scope differs')
    return doc


def independent_crowd(values):
    values = sorted(map(float, values), reverse=True)
    if not values: raise ValueError('Empty money population')
    threshold = values[int(len(values)*.05)]
    selected = [v for v in values if v >= threshold]
    total = math.fsum(values)
    return {'threshold': threshold, 'selected': len(selected), 'rows': len(values),
            'total': total, 'top_money': math.fsum(selected),
            'rounded': round(math.fsum(selected)/total*100, 1) if total else None}


def diagnostics():
    seven = namespace(0); crowd = namespace(1); rows = []
    def capture(name, fn):
        out = StringIO()
        try:
            with redirect_stdout(out): value = fn()
            value = json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
            def finite(item):
                if isinstance(item, float) and not math.isfinite(item): return None
                if isinstance(item, list): return [finite(v) for v in item]
                if isinstance(item, dict): return {k:finite(v) for k,v in item.items()}
                return item
            value = finite(value)
            error = None
        except Exception as exc: value = None; error = {'type': type(exc).__name__, 'message': str(exc)}
        rows.append({'case': name, 'value': value, 'error': error, 'stdout': out.getvalue(), 'synthetic': True})
    for size, equal, zero in ((20, False, False), (20, True, False), (1, False, False), (2, True, True)):
        amounts = [0.]*size if zero else ([1.]*size if equal else list(range(1, size+1)))
        frame = pd.DataFrame({'money': amounts}, index=pd.DatetimeIndex(['2021-01-04']*size))
        capture(f'crowd_n{size}_tie{equal}_zero{zero}', lambda frame=frame:
            crowd['get_crowd_rate_df'](frame).reset_index().to_dict('records'))
    capture('crowd_empty', lambda: crowd['get_crowd_rate_df'](pd.DataFrame({'money': []}, index=pd.DatetimeIndex([]))).to_dict())
    for factor in (1.7, np.nextafter(1.7, np.inf)):
        v = np.power(factor, np.arange(9, dtype=float))
        capture(f'volume_8_comparisons_{factor}', lambda v=v: bool(seven['count_multiple_stocks'](v, 7, 8)))
    capture('volume_eighth_comparison_only', lambda: bool(seven['count_multiple_stocks'](np.array([1.,2.,2.,2.,2.,2.,2.,2.,2.]), 7, 1)))
    capture('avg_equal', lambda: float(seven['get_avg_array'](np.array([2.,2.,2.]))))
    capture('avg_descending', lambda: float(seven['get_avg_array'](np.array([4.,3.,2.]))))
    capture('avg_wrong_order', lambda: float(seven['get_avg_array'](np.array([2.,3.,4.]))))
    for price in (100.,100.001,102.999,103.):
        ns = namespace(0); ns['get_price'] = lambda *a,price=price,**kw:pd.DataFrame({'open':[price],'pre_close':[100.]})
        ctx = SimpleNamespace(current_dt=pd.Timestamp('2021-01-05 09:30'))
        capture(f'current_open_{price}',lambda ns=ns,ctx=ctx:ns['get_open_high_history'](ctx,['A']))
    setup = []; ns = namespace(0)
    for name in ('set_benchmark','set_option','set_order_cost','set_slippage','run_daily'):
        def recorder(*args,name=name,**kw):
            setup.append({'api':name,'args':[a.__name__ if callable(a) else a for a in args],'kwargs':kw})
        ns[name] = recorder
    ns['OrderCost'] = lambda **kw:kw; ns['FixedSlippage'] = lambda x:{'fixed_slippage':x}
    capture('initialize_active_benchmark_fees_and_schedule',lambda:(ns['initialize'](None),setup)[1])
    for held in (0, 1, 2, 3):
        orders = []
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={f'H{i}': object() for i in range(held)}, available_cash=600.))
        # Stubbed nested calls explicitly isolate the original allocation callback.
        ns = namespace(0); ns['g'].security_first = ['H0','B','C']
        ns['stop_loss'] = lambda context: None
        ns['get_open_high_history'] = lambda context, stocks: stocks
        ns['order_value'] = lambda stock, value, **kw: orders.append({'stock':stock,'value':value})
        capture(f'trade_{held}_holds_includes_held_target', lambda ns=ns,ctx=ctx,orders=orders: (ns['trade'](ctx),orders)[1])
    for label, high, price, days in (('loss_exact', 1., .93, 0), ('loss_below',1., .929,0),
        ('gain20_exact',1.2,1.14,0), ('gain20_above',1.201,1.16,0), ('day3_5pct_exact',1.05,1.05,3),
        ('day3_below5pct',1.05,1.049,3), ('day1_equal_cost',1.,1.,1), ('day1_below_cost',1.,.999,1)):
        ns = namespace(0); orders = []
        pos = SimpleNamespace(init_time=date(2021,1,4),avg_cost=1.,price=price)
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'A':pos}),previous_date=date(2021,1,5))
        ns['get_price'] = lambda *a, high=high, **kw: pd.DataFrame({'high':[high]})
        ns['hold_days'] = lambda *a, days=days: days
        ns['order_target'] = lambda code,qty: orders.append([code,qty])
        capture('stop_'+label, lambda ns=ns,ctx=ctx,orders=orders:(ns['stop_loss'](ctx),orders)[1])
    # Empty daycount and zero piece_count are left as original failures.
    ns = namespace(1); ns['get_all_securities'] = lambda **kw: pd.DataFrame(index=['A'])
    ns['get_trade_days'] = lambda **kw: pd.DatetimeIndex([])
    capture('crowd_zero_days', lambda: ns['get_all_crowd_rate_df'](0,'2021-01-04',100).to_dict())
    capture('crowd_zero_piece', lambda: ns['get_all_crowd_rate_df'](1,'2021-01-04',0).to_dict())
    ns = namespace(0); ctx = SimpleNamespace(current_dt=date(2021,1,5),
        portfolio=SimpleNamespace(positions={'A':SimpleNamespace(init_time=date(2021,1,4))}))
    ns['jqdata'] = SimpleNamespace(get_trade_days=lambda start,end:[start,end])
    capture('inclusive_hold_days',lambda:ns['hold_days'](ctx,'A'))
    return {'scope':'original_definitions_on_explicit_fabricated_inputs','cases':rows,'case_count':len(rows),
            'nonfinite_result_serialization':'null; zero-money diagnostic retains original invalid arithmetic',
            'orders_are_recorded_intents_not_fills':True,'new_trading_backtests':0}


def trend_components():
    ns = namespace(0); prices = pd.read_parquet(DIRECTORY/'prices.parquet')
    cal = pd.read_parquet(DIRECTORY/'calendar.parquet'); days = pd.DatetimeIndex(cal.loc[cal.is_open,'date'])
    results = []; differences = []; maximum = 0.
    for symbol in SYMBOLS:
        frame = prices.loc[prices.instrument.eq(symbol)].sort_values('date').reset_index(drop=True)
        frame['date'] = pd.to_datetime(frame.date)
        first, last = frame.date.min(), frame.date.max()
        calendar = days[(days>=first)&(days<=last)]
        missing = pd.Series(~calendar.isin(frame.date), index=calendar).astype(int).cumsum()
        frame = frame.loc[frame.is_trading].reset_index(drop=True)
        for j in range(119,len(frame)):
            win = frame.iloc[j-119:j+1]; start = win.date.iloc[0]; end = win.date.iloc[-1]
            gap = int(missing.loc[end] - missing.loc[start])
            if gap: continue  # Absent observations are never treated as verified suspension.
            close = win.close.to_numpy(float); volume = win.volume.to_numpy(float); opens = win.open.to_numpy(float)
            if not (np.isfinite(close).all() and np.isfinite(volume).all() and (close>0).all() and (opens>0).all()): continue
            ma = {n:ns['get_ma'](close,n) for n in (5,13,21,55)}
            count13 = int(ns['get_bigger_than_val_counter_noflow'](ma[13],ma[21],6))
            count21 = int(ns['get_bigger_than_val_counter_noflow'](ma[21],ma[55],6))
            volpass = bool(ns['count_multiple_stocks'](volume,7,3))
            today = bool(not(volume[-1]<volume[-2]*1.5 or close[-1]/opens[-1]<1.05 or close[-1]/close[-2]<1))
            avgs = np.array([ma[13],ma[21],ma[55]]).T
            box = sum(ns['get_std_percentage'](avgs[-i]) for i in range(2,22))
            array = sum(ns['get_avg_array'](avgs[-i]) for i in range(1,6)); score = float(box-array)
            # Independent fsum only for relevant endpoints, not the implementation's rolling accumulator.
            stable = {n:[math.fsum(close[k-n+1:k+1])/n for k in range(98,120)] for n in (13,21,55)}
            c13 = sum(a>b for a,b in zip(stable[13][-6:],stable[21][-6:],strict=True))
            c21 = sum(a>b for a,b in zip(stable[21][-6:],stable[55][-6:],strict=True))
            vc = sum(float(volume[-i])>1.7*float(volume[-i-1]) for i in range(1,9))
            independent_array = 0.
            # box excludes the last observation: original -2..-21 -> absolute 99..118.
            independent_box = math.fsum(math.sqrt(math.fsum(((x-sorted(v)[1])/sorted(v)[1])**2 for x in v)/3)
                for k in range(99,119) for v in [[stable[n][k-98] for n in (13,21,55)]])
            for k in range(115,120):
                a,b,c = (stable[n][k-98] for n in (13,21,55))
                independent_array += 0. if a<b or b<c else (a-b)/b+(b-c)/c
            independent_score = independent_box-independent_array
            maximum = max(maximum,abs(score-independent_score))
            require(math.isclose(score,independent_score,rel_tol=2e-9,abs_tol=2e-10), 'Independent score differs materially')
            original = (count13>=5,count21>=5,volpass,today,score>.005)
            independent_today = (float(volume[-1])>=float(volume[-2])*1.5 and
                float(close[-1])/float(opens[-1])>=1.05 and float(close[-1])/float(close[-2])>=1)
            independent = (c13>=5,c21>=5,vc>=3,independent_today,independent_score>.005)
            if original != independent: differences.append({'instrument':symbol,'date':str(end.date()),'original':original,'stable':independent})
            results.append({'instrument':symbol,'date':end,'window_start':start,'window_rows':120,'ma13_count':count13,
                'ma21_count':count21,'volume_pass':volpass,'last_bar_pass':today,'score':score,'stable_score':independent_score,
                'price_volume_prefilter':all(original[:4]),'score_threshold':original[4],
                'stable_predicates_match':original==independent})
    out = pd.DataFrame(results)
    require(len(out)>0 and set(out.instrument)==set(SYMBOLS), 'Empty or incomplete arithmetic scope')
    return out, {'reference_windows':len(out),'first':str(out.date.min().date()),'last':str(out.date.max().date()),
        'price_volume_prefilter_true':int(out.price_volume_prefilter.sum()),'maximum_score_difference':maximum,
        'independent_predicate_differences':differences,'original_full_pool_selection_computed':False,
        'limits':['Ten existing history symbols, raw unadjusted OHLC/volume, missing observations blocked',
            '120 trading observations recomputed each window; no total-cap/name preselection, current-open or minute stops',
            'Reference predicates are not original signals or targets; floating-boundary differences retained']}


def crowd_components():
    ns = namespace(1); amounts = pd.read_parquet(DIRECTORY/'amounts.parquet')
    amounts['date'] = pd.to_datetime(amounts.date)
    dates = pd.DatetimeIndex(sorted(amounts.date.unique())); calls = []
    population = sorted(amounts.instrument.unique())
    def securities(**kw):
        calls.append({'api':'get_all_securities','parameters':kw}); return pd.DataFrame(index=population)
    def calendar(**kw):
        calls.append({'api':'get_trade_days','parameters':kw}); return dates
    def prices(stocks, **kw):
        calls.append({'api':'get_price','population_count':len(stocks),'parameters':{k:str(v) for k,v in kw.items()}})
        view = amounts.loc[amounts.date.between(kw['start_date'],kw['end_date'])]
        return view.rename(columns={'date':'time','amount':'money'}).copy()
    ns.update(get_all_securities=securities,get_trade_days=calendar,get_price=prices)
    original = ns['get_all_crowd_rate_df'](len(dates),'2026-09-29',100)
    require(len(original)==len(dates) and sum(c['api']=='get_all_securities' for c in calls)==1,
            'Original chunk population or day coverage differs')
    results = []
    for day, group in amounts.groupby('date', sort=True):
        independent = independent_crowd(group.amount)
        actual = float(original.at[day.strftime('%Y-%m-%d'),'crowd_rate'])
        require(actual==independent['rounded'], 'Independent crowd arithmetic differs')
        results.append({'date':day,'crowd_rate':actual,**independent})
    out = pd.DataFrame(results)
    return out, {'rows':len(out),'money_observations':len(amounts),'population_count':len(population),
        'first':str(out.date.min().date()),'last':str(out.date.max().date()),'api_trace':calls,
        'independent_rounded_differences':0,'original_historical_population_proven':False,
        'limits':['Available daily observation population, not proven original end-date stock universe',
            'Original 0-based floor5pct threshold is inclusive and admits ties; this is not exact top5pct',
            'Single end-date universe has survivorship bias; original is a chart script with no trading rules']}


def calculate(destination):
    require(not destination.exists(), 'Immutable output directory exists')
    destination.mkdir(parents=True,exist_ok=False)
    with network_blocked():
        diag = diagnostics(); print('Calculating seven-day reference', flush=True)
        trend, ts = trend_components(); print('Calculating crowd reference', flush=True)
        crowd, cs = crowd_components()
    for name,frame in (('trend-components.parquet',trend),('crowd-components.parquet',crowd)):
        with (destination/name).open('xb') as out: frame.to_parquet(out,index=False)
    save(destination/'diagnostics.json',diag)
    yearly = crowd.assign(year=crowd.date.dt.year).groupby('year').agg(days=('crowd_rate','size'),
        minimum=('crowd_rate','min'),maximum=('crowd_rate','max'),mean=('crowd_rate','mean')).reset_index().to_dict('records')
    save(destination/'summary.json',{'status':'ok','scope':'source_diagnostics_and_real_data_arithmetic_only',
        'trend':ts,'crowd':cs,'crowd_yearly_statistics_not_returns':yearly,'diagnostic_cases':diag['case_count'],
        'network_forbidden':True,'new_trading_backtests':0,'new_strategy_parent_runs':0,
        'original_strategy_complete':False,'formal_source_research_acceptance':False})
    return {'status':'ok','outputs_sha256':{n:file_sha(destination/n) for n in OUTPUTS}}


def run():
    binding(); result = calculate(DIRECTORY/'study')
    save(DIRECTORY/'study-receipt.json',{'prepared_sha256':file_sha(DIRECTORY/'prepared.json'),**result})
    return result


def replay():
    binding(); receipt = read(DIRECTORY/'study-receipt.json')
    require(receipt['prepared_sha256']==file_sha(DIRECTORY/'prepared.json') and
            set(receipt['outputs_sha256'])==set(OUTPUTS), 'Study receipt differs')
    for name,digest in receipt['outputs_sha256'].items(): require(file_sha(DIRECTORY/'study'/name)==digest, 'Study output changed')
    folder = DIRECTORY/'fresh-worker'; require(not folder.exists(), 'Worker archive exists'); folder.mkdir()
    command = [sys.executable,'-B','-m','scripts.research_strategy_batch97','worker']
    with (folder/'stdout.log').open('xb') as log:
        proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd())
        rc = proc.wait()
    save(folder/'command.json',{'command':command,'worker_pid':proc.pid,'returncode':rc,
        'log_sha256':file_sha(folder/'stdout.log'),'network_forbidden_inside_worker':True})
    require(rc==0, 'Fresh worker failed; retained log')
    result = json.loads((folder/'stdout.log').read_text().splitlines()[-1])
    require(result['status']=='ok' and result['outputs_sha256']==receipt['outputs_sha256'], 'Worker stdout differs')
    for name,digest in receipt['outputs_sha256'].items():
        require(file_sha(DIRECTORY/'replay'/name)==digest, 'Worker bytes differ')
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study'/name),pd.read_parquet(DIRECTORY/'replay'/name),check_exact=True)
        else: require(read(DIRECTORY/'study'/name)==read(DIRECTORY/'replay'/name),'Worker JSON differs')
    result = {'status':'match','differences':0,'files':len(OUTPUTS),
        'fresh_worker_command':str(folder/'command.json'),'prepared_sha256':file_sha(DIRECTORY/'prepared.json')}
    save(DIRECTORY/'offline-verification.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker','replay'))
    action = parser.parse_args().action
    if action == 'worker': binding(); result = calculate(DIRECTORY/'replay')
    else: result = {'prepare':prepare,'run':run,'replay':replay}[action]()
    print(json.dumps(result,ensure_ascii=False))
