"""Batch96 source diagnostics and daily arithmetic stage; never a trading backtest."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime
import hashlib
import inspect
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

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch96/20261011-original-callback-stage')
PREREAD = Path('data/staging/strategies-batch96/20261011-source-preread')
PARENT = Path('docs/handoff/2026-10-11-batch95-final-review.json')
PARENT_SHA = 'c48fefcbd1391a23564a49baa7c0af8672cd2991b56e522b6ac34ec687fdc9c5'
SHA = ('28b6333c10cf48b6ccc53a3bdfb3ae7dd15a09eabca0362efbb3c425dce8882d',
       'e88037b85d55b6d9879e0514663ac76ee667ca23cb140985191607216d8b9353')
FILES = ('scripts/review_strategy_batch96.py', 'tests/unit/test_batch96_stage.py',
    'scripts/archive_strategy_progress.py', 'scripts/recover_strategy_batch94_reference.py',
    'scripts/review_strategy_batch18.py', 'scripts/review_strategy_batch73.py',
    'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py',
    'src/observe/runs.py', 'src/observe/data/store.py', 'src/observe/data/locks.py',
    'src/observe/strategy_catalog.py')
CANDIDATES = (
    ('399001.SZ', 'data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight/probe-results.json', 'index-sz399001'),
    ('000016.SH', 'data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/results.json', 'index-sh000016'))
SYMBOLS = ('000001.SH', '399001.SZ', '000016.SH', '399333.SZ', '159901.SZ')
OUTPUTS = ('diagnostics.json', 'daily-arithmetic.parquet', 'component-research.json')


def parent_binding():
    require(file_sha(PARENT) == PARENT_SHA, 'Accepted Batch95 receipt changed')
    doc = read(PARENT); binding = read(doc['final_binding_file'])
    require(doc['status'] == 'ok' and doc['formal_source_research_acceptance'] and
        file_sha(doc['final_binding_file']) == doc['final_binding_sha256'] and
        file_sha(doc['receipt_file']) == doc['receipt_sha256'] == binding['receipt_sha256'], 'Parent binding differs')
    for name, digest in binding['evidence_sha256'].items():
        require(file_sha(name) == digest, 'Accepted parent evidence changed: '+name)
    return {'receipt': str(PARENT), 'sha256': PARENT_SHA,
        'final_binding_sha256': doc['final_binding_sha256'], 'bound_files': len(binding['evidence_sha256'])}


def literal(directory, number, namespace=None):
    require(number in (0, 1), 'Unknown Batch96 source')
    path = directory/'sources'/f'{number}.source'
    require(file_sha(path) == SHA[number], 'Source identity differs')
    text = read_source(path)[0]
    # RF has eleven frozen prose/comment lines. Blank them to retain original traceback line numbers.
    if number == 0:
        lines = text.splitlines(keepends=True)
        require(lines[11].strip() == 'import math', 'RF code-start contract differs')
        text = '\n'*11 + ''.join(lines[11:])
    tree = ast.parse(text); nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (36 if number == 0 else 16) and all(safe_function(n) for n in nodes), 'Definition contract differs')
    ns = {} if namespace is None else namespace
    # The raw imports, read_file, top-level platform code and missing user modules never execute.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def candidate_frame(row):
    require(file_sha(row['file']) == row['sha256'], 'Candidate byte SHA differs')
    for wire in row['wire_responses']:
        require(file_sha(wire['file']) == wire['sha256'], 'Candidate wire SHA differs')
    frame = pd.read_parquet(row['file']).copy()
    require(len(frame) == row['rows'], 'Candidate count differs')
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    require(frame.date.is_monotonic_increasing and not frame.date.duplicated().any(), 'Ambiguous candidate dates')
    for name in ('close', 'high', 'low'):
        frame[name] = pd.to_numeric(frame[name], errors='raise').astype(float)
        require(np.isfinite(frame[name]).all() and (frame[name] > 0).all(), 'Invalid candidate OHLC')
    return frame.loc[frame.date.between('2005-01-05', '2026-09-29')].reset_index(drop=True)


def inventory():
    state = Store('data').published(); tables = {}; candidates = []
    require(state['batch_id'] == '20261006-145049-5ceb', 'Publication differs')
    with duckdb.connect() as con:
        for name in ('index_1d', 'bars_1d', 'bars_5m', 'minute_universe', 'adj_factors', 'adj_coverage', 'corp_actions'):
            files = [str(Path('data')/r['file']) for r in state['tables'].get(name, {}).values()]
            column = 'index' if name == 'index_1d' else 'instrument'
            counts = dict.fromkeys(SYMBOLS, 0)
            if files:
                rows = con.execute(f'SELECT "{column}", count(*) FROM read_parquet(?, union_by_name=true) '
                    f'WHERE "{column}" IN (?, ?, ?, ?, ?) GROUP BY "{column}"', [files, *SYMBOLS]).fetchall()
                counts.update({s: int(n) for s,n in rows})
            tables[name] = {'counts': counts, 'partition_count': len(files)}
    for symbol, registry, case in CANDIDATES:
        check = Path(registry).parent/'checked-state.json'
        if case == 'index-sh000016': check = Path(registry).parent.parent/'checked-state.json'
        relative = str(Path(registry).relative_to(check.parent))
        require(file_sha(registry) == read(check)['evidence_sha256'][relative], 'Prior candidate registry changed')
        row = next(r for r in read(registry)['results'] if r['case'] == case)
        frame = candidate_frame(row)
        candidates.append({'symbol': symbol, 'row': row, 'registry': registry,
            'registry_sha256': file_sha(registry), 'prior_checked_state': str(check),
            'prior_checked_state_sha256': file_sha(check), 'rows_to_cutoff': len(frame),
            'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'strict_usable': False})
    return {'published_batch': state['batch_id'], 'tables': tables, 'candidates': candidates,
        'bars_1m_registered': 'bars_1m' in state['tables'], 'original_trade_inputs_admitted': False,
        'limits': ['Daily candidates are not original 14:30/14:50 minute input',
            'No replacement index or regenerated RF predictions', 'Fund fees/events/state/execution still need proof'],
        'partition_check': 'Filtered row inventory; old partition protection separately uses size/mtime'}


def prepare(directory):
    require(directory == DIRECTORY and not directory.exists(), 'Stage exists or is not canonical')
    parents = parent_binding(); inputs = inventory(); sources = read(PREREAD/'source-preread.json')['sources']
    for number, row in enumerate(sources):
        require(file_sha(row['frozen_source']) == file_sha(Path('repo/量化策略源代码')/row['path']) == SHA[number], 'Original source differs')
    hashes = {name: file_sha(name) for name in FILES}
    with operation_lock('data', DATA_WRITER):
        require(Store('data').published()['batch_id'] == inputs['published_batch'], 'Publication changed in preparation')
        checkpoint('data', directory, 'Batch96 original callback stage started; formal276/trade12/parents77 unchanged')
        save(directory/'parent-binding.json', parents); save(directory/'input-inventory.json', inputs)
        (directory/'sources').mkdir()
        for number in range(2): shutil.copyfile(PREREAD/f'{number}.source', directory/'sources'/f'{number}.source')
        for name in FILES:
            target = directory/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        save(directory/'implementation-binding.json', hashes)
    return {'status': 'prepared', 'candidates': [(r['symbol'], r['rows_to_cutoff']) for r in inputs['candidates']]}


def validate(directory):
    require(read(directory/'parent-binding.json') == parent_binding(), 'Parent differs')
    hashes = read(directory/'implementation-binding.json')
    require(set(hashes) == set(FILES), 'Implementation scope differs')
    for name, digest in hashes.items():
        require(file_sha(name) == file_sha(directory/'implementation'/name) == digest, 'Frozen implementation differs: '+name)
    for number in range(2): literal(directory, number)
    for row in read(directory/'input-inventory.json')['candidates']:
        require(file_sha(row['registry']) == row['registry_sha256'] and
            file_sha(row['prior_checked_state']) == row['prior_checked_state_sha256'], 'Candidate provenance differs')
        candidate_frame(row['row'])


def logger():
    return SimpleNamespace(**{k: lambda *a, **kw: None for k in ('info','debug','error','warn')})


def rf_namespace(directory):
    ns = literal(directory, 0, {'g': SimpleNamespace(), 'np': np, 'pd': pd,
        'datetime': datetime, 'math': math, 'log': logger()})
    ns['set_param'](); return ns


def zhu_namespace(directory):
    ns = literal(directory, 1, {'g': SimpleNamespace(), 'np': np, 'datetime': datetime, 'log': logger()})
    for name in ('top', 'topgender', 'buttom', 'buttomgender'): ns[name] = [False]*199
    return ns


def captured(label, function):
    out = StringIO(); error = None; value = None
    try:
        with redirect_stdout(out): value = function()
    except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)[:500]}
    return {'case': label, 'value': value, 'error': error, 'stdout': out.getvalue()}


def diagnostics(directory):
    rows = []
    # Every case runs original definitions; stubs and fabricated feedback are explicit research inputs.
    for case in ('csv_order', 'csv_missing_date', 'csv_duplicate_date'):
        ns = rf_namespace(directory); index = ['2020-01-02']*(2 if case == 'csv_duplicate_date' else 1)
        ns['g'].data_ = pd.DataFrame([[1, 0, 1.0, np.nan]]*len(index), index=index,
            columns=['600519.XSHG', '000001.XSHE', '600036.XSHG', '000002.XSHE'])
        ctx = SimpleNamespace(previous_date=datetime.date(2020,1,3 if case == 'csv_missing_date' else 2))
        def invoke():
            ns['before_trading_start'](ctx); return ns['g'].stocks.tolist()
        rows.append(captured(case, invoke))
    for count in (0, 1, 2, 3, -1):
        for growth in ((.01,.01), (np.nextafter(.01, math.inf),.01), (math.nan,math.nan)):
            ns = rf_namespace(directory); ns['g'].day_count = count; ns['g'].stocks = ['A','B','C','D','E','F']
            events = []; ns['get_growth_rate'] = lambda symbol: growth[symbol == '399333.XSHE']
            ns['get_security_info'] = lambda s: SimpleNamespace(display_name=s)
            ns['clear_position'] = lambda ctx: events.append('clear')
            ns['adjust_position'] = lambda ctx, stocks: events.append(['adjust',list(stocks)])
            def invoke():
                ns['do_handle_data'](SimpleNamespace(), {}); return {'count': ns['g'].day_count, 'events': events}
            rows.append(captured(f'rf_clock_{count}_{list(growth)}', invoke))
    for label in ('datetime', 'range'):
        ns = rf_namespace(directory)
        ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': [90.,100.]},
            index=pd.date_range('2020-01-01',periods=2) if label == 'datetime' else range(2))
        rows.append(captured('rf_history_'+label, lambda: ns['get_close_price']('x',2)))
    ns = rf_namespace(directory); ns['get_close_price'] = lambda *a: 100.
    rows.append(captured('rf_missing_isnan', lambda: ns['get_growth_rate']('x')))
    for mode in ('plain_empty', 'reject', 'partial', 'full', 'mutating_sell'):
        ns = rf_namespace(directory); orders = []
        class ZeroPositions(dict):
            def __missing__(self,key): return SimpleNamespace(security=key,total_amount=0)
        positions = {} if mode == 'plain_empty' else ZeroPositions()
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=positions,cash=1000.))
        ns['get_close_price'] = lambda *a:100.
        ns['g'].last_high = {}; ns['g'].trade_stat = SimpleNamespace(watch=lambda *a:None)
        ns['OrderStatus'] = SimpleNamespace(held='held')
        if mode == 'mutating_sell':
            positions['OLD'] = SimpleNamespace(security='OLD',total_amount=100,avg_cost=100.,price=100.)
        def order(symbol,value):
            orders.append([symbol,value])
            if mode == 'reject': return None
            if value == 0:
                del positions[symbol]
            else:
                positions[symbol] = SimpleNamespace(security=symbol,total_amount=1)
            return SimpleNamespace(filled=1,amount=2 if mode=='partial' else 1,status='held')
        ns['order_target_value'] = order
        row = captured('rf_adjust_'+mode,lambda:ns['adjust_position'](ctx,['A','B','C','D','E','F']))
        row.update(orders=orders,remaining_positions=list(positions)); rows.append(row)
    for case, values in (('flat',[100.]*200), ('oscillating',(100+10*np.sin(np.arange(200)/5)).tolist())):
        ns = zhu_namespace(directory)
        def invoke():
            crosses = ns['hasCross'](values); deads,golds = map(ns['barslast'],crosses)
            result = ns['core_ssdd'](deads,golds,values,ns['top'],ns['topgender'],ns['buttom'],ns['buttomgender'])
            return {'deads':deads, 'golds':golds, 'lengths':[len(s) for s in result], 'last':[bool(s[-1]) for s in result]}
        row = captured('zhu_core_'+case, invoke); row['state_lengths_after'] = [len(ns[k]) for k in ('top','topgender','buttom','buttomgender')]
        rows.append(row)
    ns = zhu_namespace(directory)
    # Force bottom history shortage after the original top branch has appended. No source edits or rollback.
    row = captured('zhu_partial_state_on_error', lambda: ns['core_ssdd']([80,20], [100,60,40],
        np.linspace(100,110,200), ns['top'],ns['topgender'],ns['buttom'],ns['buttomgender']))
    row['state_lengths_after'] = [len(ns[k]) for k in ('top','topgender','buttom','buttomgender')]; rows.append(row)
    for size in (0,199,200):
        ns = zhu_namespace(directory); flags = [False]*size
        if size: flags[-1] = True
        rows.append(captured('barslast_'+str(size), lambda: ns['barslast'](flags)))
    for age in (23,24):
        ns = zhu_namespace(directory); flags = [False]*199; flags[199-age] = True
        rows.append(captured('tradeHelper1_'+str(age), lambda: ns['tradeHelper1']([40],flags)))
    ns = zhu_namespace(directory)
    rows.append(captured('tradeHelper_empty',lambda: ns['tradeHelper1']([40],[False]*199)))
    for day in (datetime.datetime(2016,12,30), datetime.datetime(2017,1,2)):
        ns = zhu_namespace(directory)
        def invoke():
            ns['before_market_open'](SimpleNamespace(current_dt=day)); return ns['g'].security
        rows.append(captured('index_switch_'+str(day.date()),invoke))
    ns = zhu_namespace(directory); events = []
    ns.update(set_benchmark=lambda s:events.append(['benchmark',s]), set_option=lambda *a:None,
        OrderCost=lambda **kw:kw, set_order_cost=lambda cost,**kw:events.append(['cost',cost,kw]),
        run_daily=lambda fn,**kw:events.append(['schedule',fn.__name__,kw]))
    row = captured('zhu_initialize_stock_cost',lambda:ns['initialize'](SimpleNamespace()))
    row['events'] = events; rows.append(row)
    ns = zhu_namespace(directory); ns['g'].security = '399001.XSHE'; orders = []; calls = []
    def bars(symbol,count,unit,fields):
        calls.append([symbol,count,unit,list(fields)])
        return {c:np.full(count,100.) for c in fields}
    ns.update(get_bars=bars,order_target_value=lambda *a:orders.append(list(a)))
    ctx = SimpleNamespace(current_dt=datetime.datetime(2020,1,2,14,30),
        portfolio=SimpleNamespace(available_cash=1000.,total_value=1000.))
    row = captured('zhu_market_flat_before_orders',lambda:ns['market_open'](ctx))
    row.update(orders=orders,history_queries=calls); rows.append(row)
    return {'scope':'controlled_original_definition_diagnostics', 'cases': rows,
        'source_sha256': list(SHA), 'new_trading_backtests':0, 'original_platform_equivalence_proven':False,
        'injections':['np/datetime/log/platform feedback supplied explicitly; no ind/tradestat module reconstruction',
            'RF adjustment and clearing spies are not fills; all CSV rows are synthetic',
            'RF order feedback/virtual zero positions/deletion are controlled and have no fees/NAV',
            'Zhu price/flags are synthetic; no minute data or original signal path claimed']}


def scalar_ema(values, period):
    # Independent expanding prefix weights, then fixed-period recurrence; first observation seed.
    previous = float(values[0]); result = [previous]
    for ordinal, value in enumerate(values[1:], 2):
        alpha = 2./(min(ordinal,period)+1)
        previous = float(value)*alpha + (1-alpha)*previous; result.append(previous)
    return np.asarray(result)


def components(directory):
    inputs = read(directory/'input-inventory.json'); row = next(r for r in inputs['candidates'] if r['symbol']=='399001.SZ')
    frame = candidate_frame(row['row']); ns = zhu_namespace(directory); rows = []; maximum = 0.; cross_differences = 0
    for k in range(198,len(frame)):
        window = frame.iloc[k-198:k+1]; close,high,low = (window[c].to_numpy(float) for c in ('close','high','low'))
        actual = ns['macdx'](close); dif = scalar_ema(close,12)-scalar_ema(close,26); dea = scalar_ema(dif,9)
        expected = [dif,dea,(dif-dea)*2]; trends = ns['trend'](high,low)
        reference = [scalar_ema(high,25),scalar_ema(low,25),scalar_ema(high,90),scalar_ema(low,90)]
        for a,b in zip([*actual,*trends],[*expected,*reference],strict=True):
            maximum = max(maximum,float(np.max(np.abs(a-b))))
            require(np.allclose(a,b,rtol=1e-12,atol=1e-9),'Independent EMA/MACD mismatch')
        aa,bb = np.asarray(actual[2]),expected[2]
        cross_differences += int(np.count_nonzero(((aa[1:]<0)&(aa[:-1]>=0)) != ((bb[1:]<0)&(bb[:-1]>=0))))
        cross_differences += int(np.count_nonzero(((aa[1:]>0)&(aa[:-1]<=0)) != ((bb[1:]>0)&(bb[:-1]<=0))))
        rows.append({'date':frame.date.iloc[k], 'dif':float(actual[0][-1]), 'dea':float(actual[1][-1]),
            'macd':float(actual[2][-1]), 'short_high':float(trends[0][-1]), 'short_low':float(trends[1][-1]),
            'long_high':float(trends[2][-1]), 'long_low':float(trends[3][-1])})
    require(cross_differences == 0,'Independent crossing predicate differs')
    return pd.DataFrame(rows), {'scope':'199_daily_row_indicator_reference_only', 'symbol':'399001.SZ',
        'windows':len(rows), 'first':rows[0]['date'], 'last':rows[-1]['date'],
        'full_array_independent_maximum_difference':maximum, 'cross_predicate_differences':cross_differences,
        'original_14_30_signals':0, 'state_machine_history_computed':False, 'new_trading_backtests':0,
        'limits':['199 daily rows only; original callback needs 199 daily plus one minute bar (200 values)',
            '199-row EMA reseeded each window as source helper; TA-Lib would change early seed',
            'Pre2017 Shenzhen candidate is arithmetic evidence, not the original Shanghai reference index',
            'No targets, costs, ledger, NAV, or platform-equivalence claim']}


def compute(directory):
    validate(directory)
    require(not any((directory/name).exists() for name in OUTPUTS), 'Stage already computed')
    with network_blocked():
        result = diagnostics(directory); frame, summary = components(directory)
    save(directory/'diagnostics.json',result)
    with (directory/'daily-arithmetic.parquet').open('xb') as stream: frame.to_parquet(stream,index=False)
    save(directory/'component-research.json',summary)
    return {'diagnostics':len(result['cases']), **summary}


def worker(directory):
    validate(directory)
    with network_blocked(): result = diagnostics(directory); frame,summary = components(directory)
    require(result == read(directory/'diagnostics.json'),'Diagnostic replay differs')
    pd.testing.assert_frame_equal(frame,pd.read_parquet(directory/'daily-arithmetic.parquet'),check_exact=True)
    require(summary == read(directory/'component-research.json'),'Component replay differs')
    return {'status':'ok','differences':0,'diagnostics':len(result['cases']),'windows':len(frame),
        'network':'DNS, socket connect/connect_ex/create_connection and requests forbidden'}


def replay(directory):
    validate(directory); require(not (directory/'offline-verification.json').exists(),'Replay archive exists')
    command = [sys.executable,'-m','scripts.review_strategy_batch96','worker','--directory',str(directory)]
    process = subprocess.run(command,capture_output=True,text=True,timeout=120)
    save(directory/'offline-command.json',{'command':command,'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr})
    require(process.returncode == 0,'Fresh forbidden-network replay failed')
    result = json.loads(process.stdout); require(result['status']=='ok' and result['differences']==0,'Replay result differs')
    save(directory/'offline-verification.json',result); return result


def api_inventory(directory):
    validate(directory)
    import akshare as ak
    functions = {}
    for name in ('stock_zh_index_daily','fund_etf_hist_sina','index_zh_a_hist_min_em'):
        fn = getattr(ak,name); code = inspect.getsource(fn)
        functions[name] = {'signature':str(inspect.signature(fn)), 'source':code,
            'sha256':hashlib.sha256(code.encode()).hexdigest()}
    save(directory/'existing-apis.json',{'akshare':ak.__version__, 'functions':functions,
        'minute_limit':'Installed index period1 explicitly requests ndays5; date arguments cannot restore decades',
        'original_RF_CSV_regenerable_from_market_API':False})
    command = [sys.executable,'-c',"import json,socket; socket.getaddrinfo('finance.sina.com.cn',443); print(json.dumps({'status':'resolver_ok'}))"]
    try:
        process = subprocess.run(command,capture_output=True,text=True,timeout=8)
        result = {'status':'resolver_ok' if process.returncode == 0 else 'resolver_failed',
            'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    except subprocess.TimeoutExpired:
        result = {'status':'resolver_timeout','deadline_seconds':8}
    save(directory/'supplement-capability.json',{'host':'finance.sina.com.cn','observation':result,
        'DNS_observations':1,'HTTP_attempts':0,'new_provider_rows':0,'published':False,
        'next_step':'If resolved, make bounded native daily requests in a new directory; otherwise no symbol-only retries',
        'limits':['Resolver failure describes this host/environment now, not supplier history availability',
            'Original predictions and long minute/event/fund inputs remain missing']})
    return result


def finish_stage(directory):
    validate(directory); require(read(directory/'offline-verification.json')['status']=='ok','Offline check missing')
    evidence = {str(p):file_sha(p) for name in (*OUTPUTS,'offline-command.json','offline-verification.json',
        'input-inventory.json','existing-apis.json','supplement-capability.json') if (p:=directory/name).is_file()}
    require(len(evidence)==8,'Stage evidence incomplete')
    with operation_lock('data',DATA_WRITER):
        protection = protect('data',directory)
        progress = archive('data','Batch96 original CSV/clock/structure diagnostics and daily EMA reference; pending formal;276/12/77 retained')
        require((progress['manually_reviewed'],progress['locally_reproduced_sources'],progress['local_strategy_runs'])==(276,12,77),'Unexpected progress counts')
        save(directory/'stage-checkpoint.json',{'status':'in_progress','progress':progress,'evidence_sha256':evidence,
            'protection':protection,'formal_source_research_acceptance':False,'formal_trading_acceptance':False})
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','compute','worker','replay','api_inventory','finish_stage'))
    parser.add_argument('--directory',type=Path,default=DIRECTORY); args=parser.parse_args()
    require(args.directory == DIRECTORY,'Noncanonical Batch96 directory')
    print(json.dumps(globals()[args.action](args.directory),ensure_ascii=False,default=str))


if __name__ == '__main__': main()
