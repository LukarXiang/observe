"""Freeze dynamic-emotion/two-eight rotation and longest available reference operands."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
from fractions import Fraction
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
from scripts import probe_strategy_batch74_fallback as native
from scripts import recover_strategy_batch75 as recovery
from scripts import review_strategy_batch75 as previous
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2022年度精选策略/28.etf轮动的优化，680%收益.txt',
    '2022年度精选策略/52.用指数战胜指数，ETF二八轮动对冲模型.txt')
SOURCE_SHA = ('c140b79604a77fd939128c7ebdc48a4d92ece652a68efc16806957d426bee369',
    'b9965d411543db3075a4fa46dddd96e6b13398f6395fab9c90eb838f56681068')
PAIRS = {'399001.XSHE': '150019.XSHE', '000016.XSHG': '510050.XSHG',
    '000010.XSHG': '510180.XSHG', '000015.XSHG': '510880.XSHG',
    '399324.XSHE': '159905.XSHE', '399006.XSHE': '159915.XSHE',
    '000905.XSHG': '510500.XSHG', '399673.XSHE': '159949.XSHE'}
FUNDS = (*PAIRS.values(), '511880.XSHG', '150051.XSHE', '510300.XSHG')
QUERIES = {'fund-'+code: {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz'+code if code.startswith(('15',)) else 'sh'+code}}
    for code in ('150019', '510180', '510880', '159905', '159949', '511880', '150051')}
QUERIES.update({'index-'+symbol: {'function': 'stock_zh_index_daily', 'parameters': {'symbol': symbol}}
    for symbol in ('sz399001', 'sh000010', 'sh000015', 'sz399324', 'sh000905', 'sz399673')})
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/akshare', 'akshare/index/index_stock_zh.py'),
    ('repo/backtrader', 'backtrader/indicators/sma.py'))
FILES = tuple(sorted(set(recovery.FILES) | {'scripts/review_strategy_batch76.py', 'tests/unit/test_batch76_research.py',
    'scripts/review_strategy_batch73.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'reused-inputs.json', 'dependency-inventory.json', 'existing-apis.json',
    'probe-results.json', 'candidate-audit.json', 'diagnostics.json', 'fund-history-components.parquet',
    'index-components.parquet', 'paired-power-components.parquet', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json'}
CORE |= {'driver-before-fixture-correction.py', 'test-before-fixture-correction.py', 'initial-test-failure.json'}
CORE |= {'driver-before-probe-binding-correction.py', 'probe-binding-red-regression.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-08-batch75-verification.json')
    final = Path('docs/handoff/2026-10-08-batch75-final-review.json')
    proof = read(previous.DIRECTORY / 'final-binding-verification.json')
    accepted = read(final)
    require(read(receipt)['status'] == proof['status'] == accepted['status'] == 'ok' and
        file_sha(receipt) == proof['receipt_sha256'] == accepted['receipt_sha256'] and
        file_sha(previous.DIRECTORY / 'checked-state.json') == proof['checked_state_sha256'] == accepted['checked_state_sha256'] and
        file_sha(previous.DIRECTORY / 'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted batch75 changed')
    for name in ('scripts/review_strategy_batch75.py', 'scripts/recover_strategy_batch75.py',
            'tests/unit/test_batch75_research.py', 'tests/unit/test_batch75_recovery.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(final), 'review_sha256': file_sha(final), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    return result


def reused_inputs():
    result = {}
    checked75 = read(previous.DIRECTORY / 'checked-state.json')['evidence_sha256']
    for name in ('reused-ETF.json', 'probe-results.json', 'index-input.parquet'):
        require(file_sha(previous.DIRECTORY / name) == checked75[name], 'Prior75 input changed')
    result['index-000300'] = {'file': str(previous.DIRECTORY / 'index-input.parquet'),
        'sha256': checked75['index-input.parquet'], 'rows': 5280, 'origin': 'Accepted75 snapshot reference'}
    result['fund-510300'] = read(previous.DIRECTORY / 'reused-ETF.json')['payload']
    for row in read(previous.DIRECTORY / 'probe-results.json')['results']:
        require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'], 'Prior75 ETF changed')
        result['fund-'+row['case']] = row
    prior = native.DIRECTORY / 'results.json'
    require(file_sha(prior) == read('docs/handoff/2026-10-08-batch74-verification.json')['checks']['evidence_sha256']['sina-fallback/results.json'],
        'Prior74 candidates changed')
    for row in read(prior)['results']:
        if row['case'] in ('fund-510050', 'index-sh000016', 'index-sz399006'):
            require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'], 'Prior74 payload changed')
            result[row['case'].replace('index-sh', 'index-').replace('index-sz', 'index-')] = row
    require(len(result) == 7, 'Reuse scope incomplete')
    for row in result.values(): require(file_sha(row['file']) == row['sha256'] and len(pd.read_parquet(row['file'])) == row['rows'], 'Reuse payload changed')
    return {'inputs': result, 'new_requests': 0, 'strict_usable': False, 'not_a_backtest': True}


def api_evidence():
    with patch.object(native, 'QUERIES', QUERIES): result = native.api_evidence()
    result['talib'] = {'version': talib.__version__, 'C_version': talib.__ta_version__.decode(),
        'compatibility': talib.get_compatibility(), 'MA': 'Default matype0,100-row reinitialization per original callback'}
    return result


def inventory(root, directory):
    store = Store(root); state = store.state(SNAPSHOT)
    canonical = [s[:6]+('.SH' if s.endswith('XSHG') else '.SZ') for s in FUNDS]
    tables = ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')
    counts = {name: len(store.load_state(state, name, filters=[('instrument', 'in', canonical)])) for name in tables}
    return {'snapshot': SNAPSHOT, 'funds': list(FUNDS), 'dynamic_index_map': PAIRS, 'registered_fund_rows': counts,
        'reused_inputs': read(directory / 'reused-inputs.json'), 'original_trades_admitted': 0, 'not_a_backtest': True,
        'gates': ['150019/150051 historical classified-fund identity/conversion/termination/events and execution rules unproved',
            'ETF/fallback historical adjustment, events, states and fund fee versions incomplete',
            'Original11:30/14:40/14:53 prices and callback/instant-fill semantics unproved',
            'Original52 sort_index(by=power) and label[-1] compatibility unresolved; no silent economic substitution',
            'Fixed ex-post pool and supplier historical vintage remain limitations']}


def preflight(root, directory):
    bound = binding(root); prior = read(previous.DIRECTORY / 'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in prior['implementation_sha256'].items()), 'Accepted implementation changed before launch')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    reused = reused_inputs(); checkpoint(root, directory, 'Batch76 dynamic-emotion/two-eight launch baseline; classified-fund inputs checked first')
    save(directory / 'input-binding.json', bound); save(directory / 'reused-inputs.json', reused)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    save(directory / 'dependency-inventory.json', inventory(root, directory)); save(directory / 'existing-apis.json', api_evidence())
    return {'status': 'ok', 'reused_inputs': len(reused['inputs']), 'new_request_cases': len(QUERIES)}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码') / name; copied = folder / (strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    refs = []; (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository) / name; copied = directory / 'references' / source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Existing native fund/index parser and moving average reference; single existing trading ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder / 'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch76 two complete dynamic-emotion/two-eight sources frozen; original trade gates retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(SOURCES) and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[number]) and row['review'] == REVIEWS[SOURCES[number]] and
            row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, names, namespace):
    source = directory / 'source-reviews' / (strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    nodes = [n for n in ast.parse(read_source(source)[0]).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names) and all(safe_function(n) for n in nodes), 'Unsafe/missing original definition')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch76-definitions>', 'exec'), namespace)
    return namespace


def diagnostics(directory):
    validate_sources(directory); rows = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, error, fn):
        try: fn()
        except error as exc: record(case, exception=type(exc).__name__, detail=str(exc)[:200])
        else: raise AssertionError('Expected original error: '+case)
    log = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    ns = selected(directory, 0, ['initialize', 'set_params', 'set_variables', 'set_backtest', 'set_slip_fee',
        'ETFtrade1', 'ETFtrade2', 'get_signal', 'EmotionMonitor'], {'g': SimpleNamespace(), 'pd': pd, 'talib': talib, 'log': log})
    schedules = []; costs = []; slips = []
    ns.update(set_benchmark=lambda *a: None, set_option=lambda *a: None,
        get_security_info=lambda s: SimpleNamespace(code=s, display_name=s, start_date='synthetic'),
        run_daily=lambda fn, **kw: schedules.append([fn.__name__, kw]), set_slippage=lambda v: slips.append(v),
        FixedSlippage=lambda v: v, set_commission=lambda c: costs.append(c), PerTrade=lambda **kw: kw)
    ns['initialize'](None); ns['set_slip_fee'](None); g = ns['g']
    record('28_actual_pool_schedules_cost', pool=g.ETFList.copy(), schedules=schedules, costs=costs, slips=slips, lags=[g.lag, g.lag0, g.lag1, g.lag2])
    original_emotion = ns['EmotionMonitor']; ns['EmotionMonitor'] = lambda c: 0
    ns['attribute_history'] = lambda *a, **kw: {'close': np.full(13, 100.)}
    prices = dict(zip(PAIRS.values(), [104., 103., 102., 101., 100., 99., 98., 97.], strict=True))
    ns['get_current_data'] = lambda: {s: SimpleNamespace(last_price=v) for s, v in prices.items()}
    signal = ns['get_signal'](None)
    record('28_dynamic_third_and_actual_two', signal=signal, buy=g.buy.copy(), reference=g.target_market)
    ns['get_signal'](None); record('28_repeated_signal_accumulates_buy', buy=g.buy.copy())
    for case, levels, held, emotion in [('one_eligible', [101.,100.,100.,100.,100.,100.,100.,100.], [], 0),
            ('cash_KEEP', [100.]*8, [], 0), ('holding_sell', [100.]*8, ['OLD'], 0),
            ('emotion_sell', [104.,103.,102.,101.,100.,99.,98.,97.], ['OLD'], -1)]:
        prices.update(zip(PAIRS.values(), levels, strict=True)); g.buy=[]; g.last=held.copy(); ns['EmotionMonitor']=lambda c,e=emotion:e
        result=ns['get_signal'](None); record('28_'+case, signal=result, buy=g.buy.copy(), last=g.last.copy())
    ns['EmotionMonitor'] = original_emotion
    for count in (2,3,5,6):
        volume=np.full(100,100.); volume[-count:]=200. if count<4 else 50.
        if count<4:volume[-count-1]=50.
        ns['attribute_history']=lambda *a,v=volume,**kw:pd.DataFrame({'volume':v})
        record('28_emotion_run_'+str(count), signal=ns['EmotionMonitor'](None), rate=g.emotion_rate)
    ns['attribute_history']=lambda *a,**kw:pd.DataFrame({'volume':np.ones(100)})
    record('28_emotion_no_cross_returns_None', signal=ns['EmotionMonitor'](None), rate=g.emotion_rate)
    ns['attribute_history']=lambda *a,**kw:pd.DataFrame({'volume':[100.]*97+[200.]*3})
    record('28_positive_run_without_negative_returns_None', signal=ns['EmotionMonitor'](None))
    orders=[]
    ctx=SimpleNamespace(portfolio=SimpleNamespace(positions={}, total_value=10000., available_cash=1234., cash=1234., returns=0.))
    ns['order_target_value']=lambda s,v:orders.append(['target',s,v]); ns['order_value']=lambda s,v:orders.append(['value',s,v])
    g.signal='KEEP'; ns['ETFtrade2'](ctx); record('28_empty_KEEP_buys_fallback', orders=orders.copy()); orders.clear()
    stocks=list(PAIRS.values())[:2]
    for value in (7500.,7501.):
        ctx.portfolio.positions={s:SimpleNamespace(security=s,value=value) for s in stocks}; g.signal='BUY'; g.buy=stocks.copy(); g.clear=[]
        ns['ETFtrade2'](ctx); record('28_trim_'+str(value), orders=orders.copy()); orders.clear()
    ctx.portfolio.positions={}; g.signal='BUY'; g.buy=stocks.copy(); g.clear=[]
    ns['ETFtrade2'](ctx); record('28_incremental_targets_ignore_available_cash', orders=orders.copy(), last=g.last.copy()); orders.clear()
    g.signal='BUY'; g.buy=[]; failure('28_empty_BUY_division', ZeroDivisionError, lambda:ns['ETFtrade2'](ctx))
    ctx.portfolio.positions={s:SimpleNamespace(security=s,value=1.) for s in stocks}; g.signal='sell_the_stocks'
    ns['order_target_value']=lambda s,v:ctx.portfolio.positions.pop(s)
    failure('28_immediate_delete_during_sell', RuntimeError, lambda:ns['ETFtrade2'](ctx))

    ns2=selected(directory,1,['initialize','set_parms','before_trading_start','market_trade','get_signal','get_power'],
        {'g':SimpleNamespace(),'pd':pd,'np':np,'datetime':dt,'log':log})
    schedules=[]; costs=[]; slips=[]
    ns2.update(set_benchmark=lambda *a:None,set_option=lambda *a:None,
        set_order_cost=lambda v,**kw:costs.append([v,kw]),OrderCost=lambda **kw:kw,
        run_daily=lambda fn,**kw:schedules.append([fn.__name__,kw]),set_slippage=lambda v:slips.append(v),
        FixedSlippage=lambda v:v,set_commission=lambda v:costs.append(v),PerTrade=lambda **kw:kw,
        get_security_info=lambda s:SimpleNamespace(display_name=s))
    ns2['initialize'](None); g2=ns2['g']
    record('52_actual_parameters', indices=g2.index_list, funds=g2.etf_list, fallback=g2.bond, cycle=g2.cycle, schedules=schedules, initialize_cost=costs.copy())
    for date in ('2009-01-01','2009-01-02','2011-01-01','2011-01-02','2013-01-01','2013-01-02'):
        costs.clear(); ns2['before_trading_start'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(date)))
        record('52_cost_'+date, cost=costs.copy())
    original_power=ns2['get_power']; ns2['attribute_history']=lambda *a,**kw:pd.DataFrame({'close':np.arange(20)+100.})
    record('52_power_20_endpoints', power=original_power('INDEX',20))
    power=dict(zip(g2.index_list,[-.1,0.],strict=True)); ns2['get_power']=lambda s,*a:power[s]
    ctx2=SimpleNamespace(portfolio=SimpleNamespace(positions={},cash=1234.))
    record('52_both_nonpositive_fallback', signal=ns2['get_signal'](ctx2))
    ctx2.portfolio.positions={g2.etf_list[0]:SimpleNamespace()}; power[g2.index_list[1]]=.2
    record('52_negative_held_clear_first',signal=ns2['get_signal'](ctx2))
    power[g2.index_list[0]]=.1
    failure('52_positive_held_sort_API_error',TypeError,lambda:ns2['get_signal'](ctx2))
    ctx2.portfolio.positions={}; failure('52_positive_cash_sort_API_error',TypeError,lambda:ns2['get_signal'](ctx2))
    positional=pd.Series([1.,2.],index=['A','B']); failure('52_label_minus_one_error',KeyError,lambda:positional[-1])
    ns2['order_target_value']=lambda s,v:orders.append(['target',s,v]); ns2['order_value']=lambda s,v:orders.append(['value',s,v])
    ctx2.portfolio.positions={s:SimpleNamespace() for s in g2.etf_list}; ns2['get_signal']=lambda c:'clear'
    ns2['market_trade'](ctx2); record('52_clear_no_same_day_winner',orders=orders.copy()); orders.clear()
    ctx2.portfolio.positions={g2.bond:SimpleNamespace()}; ns2['get_signal']=lambda c:g2.etf_list[1]
    ns2['market_trade'](ctx2); record('52_sell_then_incremental_existing_cash',orders=orders.copy()); orders.clear()
    ns2['market_trade'](ctx2); record('52_repeated_incremental_buy',orders=orders.copy()); orders.clear()
    ctx2.portfolio.positions={s:SimpleNamespace() for s in g2.etf_list}; ns2['get_signal']=lambda c:'clear'
    ns2['order_target_value']=lambda s,v:ctx2.portfolio.positions.pop(s)
    failure('52_immediate_delete_during_clear',RuntimeError,lambda:ns2['market_trade'](ctx2))
    return {'cases':rows,'not_a_backtest':True,'original_formulas_unchanged':True}


def worker(directory, case):
    with patch.object(native,'QUERIES',QUERIES): return native.worker(directory / 'probes',case)


def probe(root,directory):
    binding(root,directory); folder=directory/'probes'; folder.mkdir(exist_ok=False)
    with patch.object(native,'QUERIES',QUERIES): save(folder/'existing-apis.json',native.api_evidence())
    rows=[]
    for case in QUERIES:
        command=[sys.executable,'-m','scripts.review_strategy_batch76','worker','--directory',str(directory),'--case',case]
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=45)
            save(directory/('worker-'+case+'.json'),{'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode==0,'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/('worker-'+case+'.json'),{'command':command,'status':'timeout','seconds':45})
            path=folder/case; path.mkdir(exist_ok=True)
            if not (path/'result.json').exists():save(path/'result.json',{'case':case,'query':QUERIES[case],'status':'timeout',
                'attempts':[],'wire_responses':[],'error':'45s deadline; completion unproved',
                'partial_files':{str(p):file_sha(p) for p in path.iterdir() if p.is_file()},
                'api_sha256':file_sha(folder/'existing-apis.json'),'published':False,'strict_usable':False})
        rows.append(read(folder/case/'result.json')); print(case+': '+rows[-1]['status'],file=sys.stderr,flush=True)
    result={'results':rows,'new_original_input_rows':0,'new_trading_backtests':0}
    save(folder/'results.json',result);save(directory/'probe-results.json',result)
    return archive(root,'Batch76 bounded missing classified-fund/ETF/index candidate attempts archived')


def candidate_audit(directory):
    validate_sources(directory);bound=read(directory/'input-binding.json')
    require(file_sha(bound['receipt'])==bound['receipt_sha256'] and file_sha(bound['review'])==bound['review_sha256'], 'Candidate parent binding changed')
    require(api_evidence()==read(directory/'existing-apis.json'), 'Installed API/TA-Lib changed')
    require(read(directory/'probe-results.json')==read(directory/'probes/results.json'),'Probe summary changed')
    folder=directory/'probes';results=read(folder/'results.json')['results'];profiles=[]
    require(len(results)==len(QUERIES) and {r['case'] for r in results}==set(QUERIES),'Candidate scope changed')
    with patch.object(native,'QUERIES',QUERIES):
        require(native.api_evidence()==read(folder/'existing-apis.json'),'Native parser API changed')
        for row in results:
            case=row['case'];require(row['query']==QUERIES[case] and row['api_sha256']==file_sha(folder/'existing-apis.json'),'Candidate query binding changed')
            for wire in row['wire_responses']:require(file_sha(wire['file'])==wire['sha256'],'Raw response changed')
            for name,sha in row.get('partial_files',{}).items():require(file_sha(name)==sha,'Partial response changed')
            profile={'case':case,'status':row['status'],'strict_usable':False}
            if row['status'] in ('success','empty'):
                require(file_sha(row['file'])==row['sha256'] and len(row['attempts'])==len(row['wire_responses'])==1,'Parsed candidate binding changed')
                calls=[]
                def replay(session,method,url,**kwargs):
                    endpoint,params=native.expected_request(case)
                    require(not calls and method.upper()=='GET' and url==endpoint and kwargs.get('params')==params,'Unexpected native replay')
                    calls.append(True);wire=row['wire_responses'][0];response=requests.Response()
                    response.status_code=wire['status_code'];response._content=Path(wire['file']).read_bytes();response.encoding=wire['encoding'];response.url=url
                    response.raise_for_status();return response
                with patch.object(requests.sessions.Session,'request',replay),redirect_stdout(StringIO()):frame=native.invoke(case)
                pd.testing.assert_frame_equal(frame,pd.read_parquet(row['file']),check_exact=True)
                require(len(calls)==1 and len(frame)==row['rows'] and list(frame)==row['columns'],'Native parse scope differs')
                if len(frame):
                    dates=pd.to_datetime(frame.date);profile.update(rows=len(frame),first=str(dates.min().date()),last=str(dates.max().date()),
                        post_cutoff_rows=int((dates>pd.Timestamp('2026-09-29')).sum()),unique_dates=dates.nunique())
                profile['native_parser_match']=True
            else:require(row.get('error'),'Failed candidate lacks reason');profile['error']=row['error']
            profiles.append(profile)
        return {'profiles':profiles,'original_input_rows_admitted':0,'not_a_backtest':True,'limits':native.api_evidence()['limits']}


def audit(root,directory):
    binding(root,directory);save(directory/'candidate-audit.json',candidate_audit(directory))
    return archive(root,'Batch76 native candidate parse replay and coverage audited; original identities/events remain unproved')


def inputs(directory):
    result=read(directory/'reused-inputs.json')['inputs'].copy()
    for row in read(directory/'probe-results.json')['results']:
        if row['status']=='success':result[row['case'].replace('index-sh','index-').replace('index-sz','index-')]=row
    frames={};profiles=[]
    for case,row in result.items():
        require(file_sha(row['file'])==row['sha256'],'Operand changed');frame=pd.read_parquet(row['file']).copy()
        dates=pd.to_datetime(frame.date);require(dates.notna().all(),'Invalid supplier dates')
        profiles.append({'case':case,'file':row['file'],'sha256':row['sha256'],'rows':len(frame),'post_cutoff_rows':int((dates>pd.Timestamp('2026-09-29')).sum())})
        frame=frame.loc[dates<=pd.Timestamp('2026-09-29')].copy();frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        frame=frame.sort_values('date').reset_index(drop=True);require(not frame.date.duplicated().any(),'Duplicate operand date')
        for name in ('close','volume'):
            if name in frame:frame[name]=frame[name].to_numpy(float)
        frames[case]=frame
    return frames,profiles


def independent_emotion(volume):
    ratios=[v/(math.fsum(volume[i-6:i+1])/7)-1 for i,v in enumerate(volume) if i>=6]
    for i in range(30):
        if ratios[-1]>=0:
            if ratios[-1-i]<0:return 1 if i>=3 else 0
        elif ratios[-1-i]>=0:return -1 if i>=6 else 0
    return None


def components(directory):
    frames,profiles=inputs(directory);funds=[];indices=[];paired=[];blocked=[];max_mean=0.;max_power=0.;emotion_differences=[]
    ns=selected(directory,0,['EmotionMonitor'],{'g':SimpleNamespace(target_market='REFERENCE',lag0=7,lag=6),'talib':talib})
    power=selected(directory,1,['get_power'],{})
    powers={}
    for case,frame in frames.items():
        if case.startswith('fund-'):
            for k in range(12,len(frame)):
                close=frame.close.iloc[k-12:k+1].to_numpy(float)
                if not np.isfinite(close).all() or (close<=0).any():blocked.append([case,str(frame.date.iloc[k]),'invalid13close']);continue
                mean=float(close.mean());other=math.fsum(close)/13;max_mean=max(max_mean,abs(mean-other))
                funds.append({'case':case,'history_end':str(frame.date.iloc[k]),'first_close':float(close[0]),'mean13':mean,
                    'independent_mean13':other,'intraday_current_price_available':False})
            continue
        for k in range(19,len(frame)):
            block=frame.iloc[k-19:k+1].copy();close=block.close.to_numpy(float)
            if not np.isfinite(close).all() or (close<=0).any():blocked.append([case,str(frame.date.iloc[k]),'invalid20close']);continue
            power['attribute_history']=lambda *a,b=block,**kw:b.copy()
            actual=float(power['get_power'](case,20));other=float(Fraction.from_float(float(close[-1]))/Fraction.from_float(float(close[0]))-1)
            max_power=max(max_power,abs(actual-other));row={'case':case,'history_end':str(frame.date.iloc[k]),'power20':actual,
                'independent_power20':other,'emotion100_available':False,'emotion':None,'emotion_rate':None}
            powers.setdefault(case,{})[str(frame.date.iloc[k])]=(actual,block.date.tolist())
            if k>=99 and case in {'index-'+s[:6] for s in PAIRS}:
                volume=frame.volume.iloc[k-99:k+1].to_numpy(float)
                if np.isfinite(volume).all() and (volume>0).all():
                    ns['attribute_history']=lambda *a,v=volume,**kw:pd.DataFrame({'volume':v})
                    signal=ns['EmotionMonitor'](None);expected=independent_emotion(volume)
                    row.update(emotion100_available=True,emotion=signal,emotion_rate=ns['g'].emotion_rate)
                    if signal!=expected:emotion_differences.append({'case':case,'history_end':row['history_end'],'original':signal,'independent':expected})
                else:blocked.append([case,str(frame.date.iloc[k]),'invalid100volume'])
            indices.append(row)
    p300=powers.get('index-000300',{});p905=powers.get('index-000905',{})
    for date in sorted(set(p300)&set(p905)):
        a,dates_a=p300[date];b,dates_b=p905[date]
        if dates_a!=dates_b:blocked.append(['pair',date,'20date_mismatch']);continue
        paired.append({'history_end':date,'power300':a,'power905':b,'both_nonpositive':a<=0 and b<=0,
            'original_sort_API_compatible':False})
    fund_frame=pd.DataFrame(funds,columns=['case','history_end','first_close','mean13','independent_mean13','intraday_current_price_available'])
    index_frame=pd.DataFrame(indices,columns=['case','history_end','power20','independent_power20','emotion100_available','emotion','emotion_rate'])
    pair_frame=pd.DataFrame(paired,columns=['history_end','power300','power905','both_nonpositive','original_sort_API_compatible'])
    expected={'fund-'+s[:6] for s in FUNDS}|{'index-'+s[:6] for s in PAIRS}|{'index-000300'}
    summary={'inputs':profiles,'missing_inputs':sorted(expected-set(frames)),'fund13_windows':len(funds),'index20_windows':len(indices),
        'emotion100_windows':sum(r['emotion100_available'] for r in indices),'paired20_windows':len(paired),
        'paired_first_history_end':paired[0]['history_end'] if paired else None,'paired_last_history_end':paired[-1]['history_end'] if paired else None,
        'blocked_windows':len(blocked),'blocked':blocked,'max_mean13_abs_difference':max_mean,'max_power20_abs_difference':max_power,
        'emotion_signal_differences':emotion_differences,'not_a_backtest':True,'new_trading_backtests':0,
        'limits':['Supplier historical rows are reference operands, not proven original skip-paused/vintage/adjustment or identity',
            'No daily close substituted for11:30 current price; fund first/mean only, no invented28 ranking or trading signal',
            'Emotion per-index100-row original TA-Lib reinitialization; dynamic third index needs original full-pool intraday ranking',
            'Paired powers require same20 native dates; no original52 compatibility repair, stateful trades or returns',
            'Original strict comparisons retained; any independent emotion sign differences archived rather than repaired']}
    return (fund_frame,index_frame,pair_frame),summary


def study(root,directory):
    require(all(not (directory/n).exists() for n in ('diagnostics.json','fund-history-components.parquet','index-components.parquet','paired-power-components.parquet','component-research.json')),'Study archive exists')
    binding(root,directory);require(api_evidence()==read(directory/'existing-apis.json'),'Installed API/TA-Lib changed')
    before=implementation();diag=diagnostics(directory);frames,summary=components(directory)
    require(before==implementation(),'Implementation changed during study')
    for name,frame in zip(('fund-history-components.parquet','index-components.parquet','paired-power-components.parquet'),frames,strict=True):
        with (directory/name).open('xb') as stream:frame.to_parquet(stream,index=False)
    save(directory/'diagnostics.json',diag);save(directory/'component-research.json',{**summary,'implementation_sha256':before})
    return archive(root,'Batch76 original diagnostics and longest verified supplier history operands archived; no intraday substitute')


def offline(root,directory):
    require(not (directory/'offline-verification.json').exists(),'Offline archive exists');binding(root,directory)
    require(api_evidence()==read(directory/'existing-apis.json'),'Installed API/TA-Lib changed');before=implementation()
    def denied(*a,**kw):raise ValueError('Batch76 attempted network')
    with patch.object(socket,'create_connection',denied),patch.object(socket.socket,'connect',denied),patch.object(requests.sessions.Session,'request',denied),redirect_stdout(StringIO()):
        diag=diagnostics(directory);frames,summary=components(directory);audit=candidate_audit(directory);dep=inventory(root,directory);cat=offline_catalog(root,directory)
    for name,frame in zip(('fund-history-components.parquet','index-components.parquet','paired-power-components.parquet'),frames,strict=True):
        pd.testing.assert_frame_equal(frame,pd.read_parquet(directory/name),check_exact=True)
    saved=read(directory/'component-research.json');require(saved=={**summary,'implementation_sha256':before} and before==implementation() and
        diag==read(directory/'diagnostics.json') and audit==read(directory/'candidate-audit.json') and dep==read(directory/'dependency-inventory.json'),'Offline evidence differs')
    save(directory/'offline-catalog.json',cat);validate_catalog(root,directory)
    save(directory/'offline-verification.json',{'result':'match','differences':0,'socket_network_disabled':True,'implementation_sha256':before,'not_a_backtest':True})
    return archive(root,'Batch76 forbidden-network diagnostics/reference operands/native parsers and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests','scripts/review_strategy_batch76.py'],
        [sys.executable,'-m','scripts.verify_offline_tests','-q','tests/unit/test_batch76_research.py','tests/unit/test_batch75_research.py',
            'tests/unit/test_batch75_recovery.py','tests/unit/test_external_cash_flows.py','tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py','tests/unit/test_research_metadata.py','tests/integration/test_strategy_merge.py'],['git','diff','--check']]


def checks(root,directory):
    binding(root,directory);validate_sources(directory);validate_catalog(root,directory);before=implementation();rows=[]
    require(not (directory/'checked-state.json').exists(),'Checked archive exists');folder=directory/'checks';folder.mkdir(exist_ok=False)
    for number,command in enumerate(commands()):
        path=folder/f'check-{number}.log'
        with path.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        rows.append({'command':command,'returncode':result.returncode,'log':str(path),'sha256':file_sha(path)});require(result.returncode==0,f'Check failed: {path}')
    require(before==implementation(),'Checked implementation changed')
    for name in FILES:
        copied=directory/'implementation-checked'/name;copied.parent.mkdir(parents=True,exist_ok=True);require(not copied.exists(),'Checked copy exists');shutil.copyfile(name,copied)
    save(directory/'checked-state.json',{'status':'passed','commands':rows,'implementation_sha256':before,
        'evidence_sha256':{p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status':'passed'}


def finish(root,directory):
    binding(root,directory);validate_sources(directory);validate_catalog(root,directory)
    checked=read(directory/'checked-state.json');off=read(directory/'offline-verification.json');current=implementation()
    require(checked['status']=='passed' and checked['implementation_sha256']==current==off['implementation_sha256'] and CORE<=checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']]==commands() and all(r['returncode']==0 and file_sha(r['log'])==r['sha256'] for r in checked['commands']),'Checks changed')
    for name,sha in checked['evidence_sha256'].items():require(file_sha(directory/name)==sha,'Evidence changed')
    for name,sha in current.items():require(file_sha(directory/'implementation-checked'/name)==sha,'Checked copy changed')
    require(off['result']=='match' and off['differences']==0 and off['socket_network_disabled'] and Store(root).published()==read(directory/'baseline.json')['published'],'Offline/publication changed')
    protection=protect(root,directory);progress=archive(root,'Batch76 dynamic-emotion/two-eight source and long reference research accepted; original trades blocked')
    receipt=Path('docs/handoff/2026-10-09-batch76-verification.json')
    save(receipt,{'status':'ok','snapshot':SNAPSHOT,'reviews':read(directory/'source-reviews/review.json'),'checks':checked,'offline':off,
        'diagnostics':read(directory/'diagnostics.json'),'components':read(directory/'component-research.json'),
        'inventory':read(directory/'dependency-inventory.json'),'candidates':read(directory/'candidate-audit.json'),'protection':protection,'progress':progress,
        'newly_reviewed':2,'not_a_backtest':True,'new_trading_backtests':0,'original_strategy_complete':False})
    save(directory/'final-binding-verification.json',{'status':'ok','receipt_sha256':file_sha(receipt),'checked_state_sha256':file_sha(directory/'checked-state.json'),
        'final_script_sha256':file_sha(__file__),'catalog_sha256':file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status':'ok','checkpoint':progress['checkpoint'],'manually_reviewed':progress['manually_reviewed']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','start','worker','probe','audit','study','offline','checks','finish'])
    parser.add_argument('--root',default='data');parser.add_argument('--directory',type=Path,default=DIRECTORY);parser.add_argument('--case',choices=QUERIES)
    args=parser.parse_args();result=worker(args.directory,args.case) if args.action=='worker' else globals()[args.action](args.root,args.directory)
    print(json.dumps(result,ensure_ascii=False,default=str))
