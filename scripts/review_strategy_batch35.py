"""Archive BBI/damped RSRS/V2.1 sources and verify explicitly limited components."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
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

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts import review_strategy_batch34 as previous
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch31 import reference_ols, reference_z
from scripts.review_strategy_batch32 import source_tree
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch34/20261006-momentum-rsrs-bias')
RECEIPT = Path('docs/handoff/2026-10-06-batch34-verification.json')
SOURCES = ('2023年度精选策略/68.宽基BBI动量追涨，完美避开这波大跌~.txt',
    '2024年度精选策略2/24.宽基ETF动量轮动钝化RSRS择时-回撤小.txt',
    '2024年度精选策略2/50.ETF动量轮动RSRS择时-V2.1.txt')
REFERENCES = previous.REFERENCES
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch35.py', 'tests/unit/test_batch35_research.py'}))
CORE = previous.CORE
QUERIES = {
    'index1000': {'provider': 'baostock', 'function': 'query_history_k_data_plus',
        'parameters': {'code': 'sh.000852', 'fields': 'date,code,open,high,low,close,volume,amount',
            'start_date': '2005-01-01', 'end_date': '2026-09-29', 'frequency': 'd', 'adjustflag': '3'},
        'limit': 'Daily index sample cannot supply BBI30m/current prices or fund events/states'},
    'fund1000': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh512100'},
        'limit': 'Price sample does not prove original adjustment/events/states/rules or intraday execution'},
    'fundconsumer': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz159928'},
        'limit': 'Price sample does not supply MA90 ranking pool or60m/13:00 fills/events/states'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch34 differs')
    require(file_sha(ACCEPTED/'input-binding.json') == receipt['checks']['evidence_sha256']['input-binding.json'], 'Accepted binding changed')
    upstream = previous.binding(root, ACCEPTED); rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED/name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': upstream, 'files': rows, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import inspect
    import akshare as ak
    import baostock as bs
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(bs if query['provider'] == 'baostock' else ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    checkpoint(root, directory, 'Batch35 BBI/damped RSRS/V2.1 source research started')
    save(directory/'input-binding.json', binding(root)); save(directory/'existing-apis.json', api_evidence())
    old_path = Path(read(Path(root)/'catalog/strategies/latest.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory/'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码')/name; copied = folder/f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path), 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory/'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository)/name; copied = folder/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only OLS/SMA comparison; no platform replacement'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory/'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqlib', 'jqfactor')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch35 three full sources frozen; wrong targets and original index orders retained')


def validate_sources(directory):
    doc = accepted_review(directory)
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码')/name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def accepted_review(directory):
    correction=directory/'review-correction.json'
    if not correction.exists(): return read(directory/'source-reviews/review.json')
    proof=read(correction); original=directory/'source-reviews/review.json'; corrected=directory/'source-reviews/review-corrected.json'
    require(proof['original_sha256']==file_sha(original) and proof['corrected_sha256']==file_sha(corrected),'Review correction binding changed')
    before=read(original); after=read(corrected)
    require(len(before['sources'])==len(after['sources'])==3,'Correction scope differs')
    for k,(a,b) in enumerate(zip(before['sources'],after['sources'],strict=True)):
        require({key:value for key,value in a.items() if key!='review'}=={key:value for key,value in b.items() if key!='review'},'Correction changed source')
        require(b['review']==REVIEWS[SOURCES[k]],'Corrected review differs')
        aa=dict(a['review']); bb=dict(b['review']); aa['rules']=dict(aa['rules']); bb['rules']=dict(bb['rules'])
        if k==2:
            require(aa['rules']['seed']==proof['before'] and bb['rules']['seed']==proof['after'],'Correction seed evidence differs')
            aa['rules'].pop('seed'); bb['rules'].pop('seed')
        require(aa==bb,'Correction changed other rules')
    require({key:value for key,value in before.items() if key not in ('sources','catalog')}==
        {key:value for key,value in after.items() if key not in ('sources','catalog')},'Correction changed review metadata')
    return after


def correct(root,directory):
    binding(root,directory); doc=read(directory/'source-reviews/review.json')
    original=doc['sources'][2]['review']['rules']['seed']; corrected=REVIEWS[SOURCES[2]]['rules']['seed']
    require(original!=corrected,'No review correction required')
    for name,row in zip(SOURCES,doc['sources'],strict=True): row['review']=REVIEWS[name]
    doc['catalog']=catalog_strategies(root,'repo/量化策略源代码'); save(directory/'source-reviews/review-corrected.json',doc)
    save(directory/'review-correction.json',{'original_sha256':file_sha(directory/'source-reviews/review.json'),
        'corrected_sha256':file_sha(directory/'source-reviews/review-corrected.json'),'before':original,'after':corrected,
        'arithmetic':'608 slopes; r2s[i-8] resolves index600+i; z window ends599+i; offset1, not9',
        'source_bytes_changed':False,'formula_changed':False})
    validate_sources(directory); return archive(root,'Batch35 seed offset description corrected1; original mistaken review retained')


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory/name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path)
        profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns),
            'first': frame.date.min(), 'last': frame.date.max()}
    save(directory/'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['Original HS300 timing reference only; no fund pool replacement or intraday/BBI validation',
            'Damped RSRS13 sampled complete windows only; V2.1 longest continuous daily component']})
    return archive(root, 'Batch35 accepted HS300/calendar input bytes frozen')


def inputs(directory):
    frame, sessions = previous.inputs(directory)
    require(np.isfinite(frame[['high','low','close','volume']]).all().all() and
        frame.low.gt(0).all() and frame.high.ge(frame.low).all(), 'Invalid OHLC component input')
    dates = [d for d in sessions if frame.date.min() <= d <= frame.date.max()]
    require(frame.date.tolist() == dates, 'Benchmark has missing verified session')
    return frame, sessions


def timing_namespace(directory, number, frame, end):
    holder = {'end': end}; observed = []; log = SimpleNamespace(info=lambda *a: None)
    g = SimpleNamespace(ref_stock='000300.XSHG', N=18, M=600, K=8, score_threshold=.7)
    if number == 1: g = SimpleNamespace(N={'000300.XSHG':18}, M={'000300.XSHG':700}, score_threshold={'000300.XSHG':.7})
    def history(stock, count, unit, fields):
        require(stock == '000300.XSHG' and unit == '1d' and count in (18,626,718), 'Original request differs')
        require(count-1 <= holder['end'] < len(frame), 'Insufficient component window')
        result = frame.iloc[holder['end']-count+1:holder['end']+1][fields].reset_index(drop=True).copy()
        if number == 1: result.index = range(-count, 0)
        return result
    ns = {'g':g, 'np':np, 'attribute_history':history, 'log':log, 'record':lambda **kw: observed.append(kw)}
    names = ('get_ols','get_zscore','get_timing_signal')
    if number == 2: names += ('initial_slope_series','get_zscore_slope')
    sha = selected(directory, number, names, ns)
    if number == 1:
        original = ns['get_zscore']
        def zscore(values):
            value = original(values); observed.append({'zscore':float(value)}); return value
        ns['get_zscore'] = zscore
    return ns, holder, observed, sha


def configuration(directory,number):
    calls=[]; g=SimpleNamespace(); log=SimpleNamespace(info=lambda *a:None,set_level=lambda *a:None)
    ns={'g':g,'log':log,'set_benchmark':lambda value:calls.append(['benchmark',value]),
        'set_option':lambda *a:calls.append(['option',*a]),'set_slippage':lambda value:calls.append(['slippage',value]),
        'FixedSlippage':lambda value:value,'OrderCost':lambda **kw:kw,
        'set_order_cost':lambda value,**kw:calls.append(['cost',value,kw]),
        'run_daily':lambda fn,**kw:calls.append(['schedule',fn,kw]),'initial_slope_series':lambda:([0.]*608,[0.]*8)}
    for name in ('make_sure_etf_ipo','market_buy','calculate','market_open','my_trade','check_lose','hold_check'): ns[name]=name
    selected(directory,number,('initialize',),ns); ns['initialize'](None)
    return g,calls


def reference_damped(frame):
    require(len(frame) == 718 and np.isfinite(frame[['low','high','close']]).all().all() and frame.close.gt(0).all(), 'Invalid damped window')
    close = list(map(float, frame.close)); returns = [close[k]/close[k-1]-1 for k in range(1, len(close))]
    stds = []
    for end in range(18, 718):
        values = returns[end-18:end]; mean = math.fsum(values)/18
        stds.append(math.sqrt(math.fsum((v-mean)**2 for v in values)/17))
    # pandas rank(method='average', pct=True) gives half the tied group including self.
    last = stds[-1]; quantile = (sum(v < last for v in stds)+(sum(v == last for v in stds)+1)/2)/700
    slopes = [reference_ols(frame.low.iloc[k:k+18], frame.high.iloc[k:k+18])[1] for k in range(701)]
    _, _, r2 = reference_ols(frame.low.iloc[-18:], frame.high.iloc[-18:])
    z = reference_z(slopes[-700:]); score = z*r2**(2*quantile)
    return {'quantile':quantile, 'zscore':z, 'r2':r2, 'score':score,
        'signal':'BUY' if score > .7 else 'SELL' if score < -.7 else 'KEEP'}


def v21_signal(score, derivative):
    return 'SELL' if derivative < 0 and score > 0 else 'BUY' if score > -.7 else 'SELL'


def bbi_order_fragment(directory):
    fn = next(n for n in source_tree(directory,0).body if isinstance(n,ast.FunctionDef) and n.name=='market_buy')
    start = next(k for k,n in enumerate(fn.body) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='holdings' for t in n.targets))
    fragment=ast.FunctionDef(name='orders',args=fn.args,body=fn.body[start:],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fragment],type_ignores=[]))
    return compile(module,'<original-bbi-post-ranking-orders>','exec'),hashlib.sha256(ast.dump(module).encode()).hexdigest()


def compute(directory):
    validate_sources(directory); frame, _ = inputs(directory)
    ns, holder, observed, sha = timing_namespace(directory, 2, frame, 625)
    ns['g'].slope_series, ns['g'].rsrs_score_hisitory = ns['initial_slope_series']()
    initial = frame.iloc[:626]
    regressions = [reference_ols(initial.low.iloc[k:k+18], initial.high.iloc[k:k+18]) for k in range(608)]
    slopes = [r[1] for r in regressions]; scores = [reference_z(slopes[k:k+600])*regressions[k-8][2] for k in range(8)]
    require(len(ns['g'].slope_series) == 608 and len(ns['g'].rsrs_score_hisitory) == 8, 'Seed lengths differ')
    seed_error = max(abs(float(a)-b) for a,b in zip(ns['g'].slope_series, slopes, strict=True))
    seed_score_error = max(abs(float(a)-b) for a,b in zip(ns['g'].rsrs_score_hisitory, scores, strict=True))
    max_score = 0.; max_derivative = 0.; boundaries = []; counts = {}; digest = hashlib.sha256()
    for end in range(625, len(frame)):
        holder['end'] = end; signal = ns['get_timing_signal'](None, 'ignored')
        window = frame.iloc[end-17:end+1]; _, slope, r2 = reference_ols(window.low, window.high)
        slopes.append(slope); score = reference_z(slopes[-600:])*r2; scores.append(score)
        derivative = reference_ols(range(8), scores[-8:])[1]
        actual = observed[-1]; max_score = max(max_score, abs(float(actual['rsrs_score'])-score))
        max_derivative = max(max_derivative, abs(float(actual['rsrs_slope'])-derivative))
        expected = v21_signal(score, derivative)
        if signal != expected: boundaries.append({'date':frame.date.iloc[end], 'actual':signal, 'expected':expected})
        counts[signal] = counts.get(signal,0)+1
        digest.update(json.dumps({'date':frame.date.iloc[end], 'score':float(actual['rsrs_score']),
            'derivative':float(actual['rsrs_slope']), 'signal':signal}, sort_keys=True).encode())
    dns, dh, dobserved, dsha = timing_namespace(directory, 1, frame, 717); sampled = []
    for end in np.linspace(717, len(frame)-1, 13, dtype=int):
        dh['end'] = int(end); signal = dns['get_timing_signal'](None, '000300.XSHG')
        window = frame.iloc[end-717:end+1]; expected = reference_damped(window)
        ret_std = window.close.pct_change().rolling(18).std()
        quantile = float(ret_std.tail(700).rank(pct=True).iloc[-1])
        r2 = float(dns['get_ols'](window.low.iloc[-18:],window.high.iloc[-18:])[2])
        actual = dobserved[-1]['zscore']*r2**(2*quantile)
        require(abs(quantile-expected['quantile']) < 1e-12 and abs(actual-expected['score']) < 1e-9 and signal == expected['signal'], 'Damped component differs')
        sampled.append({'end_date':frame.date.iloc[end], 'quantile':quantile, 'score':actual, 'signal':signal,
            'reference':expected, 'difference':abs(actual-expected['score'])})
    require(max(seed_error, seed_score_error, max_score, max_derivative) < 1e-9, 'V2.1 component differs')
    require(not boundaries, 'V2.1 strict boundary differs')
    return {'snapshot':SNAPSHOT, 'not_a_backtest':True, 'original_strategy_complete':False, 'strategy_results':[],
        'backend':{'numpy':np.__version__, 'pandas':pd.__version__},
        'input_sha256':{n:file_sha(directory/n) for n in ('index-input.parquet','calendar-input.parquet')},
        'v21':{'source_ast_sha256':sha, 'windows':len(frame)-625, 'first_end_date':frame.date.iloc[625], 'last_end_date':frame.date.iloc[-1],
            'seed_slopes':608, 'seed_scores':8, 'seed_last_end_index':624, 'first_append_end_index':625,
            'seed_r2_end_offset':1, 'max_seed_difference':seed_error, 'max_seed_score_difference':seed_score_error,
            'max_score_difference':max_score, 'max_derivative_difference':max_derivative, 'boundaries':boundaries,
            'signals':counts, 'sha256':digest.hexdigest()},
        'damped':{'source_ast_sha256':dsha, 'sampled_windows':len(sampled), 'exhaustive':False, 'samples':sampled},
        'limits':['Completed HS300 daily end dates only; neither intraday execution dates nor fund ranking/backtest',
            'Original626/608/eight R2 misalignment preserved; no adjustment to source economics',
            'Damped original full loop retained; negative labels explicitly adapt Series[-1] legacy semantics',
            'BBI unavailable; no invented BBI kernel, trades, fees or NAV']}


def diagnostics(directory):
    cases = []; orders = []; log = SimpleNamespace(info=lambda *a:None)
    for number in range(3):
        g,calls=configuration(directory,number)
        cases.append({'case':f'configuration_{number}','parameters':vars(g),'calls':calls})
    ns = {'g':SimpleNamespace(check_out_list='000300.XSHG',timing_signal='BUY'), 'order_target_value':lambda *a:orders.append(['target',*a]),
        'order_value':lambda *a:orders.append(['value',*a]), 'get_timing_signal':lambda *a:'SELL'}
    selected(directory,1,('market_open',),ns)
    for positions in ({}, {'one':None,'two':None}):
        orders.clear(); ns['market_open'](SimpleNamespace(portfolio=SimpleNamespace(positions=positions, available_cash=1000.,positions_value=0)))
        cases.append({'case':'empty_index_order' if not positions else 'concatenated_holdings', 'orders':[list(x) for x in orders]})
    orders.clear(); ns = {'g':SimpleNamespace(max_value=100.,last_value=95.), 'log':log, 'order_target':lambda *a:orders.append(list(a))}
    selected(directory,1,('loss_ctrl',),ns)
    ns['loss_ctrl'](SimpleNamespace(portfolio=SimpleNamespace(total_value=80.,positions={'fund':None})))
    cases.append({'case':'duplicate_account_stop', 'orders':orders.copy(), 'max_value':ns['g'].max_value, 'last_value':ns['g'].last_value})
    dated = pd.DataFrame({'low':np.arange(718.)+100.,'high':np.arange(718.)+102.,'close':np.arange(718.)+101.}, index=pd.date_range('2020-01-01',periods=718))
    ns = {'g':SimpleNamespace(N={'x':18},M={'x':700},score_threshold={'x':.7}), 'np':np, 'attribute_history':lambda *a:dated.copy()}
    selected(directory,1,('get_ols','get_zscore','get_timing_signal'),ns)
    try: ns['get_timing_signal'](None,'x')
    except KeyError as exc: cases.append({'case':'damped_legacy_index_error', 'error':str(exc)})
    else: raise AssertionError('Legacy index fault disappeared')
    rank_ns = {'g':SimpleNamespace(stock_pool=['fund'],momentum_day=20), 'np':np,
        'attribute_history':lambda *a:pd.DataFrame({'close':100+np.arange(110.)})}
    selected(directory,2,('get_rank',),rank_ns)
    try: rank_ns['get_rank'](None,[])
    except KeyError as exc: cases.append({'case':'bias_first_label_error','error':str(exc)})
    else: raise AssertionError('Original bias first-label fault disappeared')
    orders = []; ns = {'order_target_value':lambda *a:orders.append(list(a))}; selected(directory,2,('check_lose',),ns)
    pos = SimpleNamespace(security='fund',avg_cost=100.,price=9.)
    try: ns['check_lose'](SimpleNamespace(portfolio=SimpleNamespace(positions={'fund':pos})))
    except NameError as exc: cases.append({'case':'stop_orders_before_undefined_value','orders':orders.copy(),'error':str(exc)})
    else: raise AssertionError('Original stop fault disappeared')
    class Positions(dict):
        def __missing__(self,key): return SimpleNamespace(total_amount=0)
    orders = []; ns = {'g':SimpleNamespace(stock_num=1), 'open_position':lambda *a:orders.append(list(a)) or False,
        'close_position':lambda *a:False}; selected(directory,2,('adjust_position',),ns)
    ns['adjust_position'](SimpleNamespace(portfolio=SimpleNamespace(positions=Positions(),cash=1000.)), ['fund',.001])
    cases.append({'case':'mixed_targets','orders':orders.copy()})
    orders=[]; messages=[]; bars=pd.DataFrame({'close':np.r_[np.ones(21)*100,90.]},index=range(-22,0))
    ns={'log':log,'attribute_history':lambda *a:bars.copy(), 'send_message':lambda *a:messages.append(list(a)), 'order_target_value':lambda *a:orders.append(list(a))}
    selected(directory,2,('hold_check',),ns)
    for amount in (0,100):
        orders.clear(); messages.clear(); ns['hold_check'](SimpleNamespace(portfolio=SimpleNamespace(positions={'fund':SimpleNamespace(total_amount=100,closeable_amount=amount)})))
        cases.append({'case':f'hold_stop_{amount}','orders':orders.copy(),'record_only_messages':messages.copy()})
    frame=pd.DataFrame({'low':np.arange(626.)+100, 'high':np.arange(626.)+np.sin(np.arange(626.)/3)+103, 'volume':np.ones(626)})
    ns,_,_,_=timing_namespace(directory,2,frame,625)
    ns['g'].slope_series=[1.,2.]*300; ns['g'].rsrs_score_hisitory=[1.]*8
    for score,derivative in [(0.,-1.),(.1,-1.),(-.7,0.),(-.699999,0.),(.7,0.)]:
        ns['get_zscore']=lambda *a,score=score:score; ns['get_ols']=lambda *a:(0.,1.,1.); ns['get_zscore_slope']=lambda derivative=derivative:derivative
        cases.append({'case':f'timing_{score}_{derivative}','signal':ns['get_timing_signal'](None,None)})
    orders=[]; bbi_ns={'pd':pd,'log':log,'BBI':lambda *a,**kw:{'index':1.},'get_bars':lambda *a,**kw:{'close':[1.]},
        'g':SimpleNamespace(unit='30m',available_indexs=['index'])}
    selected(directory,0,('market_buy',),bbi_ns)
    try: bbi_ns['market_buy'](SimpleNamespace(current_dt='diagnostic'))
    except AttributeError as exc: cases.append({'case':'bbi_removed_append','error':str(exc)})
    else: raise AssertionError('Original append fault disappeared')
    ipo=SimpleNamespace(not_ipo_list={'index':'fund'},available_indexs=[],lag1=5)
    ns={'g':ipo,'log':log,'get_before_after_trade_days':lambda *a:'2020-01-01',
        'get_all_securities':lambda **kw:pd.DataFrame({'start_date':['2019-01-01']},index=['fund' if kw['types']=='fund' else 'index'])}
    selected(directory,0,('make_sure_etf_ipo',),ns); ctx=SimpleNamespace(previous_date='2020-01-08')
    ns['make_sure_etf_ipo'](ctx); ns['get_all_securities']=lambda **kw:pd.DataFrame(); ns['make_sure_etf_ipo'](ctx)
    cases.append({'case':'ipo_once_permanent','available':ipo.available_indexs,'pending':ipo.not_ipo_list})
    code,sha=bbi_order_fragment(directory)
    for signal,positions in [('CLEAR',{'fundA':None,'fundB':None}),('BUY',{'fundB':None}),('BUY',{})]:
        orders=[]; ns={'g':SimpleNamespace(signal=signal,increase_days=0,decrease_days=0,ETF_list={'a':'fundA','b':'fundB'}),
            'log':log,'target':'a','target_02':'b','order_target':lambda *a:orders.append(['target',*a]),
            'order_value':lambda *a:orders.append(['value',*a])}
        exec(code,ns); ns['orders'](SimpleNamespace(portfolio=SimpleNamespace(positions=positions,available_cash=1000.)))
        # set order is not proven; retain only the unordered submitted sell set in this diagnostic.
        normalized=[['target','one_of_fundA_fundB',0]] if signal=='CLEAR' else orders
        require(signal!='CLEAR' or len(orders)==1 and orders[0][1] in positions,'Original CLEAR branch differs')
        cases.append({'case':f'bbi_orders_{signal}_{len(positions)}','orders':normalized,'fragment_ast_sha256':sha})
    return {'not_a_backtest':True,'cases':cases,'limits':['Record-only message stub never contacts external services',
        'Synthetic order intentions, no fills, cash evolution, costs or NAV; economic corrections remain unapproved']}


def study(root,directory):
    binding(root,directory); before=implementation(); result=compute(directory); diagnostic=diagnostics(directory)
    require(before == implementation(),'Study code changed'); save(directory/'component-research.json',result); save(directory/'diagnostics.json',diagnostic)
    return archive(root,'Batch35 original V2.1 long components/damped13 samples and original defects frozen')


def worker(root,directory,endpoint):
    import akshare as ak
    import baostock as bs
    require(api_evidence()==read(directory/'existing-apis.json'),'Probe API differs')
    query=QUERIES[endpoint]; folder=directory/'probe'/endpoint; folder.mkdir(parents=True,exist_ok=False); socket.setdefaulttimeout(8)
    row={'endpoint':endpoint,'query':query,'api_sha256':file_sha(directory/'existing-apis.json'),'status':'failed','published':False,'files':[],'wire':[]}
    original=requests.sessions.Session.request
    def request(session,method,url,**kw):
        require(len(row['wire'])<10,'Response cap exceeded'); session.trust_env=False; kw['timeout']=(8,10)
        response=original(session,method,url,**kw); path=folder/f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file':str(path),'sha256':file_sha(path),'url':response.url,'status':response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session,'request',request),redirect_stdout(io.StringIO()):
            if query['provider']=='baostock':
                with BaoStock(root).session():
                    result=getattr(bs,query['function'])(**query['parameters']); values=[]
                    require(result.error_code=='0',f'Provider error: {result.error_code}/{result.error_msg}')
                    while result.next(): values.append(result.get_row_data()); require(len(values)<=10000,'Provider row cap exceeded')
                    frame=pd.DataFrame(values,columns=result.fields)
            else: frame=getattr(ak,query['function'])(**query['parameters'])
        require(0<len(frame)<=100000,'Empty/excessive provider sample'); path=raw.save(root,'batch35_dependency_probe',endpoint,directory.name,frame)
        row.update(status='sample',files=[{'file':str(path),'sha256':file_sha(path),'rows':len(frame),'columns':list(frame.columns)}],limit=query['limit'],strict_usable=False)
    except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
    save(directory/f'probe-{endpoint}.json',row); return row


def validate_probes(directory):
    with patch.multiple(previous, QUERIES=QUERIES, api_evidence=api_evidence):
        return previous.validate_probes(directory)


def offline_catalog(root,directory):
    proof=Path(root)/'catalog/strategies/implementation-evidence.json'
    require(file_sha(proof)==file_sha(directory/'implementation-evidence.before.json'),'Implementation registry changed')
    folder=directory/'catalog-offline'; folder.mkdir(exist_ok=False); copied=folder/'catalog/strategies/implementation-evidence.json'
    copied.parent.mkdir(parents=True); shutil.copyfile(proof,copied); result=catalog_strategies(folder,'repo/量化策略源代码')
    original=accepted_review(directory)['catalog']; require(result['catalog_id']==original['catalog_id'],'Offline catalog identity differs'); rows=[]
    for name in ('catalog.json','summary.json','catalog.parquet'):
        a=Path(original['output'])/name; b=Path(result['output'])/name; require(file_sha(a)==file_sha(b),'Offline catalog bytes differ')
        rows.append({'name':name,'original_file':str(a),'repeated_file':str(b),'sha256':file_sha(a)})
    return {'result':'match','differences':0,'socket_network_disabled':True,'catalog_id':result['catalog_id'],
        'files':rows,'implementation_evidence_sha256':file_sha(proof)}


def validate_catalog(root,directory):
    doc=read(directory/'offline-catalog.json'); original=accepted_review(directory)['catalog']
    require(doc['result']=='match' and doc['differences']==0 and doc['socket_network_disabled'] is True and doc['catalog_id']==original['catalog_id'] and
        len(doc['files'])==3 and {r['name'] for r in doc['files']}=={'catalog.json','summary.json','catalog.parquet'},'Offline catalog scope differs')
    require(doc['implementation_evidence_sha256']==file_sha(Path(root)/'catalog/strategies/implementation-evidence.json')==
        file_sha(directory/'implementation-evidence.before.json'),'Implementation registry differs')
    for row in doc['files']:
        a=Path(original['output'])/row['name']; b=directory/'catalog-offline/catalog/strategies'/doc['catalog_id']/row['name']
        require(row['original_file']==str(a) and row['repeated_file']==str(b) and file_sha(a)==row['sha256']==file_sha(b),'Offline catalog bytes/path differ')
    require(read(Path(root)/'catalog/strategies/latest.json')['catalog_id']==doc['catalog_id'],'Latest catalog differs')


def probe(root,directory):
    binding(root,directory)
    for endpoint,query in QUERIES.items():
        command=[sys.executable,'-m','scripts.review_strategy_batch35','worker','--root',root,'--directory',str(directory),'--endpoint',endpoint]
        with (directory/f'probe-{endpoint}.stdout').open('x') as out,(directory/f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command,stdout=out,stderr=err,timeout=45,check=True)
            except subprocess.TimeoutExpired:
                if not (directory/f'probe-{endpoint}.json').exists(): save(directory/f'probe-{endpoint}.json',{'endpoint':endpoint,'query':query,
                    'api_sha256':file_sha(directory/'existing-apis.json'),'status':'timeout','published':False,'files':[],'wire':[],'error':'Child exceeded45s'})
    save(directory/'probe-results.json',{'results':validate_probes(directory),'published':False})
    return archive(root,'Batch35 index1000/ETF supplementation attempts frozen; no publication')


def offline(root,directory):
    binding(root,directory); before=implementation()
    def denied(*a,**kw): raise AssertionError('Batch35 offline verification attempted network')
    with patch.object(socket,'socket',denied),patch.object(socket,'create_connection',denied):
        result=compute(directory); diagnostic=diagnostics(directory); catalog=offline_catalog(root,directory)
    require(before == implementation() and diagnostic == read(directory/'diagnostics.json'),'Offline code/diagnostics differ')
    save(directory/'component-offline.json',result); require(file_sha(directory/'component-offline.json') == file_sha(directory/'component-research.json'),'Offline components differ')
    save(directory/'offline-catalog.json',catalog); save(directory/'offline-verification.json',{'result':'match','differences':0,
        'socket_network_disabled':True,'implementation_sha256':before,'sha256':file_sha(directory/'component-offline.json'),'not_a_backtest':True})
    validate_catalog(root,directory); return archive(root,'Batch35 forbidden-network components/diagnostics/catalog match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests','scripts/review_strategy_batch35.py',
        'scripts/prepare_handoff.py','scripts/build_strategy_catalog.py','scripts/verify_financial_import.py'],
        [sys.executable,'-m','scripts.verify_offline_tests','-q','tests/unit/test_batch35_research.py','tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py','tests/unit/test_batch34_research.py','tests/unit/test_signal_slots.py','tests/integration/test_strategy_merge.py'],['git','diff','--check']]


def checks(root,directory):
    binding(root,directory); validate_catalog(root,directory); validate_probes(directory); before=implementation(); rows=[]
    require(all((directory/n).is_file() for n in CORE),'Core evidence missing')
    for k,command in enumerate(commands()):
        path=directory/f'check-{k}.log'
        with path.open('x') as stream: result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        rows.append({'command':command,'returncode':result.returncode,'log':str(path),'sha256':file_sha(path)})
        require(result.returncode == 0,f'Check failed: {path}')
    require(before == implementation(),'Checked code changed')
    for name in FILES:
        copied=directory/'implementation-checked'/name; copied.parent.mkdir(parents=True,exist_ok=True)
        require(not copied.exists(),'Checked copy exists'); shutil.copyfile(name,copied)
    save(directory/'checked-state.json',{'status':'passed','commands':rows,'implementation_sha256':before,
        'evidence_sha256':{p.relative_to(directory).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status':'passed','commands':rows}


def finish(root,directory):
    checked=read(directory/'checked-state.json'); off=read(directory/'offline-verification.json')
    require(checked['status']=='passed' and checked['implementation_sha256']==implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']]==commands() and all(r['returncode']==0 for r in checked['commands']),'Checked code/commands/core differ')
    for name,sha in checked['evidence_sha256'].items(): require(file_sha(directory/name)==sha,'Checked evidence changed')
    for name,sha in checked['implementation_sha256'].items(): require(file_sha(directory/'implementation-checked'/name)==sha,'Checked code copy changed')
    require(off['implementation_sha256']==implementation() and off['result']=='match' and off['differences']==0 and off['socket_network_disabled'] is True and
        off['sha256']==file_sha(directory/'component-research.json')==file_sha(directory/'component-offline.json'),'Offline evidence changed')
    binding(root,directory); validate_catalog(root,directory)
    require(read(directory/'probe-results.json')=={'results':validate_probes(directory),'published':False},'Probe summary differs')
    require(compute(directory)==read(directory/'component-research.json') and diagnostics(directory)==read(directory/'diagnostics.json'),'Recomputed components differ')
    require(Store(root).published()==read(directory/'baseline.json')['published'],'Publication changed')
    protection=protect(root,directory); progress=archive(root,'Batch35 original components/source audit accepted; original trading dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch35-verification.json'),{'status':'ok','reviews':accepted_review(directory),
        'progress':progress,'snapshot':SNAPSHOT,'checks':checked,'offline':off,'offline_catalog':read(directory/'offline-catalog.json'),
        'protection':protection,'probes':read(directory/'probe-results.json'),'not_a_backtest':True,'original_strategy_complete':False})
    return {'status':'ok','checkpoint':progress['checkpoint'],'manually_reviewed':progress['manually_reviewed']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['start','correct','prepare','study','worker','probe','offline','checks','finish'])
    parser.add_argument('--root',default='data'); parser.add_argument('--directory',default='data/staging/strategies-batch35/20261006-bbi-damped-v21')
    parser.add_argument('--endpoint',choices=list(QUERIES)); args=parser.parse_args()
    result=worker(args.root,Path(args.directory),args.endpoint) if args.action=='worker' else globals()[args.action](args.root,Path(args.directory))
    print(json.dumps(result,ensure_ascii=False,default=str))
