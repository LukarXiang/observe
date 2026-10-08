"""Freeze multihorizon/LR/breadth research and verify available daily operators."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import inspect
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
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch35 as previous
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch35/20261006-bbi-damped-v21')
RECEIPT = Path('docs/handoff/2026-10-06-batch35-verification.json')
SOURCES = ('2020年度精选策略/15 基于多期限的选股策略（一）.txt',
    '2021年度精选策略/60.【分享】对RSRS模型的一次修改.txt', '2021年度精选策略/45.市场宽度.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/sma.py'),
    ('repo/akshare', 'akshare/stock/stock_industry_sw.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch36.py', 'tests/unit/test_batch36_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json',
    'index-input.parquet', 'calendar-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {
    'pool180': {'provider':'akshare','function':'index_stock_cons_csindex','parameters':{'symbol':'000010'},
        'limit':'Current provider composition cannot replace the original initialization-date pool'},
    'poolcirculating': {'provider':'akshare','function':'index_stock_cons_csindex','parameters':{'symbol':'000902'},
        'limit':'Current circulating-index membership is not30 historical daily platform pools'},
    'industry': {'provider':'akshare','function':'stock_industry_clf_hist_sw','parameters':{},
        'limit':'Historical classification sample still needs platform version/date mapping; cannot replace sw_l1'}}

def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch35 differs')
    require(file_sha(ACCEPTED / 'input-binding.json') == receipt['checks']['evidence_sha256']['input-binding.json'], 'Accepted batch35 binding changed')
    upstream = previous.binding(root, ACCEPTED)
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted index/calendar bytes differ')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': upstream, 'files': rows, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    import baostock as bs
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(bs if query['provider'] == 'baostock' else ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    checkpoint(root, directory, 'Batch36 multihorizon/LR/breadth source research started')
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
            'use': 'Read-only SMA and historical classification schema; no platform replacement'})
    search=['rg','--files','--hidden','repo','data','-g','*Creat_RSRS*']
    result=subprocess.run(search,capture_output=True,text=True); require(result.returncode in (0,1),result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory/'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search':{'command':search,'returncode':result.returncode,'matches':result.stdout.splitlines()},
        'modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata','jqlib','statsmodels','pyfolio','tushare','Creat_RSRS')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch36 three full sources frozen; pool/core/industry gaps and original label faults retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path)
        profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns),
            'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['HS300 close supports original LR reference; only a multihorizon formula sample, not stock-pool replacement',
            'No original industry/BIAS histories or Creat_RSRS base; no trading results, NAV or fees calculated']})
    return archive(root, 'Batch36 accepted HS300/calendar bytes frozen for limited LR/multihorizon operator research')


def inputs(directory):
    doc = read(directory / 'input-analysis.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and doc['platform_equivalent'] is False and
        doc['component_instrument'] == '000300.SH' and set(doc['profiles']) == {'index-input.parquet', 'calendar-input.parquet'}, 'Input scope differs')
    frames = {}
    for name, row in doc['profiles'].items():
        path = directory / name; require(row['file'] == str(path) and file_sha(path) == row['sha256'], 'Input binding changed')
        frame = pd.read_parquet(path); keys = ['date', 'index'] if name.startswith('index') else ['date']
        require(len(frame) == row['rows'] and list(frame.columns) == row['columns'] and frame.date.min() == row['first'] and
            frame.date.max() == row['last'] and not frame[keys].isna().any().any() and not frame.duplicated(keys).any(), 'Input profile differs')
        frames[name] = frame
    frame = frames['index-input.parquet'].sort_values('date').reset_index(drop=True)
    calendar = frames['calendar-input.parquet']; sessions = set(calendar[calendar.is_open].date)
    require(set(frame['index']) == {'000300.SH'} and set(frame.date) <= sessions and
        np.isfinite(frame[['close', 'volume']]).all().all() and frame.close.gt(0).all() and frame.volume.ge(0).all(), 'Invalid component input')
    return frame, sorted(sessions)


PERIODS = (3,5,10,20,30,60,90,120,180,240,270,300)


def original_class(directory,name):
    node=next(n for n in source_tree(directory,1).body if isinstance(n,ast.ClassDef) and n.name==name)
    ns={'np':np,'pd':pd,'RSRS':object}
    module=ast.Module(body=[node],type_ignores=[]); exec(compile(module,'<original-class-static-operator-only>','exec'),ns)
    return ns[name],hashlib.sha256(ast.dump(module).encode()).hexdigest()


def reference_lr(close):
    values=list(map(float,close)); require(len(values)>=240 and np.isfinite(values).all() and min(values)>0,'Invalid LR input')
    return [sum(values[k]>math.fsum(values[k-p+1:k+1])/p for p in range(10,250,10))/24
        for k in range(239,len(values))]


def multihorizon_namespace(directory,frame):
    holder={'end':len(frame)-1}; calls=[]; closes=frame[['close']]
    def price(stock,end_date,frequency,fields,skip_paused,fq=None,count=None):
        require(stock=='index_operator_sample' and frequency=='daily' and skip_paused is True and fq=='pre' and
            fields=='close' and count in PERIODS,'Unexpected multihorizon request')
        require(end_date==frame.date.iloc[holder['end']] and holder['end']>=count-1,'Insufficient multihorizon window')
        calls.append({'count':count,'fq':fq,'skip_paused':skip_paused,'end_date':end_date})
        return closes.iloc[holder['end']-count+1:holder['end']+1].copy()
    # The platform's undefined mean is explicitly modeled as column-wise pandas mean.
    ns={'get_price':price,'mean':lambda data:data.mean()}
    sha=selected(directory,0,('calAt','calAtevery'),ns)
    return ns,holder,calls,sha


def configuration(directory):
    calls=[]; g=SimpleNamespace()
    ns={'g':g,'log':SimpleNamespace(set_level=lambda *a:None),'set_benchmark':lambda x:calls.append(['benchmark',x]),
        'set_option':lambda *a:calls.append(['option',*a]),'OrderCost':lambda **kw:kw,
        'set_order_cost':lambda *a,**kw:calls.append(['cost',a[0],kw]),
        'set_slippage':lambda *a:calls.append(['slippage',*a]),
        'get_index_stocks':lambda x:calls.append(['pool',x]) or ['synthetic_pool'],
        'run_daily':lambda fn,**kw:calls.append(['schedule',fn,kw]),
        'before_market_open':'before_market_open','market_open':'market_open','after_market_close':'after_market_close'}
    selected(directory,0,('initialize','set_pas','set_variables'),ns); ns['initialize'](None)
    return g,calls


def compute(directory):
    validate_sources(directory); frame,sessions=inputs(directory)
    dates=[d for d in sessions if frame.date.min()<=d<=frame.date.max()]
    require(frame.date.tolist()==dates,'Benchmark has missing verified session')
    ns={'np':np,'pd':pd}; sha=selected(directory,1,('cala_LR','add_flag'),ns)
    series=pd.Series(frame.close.to_numpy(),index=pd.to_datetime(frame.date))
    lr_results={}
    for label,values in [('long',series),('original_lr_dates',series.loc['2008-01-01':'2020-08-13'])]:
        actual=ns['cala_LR'](values); expected=reference_lr(values); require(len(actual)==len(values)-239,'LR warmup differs')
        boundaries=[]
        for k,(a,b) in enumerate(zip(actual,expected,strict=True)):
            if a!=b:
                end=k+239; operands=[]
                for p in range(10,250,10):
                    original=float(values.rolling(p).mean().iloc[end]); independent=math.fsum(map(float,values.iloc[end-p+1:end+1]))/p
                    if (values.iloc[end]>original)!=(values.iloc[end]>independent):
                        operands.append({'period':p,'close':float(values.iloc[end]),'rolling':original,'stable_sum':independent})
                boundaries.append({'end_date':actual.index[k].date().isoformat(),'actual':float(a),'reference':b,'operands':operands})
        lr_results[label]={'windows':len(actual),'first_end_date':actual.index[0].date().isoformat(),'last_end_date':actual.index[-1].date().isoformat(),
            'boundaries':boundaries,'max_difference':max(abs(float(a)-b) for a,b in zip(actual,expected,strict=True)),
            'sha256':hashlib.sha256(actual.to_json(date_format='iso',double_precision=15).encode()).hexdigest()}
    long_lr=ns['cala_LR'](series)
    original_rank=next(n for n in ast.walk(source_tree(directory,1)) if isinstance(n,ast.Lambda) and
        isinstance(n.body,ast.Subscript) and isinstance(n.body.value,ast.Call) and isinstance(n.body.value.func,ast.Attribute) and n.body.value.func.attr=='rank')
    rank_fn=eval(compile(ast.Expression(original_rank),'<original-LR-percentile-lambda>','eval'))
    rank_max=0.; rank_digest=hashlib.sha256(); rank_samples=[]; values=(1-long_lr).to_numpy()
    for end in range(599,len(values)):
        window=values[end-599:end+1]; actual=float(rank_fn(pd.Series(window,index=range(-600,0))))
        expected=(sum(window<window[-1])+(sum(window==window[-1])+1)/2)/600
        rank_max=max(rank_max,abs(actual-expected)); rank_digest.update(f'{actual:.17g}\n'.encode())
        if end in (599,len(values)-1): rank_samples.append({'date':long_lr.index[end].date().isoformat(),'value':actual})
    require(rank_max==0,'Original LR average rank differs')
    improve,improve_sha=original_class(directory,'RSRS_improve1'); quantiles=pd.Series([0.,.25,.5,.75,1.],index=list('abcde'))
    table=pd.DataFrame({'RSRS_z':[-2.,-1.,0.,1.,2.],'R_2':[.1,.3,.5,.7,.9]},index=list('abcde'))
    damp_actual=improve.cala_passivation_RSRS(table.copy(),quantiles.iloc[::-1])['RSRS_passivation']
    independent=[float(table.loc[k,'RSRS_z'])*float(table.loc[k,'R_2'])**(2*(1-float(quantiles.loc[k]))) for k in table.index]
    require(np.allclose(damp_actual,independent,rtol=0,atol=1e-15),'Reverse dampening differs')
    multi,holder,_,multi_sha=multihorizon_namespace(directory,frame); multi_rows=[]; max_error=0.; total=0
    for period in PERIODS:
        digest=hashlib.sha256(); count=0
        for end in range(period-1,len(frame)):
            holder['end']=end; actual=multi['calAt']('index_operator_sample',frame.date.iloc[end],period)
            expected=math.fsum(map(float,frame.close.iloc[end-period+1:end+1]))/period/float(frame.close.iloc[end])
            require(math.isfinite(actual),'Nonfinite multihorizon value'); max_error=max(max_error,abs(actual-expected)); count+=1
            digest.update(f'{actual:.17g}\n'.encode())
        total+=count; multi_rows.append({'period':period,'windows':count,'first_end_date':frame.date.iloc[period-1],
            'last_end_date':frame.date.iloc[-1],'sha256':digest.hexdigest()})
    require(max_error<1e-12,'Multihorizon ratio differs')
    return {'snapshot':SNAPSHOT,'not_a_backtest':True,'original_strategy_complete':False,'strategy_results':[],
        'input_sha256':{n:file_sha(directory/n) for n in ('index-input.parquet','calendar-input.parquet')},
        'backend':{'numpy':np.__version__,'pandas':pd.__version__},
        'lr':{'source_ast_sha256':sha,'periods':list(range(10,250,10)),'results':lr_results},
        'inverse_lr_rank':{'windows':len(values)-599,'max_difference':rank_max,'samples':rank_samples,'sha256':rank_digest.hexdigest(),
            'source_ast_sha256':hashlib.sha256(ast.dump(original_rank).encode()).hexdigest(),'legacy_adapter':'negative integer labels for [-1]'},
        'reverse_dampening':{'source_ast_sha256':improve_sha,'actual':list(map(float,damp_actual)),
            'cases':5,'input':table.to_dict(),'quantiles':quantiles.to_dict()},
        'multihorizon':{'source_ast_sha256':multi_sha,'windows':total,'max_difference':max_error,'periods':multi_rows,
            'mean_adapter':'Explicit pandas column mean models undefined platform mean; platform equivalence unproved'},
        'limits':['Overlapping mean/LR component samples only; no original stock-pool replacement, ranking or fitted model',
            'LR two date ranges overlap; endpoints are completed window dates, not intraday/trading dates',
            'No Creat_RSRS base reconstruction or original benchmark-return/fee/NAV loops',
            'No actual industry breadth/BIAS validation; synthetic aggregation diagnosis only; strict numeric boundary evidence retained']}


def breadth_fragment(directory):
    loop=next(n for n in source_tree(directory,2).body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='day')
    start=next(k for k,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='day_market_capacity' for t in n.targets))
    module=ast.Module(body=loop.body[start:],type_ignores=[])
    return compile(module,'<original-breadth-aggregation-only>','exec'),hashlib.sha256(ast.dump(module).encode()).hexdigest()


def diagnostics(directory):
    import datetime
    import importlib
    from sklearn import linear_model
    cases=[]; rows=[]
    g,calls=configuration(directory); cases.append({'case':'multihorizon_initial_configuration','parameters':vars(g),'calls':calls})
    try: importlib.import_module('sklearn.preprocessing').Imputer
    except AttributeError as exc: cases.append({'case':'removed_imputer','error':str(exc)})
    else: raise AssertionError('Original Imputer fault disappeared')
    try: importlib.import_module('Creat_RSRS')
    except ModuleNotFoundError as exc: cases.append({'case':'missing_core_module','error':str(exc)})
    else: raise AssertionError('New RSRS dependency needs review')
    ns={'pd':pd,'get_price':lambda *a,**kw:pd.DataFrame({'close':[1.,2.,3.],'open':[10.,20.,40.]})}
    selected(directory,0,('calAt',),ns)
    try: ns['calAt']('sample','2020-01-01',3)
    except NameError as exc: cases.append({'case':'undefined_mean','error':str(exc)})
    else: raise AssertionError('Original mean fault disappeared')
    requests=[]; sample=pd.DataFrame({'close':[11.,12.,13.,14.,15.],'open':[20.,40.,80.,160.,320.]})
    def get_price(stock,**kw): requests.append({'stock':stock,**kw,'end_date':kw['end_date'].isoformat()}); return sample.copy()
    ns={'datetime':datetime,'get_price':get_price}; selected(directory,0,('calRFlist','calATlist'),ns)
    labels=ns['calRFlist']('sample',datetime.datetime(2020,1,8)); require(labels==[-.5]*25,'Original reversed open label differs')
    cases.append({'case':'reversed_open_labels','labels':labels,'requests':requests.copy()}); requests.clear()
    def at(stock,date,n): rows.append({'date':date.isoformat(),'period':n}); return pd.DataFrame({'close':[1.]})
    ns['calAtevery']=at; ns['calATlist']('sample',datetime.datetime(2020,1,8),3)
    cases.append({'case':'feature_natural_week_lag','requests':rows.copy()})
    ns={'pd':pd,'linear_model':linear_model,'calATlist':lambda *a:list(np.arange(25.)+a[-1]),
        'calRFlist':lambda *a:list(np.arange(25.)*.01),'calAt':lambda *a:1.}
    selected(directory,0,('calEstYeild',),ns)
    try: ns['calEstYeild']('sample','2020-01-01')
    except ValueError as exc: cases.append({'case':'predict_one_dimensional_error','error':str(exc)})
    else: raise AssertionError('Original predict shape fault disappeared')
    ns={'pd':pd,'calEstYeild':lambda *a:[1.]}; selected(directory,0,('SortStockList',),ns)
    try: ns['SortStockList'](['sample'],'2020-01-01')
    except AttributeError as exc: cases.append({'case':'removed_sort','error':str(exc)})
    else: raise AssertionError('Original sort fault disappeared')
    g=SimpleNamespace(tc=7,t=0,num_stocks=5,if_trade=False,stocks=['synthetic_pool']); orders=[]
    ns={'g':g,'log':SimpleNamespace(info=lambda *a:None),'SortStockList':lambda *a:pd.DataFrame(index=['a','b','c','d','e']),
        'order_target_value':lambda *a:orders.append(list(a)),'get_trades':lambda:{}}
    selected(directory,0,('before_market_open','market_open','after_market_close'),ns)
    ctx=SimpleNamespace(current_dt=datetime.datetime(2020,1,8,9,30),portfolio=SimpleNamespace(portfolio_value=10000.,positions={'old':None}))
    schedule=[]
    for day in range(15):
        ns['before_market_open'](ctx)
        if g.if_trade: schedule.append(day)
        ns['market_open'](ctx); ns['after_market_close'](ctx)
    cases.append({'case':'seven_callback_schedule_and_equal_orders','days':schedule,'orders':orders,'end_flag':g.if_trade})
    ns={'np':np,'pd':pd}; selected(directory,1,('add_flag',),ns)
    signals=pd.Series([.7,.700001,np.nan,-.7,-.700001,.7],index=pd.date_range('2020-01-01',periods=6))
    cases.append({'case':'flag_strict_threshold_nan_hold','flags':list(map(float,ns['add_flag'](signals,.7)))})
    try: ns['add_flag'](pd.Series(dtype=float),.7)
    except IndexError as exc: cases.append({'case':'flag_empty_error','error':str(exc)})
    else: raise AssertionError('Original empty flag fault disappeared')
    cls,sha=original_class(directory,'RSRS_improve2'); local=source_tree(directory,1)
    ranked=next(n for n in ast.walk(local) if isinstance(n,ast.Lambda) and isinstance(n.body,ast.Subscript) and
        isinstance(n.body.value,ast.Call) and isinstance(n.body.value.func,ast.Attribute) and n.body.value.func.attr=='rank')
    rank=eval(compile(ast.Expression(ranked),'<original-rank-lambda>','eval'))
    try: rank(pd.Series([1.,2.,2.],index=pd.date_range('2020-01-01',periods=3)))
    except KeyError as exc: cases.append({'case':'LR_rank_legacy_index_error','error':str(exc),'source_ast_sha256':sha})
    else: raise AssertionError('Original rank fault disappeared')
    require(cls.__bases__==(object,),'Diagnostic base not isolated')
    ns={'pd':pd,'get_industry':lambda names,**kw:{'a':{'sw_l1':{'industry_name':'A'}},'b':{'sw_l1':None},'c':{'sw_l1':{'industry_name':'B'}}}}
    selected(directory,2,('getStockIndustry','getStockBIAS'),ns)
    industries=ns['getStockIndustry'](['a','b','c'],'sw_l1','2020-01-01')
    cases.append({'case':'industry_missing_excluded','stocks':industries.index.tolist(),'industries':industries.sw_l1.tolist()})
    calls=[]
    def bias(*a,**kw): calls.append({'args':list(a),'kwargs':kw}); return {'a':0.},None,None
    ns['BIAS']=bias; result=ns['getStockBIAS'](['a'],'2020-01-01')
    cases.append({'case':'original_BIAS_call_only','calls':calls,'returned':result.to_dict()})
    code,bsha=breadth_fragment(directory)
    for label,values in [('zero_and_nan',[1.,0.,np.nan,-1.]),('half_percent_rounding',[1.]+[0.]*199)]:
        data=pd.DataFrame({'sw_l1':['A']*len(values),'bias':values}); out=pd.DataFrame(index=['A','index_all'])
        local={'stock_df':data,'industries_type':'sw_l1','market_capacity_day':out,'day':'synthetic'}; exec(code,local)
        cases.append({'case':f'breadth_{label}','industry':float(out.loc['A','synthetic']),
            'index_all':float(out.loc['index_all','synthetic']),'source_ast_sha256':bsha})
    return {'not_a_backtest':True,'cases':cases,'limits':['Synthetic source requests/orders only; no actual original universe/industry/BIAS data',
        'No root module imports, token queries, plotting, original return/NAV/fee loops or external messages']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch36 long LR/multihorizon operators and original defects/request diagnostics frozen')


def worker(root, directory, endpoint):
    import akshare as ak
    import baostock as bs
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API differs')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(8)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kw):
        require(len(row['wire']) < 10, 'Response cap exceeded'); session.trust_env = False; kw['timeout'] = (8, 10)
        response = original(session, method, url, **kw); path = folder / f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status': response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()):
            if query['provider'] == 'baostock':
                with BaoStock(root).session():
                    result = getattr(bs, query['function'])(**query['parameters']); values = []
                    require(result.error_code == '0', f'Provider error: {result.error_code}/{result.error_msg}')
                    while result.next(): values.append(result.get_row_data()); require(len(values) <= 10000, 'Provider row cap exceeded')
                    frame = pd.DataFrame(values, columns=result.fields)
            else: frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch36_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}],
            limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['published'] is False, 'Probe binding differs')
        for item in [*row['files'], *row['wire']]: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] == 'sample':
            require(row['strict_usable'] is False and row['limit'] == query['limit'] and len(row['files']) == 1, 'Unproven sample admitted')
            item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(0 < len(frame) == item['rows'] <= 100000 and list(frame.columns) == item['columns'], 'Sample profile differs')
        else: require(row['status'] in ('failed', 'timeout') and not row['files'] and row.get('error'), 'Failed probe admitted data')
        rows.append(row)
    return rows


def probe(root, directory):
    binding(root, directory)
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch36', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch36 composition/industry supplementation attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch36 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch36 forbidden-network operator/diagnostic/catalog byte match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch36.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch36_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch35_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked code changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True); require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch36 LR/multihorizon/breadth components accepted; original pool/core/industry dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch36-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch36/20261006-multihorizon-lr-breadth')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
