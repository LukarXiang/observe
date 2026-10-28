"""Execute reviewed original breadth fragments and listing-age functions; never synthesize NHNL trades."""
import ast
from contextlib import redirect_stdout
import copy
from io import StringIO
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch101 import canonical
from scripts.verify_strategy_batch3 import require

PREREAD = Path('data/staging/strategies-batch104/20261011-next-source-preread')
SOURCES = ('2024年度精选策略2/52.市场宽度——简洁版.txt',
    '2024年度精选策略1/89.【复现·水】市场情绪指标专题行业指数顶部和底部信号.txt')
SOURCE_SHA = ('99d10bce41b0c7e1c2669c3c0c74afefea384ba6a099673c24e041fde5b89aaf',
    'f41f457f69e71ddb4b0b22697ca8a9fd329d7ba2954f2c7a4571e7296896f4e8')


def tree(number):
    require(number in (0,1),'Unknown source')
    path = PREREAD/f'{number}.source'
    require(file_sha(path) == file_sha(Path('repo/量化策略源代码')/SOURCES[number]) == SOURCE_SHA[number],
        'Original source bytes changed')
    return ast.parse(read_source(path)[0])


def fragment(names, *, include_industry_loop=False):
    nodes = []
    for node in tree(0).body:
        if isinstance(node,ast.Assign):
            require(len(node.targets) == 1,'Unexpected multiple assignment')
            target = node.targets[0]
            name = target.id if isinstance(target,ast.Name) else target.value.id if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) else None
            if name in names: nodes.append(node)
        elif include_industry_loop and isinstance(node,ast.For): nodes.append(node)
    require(bool(nodes),'Empty original fragment')
    return compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged-source52-fragment>','exec')


def original_bias(close):
    ns = {'df_close':close}
    exec(fragment({'df_bias'}),ns)
    return ns['df_bias']


def breadth_fixture(*, missing=False, duplicate_rows=False, duplicate_names=False, no_industries=False):
    dates = pd.bdate_range(end='2022-09-02',periods=40)
    raw = pd.DataFrame({'time':np.tile(dates,3),'code':np.repeat(['A','B','C'],40),
        'close':np.r_[np.arange(1.,41.),np.arange(40.,0.,-1.),np.full(40,10.)]})
    if missing: raw.loc[39,'close'] = np.nan
    if duplicate_rows: raw = pd.concat([raw,raw.iloc[:1]],ignore_index=True)
    industries = pd.DataFrame({'name':['one','one' if duplicate_names else 'two','empty']},index=['I','II','III'])
    if no_industries: industries = industries.iloc[:0]
    members = {'I':['A','B','B','outside'],'II':['C'],'III':['outside']}; calls = []
    def query(name,value): calls.append([name,value]); return value
    ns = {'pd':pd,'get_trade_days':lambda **k: query('calendar_parameters',k) and dates,
        'get_all_securities':lambda **k: query('securities_date',k) and pd.DataFrame(index=['A','B','C']),
        'get_price':lambda *a,**k: query('price_parameters',{'args':list(a),'kwargs':k}) and raw,
        'get_industries':lambda *a,**k: query('industry_parameters',{'args':list(a),'kwargs':k}) and industries,
        'get_industry_stocks':lambda code,**k: query('membership_parameters',{'code':code,**k}) and members[code]}
    names = {'end_date','count_','by_date','stock_list','df_close','df_bias','df_industries','df','columns',
        'df_1','vmax','df_2','df_3'}
    exec(fragment(names,include_industry_loop=True),ns)
    return {'calls':canonical(calls),'rows':len(ns['df_bias']),'columns':list(ns['df_bias'].columns),
        'last_bias':ns['df_bias'].iloc[-1].to_dict(),'industry_last':ns['df'].iloc[0].to_dict() if len(ns['df']) else {},
        'scheme1':ns['df_1']['总体'].tolist(),'scheme2':ns['df_2']['总体'].tolist(),
        'scheme3':ns['df_3']['总体'].tolist(),'vmax':ns['vmax'],'synthetic_only':True}


def listing_function(*, explicit_ns_dtype=False, basic=None, periods=None):
    node = copy.deepcopy(next(n for n in tree(1).body if isinstance(n,ast.FunctionDef)))
    require(node.name == 'get_ind_overthreshold_num' and not node.decorator_list and not node.type_params,
        'Original function contract differs')
    annotations = [arg.annotation for arg in node.args.args]+[node.returns]
    require(all(isinstance(a,ast.Name) and a.id in {'Dict','str','int'} or
        isinstance(a,ast.Attribute) and isinstance(a.value,ast.Name) and a.value.id == 'pd' and a.attr == 'DataFrame'
        for a in annotations),'Unreviewed executable annotation')
    if explicit_ns_dtype:
        matches = [call for call in ast.walk(node) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and call.func.attr == 'astype' and len(call.args) == 1 and isinstance(call.args[0],ast.Attribute)
            and isinstance(call.args[0].value,ast.Name) and call.args[0].value.id == 'np' and call.args[0].attr == 'datetime64']
        require(len(matches) == 1,'Datetime compatibility patch is not exactly one argument')
        matches[0].args[0] = ast.copy_location(ast.Constant('datetime64[ns]'),matches[0].args[0])
    ns = {'pd':pd,'np':np,'Dict':Dict,'List':List,'get_ts_stock_basic':lambda:basic,
        'get_trade_days':lambda *a: periods}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        '<source89-one-explicit-dtype-reference>' if explicit_ns_dtype else '<unchanged-source89-function>','exec'),ns)
    return ns['get_ind_overthreshold_num']


def age_fixture(*, explicit=False, missing=False, duplicate=False):
    basic = pd.DataFrame({'code':['A','B','outside'],
        'list_date':pd.to_datetime(['2020-01-01','2020-01-02','2010-01-01'])})
    if missing: basic.loc[1,'list_date'] = pd.NaT
    if duplicate: basic = pd.concat([basic,basic.iloc[:1]],ignore_index=True)
    periods = pd.to_datetime(['2020-09-09','2020-09-10']).to_numpy(dtype='datetime64[ns]')
    out = listing_function(explicit_ns_dtype=explicit,basic=basic,periods=periods)(
        {'A':'same','B':'same'},'2020-09-09','2020-09-10',252)
    return {'index':out.index.strftime('%Y-%m-%d').tolist(),'columns':list(out.columns),'values':out.to_numpy().tolist(),
        'explicit_ns_dtype_reference':explicit,'not_original_industry_data':True}


def diagnostics():
    cases = []
    def add(name,callback):
        output = StringIO(); notes = []; error = None; value = None
        with redirect_stdout(output),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try: value = callback()
            except Exception as exc: error = {'type':type(exc).__name__,'message':str(exc)}
            notes = [{'type':type(w.message).__name__,'message':str(w.message)} for w in caught]
        cases.append({'case':name,'value':canonical(value),'error':error,'warnings':notes,'stdout':output.getvalue()})
    add('breadth_three_different_denominators',breadth_fixture)
    add('breadth_missing_close_kept_in_denominator',lambda:breadth_fixture(missing=True))
    add('breadth_duplicate_time_code_pivot_error',lambda:breadth_fixture(duplicate_rows=True))
    add('breadth_duplicate_industry_names_overwrite',lambda:breadth_fixture(duplicate_names=True))
    add('breadth_no_industry_still_global_scheme3',lambda:breadth_fixture(no_industries=True))
    add('age_original_generic_datetime_dtype',age_fixture)
    add('age_explicit_ns_strict252_natural_days',lambda:age_fixture(explicit=True))
    add('age_explicit_ns_missing_listing',lambda:age_fixture(explicit=True,missing=True))
    add('age_explicit_ns_duplicate_basic_code',lambda:age_fixture(explicit=True,duplicate=True))
    for length in (0,19,20,21,40):
        add('bias_output_rows_'+str(length),lambda length=length:{'rows':len(original_bias(pd.DataFrame({'A':np.arange(float(length))})))})
    add('bias_constant_strict_equality',lambda:original_bias(pd.DataFrame({'A':[10.]*40})).any().to_dict())
    return {'status':'original_fragments_and_compatibility_reference_only','case_count':len(cases),'cases':cases,
        'compatibility_patch':'Only one astype argument: np.datetime64 to datetime64[ns];not an economic/trading approval',
        'original_nhnl_core_present':False,'original_trade_loop_present':False,'new_trading_backtests':0,
        'formal_source_research_acceptance':False,'library_goal_complete':False}
