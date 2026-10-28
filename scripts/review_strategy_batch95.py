"""Batch95 frozen original callbacks and long arithmetic references; no trading engine."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch94 as prior
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch59 import index_operand
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch95/20261010-magic3-v2-numerical-stage')
SOURCES = ('2023年度精选策略/7.ETF动量轮动RSRS择时-魔改3小优化.txt',
    '2024年度精选策略2/2.ETF动量轮动RSRS择时-魔改3小优化.txt',
    '2024年度精选策略2/60.ETF动量轮动RSRS择时-v2.txt')
SOURCE_SHA = ('e853b55be636530723810ba09160b80091dfd03a1674f8156681279d080b4938',
    'e853b55be636530723810ba09160b80091dfd03a1674f8156681279d080b4938',
    '7de6da9d4efc8e504a73e535393f2dcaf6d46d91de8ceb5618a05bd4acc6c9c6')
POOL = (('510050.XSHG', '159928.XSHE', '510300.XSHG', '159949.XSHE'),)*2 + (
    ('510050.XSHG', '510300.XSHG', '159949.XSHE', '159928.XSHE'),)
PARENTS = ('docs/handoff/2026-10-10-batch94-final-review.json',
    'docs/handoff/2026-10-10-yili-cmb-earlier-window-final-review.json')
PARENT_SHA = ('68245df3408d6a8aaf78cd8795d9a4502cce954863f4d05f42cb5cf95f565c50',
    '302a21b28c492fff0765ed9f99aef0d24f6bcfccc37eb3705f11f279a0f6929a')
FILES = ('scripts/review_strategy_batch95.py', 'tests/unit/test_batch95_stage.py')
OUTPUTS = ('diagnostics.json', 'rsrs-magic3.parquet', 'rsrs-v2-reference.parquet',
    'native-momentum-reference.parquet', 'component-research.json')


def literal(directory, number, namespace):
    require(number in (0, 1, 2), 'Unknown source number')
    path = directory/'sources'/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number], 'Batch95 source identity differs')
    tree = ast.parse(read_source(path)[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (15 if number == 2 else 13) and all(safe_function(n) for n in nodes),
        'Batch95 literal definition contract differs')
    # SHA allowlisted definitions only: platform imports/top-level code do not execute.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def parent_binding():
    result = {}
    for name, expected in zip(PARENTS, PARENT_SHA, strict=True):
        require(file_sha(name) == expected, 'Parent receipt identity changed')
        doc = read(name); require(doc['status'] == 'ok' and doc['manually_reviewed'] == 273, 'Parent status differs')
        result[name] = expected
        for key in ('receipt', 'final_binding', 'formal_report'):
            path = doc[key] if key == 'formal_report' else doc[key+'_file']
            require(file_sha(path) == doc[key+'_sha256'], 'Parent archive changed: '+path)
            result[path] = doc[key+'_sha256']
    for name, sha in read(PARENTS[0])['reviewed_implementation_sha256'].items():
        require(file_sha(name) == sha, 'Upstream implementation changed: '+name)
        result[name] = sha
    return result


def prepare():
    require(not DIRECTORY.exists(), 'Batch95 stage already exists')
    parents = parent_binding(); funds = prior.candidate_inputs(); benchmark = prior.benchmark_candidates()
    supplement = prior.supplement_evidence(); frame, operand = index_operand('data')
    require(Store('data').published()['batch_id'] == '20261006-145049-5ceb', 'Published input differs')
    rows = []
    for number, name in enumerate(SOURCES):
        path = Path('repo/量化策略源代码')/name
        require(file_sha(path) == SOURCE_SHA[number], 'Original source changed')
        text, encoding = read_source(path); tree = ast.parse(text)
        rows.append({'path': name, 'sha256': SOURCE_SHA[number], 'lines': len(text.splitlines()),
            'encoding': encoding, 'functions': [n.name for n in tree.body if isinstance(n, ast.FunctionDef)],
            'pool_order': list(POOL[number]), 'duplicate_of': 0 if number == 1 else None,
            'minimum_commission': 0 if number == 2 else 5,
            'source_copy': f'sources/{number}.source', 'original_platform_equivalence_proven': False})
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch95 magic3/v2 frozen arithmetic stage; formal273/trade12/parents77 unchanged')
        save(DIRECTORY/'parent-binding.json', parents)
        save(DIRECTORY/'candidate-inputs.json', funds)
        save(DIRECTORY/'benchmark-candidates.json', benchmark)
        # Reuse actual archived host failure; no claim that today's host was tested.
        supplement['missing'] = [s for s in supplement['missing'] if 'pykalman' not in s]
        supplement['missing'].append('v2 four-fund 60m history and callback-time visibility')
        save(DIRECTORY/'supplement-decision.json', supplement)
        (DIRECTORY/'sources').mkdir()
        for number, name in enumerate(SOURCES):
            shutil.copyfile(Path('repo/量化策略源代码')/name, DIRECTORY/'sources'/f'{number}.source')
        save(DIRECTORY/'source-reviews.json', {'sources': rows, 'formal_source_research_acceptance': False,
            'source_strategy_count': 3, 'unique_source_bytes': 2, 'active_buy_back_callback': False})
        with (DIRECTORY/'index-input.parquet').open('xb') as out: frame.to_parquet(out, index=False)
        save(DIRECTORY/'index-binding.json', {**operand, 'frozen_sha256': file_sha(DIRECTORY/'index-input.parquet')})
        hashes = {name: file_sha(name) for name in FILES}
        for name in FILES:
            target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        save(DIRECTORY/'implementation-binding.json', hashes)
    return {'status': 'prepared', 'index_rows': len(frame), 'native_funds': len(funds['funds']), 'new_trading_backtests': 0}


def validate(directory):
    require(read(directory/'parent-binding.json') == parent_binding(), 'Stage parent binding differs')
    hashes = read(directory/'implementation-binding.json')
    require(set(hashes) == set(FILES), 'Stage implementation manifest differs')
    for name, sha in hashes.items():
        require(file_sha(name) == file_sha(directory/'implementation'/name) == sha, 'Stage implementation changed')
    require(file_sha(directory/'index-input.parquet') == read(directory/'index-binding.json')['frozen_sha256'], 'Index changed')
    for number in range(3): literal(directory, number, {})
    return hashes


def log_stub(events):
    return SimpleNamespace(info=lambda *a: events.append(['info', *map(str, a)]),
        debug=lambda *a: None, set_level=lambda *a: events.append(['log_level', *a]))


def position(symbol, amount=100, price=100., cost=100.):
    return SimpleNamespace(security=symbol, total_amount=amount, price=price, avg_cost=cost, value=amount*price)


class ZeroPositions(dict):
    def __missing__(self, key): return position(key, 0)


class Harness:
    """Controlled API feedback, never fees, NAV, or an executable portfolio simulation."""
    def __init__(self, directory, number, mode='reject', held=(), mapping='plain'):
        self.orders = []; self.events = []; self.mode = mode; self.number = number
        positions = {} if mapping == 'plain' else ZeroPositions()
        positions.update({s: position(s) for s in held})
        self.context = SimpleNamespace(current_dt=datetime(2020, 1, 2, 9, 30),
            portfolio=SimpleNamespace(positions=positions, cash=1000.))
        self.g = SimpleNamespace(stock_pool=list(POOL[number]), stock_num=1, momentum_day=20, ref_stock='000300.XSHG',
            N=18, M=600, score_threshold=.7, mean_day=30, mean_diff_day=2, rsrs_score_hisitory=[],
            checktime='13:00', checktime2='13:50', hold='', slope_series=list(np.linspace(.5, 1.5, 599)))
        self.ns = literal(directory, number, {'np': np, 'g': self.g, 'log': log_stub(self.events),
            'OrderStatus': SimpleNamespace(held='held'), 'order_target_value': self.order,
            'set_benchmark': lambda s: self.events.append(['benchmark', s]),
            'set_option': lambda k,v: self.events.append(['option', k,v]),
            'FixedSlippage': lambda v: {'FixedSlippage':v}, 'set_slippage': lambda v: self.events.append(['slippage',v]),
            'OrderCost': lambda **kw: kw, 'set_order_cost': lambda c,**kw: self.events.append(['cost',c,kw]),
            'run_daily': lambda f,**kw: self.events.append(['schedule',f.__name__,kw]),
            'record': lambda **kw: self.events.append(['record',kw])})

    def order(self, symbol, value):
        require(isinstance(symbol, (str, float, int, np.number)), 'Unexpected diagnostic order operand')
        self.orders.append({'security': str(symbol) if isinstance(symbol,str) else float(symbol), 'value': float(value)})
        if self.mode == 'reject': return None
        if self.mode == 'zero': return SimpleNamespace(filled=0, amount=100, status='canceled')
        filled = 30 if self.mode == 'partial' else 100
        if self.mode != 'deferred' and value == 0:
            p = self.context.portfolio.positions[symbol]
            if filled == p.total_amount: del self.context.portfolio.positions[symbol]
            else: p.total_amount -= filled
        elif value != 0:
            self.context.portfolio.positions[symbol] = position(symbol, filled)
        return SimpleNamespace(filled=filled, amount=100, status='held' if self.mode == 'full' else 'canceled')


CASES = ('initialize', 'rank_datetime', 'rank_range', 'rank_controlled_positional', 'rank_equal_scores',
    'mean_11', 'mean_12_missing', 'mean_12_injected', 'BUY_plain_empty', 'BUY_reject_numeric',
    'BUY_partial_masks_numeric', 'BUY_full_masks_numeric', 'SELL_mutating', 'rank_error_before_SELL',
    'KEEP_rotates', 'partial_sell_blocks_rotation', 'deferred_sell_blocks_rotation',
    'stop_exact_minus90', 'stop_above_minus90', 'stop_zero_cost', 'time_0929', 'time_0931',
    'positive_below_mean', 'positive_equal_mean', 'score_zero', 'score_minus07', 'score_above_minus07',
    'score_plus07', 'score_above_plus07', 'score_below_minus07', 'hold_datetime', 'hold_range',
    'hold_below_injected_label', 'hold_equal_injected_label', 'hold_wrong_time', 'hold_two_mutating',
    'open_reject', 'open_zero', 'open_partial', 'close_partial', 'close_full', 'close_full_canceled')


def diagnostic_case(directory, number, case):
    require(case in CASES, 'Unknown diagnostic case')
    a,b = POOL[number][0], POOL[number][2]
    mode = ('partial' if case in ('BUY_partial_masks_numeric','partial_sell_blocks_rotation','open_partial','close_partial')
        else 'full' if case in ('BUY_full_masks_numeric','SELL_mutating','KEEP_rotates','stop_exact_minus90',
            'hold_two_mutating','close_full','close_full_canceled') else 'deferred' if case == 'deferred_sell_blocks_rotation'
        else 'zero' if case == 'open_zero' else 'reject')
    held = (a,) if case.startswith(('stop_','hold_','close_')) or case in ('SELL_mutating','rank_error_before_SELL') else (
        (b,) if case in ('KEEP_rotates','partial_sell_blocks_rotation','deferred_sell_blocks_rotation') else ())
    if case == 'hold_two_mutating': held = (a,b)
    h = Harness(directory, number, mode, held, 'virtual' if case.startswith('BUY_') and case != 'BUY_plain_empty' else 'plain')
    ns,g,ctx = h.ns,h.g,h.context; result = None; error = None; output = StringIO()
    def history(symbol,count,unit,fields):
        h.events.append(['history',symbol,count,unit,list(fields)])
        if unit == '60m':
            closes = [100.]*21 + ([100.] if case == 'hold_equal_injected_label' else [50.])
            index = (pd.date_range('2020-01-01',periods=count,freq='h') if case == 'hold_datetime' else
                list(range(-count,0)) if case in ('hold_below_injected_label','hold_equal_injected_label','hold_two_mutating') else range(count))
            return pd.DataFrame({'close':closes},index=index)
        index = (pd.date_range('2020-01-01',periods=count) if case in ('rank_datetime','rank_error_before_SELL') else
            range(-90,20) if number == 2 and case in ('rank_controlled_positional','rank_equal_scores') else range(count))
        closes = np.linspace(100.,120.,count) if case != 'rank_equal_scores' else np.full(count,100.)
        return pd.DataFrame({'close':closes, 'low':closes-1., 'high':closes+1., 'volume':1000.},index=index)[fields]
    ns['attribute_history'] = history
    try:
        with redirect_stdout(output):
            if case == 'initialize':
                ns['initial_slope_series'] = lambda: list(range(600)); ns['initialize'](ctx)
                result = {'seed_length':len(g.slope_series), 'pool':g.stock_pool, 'hold': getattr(g,'hold',None)}
            elif case.startswith('rank_') and case != 'rank_error_before_SELL': result = ns['get_rank'](ctx, ['ignored'])
            elif case.startswith('mean_'):
                if number != 2: return None
                g.rsrs_score_hisitory = list(range(10 if case == 'mean_11' else 11))
                if case == 'mean_12_injected': ns['mean'] = np.mean
                result = ns['get_rsrs_score_mean'](11.)
            elif case.startswith(('positive_','score_')):
                scores = {'positive_below_mean':.2,'positive_equal_mean':.2,'score_zero':0.,'score_minus07':-.7,
                    'score_above_minus07':np.nextafter(-.7,math.inf),'score_plus07':.7,
                    'score_above_plus07':np.nextafter(.7,math.inf),'score_below_minus07':np.nextafter(-.7,-math.inf)}
                ns['get_ols'] = lambda *args:(0.,1.,1.); ns['get_zscore'] = lambda *args:scores[case]
                ns['get_rsrs_score_mean'] = lambda *args:.5 if case == 'positive_below_mean' else .2
                result = ns['get_timing_signal'](ctx,'ignored')
            elif case.startswith('stop_'):
                ctx.portfolio.positions[a].price = 10. if case == 'stop_exact_minus90' else 10.1
                if case == 'stop_zero_cost': ctx.portfolio.positions[a].avg_cost = 0.
                result = ns['check_lose'](ctx)
            elif case.startswith('hold_'):
                if number != 2: return None
                ctx.current_dt = datetime(2020,1,2,12 if case == 'hold_wrong_time' else 13,0)
                result = ns['hold_check'](ctx)
            elif case.startswith('open_'):
                result = ns['open_position'](a,1000.)
            elif case.startswith('close_'):
                if case == 'close_full_canceled': h.mode = 'canceled_full'
                result = ns['close_position'](ctx.portfolio.positions[a])
            else:
                if case == 'rank_error_before_SELL':
                    ns['get_timing_signal'] = lambda *args:h.events.append(['timing']) or 'SELL'
                else:
                    ns['get_rank'] = lambda *args:h.events.append(['rank']) or [a,.123]
                    ns['get_timing_signal'] = lambda *args:h.events.append(['timing']) or ('SELL' if case == 'SELL_mutating' else 'KEEP' if case == 'KEEP_rotates' else 'BUY')
                if case.startswith('time_'): ctx.current_dt = datetime(2020,1,2,9,29 if case == 'time_0929' else 31)
                result = ns['my_trade'](ctx)
    except (KeyError, NameError, RuntimeError, ZeroDivisionError, TypeError) as exc:
        error = {'type':type(exc).__name__, 'message':str(exc)}
    if isinstance(result,np.generic): result = result.item()
    if isinstance(result,list): result = [x.item() if isinstance(x,np.generic) else x for x in result]
    return {'source':number,'case':case,'result':result,'error':error,'orders':h.orders,'events':h.events,
        'held':list(ctx.portfolio.positions),'hold_state':g.hold,'score_history_length':len(g.rsrs_score_hisitory),
        'slope_history_length':len(g.slope_series),'stdout':output.getvalue(),'synthetic_not_backtest':True}


def diagnostics(directory):
    rows = [diagnostic_case(directory,n,c) for n in range(3) for c in CASES]
    rows = [r for r in rows if r is not None]
    return {'cases':rows,'count':len(rows),'controlled_namespace_only':True,
        'jqdata_mean_export_proven':False,'platform_position_refresh_semantics_proven':False,'new_trading_backtests':0}


def scalar_ols(x,y):
    x,y = list(map(float,x)),list(map(float,y)); mx,my = math.fsum(x)/len(x),math.fsum(y)/len(y)
    variance = math.fsum((v-mx)**2 for v in x)
    slope = math.fsum((a-mx)*(b-my) for a,b in zip(x,y,strict=True))/variance
    intercept = my-slope*mx
    r2 = 1-math.fsum((b-slope*a-intercept)**2 for a,b in zip(x,y,strict=True))/math.fsum((b-my)**2 for b in y)
    return intercept,slope,r2


def scalar_z(values):
    mean = math.fsum(map(float,values))/len(values)
    return (values[-1]-mean)/math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))


def close_enough(a,b,label):
    require(math.isclose(float(a),float(b),rel_tol=2e-9,abs_tol=2e-10),f'{label} independent formula differs')


def signal(score, number, history):
    if number != 2: return 'BUY' if score > .7 else 'SELL' if score < -.7 else 'KEEP'
    average = math.fsum(history[-12:])/12 if len(history) >= 12 else 0.
    return 'SELL' if score > 0 and score < average else 'BUY' if score > -.7 else 'SELL'


def rsrs(directory,number):
    frame = pd.read_parquet(directory/'index-input.parquet'); frame['date'] = frame.date.astype(str)
    values = frame[['low','high']].to_numpy(float); cursor = [618]
    def history(symbol,count,unit,fields):
        require(symbol == '000300.XSHG' and unit == '1d' and count in (18,35,618),'RSRS query differs')
        return frame.iloc[cursor[0]-count:cursor[0]][fields]
    g = SimpleNamespace(N=18,M=600,ref_stock='000300.XSHG',score_threshold=.7,mean_day=30,
        mean_diff_day=2,rsrs_score_hisitory=[])
    ns = literal(directory,number,{'np':np,'g':g,'attribute_history':history,'mean':np.mean,
        'log':SimpleNamespace(info=lambda *a:None),'record':lambda **kw:None})
    seeds = ns['initial_slope_series'](); independent = [scalar_ols(*values[i:i+18].T)[1] for i in range(600)]
    require(len(seeds) == len(independent) == 600,'Seed count differs')
    for a,b in zip(seeds,independent,strict=True): close_enough(a,b,'seed')
    g.slope_series = seeds if number == 2 else seeds[:-1]
    independent = independent if number == 2 else independent[:-1]
    scores = []; rows = []; maximum = 0.
    for end in range(618,len(frame)):
        cursor[0] = end; actual_signal = ns['get_timing_signal'](None,'ignored')
        a_intercept,a_slope,a_r2 = ns['get_ols'](*values[end-18:end].T)
        b_intercept,b_slope,b_r2 = scalar_ols(*values[end-18:end].T); independent.append(b_slope)
        actual_z = ns['get_zscore'](g.slope_series[-600:]); expected_z = scalar_z(independent[-600:])
        score,expected_score = actual_z*a_r2,expected_z*b_r2; scores.append(expected_score)
        for a,b in ((a_intercept,b_intercept),(a_slope,b_slope),(a_r2,b_r2),(actual_z,expected_z),(score,expected_score)):
            close_enough(a,b,'RSRS'); maximum = max(maximum,abs(float(a)-float(b)))
        require(actual_signal == signal(expected_score,number,scores),'RSRS strict state differs')
        mean = float(np.mean(g.rsrs_score_hisitory[-12:])) if number == 2 and len(scores)>=12 else 0.
        if number == 2: close_enough(mean,math.fsum(scores[-12:])/12 if len(scores)>=12 else 0.,'12 mean')
        rows.append({'decision_date':frame.date.iloc[end],'last_input_date':frame.date.iloc[end-1],
            'slope':a_slope,'r2':a_r2,'zscore':actual_z,'score':score,'mean12':mean,
            'independent_score':expected_score,'signal':actual_signal})
    result = pd.DataFrame(rows)
    return result, {'rows':len(rows),'first':rows[0]['decision_date'],'last':rows[-1]['decision_date'],
        'seed_slopes':600 if number == 2 else 599,'maximum_independent_error':maximum,
        'signals':dict(Counter(result.signal)),'controlled_mean_injection':number == 2,
        'original_platform_mean_available_proven':False,'not_a_trading_backtest':True}


def momentum(directory):
    inputs = read(directory/'candidate-inputs.json')['funds']; rows = []; profiles = []
    for number in (0,2):
        count = 110 if number == 2 else 20
        for symbol in POOL[number]:
            if symbol not in inputs: continue
            frame,_ = prior.native_frame(inputs[symbol]); closes = frame.close.to_numpy(float); cursor = [count]
            def history(stock,n,unit,fields):
                require(stock == symbol and n == count and unit == '1d' and fields == ['close'],'Momentum query differs')
                # Explicit index-label injection permits original literal arithmetic, not platform compatibility.
                return pd.DataFrame({'close':closes[cursor[0]-count:cursor[0]]},
                    index=range(-90,20) if number == 2 else range(20))
            ns = literal(directory,number,{'np':np,'g':SimpleNamespace(stock_pool=[symbol],momentum_day=20),
                'attribute_history':history,'log':SimpleNamespace(info=lambda *a:None)})
            start = len(rows); maximum = 0.
            for end in range(count,len(frame)+1):
                cursor[0] = end; selected,score = ns['get_rank'](None,[]); require(selected == symbol,'Single operand identity changed')
                window = list(map(float,closes[end-count:end]))
                if number == 2:
                    y = [window[i]/(math.fsum(window[i-89:i+1])/90) for i in range(90,110)]
                else: y = window
                normalized = [v/y[0] for v in y]
                expected = math.fsum((i-9.5)*v for i,v in enumerate(normalized))/665.
                close_enough(score,expected,'native slope'); maximum=max(maximum,abs(float(score)-expected))
                rows.append({'variant':'v2_bias90' if number == 2 else 'magic3_return','symbol':symbol,
                    'last_input_date':frame.date.iloc[end-1],'window_first':frame.date.iloc[end-count],
                    'score':float(score),'independent_score':expected,'input_rows':count})
            profiles.append({'variant':number,'symbol':symbol,'windows':len(rows)-start,
                'first_last_input':frame.date.iloc[count-1],'last_last_input':frame.date.iloc[-1],
                'maximum_independent_error':maximum})
    return pd.DataFrame(rows), {'windows':len(rows),'profiles':profiles,'missing_pool':['159928.XSHE'],
        'full_original_pool_ranking_rows':0,'native_unadjusted_reference':True,
        'history_index_labels_controlled':True,'original_adjusted_price_equivalence_proven':False,
        'last_input_has_no_proven_next_execution_session':True,'not_a_trading_backtest':True}


def compute(directory):
    a,summary_a=rsrs(directory,0); b,summary_b=rsrs(directory,2); native,summary_native=momentum(directory)
    return {'diagnostics.json':diagnostics(directory),'rsrs-magic3.parquet':a,'rsrs-v2-reference.parquet':b,
        'native-momentum-reference.parquet':native,'component-research.json':{
            'magic3_RSRS':summary_a,'v2_RSRS_controlled_mean_reference':summary_b,'native_momentum':summary_native,
            'new_trading_backtests':0,'formal_source_research_acceptance':False,'original_strategy_complete':False}}


def study():
    validate(DIRECTORY)
    require(not any((DIRECTORY/n).exists() for n in OUTPUTS),'Study output already exists')
    with network_blocked(): outputs = compute(DIRECTORY)
    for name,value in outputs.items():
        if isinstance(value,pd.DataFrame):
            with (DIRECTORY/name).open('xb') as out: value.to_parquet(out,index=False)
        else: save(DIRECTORY/name,value)
    save(DIRECTORY/'output-binding.json',{name:file_sha(DIRECTORY/name) for name in OUTPUTS})
    return {'status':'computed',**outputs['component-research.json'],'diagnostic_cases':outputs['diagnostics.json']['count']}


def replay():
    hashes=validate(DIRECTORY); bindings=read(DIRECTORY/'output-binding.json')
    require(set(bindings)==set(OUTPUTS),'Replay inventory changed')
    for name,sha in bindings.items(): require(file_sha(DIRECTORY/name)==sha,'Archived output changed')
    with network_blocked(): outputs=compute(DIRECTORY)
    for name,value in outputs.items():
        if isinstance(value,pd.DataFrame): pd.testing.assert_frame_equal(value,pd.read_parquet(DIRECTORY/name),check_exact=True)
        else: require(value==read(DIRECTORY/name),'Replay JSON differs: '+name)
    result={'status':'match','differences':0,'network_forbidden':True,'fresh_process_required_by_cli':True,
        'implementation_sha256':hashes,'files_sha256':bindings,'new_trading_backtests':0,
        'formal_source_research_acceptance':False}
    save(DIRECTORY/'offline-verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','study','replay','protect','checkpoint'))
    args=parser.parse_args()
    if args.action=='prepare': result=prepare()
    elif args.action=='study': result=study()
    elif args.action=='replay': result=replay()
    elif args.action=='protect': result=protect('data',DIRECTORY)
    else:
        validate(DIRECTORY)
        result=archive('data','Batch95 long magic3/v2 arithmetic and original callback stage, no formal95 acceptance/trades')
        save(DIRECTORY/'progress-stage.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
