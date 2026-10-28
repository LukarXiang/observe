"""Original C-VIX/option arithmetic with frozen inputs; never a trading backtest."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
from decimal import Decimal, ROUND_HALF_EVEN
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import talib

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-original-options-v1')
SOURCE_PATHS = (
    '2021年度精选策略/64.【复现】C-VIX 中国版 VIX 编制手册.txt',
    '2022年度精选策略/18.期权突破策略研究 v1.txt')
SOURCE_SHA = ('5483ce9fea22822a8ca84e0f004064c53b384f54bdd0a2a51eee5d162761f983',
              '6259fb771cf0bc1a857d10806d4282241053409d5079c5478862a221f6a63e1d')
PARENT = Path('docs/handoff/2026-10-11-batch98-final-review.json')
PARENT_SHA = 'c6861c7afb7599f27a24223b7f41ac82bc6926c40f0d04ca425d80d24ee25450'
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
ETF_DIRECTORY = Path('data/staging/strategies-batch74/20261008-FOF-two-eight')
ETF_RESULTS_SHA = 'd258e2fb847f2441c1f646e15aba2fa98360911e1e63dcefec25729b67b8ac68'
ETF_SHA = '17897aa46f5d66ba2878ad90f537ddd6c2d3d2d1baaf2b321cdc333d6b4237f2'
FILES = ('scripts/research_strategy_batch99.py', 'tests/unit/test_batch99_research.py',
         'scripts/recover_strategy_batch94_reference.py', 'scripts/review_strategy_batch18.py',
         'scripts/run_strategy_batch4.py', 'scripts/archive_strategy_progress.py',
         'scripts/verify_strategy_batch3.py', 'src/observe/data/store.py',
         'src/observe/data/locks.py', 'src/observe/runs.py', 'src/observe/strategy_catalog.py',
         'pyproject.toml', 'uv.lock')
OUTPUTS = ('diagnostics.json', 'ETF-range-components.parquet', 'summary.json')
METHODS = {'__init__', 'GetCBOE_VIX', 'GetOptContractBasicInfo', 'GetOptDailyPrice',
           'PrepareData', 'GetShiBor', 'filter_contract', '_check_fields',
           'cal_forward_price', 'cal_mid_price', '_nearest_k', 'cal_k_diff',
           'cal_vix_sub', 'cal_vix', 'cal_skew', 'cal_epsilon', 'cal_moments_sub'}


def definitions(source, number, namespace=None):
    """SHA-locked definitions only; imports, profiling and credential setup are omitted."""
    require(number in (0, 1) and file_sha(source) == SOURCE_SHA[number], 'Source SHA differs')
    tree = ast.parse(read_source(source)[0])
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    require(len(nodes) == (10 if number == 0 else 5), 'Definition inventory differs')
    selected = [n for n in nodes if n.name != 'set_variables']
    for n in selected:
        if isinstance(n, ast.ClassDef):
            require(n.name == 'CH_VIX' and not n.decorator_list and not n.keywords and
                    not n.type_params and len(n.bases) == 1 and
                    isinstance(n.bases[0], ast.Name) and n.bases[0].id == 'object',
                    'Unreviewed class declaration')
            require(all(isinstance(f, ast.FunctionDef) for f in n.body) and
                    {f.name for f in n.body} == METHODS, 'Unreviewed class body')
        for f in n.body if isinstance(n, ast.ClassDef) else [n]:
            require(not f.type_params and all(isinstance(d, ast.Name) and
                    d.id == 'staticmethod' for d in f.decorator_list), 'Unreviewed decorator')
            args = (*f.args.posonlyargs, *f.args.args, *f.args.kwonlyargs,
                    *[a for a in (f.args.vararg, f.args.kwarg) if a])
            for annotation in [f.returns, *(a.annotation for a in args)]:
                if annotation is None: continue
                require((isinstance(annotation, ast.Name) and annotation.id in
                         {'int', 'float', 'bool', 'str', 'list', 'dict', 'tuple'}) or
                        (isinstance(annotation, ast.Attribute) and
                         isinstance(annotation.value, ast.Name) and annotation.value.id == 'pd' and
                         annotation.attr in {'DataFrame', 'Series'}), 'Unreviewed annotation')
            for default in (*f.args.defaults, *[d for d in f.args.kw_defaults if d is not None]):
                ast.literal_eval(default)
    ns = {'pd': pd, 'np': np, 'talib': talib, 'interp1d': interp1d, 'dt': dt,
          'timedelta': dt.timedelta, 'g': SimpleNamespace(), 'record': lambda **kw: None}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def source_namespace(number, **kwargs):
    return definitions(DIRECTORY/'sources'/f'{number}.source', number, kwargs)


def canonical(value):
    """Preserve values and container shape; reject unreviewed fallback representations."""
    if isinstance(value, pd.DataFrame): return canonical(value.reset_index().to_dict('records'))
    if isinstance(value, pd.Series): return canonical(value.tolist())
    if isinstance(value, (np.ndarray, pd.api.extensions.ExtensionArray)): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)): return value.isoformat()
    if isinstance(value, complex): return {'complex': [canonical(value.real), canonical(value.imag)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else 'inf' if value > 0 else '-inf'}
    if value is None or isinstance(value, (str, bool, int, float)): return value
    raise TypeError('Unreviewed archive type: '+type(value).__name__)


def inventory():
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA and
            state['tables'] == store.published()['tables'], 'Active snapshot differs')
    path = ETF_DIRECTORY/'sina-fallback/results.json'
    require(file_sha(path) == ETF_RESULTS_SHA and
            read(ETF_DIRECTORY/'checked-state.json')['evidence_sha256']['sina-fallback/results.json'] == ETF_RESULTS_SHA,
            'Candidate registry differs')
    rows = [r for r in read(path)['results'] if r['case'] == 'fund-510050']
    require(len(rows) == 1, 'Candidate identity differs'); row = rows[0]
    require(row['status'] == 'success' and row['sha256'] == ETF_SHA and
            file_sha(row['file']) == ETF_SHA, 'Candidate bytes differ')
    for response in row['wire_responses']:
        require(file_sha(response['file']) == response['sha256'], 'Wire evidence differs')
    frame = pd.read_parquet(row['file'])
    require(len(frame) == row['rows'] == 5254, 'Candidate rows differ')
    rates = Path('data/staging/strategies-batch25/20261006-macro-open-fund/probes/shibor/result.json')
    return {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
            'published_batch': store.published()['batch_id'], 'registered_tables': sorted(state['tables']),
            'required_option_tables_registered': sorted(set(state['tables']) &
                {'option_contracts', 'option_daily', 'option_minute', 'shibor'}),
            'ETF_candidate': {**row, 'first': str(frame.date.min()), 'last': str(frame.date.max()),
                              'registry_file': str(path), 'registry_sha256': ETF_RESULTS_SHA},
            'old_shibor_probe': {'file': str(rates), 'sha256': file_sha(rates),
                                 'result': read(rates)},
            'gates': ['Historical full option chain/expiry/adjusted-contract identity and prices missing',
                      'Original eight-tenor SHIBOR history not recovered',
                      'Historical 510050 and both option one-minute closes missing',
                      'ETF candidate is not admitted price/state/event/fee input',
                      'Book option margin/short/expiry/settlement execution not admitted'],
            'strict_usable': False, 'new_trading_backtests': 0}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable directory exists')
    require(file_sha(PARENT) == PARENT_SHA and read(PARENT)['status'] == 'ok' and
            read(PARENT)['progress']['manually_reviewed'] == 282, 'Parent acceptance differs')
    for name, digest in read(PARENT)['evidence_sha256'].items():
        require(file_sha(name) == digest, 'Parent evidence differs: '+name)
    reviews = []
    for number, name in enumerate(SOURCE_PATHS):
        source = Path('repo/量化策略源代码')/name; text = read_source(source)[0]
        definitions(source, number)
        reviews.append({'path': name, 'sha256': SOURCE_SHA[number], 'lines': len(text.splitlines()),
            'definitions': [{'name': n.name, 'kind': type(n).__name__, 'first_line': n.lineno,
                             'last_line': n.end_lineno,
                             'methods': [f.name for f in n.body if isinstance(f, ast.FunctionDef)]}
                            for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))],
            'top_level_executed': False, 'credential_setup_executed': False})
    deps = inventory(); etf = pd.read_parquet(deps['ETF_candidate']['file'])
    etf['date'] = pd.to_datetime(etf.date).dt.strftime('%Y-%m-%d')
    etf = etf[etf.date.le('2026-09-29')][['date', 'open', 'high', 'low', 'close']].reset_index(drop=True)
    require(etf.date.is_monotonic_increasing and not etf.date.duplicated().any() and
            np.isfinite(etf[['open', 'high', 'low', 'close']]).all().all(), 'Candidate OHLC invalid')
    require((etf.low <= etf[['open', 'close']].min(axis=1)).all() and
            (etf.high >= etf[['open', 'close']].max(axis=1)).all() and (etf.low > 0).all(),
            'Candidate OHLC constraints differ')
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch99 full source/dependency arithmetic started; formal acceptance remains282')
        (DIRECTORY/'sources').mkdir()
        for number, name in enumerate(SOURCE_PATHS):
            shutil.copyfile(Path('repo/量化策略源代码')/name, DIRECTORY/'sources'/f'{number}.source')
        etf.to_parquet(DIRECTORY/'ETF-input.parquet', index=False)
        for name in FILES:
            target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        save(DIRECTORY/'prepared.json', {'parent': str(PARENT), 'parent_sha256': PARENT_SHA,
             'source_inventory': reviews, 'operand_sha256': file_sha(DIRECTORY/'ETF-input.parquet'),
             'implementation_sha256': {n: file_sha(n) for n in FILES}, 'candidate': deps['ETF_candidate'],
             'candidate_rows': len(etf), 'future_rows_excluded': 5254-len(etf), 'not_a_backtest': True})
        save(DIRECTORY/'dependencies.json', deps)
    return {'status': 'prepared', 'directory': str(DIRECTORY), 'rows': len(etf)}


def chain():
    return pd.DataFrame([{'date': dt.date(2018, 7, 20), 'exercise_price': k,
                         'maturity': days/365, 'contract_type': kind, 'close': price}
                        for days in (5, 40) for k, c, p in ((2., .35, .05), (2.5, .2, .1), (3., .05, .25))
                        for kind, price in (('call', c), ('put', p))])


def scalar_vix(frame, forward, rate, maturity, k0):
    terms = [float(r.diff_k)*math.exp(rate*maturity)*float(r.mid_p)/float(r.exercise_price)**2
             for r in frame.itertuples(index=False)]
    return 2*math.fsum(terms)/maturity-(forward/k0-1)**2/maturity


def scalar_skew(frame, forward, rate, maturity, k0):
    log = math.log(k0/forward)
    e1 = -(1+math.log(forward/k0)-forward/k0)
    e2 = 2*log*(k0/forward-1)+log*log/2
    e3 = 3*log*log*(log/3-1+forward/k0)
    terms = ([], [], [])
    for r in frame.itertuples(index=False):
        base = float(r.mid_p)*float(r.diff_k)/float(r.exercise_price)**2
        x = math.log(float(r.exercise_price)/forward)
        terms[0].append(-base); terms[1].append(base*2*(1-x)); terms[2].append(base*3*(2*x-x*x))
    p1, p2, p3 = [math.exp(rate*maturity)*math.fsum(a)+e for a, e in zip(terms, (e1, e2, e3))]
    return (p3-3*p1*p2+2*p1**3)/(p2-p1**2)**1.5


def paper_case(namespace, prices, flag=('Buy', 'Sell'), codes=None):
    """Call original paper function on fixtures; this is not an execution engine."""
    days = [dt.date(2018, 7, 20)+dt.timedelta(i) for i in range(len(prices))]
    namespace['get_trade_days'] = lambda *args: days
    namespace['get_option_code'] = lambda date: codes or (['10001355.XSHG']*3, ['10001356.XSHG']*3)
    namespace['Price_Range'] = lambda *args: (3., 2.)
    namespace['get_price'] = lambda code, start, end, **kw: {'close': prices[days.index(start)]}
    return namespace['option_strategy'](flag=list(flag))


def contract_case(namespace, frame, opening=2.5):
    """Reviewed query fixture returns specified pre-open rows, with no live query."""
    class Field:
        def __eq__(self, other): return self
    class Query:
        def filter(self, *args): return self
    namespace['jy'] = SimpleNamespace(Opt_DailyPreOpen=SimpleNamespace(
        TradingDate=Field(), ULAName=Field()), run_query=lambda query: frame.copy())
    namespace['query'] = lambda *args: Query()
    namespace['get_price'] = lambda *args, **kwargs: pd.DataFrame({'open': [opening]})
    return namespace['get_option_code'](dt.date(2018, 7, 20))


def diagnostics(namespace):
    ns, option = namespace; obj = ns['CH_VIX']('510050.XSHG', '', '')
    cases = []
    def case(name, callback, scope='synthetic_original_definition'):
        try:
            with redirect_stdout(StringIO()), np.errstate(all='ignore'):
                value = canonical(callback())
            row = {'case': name, 'scope': scope, 'value': value, 'error': None}
        except Exception as exc:
            row = {'case': name, 'scope': scope, 'value': None, 'error': type(exc).__name__}
        require(json.loads(json.dumps(row, allow_nan=False)) == row, 'Actual JSON shape differs')
        cases.append(row)
    maturity, near, far = obj.filter_contract(chain())
    case('five_day_expiry_included', lambda: maturity)
    case('less_than_two_expiries', lambda: obj.filter_contract(chain().query('maturity < .1')))
    case('missing_put_field', lambda: obj._check_fields(pd.DataFrame({'call': [.1]}, index=[2.])))
    duplicate = pd.concat([chain(), chain().iloc[[0]].assign(close=.55)], ignore_index=True)
    case('duplicate_strike_type_mean', lambda: obj.filter_contract(duplicate)[1])
    case('K0_is_min_diff_not_largest_strike_below_F', lambda: obj._nearest_k(
        pd.DataFrame({'diff': [.01, .2, .3]}, index=[2., 2.5, 3.]), 2.9))
    case('F_below_all_strikes_falls_back_all', lambda: obj._nearest_k(near, 1.))
    case('single_strike_deltaK', lambda: obj.cal_k_diff(near.iloc[:1]))
    arithmetic = []
    for frame, days in ((near, 5), (far, 40)):
        forward = obj.cal_forward_price(days/365, .03, frame)
        k0 = obj._nearest_k(frame, forward)
        term = pd.concat([obj.cal_mid_price(days/365, frame, forward), obj.cal_k_diff(frame)], axis=1).reset_index()
        sigma = obj.cal_vix_sub(term, forward, .03, days/365, k0)
        skew = obj.cal_moments_sub(term, days/365, .03, forward, k0)
        reference = scalar_vix(term, float(forward), .03, days/365, float(k0))
        skew_reference = scalar_skew(term, float(forward), .03, days/365, float(k0))
        require(abs(sigma-reference) < 1e-12 and abs(skew-skew_reference) < 1e-10,
                'Independent VIX/SKEW differs')
        arithmetic.append({'days': days, 'F': float(forward), 'K0': float(k0), 'sigma': float(sigma),
                           'skew': float(skew), 'sigma_error': abs(sigma-reference),
                           'skew_error': abs(skew-skew_reference)})
    case('scalar_vix_skew_comparison', lambda: arithmetic)
    near_term = pd.concat([obj.cal_mid_price(5/365, near, arithmetic[0]['F']),
                           obj.cal_k_diff(near)], axis=1).reset_index()
    far_term = pd.concat([obj.cal_mid_price(40/365, far, arithmetic[1]['F']),
                          obj.cal_k_diff(far)], axis=1).reset_index()
    def combined(days):
        a, b = days; w = (b-30)/(b-a)
        args = (near_term, arithmetic[0]['F'], .03, a/365, arithmetic[0]['K0'],
                far_term, arithmetic[1]['F'], .03, b/365, arithmetic[1]['K0'])
        value, skew = obj.cal_vix(*args), obj.cal_skew(*args)
        variance = (a/365*scalar_vix(near_term, args[1], .03, a/365, args[4])*w+
                    b/365*scalar_vix(far_term, args[6], .03, b/365, args[9])*(1-w))*365/30
        oracle = 100*math.sqrt(variance) if variance >= 0 else float('nan')
        skew_oracle = 100-10*(w*scalar_skew(near_term, args[1], .03, a/365, args[4])+
                              (1-w)*scalar_skew(far_term, args[6], .03, b/365, args[9]))
        require((math.isnan(value) and math.isnan(oracle) or abs(value-oracle) < 1e-10) and
                abs(skew-skew_oracle) < 1e-10, 'Combined VIX/SKEW differs')
        return {'days': days, 'weight': w, 'VIX': value, 'SKEW': skew,
                'VIX_error': abs(value-oracle), 'SKEW_error': abs(skew-skew_oracle)}
    case('combined_vix_skew_original_weights', lambda: combined((5, 40)))
    case('both_expiries_after30_weight_unclamped', lambda: combined((55, 80)))
    term = pd.DataFrame({'exercise_price': [2.], 'diff_k': [.1], 'mid_p': [0.]})
    case('negative_variance_returns_nan', lambda: obj.cal_vix(term, 5., 0., .1, 2., term, 5., 0., .2, 2.))
    case('equal_maturity_division_by_zero', lambda: obj.cal_vix(term, 2., 0., .1, 2., term, 2., 0., .1, 2.))
    case('zero_forward_epsilon_nonfinite', lambda: obj.cal_epsilon(0., 2.))
    calls = []
    def shibor(**kw):
        calls.append(kw)
        return pd.DataFrame({'date': ['20180720'], **{str(i): [3.] for i in range(8)}})
    ns['g'].pro = SimpleNamespace(shibor=shibor)
    ns['get_trade_days'] = lambda *args: pd.date_range('2005-01-01', periods=2001)
    case('shibor_2001days_unbound_local', lambda: obj.GetShiBor(dt.date(2005, 1, 1), dt.date(2018, 7, 20)))
    case('shibor_first_page_is_one_day', lambda: calls.copy())
    ns['get_trade_days'] = lambda *args: [dt.date(2018, 7, 20)]
    case('shibor_string_date_attribute_error', lambda: obj.GetShiBor('2018-07-20', '2018-07-20'))
    case('shibor_cubic_flat_percent_to_fraction', lambda: obj.GetShiBor(dt.date(2018, 7, 20), dt.date(2018, 7, 20)))
    # A reviewed fixture injects data, never replaces missing historical inputs.
    obj.PrepareData = lambda: setattr(obj, 'OptData', chain())
    obj.GetShiBor = lambda *args: pd.DataFrame(.03, index=[dt.date(2018, 7, 20)], columns=range(1, 361))
    case('GetCBOE_VIX_pandas3_pad_compatibility', obj.GetCBOE_VIX)
    case('EMA_equal_buys', lambda: ns['GetSingal'](pd.Series([20.]*15), 5, 10))
    case('EMA_rising_sells', lambda: ns['GetSingal'](pd.Series(np.arange(15., 30.)), 5, 10))
    case('EMA_falling_buys', lambda: ns['GetSingal'](pd.Series(np.arange(30., 15., -1)), 5, 10))
    case('EMA_short_window_false', lambda: ns['GetSingal'](pd.Series([20.]*4), 5, 10))
    frame = pd.DataFrame({'StrikePrice': [2., 2.5, 3.], 'ContractCode': ['low', 'mid', 'high']})
    case('call_put_order', lambda: [option['option_code'](frame, 2.5, kind) for kind in ('c', 'p')])
    case('edge_strike_only_two', lambda: option['option_code'](frame, 2., 'c'))
    case('equidistant_prefers_lower_strike', lambda: option['option_code'](frame, 2.75, 'c'))
    contracts = pd.DataFrame([{'ContractCode': str(10000000+i+j*10),
        'TradingCode': '510050'+kind+'1807M02500', 'StrikePrice': k,
        'ExerciseDate': dt.date(2018, 7, 25)}
        for j, kind in enumerate(('C', 'P')) for i, k in enumerate((2., 2.5, 3.))])
    case('preopen_standard_contracts', lambda: contract_case(option, contracts))
    case('preopen_empty_None_strings', lambda: contract_case(option, contracts.iloc[:0]))
    case('preopen_short_TradingCode', lambda: contract_case(option, contracts.assign(TradingCode='short')))
    adjusted = contracts.assign(TradingCode=contracts.TradingCode.str.replace('M', 'A'))
    later = contracts.assign(ExerciseDate=dt.date(2018, 8, 22))
    case('preopen_first_expiry_adjusted_only_no_next_fallback', lambda: contract_case(
        option, pd.concat([adjusted, later], ignore_index=True)))
    case('Indicators_sqrt_not_imported', lambda: option['Indicators']([1., 1.1, 1.05]))
    up = pd.DataFrame([[3., .02, .03], [3.1, .02, .03], [1.9, .03, .01]])
    down = pd.DataFrame([[2., .02, .03], [1.9, .02, .03], [3.1, .03, .01]])
    for flag in (('Buy', 'Sell'), ('Sell', 'Buy'), ('Buy', 'Buy'), ('Sell', 'Sell')):
        for direction, prices in (('up', up), ('down', down)):
            case('paper_'+direction+'_'+flag[0]+'_'+flag[1],
                 lambda prices=prices, flag=flag: paper_case(option, [prices], flag), 'paper_callback_not_Book')
    case('paper_state_resets_next_day', lambda: paper_case(option, [up, down]), 'paper_callback_not_Book')
    case('paper_equal_no_breakout', lambda: paper_case(option, [up.iloc[[0]]]), 'paper_callback_not_Book')
    case('paper_zero_contracts', lambda: paper_case(option,
         [pd.DataFrame([[3.1, .2, .3], [3.2, .3, .2]])]), 'paper_callback_not_Book')
    case('paper_zero_premium', lambda: paper_case(option, [pd.DataFrame([[3.1, 0., .1]])]), 'paper_callback_not_Book')
    case('paper_missing_entry', lambda: paper_case(option, [pd.DataFrame([[3.1, np.nan, .1]])]), 'paper_callback_not_Book')
    case('paper_None_string_skips', lambda: paper_case(option, [up], codes=('None', 'None')), 'paper_callback_not_Book')
    return {'cases': cases, 'case_count': len(cases), 'top_level_executed': False,
            'actual_json_roundtrip': True, 'historical_option_chain_used': False,
            'new_trading_backtests': 0, 'three_cost_scenarios': 'not_applicable_to_component_diagnostics'}


def components(namespace, frame):
    """Longest candidate reference, excluding later rows; original Num3/K.3 only."""
    rows = []
    for i in range(3, len(frame)):
        window = frame.iloc[i-3:i+1][['open', 'high', 'low', 'close']]
        namespace['get_price'] = lambda *a, window=window, **kw: window
        up, down = namespace['Price_Range'](frame.date.iloc[i], [3, .3, .3])
        hh, hc = max(map(float, window.high.iloc[:-1])), max(map(float, window.close.iloc[:-1]))
        lc, ll = min(map(float, window.close.iloc[:-1])), min(map(float, window.low.iloc[:-1]))
        width = max(hh-lc, hc-ll); opening = float(window.open.iloc[-1])
        decimal_width = max(max(map(lambda x: Decimal(str(x)), window.high.iloc[:-1]))-
                            min(map(lambda x: Decimal(str(x)), window.close.iloc[:-1])),
                            max(map(lambda x: Decimal(str(x)), window.close.iloc[:-1]))-
                            min(map(lambda x: Decimal(str(x)), window.low.iloc[:-1])))
        decimal_up, decimal_down = [float((Decimal(str(opening))+sign*Decimal('.3')*decimal_width)
                                         .quantize(Decimal('.001'), rounding=ROUND_HALF_EVEN))
                                   for sign in (1, -1)]
        rows.append({'date': frame.date.iloc[i], 'original_up': float(up), 'original_down': float(down),
            'scalar_up': round(opening+.3*width, 3), 'scalar_down': round(opening-.3*width, 3),
            'decimal_up': decimal_up, 'decimal_down': decimal_down, 'range': width,
            'open': opening, 'high': float(window.high.iloc[-1]), 'low': float(window.low.iloc[-1]),
            'last_previous_date': frame.date.iloc[i-1]})
    output = pd.DataFrame(rows)
    errors = {kind: int((output.original_up.ne(output[kind+'_up']) |
                        output.original_down.ne(output[kind+'_down'])).sum()) for kind in ('scalar', 'decimal')}
    require((output.last_previous_date < output.date).all(), 'Reference history includes current extrema')
    return output, {'rows': len(output), 'input_rows': len(frame), 'first': output.date.min(),
        'last': output.date.max(), 'window': 3, 'K_up': .3, 'K_down': .3,
        'rounding_difference_days': errors, 'original_numpy_round_preserved': True,
        'intraday_breakout_not_inferred_from_daily_extrema': True,
        'source_candidate_strict_usable': False, 'new_trading_backtests': 0,
        'original_strategy_complete': False, 'three_cost_scenarios': 'not_applicable_to_reference',
        'scope': '50ETF candidate daily DualThrust bands only; not minute signal, option trade or VIX'}


def worker(mode):
    require(mode in ('study', 'replay'), 'Unreviewed worker mode')
    prepared = read(DIRECTORY/'prepared.json')
    require(file_sha(DIRECTORY/'ETF-input.parquet') == prepared['operand_sha256'], 'Operand differs')
    for name, digest in prepared['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == digest, 'Worker code differs: '+name)
    target = DIRECTORY/mode; require(not target.exists(), 'Immutable worker directory exists')
    with network_blocked():
        ns = [source_namespace(n) for n in (0, 1)]
        diag = diagnostics(ns)
        frame, summary = components(ns[1], pd.read_parquet(DIRECTORY/'ETF-input.parquet'))
        target.mkdir(); save(target/'diagnostics.json', diag)
        require(read(target/'diagnostics.json') == diag, 'Persisted diagnostic shape differs')
        frame.to_parquet(target/'ETF-range-components.parquet', index=False)
        summary['synthetic_cases'] = diag['case_count']; save(target/'summary.json', summary)
    return {'status': 'ok', 'mode': mode, 'output_sha256': {n: file_sha(target/n) for n in OUTPUTS}}


def run(mode):
    folder = DIRECTORY/(mode+'-command'); folder.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.research_strategy_batch99', 'worker', '--mode', mode]
    process = subprocess.run(command, capture_output=True, text=True, timeout=240)
    for name, content in (('stdout.log', process.stdout), ('stderr.log', process.stderr)):
        (folder/name).write_text(content, encoding='utf-8')
    save(folder/'command.json', {'command': command, 'returncode': process.returncode,
         'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Worker failed; logs retained')
    return json.loads(process.stdout.splitlines()[-1])


def offline():
    require(all(read(DIRECTORY/(m+'-command')/'command.json')['returncode'] == 0
                for m in ('study', 'replay')), 'Worker execution failed')
    require(all(file_sha(DIRECTORY/'study'/n) == file_sha(DIRECTORY/'replay'/n) for n in OUTPUTS),
            'Offline outputs differ')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study/ETF-range-components.parquet'),
        pd.read_parquet(DIRECTORY/'replay/ETF-range-components.parquet'), check_exact=True)
    for mode in ('study', 'replay'):
        diag = read(DIRECTORY/mode/'diagnostics.json')
        require(diag['case_count'] == len(diag['cases']) and diag['actual_json_roundtrip'] is True,
                'Persisted cases incomplete')
    result = {'status': 'match', 'differences': 0, 'fresh_process': True, 'network_blocked': True,
              'output_sha256': {n: file_sha(DIRECTORY/'study'/n) for n in OUTPUTS},
              'prepared_sha256': file_sha(DIRECTORY/'prepared.json'), 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'replay', 'worker', 'offline', 'protect'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    if args.action == 'prepare': result = prepare()
    elif args.action == 'worker': result = worker(args.mode)
    elif args.action == 'offline': result = offline()
    elif args.action == 'protect': result = protect('data', DIRECTORY)
    else: result = run(args.action)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
