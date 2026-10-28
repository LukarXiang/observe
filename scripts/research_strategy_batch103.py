"""Unchanged ETF6/81 callbacks on explicit diagnostic operands; never trades."""
import ast
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import talib

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch101 import canonical
from scripts.research_strategy_batch102 import safe_definition
from scripts.verify_strategy_batch3 import require

PREREAD = Path('data/staging/strategies-batch102/20261011-next-source-preread')
SOURCES = ('2023年度精选策略/6.21年度6.为了挣点积分自己实盘的策略分享出来.txt',
           '聚宽2025年精选/81搞市场最靓的仔！指数ETF动量轮动策略-2.txt')
SOURCE_SHA = ('b4ffbfb4d757aaa417406591e7d3f1481aee32ef37e0e2f94fd82abb99defeb5',
              '8d1a703edbb5c173d3f6136da01556d2295d19a8800d5a393927bf1982277210')
REAL_POOL = ('159901.XSHE', '162605.XSHE', '161005.XSHE', '510050.XSHG',
             '510180.XSHG', '510880.XSHG', '159905.XSHE', '159915.XSHE',
             '510300.XSHG', '510500.XSHG', '159949.XSHE')
MOMENTUM_POOL = ('510500.XSHG', '510300.XSHG', '510050.XSHG', '159949.XSHE',
                 '513100.XSHG', '513500.XSHG', '159920.XSHE', '513520.XSHG',
                 '513030.XSHG', '162411.XSHE', '159985.XSHE')


def tree(number):
    require(number in (0, 1), 'Unknown source')
    path = PREREAD/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number] == file_sha(Path('repo/量化策略源代码')/SOURCES[number]),
            'Original source changed')
    return ast.parse(read_source(path)[0])


def literal(number, namespace=None):
    nodes = [n for n in tree(number).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (10 if number == 0 else 5) and all(safe_definition(n) for n in nodes),
            'Original definition contract differs')
    ns = {'pd': pd, 'np': np, 'talib': talib, 'random': random, 'g': SimpleNamespace(),
          'log': SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, set_level=lambda *a: None)}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREREAD/f'{number}.source'), 'exec'), ns)
    return ns


def initialize(number, namespace=None):
    calls = []
    ns = literal(number, {'set_benchmark': lambda s: calls.append(['benchmark', s]),
        'set_option': lambda *a: calls.append(['option', list(a)]),
        'run_daily': lambda f, **k: calls.append(['daily', f.__name__, k]),
        'set_order_cost': lambda *a, **k: calls.append(['cost', list(a), k]),
        'set_slippage': lambda *a, **k: calls.append(['slippage', list(a), k]),
        'set_commission': lambda v: calls.append(['commission', v]),
        'OrderCost': lambda **k: k, 'PerTrade': lambda **k: k,
        'PriceRelatedSlippage': lambda v: v, 'FixedSlippage': lambda v: v,
        'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, start_date=dt.date(2000, 1, 1)),
        **(namespace or {})})
    ns['initialize'](SimpleNamespace())
    return ns, calls


def real_signal(histories, current, *, emotion=1, last=()):
    """Full original 11-value mapping with injected heat result, never an original signal."""
    ns, _ = initialize(0)
    require(set(histories) == set(current) == set(REAL_POOL), 'Diagnostic original pool incomplete')
    calls = []
    def history(symbol, count, *a, **k):
        calls.append([symbol, count, list(a), k])
        return {'close': np.asarray(histories[symbol], dtype=float)[-count:]}
    ns.update(attribute_history=history,
        get_current_data=lambda: {s: SimpleNamespace(last_price=p) for s, p in current.items()},
        EmotionMonitor=lambda context: emotion)
    ns['g'].last = list(last)
    result = ns['get_signal'](SimpleNamespace())
    return {'signal': result, 'buy': ns['g'].buy, 'last': ns['g'].last,
            'target_market': ns['g'].target_market, 'rows': ns['g'].df.to_dict('records'),
            'history_calls': calls, 'emotion_is_injected': True, 'synthetic_operands': True}


def emotion_monitor(volume=None, *, fail=False):
    ns, _ = initialize(0)
    calls = []
    def history(*a, **k):
        calls.append([list(a), k])
        if fail: raise RuntimeError('explicit missing-volume diagnostic')
        return pd.DataFrame({'volume': np.asarray(volume, dtype=float)})
    ns['attribute_history'] = history
    value = ns['EmotionMonitor'](SimpleNamespace())
    return {'result': value, 'rate': ns['g'].emotion_rate, 'calls': calls, 'synthetic_operands': True}


def real_orders(signal, buy=(), holdings=None, *, last=(), clear=(), immediate=False):
    """Original order intents with stated feedback; no fills, fees, NAV or Book."""
    ns, _ = initialize(0); calls = []
    positions = {s: SimpleNamespace(security=s, value=float(v)) for s, v in (holdings or {}).items()}
    portfolio = SimpleNamespace(positions=positions, total_value=1000., available_cash=100., returns=0.)
    ns['g'].signal = signal; ns['g'].buy = list(buy); ns['g'].last = list(last); ns['g'].clear = list(clear)
    def order(kind, symbol, value):
        calls.append([kind, symbol, float(value)])
        if immediate and kind == 'target' and value == 0:
            old = positions.pop(symbol, None)
            portfolio.available_cash += 0. if old is None else old.value
        return None
    ns['order_target_value'] = lambda s, v: order('target', s, v)
    ns['order_value'] = lambda s, v: order('value', s, v)
    error = None
    try: ns['ETFtrade2'](SimpleNamespace(portfolio=portfolio))
    except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'calls': calls, 'buy': ns['g'].buy, 'last': ns['g'].last, 'clear': ns['g'].clear,
            'holding_keys': list(positions), 'error': error, 'immediate_feedback_stub': immediate,
            'orders_return_None_not_fills': True, 'synthetic_operands': True}


def momentum(histories, current, *, status=0, holding='0', ordered=None,
             random_values=(.5,), immediate=False):
    """Explicit positional/history and feedback operands; preserve original function body."""
    ns, _ = initialize(1); logs = []; calls = []; random_calls = []
    symbols = list(ordered if ordered is not None else histories)
    require(set(symbols) == set(histories) == set(current), 'Momentum diagnostic operand mismatch')
    ns['g'].stocks = symbols  # diagnostic order override only, not a proposed production fix
    ns['g'].position.update(STATUS=status, ETF_HOLD=holding)
    values = iter(random_values)
    def draw():
        value = next(values); random_calls.append(value); return value
    portfolio = SimpleNamespace(cash=100.)
    def target(symbol, value):
        calls.append([symbol, float(value), dict(ns['g'].position)])
        if immediate and value == 0: portfolio.cash += 900.
        return None
    ns.update(attribute_history=lambda s, *a, **k: {'close': histories[s]},
        get_current_data=lambda: {s: SimpleNamespace(last_price=p, name=s) for s, p in current.items()},
        random=SimpleNamespace(random=draw), order_target_value=target,
        log=SimpleNamespace(info=lambda *a: logs.append(list(a)), warning=lambda *a: logs.append(list(a))))
    ns['my_trade'](SimpleNamespace(portfolio=portfolio))
    return {'calls': calls, 'position': ns['g'].position, 'logs': logs, 'random_calls': random_calls,
            'diagnostic_order': symbols, 'cash': portfolio.cash, 'synthetic_operands': True,
            'controlled_random_not_approved_seed': True, 'immediate_feedback_stub': immediate}


def actual_set_trial():
    """Retain source set in a fresh process, but use equal synthetic operands/fixed draw."""
    ns, _ = initialize(1); symbols = list(ns['g'].stocks)
    draws = []; calls = []
    def draw(): draws.append(.5); return .5
    ns.update(attribute_history=lambda *a, **k: {'close': np.full(30, 100.)},
        get_current_data=lambda: {s: SimpleNamespace(last_price=101., name=s) for s in symbols},
        random=SimpleNamespace(random=draw), order_target_value=lambda *a: calls.append(list(a)))
    ns['my_trade'](SimpleNamespace(portfolio=SimpleNamespace(cash=1000.)))
    return {'iteration': symbols, 'calls': calls, 'position': ns['g'].position, 'draws': draws,
            'fixed_draw_is_diagnostic_only': True, 'synthetic_operands': True}


def diagnostics():
    rows = []
    def case(name, function):
        out = StringIO()
        try:
            with redirect_stdout(out): value = function()
            rows.append({'case': name, 'value': canonical(value), 'error': None, 'stdout': out.getvalue()})
        except Exception as exc:
            rows.append({'case': name, 'value': None, 'error': type(exc).__name__, 'message': str(exc), 'stdout': out.getvalue()})
    def initial(number):
        ns, calls = initialize(number)
        if number == 0: ns['before_trading_start'](SimpleNamespace())
        state = vars(ns['g']).copy()
        if number == 0: state['df'] = state['df'].to_dict('records')
        else: state['stocks'] = sorted(state['stocks'])
        return {'calls': calls, 'state': state, 'listing_info_is_synthetic': number == 0}
    case('real_initial_schedule_cost', lambda: initial(0)); case('momentum_initial_set_cost', lambda: initial(1))
    base = {s: np.full(13, 100.) for s in REAL_POOL}
    for name, price, emotion, last in (('below_code_point1_KEEP', 100., 1, ()),
            ('below_code_point1_existing_last_sell', 100., 1, ('old',)),
            ('point2_above_code_not_comment1', 100.2, 1, ()),
            ('all11_pass_choose1', 102., 1, ()), ('emotion_None_not_sell', 102., None, ()),
            ('emotion_minus1_clears_state', 102., -1, ('old',)), ('all_nan_fill_minus100_KEEP', np.nan, 1, ())):
        case('real_'+name, lambda p=price, e=emotion, l=last: real_signal(base, dict.fromkeys(REAL_POOL, p), emotion=e, last=l))
    prices = dict.fromkeys(REAL_POOL, 102.); prices[REAL_POOL[-1]] = 98.
    case('real_dynamic_heat_uses_worst_fund_inverse', lambda: real_signal(base, prices))
    case('real_empty_history_IndexError', lambda: real_signal(dict.fromkeys(REAL_POOL, []), dict.fromkeys(REAL_POOL, 102.)))
    for count in (4, 5, 6):
        case(f'emotion_{count}_low_volume_days', lambda n=count: emotion_monitor([1000.]*(100-n)+[100.]*n))
    case('emotion_flat_no_cross_returns_None', lambda: emotion_monitor([100.]*100))
    case('emotion_all_nan_returns_None', lambda: emotion_monitor([np.nan]*100))
    case('emotion_missing_exception_returns1', lambda: emotion_monitor(fail=True))
    for signal in ('KEEP', 'sell_the_stocks'):
        case('real_'+signal+'_empty_buys_cash_fund', lambda s=signal: real_orders(s))
    case('real_sell_keeps_cash_fund', lambda: real_orders('sell_the_stocks', holdings={'511880.XSHG': 900.}))
    case('real_sell_immediate_dict_mutation', lambda: real_orders('sell_the_stocks', holdings={'A': 900.}, immediate=True))
    case('real_BUY_empty_target_ZeroDivisionError', lambda: real_orders('BUY'))
    case('real_BUY_value_uses_equity_not_cash', lambda: real_orders('BUY', ('A',)))
    case('real_BUY_new_cash_fund_stale_last', lambda: real_orders('BUY', ('A',), {'511880.XSHG': 900.}))
    case('real_BUY_cash_fund_immediate_feedback', lambda: real_orders('BUY', ('A',), {'511880.XSHG': 900.}, immediate=True))
    case('real_BUY_clear_stale_missing_last', lambda: real_orders('BUY', ('A',), {'A': 900.}, clear=('old',)))
    case('real_rebalance_nested_duplicate_targets', lambda: real_orders('BUY', ('A', 'B'), {'A': 800., 'B': 800.}))
    case('real_rebalance_exact1point5_no_target', lambda: real_orders('BUY', ('A', 'B'), {'A': 750., 'B': 750.}))
    flat = np.full(30, 100.)
    for name, price, status, holding in (('entry_equal1point001_no_order', 100.*1.001, 0, '0'),
            ('entry_above_boundary', np.nextafter(100.*1.001, np.inf), 0, '0'),
            ('exit_equal_point999_no_order', 100.*.999, 1, 'A'),
            ('exit_below_boundary', np.nextafter(100.*.999, -np.inf), 1, 'A'),
            ('invalid_nan_current_returns', np.nan, 0, '0'), ('invalid_tiny_current_returns', .001, 0, '0')):
        case('momentum_'+name, lambda p=price, s=status, h=holding: momentum({'A': flat}, {'A': p}, status=s, holding=h))
    for count in (20, 21, 29):
        case(f'momentum_short{count}_rows', lambda n=count: momentum({'A': np.full(n, 100.)}, {'A': 101.}))
    case('momentum_datetime_index_KeyError', lambda: momentum({'A': pd.Series(flat, index=pd.bdate_range('2024-01-02', periods=30))}, {'A': 101.}))
    case('momentum_RangeIndex_negative_KeyError', lambda: momentum({'A': pd.Series(flat)}, {'A': 101.}))
    for order in (('A', 'B'), ('B', 'A')):
        case('momentum_equal_scores_order_'+''.join(order), lambda o=order: momentum(dict.fromkeys(o, flat), dict.fromkeys(o, 101.), ordered=o))
    case('momentum_zero_draw_ties_use_tuple_symbol', lambda: momentum(dict.fromkeys(('A', 'B'), flat), dict.fromkeys(('A', 'B'), 101.), random_values=(0.,)))
    for immediate in (False, True):
        case('momentum_switch_'+('immediate' if immediate else 'delayed'), lambda i=immediate: momentum({'A': flat}, {'A': 101.}, status=1, holding='old', immediate=i))
    case('momentum_other_winner_below_entry_keeps_old', lambda: momentum({'A': flat}, {'A': 99.}, status=1, holding='old'))
    return {'case_count': len(rows), 'cases': rows, 'original_function_bodies_changed': False,
            'controlled_operands_not_platform_equivalence': True, 'new_trading_backtests': 0}
