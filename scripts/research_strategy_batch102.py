"""Original macro48/ETF58 functions and explicit operand diagnostics, never trades."""
import ast
from contextlib import redirect_stdout
import datetime as dt
from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import optimize as op

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch101 import canonical
from scripts.verify_strategy_batch3 import require

PREREAD = Path('data/staging/strategies-batch101/20261011-next-source-preread')
SOURCES = ('2024年度精选策略2/48.一种宏观数据的中长线策略，年化15%，最大回撤9%.txt',
           '聚宽2025年精选/58Debug-输出信息-多标的版ETF策略(ETF复现之三）.txt')
SOURCE_SHA = ('5d55e30ef811ceda9ef64f13f874d38ae84204dcbf2fb248274188563d08c88b',
              'e7ff4f030f9549890c3277b2ca7eefbb068893832a12c7d0b685206b2e4f7401')
POOL = ('516510.XSHG', '159985.XSHE', '515880.XSHG', '513060.XSHG', '512660.XSHG',
        '513520.XSHG', '513330.XSHG', '515790.XSHG', '512100.XSHG', '512100.XSHG',
        '159998.XSHE', '513500.XSHG', '162719.XSHE', '159949.XSHE', '515030.XSHG',
        '159928.XSHE', '513100.XSHG', '159995.XSHE')


def tree(number):
    require(number in (0, 1), 'Unknown source')
    path = PREREAD/f'{number}.source'
    require(file_sha(path) == file_sha(Path('repo/量化策略源代码')/SOURCES[number]) == SOURCE_SHA[number], 'Original source changed')
    return ast.parse(read_source(path)[0])


def safe_definition(node):
    """Allow original builtin annotations without evaluating arbitrary annotation calls."""
    builtin_types = {'str', 'float', 'tuple', 'dict', 'bool', 'int', 'list'}
    def annotation(value): return value is None or isinstance(value, ast.Name) and value.id in builtin_types
    return (isinstance(node, ast.FunctionDef) and not node.decorator_list and not node.type_params and annotation(node.returns)
        and all(annotation(arg.annotation) for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
        and all(arg is None or annotation(arg.annotation) for arg in (node.args.vararg, node.args.kwarg))
        and all(isinstance(value, ast.Constant) for value in node.args.defaults)
        and all(value is None or isinstance(value, ast.Constant) for value in node.args.kw_defaults))


def literal(number, namespace=None):
    definitions = [node for node in tree(number).body if isinstance(node, ast.FunctionDef)]
    require(len(definitions) == (13 if number == 0 else 6) and all(safe_definition(node) for node in definitions), 'Unsafe/missing definition')
    ns = {'np': np, 'pd': pd, 'op': op, 'Decimal': Decimal, 'datetime': dt, 'g': SimpleNamespace(),
          'log': SimpleNamespace(info=lambda *a: None)}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<unchanged-batch102-definitions>', 'exec'), ns)
    return ns


def initialize(number, namespace=None):
    calls = []
    ns = literal(number, {'set_option': lambda *a: calls.append(['option', list(a)]),
        'set_benchmark': lambda value: calls.append(['benchmark', value]),
        'OrderCost': lambda **k: k, 'set_order_cost': lambda *a, **k: calls.append(['cost', list(a), k]),
        'run_daily': lambda function, *a, **k: calls.append(['daily', function.__name__, list(a), k]),
        **(namespace or {})})
    ns['initialize'](SimpleNamespace(portfolio=SimpleNamespace(starting_cash=1000.)))
    return ns, calls


class CloseOperands:
    """Explicit positional method stub, not a pandas adapter or approved compatibility fix."""
    def __init__(self, values): self.values = np.asarray(values, dtype=float)
    def __getitem__(self, number): return self.values[number]
    def mean(self): return pd.Series(self.values).mean()
    def tail(self, count): return CloseOperands(self.values[-count:])


def reference_metrics(frames):
    """Unchanged metric expressions receive explicitly positional close operands."""
    ns = literal(1, {'attribute_history': lambda symbol, count:
        {'close': CloseOperands(frames[symbol][-count:])}})
    return ns['calculate_etf_metrics'](list(frames), 20, 5)


class Field:
    def __init__(self, name): self.name = name
    def __ge__(self, value): return ['ge', self.name, value]
    def __le__(self, value): return ['le', self.name, value]
    def __lt__(self, value): return ['lt', self.name, value]
    def asc(self): return ['asc', self.name]
    def desc(self): return ['desc', self.name]


class Query:
    """Record original query clauses; synthetic rows do not prove macro versions."""
    def __init__(self, fields): self.fields = [field.name for field in fields]; self.filters = []; self.order = None
    def filter(self, *clauses): self.filters = list(clauses); return self
    def order_by(self, order): self.order = order; return self


def macro_query(name):
    rows = []; fields = SimpleNamespace(**{n: Field(n) for n in
        ('pmi', 'stat_month', 'produce_idx', 'new_orders_idx', 'raw_material_idx', 'finished_produce_idx')})
    def run(query):
        rows.append({'fields': query.fields, 'filters': query.filters, 'order': query.order})
        if name == 'get_PMI': return pd.DataFrame({'pmi': np.arange(13.)+50})
        return pd.DataFrame({'stat_month': ['2024-03', '2024-02', '2024-01'],
            query.fields[1]: [50., 49., 51.], query.fields[2]: [52., 50., 51.]})
    ns = literal(0, {'query': lambda *f: Query(f), 'macro': SimpleNamespace(MAC_MANUFACTURING_PMI=fields, run_query=run)})
    value = ns[name]('2024-04') if name == 'get_PMI' else ns[name]('2013-01', '2024-04')
    return {'value': value, 'queries': rows, 'macro_rows_are_synthetic': True}


def macro_state(mode, price, *, gap=1, current_as_dataframe=False):
    calls = []; ns, _ = initialize(0)
    ns['g'].mode = mode; ns['g'].times = 0; ns['g'].record = 100.
    ns.update(get_PMI=lambda *a: [49.]*13, change_to_yeak_k=lambda: {'2000': 1000.},
        year_move_average=lambda *a: 0., day_move_average=lambda *a: 100., sell_stock_security=lambda *a: None)
    def bars(symbol, count, frequency, **kwargs):
        calls.append(['bars', symbol, count, frequency, kwargs])
        values = [price if symbol == '000300.XSHG' else 10.]
        return pd.DataFrame({'close': values}) if current_as_dataframe else pd.Series(values)
    ns.update(get_bars=bars, order_target=lambda *a: calls.append(['target', *a]),
        order_value=lambda *a: calls.append(['value', *a]))
    now = dt.datetime(2024, 4, 8, 14, 55)
    context = SimpleNamespace(current_dt=now, previous_date=now.date()-dt.timedelta(days=gap),
        portfolio=SimpleNamespace(total_value=1000., available_cash=1000., positions={}))
    ns['handle'](context)
    return {'calls': calls, 'times': ns['g'].times, 'record': ns['g'].record,
        'mode': ns['g'].mode, 'price_shape_stub': not current_as_dataframe, 'other_helpers_are_stubs': True}


def etf_orders(metrics, holdings=(), immediate=False):
    ns, _ = initialize(1); calls = []
    portfolio = SimpleNamespace(positions=dict.fromkeys(holdings, object()), available_cash=1000.)
    context = SimpleNamespace(portfolio=portfolio)
    ns['calculate_etf_metrics'] = lambda *a: metrics.copy(deep=True)
    def target(symbol, value):
        calls.append([symbol, float(value)])
        if immediate:
            if value == 0: portfolio.positions.pop(symbol, None)
            else: portfolio.positions[symbol] = object(); portfolio.available_cash -= value
    ns['order_target_value'] = target
    ns['trade'](context)
    return {'calls': calls, 'buy': ns['g'].buy_etf, 'sell': ns['g'].sell_etf, 'day': ns['g'].day,
        'holding_keys': list(portfolio.positions), 'metrics_are_injected': True, 'immediate_feedback_stub': immediate}


def diagnostics():
    rows = []
    def case(name, function):
        stream = StringIO()
        try:
            with redirect_stdout(stream): value = function()
            rows.append({'case': name, 'value': canonical(value), 'error': None, 'stdout': stream.getvalue()})
        except Exception as exc:
            rows.append({'case': name, 'value': None, 'error': type(exc).__name__, 'message': str(exc), 'stdout': stream.getvalue()})
    def initial(number):
        ns, calls = initialize(number); return {'calls': calls, 'state': vars(ns['g'])}
    case('macro_initial_rules', lambda: initial(0)); case('ETF_initial_duplicate_pool', lambda: initial(1))
    for name in ('get_PMI', 'calc_provide_and_need', 'calc_save'): case('macro_query_'+name, lambda n=name: macro_query(n))
    case('prior_month_january_boundary', lambda: literal(0)['calc_last_month']('2024-01'))
    for name, values, target in (('percentile_ties', [1., 2., 2.], 2.),
            ('percentile_single_zero_denominator', [1.], 1.), ('percentile_empty', [], 0.),
            ('percentile_nan_missing_values', [1., np.nan, 2.], 2.)):
        case(name, lambda v=values, t=target: literal(0)['percentile'](np.asarray(v), t))
    case('year_mean_fixed10_even_three_values', lambda: literal(0)['year_move_average']({'a': 10., 'b': 20., 'c': 30.}))
    def order(times, cash):
        ns, _ = initialize(0); ns['g'].times = times; calls = []; ns['order_value'] = lambda *a: calls.append(list(a))
        ns['order_func'](1000., cash, 1.); return calls
    for name, count, cash in (('first4percent', 0, 1000.), ('next8percent', 1, 1000.),
            ('cash_below_lot_no_order', 5, 99.), ('lot_exact_fallback_cash', 5, 100.)):
        case('allocation_'+name, lambda c=count, money=cash: order(c, money))
    case('macro_DataFrame_current_scalar_error', lambda: macro_state(1, 100., current_as_dataframe=True))
    for name, mode, price, gap in (('mode1_exact_minus2_hold', 1, 98., 1), ('mode1_below_minus2_add', 1, 97.9, 1),
            ('mode2_exact_minus5_hold', 2, 95., 1), ('mode2_below_minus5_add', 2, 94.9, 1),
            ('mode1_gap3_regular_order', 1, 100., 3), ('mode1_gap2_no_regular_order', 1, 100., 2)):
        case(name+'_shape_stub', lambda m=mode, p=price, d=gap: macro_state(m, p, gap=d))
    def exit_stock():
        ns, _ = initialize(0); ns['g'].mode = 2; ns['g'].times = 3; calls = []
        ns.update(get_bars=lambda *a, **k: pd.Series([101.]),
            day_move_average=lambda n, now: {(5, True): 98., (5, False): 102., (20, True): 100.,
                (20, False): 100., (120, True): 100.}[n, now],
            order_target=lambda *a: calls.append(['target', *a]), order_value=lambda *a: calls.append(['value', *a]))
        ns['sell_stock_security']('2024-04', 1000., ['510300.XSHG'])
        return {'calls': calls, 'mode': ns['g'].mode, 'times': ns['g'].times, 'shape_stub': True}
    case('exit_buys_full_equity_not_half_stub', exit_stock)
    def metric_failure():
        ns = literal(1, {'attribute_history': lambda *a: pd.DataFrame({'close': np.arange(20.)+100.},
            index=pd.bdate_range('2024-01-02', periods=20))})
        return ns['calculate_etf_metrics'](['X'], 20, 5)
    case('ETF_datetime_index_minus1_KeyError', metric_failure)
    case('ETF20_native_rows19_return_intervals_protocol_stub', lambda: reference_metrics({'X': np.linspace(100., 110., 20)}).to_dict())
    for name, rates, keys, held, immediate in (
        ('strict5_and20_buy_excluded', [.20, .05], ['A', 'B'], (), False),
        ('strict20_held_not_sold', [.20, .10], ['A', 'B'], ('A',), False),
        ('above20_held_sold', [.21, .10], ['A', 'B'], ('A',), False),
        ('duplicate_top2_two_orders_same_key', [.10, .10, .06], ['A', 'A', 'B'], (), True),
        ('held_duplicate_only_one_removed', [.10, .10, .06], ['A', 'A', 'B'], ('A',), False),
        ('sell_feedback_deferred_blocks_capacity', [.10, .08], ['A', 'B'], ('old1', 'old2'), False),
        ('sell_feedback_immediate_frees_capacity', [.10, .08], ['A', 'B'], ('old1', 'old2'), True)):
        frame = pd.DataFrame({'inc': rates, 'malong': [100.]*len(rates), 'mashort': [101.]*len(rates)}, index=keys)
        case(name, lambda f=frame, h=held, i=immediate: etf_orders(f, h, i))
    def print_info():
        ns, _ = initialize(1); ns['get_security_info'] = lambda *a: SimpleNamespace(display_name='synthetic')
        context = SimpleNamespace(portfolio=SimpleNamespace(positions={}, portfolio_value=1100., starting_cash=1000.))
        ns['print_position_info'](context); ns['print_position_info'](context)
        return {'last_portfolio_value': ns['g'].previous_portfolio_value, 'total_return_percent': ns['g'].total_return_rate,
            'printed_return_is_not_Book_NAV': True}
    case('return_print_updates_previous_value_not_ledger', print_info)
    doc = {'case_count': len(rows), 'cases': rows, 'synthetic_only': True,
        'method_protocol_and_scalar_shapes_are_stubs': True, 'new_trading_backtests': 0, 'original_strategy_complete': False}
    import json
    require(__import__('json').loads(json.dumps(doc, ensure_ascii=False, allow_nan=False)) == doc, 'Actual JSON shape differs')
    doc['actual_json_roundtrip'] = True
    return doc
