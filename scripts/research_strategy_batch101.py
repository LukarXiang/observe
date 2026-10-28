"""Original all-weather72/futures36 diagnostics; isolated operands are not trades."""
import ast
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.research_strategy_batch100 import canonical as primitive_canonical
from scripts.review_strategy_batch73 import safe_function
from scripts.verify_strategy_batch3 import require

SOURCES = ('2024年度精选策略1/72.桥水 全天候策略 增加一致性度量ES 风险控制.txt',
           '2024年度精选策略2/36.致敬市场(6) ，指数期货贴水.txt')
SOURCE_SHA = ('b30d9ba30a2e18bc162261a783479d853ff6768b4ae85963d39315fc85683835',
              '57d529dce6eb3ba8a76e4e79b1f5340df98d1993118d62b50cc34dec0b7ded4d')
PREREAD = Path('data/staging/strategies-batch100/20261011-next-source-preread')
FUNDS = ('510310.XSHG', '513100.XSHG', '513500.XSHG', '511010.XSHG',
         '511260.XSHG', '518880.XSHG', '510170.XSHG')
INDICES = ('000300.XSHG', '000016.XSHG', '000905.XSHG', '000852.XSHG')


def canonical(value):
    """Encode original listing-date objects while preserving diagnostic list shapes."""
    if isinstance(value, (dt.datetime, dt.date)): return value.isoformat()
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [canonical(v) for v in value]
    return primitive_canonical(value)


def tree(number):
    require(number in (0, 1), 'Unknown original source')
    path = PREREAD/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number] and
            file_sha(Path('repo/量化策略源代码')/SOURCES[number]) == SOURCE_SHA[number], 'Original source SHA differs')
    return ast.parse(read_source(path)[0])


def literal(number, namespace=None):
    nodes = [n for n in tree(number).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (10 if number == 0 else 4) and all(safe_function(n) for n in nodes),
            'Original definitions differ')
    ns = {'pd': pd, 'np': np, 'isnan': np.isnan, 'datetime': dt, 'g': SimpleNamespace(),
        'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREREAD/f'{number}.source'), 'exec'), ns)
    return ns


def init_allweather(**kwargs):
    calls = []
    ns = literal(0, {'set_benchmark': lambda s: calls.append(['benchmark', s]),
        'set_option': lambda *a: calls.append(['option', list(a)]),
        'set_order_cost': lambda *a, **k: calls.append(['cost', list(a), k]),
        'OrderCost': lambda **k: k,
        'run_monthly': lambda f, **k: calls.append(['monthly', f.__name__, k]), **kwargs})
    ns['initialize'](SimpleNamespace())
    return ns, calls


def es_kernel(daily_returns, confidence=2.58):
    """Call unchanged nested get_stock_ES with explicitly injected return operands."""
    nodes = [n for n in ast.walk(tree(0)) if isinstance(n, ast.FunctionDef) and n.name == 'get_stock_ES']
    require(len(nodes) == 1 and safe_function(nodes[0]), 'Nested ES definition differs')
    values = np.asarray(daily_returns, dtype=float).reshape(-1, 1)
    ns = {'getdailyreturn': lambda *a: values.copy()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original72-ES-only>', 'exec'), ns)
    return ns['get_stock_ES']('component_only', '1d', 120, confidence)


class ReturnOperands:
    """Controlled method protocol, never a replacement calendar/fill implementation."""
    def __init__(self, values): self.values = np.asarray(values, dtype=float).reshape(-1, 1)
    def resample(self, *a, **k): return self
    def pct_change(self, *a, **k): return self
    def fillna(self, *a, **k): return self


def injected_inverse_ES(operands, change_class_rates=False):
    ns, _ = init_allweather(history=lambda lag, freq, field, stock, **kw: ReturnOperands(operands[stock]))
    if change_class_rates:
        for asset in ns['g'].pool: ns['g'].pool[asset]['rate'] = 999.
    return ns['cal_stocks_ratio'](None, list(operands))


def futures_namespace(prices=None, contracts=None, holdings=(), equity=1_000_000., initialize_state=True):
    calls = []; prices = dict(prices or {}); contracts = dict(contracts or {})
    ns = literal(1, {'history': lambda count, unit, field, symbol:
        calls.append(['history', count, unit, field, symbol]) or pd.DataFrame({symbol: [prices[symbol]]}),
        'get_future_contracts': lambda prefix: list(contracts.get(prefix, [])),
        'get_security_info': lambda symbol: SimpleNamespace(get_contract_multiplier=lambda: 300.),
        'order': lambda symbol, qty: calls.append(['order', symbol, qty]),
        'order_target': lambda symbol, qty: calls.append(['target', symbol, qty])})
    if initialize_state: ns['after_code_changed'](None)
    context = SimpleNamespace(portfolio=SimpleNamespace(positions=dict.fromkeys(holdings, object()), total_value=equity))
    return ns, context, calls


def diagnostics():
    rows = []
    def case(name, function):
        try:
            with redirect_stdout(StringIO()): value = function()
            rows.append({'case': name, 'value': canonical(value), 'error': None})
        except Exception as exc:
            rows.append({'case': name, 'value': None, 'error': type(exc).__name__, 'message': str(exc)})

    case('allweather_initial_rules_and_schedule', lambda: allweather_initial())
    for date in ('2011-01-24', '2011-01-25', '2013-03-24', '2013-03-25', '2013-07-29', '2017-08-24'):
        def target(d=date):
            ns, _ = init_allweather()
            return ns['get_trade_target'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(d)))
        case('targets_'+date, target)

    def period():
        ns, _ = init_allweather(); calls = []
        ns['rebalance'] = lambda context, pool: calls.append({'callback_count': ns['g'].run_count, 'pool': pool})
        for i in range(25):
            ns['market_open'](SimpleNamespace(current_dt=dt.datetime(2020+i//12, 1+i%12, 2)))
        return {'calls': calls, 'final_count': ns['g'].run_count}
    case('rebalance_on_callbacks_0_12_24', period)

    def raise_keys(class_keys=False):
        ns, _ = init_allweather()
        ns['g'].rebalanced_asset_values = {'stock': 100.} if class_keys else {'510310.XSHG': 100.}
        context = SimpleNamespace(portfolio=SimpleNamespace(positions={'510310.XSHG': SimpleNamespace(value=140.)}))
        return {'original_result': ns['calc_asset_max_raise'](context), 'baseline': ns['g'].rebalanced_asset_values,
                'configured_raise_rate': ns['g'].raise_rate}
    case('security_keys_do_not_match_asset_values', raise_keys)
    case('class_keys_ratio_not_gain_probe_only', lambda: raise_keys(True))

    def actual_pipeline_failure():
        ns, _ = init_allweather(history=lambda *a, **k:
            pd.DataFrame({'A': np.linspace(100., 110., 120)}, index=pd.bdate_range('2020-01-01', periods=120)))
        return ns['cal_stocks_ratio'](None, ['A'])
    case('original_daily_resample_how_current_pandas_fails', actual_pipeline_failure)

    def risk_orders():
        ns, _ = init_allweather(); calls = []
        ns['cal_stocks_ratio'] = lambda *a: {'510310.XSHG': .25, '513100.XSHG': .75}
        ns['order_target_value'] = lambda s, v: calls.append([s, v])
        context = SimpleNamespace(portfolio=SimpleNamespace(total_value=1000., positions={
            'old': SimpleNamespace(value=50.), '510310.XSHG': SimpleNamespace(value=600.),
            '513100.XSHG': SimpleNamespace(value=100.)}))
        ns['rebalance'](context, ['510310.XSHG', '513100.XSHG'])
        return {'order_intents': calls, 'baseline_after_orders': ns['g'].rebalanced_asset_values}
    case('old_exit_reduction_before_remaining_targets', risk_orders)

    def equal_ES():
        values = {'510310.XSHG': [-.1, -.05]+[0.]*118, '518880.XSHG': [-.1, -.05]+[0.]*118}
        return {'original_outer_inverse_ES': injected_inverse_ES(values),
            'changed_class_rates_probe': injected_inverse_ES(values, True),
            'daily_returns_are_injected': True, 'legacy_method_protocol_is_a_stub': True}
    case('global_inverse_ES_ignores_class_rates_probe_only', equal_ES)
    case('zero_ES_inverse_weights_nonfinite_probe_only', lambda:
         injected_inverse_ES({'510310.XSHG': [0.]*120, '518880.XSHG': [0.]*120}))
    def empty_target():
        ns, _ = init_allweather(); calls = []
        ns['order_target_value'] = lambda s, v: calls.append([s, v])
        context = SimpleNamespace(portfolio=SimpleNamespace(total_value=1000., positions={'old': SimpleNamespace(value=100.)}))
        ns['rebalance'](context, [])
        return {'order_intents': calls, 'baseline': ns['g'].rebalanced_asset_values}
    case('empty_targets_exit_all_old_positions', empty_target)

    for name, values, confidence in (
        ('tail_two_values_divided_by_one', [-.1, -.05, .2]*40, 2.58),
        ('negative_ES_on_positive_operands', [.01]*120, 2.58),
        ('zero_ES', [0.]*120, 2.58),
        ('confidence5_zero_denominator', [-.1]+[0.]*119, 5),
        ('fallback_confidence_fixed_denominator', [-.1, -.05, .2]*40, 0),
        ('short_input_uses_fixed120_denominator', [-.1], 2.58)):
        case(name, lambda v=values, c=confidence: es_kernel(v, c))

    def init_future():
        calls = []; ns, c, _ = futures_namespace(initialize_state=False)
        ns.update(set_option=lambda *a: calls.append(['option', list(a)]),
            set_subportfolios=lambda *a: calls.append(['subportfolios', list(a)]),
            SubPortfolioConfig=lambda **k: k,
            run_daily=lambda f, **k: calls.append(['daily', f.__name__, k]))
        c.portfolio.starting_cash = 1_000_000.
        ns['initialize'](c)
        return {'calls': calls, 'state_after_initialize': vars(ns['g'])}
    case('futures_initialize_does_not_initialize_selection', init_future)
    def no_state():
        ns, c, _ = futures_namespace(initialize_state=False); ns['Trader_buy'](c)
    case('futures_buy_without_after_code_changed', no_state)

    def buy(future_price=100., index_price=108., held=(), equity=1_000_000., empty=False):
        prices = dict.fromkeys(INDICES, index_price); prices.update(IF1=future_price, IF2=105.)
        ns, c, calls = futures_namespace(prices, {} if empty else {'IF': ['IF1', 'IF2']}, held, equity)
        ns['Trader_buy'](c)
        return {'selected': ns['g'].future_contract, 'calls': calls}
    for name, args in (
        ('below_threshold_still_updates_selection', {'index_price': 100., 'future_price': 100.}),
        ('strict_six_percent_equality_no_order', {'index_price': 106., 'future_price': 100.}),
        ('largest_raw_discount_not_annualized', {}),
        ('selected_already_held_no_increment', {'held': ('IF1',)}),
        ('integer_zero_order_intent_is_retained', {'equity': 100.}),
        ('empty_contract_pool_index_error', {'empty': True}),
        ('zero_future_price_nonfinite_sort', {'future_price': 0.})):
        case(name, lambda a=args: buy(**a))
    def sell(selection):
        ns, c, calls = futures_namespace(holdings=('IF1', 'IF2'))
        ns['g'].future_contract = selection; ns['Trader_sell'](c)
        return calls
    case('morning_uses_previous_selection', lambda: sell('IF1'))
    case('empty_previous_selection_sells_all', lambda: sell(''))
    return {'cases': rows, 'case_count': len(rows), 'synthetic_only': True,
        'namespace_isnan_explicitly_numpy': True, 'ES_operands_are_injected_not_original_returns': True,
        'real_futures_contracts_or_minutes': False, 'new_trading_backtests': 0,
        'original_strategy_complete': False, 'compatibility_or_economic_repairs': False}


def allweather_initial():
    ns, calls = init_allweather(); g = ns['g']
    return {'calls': calls, 'pool': g.pool, 'stock_map_asset': g.stock_map_asset,
        'confidencelevel': g.confidencelevel, 'period': g.period, 'raise_rate': g.raise_rate,
        'run_count': g.run_count}


if __name__ == '__main__':
    # No output directory or catalog is mutated by this source-only command.
    print(json.dumps(diagnostics(), ensure_ascii=False, allow_nan=False))
