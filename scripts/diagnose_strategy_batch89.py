"""Diagnose SHA-pinned all-weather callbacks with synthetic API state only."""
import argparse
import ast
import datetime as dt
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import requests

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch89 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch89/20261009-all-weather-callbacks')
RECEIPT = Path('docs/handoff/2026-10-09-batch89-stage-verification.json')
FUNCTIONS = {'initialize', 'fun_initialize', 'before_market_open', 'after_market_close',
             'market_open', 'rebalance', 'fun_calc_trade_ratio', 'fun_trade', 'need_rebalance'}
FILES = ('scripts/diagnose_strategy_batch89.py', 'tests/unit/test_batch89_callbacks.py')


def binding():
    receipt = read(RECEIPT)
    require(receipt['formal_acceptance'] is False and receipt['new_trading_backtests'] == 0,
            'Unexpected parent scope')
    for group in ('implementation_sha256', 'implementation_copy_sha256', 'evidence_sha256'):
        require(all(file_sha(name) == sha for name, sha in receipt[group].items()), 'Parent stage changed')
    stage.validate_sources(stage.DIRECTORY)
    return {'parent_receipt': str(RECEIPT), 'parent_receipt_sha256': file_sha(RECEIPT),
            'source_sha256': list(stage.SOURCE_SHA), 'snapshot': stage.SNAPSHOT,
            'synthetic_only': True, 'not_a_backtest': True, 'formal_acceptance': False}


def original_definitions(source, namespace):
    require(file_sha(source) == stage.SOURCE_SHA[0], 'Unreviewed callback source')
    tree = ast.parse(read_source(Path(source))[0])
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    functions = [n for n in nodes if isinstance(n, ast.FunctionDef)]
    classes = [n for n in nodes if isinstance(n, ast.ClassDef)]
    require(len(functions) == 9 and {n.name for n in functions} == FUNCTIONS and
            len(classes) == 1 and classes[0].name == 'QuantLib', 'Definition scope changed')
    cls = classes[0]
    require(all(safe_function(n) for n in functions) and not cls.decorator_list and
            not cls.keywords and not cls.type_params and len(cls.bases) == 1 and
            isinstance(cls.bases[0], ast.Name) and cls.bases[0].id == 'object' and
            all(safe_function(n) for n in cls.body), 'Unsafe callback definition')
    # Execute only pinned definitions; platform imports and top-level statements stay excluded.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<batch89-pinned-callbacks>', 'exec'), namespace)
    return namespace


class Harness:
    """Synthetic callback state, not a fill engine, fee model, or portfolio ledger."""

    def __init__(self, source, *, quotes=None, positions=None, cash=100000., total=100000.,
                 history_shape='numpy', delivery='deferred'):
        require(history_shape in ('numpy', 'list'), 'Unknown history shape')
        require(delivery in ('deferred', 'immediate_keep_zero', 'immediate_delete_zero'), 'Unknown delivery')
        self.quotes = {s: 10. for s in stage.POOL} | (quotes or {})
        self.context = SimpleNamespace(current_dt=dt.datetime(2021, 1, 4, 9, 30), portfolio=SimpleNamespace(
            positions=positions or {}, available_cash=cash, total_value=total, returns=0.))
        self.calls = []; self.orders = []; self.settings = []; self.logs = []
        self.history_shape = history_shape; self.delivery = delivery
        self.g = SimpleNamespace()
        self.ns = original_definitions(source, {
            'g': self.g, 'dt': dt, 'log': SimpleNamespace(info=lambda *args: self.logs.append(str(args)),
                set_level=lambda *args: self.settings.append(['log_level', list(args)])),
            'history': self.history, 'order_target_value': self.order,
            'set_benchmark': lambda s: self.settings.append(['benchmark', s]),
            'set_option': lambda key, value: self.settings.append(['option', key, value]),
            'FixedSlippage': lambda value: value,
            'set_slippage': lambda value: self.settings.append(['slippage', value]),
            'run_daily': lambda fn, **kw: self.settings.append(['schedule', fn.__name__, kw])})
        self.ns['initialize'](self.context)
        self.ns['before_market_open'](self.context)
        self.g.tradeRecord = ''

    def history(self, lag, freq, field, stock, df=False):
        self.calls.append({'lag': lag, 'freq': freq, 'field': field, 'stock': stock, 'df': df})
        require(lag == 1 and freq == '1d' and field == 'close' and not df,
                'Real risk-history calculation forbidden in synthetic callback harness')
        values = {s: [self.quotes[s]] for s in ([stock] if isinstance(stock, str) else stock)}
        return {s: np.array(v) for s, v in values.items()} if self.history_shape == 'numpy' else values

    def order(self, stock, value):
        self.orders.append([stock, float(value)])
        if self.delivery != 'deferred':
            # Deliberately simple API-state injection, with no cash/fees/rounding or NAV simulation.
            amount = float(value) / self.quotes[stock]
            if amount == 0 and self.delivery == 'immediate_delete_zero':
                self.context.portfolio.positions.pop(stock, None)
            else:
                self.context.portfolio.positions[stock] = position(amount, self.quotes[stock])

    def call(self, function, *args):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                value = function(self.context, *args)
                result = {'exception': None, 'return': value}
            except (KeyError, ZeroDivisionError, TypeError) as exc:
                result = {'exception': type(exc).__name__, 'message': str(exc)}
        return {**result, 'warnings': [str(w.message) for w in captured],
                'orders': [list(order) for order in self.orders]}


def position(amount, price):
    return SimpleNamespace(total_amount=amount, price=price)


def source_path():
    return Path(read(stage.DIRECTORY/'sources-frozen.json')[0]['source_copy'])


def diagnostics(source):
    cases = []
    a, b, _, fund = stage.POOL

    def record(name, actual, expected):
        require(all(actual.get(key) == value for key, value in expected.items()), f'{name}: independent expectation differs')
        cases.append({'case': name, 'actual': actual, 'expected': expected})

    h = Harness(source)
    record('initialize_and_original_pool', {'pool': h.g.pools, 'cycle': h.g.hold_cycle,
        'confidence': h.g.confidence_level, 'settings': h.settings}, {
        'pool': list(stage.POOL), 'cycle': 30, 'confidence': 2.58,
        'settings': [['benchmark', '511010.XSHG'], ['option', 'use_real_price', True],
            ['log_level', ['order', 'error']], ['slippage', .002],
            ['schedule', 'before_market_open', {'time': 'before_open', 'reference_security': '000300.XSHG'}],
            ['schedule', 'market_open', {'time': 'open', 'reference_security': '000300.XSHG'}],
            ['schedule', 'after_market_close', {'time': 'after_close', 'reference_security': '000300.XSHG'}]]})
    h = Harness(source)
    recalcs = []; trades = []
    def synthetic_ratio(context):
        recalcs.append(callback)
        return {a: .25}
    h.ns['fun_calc_trade_ratio'] = synthetic_ratio
    h.ns['fun_trade'] = lambda context, weights: trades.append(dict(weights))
    for callback in range(1, 64):
        h.ns['market_open'](h.context)
    record('synthetic_weights_63_callback_cycle', {'recalcs': recalcs, 'trade_calls': len(trades),
        'anchor': h.g.positions[a]}, {'recalcs': [1, 32, 63], 'trade_calls': 63, 'anchor': 0.})
    h.context.portfolio.positions[a] = position(100, 100.)
    record('initial_zero_anchor_skips_price_move', h.call(h.ns['need_rebalance']), {'exception': None, 'return': None})
    for price, expected in ((115., None), (115.01, True), (85., None), (84.99, True)):
        h.g.positions[a] = 100.; h.context.portfolio.positions[a].price = price
        record(f'strict_15pct_{price}', h.call(h.ns['need_rebalance']), {'exception': None, 'return': expected})
    h.context.portfolio.positions[b] = position(10, 10.)
    h.context.portfolio.positions[a].price = 100.
    record('untracked_position_anchor_key', h.call(h.ns['need_rebalance']), {'exception': 'KeyError'})

    # Exact thresholds are checked with whole-number price/value inputs to avoid float ambiguity.
    for name, amount, ratio, cash, avg, orders in (
        ('delta_1000_equal_block', 300, .04, 100000., 0., []),
        ('delta_1010_buy', 299, .04, 1000., 0., [[a, 4000.]]),
        ('relative_25pct_and_cash_equal_pass', 600, .08, 2000., 0., [[a, 8000.]]),
        ('relative_below_25pct_block', 601, .08, 2000., 0., []),
        ('cash_below_quarter_block', 600, .08, 1999., 0., []),
        ('price_equals_internal_cost_block', 600, .08, 2000., 10., []),
        ('price_below_internal_cost_block', 600, .08, 2000., 11., []),
        ('sell_does_not_require_profit_or_cash', 1000, .04, 0., 20., [[a, 4000.]])):
        h = Harness(source, positions={a: position(amount, 10.)}, cash=cash)
        h.g.transactionRecord[a]['avg_cost'] = avg
        record(name, h.call(h.ns['fun_trade'], {a: ratio}), {'exception': None, 'orders': orders})

    for shape, amount, exception, orders in (
        ('numpy', 0, None, []), ('list', 0, 'ZeroDivisionError', []),
        ('numpy', 1000, None, [[a, 0.]]), ('list', 1000, 'ZeroDivisionError', [])):
        h = Harness(source, positions={a: position(amount, 10.)}, history_shape=shape)
        result = h.call(h.ns['fun_trade'], {a: 0.})
        record(f'zero_target_{shape}_holding{amount}', result, {'exception': exception, 'orders': orders})
        require(bool(result['warnings']) == (shape == 'numpy'), 'Numeric zero warning differs')

    for name, amount, value, cash, orders in (
        ('special_delta_100shares_equal_block', 0, 1000., 100000., []),
        ('special_cash_100shares_equal_block', 0, 2000., 1000., []),
        ('special_cash_above_pass', 0, 2000., 1001., [[fund, 2000.]]),
        ('special_sell_no_cash_profit_check', 300, 0., 0., [[fund, 0.]])):
        h = Harness(source, positions={fund: position(amount, 10.)}, cash=cash)
        h.g.transactionRecord[fund]['avg_cost'] = 100.
        record(name, h.call(h.g.QuantLib.fun_tradeBond, fund, value), {'exception': None, 'orders': orders})

    h = Harness(source, positions={b: position(100, 10.), a: position(8000, 10.), 'EXTRA': position(1, 10.)},
                quotes={'EXTRA': 10.})
    record('reduction_first_extra_holding_not_liquidated', h.call(h.ns['fun_trade'], {b: .3, a: .2}),
           {'exception': None, 'orders': [[a, 20000.], [b, 30000.]]})

    for delivery, initial, target, exception, amount, buys, sells in (
        ('deferred', None, 20000., 'KeyError', 0., 0, 0),
        ('deferred', 100., 20000., None, 100., 0, 0),
        ('immediate_keep_zero', None, 20000., None, 2000., 1, 0),
        ('immediate_keep_zero', 100., 0., None, 0., 0, 1),
        ('immediate_delete_zero', 100., 0., 'KeyError', 100., 0, 0)):
        h = Harness(source, positions={} if initial is None else {a: position(initial, 10.)}, delivery=delivery)
        h.g.transactionRecord[a].update(amount=initial or 0., avg_cost=10. if initial else 0.)
        result = h.call(h.g.QuantLib.fun_trade, a, target)
        record(f'order_record_{delivery}_{initial}_{target}', {**result, 'record': h.g.transactionRecord[a]},
               {'exception': exception, 'orders': [[a, target]]})
        require(h.g.transactionRecord[a]['amount'] == amount and h.g.transactionRecord[a]['buy_times'] == buys and
                h.g.transactionRecord[a]['sell_times'] == sells, 'Post-order record differs')

    h = Harness(source, positions={a: position(100., 10.)})
    h.g.transactionRecord[a].update(amount=100., avg_cost=10., standPrice=10.)
    record_states = []
    for amount, price in ((50., 12.), (100., 12.), (0., 7.)):
        h.context.portfolio.positions[a] = position(amount, price)
        h.g.QuantLib.fun_record(h.context, a)
        record_states.append(dict(h.g.transactionRecord[a]))
    record('record_partial_sale_buy_full_loss', {'costs': [r['avg_cost'] for r in record_states],
        'final_amount': record_states[-1]['amount'], 'margin': record_states[-1]['Margin'],
        'max_loss': record_states[-1]['max_loss'], 'loss': record_states[-1]['loss'],
        'buy_times': record_states[-1]['buy_times'], 'sell_times': record_states[-1]['sell_times']},
        {'costs': [8., 10., 0.], 'final_amount': 0., 'margin': -300., 'max_loss': -300.,
         'loss': 1, 'buy_times': 1, 'sell_times': 2})
    h.ns['after_market_close'](h.context)
    require('stock, Win, loss' in h.g.message and 'Returns' in h.g.message, 'Close record output missing')
    record('close_callback_reports_without_orders', {'orders': h.orders, 'hold_before': h.g.hold[a]},
           {'orders': [], 'hold_before': 100.})
    return {'cases': cases, 'case_count': len(cases), 'synthetic_only': True, 'not_a_backtest': True,
            'history_shapes': ['dict[str,numpy.ndarray]', 'dict[str,list]'],
            'platform_history_shape_proven': False, 'platform_immediate_fill_proven': False,
            'synthetic_weights_injected_only_for_callback_cycle': True,
            'real_ES_or_original_weights_computed': False, 'return_semantics': 'pending_D-allweather-fill',
            'limits': ['Synthetic orders are intentions; immediate cases inject API state only',
                       'No fees/cash settlement/lot rounding/NAV simulation or second ledger',
                       'No historical fund events, adjustment, identity, state or fee gates released',
                       'Native zero-target behavior depends on supplied scalar type and position retention']}


def offline_diagnostics(source):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        result = diagnostics(source)
    return {**result, 'socket_HTTP_disabled': True}


def preflight(root, directory):
    bound = binding()
    checkpoint(root, directory, 'Batch89 original callback diagnostics start; synthetic only; accepted258 retained')
    save(directory/'input-binding.json', bound)
    save(directory/'scope.json', {'definitions': sorted(FUNCTIONS), 'class': 'QuantLib(object)',
        'source_file': str(source_path()), 'source_sha256': stage.SOURCE_SHA[0],
        'pinned_original_definitions_unmodified': True, 'new_requests': 0, 'new_trading_backtests': 0})
    return {'status': 'ok', 'formal_acceptance': False}


def study(root, directory):
    require(binding() == read(directory/'input-binding.json'), 'Callback input binding changed')
    save(directory/'diagnostics.json', offline_diagnostics(source_path()))
    return archive(root, 'Batch89 original callback/threshold/lifecycle diagnostics archived; synthetic intentions only; no ES repair or trades')


def verify(root, directory):
    require(binding() == read(directory/'input-binding.json'), 'Callback input binding changed')
    expected = read(directory/'diagnostics.json')
    require(offline_diagnostics(source_path()) == expected, 'Offline callback replay differs')
    save(directory/'offline-verification.json', {'status': 'match', 'differences': 0,
        'case_count': expected['case_count'], 'socket_HTTP_disabled': True,
        'independent_expected_checks': 'Explicit integer/value/state expectations within diagnostic cases',
        'not_a_backtest': True, 'platform_equivalence_proven': False})
    return protect(root, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'study', 'verify'))
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(globals()[args.action](args.root, args.directory))
