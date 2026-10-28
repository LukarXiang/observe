"""SHA-bound original ETF callback diagnostics. Synthetic state, no ledger or NAV."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from observe.runs import file_sha
from scripts import review_strategy_batch94 as stage
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.reference_strategy_batch94 import historical_stage
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch94/20261010-original-order-callbacks')
PARENT = Path('docs/handoff/2026-10-10-batch94-kalman-reference.json')
PARENT_SHA = '03fb11ce657a9fbac67147cf006f47a0cae885694d436fb8cd930597809324a4'
FILES = ('scripts/diagnose_strategy_batch94.py', 'tests/unit/test_batch94_callbacks.py')
A, B = stage.POOL[0], stage.POOL[2]


def encoded(value):
    if isinstance(value, np.ndarray): return {'kind': 'array', 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, str): return {'kind': 'security', 'value': value}
    if isinstance(value, (float, int, np.number)): return {'kind': 'number', 'value': float(value)}
    raise ValueError('Unexpected diagnostic value')


def position(security, amount=100, cost=100., price=100.):
    return SimpleNamespace(security=security, total_amount=amount, avg_cost=cost, price=price, value=amount*price)


class ZeroPositions(dict):
    def __init__(self, insert=False): super().__init__(); self.insert = insert
    def __missing__(self, key):
        value = position(key, 0)
        if self.insert: self[key] = value
        return value


class Harness:
    """Explicit alternative API feedback models, not a second execution engine."""
    def __init__(self, number, mode='reject', mapping='plain', held=(), price=100.):
        self.orders = []; self.events = []; self.mode = mode; self.price = price
        positions = {} if mapping == 'plain' else ZeroPositions(insert=mapping == 'inserting')
        positions.update({s: position(s, price=price) for s in held})
        self.context = SimpleNamespace(current_dt=datetime(2020, 1, 2, 9, 30),
            portfolio=SimpleNamespace(positions=positions, cash=1000.))
        self.g = SimpleNamespace(stock_pool=list(stage.POOL), stock_num=1, momentum_day=20,
            ref_stock='000300.XSHG', N=18, M=600, score_threshold=.7)
        self.namespace = stage.literal(stage.DIRECTORY, number, {'np': np, 'g': self.g,
            'log': SimpleNamespace(info=lambda *args: None, debug=lambda *args: None,
                set_level=lambda *args: self.events.append(['log_level', *args])),
            'OrderStatus': SimpleNamespace(held='held'), 'order_target_value': self.order,
            'set_benchmark': lambda s: self.events.append(['benchmark', s]),
            'set_option': lambda k, v: self.events.append(['option', k, v]),
            'FixedSlippage': lambda v: {'FixedSlippage': v},
            'set_slippage': lambda v: self.events.append(['slippage', v]),
            'OrderCost': lambda **kwargs: kwargs,
            'set_order_cost': lambda cost, **kwargs: self.events.append(['cost', cost, kwargs]),
            'run_daily': lambda f, **kwargs: self.events.append(['schedule', f.__name__, kwargs])})

    def order(self, security, value):
        self.orders.append({'security': encoded(security), 'value': float(value)})
        if self.mode == 'reject': return None
        if self.mode == 'zero': return SimpleNamespace(filled=0, amount=100, status='canceled')
        portfolio = self.context.portfolio
        filled = 30 if self.mode == 'partial' else 100
        status = 'canceled' if self.mode in ('partial', 'canceled_full') else 'held'
        if value == 0:
            held = portfolio.positions[security]
            if self.mode != 'deferred':
                portfolio.cash += held.price*filled
                if filled == held.total_amount: del portfolio.positions[security]
                else: held.total_amount -= filled
        else:
            portfolio.cash -= value*filled/100
            portfolio.positions[security] = position(security, filled, cost=self.price, price=self.price)
        return SimpleNamespace(filled=filled, amount=100, status=status)

    def snapshot(self):
        return {'orders': self.orders, 'positions': [{'security': encoded(s), 'amount': p.total_amount}
            for s, p in self.context.portfolio.positions.items()], 'cash': self.context.portfolio.cash}


SCENARIOS = (
    'initialize_with_seed_spy', 'open_rejected', 'open_zero_fill', 'open_partial_canceled', 'open_full',
    'close_rejected', 'close_zero_fill', 'close_partial_canceled', 'close_full_held', 'close_full_canceled',
    'empty_plain_dict', 'rejected_buy_numeric_score', 'zero_buy_numeric_score', 'rejected_buy_array_score',
    'full_buy_masks_score', 'partial_buy_masks_score', 'inserting_zero_positions_pollute_count',
    'sell_mutates_iteration', 'partial_sell_blocks_buy', 'deferred_sell_blocks_buy',
    'held_target_no_topup', 'KEEP_rotates', 'SELL_rank_before_timing', 'rank_error_blocks_SELL',
    'not_0930_0929', 'not_0930_0931', 'not_0930_1500',
    'stop_exact_minus90', 'stop_above_minus90', 'stop_zero_cost_error', 'stop_snapshot_survives_deletion',
    'trade_then_stop', 'stop_then_trade', 'after_close_prints_trades',
)


def run_case(number, name):
    require(name in SCENARIOS, 'Unknown callback case')
    mode = 'reject'; mapping = 'plain'; held = (); price = 100.
    if name.startswith('open_') or name.startswith('close_'):
        if 'zero' in name: mode = 'zero'
        elif 'partial' in name: mode = 'partial'
        elif 'full' in name: mode = 'canceled_full' if name == 'close_full_canceled' else 'full'
        if name.startswith('close_'): held = (A,)
    if name in ('rejected_buy_numeric_score', 'zero_buy_numeric_score', 'rejected_buy_array_score',
            'full_buy_masks_score', 'partial_buy_masks_score', 'inserting_zero_positions_pollute_count',
            'KEEP_rotates', 'trade_then_stop', 'stop_then_trade'): mapping = 'virtual'
    if name == 'inserting_zero_positions_pollute_count': mapping = 'inserting'
    if name in ('full_buy_masks_score', 'trade_then_stop', 'stop_then_trade', 'stop_exact_minus90',
            'stop_snapshot_survives_deletion', 'sell_mutates_iteration', 'SELL_rank_before_timing'): mode = 'full'
    if name == 'partial_buy_masks_score' or name == 'partial_sell_blocks_buy': mode = 'partial'
    if name == 'deferred_sell_blocks_buy': mode = 'deferred'
    if name == 'zero_buy_numeric_score': mode = 'zero'
    if name in ('partial_sell_blocks_buy', 'deferred_sell_blocks_buy', 'KEEP_rotates'): held = (B,)
    if name == 'KEEP_rotates': mode = 'deferred'
    if name == 'held_target_no_topup': held = (A,)
    if name == 'rank_error_blocks_SELL': held = (B,)
    if name in ('trade_then_stop', 'stop_then_trade', 'stop_exact_minus90', 'stop_above_minus90', 'stop_zero_cost_error'):
        held = (A,); price = 10.
    if name in ('sell_mutates_iteration', 'SELL_rank_before_timing', 'stop_snapshot_survives_deletion'): held = (A, B)
    if name == 'stop_snapshot_survives_deletion': price = 10.
    h = Harness(number, mode, mapping, held, price); ns = h.namespace; context = h.context
    result = None; error = None; steps = []; output = StringIO()
    with redirect_stdout(output):
        try:
            if name == 'initialize_with_seed_spy':
                ns['initial_slope_series'] = lambda: list(range(600))
                ns['initialize'](context)
                result = {'seed_count': len(h.g.slope_series), 'seed_last': h.g.slope_series[-1],
                    'stock_pool': h.g.stock_pool, 'stock_num': h.g.stock_num, 'N': h.g.N, 'M': h.g.M,
                    'threshold': h.g.score_threshold, 'events': h.events}
            elif name.startswith('open_'): result = ns['open_position'](A, 1000.)
            elif name.startswith('close_'): result = ns['close_position'](context.portfolio.positions[A])
            elif name in ('stop_exact_minus90', 'stop_above_minus90', 'stop_zero_cost_error', 'stop_snapshot_survives_deletion'):
                if name == 'stop_above_minus90': context.portfolio.positions[A].price = 10.001
                if name == 'stop_zero_cost_error': context.portfolio.positions[A].avg_cost = 0.
                ns['check_lose'](context)
            elif name == 'after_close_prints_trades':
                ns['get_trades'] = lambda: {'t1': 'synthetic trade'}
                ns['print_trade_info'](context)
            elif name.startswith('not_0930_') or name in ('KEEP_rotates', 'SELL_rank_before_timing',
                    'rank_error_blocks_SELL', 'trade_then_stop', 'stop_then_trade'):
                def rank(*args):
                    steps.append('rank')
                    if name == 'rank_error_blocks_SELL': raise KeyError('original date-index shape failure')
                    return [A, .125]
                def timing(*args):
                    steps.append('timing')
                    return 'SELL' if name in ('SELL_rank_before_timing', 'rank_error_blocks_SELL') else 'KEEP'
                ns['get_rank'] = rank; ns['get_timing_signal'] = timing
                if name == 'not_0930_0929': context.current_dt = datetime(2020, 1, 2, 9, 29)
                if name == 'not_0930_0931': context.current_dt = datetime(2020, 1, 2, 9, 31)
                if name == 'not_0930_1500': context.current_dt = datetime(2020, 1, 2, 15)
                if name == 'stop_then_trade': ns['check_lose'](context)
                ns['my_trade'](context)
                if name == 'trade_then_stop': ns['check_lose'](context)
            else:
                rank = [A, np.array([.125])] if name == 'rejected_buy_array_score' else [A, .125]
                ns['adjust_position'](context, rank)
        except (KeyError, TypeError, RuntimeError, ZeroDivisionError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'case': name, 'source_number': number, 'synthetic': True, 'result': result,
        'error': error, 'steps': steps, 'printed': output.getvalue(), **h.snapshot()}


def diagnostics():
    rows = [run_case(number, name) for number in range(3) for name in SCENARIOS]
    return {'cases': rows, 'case_count': len(rows), 'unique_scenarios': len(SCENARIOS),
        'source_sha256': list(stage.SOURCE_SHA), 'platform_top_level_executed': False,
        'synthetic_API_feedback_models': ['plain dict', 'virtual noninserting zero', 'inserting zero',
            'rejected/zero/partial/full/canceled anddeferred feedback'],
        'platform_feedback_equivalence_proven': False, 'duplicate_is_independent_economic_sample': False,
        'new_requests': 0, 'new_trading_backtests': 0, 'not_a_trading_backtest': True}


def parent_binding():
    historical_stage()
    require(file_sha(PARENT) == PARENT_SHA, 'Parent reference receipt changed')
    doc = read(PARENT)
    require(file_sha(doc['stage_binding']) == doc['stage_binding_sha256'] and
        file_sha(doc['report']) == doc['report_sha256'], 'Parent reference binding changed')
    for name, sha in doc['payload_evidence_sha256'].items():
        require(file_sha(name) == sha, 'Parent payload changed')
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == sha, 'Parent implementation changed')
    return {'parent_receipt': str(PARENT), 'parent_receipt_sha256': PARENT_SHA,
        'source_sha256': list(stage.SOURCE_SHA), 'not_a_trading_backtest': True}


def validate_implementations(directory):
    """Reject incomplete inventories and check live AND frozen implementation bytes."""
    root = Path.cwd().resolve()

    def regular(path):
        require(path.is_file() and path.resolve().is_relative_to(root) and
            not any(p.is_symlink() for p in (path, *path.parents)),
            'Callback implementation path is not a rooted regular file')

    manifest = directory/'implementation-binding.json'
    regular(manifest)
    hashes = read(manifest)
    require(isinstance(hashes, dict) and set(hashes) == set(FILES),
        'Callback implementation inventory differs')
    for name in FILES:
        for path in (Path(name), directory/'implementation'/name):
            regular(path)
            require(file_sha(path) == hashes[name], 'Callback implementation changed: '+str(path))
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compute', 'offline'))
    args = parser.parse_args(); bound = parent_binding()
    if args.action == 'compute':
        require(not DIRECTORY.exists(), 'Callback directory already exists')
        with network_blocked(): result = diagnostics()
        require(result['case_count'] == 102, 'Callback scope differs')
        DIRECTORY.mkdir(parents=True, exist_ok=False)
        save(DIRECTORY/'parent-binding.json', bound); save(DIRECTORY/'diagnostics.json', result)
        for name in FILES:
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
        save(DIRECTORY/'implementation-binding.json', {name: file_sha(name) for name in FILES})
        print(json.dumps({'status': 'computed', 'cases': result['case_count'], 'new_trading_backtests': 0}))
    else:
        require(read(DIRECTORY/'parent-binding.json') == bound, 'Callback parent differs')
        validate_implementations(DIRECTORY)
        with network_blocked(): result = diagnostics()
        require(result == read(DIRECTORY/'diagnostics.json'), 'Callback replay differs')
        save(DIRECTORY/'offline-verification.json', {'status': 'match', 'differences': 0,
            'cases': result['case_count'], 'network_forbidden': True, 'not_a_trading_backtest': True,
            'new_trading_backtests': 0, 'diagnostics_sha256': file_sha(DIRECTORY/'diagnostics.json')})
        print(json.dumps({'status': 'match', 'cases': result['case_count'], 'differences': 0}))


if __name__ == '__main__': main()
