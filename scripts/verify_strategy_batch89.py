"""Accept all-weather source research; pending returns and fund trading stay blocked."""
import argparse
import ast
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source
from scripts import diagnose_strategy_batch89 as callbacks
from scripts import review_strategy_batch89 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch89/20261009-all-weather-source-acceptance')
STAGE_RECEIPT = Path('docs/handoff/2026-10-09-batch89-callback-events-verification.json')
RECEIPT = Path('docs/handoff/2026-10-09-batch89-verification.json')
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch89.py',
         'tests/unit/test_batch89_verification.py', *callbacks.FILES,
         'scripts/probe_strategy_batch89_events.py', 'tests/unit/test_batch89_events.py',
         'scripts/review_strategy_batch89.py', 'tests/unit/test_batch89_stage.py')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
        'source-reviews/review.json', 'risk-diagnostics.json', 'offline-verification.json',
        'offline-catalog.json', 'checked-state.json'}


def source_checks():
    rows = read(stage.DIRECTORY/'sources-frozen.json')
    require(len(rows) == 2, 'Original source scope changed')
    for source, sha, row in zip(stage.SOURCES, stage.SOURCE_SHA, rows, strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码')/source) and
                file_sha(row['source_path']) == file_sha(row['source_copy']) == sha, 'Original source changed')
        tree = ast.parse(read_source(Path(row['source_copy']))[0])
        require(hashlib.sha256(ast.dump(tree).encode()).hexdigest() == stage.AST_SHA, 'Same AST relation changed')
    return rows


def binding(directory=None):
    receipt = read(STAGE_RECEIPT)
    require(receipt['status'] == 'stage_checks_passed' and not receipt['formal_acceptance'] and
            receipt['callback_cases'] == 32 and receipt['new_real_ES_weights_or_original_signals'] == 0 and
            receipt['new_trading_backtests'] == 0, 'Parent stage scope changed')
    for group in ('implementation_sha256', 'evidence_sha256'):
        require(all(file_sha(name) == sha for name, sha in receipt[group].items()), 'Parent callback stage changed')
    require(file_sha(receipt['parent_receipt']) == receipt['parent_receipt_sha256'] and
            file_sha(receipt['stage_report']) == receipt['stage_report_sha256'], 'Parent document changed')
    source_checks()
    result = {'snapshot': stage.SNAPSHOT, 'parent_receipt': str(STAGE_RECEIPT),
        'parent_receipt_sha256': file_sha(STAGE_RECEIPT), 'return_semantics': 'pending_D-allweather-fill',
        'new_requests': 0, 'not_a_backtest': True}
    if directory is not None:
        require(read(directory/'input-binding.json') == result, 'Acceptance input binding changed')
    return result


def helpers(source, daily_returns, confidence=2.58):
    require(file_sha(source) == stage.SOURCE_SHA[0], 'Unreviewed risk source')
    tree = ast.parse(read_source(Path(source))[0])
    parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fun_calc_trade_ratio')
    nested = [n for n in parent.body if isinstance(n, ast.FunctionDef)]
    require([n.name for n in nested] == ['__fun_getdailyreturn', '__fun_get_portfolio_ES',
            '__fun_calc_stock_risk_ES', '__fun_calc_trade_ratio'] and all(safe_function(n) for n in nested),
            'Original nested helper scope changed')
    calls = []
    def synthetic_returns(stock, freq, lag):
        calls.append({'stock': stock, 'freq': freq, 'lag': lag})
        return np.array(daily_returns[stock], dtype=float)
    namespace = {'__fun_getdailyreturn': synthetic_returns, 'math': math,
                 'g': SimpleNamespace(confidence_level=confidence)}
    # Isolate unchanged arithmetic helpers; the original incompatible price-return helper stays inert.
    exec(compile(ast.Module(body=nested[1:], type_ignores=[]), '<batch89-pinned-risk-helpers>', 'exec'), namespace)
    return namespace, calls


class SyntheticReturnAPI:
    """Record the literal call chain and return injected values, without price computation."""

    def __init__(self, operands, operations):
        self.operands = np.array(operands, dtype=float)
        self.operations = operations

    def resample(self, frequency):
        require(frequency == 'D', 'Original resample frequency changed')
        self.operations.append(['resample', frequency])
        return self

    def last(self):
        self.operations.append(['last'])
        return self

    def pct_change(self):
        self.operations.append(['pct_change_default'])
        return self

    def fillna(self, *, value, method, axis):
        require((value, method, axis) == (0, None, 0), 'Original fill arguments changed')
        self.operations.append(['fillna', value, method, axis])
        return self

    @property
    def iloc(self):
        return self

    def __getitem__(self, key):
        require(key == (slice(None), 0), 'Original first-column extraction changed')
        self.operations.append(['iloc', 'all_rows', 0])
        return SimpleNamespace(values=self.operands)


def synthetic_full_weights(returns):
    calls = []; operations = []
    def history(lag, frequency, field, stock, df):
        require((lag, frequency, field, df) == (120, '1d', 'close', True), 'Original risk-history arguments changed')
        calls.append(stock)
        return SyntheticReturnAPI(returns[stock], operations)
    function = stage.original_calc(stage.DIRECTORY, history)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            weights = function(None)
            result = {'weights': {key: float(value) for key, value in weights.items()}, 'exception': None}
        except ZeroDivisionError as exc:
            result = {'exception': type(exc).__name__, 'message': str(exc)}
    return {**result, 'history_stock_calls': calls, 'synthetic_operations': operations,
            'warnings': [str(w.message) for w in caught]}


def risk_diagnostics(source):
    a, b, c, fund = stage.POOL
    rows = []

    def scalar(name, actual, expected, details=None):
        require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14), f'{name}: scalar differs')
        rows.append({'case': name, 'actual': float(actual), 'expected': float(expected), **(details or {})})

    # Equal negative operands make the selected count independently recoverable from the ES value.
    for confidence, probability, count in ((1.96, .95, 7), (2.06, .96, 5), (2.18, .97, 4),
                                          (2.34, .98, 3), (2.58, .99, 2), (5., .99999, 1), (0., .95, 7)):
        ns, calls = helpers(source, {a: [-.01]*166})
        actual = ns['__fun_get_portfolio_ES']({a: 1.}, '1d', 120, confidence)
        denominator = Decimal.from_float(1-probability)*120
        expected = Decimal.from_float(.01)*count/denominator
        recovered = round(float(actual)*float(denominator)/.01)
        require(recovered == count, 'Independent selected-count expectation differs')
        scalar(f'ES_confidence_{confidence}', actual, expected, {'calendar_return_rows': 166,
            'lag': 120, 'denominator_hex': float(denominator).hex(), 'selected_count_from_equal_operands': recovered,
            'expected_selected_count': count, 'history_calls': calls})

    ns, calls = helpers(source, {a: [-.01, -.02], b: [-.1, -.2]})
    scalar('ES_short_array_still_divides_by_lag_probability',
           ns['__fun_get_portfolio_ES']({a: 1.}, '1d', 120, 1.96), Decimal('.03')/Decimal.from_float((1-.95)*120))
    calls.clear()
    scalar('ES_empty_ratio_no_history', ns['__fun_get_portfolio_ES']({}, '1d', 120, 2.58), 0)
    require(calls == [], 'Empty portfolio queried synthetic returns')
    scalar('ES_only_first_ratio_key', ns['__fun_get_portfolio_ES']({a: .2, b: .8}, '1d', 120, 2.58),
           Decimal('.03')/Decimal.from_float((1-.99)*120), {'ratio_weights_ignored': True})
    scalar('ES_key_order_changes_operand', ns['__fun_get_portfolio_ES']({b: .8, a: .2}, '1d', 120, 2.58),
           Decimal('.3')/Decimal.from_float((1-.99)*120))
    value, weights = ns['__fun_calc_stock_risk_ES']([a, b])
    scalar('risk_stock_list_uses_first_only', value, Decimal('.03')/Decimal.from_float((1-.99)*120),
           {'ratio': weights, 'expected_ratio': {a: 1}})
    require(weights == {a: 1}, 'First-stock risk mapping changed')
    value, weights = ns['__fun_calc_stock_risk_ES']([])
    scalar('risk_empty_class', value, 0, {'ratio': weights})
    require(weights == {}, 'Empty class created stock weights')
    for name, operands, expected in (('risk_all_zero', [0.]*166, 0.),
                                     ('risk_all_positive_is_negative', [.01]*166, -.02/((1-.99)*120)),
                                     ('risk_nan_is_reset_to_zero', [math.nan], 0.)):
        ns, _ = helpers(source, {a: operands})
        value, _ = ns['__fun_calc_stock_risk_ES']([a])
        scalar(name, value, expected)
    ns, _ = helpers(source, {a: [-math.inf]})
    value, _ = ns['__fun_calc_stock_risk_ES']([a])
    require(value == math.inf, 'Infinite original ES was unexpectedly filtered')
    rows.append({'case': 'risk_infinite_is_retained', 'actual_hex': float(value).hex(),
                 'expected_hex': 'inf', 'trade_usable': False})

    ns, _ = helpers(source, {})
    weight = ns['__fun_calc_trade_ratio']
    for name, positions, total, expected in (
        ('weights_round_total_below_one', [1., 1., 1., 0.], 3., {a: .333, b: .333, c: .333, fund: 0.}),
        ('weights_round_total_above_one', [.3336, .3336, .3328, 0.], 1., {a: .334, b: .334, c: .333, fund: 0.})):
        result = {}
        for stock, pos in zip(stage.POOL, positions, strict=True):
            result = weight(result, [stock], {stock: 1.}, pos, total)
        require(result == expected, f'{name}: independent rounded weights differ')
        rows.append({'case': name, 'synthetic_positions': positions, 'synthetic_total_position': total,
                     'actual': result, 'expected': expected, 'rounded_total': math.fsum(result.values())})
    result = weight({a: .25}, [a], {a: 1.}, 1., 4.)
    require(result == {a: .5}, 'Repeated stock helper accumulation changed')
    rows.append({'case': 'weights_existing_key_accumulates', 'actual': result, 'expected': {a: .5}})
    for name, stocks, ratios, position, total, exception in (
        ('weights_zero_total_raises', [a], {a: 1.}, 0., 0., 'ZeroDivisionError'),
        ('weights_multi_stock_class_mapping_missing', [a, b], {a: 1.}, 1., 1., 'KeyError')):
        try:
            weight({}, stocks, ratios, position, total)
        except (ZeroDivisionError, KeyError) as exc:
            require(type(exc).__name__ == exception, 'Unexpected helper failure')
            rows.append({'case': name, 'exception': type(exc).__name__, 'expected_exception': exception,
                         'original_active_pool_multi_stock': False})
        else:
            raise AssertionError('Expected original helper failure')
    result = weight({}, [], {}, 0., 0.)
    require(result == {}, 'Empty class unexpectedly divides by zero')
    rows.append({'case': 'weights_empty_class_no_division', 'actual': result, 'expected': {}})
    for name, values, expected in (
        ('full_synthetic_four_class_inverse_risk', [-.01, -.02, -.03, -.04], {a: .48, b: .24, c: .16, fund: .12}),
        ('full_synthetic_round_below_one', [-.01, -.01, -.01, 0.], {a: .333, b: .333, c: .333, fund: 0.}),
        ('full_synthetic_round_above_one', [-1/.3336, -1/.3336, -1/.3328, 0.], {a: .334, b: .334, c: .333, fund: 0.}),
        ('full_synthetic_negative_risks_not_rejected', [.01, .02, .03, .04], {a: .48, b: .24, c: .16, fund: .12})):
        result = synthetic_full_weights({stock: [value]*166 for stock, value in zip(stage.POOL, values, strict=True)})
        require(result['exception'] is None and result['weights'] == expected and
                result['history_stock_calls'] == list(stage.POOL), f'{name}: full synthetic expectations differ')
        rows.append({'case': name, 'actual': result, 'expected_weights': expected,
                     'synthetic_return_values': values, 'native_pandas_computation': False})
    result = synthetic_full_weights({stock: [0.]*166 for stock in stage.POOL})
    require(result['exception'] == 'ZeroDivisionError', 'Full zero-risk original pipeline did not fail')
    rows.append({'case': 'full_synthetic_zero_risk_raises', 'actual': result,
                 'expected_exception': 'ZeroDivisionError'})
    result = synthetic_full_weights({stock: [-math.inf] for stock in stage.POOL})
    require(result['exception'] is None and all(math.isnan(v) for v in result['weights'].values()) and
            result['warnings'], 'Full infinite-risk original behavior changed')
    result['weights'] = {key: value.hex() for key, value in result['weights'].items()}
    rows.append({'case': 'full_synthetic_infinite_risk_nan_weights', 'actual': result,
                 'expected_weight_hex': {stock: 'nan' for stock in stage.POOL}, 'trade_usable': False})
    return {'cases': rows, 'case_count': len(rows), 'synthetic_only': True, 'not_a_backtest': True,
        'independent_reference': 'Explicit selected counts and Decimal.from_float rational arithmetic; literal rounded dictionaries',
        'source_helpers_body_unmodified': True, 'helper_daily_returns_synthetically_injected': True,
        'original_full_weight_function_executed': True, 'synthetic_price_return_API_adapter': True,
        'original_prices_or_pandas_returns_computed': False,
        'real_ES_or_original_weights_computed': False, 'return_semantics': 'pending_D-allweather-fill',
        'limits': ['Isolated helpers and full original function run only on explicit synthetic return/API contracts',
                   'The synthetic API records resample/pct_change/fillna calls without implementing any of their price semantics',
                   'Only original confidence2.58 is active; other confidence branches and multi-stock classes are boundary diagnostics',
                   'No candidate historical returns filled, adjusted, computed or substituted',
                   'Nonfinite/negative risk and rounded totals retained; no unapproved economic repair']}


def network_disabled(function):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        return function()


def preflight(root, directory):
    bound = binding()
    callbacks.binding()
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch89 formal source research starts; return repair pending; no historical weights or trades')
    save(directory/'input-binding.json', bound)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    return {'status': 'ok', 'formal_trading_acceptance': False}


def audit(root, directory):
    binding(directory)
    result = network_disabled(lambda: risk_diagnostics(callbacks.source_path()))
    save(directory/'risk-diagnostics.json', {**result, 'socket_HTTP_disabled': True})
    return {'status': 'ok', 'risk_cases': result['case_count'], 'new_trading_backtests': 0}


def catalog(root, directory):
    binding(directory)
    require(read(directory/'risk-diagnostics.json')['real_ES_or_original_weights_computed'] is False, 'Risk scope changed')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}
    rows = []
    for name, frozen in zip(stage.SOURCES, source_checks(), strict=True):
        require(name in REVIEWS and old[name]['review_status'] != '人工审查完成', 'Missing/new source review differs')
        rows.append({**frozen, 'formal_acceptance': 'source_research_only', 'review': REVIEWS[name], 'newly_reviewed': True})
    now = catalog_strategies(root, 'repo/量化策略源代码')
    records = read(Path(now['output'])/'catalog.json')
    require(len(old) == len(records) == 695 and
            {r['path'] for r in records if r != old[r['path']]} == set(stage.SOURCES), 'Unrelated source catalog changed')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    save(folder/'review.json', {'sources': rows, 'snapshot': stage.SNAPSHOT, 'catalog': now,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'same_AST_sha256': stage.AST_SHA, 'different_article_headers_not_byte_duplicates': True,
        'original_trading_complete': False, 'not_a_backtest': True})
    return archive(root, 'Batch89 two all-weather source rules reviewed; final checks pending; synthetic arithmetic only; no trades')


def validate_sources(directory):
    binding(directory)
    doc = read(directory/'source-reviews/review.json')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(stage.SOURCES, doc['sources'], strict=True):
        require(row['review'] == REVIEWS[name], 'Reviewed rules changed')


def offline(root, directory):
    validate_sources(directory)
    result = network_disabled(lambda: risk_diagnostics(callbacks.source_path()))
    require({**result, 'socket_HTTP_disabled': True} == read(directory/'risk-diagnostics.json'), 'Risk diagnostic replay differs')
    callback_result = network_disabled(lambda: callbacks.offline_diagnostics(callbacks.source_path()))
    require(callback_result == read(callbacks.DIRECTORY/'diagnostics.json'), 'Old callback replay differs')
    off = network_disabled(lambda: offline_catalog(root, directory))
    save(directory/'offline-catalog.json', off)
    validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'status': 'match', 'differences': 0,
        'risk_cases': result['case_count'], 'callback_cases': callback_result['case_count'],
        'socket_HTTP_disabled': True, 'synthetic_only': True, 'not_a_backtest': True,
        'platform_equivalence_proven': False})
    return {'status': 'match', 'differences': 0, 'risk_cases': result['case_count']}


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
             'scripts/verify_strategy_batch89.py', 'scripts/diagnose_strategy_batch89.py',
             'scripts/probe_strategy_batch89_events.py'],
            [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch89_verification.py',
             'tests/unit/test_batch89_callbacks.py', 'tests/unit/test_batch89_events.py', 'tests/unit/test_batch89_stage.py',
             'tests/unit/test_batch88_verification.py', 'tests/unit/test_research_metadata.py',
             'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
             'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    require(CORE-{'checked-state.json'} <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()},
            'Core evidence missing')
    before = {name: file_sha(name) for name in FILES}; logs = []; folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for i, command in enumerate(commands()):
        path = folder/f'{i}.log'
        with path.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        logs.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed; preserve log: {path}')
    require(before == {name: file_sha(name) for name in FILES}, 'Implementation changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': logs, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': logs}


def finish(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == {n: file_sha(n) for n in FILES} and
            CORE-{'checked-state.json'} <= checked['evidence_sha256'].keys() and
            [r['command'] for r in checked['commands']] == commands(), 'Frozen check scope changed')
    require(all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Check logs changed')
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
            'Frozen code copies changed')
    require(off['status'] == 'match' and off['differences'] == 0 and off['socket_HTTP_disabled'] and
            off['callback_cases'] == 32 and off['risk_cases'] == 29, 'Offline scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed')
    protected = protect(root, directory)
    progress = archive(root, 'Batch89 two same-AST all-weather source reviews accepted;29 risk/32 callback synthetic cases match0; real weights/trades0; return variant pending')
    save(RECEIPT, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': binding(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked,
        'risk_diagnostics': read(directory/'risk-diagnostics.json'), 'offline_verification': off,
        'offline_catalog': read(directory/'offline-catalog.json'), 'protection': protected, 'progress': progress,
        'newly_reviewed': 2, 'new_real_ES_weights_or_original_signals': 0, 'new_trading_backtests': 0,
        'return_variant': 'pending_D-allweather-fill', 'formal_source_research_acceptance': True,
        'original_strategy_complete': False, 'not_a_backtest': True})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(RECEIPT),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'audit', 'catalog', 'offline', 'checks', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
