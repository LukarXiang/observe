"""Review duplicate low-momentum and intraday low-price sources without inventing trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
from heapq import nlargest
import inspect
from io import StringIO
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

from observe.data.store import Store, fingerprint
from observe.runs import canonical, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch64 import safe_default
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch67/20261008-momentum-low-price-json-recovery')
SNAPSHOT = '20261006-145049-e06c'
ACCEPTED = Path('data/staging/strategies-batch64/20261008-price-shape-candles-recovery')
RECEIPT = Path('docs/handoff/2026-10-07-batch64-verification.json')
SOURCES = ('2024年度精选策略2/59.年化62%的动量策略.txt',
    '2024年度精选策略1/25.低价股优化，18年至今10625.40%，已加防未来函数.txt')
SOURCE_SHA = ('31ede8b8c288864c2315ab32addd7efde4ca88031ef8832b170731d03ec7639c',
    'd9e4fe2d88b8f700647098221f7728b22b0b9236f014043c82c549c956e8707b')
CANONICAL = '2024年度精选策略1/49.年化62%的动量策略.txt'
REFERENCES = (('repo/backtrader', 'backtrader/indicators/momentum.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_simulation/simulation_broker.py'),
    ('repo/akshare', 'akshare/stock/stock_info.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'))
FILES = ('scripts/review_strategy_batch67.py', 'tests/unit/test_batch67_research.py',
    'src/observe/strategy_catalog.py', 'src/observe/ledger/book.py', 'src/observe/execution.py',
    'src/observe/data/store.py', 'src/observe/data/prices.py', 'scripts/archive_strategy_progress.py',
    'scripts/run_strategy_batch4.py', 'scripts/review_strategy_batch18.py',
    'scripts/review_strategy_batch32.py', 'scripts/review_strategy_batch58.py',
    'scripts/review_strategy_batch64.py', 'scripts/verify_strategy_batch3.py')
PROBES = ('old-minute', 'sz-delisted')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'sources-frozen.json',
    'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'price-input.parquet', 'calendar-input.parquet', 'probes.json', 'supplement-decision.json',
    'component-research.json', 'component-values.parquet', 'diagnostics.json',
    'offline-verification.json', 'offline-catalog.json', 'dependency-inputs.json', 'stock-input-audit.json',
    'fingerprint-failure-binding.json', 'recovery-binding.json', 'diagnostic-json-failure-binding.json'}


def implementation():
    return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    proof = read(ACCEPTED / 'final-binding-verification.json'); receipt = read(RECEIPT)
    require(proof['status'] == receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch64 changed')
    inputs = read(ACCEPTED / 'input-binding.json')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json')}
    for name in ('price', 'calendar'):
        path = ACCEPTED / f'{name}-input.parquet'
        require(file_sha(path) == inputs[f'{name}_sha256'] ==
            receipt['checks']['evidence_sha256'][f'{name}-input.parquet'], 'Frozen input SHA differs')
        result.update({f'{name}_file': str(path), f'{name}_sha256': file_sha(path)})
    if directory is not None:
        require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Published input changed')
    return result


def api_evidence():
    import akshare as ak
    import baostock as bs
    return {'pandas': pd.__version__, 'numpy': np.__version__, 'akshare': ak.__version__,
        'minute_signature': str(inspect.signature(ak.stock_zh_a_hist_min_em)),
        'minute_source': inspect.getsource(ak.stock_zh_a_hist_min_em),
        'delisted_source': inspect.getsource(ak.stock_info_sz_delist),
        'historical_universe_source': inspect.getsource(bs.query_all_stock),
        'limits': ['Installed minute API declares 1/5/15/30/60, not original 50m bars',
            'Historical date argument alone does not prove historical names; batch65 documented future names']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {n: len(store.load_state(state, n)) for n in
            ('instruments', 'instrument_status', 'bars_1m', 'index_constituents')},
        'historical_names_evidence': {'file': 'docs/tasks/2026-10-08-历史名称与股票池第一方核查.md',
            'sha256': file_sha('docs/tasks/2026-10-08-历史名称与股票池第一方核查.md')},
        'not_a_backtest': True,
        'limits': ['Minute rows do not prove selected historical whole-pool stock or 50m query coverage',
            'No historical name/status/identity chain or full pre-open candidate pool was admitted']}


def start(root, directory):
    bound = binding(root)
    require((directory / 'baseline.json').exists(), 'Create the existing checkpoint baseline before edits')
    frozen = read(directory / 'sources-frozen.json'); old_pointer = read(directory / 'catalog-before.json')
    prior = Path(old_pointer['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    require(file_sha(Path('repo/量化策略源代码') / CANONICAL) == SOURCE_SHA[0] and
        old[CANONICAL]['review_status'] == '人工审查完成', 'Canonical duplicate not proven')
    rows = []
    for i, name in enumerate(SOURCES):
        path = Path('repo/量化策略源代码') / name; copied = directory / 'source-reviews' / (strategy_id(name)+'.source')
        text, encoding = read_source(copied)
        require(old[name]['review_status'] != '人工审查完成' and frozen[i] == {'path': name, 'sha256': SOURCE_SHA[i]} and
            file_sha(path) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source changed/reviewed')
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()),
            'definition_ast': definitions(ast.parse(text)), 'review': REVIEWS[name], 'newly_reviewed': True,
            'duplicate_of': old[name]['duplicate_of']})
    folder = directory / 'references'; folder.mkdir(exist_ok=False); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True,
                text=True, check=True).stdout.strip()})
    for name in ('price', 'calendar'):
        shutil.copyfile(bound[f'{name}_file'], directory / f'{name}-input.parquet')
    save(directory / 'input-binding.json', bound)
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'snapshot': SNAPSHOT,
        'catalog': catalog, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'canonical_duplicate': {'path': CANONICAL, 'sha256': SOURCE_SHA[0]}, 'not_a_backtest': True})
    return archive(root, 'Batch67 complete momentum duplicate and low-price intraday rules archived; no trades')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(safe_default(d) for d in n.args.defaults) and
        all(d is None or safe_default(d) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch67-functions>', 'exec'), namespace)
    return namespace


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    require(file_sha(Path('repo/量化策略源代码') / CANONICAL) == SOURCE_SHA[0], 'Canonical changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Original rules changed')
    require(len(doc['sources']) == 2 and len(doc['references']) == len(REFERENCES), 'Review scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and
            file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference bytes changed')
        require(subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True,
            text=True, check=True).stdout.strip() == row['commit'], 'Reference HEAD changed')
    for name in ('price', 'calendar'):
        require(file_sha(directory / f'{name}-input.parquet') == read(directory / 'input-binding.json')[f'{name}_sha256'], 'Operand changed')
    failure = read(directory / 'fingerprint-failure-binding.json')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 9 passed' in red['stdout'] and
        'Dependency snapshot SHA differs' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Fingerprint red regression binding changed')
    failure = read(directory / 'diagnostic-json-failure-binding.json')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'JSON failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 10 passed' in red['stdout'] and
        'test_diagnostic_result_matches_json_storage_roundtrip' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'JSON red regression binding changed')
    validate_recovery(directory)


def validate_recovery(directory):
    seen = {directory.resolve()}; current = directory
    require((current / 'recovery-binding.json').is_file(), 'Missing recovery binding')
    while (current / 'recovery-binding.json').is_file():
        recovered = read(current / 'recovery-binding.json'); original = Path(recovered['original_directory'])
        require(original.resolve() not in seen and recovered['original_left_read_only'] is True, 'Recovery scope changed')
        seen.add(original.resolve())
        for name, sha in recovered['original_files_sha256'].items():
            require(file_sha(original / name) == sha, 'Original failed attempt changed')
        for row in recovered['failure_files'].values():
            require(file_sha(row['file']) == row['sha256'], 'Recovery failure evidence changed')
        current = original


def numeric_values(price, dates, tree):
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_momentum')
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign) and
        len(n.targets) == 1 and ast.unparse(n.targets[0]) == 'stock_momentum[stock]']
    require(len(assignments) == 1, 'Original momentum expression changed')
    expression = compile(ast.Expression(assignments[0].value), '<frozen-momentum-expression>', 'eval')
    records = []; errors = {'momentum91': 0., 'mean5': 0.}; skipped = 0
    require(dates.is_unique and dates.is_monotonic_increasing and not price.duplicated(['date', 'instrument']).any(), 'Duplicate calendar/price')
    for stock, frame in price.groupby('instrument', sort=True):
        frame = frame.sort_values('date').copy(); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.isin(dates).all() and frame.is_trading.notna().all(), 'Unknown date/trading status')
        absent = (~dates.isin(frame.date)).astype(int).cumsum()
        active = frame[frame.is_trading].reset_index(drop=True); skipped += int((~frame.is_trading).sum())
        positions = dates.get_indexer(active.date); adjusted = active.close_adj.to_numpy(dtype=float)
        # The 91-row query skips only explicit pauses; the 5-row component uses calendar rows.
        by_day = frame.set_index('date').reindex(dates)
        for end, row in enumerate(active.itertuples()):
            position = positions[end]; basis = float(row.back_factor)
            record = {'date': row.date.strftime('%Y-%m-%d'), 'instrument': stock,
                'momentum91': math.nan, 'momentum91_status': 'insufficient_91_rows',
                'raw_close_ge5': bool(row.close >= 5), 'mean5': math.nan, 'mean5_status': 'insufficient_5_calendar_rows'}
            if end >= 90:
                window = adjusted[end-90:end+1]
                if absent[position] - absent[positions[end-90]]: reason = 'unknown_internal_calendar_day'
                elif not np.isfinite(window).all() or not (window > 0).all(): reason = 'invalid_price'
                else:
                    reason = 'eligible'; value = float(eval(expression, {'df': {'close': window}, '__builtins__': {}}))
                    expected = float(window[-1]/window[0]-1)
                    require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), 'Momentum arithmetic differs')
                    record['momentum91'] = value; errors['momentum91'] = max(errors['momentum91'], abs(value-expected))
                record['momentum91_status'] = reason
            if position >= 4:
                window = by_day.iloc[position-4:position+1]
                if window.instrument.isna().any(): reason = 'unknown_calendar_day'
                elif not window.is_trading.eq(True).all(): reason = 'known_pause_without_platform_fill_proof'
                elif not math.isfinite(basis) or basis <= 0 or not np.isfinite(window.close_adj).all() or not window.close_adj.gt(0).all(): reason = 'invalid_price_or_basis'
                else:
                    reason = 'eligible'; anchored = window.close_adj.to_numpy(dtype=float)/basis
                    value = float(pd.Series(anchored).mean()); expected = math.fsum(anchored)/5
                    require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10), 'Mean arithmetic differs')
                    record['mean5'] = value; errors['mean5'] = max(errors['mean5'], abs(value-expected))
                record['mean5_status'] = reason
            records.append(record)
    values = pd.DataFrame(records)
    return values, {'rows': len(values), 'first': values.date.min(), 'last': values.date.max(),
        'known_paused_skipped': skipped, 'max_abs_error': errors,
        'statuses': {n: values[f'{n}_status'].value_counts().to_dict() for n in errors},
        'arithmetic_ast': ast.dump(assignments[0].value, include_attributes=False),
        'limits': ['Ten frozen stocks are numeric operands, not the original historical whole pool',
            '91 prices span 90 active intervals; adjusted ratio is scale invariant; raw floor is recorded separately',
            'Five-row mean is anchored to the operand day; platform default fill/adjustment and 09:31 query cutoff are unproved',
            'Original mean skips NaNs; this complete-window component blocks missing/pause rows rather than assigning a replacement value']}


def diagnostics(directory):
    quiet = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    cases = []; context = SimpleNamespace(current_dt=dt.datetime(2024, 1, 2, 9, 0), previous_date=dt.date(2024, 1, 1),
        portfolio=SimpleNamespace(available_cash=10000., positions={}))
    g = SimpleNamespace(day_count=91, stock_num=2, stock_price=5, security1=['OLD'], security2=[])
    ns = selected(directory, 0, ['get_momentum', 'min_dict', 'max_dict', 'market_open', 'filter_new_stock',
        'filter_trade_stock', 'filter_paused_stock', 'filter_st_stock'],
        {'g': g, 'nlargest': nlargest, 'log': quiet, 'get_price': lambda *a, **kw: None})
    def failure(label, error, call):
        try: call()
        except error as exc: cases.append({'case': label, 'error': type(exc).__name__, 'message': str(exc)})
        else: raise ValueError(f'Expected {label}')
    failure('datetime_not_imported_outside_platform', NameError, lambda: ns['get_momentum'](context, ['A']))
    ns['datetime'] = dt
    requests_seen = []
    def query(*a, **kw):
        requests_seen.append(kw)
        return pd.DataFrame({'close': np.linspace(5., 6., 91)}, index=pd.bdate_range('2023-08-01', periods=91))
    ns['get_price'] = query
    failure('datetime_series_minus1_is_label', KeyError, lambda: ns['get_momentum'](context, ['A']))
    require(requests_seen[-1]['end_date'] == context.current_dt-dt.timedelta(days=1) and requests_seen[-1]['skip_paused'], 'Query cutoff changed')
    cases.append({'case': 'calendar_day_cutoff_and_skip_paused', 'request': {**requests_seen[-1], 'end_date': str(requests_seen[-1]['end_date'])}})
    lowest = ns['min_dict']({'A': .2, 'B': -.3, 'C': -.3, 'D': .1}, 2)
    require(lowest == ['B', 'C'], 'Low momentum or tie order changed')
    cases.append({'case': 'lowest_momentum_not_title_and_insertion_ties', 'selected': lowest})
    ns['get_security_info'] = lambda stock: SimpleNamespace(start_date=context.previous_date-dt.timedelta(days=250 if stock == 'A' else 249))
    require(ns['filter_new_stock'](context, ['A', 'B']) == ['A'], '250 calendar day boundary changed')
    cases.append({'case': 'IPO_at_least250_calendar_days', 'selected': ['A']})
    chosen = ns['filter_trade_stock'](context, ['600001.XSHG', '000001.XSHE', '300001.XSHE', '688001.XSHG'])
    require(chosen == ['600001.XSHG', '000001.XSHE'], 'Prefix rules changed')
    cases.append({'case': 'literal60_or00_prefix', 'selected': chosen})
    statuses = {s: SimpleNamespace(paused=p, is_st=st, name=name) for s, p, st, name in
        [('A', False, False, 'normal'), ('B', True, False, 'normal'), ('C', False, True, 'normal'),
         ('D', False, False, 'STname'), ('E', False, False, '*name'), ('F', False, False, '\u9000name')]}
    ns['get_current_data'] = lambda: statuses
    require(ns['filter_st_stock'](ns['filter_paused_stock'](list(statuses))) == ['A'], 'Name/status filters changed')
    cases.append({'case': 'pause_ST_star_delist_name_filters', 'selected': ['A']})
    orders = []; ns.update(order_target=lambda *a: orders.append(list(a)), order_value=lambda *a: orders.append(list(a)))
    failure('empty_new_targets_divide_by_zero_after_sell', ZeroDivisionError, lambda: ns['market_open'](context))
    require(orders == [['OLD', 0]], 'Sell-before-error changed')
    cases.append({'case': 'sell_intent_before_empty_target_failure', 'orders': orders.copy()})
    g.security2 = ['A', 'B']; ns['market_open'](context)
    require(orders[-2:] == [['A', 5000.], ['B', 5000.]], 'Original fixed cash split changed')
    cases.append({'case': 'new_targets_share_cash_once_not_rebalance', 'orders': orders[-2:]})

    g = SimpleNamespace(sel_stock='A', strategy_starttime=dt.time(10, 20), strategy_endtime=dt.time(14, 55))
    ns = selected(directory, 1, ['weekly', 'before_market_open', 'get_buy_price', 'get_sell_price', 'market_open'],
        {'g': g, 'pd': pd, 'log': quiet})
    panel = pd.DataFrame({'LOW': [1.4]*5, 'PAUSE': [1.5]*5, 'ST': [1.6]*5, 'A': [2.]*5, 'B': [3.]*5})
    statuses = {s: SimpleNamespace(paused=s == 'PAUSE', is_st=s == 'ST', name=s) for s in panel}
    sells = []; ns.update(get_all_securities=lambda: pd.DataFrame(index=panel.columns), history=lambda **kw: panel,
        get_current_data=lambda: statuses, get_security_info=lambda s: SimpleNamespace(display_name=s),
        order_target_value=lambda *a: sells.append(list(a)))
    context.portfolio.positions = {'OLD': SimpleNamespace(closeable_amount=100)}
    with redirect_stdout(StringIO()): ns['weekly'](context)
    require(g.sel_stock == 'A' and sells == [['OLD', 0]], 'Weekly selection changed')
    cases.append({'case': 'weekly_lowest5_mean_above1p5_then_status_filter', 'selected': g.sel_stock, 'sell_intents': sells.copy()})
    panel.loc[0, 'A'] = 1.6; panel.loc[1:, 'A'] = np.nan
    with redirect_stdout(StringIO()): ns['weekly'](context)
    require(g.sel_stock == 'A', 'Original missing mean behavior changed')
    cases.append({'case': 'weekly_mean_skips_four_missing_closes', 'selected': g.sel_stock, 'finite_rows': 1})
    calls = []
    def history(*a, **kw):
        calls.append(kw)
        return pd.DataFrame({'open': [2.], 'low': [1.9], 'high': [2.1], 'close': [2.05]}, index=[pd.Timestamp('2024-01-02 10:19')])
    ns['attribute_history'] = history
    context.current_dt = dt.datetime(2024, 1, 2, 10, 20)
    failure('50m_date_series_zero_is_label', KeyError, lambda: ns['get_buy_price'](context))
    cases.append({'case': 'original50m_pre_adjusted_query', 'request': calls[-1]})
    ns['attribute_history'] = lambda *a, **kw: {n: np.array([v]) for n, v in [('low', 1.9), ('high', 2.1), ('close', 2.05)]}
    boundaries = []
    for hour, minute in ((10, 19), (10, 20), (14, 54), (14, 55)):
        context.current_dt = dt.datetime(2024, 1, 2, hour, minute)
        boundaries.append({'time': context.current_dt.strftime('%H:%M'), 'buy': float(ns['get_buy_price'](context)), 'sell': float(ns['get_sell_price'](context))})
    require([(r['buy'], r['sell']) for r in boundaries] == [(2.05, 2.05), (1.9, 2.1), (1.9, 2.1), (2.05, 2.05)], 'Time branches changed')
    cases.append({'case': '10p20_14p55_price_boundaries_array_fixture', 'prices': boundaries})
    context.current_dt = dt.datetime(2024, 1, 2, 10, 20)
    context.portfolio.positions = {'A': SimpleNamespace(closeable_amount=100)}
    def minute_case(result_status, cash, positions):
        ns['before_market_open'](context); context.portfolio.available_cash = cash; context.portfolio.positions = positions
        g.last_buy_orderid = 'previous'; operations = []
        ns.update(get_buy_price=lambda c: 2., get_sell_price=lambda c: 3., LimitOrderStyle=lambda p: p,
            get_open_orders=lambda: {'previous': SimpleNamespace(order_id='previous')},
            cancel_order=lambda o: operations.append(['cancel', o.order_id]))
        def order(stock, qty, limit):
            operations.append(['order', stock, qty, limit])
            return SimpleNamespace(status=result_status, order_id='new')
        ns['order'] = order; ns['market_open'](context)
        return {'operations': operations, 'buy_flag': g.not_buy_flg, 'sell_flag': g.not_sell_flg,
            'last_buy_orderid': g.last_buy_orderid, 'last_sell_orderid': g.last_sell_orderid}
    result = minute_case('held', 1000., {'A': SimpleNamespace(closeable_amount=100)})
    require(result['operations'] == [['cancel', 'previous'], ['order', 'A', 500, 2.]] and result['buy_flag'] == 1 and result['sell_flag'] == 0, 'Held-buy flags changed')
    cases.append({'case': 'held_buy_disables_sell_but_buy_remains_enabled', **result})
    result = minute_case('open', 1000., {'A': SimpleNamespace(closeable_amount=100)})
    require(result['operations'][-1] == ['order', 'A', -100, 3.] and result['last_buy_orderid'] == result['last_sell_orderid'] == 'new', 'Pending-order intentions changed')
    cases.append({'case': 'cancel_pending_buy_then_repost_and_sell_closeable_only', **result})
    result = minute_case('open', 0., {'A': SimpleNamespace(closeable_amount=0)})
    require(result['sell_flag'] == 1 and not any(o[0] == 'order' for o in result['operations']), 'Comparison/no-closeable branch changed')
    cases.append({'case': 'no_closeable_comparison_does_not_assign_sell_flag', **result})
    context.current_dt = dt.datetime(2024, 1, 2, 14, 55)
    result = minute_case('open', 1000., {'A': SimpleNamespace(closeable_amount=100)})
    require(len([o for o in result['operations'] if o[0] == 'order']) == 1, 'End-time sell boundary changed')
    cases.append({'case': '14p55_still_buys_but_stops_selling', **result})
    return canonical({'cases': cases, 'synthetic_fixture': True, 'not_a_backtest': True,
        'real_historical_trade_windows': 0, 'platform_equivalent': False,
        'limits': ['Array fixtures isolate source logic; pandas label-index failures are independently retained',
            'Order return statuses are fixtures, not a matching engine, fill evidence or account balance simulation']})


def worker(directory, case):
    import akshare as ak
    evidence = directory / case; evidence.mkdir(exist_ok=False)
    fn = ak.stock_zh_a_hist_min_em if case == 'old-minute' else ak.stock_info_sz_delist
    params = {'symbol': '600568', 'start_date': '2018-01-02 09:30:00', 'end_date': '2018-01-03 15:00:00',
        'period': '5', 'adjust': ''} if case == 'old-minute' else {'symbol': '\u7ec8\u6b62\u4e0a\u5e02\u516c\u53f8'}
    parser = evidence / 'parser.original.py'; shutil.copyfile(inspect.getsourcefile(fn), parser)
    result = {'case': case, 'status': 'failed', 'request': params, 'wire_responses': [],
        'parser_file': str(parser), 'parser_sha256': file_sha(parser), 'strict_usable': False,
        'requested_at': dt.datetime.now().isoformat()}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        session.trust_env = False; kwargs['timeout'] = 20
        response = original(session, method, url, **kwargs)
        path = evidence / f'response-{len(result["wire_responses"]):03d}.bin'; path.write_bytes(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code,
            'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr):
            frame = fn(**params)
        path = evidence / 'parsed.parquet'; frame.to_parquet(path, index=False)
        result.update(status='success' if len(frame) else 'empty', rows=len(frame), columns=list(frame),
            parsed_file=str(path), parsed_sha256=file_sha(path))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    save(evidence / 'result.json', result)
    return result


def probe(root, directory):
    binding(root, directory); results = []
    require(not (directory / 'probes.json').exists() and not any((directory / c).exists() for c in PROBES), 'Probe already exists')
    for case in PROBES:
        try:
            child = subprocess.run([sys.executable, '-m', 'scripts.review_strategy_batch67', 'worker',
                '--directory', str(directory), '--case', case], capture_output=True, text=True, timeout=60)
            result = read(directory / case / 'result.json') if (directory / case / 'result.json').exists() else {
                'case': case, 'status': 'failed', 'error': child.stderr[-1000:], 'returncode': child.returncode}
            save(directory / f'{case}-process.json', {'returncode': child.returncode, 'stderr': child.stderr, 'stdout': child.stdout})
        except subprocess.TimeoutExpired:
            result = {'case': case, 'status': 'timeout', 'timeout_seconds': 60, 'strict_usable': False}
        results.append(result)
        print(json.dumps({'case': case, 'status': result['status'], 'rows': result.get('rows')}, ensure_ascii=False), flush=True)
    save(directory / 'probes.json', {'probes': results, 'not_a_backtest': True, 'strict_usable': False})
    return archive(root, 'Batch67 bounded old-minute/delisting interface responses or failures archived; no input publication')


def supplements(directory):
    probes = read(directory / 'probes.json'); verified = 0
    for row in probes['probes']:
        if 'parser_file' in row:
            require(file_sha(row['parser_file']) == row['parser_sha256'], 'Probe parser changed'); verified += 1
        for wire in row.get('wire_responses', []):
            require(file_sha(wire['file']) == wire['sha256'], 'Wire response changed'); verified += 1
        if 'parsed_file' in row:
            require(file_sha(row['parsed_file']) == row['parsed_sha256'], 'Parsed probe changed'); verified += 1
            frame = pd.read_parquet(row['parsed_file'])
            require(len(frame) == row['rows'] and list(frame) == row['columns'], 'Parsed probe scope changed')
    return {'probes': probes, 'probe_files_sha_checked': verified, 'strict_usable': False, 'not_a_backtest': True,
        'decision': 'Original full-pool trades remain blocked; no daily or current-survivor replacement',
        'limits': ['Delisted list alone is not historical ST/name/identity/announcement availability proof',
            'Supported 5m sample cannot silently replace original platform 50m cutoff/fill/cancel semantics',
            'Existing batch65 names/universe future-name counterexamples remain blocking',
            'No economic variant newly approved; original fund-cost scope and every-bar rules remain unresolved']}


def delisted_audit(delisted, master):
    codes = delisted['证券代码'].astype(str)
    require(codes.str.fullmatch(r'\d{6}').all() and not codes.duplicated().any(), 'Invalid/duplicate delisted codes')
    listing = pd.to_datetime(delisted['上市日期'], errors='coerce')
    ending = pd.to_datetime(delisted['终止上市日期'], errors='coerce')
    require(master.instrument.is_unique, 'Duplicate security master identity')
    lookup = master.set_index('instrument'); missing = []; differences = []; matched = 0
    for code, first, last in zip(codes, listing, ending, strict=True):
        instrument = f'{code}.SZ'
        if instrument not in lookup.index:
            missing.append(instrument); continue
        matched += 1; row = lookup.loc[instrument]
        current_first = pd.to_datetime(row.list_date, errors='coerce')
        current_last = pd.to_datetime(row.delist_date, errors='coerce')
        if pd.isna(current_first) or pd.isna(current_last) or current_first != first or current_last != last:
            differences.append({'instrument': instrument, 'official_list_date': None if pd.isna(first) else str(first.date()),
                'official_delist_date': None if pd.isna(last) else str(last.date()),
                'master_list_date': None if pd.isna(current_first) else str(current_first.date()),
                'master_delist_date': None if pd.isna(current_last) else str(current_last.date())})
    return {'official_rows': len(delisted), 'official_unique_codes': codes.nunique(), 'master_rows': len(master),
        'official_missing_list_dates': int(listing.isna().sum()), 'official_missing_delist_dates': int(ending.isna().sum()),
        'official_delist_before_listing': int((ending < listing).sum()),
        'official_delist_after_numeric_end': int((ending > pd.Timestamp('2026-09-29')).sum()),
        'official_first_delist': None if ending.dropna().empty else str(ending.min().date()),
        'official_last_delist': None if ending.dropna().empty else str(ending.max().date()),
        'matched_master_codes': matched, 'missing_master_codes': missing,
        'master_date_differences_or_missing': differences, 'strict_usable': False,
        'limits': ['Current official list is independently captured evidence, not an historical version guarantee',
            'Dates and code membership do not prove the name, risk status, announcement time or code identity chain']}


def dependency_audit(root, directory, freeze=False):
    store = Store(root); state = store.state(SNAPSHOT); inputs = {}
    for name in ('instruments', 'minute_universe'):
        inputs[name] = []
        for entry in state['tables'].get(name, {}).values():
            path = Path(root) / entry['file']; frame = pd.read_parquet(path)
            require(fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'], 'Dependency content fingerprint differs')
            inputs[name].append({'file': str(path), 'sha256': file_sha(path),
                'content_fingerprint': entry['sha'], 'rows': entry['rows']})
    if freeze: save(directory / 'dependency-inputs.json', inputs)
    else: require(inputs == read(directory / 'dependency-inputs.json'), 'Dependency binding changed')
    master = store.load_state(state, 'instruments'); universe = store.load_state(state, 'minute_universe')
    require(len(master) == sum(e['rows'] for e in inputs['instruments']) and
        len(universe) == sum(e['rows'] for e in inputs['minute_universe']), 'Dependency row scope changed')
    entries = state['tables'].get('bars_5m', {})
    probes = read(directory / 'probes.json')['probes']; official = next(r for r in probes if r['case'] == 'sz-delisted')
    if official['status'] in ('success', 'empty'):
        require(file_sha(official['parsed_file']) == official['parsed_sha256'], 'Official list SHA changed')
        comparison = delisted_audit(pd.read_parquet(official['parsed_file']), master)
    else: comparison = {'status': official['status'], 'strict_usable': False}
    return {'snapshot': SNAPSHOT, 'inputs': inputs, 'official_delisted_vs_master': comparison,
        'minute_snapshot': {'rows_from_partition_metadata': sum(e['rows'] for e in entries.values()),
            'partition_keys': sorted(entries), 'universe_rows': len(universe),
            'universe_years': sorted(int(y) for y in universe.year.unique()),
            '600568_in_minute_universe': bool(universe.instrument.eq('600568.SH').any()),
            'fresh_all_minute_partition_hash': False}, 'not_a_backtest': True,
        'limits': ['Minute counts are manifest metadata; no fresh scan or hash of all 44 million minute rows',
            'The selected 2018 example predates registered minute partition months; original weekly whole-pool targets remain unknown']}


def component(directory):
    validate_sources(directory); price = pd.read_parquet(directory / 'price-input.parquet')
    calendar = pd.read_parquet(directory / 'calendar-input.parquet')
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date'])).sort_values()
    require(len(dates) == 5280, 'Calendar scope differs')
    values, arithmetic = numeric_values(price, dates, source_tree(directory, 0))
    return {'snapshot': SNAPSHOT, 'input_rows': len(price), 'stocks': sorted(price.instrument.unique()),
        'numeric_components': arithmetic, 'not_a_backtest': True, 'platform_equivalent': False,
        'original_strategy_complete': False, 'strategy_trade_runs': 0, 'cost_backtests': []}, values


def study(root, directory):
    binding(root, directory); before = implementation(); comp, values = component(directory); diag = diagnostics(directory)
    supplement = supplements(directory)
    dependencies = dependency_audit(root, directory, freeze=not (directory / 'dependency-inputs.json').exists())
    require(before == implementation() and not (directory / 'component-values.parquet').exists(), 'Study already exists/changed')
    values.to_parquet(directory / 'component-values.parquet', index=False)
    save(directory / 'component-research.json', comp); save(directory / 'diagnostics.json', diag)
    save(directory / 'supplement-decision.json', supplement); save(directory / 'stock-input-audit.json', dependencies)
    return archive(root, 'Batch67 2005-2026 numeric operands and original order-state diagnostics archived; not trades')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch67 network forbidden')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        comp, values = component(directory); diag = diagnostics(directory); catalog = offline_catalog(root, directory)
        supplement = supplements(directory); dep = inventory(root); dependencies = dependency_audit(root, directory)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    require(before == implementation() and comp == read(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and supplement == read(directory / 'supplement-decision.json') and
        dependencies == read(directory / 'stock-input-audit.json') and
        dep == read(directory / 'dependency-inventory.json') and api_evidence() == read(directory / 'existing-apis.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before,
        'evidence_sha256': {n: file_sha(directory / n) for n in
            ('component-values.parquet', 'component-research.json', 'diagnostics.json', 'supplement-decision.json', 'stock-input-audit.json')},
        'not_a_backtest': True, 'formal_strategy_acceptance': False})
    return archive(root, 'Batch67 forbidden-network numeric values/diagnostics and catalog artifacts match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch67.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch67_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['implementation_sha256'].items():
        require(file_sha(directory / 'implementation-checked' / name) == sha, 'Implementation copy changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'], 'Offline proof changed')
    for name, sha in off['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Offline component changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory)
    progress = archive(root, 'Batch67 duplicate/low-price source reviews and components accepted; original trades remain gated')
    receipt = Path('docs/handoff/2026-10-08-batch67-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'component': read(directory / 'component-research.json'),
        'diagnostics': read(directory / 'diagnostics.json'), 'supplements': supplements(directory),
        'stock_input_audit': read(directory / 'stock-input-audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'new_unique_strategy_logic': 1,
        'not_a_backtest': True, 'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'probe', 'worker', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=PROBES)
    args = parser.parse_args()
    if args.action == 'worker':
        require(args.case is not None, 'Missing probe case'); result = worker(args.directory, args.case)
    else: result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
