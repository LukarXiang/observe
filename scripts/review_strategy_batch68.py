"""Audit low-PB source rules and monthly candidate components; never invent trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import inspect
from io import StringIO
import json
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
from scripts import review_strategy_batch67 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch68/20261008-low-PB')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY
RECEIPT = Path('docs/handoff/2026-10-08-batch67-verification.json')
SOURCES = ('2024年度精选策略2/81.低PB小市值低换手策略总结.txt',
    '2020年度精选策略/31 低PB价值投资策略分享.txt')
SOURCE_SHA = ('86717283069149897158bf81d70693f91c037bd397b709a8536bf5ccb2b4ede3',
    '5f9bf0f45618e22ecfae05849b576f9e135b8254f9065838284978bb5a9f5987')
REFERENCES = (('repo/akshare', 'akshare/stock_feature/stock_zh_valuation_baidu.py'),
    ('repo/akshare', 'akshare/stock/stock_share_changes_cninfo.py'),
    ('repo/rqalpha', 'rqalpha/apis/api_base.py'))
LEGACY = ('strategies/specs/y2024b-081.yaml', 'strategies/results.csv',
    'data/runs/20261005-132927-strategy-y2024b-081-87b5/config.json',
    'data/runs/20261005-132927-strategy-y2024b-081-87b5/metrics.json',
    'data/runs/20261005-132927-strategy-y2024b-081-87b5/limitations.json')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch68.py',
    'tests/unit/test_batch68_research.py', 'src/observe/strategy/fields.py'}))
PROBES = ('600036', '600519', '600568')
DENSE = {'600036-1y': '\u8fd1\u4e00\u5e74', '600036-5y': '\u8fd1\u4e94\u5e74',
    '600519-1y': '\u8fd1\u4e00\u5e74', '600568-1y': '\u8fd1\u4e00\u5e74'}
HELPERS = ('financial_data_filter_xiaoyu', 'financial_data_filter_dayu', 'get_sort_dataframe',
    'max_hold_days', 'sell_by_amount_or_percent_or_none', 'order_style',
    'judge_security_max_proportion', 'max_buy_value_or_amount')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json',
    'monthly-input.parquet', 'monthly-input-binding.json', 'calendar-input.parquet',
    'component-values.parquet', 'component-research.json', 'diagnostics.json',
    'probe-results.json', 'dense-probe-results.json', 'total-cap-audit.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch67 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Binding changed')
    return result


def parsed_source(text, number):
    lines = text.splitlines(); stripped = []
    if number == 1:
        require(lines[6] == '\u539f\u6587\u7b56\u7565\u6e90\u7801\u5982\u4e0b\uff1a' and lines[8] == "'''", 'Source preamble changed')
        stripped = [{'line': i+1, 'original': line} for i, line in enumerate(lines[:8])]
        lines[:8] = ['']*8
    code = '\n'.join(lines)
    return ast.parse(code), {'removed_preamble': stripped,
        'parsed_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'original_line_numbers_retained': True, 'purpose': 'AST extraction only; no full strategy execution'}


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    tree, normalization = parsed_source(read_source(Path(row['source_copy']))[0], number)
    require(row['normalization'] == normalization, 'Normalization changed')
    return tree


def allowed_default(node):
    return safe_default(node) or (isinstance(node, ast.List) and not node.elts)


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(allowed_default(d) for d in n.args.defaults) and
        all(d is None or allowed_default(d) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch68-functions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    fn = ak.stock_zh_valuation_baidu
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
        'function': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': inspect.getsource(fn),
        'request': {'indicator': '\u603b\u5e02\u503c', 'period': '\u5168\u90e8'},
        'limits': ['Parsed date/value alone does not prove units, daily historical version, share-class or whole-pool coverage']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    search = ['rg', '-n', '^def ('+'|'.join(HELPERS)+')\\(', 'repo', '-g', '*.py', '-g', '*.txt']
    result = subprocess.run(search, capture_output=True, text=True)
    require(result.returncode in (0, 1), result.stderr)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'helper_search': {'command': search, 'returncode': result.returncode, 'matches': result.stdout.splitlines(),
            'scope': 'Visible source definitions, not proof of absence on other machines or in platform modules'},
        'required_external_helpers': list(HELPERS), 'not_a_backtest': True,
        'gates': ['No admitted daily total market cap or historical total-share event chain',
            'No demonstrated strict historical revenue/net-profit growth/ROE version',
            'No admitted complete 2014-2018 minute history or wizard rank/hold-day/order semantics',
            'Historical all-stock membership and provider valuation/tie-order equivalence remain unproved']}


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch68 low-PB source preflight; original inputs and old runs protected')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    folder = directory / 'source-reviews'; folder.mkdir(); frozen = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/\u91cf\u5316\u7b56\u7565\u6e90\u4ee3\u7801') / name
        require(file_sha(path) == sha, 'Source SHA changed')
        shutil.copyfile(path, folder / f'{strategy_id(name)}.source'); frozen.append({'path': name, 'sha256': sha})
    save(directory / 'sources-frozen.json', frozen)
    return {'status': 'ok'}


def start(root, directory):
    binding(root, directory); require((directory / 'baseline.json').is_file(), 'Missing baseline')
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = []; refs = []; legacy = []
    for i, name in enumerate(SOURCES):
        original = Path('repo/\u91cf\u5316\u7b56\u7565\u6e90\u4ee3\u7801') / name
        copied = directory / 'source-reviews' / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '\u4eba\u5de5\u5ba1\u67e5\u5b8c\u6210' and
            file_sha(original) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source already reviewed/changed')
        text, encoding = read_source(copied); tree, normalization = parsed_source(text, i)
        rows.append({'source_path': str(original), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(tree),
            'normalization': normalization, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    for i, name in enumerate(LEGACY):
        path = Path(name); copied = directory / 'legacy' / f'{i}-{path.name}'; copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, copied); legacy.append({'file': name, 'copy': str(copied), 'sha256': file_sha(path)})
    catalog = catalog_strategies(root, 'repo/\u91cf\u5316\u7b56\u7565\u6e90\u4ee3\u7801')
    current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'legacy': legacy,
        'catalog': catalog, 'snapshot': SNAPSHOT, 'previous_catalog_file': str(prior),
        'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch68 two complete low-PB sources reviewed; legacy float-cap/reweight deviations and wizard gaps archived')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review scope/binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/\u91cf\u5316\u7b56\u7565\u6e90\u4ee3\u7801') / SOURCES[i]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Original rule changed')
    require(len(doc['references']) == len(REFERENCES) and len(doc['legacy']) == len(LEGACY), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference bytes changed')
        require(subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip() == row['commit'], 'Reference HEAD changed')
    for name, row in zip(LEGACY, doc['legacy'], strict=True):
        require(row['file'] == name and file_sha(name) == file_sha(row['copy']) == row['sha256'], 'Legacy evidence changed')


def monthly_schedule(calendar):
    days = sorted(pd.to_datetime(calendar.loc[calendar.is_open.eq(True), 'date']).dt.date)
    require(len(days) == len(set(days)), 'Calendar duplicate')
    return [(days[k-1], day) for k, day in enumerate(days) if k and dt.date(2021, 2, 1) <= day <= dt.date(2026, 9, 29)
        and (days[k-1].year, days[k-1].month) != (day.year, day.month)]


def prepare(root, directory):
    binding(root, directory); validate_sources(directory); store = Store(root); state = store.state(SNAPSHOT)
    cal = store.load_state(state, 'calendar'); schedule = monthly_schedule(cal)
    rows = []; checked = []; dates = {before for before, _ in schedule}
    columns = ['date', 'instrument', 'pb_mrq', 'turnover', 'volume', 'close', 'is_trading']
    for _, entry in sorted(state['tables']['bars_1d'].items()):
        path = store.root / entry['file']; frame = pd.read_parquet(path)
        require(len(frame) == entry['rows'] and fingerprint(frame) == entry['sha'], 'Daily partition content changed')
        checked.append({'file': str(path), 'file_sha256': file_sha(path), 'content_fingerprint': entry['sha'], 'rows': len(frame)})
        keep = pd.to_datetime(frame.date).dt.date.isin(dates)
        rows.append(frame.loc[keep, columns].copy())
    cal_path = directory / 'calendar-input.parquet'; require(not cal_path.exists(), 'Frozen calendar exists')
    cal.to_parquet(cal_path, index=False)
    frame = pd.concat(rows, ignore_index=True).sort_values(['date', 'instrument']).reset_index(drop=True)
    require(not frame.duplicated(['date', 'instrument']).any() and not frame.empty, 'Bad monthly input')
    path = directory / 'monthly-input.parquet'; require(not path.exists(), 'Frozen monthly input exists')
    frame.to_parquet(path, index=False)
    save(directory / 'monthly-input-binding.json', {'snapshot': SNAPSHOT, 'partitions': checked,
        'calendar_sha256': file_sha(cal_path), 'input_sha256': file_sha(path), 'rows': len(frame),
        'schedule': [[str(a), str(b)] for a, b in schedule], 'columns': columns,
        'scope': 'All stored prior-session rows, no replacement for platform historical whole pool', 'not_a_backtest': True})
    return archive(root, 'Batch68 monthly prior-session PB/turnover operands frozen after full daily partition content checks')


def numeric_values(frame, schedule, tree):
    namespace = {'can_buy': lambda day, stock: True}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter']
    require(len(nodes) == 1 and not nodes[0].decorator_list and all(allowed_default(d) for d in nodes[0].args.defaults), 'Unsafe filter')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-candidate-filter>', 'exec'), namespace)
    frame = frame.copy(); frame['date'] = pd.to_datetime(frame.date).dt.date
    require(not frame.duplicated(['date', 'instrument']).any(), 'Duplicate operands')
    groups = {day: part for day, part in frame.groupby('date', sort=True)}; rows = []; coverage = []
    for before, day in schedule:
        require(before < day, 'Future decision input')
        raw = groups.get(before, frame.iloc[:0]); data = raw.rename(columns={'instrument': 'code', 'pb_mrq': 'pb_ratio', 'turnover': 'turnover_ratio'}).copy()
        # This is a separately labelled operand diagnostic, never total market cap.
        data['float_cap_proxy'] = data.close*data.volume/data.turnover_ratio.where(data.turnover_ratio.gt(0))/1e8
        data.loc[~data.is_trading.eq(True), 'float_cap_proxy'] = np.nan
        coverage.append({'date': str(before), 'decision_date': str(day), 'stored_rows': len(data),
            'pb_missing': int(data.pb_ratio.isna().sum()), 'nonpositive_pb': int(data.pb_ratio.le(0).sum()),
            'turnover_missing': int(data.turnover_ratio.isna().sum()), 'proxy_missing': int(data.float_cap_proxy.isna().sum())})
        for branch, field in (('original_PB_prescreen', 'pb_ratio'), ('float_cap_proxy_prescreen', 'float_cap_proxy')):
            conditions = [{'by': field, 'asc': True, 'count': 70}, {'by': 'turnover_ratio', 'asc': True, 'count': 35}]
            picks = []; namespace['filter'](conditions, day, data, picks)
            require(len(picks) == min(35, len(data)) and len(set(picks)) == len(picks), 'Candidate truncation changed')
            for rank, code in enumerate(picks, 1):
                value = data[data.code.eq(code)].iloc[0]
                rows.append({'date': str(before), 'decision_date': str(day), 'branch': branch, 'rank': rank,
                    'instrument': code, 'pb_mrq': value.pb_ratio, 'turnover': value.turnover_ratio,
                    'float_cap_proxy': value.float_cap_proxy})
    values = pd.DataFrame(rows)
    return values, {'rows': len(values), 'monthly_decisions': len(schedule), 'coverage': coverage,
        'not_a_backtest': True, 'original_strategy_complete': False,
        'limits': ['Original PB branch uses every stored row, including nonpositive/missing PB; can_buy is deliberately bypassed',
            'Float-cap branch is a diagnostic of legacy proxy, not original market_cap or an approved trading variant',
            'Rows sorted by instrument give an explicit local input order; platform tie-order is unproved',
            'No target union, day-open status filter, account, trades or returns are produced']}


def component(directory):
    bound = read(directory / 'monthly-input-binding.json')
    require(file_sha(directory / 'monthly-input.parquet') == bound['input_sha256'] and
        file_sha(directory / 'calendar-input.parquet') == bound['calendar_sha256'], 'Monthly input changed')
    schedule = monthly_schedule(pd.read_parquet(directory / 'calendar-input.parquet'))
    require([[str(a), str(b)] for a, b in schedule] == bound['schedule'], 'Monthly dates changed')
    return numeric_values(pd.read_parquet(directory / 'monthly-input.parquet'), schedule, source_tree(directory, 0))


def diagnostics(directory):
    context = SimpleNamespace(current_dt=dt.datetime(2024, 1, 2, 9, 30), previous_date=dt.date(2023, 12, 29),
        portfolio=SimpleNamespace(available_cash=1000., positions={'OLD': object(), 'A': object()}))
    cases = []
    def failure(label, error, call):
        try: call()
        except error as exc: cases.append({'case': label, 'error': type(exc).__name__, 'message': str(exc)})
        else: raise ValueError(f'Expected {label}')
    data = pd.DataFrame({'code': [f'S{i:03d}' for i in range(100)], 'pb_ratio': range(100),
        'market_cap': list(range(99, -1, -1)), 'turnover_ratio': range(100)})
    ns = selected(directory, 0, ['filter', 'get_stock_pool', 'trade', 'sell_all', 'buy_stocks', 'can_buy'],
        {'get_data': lambda c: data, 'can_buy': lambda day, code: code != 'S000'})
    # Keep the unmodified can_buy separately, then supply a deterministic status stub.
    original_can_buy = ns['can_buy']; ns['can_buy'] = lambda day, code: code != 'S000'
    conditions = [{'by': 'pb_ratio', 'asc': True, 'count': 70}, {'by': 'turnover_ratio', 'asc': True, 'count': 35}]
    picks = []; ns['filter'](conditions, context.current_dt, data, picks)
    require(picks == [f'S{i:03d}' for i in range(1, 35)], 'Post-truncation filter unexpectedly refills')
    cases.append({'case': 'filter_after35_no_refill', 'count': len(picks), 'rejected_not_replaced': 'S000'})
    union = ns['get_stock_pool'](context)
    require(union == [f'S{i:03d}' for i in range(1, 65)], 'Branch union/order changed')
    cases.append({'case': 'PB_then_marketcap_ordered_dedup_union', 'selected': union})
    defaults = ns['filter'].__defaults__[0]; require(defaults == [], 'Main path mutated default list')
    cases.append({'case': 'main_path_supplies_fresh_list', 'default_after_main_path': defaults.copy()})
    def order(stock, value):
        orders.append([stock, value])
        if value == 0: context.portfolio.available_cash += 100.
    orders = []; ns['order_target_value'] = order; ns['get_stock_pool'] = lambda c: ['A', 'B']
    ns['trade'](context)
    require({tuple(v) for v in orders[:2]} == {('OLD', 0), ('A', 0)} and orders[2:] == [['A', 600.], ['B', 600.]], 'Liquidation/cash split changed')
    cases.append({'case': 'sell_all_including_retained_then_read_cash_once', 'sell_intents': sorted(orders[:2]), 'buy_intents': orders[2:]})
    orders.clear(); ns['buy_stocks'](context, [])
    require(not orders, 'Empty target orders changed')
    cases.append({'case': 'empty_targets_no_division_because_loop_empty', 'orders': []})
    ns['get_extras'] = lambda *a, **k: {'A': [False]}
    ns['get_price'] = lambda *a, **k: pd.DataFrame({'open': [10.], 'high_limit': [11.], 'paused': [0]}, index=pd.to_datetime(['2024-01-02']))
    failure('date_series_zero_is_label', KeyError, lambda: original_can_buy(context.current_dt, 'A'))
    ns['get_price'] = lambda *a, **k: pd.DataFrame({'open': [11.], 'high_limit': [11.], 'paused': [0]})
    require(not original_can_buy(context.current_dt, 'A'), 'Equality at upper limit passed')
    cases.append({'case': 'strict_open_below_limit', 'at_limit_can_buy': False})

    g = SimpleNamespace(sell_will_buy=True, selled_security_list={}, filter_holded=True,
        sell_by_amount=None, sell_by_percent=None, open_sell_securities=[])
    ns = selected(directory, 1, ['risk_management', 'selled_security_list_dict', 'holded_filter', 'sell', 'check_stocks_sort'],
        {'g': g, 'pd': pd}); orders = []
    position = SimpleNamespace(init_time=dt.datetime(2024, 1, 1), price=96.)
    context.portfolio.positions = {'A': position}
    ns['get_price'] = lambda *a, **k: pd.DataFrame({'high': [130.], 'low': [100.]})
    ns['order_target_value'] = lambda *a: orders.append(list(a))
    ns['risk_management'](context)
    require(orders == [['A', 0]], 'Range30/drawdown4 intent changed')
    cases.append({'case': 'range30_and_drawdown4_not_entry_profit_gate', 'price': 96., 'high': 130., 'low': 100., 'orders': orders.copy()})
    orders.clear(); ns['get_price'] = lambda *a, **k: pd.DataFrame({'high': [130.], 'low': [100.]})
    position.price = 126.; ns['risk_management'](context); require(not orders, 'Below4 drawdown unexpectedly sells')
    cases.append({'case': 'drawdown_below4_no_sell', 'orders': []})
    position.price = 96.; ns['get_price'] = lambda *a, **k: pd.DataFrame({'high': [np.nan], 'low': [np.nan]})
    ns['risk_management'](context); require(not orders, 'NaN risk window unexpectedly sells')
    cases.append({'case': 'NaN_minute_extrema_silently_skip', 'orders': []})
    g.sell_will_buy = False
    failure('optional_risk_branch_unbound_sell_lists', UnboundLocalError, lambda: ns['risk_management'](context))
    g.sell_will_buy = True
    require(ns['holded_filter'](context, ['A', 'B']) == ['A', 'B'], 'Default held filter changed')
    g.filter_holded = False; require(ns['holded_filter'](context, ['A', 'B']) == ['B'], 'False held filter changed')
    cases.append({'case': 'holded_filter_false_excludes_default_true_keeps', 'true': ['A', 'B'], 'false': ['B']})
    initial = context.portfolio.positions.keys(); del context.portfolio.positions['A']
    ns['selled_security_list_dict'](context, initial); require(not g.selled_security_list, 'Dynamic keys unexpectedly preserved sold stock')
    cases.append({'case': 'dynamic_keys_lose_sold_history_when_immediate_fill', 'sold_counter': {}})
    context.portfolio.positions = {'A': position}; calls = []
    ns['get_price'] = lambda *a, **k: pd.DataFrame({'high': [129.], 'low': [100.]})
    ns['max_hold_days'] = lambda *a: calls.append(a[-1]) or True
    ns['sell_by_amount_or_percent_or_none'] = lambda *a: orders.append([a[1], 'wizard_sell'])
    ns['sell'](context, ['A']); require(calls == [83] and orders == [['A', 'wizard_sell']], '83 helper delegation changed')
    cases.append({'case': 'hold_day83_delegates_unknown_unit_and_sells_retained', 'helper_argument': 83, 'orders': orders.copy()})
    ns['get_sort_dataframe'] = lambda *a: pd.DataFrame({'weighted': [1.]}, index=['A'])
    failure('removed_DataFrame_sort_after_external_scores', AttributeError,
        lambda: ns['check_stocks_sort'](context, ['A'], {'roe': ('desc', .7)}))
    return canonical({'cases': cases, 'not_a_backtest': True, 'real_historical_trade_windows': 0,
        'formal_strategy_acceptance': False, 'limits': ['Synthetic stubs establish call/intent behavior, not actual platform fills or helper definitions']})


def worker(directory, case):
    require(case in PROBES or case in DENSE, 'Unexpected probe'); folder = directory / f'probe-{case}'; folder.mkdir(exist_ok=False)
    import akshare as ak
    fn = ak.stock_zh_valuation_baidu; copied = folder / 'installed-parser.py'
    shutil.copyfile(inspect.getsourcefile(fn), copied)
    result = {'symbol': case.split('-')[0], 'status': 'failed', 'strict_usable': False, 'parser_sha256': file_sha(copied),
        'request': {'symbol': case.split('-')[0], 'indicator': '\u603b\u5e02\u503c', 'period': DENSE.get(case, '\u5168\u90e8')}, 'wire_responses': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        session.trust_env = False; kwargs['timeout'] = 20
        response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'; path.write_bytes(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr):
            frame = fn(**result['request'])
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        result.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame),
            columns=list(frame.columns), first=str(frame.date.min()), last=str(frame.date.max()),
            missing_dates=int(frame.date.isna().sum()), missing_values=int(frame.value.isna().sum()),
            duplicate_dates=int(frame.date.duplicated().sum()), examples=canonical(frame.head(5).to_dict('records')))
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'
    save(folder / 'result.json', result)
    return result


def probe_requests(root, directory, cases, manifest):
    binding(root, directory); validate_sources(directory); rows = []
    for case in cases:
        path = directory / f'probe-{case}' / 'result.json'
        require(not path.exists(), 'Probe already exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch68', 'worker', '--directory', str(directory), '--case', case]
        log = directory / f'probe-{case}.log'
        with log.open('x') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try: code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); code = -9
        if not path.exists(): save(path, {'symbol': case, 'status': 'worker_timeout_or_failure', 'returncode': code, 'strict_usable': False})
        rows.append({'symbol': case, 'result': read(path), 'result_sha256': file_sha(path), 'log_sha256': file_sha(log), 'returncode': code})
    save(directory / manifest, {'probes': rows, 'published': False, 'not_a_backtest': True,
        'limits': ['Three security samples cannot admit daily total-cap history for the original full pool',
            'Units, revision vintage, share-class and decision-time availability must be demonstrated before any admission']})
    return archive(root, 'Batch68 bounded total-cap interface samples archived; no published input or strategy trades')


def probe(root, directory): return probe_requests(root, directory, PROBES, 'probe-results.json')


def dense(root, directory): return probe_requests(root, directory, tuple(DENSE), 'dense-probe-results.json')


def cap_coverage(frame, schedule, calendar):
    values = frame.copy(); values['date'] = pd.to_datetime(values.date).dt.date
    required = {before for before, _ in schedule}; present = set(values.date.dropna())
    days = set(pd.to_datetime(calendar.loc[calendar.is_open.eq(True), 'date']).dt.date)
    window = {day for day in days if dt.date(2021, 1, 29) <= day <= dt.date(2026, 9, 29)}
    return {'rows': len(values), 'required_monthly_dates': len(required),
        'matched_monthly_dates': len(required & present), 'missing_monthly_dates': sorted(str(d) for d in required-present),
        'required_trading_dates_in_research_window': len(window), 'matched_trading_dates': len(window & present),
        'missing_trading_dates': len(window-present), 'future_rows_excluded': int(values.date.gt(dt.date(2026, 9, 29)).sum()),
        'duplicate_dates': int(values.date.duplicated().sum()), 'invalid_dates': int(values.date.isna().sum()),
        'invalid_values': int((~np.isfinite(values.value) | values.value.le(0)).sum()), 'strict_usable': False}


def total_cap_audit(directory):
    rows = []; calendar = pd.read_parquet(directory / 'calendar-input.parquet'); schedule = monthly_schedule(calendar)
    for manifest, cases in (('probe-results.json', PROBES), ('dense-probe-results.json', tuple(DENSE))):
        doc = read(directory / manifest); require(len(doc['probes']) == len(cases) and not doc['published'], 'Probe scope changed')
        for case, item in zip(cases, doc['probes'], strict=True):
            folder = directory / f'probe-{case}'; path = folder / 'result.json'; result = item['result']
            require(file_sha(path) == item['result_sha256'] and read(path) == result and
                file_sha(directory / f'probe-{case}.log') == item['log_sha256'], 'Probe record changed')
            record = {'case': case, 'status': result['status'], 'strict_usable': False}
            if 'request' in result:
                require(result['request'] == {'symbol': case.split('-')[0], 'indicator': '\u603b\u5e02\u503c',
                    'period': DENSE.get(case, '\u5168\u90e8')}, 'Probe parameters changed')
                require(file_sha(folder / 'installed-parser.py') == result['parser_sha256'], 'Parser changed')
            for response in result.get('wire_responses', []): require(file_sha(response['file']) == response['sha256'], 'Response changed')
            if result['status'] in ('success', 'empty'):
                require(file_sha(result['file']) == result['sha256'], 'Parsed input changed')
                frame = pd.read_parquet(result['file']); require(len(frame) == result['rows'], 'Parsed rows changed')
                raw = json.loads(Path(result['wire_responses'][-1]['file']).read_bytes())
                chart = raw['Result'][0]['DisplayData']['resultData']['tplData']['result']['chartInfo'][0]
                parsed = pd.DataFrame(chart['body'], columns=['date', 'value'])
                parsed['date'] = pd.to_datetime(parsed.date, errors='coerce').dt.date
                parsed['value'] = pd.to_numeric(parsed.value, errors='coerce')
                pd.testing.assert_frame_equal(parsed, frame, check_exact=True)
                record.update(cap_coverage(frame, schedule, calendar), unit=chart.get('unit'),
                    headers=chart.get('header'), parser_roundtrip='match', provider_chart_type=chart.get('type'))
            rows.append(record)
    return {'samples': rows, 'not_a_backtest': True, 'admitted_rows': 0,
        'limits': ['Every requested series is a current vendor chart version, not proven historical vintage',
            'Monthly exact-date matching is measured without interpolation or unlimited forward fill',
            'Chart unit is recorded from raw response; per-security sample coverage does not prove all-stock ranks or share classes']}


def study(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    values, summary = component(directory); diag = diagnostics(directory); cap = total_cap_audit(directory)
    require(before == implementation(), 'Implementation changed during study')
    path = directory / 'component-values.parquet'; require(not path.exists(), 'Component exists')
    values.to_parquet(path, index=False); save(directory / 'component-research.json', summary); save(directory / 'diagnostics.json', diag)
    save(directory / 'total-cap-audit.json', cap)
    return archive(root, 'Batch68 monthly prior-session candidate arithmetic and original source intent diagnostics archived; no trading backtest')


def offline(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    def denied(*a, **k): raise AssertionError('Network forbidden during batch68 reconstruction')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        values, summary = component(directory); diag = diagnostics(directory); cat = offline_catalog(root, directory)
        dep = inventory(root); cap = total_cap_audit(directory)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    require(before == implementation() and summary == read(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and dep == read(directory / 'dependency-inventory.json') and
        cap == read(directory / 'total-cap-audit.json') and api_evidence() == read(directory / 'existing-apis.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'evidence_sha256': {name: file_sha(directory / name) for name in
            ('component-values.parquet', 'component-research.json', 'diagnostics.json', 'total-cap-audit.json')}, 'not_a_backtest': True})
    return archive(root, 'Batch68 forbidden-network candidate/diagnostic/catalog reconstruction matches0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch68.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch68_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as output: result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
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
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Implementation copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'], 'Offline proof changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    monthly = read(directory / 'monthly-input-binding.json')
    require(all(file_sha(p['file']) == p['file_sha256'] for p in monthly['partitions']), 'Original daily partition bytes changed')
    protection = protect(root, directory)
    progress = archive(root, 'Batch68 low-PB source/candidate research accepted; original total-cap/wizard/minute trades remain gated')
    receipt = Path('docs/handoff/2026-10-08-batch68-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'component': read(directory / 'component-research.json'),
        'diagnostics': read(directory / 'diagnostics.json'), 'supplements': read(directory / 'probe-results.json'),
        'dense_supplements': read(directory / 'dense-probe-results.json'), 'total_cap_audit': read(directory / 'total-cap-audit.json'),
        'daily_partition_full_byte_checks': len(monthly['partitions']),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'prepare', 'worker', 'probe', 'dense', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=(*PROBES, *DENSE)); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
