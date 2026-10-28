"""Audit small-cap wizard and PE rotation with original callbacks and real coverage."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
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
from scripts import review_strategy_batch68 as cap_batch
from scripts import review_strategy_batch70 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch71/20261008-smallcap-PE-rotation')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY
RECEIPT = Path('docs/handoff/2026-10-08-batch70-verification.json')
SOURCES = ('2020年度精选策略/53 小市值策略.txt',
    '2020年度精选策略/59 小市值轮动，修改了内容，剔除了停牌股，同时PE要求为正的股票。.txt')
SOURCE_SHA = ('8464ef531e098a4067ca88a73d6cc196bbcb8ebfba7bdfd12afa746b9ca9704e',
    '4ca0e6ef4b7590881f4c3d6864efd521b7b35e75ddf61c471d6cfcd12d01994b')
REFERENCES = (('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_zh_valuation_baidu.py'))
HELPERS = ('get_sort_dataframe', 'order_style', 'judge_security_max_proportion',
    'max_buy_value_or_amount', 'sell_by_amount_or_percent_or_none')
QUERIES = {'000001-all': {'symbol': '000001', 'indicator': '总市值', 'period': '全部'},
    '000001-5y': {'symbol': '000001', 'indicator': '总市值', 'period': '近五年'}}
URL = 'https://gushitong.baidu.com/opendata'
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch71.py',
    'tests/unit/test_batch71_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'existing-apis.json', 'dependency-inventory.json',
    'diagnostics.json', 'probe-results.json', 'total-cap-coverage.json', 'daily-input-binding.json',
    'daily-predicate-values.parquet', 'calendar-input.parquet', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation():
    return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch70 changed')
    cap_receipt = read('docs/handoff/2026-10-08-batch68-verification.json'); reused = {}
    for manifest in ('probe-results.json', 'dense-probe-results.json'):
        path = cap_batch.DIRECTORY / manifest
        require(file_sha(path) == cap_receipt['checks']['evidence_sha256'][manifest], 'Frozen cap manifest changed')
        for item in read(path)['probes']:
            case = item['symbol']; row = item['result']; folder = cap_batch.DIRECTORY / f'probe-{case}'
            require(read(folder / 'result.json') == row and file_sha(folder / 'result.json') == item['result_sha256'], 'Frozen cap result changed')
            if row['status'] in ('success', 'empty'):
                require(file_sha(row['file']) == row['sha256'], 'Frozen cap values changed')
            for wire in row.get('wire_responses', []):
                require(file_sha(wire['file']) == wire['sha256'], 'Frozen cap wire changed')
            reused[case] = row
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'reused_cap_samples': reused,
        'cap_receipt_sha256': file_sha('docs/handoff/2026-10-08-batch68-verification.json'), 'not_a_backtest': True}
    if directory is not None:
        require(result == read(directory / 'input-binding.json'), 'Binding changed')
    return result


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return previous.parsed_source(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch71-functions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    fn = ak.stock_zh_valuation_baidu
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'function': fn.__name__,
        'signature': str(inspect.signature(fn)), 'source': inspect.getsource(fn), 'queries': QUERIES,
        'policy': 'Two cases, one GET each, fixed URL/parameters, no proxy/redirect/retry, timeout8/10, parent45',
        'limits': ['Current vendor chart versions and seven frozen samples cannot prove original whole-pool daily total-cap history']}


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    command = ['rg', '-n', '^def ('+'|'.join(HELPERS)+')\\(', 'repo', '-g', '*.py', '-g', '*.txt']
    result = subprocess.run(command, capture_output=True, text=True)
    require(result.returncode in (0, 1), result.stderr)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'helper_search': {'command': command, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'required_external_helpers': list(HELPERS), 'not_a_backtest': True,
        'gates': ['No admitted daily total cap, historical total-share event chain or original index/industry membership',
            'Source53 external weighted-score and incremental-order sizing definitions missing',
            'Source59 every_bar frequency and same-day valuation/paused availability unproved',
            'Source59 incremental buys, immediate cash release and mutable position iteration cannot be silently replaced by target weights',
            'Daily PE predicate is arithmetic on stored supplier rows, not original candidate ranking or trading']}


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch71 original smallcap/PE source protection baseline started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    folder = directory / 'source-reviews'; folder.mkdir(); frozen = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name
        require(file_sha(path) == sha, 'Source SHA changed')
        shutil.copyfile(path, folder / f'{strategy_id(name)}.source')
        frozen.append({'path': name, 'sha256': sha})
    save(directory / 'sources-frozen.json', frozen)
    return {'status': 'ok'}


def start(root, directory):
    binding(root, directory)
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = []; refs = []
    for i, name in enumerate(SOURCES):
        original = Path('repo/量化策略源代码') / name
        copied = directory / 'source-reviews' / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and
            file_sha(original) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        text, encoding = read_source(copied)
        rows.append({'source_path': str(original), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()),
            'definition_ast': definitions(previous.parsed_source(text)), 'original_line_numbers_retained': True,
            'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / path.name
        shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Incremental versus target stock orders and cap parser comparison; no replacement ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence())
    save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch71 two complete smallcap/PE sources reviewed; original pool/cap/wizard/timing gates retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            row['source_sha256'] == file_sha(row['source_path']) == file_sha(row['source_copy']) == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['original_line_numbers_retained'] and
            row['definition_ast'] == definitions(previous.parsed_source(text)), 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and
            file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; scheduled = []
    context = SimpleNamespace(current_dt=dt.datetime(2024, 1, 2, 9, 30),
        portfolio=SimpleNamespace(positions={}, cash=1000.))
    def error(case, kind, fn):
        try:
            fn()
        except kind as exc:
            rows.append({'case': case, 'error': type(exc).__name__, 'orders': orders.copy()})
        else:
            raise ValueError('Expected original failure: '+case)
    g = SimpleNamespace(); ns = selected(directory, 0,
        [n.name for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef)],
        {'g': g, 'pd': pd, 'log': SimpleNamespace(set_level=lambda *a: None),
            'valuation': SimpleNamespace(market_cap='market_cap'),
            'set_benchmark': lambda *a: None, 'set_slippage': lambda *a: None,
            'set_option': lambda *a: None, 'set_order_cost': lambda *a, **k: None,
            'OrderCost': lambda **k: k, 'FixedSlippage': lambda n: n,
            'run_daily': lambda fn, time: scheduled.append([fn.__name__, time])})
    ns['initialize'](context)
    require((g.max_hold_stocknum, g.check_stocks_refresh_rate, g.buy_refresh_rate, g.filter_holded) == (10, 5, 1, False), 'Wizard defaults differ')
    rows.append({'case': 'wizard_defaults_five_callbacks_ten_slots', 'scheduled': scheduled.copy(),
        'score_input': ns['get_check_stocks_sort_input_dict'](), 'outer_sort': g.check_out_lists_ascending,
        'industry_count': len(g.industry_list), 'external_helpers_proved': False})
    ns['get_index_stocks'] = lambda code: ['B', 'A', 'A']
    require(ns['get_security_universe'](context, ['000300.XSHG'], []) == ['A', 'B'], 'Pool sort/dedupe differs')
    rows.append({'case': 'wizard_pool_sorted_deduplicated', 'pool': ['A', 'B']})
    rows.append({'case': 'wizard_user_security_string_extends_characters',
        'pool': ns['get_security_universe'](context, [], ['AB'])})
    current = {name: SimpleNamespace(is_st=False, paused=False, name=name, day_open=10., high_limit=11.) for name in ('A', 'B', 'C', 'D')}
    current['A'].name = '*A'; current['B'].name = '退B'; current['C'].day_open = 11.
    ns['get_current_data'] = lambda: current
    require(ns['delisted_filter'](context, list(current)) == ['C', 'D'], 'Name filter differs')
    require(ns['high_limit_filter'](context, ['C', 'D']) == ['D'], 'Equality limit filter differs')
    rows.append({'case': 'wizard_name_star_retirement_and_open_equal_limit_filters', 'after_names': ['C', 'D'], 'after_limit': ['D']})
    context.portfolio.positions = {'D': SimpleNamespace()}
    require(ns['holded_filter'](context, ['C', 'D']) == ['C'], 'False excludes held differs')
    rows.append({'case': 'wizard_false_hold_filter_excludes_existing', 'remaining': ['C']})
    error('wizard_missing_score_helper_NameError', NameError,
        lambda: ns['check_stocks_sort'](context, ['A'], {'market_cap': ('asc', 50)}))
    ns['get_sort_dataframe'] = lambda *a: pd.DataFrame({'score_input': [1., 2.]}, index=['A', 'B'])
    error('wizard_removed_DataFrame_sort', AttributeError,
        lambda: ns['check_stocks_sort'](context, ['A', 'B'], {'market_cap': ('asc', 50)}))
    require(ns['check_stocks_sort'](context, [], {'market_cap': ('asc', 50)}) == [], 'Empty sort differs')
    rows.append({'case': 'wizard_empty_pool_avoids_unknown_sort', 'result': []})
    calls = []; ns['get_security_universe'] = lambda *a: calls.append(g.check_stocks_days) or ['D']
    for name in ('industry_filter', 'concept_filter', 'st_filter', 'delisted_filter', 'financial_statements_filter',
            'situation_filter', 'technical_indicators_filter', 'pattern_recognition_filter', 'other_func_filter'):
        ns[name] = lambda context, values, *a: values
    ns['check_stocks_sort'] = lambda context, values, *a: values
    for _ in range(11):
        ns['check_stocks'](context)
    require(calls == [0, 5, 5], 'Five callback counter differs')
    rows.append({'case': 'wizard_selection_callback_numbers_1_6_11', 'counter_at_selection': calls, 'final_counter': g.check_stocks_days})
    ns['order_target_value'] = lambda *a: orders.append(['target', *a])
    g.open_sell_securities = ['D', 'D', 'missing']; orders.clear(); ns['sell_every_day'](context)
    require(orders == [['target', 'D', 0]] and g.open_sell_securities == ['D'], 'Retry pending sells differs')
    rows.append({'case': 'wizard_None_pending_sell_retained', 'orders': orders.copy(), 'pending': g.open_sell_securities.copy()})
    g.sell_will_buy = False; context.portfolio.positions = {'A': SimpleNamespace()}; orders.clear()
    error('wizard_missing_sell_helper_NameError', NameError, lambda: ns['sell'](context, ['B']))
    def immediate_sell(context, stock, *a):
        del context.portfolio.positions[stock]
    ns['sell_by_amount_or_percent_or_none'] = immediate_sell
    ns['sell'](context, ['B']); require(g.selled_security_list == {}, 'Dynamic keys unexpectedly snapshot sold')
    rows.append({'case': 'wizard_dynamic_keys_lose_immediate_sold_history', 'sold_counter': {}})
    context.portfolio.positions = {}; g.check_out_lists = ['D']; orders.clear()
    error('wizard_missing_order_style_NameError', NameError, lambda: ns['buy'](context, ['D']))
    ns['order_style'] = lambda *a: {'D': 100.}
    ns['judge_security_max_proportion'] = lambda *a: a[2]
    ns['max_buy_value_or_amount'] = lambda *a: 100
    ns['MarketOrderStyle'] = lambda: 'synthetic-style'
    ns['order'] = lambda *a: orders.append(['increment', *a])
    ns['buy'](context, ['D'])
    rows.append({'case': 'wizard_external_size_stubs_only_increment_intent', 'orders': orders.copy(), 'real_size_rule_proved': False})
    context.portfolio.positions = {str(i): SimpleNamespace() for i in range(11)}
    candidates = [str(i) for i in range(20, 25)]; g.check_out_lists = candidates
    captured = []; ns['order_style'] = lambda context, values, *a: captured.append(values.copy()) or {}
    ns['buy'](context, candidates); require(captured == [candidates[:-1]], 'Negative slots slice differs')
    rows.append({'case': 'wizard_overfull_negative_slice_calls_allocator_but_no_order', 'allocator_candidates': captured[0]})
    scheduled.clear(); rg = SimpleNamespace(); valuations = []; price_calls = []; index_calls = []
    class Field:
        def __init__(self, name): self.name = name
        def in_(self, values): return ['in', self.name, list(values)]
        def __gt__(self, value): return ['gt', self.name, value]
        def asc(self): return ['asc', self.name]
    class Query:
        def filter(self, *args): self.filters = args; return self
        def order_by(self, *args): self.sort = args; return self
    platform = SimpleNamespace(code=Field('code'), market_cap=Field('market_cap'), pe_ratio=Field('pe_ratio'))
    rn = selected(directory, 1, ['initialize', 'daily'], {'g': rg, 'valuation': platform,
        'run_daily': lambda fn, **k: scheduled.append([fn.__name__, k]), 'query': lambda *a: Query()})
    rn['initialize'](context)
    def stocks(code):
        index_calls.append(code); return ['A', 'B'] if code == '000001.XSHG' else ['C']
    def price(*args, **kwargs):
        price_calls.append(canonical({'args': args, 'kwargs': kwargs}))
        return {'paused': pd.DataFrame([[0, 1, 0]], index=[context.current_dt], columns=['A', 'B', 'C'])}
    def fundamentals(query, **kwargs):
        valuations.append(canonical({'filters': query.filters, 'sort': query.sort, 'kwargs': kwargs}))
        return pd.DataFrame({'code': ['A', 'C']})
    rn['get_index_stocks'] = stocks; rn['get_price'] = price; rn['get_fundamentals'] = fundamentals
    rn['order_target'] = lambda *a: orders.append(['target', *a])
    rn['order_value'] = lambda *a: orders.append(['increment_value', *a])
    context.portfolio.positions = {'A': SimpleNamespace(), 'OLD': SimpleNamespace()}; context.portfolio.cash = 1000.; orders.clear()
    rn['daily'](context)
    require(orders == [['target', 'OLD', 0], ['increment_value', 'A', 100.], ['increment_value', 'C', 100.]], 'Rotation intents differ')
    rows.append({'case': 'rotation_same_date_PEgt1_two_indices_increment_retained_cash_div10',
        'orders': orders.copy(), 'scheduled': scheduled.copy(), 'query': valuations[-1], 'price_call': price_calls[-1], 'index_calls': index_calls.copy()})
    callback_selections = [1]; valuations.clear(); orders.clear()
    for k in range(2, 22):
        before = len(valuations); rn['daily'](context)
        if len(valuations) != before: callback_selections.append(k)
    require(callback_selections == [1, 11, 21], 'Ten callback counter differs')
    rows.append({'case': 'rotation_every10_callbacks_not_proved_days', 'selected_callbacks': callback_selections, 'final_counter': rg.days})
    rg.days = 1; orders.clear(); context.portfolio.positions = {'OLD': SimpleNamespace()}
    def delete_position(stock, amount):
        orders.append(['target', stock, amount]); del context.portfolio.positions[stock]
    rn['order_target'] = delete_position
    error('rotation_immediate_sell_mutates_iteration_RuntimeError', RuntimeError, lambda: rn['daily'](context))
    require(rg.days == 1, 'Failed callback must not increment')
    rows.append({'case': 'rotation_exception_preserves_counter', 'counter': rg.days})
    rn['order_target'] = lambda *a: orders.append(['target', *a]); orders.clear(); context.portfolio.positions = {}
    rn['get_fundamentals'] = lambda *a, **k: pd.DataFrame({'code': []})
    rn['daily'](context); require(not orders, 'Empty candidate list unexpectedly orders')
    rows.append({'case': 'rotation_empty_candidates_no_order_but_counter_advances', 'orders': [], 'counter': rg.days})
    rg.days = 1; rn['get_price'] = lambda *a, **k: {'paused': pd.DataFrame(columns=['A', 'B', 'C'])}; orders.clear()
    error('rotation_empty_same_day_paused_IndexError', IndexError, lambda: rn['daily'](context))
    require(len(rows) == 21, 'Diagnostic count differs')
    return canonical({'cases': rows, 'not_a_backtest': True, 'synthetic_fixture': True,
        'platform_equivalent': False, 'real_historical_trade_windows': 0,
        'limits': ['Explicit query/history/order stubs only; no platform helper reconstruction or actual fills',
            'No source59 every_bar-to-trading-day conversion or same-day valuation repair approved']})


def study(root, directory):
    binding(root, directory)
    with redirect_stdout(StringIO()):
        result = diagnostics(directory)
    save(directory / 'diagnostics.json', result)
    return archive(root, 'Batch71 original callback/filter/incremental-order diagnostics archived; no trades')


def worker(directory, case):
    folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=False)
    query = QUERIES[case]
    row = {'case': case, 'request': query, 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        params = kwargs.get('params', {})
        require(not row['attempts'] and method.upper() == 'GET' and url == URL and
            params.get('code') == query['symbol'] and params.get('query') == params.get('tag') == query['indicator'] and
            params.get('chart_select') == query['period'] and params.get('market') == 'ab', 'Unexpected or repeated request')
        row['attempts'].append({'method': method, 'url': url, 'parameters': params})
        session.trust_env = False; kwargs['timeout'] = (8, 10); kwargs['allow_redirects'] = False
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as output: output.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code})
        response.raise_for_status(); return response
    try:
        import akshare as ak
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API/settings changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr):
            frame = ak.stock_zh_valuation_baidu(**query)
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    finally:
        save(folder / 'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); rows = []
    for case in QUERIES:
        require(not (directory / 'probes' / case).exists(), 'Case already attempted')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch71', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired as exc:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'timeout_seconds': 45,
                'stdout': str(exc.stdout), 'stderr': str(exc.stderr)})
            folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'request': QUERIES[case], 'status': 'timeout',
                    'attempts': [], 'wire_responses': [], 'error': 'Worker45s deadline; request completion unproved',
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / 'probes' / case / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False, 'not_a_backtest': True})
    return archive(root, 'Batch71 two bounded new total-cap probes archived; original daily total-cap gate unchanged')


def pe_predicate(tree, values):
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and
        isinstance(n.left, ast.Attribute) and n.left.attr == 'pe_ratio']
    require(len(nodes) == 1 and isinstance(nodes[0].left.value, ast.Name) and nodes[0].left.value.id == 'valuation' and
        len(nodes[0].ops) == len(nodes[0].comparators) == 1 and isinstance(nodes[0].ops[0], ast.Gt) and
        isinstance(nodes[0].comparators[0], ast.Constant) and nodes[0].comparators[0].value == 1, 'Original PE predicate changed')
    expression = ast.Expression(body=nodes[0])
    return eval(compile(expression, '<original-batch71-PE-predicate>', 'eval'), {'valuation': SimpleNamespace(pe_ratio=values)})


def numeric_values(frame, tree):
    frame = frame.copy(); frame['date'] = pd.to_datetime(frame.date)
    dates = frame.date.astype('datetime64[ms]')
    require(frame.date.eq(dates).all(), 'Daily date precision loss')
    frame['date'] = dates
    require(not frame.duplicated(['date', 'instrument']).any(), 'Duplicate daily input')
    beyond = int(frame.date.gt('2026-09-29').sum()); frame = frame[frame.date.le('2026-09-29')]
    frame = frame.sort_values(['date', 'instrument']).reset_index(drop=True)
    frame['original_PE_gt1'] = pe_predicate(tree, frame.pe_ttm)
    expected = np.array([float(value) > 1 for value in frame.pe_ttm], dtype=bool)
    require(np.array_equal(frame.original_PE_gt1.to_numpy(), expected), 'Independent PE predicate differs')
    grouped = frame.groupby('date', sort=True)
    summary = [{'date': str(day.date()), 'stored_rows': len(group), 'predicate_passes': int(group.original_PE_gt1.sum()),
        'nonfinite_PE': int((~np.isfinite(group.pe_ttm)).sum()), 'provider_trading_rows': int(group.is_trading.sum())}
        for day, group in grouped]
    return frame, {'stored_rows': len(frame), 'dates': len(summary), 'rows_after_cutoff_excluded': beyond,
        'first_date': str(frame.date.min().date()) if len(frame) else None,
        'last_date': str(frame.date.max().date()) if len(frame) else None,
        'instruments': int(frame.instrument.nunique()),
        'predicate_passes': int(frame.original_PE_gt1.sum()), 'daily_coverage': summary,
        'not_a_backtest': True, 'original_signal_reproduced': False,
        'limits': ['Supplier stored PE_ttm arithmetic only; not proven same-day platform PE or original index/paused/total-cap ranking',
            'Original >1 predicate retained including NaN/inf comparison behavior; no additional ST or positive-value filter',
            'Historical rows have varying supplier security coverage; no 10-position candidate list, targets, fees or NAV']}


def total_cap_coverage(directory):
    calendar = pd.read_parquet(directory / 'calendar-input.parquet')
    days = set(pd.to_datetime(calendar.loc[calendar.is_open.eq(True), 'date']).dt.date)
    required = {day for day in days if dt.date(2021, 1, 29) <= day <= dt.date(2026, 9, 29)}
    samples = dict(read(directory / 'input-binding.json')['reused_cap_samples'])
    samples.update({r['case']: r for r in read(directory / 'probe-results.json')['results']})
    rows = []
    for case, row in samples.items():
        record = {'case': case, 'status': row['status'], 'strict_usable': False, 'source_pool_equivalent': False}
        for wire in row.get('wire_responses', []):
            require(file_sha(wire['file']) == wire['sha256'], 'Cap wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial cap evidence changed')
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'], 'Cap values changed')
            frame = pd.read_parquet(row['file'])
            raw = json.loads(Path(row['wire_responses'][-1]['file']).read_bytes())
            chart = raw['Result'][0]['DisplayData']['resultData']['tplData']['result']['chartInfo'][0]
            parsed = pd.DataFrame(chart['body'], columns=['date', 'value'])
            parsed['date'] = pd.to_datetime(parsed.date, errors='coerce').dt.date
            parsed['value'] = pd.to_numeric(parsed.value, errors='coerce')
            pd.testing.assert_frame_equal(parsed, frame, check_exact=True)
            require(len(frame) == row['rows'], 'Cap profile changed')
            present = set(frame.date.dropna())
            record.update({'rows': len(frame), 'required_trading_dates': len(required),
                'matched_dates': len(required & present), 'missing_dates': sorted(str(day) for day in required-present),
                'nontrading_dates': len({day for day in present if dt.date(2021,1,29) <= day <= dt.date(2026,9,29)}-days),
                'future_rows_excluded': int(frame.date.gt(dt.date(2026,9,29)).sum()),
                'duplicate_dates': int(frame.date.duplicated().sum()), 'invalid_dates': int(frame.date.isna().sum()),
                'invalid_values': int((~np.isfinite(frame.value) | frame.value.le(0)).sum()), 'unit': chart.get('unit'), 'parser_match': True})
        else:
            require(row.get('error'), 'Failure lacks reason'); record['error'] = row['error']
        rows.append(record)
    return {'samples': rows, 'new_cases': len(QUERIES), 'reused_cases': len(samples)-len(QUERIES),
        'admitted_rows': 0, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'limits': ['Exact-date coverage without fill/interpolation; current vendor chart not historic vintage',
            'Unit from raw response only; limited security samples cannot admit full index/industry pool or daily share events']}


def read_daily(root, directory=None):
    store = Store(root); state = store.state(SNAPSHOT); frames = []; partitions = []
    for _, entry in sorted(state['tables']['bars_1d'].items()):
        path = store.root / entry['file']; frame = pd.read_parquet(path)
        require(len(frame) == entry['rows'] and fingerprint(frame) == entry['sha'], 'Daily partition content changed')
        partitions.append({'file': str(path), 'file_sha256': file_sha(path), 'fingerprint': entry['sha'], 'rows': len(frame)})
        frames.append(frame[['date', 'instrument', 'pe_ttm', 'is_trading']])
    calendar = store.load_state(state, 'calendar')
    bound = {'snapshot': SNAPSHOT, 'partitions': partitions, 'calendar_fingerprint': fingerprint(calendar)}
    if directory is not None: require(bound == read(directory / 'daily-input-binding.json'), 'Daily input binding changed')
    return pd.concat(frames, ignore_index=True), calendar, bound


def prepare(root, directory):
    binding(root, directory); validate_sources(directory)
    require(not (directory / 'daily-predicate-values.parquet').exists(), 'Component already archived')
    frame, calendar, bound = read_daily(root)
    save(directory / 'daily-input-binding.json', bound)
    calendar.to_parquet(directory / 'calendar-input.parquet', index=False)
    values, summary = numeric_values(frame, source_tree(directory, 1))
    values.to_parquet(directory / 'daily-predicate-values.parquet', index=False)
    save(directory / 'component-research.json', summary)
    save(directory / 'total-cap-coverage.json', total_cap_coverage(directory))
    return archive(root, 'Batch71 long supplier PE predicate/9 exact-date cap samples archived; original selection/trades gated')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **k): raise ValueError('Batch71 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); cat = offline_catalog(root, directory); dep = inventory(root)
        frame, calendar, _ = read_daily(root, directory)
        values, summary = numeric_values(frame, source_tree(directory, 1)); caps = total_cap_coverage(directory)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'daily-predicate-values.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(calendar, pd.read_parquet(directory / 'calendar-input.parquet'), check_exact=True)
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and summary == read(directory / 'component-research.json') and
        caps == read(directory / 'total-cap-coverage.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'evidence_sha256': {name: file_sha(directory / name) for name in
            ('diagnostics.json', 'daily-predicate-values.parquet', 'component-research.json', 'total-cap-coverage.json')}, 'not_a_backtest': True})
    return archive(root, 'Batch71 forbidden-network PE/diagnostics/cap parses and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch71.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch71_research.py',
            'tests/unit/test_batch68_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
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
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'], 'Offline proof changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    for row in read(directory / 'daily-input-binding.json')['partitions']:
        require(file_sha(row['file']) == row['file_sha256'], 'Original daily bytes changed')
    protection = protect(root, directory)
    progress = archive(root, 'Batch71 smallcap/PE source research accepted; original total-cap/pool/wizard/timing trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch71-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory / 'diagnostics.json'),
        'component': read(directory / 'component-research.json'), 'supplements': read(directory / 'probe-results.json'),
        'cap_coverage': read(directory / 'total-cap-coverage.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'worker', 'probe', 'prepare', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
