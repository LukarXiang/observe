"""Audit original wizard callbacks; missing helpers and historical inputs block trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch84 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch85/20261009-industry-wizard-money-flow')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2020年度精选策略/32 测试策略1.txt', '2020年度精选策略/16 投资学作业.txt')
SOURCE_SHA = ('cc3ab398876bb13bc9abada1a10ef5716418cfc8704b7a66a5086ad536cf248e',
    '30cad23157f77f32066ca1244b025b5925030de426812913108319435c814dd5')
HELPERS = ('financial_data_filter_dayu', 'financial_data_filter_xiaoyu', 'financial_data_filter_qujian',
    'ipo_days_dayu', 'MA_judge_duotou', 'MA_judge_kongtou', 'money_flow_xiaoyu', 'money_flow_dayu',
    'sell_by_amount_or_percent_or_none', 'order_style', 'judge_security_max_proportion',
    'max_buy_value_or_amount', 'get_sort_dataframe')
REFERENCES = (('repo/akshare', 'akshare/stock/stock_industry_sw.py'),
    ('repo/akshare', 'akshare/stock/stock_fund_em.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py'))
QUERIES = {'industry-history': {'function': 'stock_industry_clf_hist_sw', 'parameters': {}},
    'money-flow-600519': {'function': 'stock_individual_fund_flow', 'parameters': {'stock': '600519', 'market': 'sh'}}}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch85.py', 'tests/unit/test_batch85_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json',
    'diagnostics.json', 'probe-results.json', 'candidate-audit.json', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {n: file_sha(n) for n in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch84-verification.json')
    review = Path('docs/handoff/2026-10-09-batch84-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted84 binding changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted84 evidence changed')
    for n in ('scripts/review_strategy_batch84.py', 'tests/unit/test_batch84_research.py'):
        require(file_sha(n) == accepted['implementation_sha256'][n], 'Accepted84 entrypoint changed')
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Input binding changed')
    return result


def parsed_source(text):
    lines = text.splitlines(); starts = [i for i, line in enumerate(lines) if line == 'from kuanke.wizard import *']
    require(starts == [8], 'Original wizard code boundary changed')
    # Padding preserves original line numbers without executing the non-Python preface.
    return ast.parse('\n'*starts[0]+'\n'.join(lines[starts[0]:]))


def api_evidence():
    import akshare as ak
    functions = {}
    for name in sorted({q['function'] for q in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        functions[name] = {'signature': str(inspect.signature(fn)), 'source': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest()}
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'functions': functions, 'queries': QUERIES,
        'policy': 'Two native single-GET attempts, fixed endpoints, no proxy/redirect/retry, timeout8/10,parent45s',
        'limits': ['SW2021 file version and start/update dates do not prove old28 industry membership/platform code equivalence',
            'Individual money-flow endpoint is recent-only without historical start/end arguments',
            'Vendor main-flow percent is a candidate, not proven original net_pct_main vintage or helper threshold semantics']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    command = ['rg', '-n', '--hidden', '--no-ignore', '^def ('+'|'.join(HELPERS)+')\\(',
        'repo', '-g', '*.py', '-g', '*.txt', '-g', '!**/.git/**']
    search = subprocess.run(command, capture_output=True, text=True)
    require(search.returncode in (0, 1), 'Helper search failed')
    fields = store.load_state(state, 'financial_fields')
    columns = ['table', 'column', 'original_label', 'description', 'unit', 'joinquant_candidate',
        'joinquant_equivalence', 'time_evidence', 'source_sha256', 'rows', 'missing']
    text = fields[['column', 'original_label', 'description', 'joinquant_candidate']].fillna('').astype(str).agg(' '.join, axis=1)
    candidates = fields.loc[text.str.contains('ROE|roe|净资产收益率|营业收入增长|经营利润|股本|实收资本|市盈率|市净率', regex=True), columns]
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'helper_search': {'command': command, 'returncode': search.returncode, 'matches': search.stdout.splitlines(),
            'matched_file_sha256': {n: file_sha(n) for n in sorted({s.split(':', 1)[0] for s in search.stdout.splitlines()})}},
        'missing_modules': [n for n in ('kuanke', 'jqdata', 'jqdatasdk') if importlib.util.find_spec(n) is None],
        'required_helpers': list(HELPERS), 'financial_dictionary_candidate_rows': len(candidates),
        'financial_dictionary_candidates': json.loads(candidates.to_json(orient='records', force_ascii=False)),
        'original_historical_pool_rows_admitted': 0, 'original_financial_rows_admitted': 0,
        'original_MA_signal_windows': 0, 'original_money_flow_signal_windows': 0,
        'new_trading_backtests': 0, 'not_a_backtest': True,
        'gates': ['No registered index_constituents or historical SW industry membership/name chain on this machine',
            'Final annual/quarterly financial records have no proven revision availability; strict usable0',
            'Capitalization shares are not paid-in capital amount; annual marketvalue is not daily cap',
            'Supplier peTTM is not pe_ratio_lyr; ROE/revenue growth/profit ratio definitions and helper endpoints are unknown',
            'MA8/89 helper price/seed/current-bar/inequality semantics unknown; no invented long MA signals',
            'Previous net_pct_main helper date/units/vendor/version unavailable; native recent sample cannot restore full history',
            'Incremental allocation, immediate/failed fills, .02 absolute slippage and platform callback order remain unproved']}


def preflight(root, directory):
    bound = binding(root); old = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in old['implementation_sha256'].items()), 'Accepted implementation changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    for repo, name in REFERENCES: require((Path(repo)/name).is_file(), 'Reference missing')
    checkpoint(root, directory, 'Batch85 parent84 verified; original wizard and historical input gates retained')
    save(directory/'input-binding.json', bound); save(directory/'dependency-inventory.json', inventory(root))
    save(directory/'existing-apis.json', api_evidence())
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False); rows = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(file_sha(path) == sha, 'Source changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'code_start_line': 9,
            'definition_ast': previous.previous.definitions(parsed_source(text))})
    save(directory/'sources-frozen.json', rows)
    return {'status': 'ok', 'sources_frozen': len(rows), 'not_a_backtest': True}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = read(directory/'sources-frozen.json'); refs = []
    for name, row in zip(SOURCES, rows, strict=True):
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == row['source_sha256'], 'Source reviewed/changed')
        row['review'] = REVIEWS[name]; row['newly_reviewed'] = True
    (directory/'references').mkdir(exist_ok=False)
    for repo, name in REFERENCES:
        path = Path(repo)/name; copied = directory/'references'/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Native history/parser and incremental-order reference; no second ledger or helper substitution'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(old) == len(current) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(directory/'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'snapshot': SNAPSHOT, 'not_a_backtest': True})
    return archive(root, 'Batch85 two full wizard sources frozen; helper/financial/pool/flow gates retained')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json'); frozen = read(directory/'sources-frozen.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(frozen) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for name, sha, row, original in zip(SOURCES, SOURCE_SHA, doc['sources'], frozen, strict=True):
        text, encoding = read_source(Path(row['source_copy']))
        require({k: row[k] for k in original} == original and row['source_path'] == str(Path('repo/量化策略源代码')/name) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == sha and
            row['review'] == REVIEWS[name] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == previous.previous.definitions(parsed_source(text)), 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'References missing')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(api_evidence() == read(directory/'existing-apis.json'), 'API changed')


def selected(directory, number, names, namespace):
    row = read(directory/'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Selected source changed')
    return previous.previous.selected_nodes(parsed_source(read_source(Path(row['source_copy']))[0]), names, namespace)


def controlled_context(directory, number, positions=None):
    events = []; g = SimpleNamespace(); context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 2, 9, 30),
        portfolio=SimpleNamespace(positions=dict(positions or {}), available_cash=1000., total_value=2000.))
    namespace = {'g': g, 'pd': pd, 'log': SimpleNamespace(set_level=lambda *a: None),
        'set_benchmark': lambda value: events.append(['benchmark', value]),
        'FixedSlippage': lambda value: value, 'set_slippage': lambda value: events.append(['slippage', value]),
        'set_option': lambda name, value: events.append(['option', name, value]),
        'OrderCost': lambda **kwargs: kwargs, 'set_order_cost': lambda cost, **kwargs: events.append(['cost', cost, kwargs]),
        'run_daily': lambda fn, when: events.append(['schedule', fn.__name__, when]),
        'MarketOrderStyle': lambda: 'controlled-market',
        'valuation': SimpleNamespace(circulating_market_cap='valuation.circulating_market_cap',
            capitalization='valuation.capitalization', pe_ratio_lyr='valuation.pe_ratio_lyr', pb_ratio='valuation.pb_ratio'),
        'indicator': SimpleNamespace(inc_revenue_year_on_year='indicator.inc_revenue_year_on_year',
            operation_profit_to_total_revenue='indicator.operation_profit_to_total_revenue', roe='indicator.roe')}
    tree = parsed_source(read_source(Path(read(directory/'source-reviews/review.json')['sources'][number]['source_copy']))[0])
    selected(directory, number, [n.name for n in tree.body if isinstance(n, ast.FunctionDef)], namespace)
    namespace['initialize'](context)
    return namespace, context, events


def diagnostics(directory):
    cases = []
    for number in range(2):
        ns, context, events = controlled_context(directory, number)
        cases.append({'case': f'{number}_defaults', 'state': vars(ns['g']), 'events': events})
        # External helpers below are trace fixtures, never reconstructed economic definitions.
        for flag in (False, True):
            ns, context, events = controlled_context(directory, number, {'held': 1}); ns['g'].filter_holded = flag
            result = ns['holded_filter'](context, ['held', 'new', 'new'])
            cases.append({'case': f'{number}_holded_{flag}', 'result': result})
        ns, context, events = controlled_context(directory, number); trace = []
        for helper in ('financial_data_filter_dayu', 'financial_data_filter_xiaoyu', 'financial_data_filter_qujian'):
            def financial(stocks, field, threshold, helper=helper):
                trace.append([helper, list(stocks), field, threshold]); return stocks
            ns[helper] = financial
        ns['financial_statements_filter'](context, ['a', 'b'])
        cases.append({'case': f'{number}_financial_helper_arguments', 'trace': previous.previous.plain(trace), 'fixture_only': True})
        ns, context, events = controlled_context(directory, number); trace = []
        ns['ipo_days_dayu'] = lambda ctx, security, days: trace.append([security, days]) or security == 'a'
        result = ns['situation_filter'](context, ['a', 'b'])
        cases.append({'case': f'{number}_IPO_helper_arguments', 'trace': trace, 'result': result, 'fixture_only': True})
        ns, context, events = controlled_context(directory, number); trace = []
        ns['get_index_stocks'] = lambda code: trace.append(['index', code]) or ['b', 'a', 'b']
        result = ns['get_security_universe'](context, ['000300.XSHG'], [])
        ns['get_industry_stocks'] = lambda code: trace.append(['industry', code]) or ['a']
        filtered = ns['industry_filter'](context, result, ['801780', '801110'])
        concepts = ns['concept_filter'](context, result, [])
        expanded = ns['get_security_universe'](context, [], ['ABC'])
        cases.append({'case': f'{number}_pool_order_and_user_string', 'pool': result, 'filtered': filtered,
            'concepts': concepts, 'user_string_expansion': expanded, 'trace': trace, 'fixture_only': True})
        ns, context, events = controlled_context(directory, number); stocks = ['z', 'a', 'z']
        result = ns['check_stocks_sort'](context, stocks, ns['get_check_stocks_sort_input_dict']())
        cases.append({'case': f'{number}_empty_sort_keeps_input', 'same_object': result is stocks, 'result': result})
        ns['get_sort_dataframe'] = lambda *a: pd.DataFrame({'controlled': [1., 2.]}, index=['a', 'b'])
        try: ns['check_stocks_sort'](context, ['a', 'b'], {'x': 1}); error = None
        except Exception as exc: error = type(exc).__name__
        cases.append({'case': f'{number}_inactive_nonempty_sort_compatibility', 'error': error, 'inactive': True})
        for fills in ('immediate', 'failed', 'partial'):
            ns, context, events = controlled_context(directory, number, {'a': 100, 'b': 100}); trace = []
            def predicate(*args, **kwargs): trace.append(['exit_predicate', list(args[1:] if number else args), kwargs]); return True
            ns['MA_judge_kongtou'] = ns['money_flow_dayu'] = predicate
            def sell(ctx, stock, amount, percent, retry):
                trace.append(['sell', stock, amount, percent])
                if fills == 'immediate': ctx.portfolio.positions.pop(stock)
                elif fills == 'partial': ctx.portfolio.positions[stock] = 1
                else: retry.append(stock)
            ns['sell_by_amount_or_percent_or_none'] = sell
            ns['sell'](context, ['a'])
            cases.append({'case': f'{number}_sell_keys_{fills}', 'trace': trace,
                'positions': context.portfolio.positions, 'sold': ns['g'].selled_security_list,
                'retry': ns['g'].open_sell_securities, 'fixture_only': True})
        ns, context, events = controlled_context(directory, number, {'a': 100}); initial = list(context.portfolio.positions)
        context.portfolio.positions.pop('a'); ns['selled_security_list_dict'](context, initial)
        ns['selled_security_list_count'](context)
        cases.append({'case': f'{number}_snapshot_sold_and_close_count', 'sold': ns['g'].selled_security_list,
            'daily_risk': ns['g'].daily_risk_management})
        for count, fills in ((0, False), (1, False), (1, True), (5, False), (6, False)):
            ns, context, events = controlled_context(directory, number, {f'p{i}': 1 for i in range(count)}); trace = []
            ns['order_style'] = lambda ctx, stocks, maximum, style, value: trace.append(['allocation', list(stocks), maximum, style, value]) or dict.fromkeys(stocks, 10)
            ns['judge_security_max_proportion'] = lambda ctx, stock, cash, ratio: trace.append(['proportion', stock, cash, ratio]) or cash
            ns['max_buy_value_or_amount'] = lambda stock, value, maxvalue, maxamount: trace.append(['size', stock, value, maxvalue, maxamount]) or 1
            def order(stock, amount, style):
                trace.append(['order', stock, amount, style])
                if fills: context.portfolio.positions[stock] = 1
            ns['order'] = order
            ns['buy'](context, ['p0', 'a', 'b', 'c', 'd', 'e'])
            cases.append({'case': f'{number}_buy_{count}_{fills}', 'trace': trace,
                'positions': list(context.portfolio.positions), 'fixture_only': True})
        for gate in ('risk_management_signal', 'daily_risk_management'):
            ns, context, events = controlled_context(directory, number); setattr(ns['g'], gate, False)
            ns['buy'](context, ['a'])
            cases.append({'case': f'{number}_buy_gate_{gate}', 'no_helpers_required': True})
        for fills in (False, True):
            ns, context, events = controlled_context(directory, number, {'a': 1, 'b': 1}); trace = []
            ns['g'].open_sell_securities = ['a', 'a', 'absent']
            def target(stock, value):
                trace.append([stock, value])
                if fills: context.portfolio.positions.pop(stock)
            ns['order_target_value'] = target; ns['sell_every_day'](context)
            cases.append({'case': f'{number}_retry_{fills}', 'orders': trace,
                'retry': ns['g'].open_sell_securities, 'fixture_only': True})
        ns, context, events = controlled_context(directory, number); trace = []
        ns['get_current_data'] = lambda: {s: SimpleNamespace(paused=s == 'paused', is_st=s == 'st',
            name={'delisted': '退', 'star': '*name'}.get(s, 'normal'), day_open=10., high_limit=10. if s == 'limit' else 11.)
            for s in ['normal', 'paused', 'st', 'delisted', 'star', 'limit']}
        stocks = ['normal', 'paused', 'st', 'delisted', 'star', 'limit']
        result = {fn: ns[fn](context, stocks) for fn in ('paused_filter', 'st_filter', 'delisted_filter', 'high_limit_filter')}
        cases.append({'case': f'{number}_literal_state_filters', 'results': result, 'fixture_only': True})
        ns, context, events = controlled_context(directory, number); trace = []; ns['g'].check_out_lists = ['a', 'b']
        for fn in ('st_filter', 'paused_filter', 'delisted_filter', 'high_limit_filter'):
            ns[fn] = lambda ctx, stocks, fn=fn: trace.append([fn, list(stocks)]) or stocks
        ns['MA_judge_duotou'] = lambda stock, short, long: trace.append(['MA_entry', stock, short, long]) or stock == 'a'
        ns['money_flow_xiaoyu'] = lambda ctx, stock, field, threshold, **kw: trace.append(['flow_entry', stock, field, threshold, kw]) or stock == 'a'
        ns['sell'] = lambda ctx, stocks: trace.append(['sell', list(stocks)])
        ns['buy'] = lambda ctx, stocks: trace.append(['buy', list(stocks)])
        ns['trade'](context); cases.append({'case': f'{number}_trade_helper_and_sell_buy_order', 'trace': list(trace),
            'buy_days': ns['g'].buy_trade_days, 'sell_days': ns['g'].sell_trade_days, 'fixture_only': True})
        ns['g'].buy_refresh_rate = ns['g'].sell_refresh_rate = 2
        trace.clear(); ns['trade'](context)
        cases.append({'case': f'{number}_trade_skipped_counters', 'trace': list(trace),
            'buy_days': ns['g'].buy_trade_days, 'sell_days': ns['g'].sell_trade_days})
    return {'cases': previous.previous.plain(cases), 'case_count': len(cases), 'not_a_backtest': True,
        'limits': ['Controlled helper return values are fixtures only; thresholds/units/MA/financial/flow kernels remain unproved',
            'Plain dictionary immediate/failed/partial scenarios diagnose Python state behavior, not JoinQuant fill equivalence']}


def expected_request(case):
    require(case in QUERIES, 'Unknown probe case')
    if case == 'industry-history': return 'https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls', None
    return 'https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get', {
        'lmt': '0', 'klt': '101', 'secid': '1.600519', 'fields1': 'f1,f2,f3,f7',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65', 'ut': 'b2884a393a59ad64002292a3e90d46a5'}


def validate_request(case, method, url, parameters):
    target, expected = expected_request(case)
    require(method.upper() == 'GET' and url == target, 'Unexpected method/URL')
    if expected is None: require(parameters is None, 'Unexpected industry parameters')
    else:
        require(isinstance(parameters, dict) and isinstance(parameters.get('_'), int) and
            {k: v for k, v in parameters.items() if k != '_'} == expected, 'Unexpected flow parameters')


def invoke(case):
    import akshare as ak
    query = QUERIES[case]; return getattr(ak, query['function'])(**query['parameters'])


def worker(directory, case):
    require(case in QUERIES, 'Unknown case'); folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(not row['attempts'], 'Repeated request'); validate_request(case, method, url, kwargs.get('params'))
        row['attempts'].append({'method': 'GET', 'url': url, 'params': kwargs.get('params')}); session.trust_env = False
        kwargs.update(timeout=(8, 10), allow_redirects=False); response = original(session, method, url, **kwargs)
        path = folder/'response.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder/'parsed.parquet'
        with path.open('xb') as out: frame.to_parquet(out, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); validate_sources(directory)
    require(not (directory/'probes').exists() and not (directory/'probe-results.json').exists(), 'Probes already attempted')
    rows = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch85', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory/f'worker-{case}.json', {'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/f'worker-{case}.json', {'command': command, 'status': 'timeout', 'seconds': 45})
            folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'result.json').exists(): save(folder/'result.json', {'case': case, 'query': QUERIES[case],
                'status': 'timeout', 'attempts': [], 'wire_responses': [], 'error': '45s deadline; completion unproved',
                'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False})
        rows.append(read(directory/'probes'/case/'result.json')); print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    save(directory/'probe-results.json', {'results': rows, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0})
    return archive(root, 'Batch85 bounded native industry history and recent flow attempts archived; no input substitution')


def candidate_audit(directory):
    require(api_evidence() == read(directory/'existing-apis.json'), 'API changed'); profiles = []
    rows = read(directory/'probe-results.json')['results']
    require(len(rows) == len(QUERIES) and {r['case'] for r in rows} == set(QUERIES), 'Probe scope changed')
    for row in rows:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory/'existing-apis.json'), 'Candidate binding changed')
        for attempt in row['attempts']: validate_request(case, attempt['method'], attempt['url'], attempt['params'])
        require(len(row['attempts']) <= 1 and len(row['wire_responses']) <= 1, 'Repeated probe')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Response changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False, 'wire_responses': len(row['wire_responses'])}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Parsed binding changed')
            calls = []
            def replay(session, method, url, **kwargs):
                require(not calls, 'Repeated replay'); validate_request(case, method, url, kwargs.get('params')); calls.append(True)
                wire = row['wire_responses'][0]; response = requests.Response(); response.status_code = wire['status_code']
                response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(frame) == row['rows'] and list(frame) == row['columns'] and len(calls) == 1, 'Parser scope differs')
            datecolumn = 'start_date' if case == 'industry-history' else '日期'
            if len(frame):
                dates = pd.to_datetime(frame[datecolumn]); profile.update(rows=len(frame), first=str(dates.min()),
                    last=str(dates.max()), post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29')).sum()))
            profile['native_parser_match'] = True
        else: require(row.get('error'), 'Failure reason missing'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0, 'not_a_backtest': True}


def component_summary(directory):
    dep = read(directory/'dependency-inventory.json'); audit = candidate_audit(directory)
    return {'original_MA_windows': 0, 'original_money_flow_windows': 0, 'original_pool_financial_rank_days': 0,
        'cost_backtests': [], 'new_trading_backtests': 0, 'not_a_backtest': True,
        'financial_dictionary_candidate_rows': dep['financial_dictionary_candidate_rows'], 'candidate_profiles': audit['profiles'],
        'longest_interval_conclusion': 'No original signal intervals admitted; unknown external helpers and historical pool/financial/flow inputs block signals and trades',
        'limits': dep['gates']}


def study(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    diag = diagnostics(directory); audit = candidate_audit(directory); summary = component_summary(directory)
    require(before == implementation(), 'Implementation changed during study')
    save(directory/'diagnostics.json', diag); save(directory/'candidate-audit.json', audit); save(directory/'component-research.json', summary)
    return archive(root, 'Batch85 original local callbacks and unknown-helper/longest-input limits archived; trading not admitted')


def offline(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    def forbidden(*a, **k): raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        diag = diagnostics(directory); audit = candidate_audit(directory); summary = component_summary(directory)
        dep = inventory(root); catalog = offline_catalog(root, directory)
    require(diag == read(directory/'diagnostics.json') and audit == read(directory/'candidate-audit.json') and
        summary == read(directory/'component-research.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    require(before == implementation(), 'Implementation changed during offline')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'catalog': catalog, 'new_trading_backtests': 0, 'not_a_backtest': True})
    return archive(root, 'Batch85 forbidden-network original diagnostics/candidate archives/inventory and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch85.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch85_research.py',
            'tests/unit/test_batch84_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Core missing')
    folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for k, command in enumerate(commands()):
        path = folder/f'check-{k}.log'
        with path.open('x') as out: process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': process.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(process.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True); require(not path.exists(), 'Checked copy exists'); shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and current == checked['implementation_sha256'] == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands(), 'Checks changed')
    for row in checked['commands']: require(row['returncode'] == 0 and file_sha(row['log']) == row['sha256'], 'Log changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch85 industry wizard/flow source and original callback research accepted; trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch85-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory/'diagnostics.json'),
        'research': read(directory/'component-research.json'), 'candidates': read(directory/'candidate-audit.json'),
        'inventory': read(directory/'dependency-inventory.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': progress['catalog_sha256']})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'newly_reviewed': 2, 'new_trading_backtests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'start', 'probe', 'worker', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
