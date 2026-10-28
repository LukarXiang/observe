"""Archive original DCA rules and bounded fund/index valuation reference evidence."""
import argparse
import ast
import bisect
from contextlib import redirect_stdout
import datetime as dt
import hashlib
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

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts import review_strategy_batch76 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch77/20261009-DCA-index-valuation')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2020年度精选策略/72 基金定投-沪深300-5年回测.txt',
    '2022年度精选策略/99.量化定投，工薪族逆袭之路(附定投模型）.txt')
SOURCE_SHA = ('40c4560c75f8d1dc104c2f1d36b16c39a51667c597a37fbf776fd4a8e5651608',
    '8281336e38a7a91050e85c073dceaaf985cee639cf07750463f709b3314be958')
FUNDS = ('160706.SZ', '159919.SZ', '511010.SH')
QUERIES = {'fund-'+s[:6]: {'function': 'fund_etf_hist_sina',
    'parameters': {'symbol': ('sh' if s.endswith('SH') else 'sz')+s[:6]}} for s in FUNDS}
QUERIES.update({name: {'function': fn, 'parameters': {'symbol': '沪深300'}}
    for name, fn in (('index-PE', 'stock_index_pe_lg'), ('index-PB', 'stock_index_pb_lg'))})
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_a_pe_and_pb.py'),
    ('repo/akshare', 'akshare/stock_feature/stock_a_indicator.py'),
    ('repo/rqalpha', 'rqalpha/portfolio/account.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch77.py', 'tests/unit/test_batch77_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json', 'probe-results.json',
    'candidate-audit.json', 'diagnostics.json', 'valuation-reference-components.parquet',
    'component-research.json', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch76-verification.json')
    review = Path('docs/handoff/2026-10-09-batch76-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted76 binding changed')
    for name in ('scripts/review_strategy_batch76.py', 'tests/unit/test_batch76_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted76 entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def source_tree(text, number):
    if number == 0:
        lines = text.splitlines()
        require(lines[6] == '原文策略源码如下：' and lines[8] == '# 导入函数库', 'Article prefix changed')
        lines[:8] = ['']*8
        text = '\n'.join(lines)
    return ast.parse(text)


def api_evidence():
    import akshare as ak
    code = {}
    for name in sorted({q['function'] for q in QUERIES.values()}):
        fn = getattr(ak, name); text = inspect.getsource(fn); module = inspect.getmodule(fn)
        row = {'source': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
        if name == 'fund_etf_hist_sina':
            row.update(decoder=module.hk_js_decode, decoder_sha256=hashlib.sha256(module.hk_js_decode.encode()).hexdigest())
        else:
            helper = inspect.getsource(module.get_cookie_csrf)
            row.update(cookie_helper=helper, cookie_helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),
                hash_code_sha256=hashlib.sha256(module.hash_code.encode()).hexdigest())
        code[name] = row
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'functions': code,
        'policy': 'Five serial native calls; fixed1/2 GET endpoints each,8/10s,parent45s,no proxy/retry/redirect; headers/cookies/generated token values not archived',
        'limits': ['Fund close candidates do not prove identities/events/states/adjustment/fund execution and fees',
            'Supplier index PE/PB methods are not original membership harmonic or strictly quartile-trimmed medians',
            'Native row windows do not prove daily coverage/vintages; post20260929 rows retained but excluded']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    tables = ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')
    counts = {t: len(store.load_state(state, t, filters=[('instrument', 'in', list(FUNDS))])) for t in tables}
    members = state['tables'].get('index_constituents', {})
    ledger = Path('src/observe/ledger/book.py')
    return {'snapshot': SNAPSHOT, 'funds': list(FUNDS), 'registered_fund_rows': counts,
        'index_membership_partitions': len(members), 'index_membership_rows': sum(p['rows'] for p in members.values()),
        'ledger_sha256': file_sha(ledger), 'external_cash_flow_support': 'Accepted74 explicit Book external_flows/unit NAV; no duplicate ledger',
        'original_index_valuation_rows_admitted': 0, 'original_trades_admitted': 0, 'not_a_backtest': True,
        'gates': ['Historical000300 member/vintage PE/PB and decision-time available versions unproved',
            '72 month-end historical fundamentals/current_dt and99 previous-date trimmed medians cannot use supplier index fields as equivalent',
            '160706/159919/511010 fund identities/events/states/adjustment/execution/fee versions unproved',
            '72 ix/month alias/bisect max and99 label[-1]/positions.total_amount original defects retained',
            '99 order quantities versus text money and wrong sale tuple item require explicit economic variant decision',
            '72 has no external cash injection; its TotalCash is not contributions or unit NAV']}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == s for n, s in checked['implementation_sha256'].items()), 'Accepted implementation changed before launch')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch77 DCA/index-valuation launch baseline; actual funds/order units checked first')
    save(directory/'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    save(directory/'dependency-inventory.json', inventory(root)); save(directory/'existing-apis.json', api_evidence())
    return {'status': 'ok', 'new_queries': len(QUERIES)}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(); rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(source_tree(text, number)),
            'article_header_blanked_for_AST_only': 8 if number == 0 else 0, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory/'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch77 two complete DCA/index-valuation sources frozen; no economic rule repairs')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/SOURCES[number]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['review'] == REVIEWS[SOURCES[number]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(source_tree(text, number)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, names, namespace):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    nodes = [n for n in source_tree(read_source(source)[0], number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names) and all(safe_function(n) for n in nodes), 'Unsafe/missing definition')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch77-definitions>', 'exec'), namespace)
    return namespace


def diagnostics(directory):
    validate_sources(directory); rows = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, error, fn):
        try: fn()
        except error as exc: record(case, exception=type(exc).__name__)
        else: raise AssertionError('Expected original error: '+case)
    log = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    orders = []; messages = []; schedules = []; costs = []
    ns = selected(directory, 0, ['initialize', 'fnud_main', 'fund_sell', 'fnud_close', 'on_strategy_end',
        'get_index_pe_date', 'get_index_pe', 'get_index_pes'], {'g': SimpleNamespace(), 'pd': pd, 'np': np, 'bisect': bisect, 'log': log})
    ns.update(set_benchmark=lambda *a: None, set_option=lambda *a: None, set_commission=lambda v: costs.append(v),
        PerTrade=lambda **kw: kw, run_monthly=lambda fn, *a, **kw: schedules.append([fn.__name__, list(a), kw]),
        run_daily=lambda fn, **kw: schedules.append([fn.__name__, [], kw]),
        get_security_info=lambda s: SimpleNamespace(display_name=s, start_date='2005-01-01'),
        send_message=lambda v: messages.append(v), order_value=lambda s, v: orders.append(['value', s, v]),
        order_target_value=lambda s, v: orders.append(['target', s, v]))
    ns['initialize'](None); g = ns['g']
    record('72_defaults_schedule_cost', fund=g.buy_index, benchmark=g.benchmark_index, schedules=schedules.copy(), costs=costs.copy(), last_cash=g.LastCash)
    original_pes = ns['get_index_pes']; context = SimpleNamespace(current_dt=dt.datetime(2020, 2, 3, 9, 30),
        previous_date=dt.date(2020, 1, 31), portfolio=SimpleNamespace(positions_value=0., available_cash=10000., returns=0.))
    for q in (.3999, .4, .6, .8):
        g.LastCash = 1000.; g.Totalmonth = 0; orders.clear()
        ns['get_index_pes'] = lambda c, value=q: SimpleNamespace(ix={(0, 2): value*100})
        ns['fnud_main'](context)
        record('72_month_'+str(q), orders=orders.copy(), last_cash=g.LastCash, total_month=g.Totalmonth, ratio=g.pe_ratio)
    ns['fnud_main'](context); record('72_repeat_month_increases_budget', orders=orders.copy(), last_cash=g.LastCash, total_month=g.Totalmonth)
    ns['get_index_pes'] = lambda c: pd.DataFrame([[1, 2, 50]])
    failure('72_ix_main_error', AttributeError, lambda: ns['fnud_main'](context))
    calls = []
    for prior, held, updated in ((.75, 100., .9), (.76, 0., .9), (.76, 100., .79), (.76, 100., .8)):
        g.pe_now = prior; context.portfolio.positions_value = held; orders.clear(); calls.clear()
        def pes(c, value=updated): calls.append(True); return SimpleNamespace(ix={(0, 2): value*100})
        ns['get_index_pes'] = pes; ns['fund_sell'](context)
        record('72_sell_'+str(prior)+'_'+str(held)+'_'+str(updated), queried=len(calls), orders=orders.copy(), pe_now=g.pe_now)
    field = SimpleNamespace(in_=lambda v: v); query = SimpleNamespace(filter=lambda *a: None)
    ns.update(valuation=SimpleNamespace(code=field), query=lambda *a: query, get_index_stocks=lambda *a: ['SYNTHETIC'])
    for name, values in (('positive', [10., 20.]), ('negative_and_nan', [10., -10., np.nan]), ('empty', [])):
        ns['get_fundamentals'] = lambda *a, v=values: pd.DataFrame({'pe_ratio': v})
        value = ns['get_index_pe_date']('INDEX', context.current_dt)
        record('72_harmonic_'+name, value=None if math.isnan(value) else value)
    ns['get_fundamentals'] = lambda *a: pd.DataFrame({'pe_ratio': [-1., 0.]})
    failure('72_no_positive_PE_division', ZeroDivisionError, lambda: ns['get_index_pe_date']('INDEX', context.current_dt))
    original_pe_date = ns['get_index_pe_date']
    failure('72_month_alias_error', ValueError, lambda: ns['get_index_pe']('INDEX', context.current_dt))
    ns['get_index_pes'] = original_pes
    ns['get_index_pe'] = lambda *a: pd.Series(np.arange(1., 12.))
    ns['get_all_securities'] = lambda *a: SimpleNamespace(ix={g.benchmark_index: SimpleNamespace(display_name='SYNTHETIC')})
    for value in (5., 0.):
        ns['get_index_pe_date'] = lambda *a, v=value: v
        frame = ns['get_index_pes'](context)
        record('72_decile_'+str(value), percentile=float(frame.iloc[0, 2]), number=int(frame.iloc[0, -1]))
    ns['get_index_pe_date'] = lambda *a: 11.
    failure('72_max_decile_bisect_error', IndexError, lambda: ns['get_index_pes'](context))
    ns['get_index_pe_date'] = lambda *a: 5.
    ns['get_index_pe'] = lambda *a: pd.Series([5.]*11)
    failure('72_constant_decile_bisect_error', IndexError, lambda: ns['get_index_pes'](context))
    ns['get_index_pe_date'] = original_pe_date
    context.portfolio.positions_value = 0.; g.TotalCash = 0.; g.MaxCash = 0.
    ns['get_orders'] = lambda: {1: SimpleNamespace(price=10., filled=200, is_buy=True),
        2: SimpleNamespace(price=10., filled=300, is_buy=False), 3: SimpleNamespace(price=10., filled=50, is_buy=True)}
    ns['fnud_close'](context); record('72_order_cash_is_not_external_flow', total_cash=g.TotalCash, max_cash=g.MaxCash, external_flows=0)

    class Table:
        def __init__(self, *a): pass
        def add_row(self, row): pass
    table = SimpleNamespace(PrettyTable=Table)
    ns2 = selected(directory, 1, ['initialize', 'before_trading', 'month_trade', 'evaluate_results', 'get_pe_pb',
        'get_quantile', 'get_save_money', 'inform'], {'g': SimpleNamespace(), 'pd': pd, 'np': np, 'log': log, 'pt': table})
    schedules.clear(); costs.clear()
    ns2.update(set_benchmark=lambda *a: None, set_option=lambda *a: None, OrderCost=lambda **kw: kw,
        set_order_cost=lambda v, **kw: costs.append([v, kw]),
        run_daily=lambda fn, **kw: schedules.append([fn.__name__, kw]),
        run_monthly=lambda fn, **kw: schedules.append([fn.__name__, kw]))
    ns2['initialize'](None); g2 = ns2['g']
    record('99_defaults_mapping_cost_schedule', fund=g2.trade_etf, fallback=g2.national_debt, mapping=g2.index_etf.copy(),
        schedules=schedules.copy(), costs=costs.copy(), minimum=g2.Base_Money_Min, maximum=g2.Base_Money_Max, years=g2.quartile_long)
    for attr, value in (('Take_Risk', False), ('Have_Work', False), ('Need_Pay', True), ('How_Long', 4),
            ('Base_Money_Min', 99), ('Base_Money_Max', 9999)):
        old = getattr(g2, attr); setattr(g2, attr, value)
        record('99_risk_'+attr, accepted=ns2['evaluate_results']()); setattr(g2, attr, old)
    context2 = SimpleNamespace(current_dt=dt.datetime(2020, 2, 3), previous_date=dt.date(2020, 1, 31),
        portfolio=SimpleNamespace(positions={}))
    for q in (.1, .45, .6, .9):
        g2.df_pe_quantile = {'quantile': {-1: q}}
        record('99_budget_'+str(q), result=list(ns2['get_save_money'](context2)))
    g2.Base_Money_Max = 15000; g2.df_pe_quantile = {'quantile': {-1: .1}}
    record('99_budget_cap', result=list(ns2['get_save_money'](context2))); g2.Base_Money_Max = 100000
    g2.df_pe_quantile = pd.DataFrame({'quantile': [.5]}, index=pd.to_datetime(['2020-01-31']))
    failure('99_date_label_minus_one_error', KeyError, lambda: ns2['get_save_money'](context2))
    g2.df_pe_quantile = {'quantile': {-1: .99}}
    failure('99_positions_total_amount_error', AttributeError, lambda: ns2['get_save_money'](context2))
    context2.portfolio.positions = SimpleNamespace(total_amount=123)
    record('99_clear_tuple_with_attribute_stub', result=list(ns2['get_save_money'](context2)))
    flows = []; orders.clear()
    ns2['inout_cash'] = lambda value, **kw: flows.append([value, kw])
    ns2['order'] = lambda s, v: orders.append(['quantity', s, v])
    ns2['order_target_value'] = lambda s, v: orders.append(['target_value', s, v])
    g2.df_pe_pb = pd.DataFrame({'pe': [1.]*5, 'pb': [1.]*5}, index=list(range(-5, 0)))
    for q in (.1, .5, .9, .99):
        g2.df_pe_quantile = {'quantile': {i: q for i in range(-5, 0)}}; flows.clear(); orders.clear()
        ns2['month_trade'](context2)
        record('99_month_'+str(q), flows=flows.copy(), orders=orders.copy())
    ns2['send_message'] = lambda text: messages.append(text)
    ns2['inform'](context2); record('99_inform_local_stub_only', messages=len(messages), external_messages=0)
    pe_field = SimpleNamespace(); pb_field = SimpleNamespace()
    ns2.update(valuation=SimpleNamespace(code=field, pe_ratio=pe_field, pb_ratio=pb_field), query=lambda *a: query,
        get_security_info=lambda *a: SimpleNamespace(start_date=context2.previous_date),
        get_trade_days=lambda **kw: [context2.previous_date], get_index_stocks=lambda *a: ['SYNTHETIC'])
    for name, pe, pb in (('strict_middle', [1., 2., 3., 100.], [4., 3., 2., 1.]),
            ('negative_kept', [-10., -5., -1., 10.], [1., 2., 3., 4.]), ('constant_empty_trim', [5.]*4, [2.]*4)):
        ns2['get_fundamentals'] = lambda *a, x=pe, y=pb: pd.DataFrame({'pe_ratio': x, 'pb_ratio': y})
        frame = ns2['get_pe_pb'](context2, 'INDEX')
        record('99_trim_'+name, pe=None if pd.isna(frame.pe.iloc[0]) else float(frame.pe.iloc[0]),
            pb=None if pd.isna(frame.pb.iloc[0]) else float(frame.pb.iloc[0]))
    prior = pd.DataFrame({'pe': [1.], 'pb': [1.]}, index=[context2.previous_date])
    failure('99_increment_single_trade_day_error', IndexError, lambda: ns2['get_pe_pb'](context2, 'INDEX', prior))
    for name, pe, pb in (('all_tied', [10.]*1708, [1.]*1708), ('missing_PB_at_end', list(range(1708)), [1.]*1707+[np.nan]),
            ('short', [10.]*1707, [1.]*1707)):
        frame = pd.DataFrame({'pe': pe, 'pb': pb}); result = ns2['get_quantile'](None, 'INDEX', frame, 'pe', 7)
        record('99_rank_'+name, rows=len(result), quantile=None if result.empty else float(result['quantile'].iloc[-1]))
    return {'cases': rows, 'not_a_backtest': True, 'original_formulas_unchanged': True,
        'legacy_index_shapes': 'Explicit synthetic ix or label[-1] mappings only to isolate unchanged downstream rules; real compatibility errors retained',
        'external_messages': 0}


def expected_request(case, number, method, url, kwargs):
    require(method.upper() == 'GET', 'Unexpected request method')
    if case.startswith('fund-'):
        endpoint = 'https://finance.sina.com.cn/realstock/company/'+QUERIES[case]['parameters']['symbol']+'/hisdata_klc2/klc_kl.js'
        require(number == 0 and url == endpoint and kwargs.get('params') is None, 'Unexpected fund endpoint')
        return {'method': 'GET', 'url': endpoint, 'parameters': None}
    endpoints = ('https://legulegu.com/stockdata/'+('sz50-ttm-lyr' if case == 'index-PE' else 'zz500-ttm-lyr'),
        'https://legulegu.com/api/stockdata/index-basic-'+('pe' if case == 'index-PE' else 'pb'))
    require(number < 2 and url == endpoints[number], 'Unexpected index endpoint/order')
    params = kwargs.get('params')
    if number == 0: require(params is None, 'Unexpected bootstrap parameters')
    else:
        require(isinstance(params, dict) and set(params) == {'token', 'indexCode'} and params['indexCode'] == '000300.SH' and
            isinstance(params['token'], str) and len(params['token']) == 32 and all(c in '0123456789abcdef' for c in params['token']), 'Unexpected index parameters')
        params = {'indexCode': '000300.SH', 'token': '<redacted-generated>'}
    return {'method': 'GET', 'url': url, 'parameters': params}


def invoke(case):
    import akshare as ak
    q = QUERIES[case]
    return getattr(ak, q['function'])(**q['parameters'])


def worker(directory, case):
    folder = directory/'probes'/case; folder.mkdir(exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        number = len(row['attempts']); row['attempts'].append(expected_request(case, number, method, url, kwargs))
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder/f'response-{number}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'Installed native API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(StringIO()): frame = invoke(case)
        path = folder/'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc:
        row['error'] = type(exc).__name__
        row['error_detail_policy'] = 'Exception text may contain generated URL token; class/status/raw response retained instead'
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); require(api_evidence() == read(directory/'existing-apis.json'), 'Installed native API changed')
    (directory/'probes').mkdir(exist_ok=False); rows = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch77', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory/('worker-'+case+'.json'), {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/('worker-'+case+'.json'), {'command': command, 'status': 'timeout', 'seconds': 45})
            folder = directory/'probes'/case; folder.mkdir(exist_ok=True)
            if not (folder/'result.json').exists():
                save(folder/'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s deadline; completion unproved',
                    'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False})
        rows.append(read(directory/'probes'/case/'result.json')); print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    save(directory/'probe-results.json', {'results': rows, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0})
    return archive(root, 'Batch77 bounded actual fund and supplier index valuation attempts archived; originals not substituted')


def candidate_audit(directory):
    validate_sources(directory); bound = read(directory/'input-binding.json')
    require(file_sha(bound['receipt']) == bound['receipt_sha256'] and file_sha(bound['review']) == bound['review_sha256'], 'Parent binding changed')
    require(api_evidence() == read(directory/'existing-apis.json'), 'Installed native API changed')
    results = read(directory/'probe-results.json')['results']; profiles = []
    require(len(results) == len(QUERIES) and {r['case'] for r in results} == set(QUERIES), 'Candidate scope changed')
    for row in results:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory/'existing-apis.json') and
            row == read(directory/'probes'/case/'result.json'), 'Candidate query changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Raw response changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial response changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False,
            'attempted_requests': len(row['attempts']), 'http_statuses': [w['status_code'] for w in row['wire_responses']]}
        if row['status'] in ('success', 'empty'):
            expected_calls = 1 if case.startswith('fund-') else 2
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == expected_calls, 'Successful capture scope changed')
            calls = []
            def replay(session, method, url, **kwargs):
                number = len(calls); require(number < expected_calls and expected_request(case, number, method, url, kwargs) == row['attempts'][number], 'Unexpected parser replay')
                calls.append(True); wire = row['wire_responses'][number]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(calls) == expected_calls and len(frame) == row['rows'] and list(frame) == row['columns'], 'Native parse scope differs')
            if len(frame):
                dates = pd.to_datetime(frame['date' if case.startswith('fund-') else '日期'])
                require(dates.notna().all(), 'Candidate has invalid dates')
                profile.update(rows=len(frame), first=str(dates.min().date()), last=str(dates.max().date()),
                    post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29')).sum()), unique_dates=int(dates.nunique()))
            profile['native_parser_match'] = True
        else:
            require(row.get('error'), 'Failed request lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'not_a_backtest': True, 'limits': api_evidence()['limits']}


def audit(root, directory):
    binding(root, directory); save(directory/'candidate-audit.json', candidate_audit(directory))
    return archive(root, 'Batch77 native fund/index valuation parse and actual coverage audited; no original input admission')


def components(directory):
    rows = {r['case']: r for r in read(directory/'probe-results.json')['results']}; result = []; blocked = []
    summary = {'not_a_backtest': True, 'new_trading_backtests': 0, 'original72_historical_windows': 0,
        'supplier_rank_windows': 0, 'independent_rank_differences': 0,
        'limits': ['Supplier PE/PB references are not original member/vintage harmonic or strictly trimmed index valuations',
            '1708 native rows are not proven seven calendar years/daily original coverage',
            'No supplier series substituted into original72 monthly or99 stateful trades/flows; no strategy returns',
            'All original99 dropna and average-rank semantics retained; missing PB is never filled']}
    columns = ['field', 'history_end', 'pe', 'pb', 'rank1708', 'independent_rank1708', 'original_index_valuation_equivalent']
    if not all(rows[c]['status'] == 'success' for c in ('index-PE', 'index-PB')):
        summary.update(blocked=['Supplier PE/PB native calls not both successful'], component_rows=0)
        return pd.DataFrame(columns=columns), summary
    frames = {}
    for case in ('index-PE', 'index-PB'):
        row = rows[case]; require(file_sha(row['file']) == row['sha256'], 'Valuation operand changed')
        frame = pd.read_parquet(row['file']).copy(); frame['日期'] = pd.to_datetime(frame['日期'])
        require(frame['日期'].notna().all() and not frame['日期'].duplicated().any(), 'Invalid/duplicate valuation dates')
        frames[case] = frame.loc[frame['日期'] <= pd.Timestamp('2026-09-29')].sort_values('日期').set_index('日期')
    ns = selected(directory, 1, ['get_quantile'], {'pd': pd})
    for field in ('等权静态市盈率', '等权滚动市盈率', '滚动市盈率', '滚动市盈率中位数'):
        data = pd.DataFrame({'pe': pd.to_numeric(frames['index-PE'][field], errors='raise'),
            'pb': pd.to_numeric(frames['index-PB']['市净率中位数'], errors='raise').reindex(frames['index-PE'].index)})
        finite = np.isfinite(data.pe.to_numpy(float))
        require((finite | data.pe.isna().to_numpy()).all(), 'Nonfinite supplier PE')
        require((np.isfinite(data.pb.to_numpy(float)) | data.pb.isna().to_numpy()).all(), 'Nonfinite supplier PB')
        actual = ns['get_quantile'](None, 'SUPPLIER_REFERENCE', data, 'pe', 7)
        values = data.pe.to_numpy(float); expected_dates = []
        for k in range(1707, len(data)):
            window = values[k-1707:k+1]; date = data.index[k]
            if not np.isfinite(window).all() or pd.isna(data.pb.iloc[k]):
                blocked.append([field, str(date.date()), 'missing_PE_window_or_current_PB']); continue
            expected_dates.append(date); current = values[k]
            expected = (np.count_nonzero(window < current)+(np.count_nonzero(window == current)+1)/2)/1708
            observed = float(actual.loc[date, 'quantile'])
            summary['independent_rank_differences'] += observed != expected
            result.append({'field': field, 'history_end': str(date.date()), 'pe': current, 'pb': float(data.pb.iloc[k]),
                'rank1708': observed, 'independent_rank1708': expected, 'original_index_valuation_equivalent': False})
        require(list(actual.index) == expected_dates, 'Original rolling/dropna scope differs')
    frame = pd.DataFrame(result, columns=columns)
    summary.update(component_rows=len(frame), supplier_rank_windows=len(frame), blocked=blocked, blocked_windows=len(blocked),
        first_history_end=frame.history_end.min() if len(frame) else None, last_history_end=frame.history_end.max() if len(frame) else None,
        candidate_PE_rows=len(frames['index-PE']), candidate_PB_rows=len(frames['index-PB']))
    return frame, summary


def study(root, directory):
    require(all(not (directory/n).exists() for n in ('diagnostics.json', 'valuation-reference-components.parquet', 'component-research.json')), 'Study archive exists')
    binding(root, directory); require(api_evidence() == read(directory/'existing-apis.json'), 'Native API changed')
    before = implementation(); diag = diagnostics(directory); frame, summary = components(directory)
    require(before == implementation(), 'Implementation changed during study')
    with (directory/'valuation-reference-components.parquet').open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', {**summary, 'implementation_sha256': before})
    return archive(root, 'Batch77 original DCA diagnostics and longest available valuation reference arithmetic archived')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch77 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); frame, summary = components(directory); candidates = candidate_audit(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory/'valuation-reference-components.parquet'), check_exact=True)
    require(read(directory/'component-research.json') == {**summary, 'implementation_sha256': before} and before == implementation() and
        diag == read(directory/'diagnostics.json') and candidates == read(directory/'candidate-audit.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch77 forbidden-network original diagnostics/native candidates/reference arithmetic and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch77.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch77_research.py', 'tests/unit/test_batch76_research.py',
            'tests/unit/test_external_cash_flows.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(not (directory/'checked-state.json').exists(), 'Checked archive exists'); folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for number, command in enumerate(commands()):
        path = folder/f'check-{number}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        copied = directory/'implementation-checked'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == current == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch77 DCA/index-valuation source and reference research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch77-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'diagnostics': read(directory/'diagnostics.json'), 'components': read(directory/'component-research.json'),
        'inventory': read(directory/'dependency-inventory.json'), 'candidates': read(directory/'candidate-audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'worker', 'probe', 'audit', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY); parser.add_argument('--case', choices=QUERIES)
    args = parser.parse_args(); result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
