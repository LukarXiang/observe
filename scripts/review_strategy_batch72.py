"""Audit dividend selectors, original order callbacks and historical input gates."""
import argparse
import ast
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

from observe.data.store import Store, fingerprint
from observe.runs import canonical, file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts import review_strategy_batch71 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch72/20261008-dividend-income')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY.with_name(previous.DIRECTORY.name+'-verified-recovery')
RECEIPT = Path('docs/handoff/2026-10-08-batch71-verification.json')
PRICE = Path('data/staging/strategies-batch47/20261007-candidate-momentum/price-input.parquet')
SOURCES = ('2021年度精选策略/70.超稳的股息率+均线选股策略.txt',
    '2022年度精选策略/12.回馈社区，分享市盈率股息率选股+均线止盈交易策略.txt')
SOURCE_SHA = ('d519c60fc103d030fd4b84482543c90765853fae863a880123a4f1ba90f235cb',
    '8d97f08edd62f687ad8eda4a7f3bdfac52ed6fc60968e0c6e15c7c6e503a2460')
REFERENCES = (('repo/akshare', 'akshare/stock_fundamental/stock_finance_sina.py'),
    ('repo/akshare', 'akshare/stock/stock_dividend_cninfo.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py'))
QUERIES = {f'{provider}-{symbol}': {'function': 'stock_history_dividend_detail' if provider == 'sina' else 'stock_dividend_cninfo',
    'parameters': {'symbol': symbol, 'indicator': '分红', 'date': ''} if provider == 'sina' else {'symbol': symbol}}
    for symbol in ('000001', '600036', '600519') for provider in ('sina', 'cninfo')}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch72.py', 'tests/unit/test_batch72_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json',
    'corp-action-input.parquet', 'corp-input-binding.json', 'price-input.parquet', 'calendar-input.parquet',
    'diagnostics.json', 'supplier-PE-components.parquet', 'historical-mean-components.parquet',
    'component-research.json', 'probe-results.json', 'dividend-input-audit.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    price_sha = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']['price-input.parquet']
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha(PRICE) == price_sha, 'Accepted inputs changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'price_file': str(PRICE), 'price_sha256': price_sha,
        'supplier_PE_file': str(ACCEPTED / 'daily-predicate-values.parquet'),
        'supplier_PE_sha256': receipt['checks']['evidence_sha256']['daily-predicate-values.parquet'], 'not_a_backtest': True}
    require(file_sha(result['supplier_PE_file']) == result['supplier_PE_sha256'], 'Frozen supplier PE changed')
    if directory is not None:
        require(result == read(directory / 'input-binding.json'), 'Binding changed')
        require(file_sha(directory / 'price-input.parquet') == price_sha, 'Frozen price copy changed')
    return result


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and not n.returns and
        all(a.annotation is None for a in (*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs)) and
        all(a is None or a.annotation is None for a in (n.args.vararg, n.args.kwarg)) and not n.type_params and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch72-functions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    names = sorted({q['function'] for q in QUERIES.values()}); code = {}
    for name in names:
        fn = getattr(ak, name); source = inspect.getsource(fn)
        code[name] = {'signature': str(inspect.signature(fn)), 'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    js = ak.stock_dividend_cninfo.__globals__['_get_file_content_ths']('cninfo.js')
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'functions': code,
        'cninfo_installed_js_sha256': hashlib.sha256(js.encode()).hexdigest(),
        'policy': 'Three securities/two providers, one request per case, no proxy/redirect/retry, timeout8/10,parent45; no auth headers archived',
        'limits': ['Per-share cash/event references are not STK_XR_XD total bonus amounts or JY daily dividend-ratio vintages']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); actions = store.load_state(state, 'corp_actions')
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'corp_action_columns': list(actions), 'corp_actions_rows': len(actions),
        'missing_fields': [name for name in ('bonus_amount_rmb', 'report_date', 'announcement_date', 'DividendRatio', 'InnerCode') if name not in actions],
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'gates': ['Source70 needs historical HS300, available income.statDate and total bonus amounts with units/versions, plus daily total cap',
            'Source12 ignores its Shanghai index list: full-market total-cap top100, JY InnerCode and exact-date DividendRatio required',
            'Existing per-share actions lack original amount/report-period/availability fields; no current-share multiplication substitution',
            'Current supplier PE and 35-row historical mean are components, not original same-day fundamentals, 09:30 price or mavg',
            'Original repeated incremental orders, same-day amount, implicit zero positions, instant cash and callback frequency unproved']}


def preflight(root, directory):
    bound = binding(root); require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch72 dividend source preflight; previous accepted/failed directories protected')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    shutil.copyfile(PRICE, directory / 'price-input.parquet')
    folder = directory / 'source-reviews'; folder.mkdir(); frozen = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; require(file_sha(path) == sha, 'Source changed')
        shutil.copyfile(path, folder / f'{strategy_id(name)}.source'); frozen.append({'path': name, 'sha256': sha})
    save(directory / 'sources-frozen.json', frozen); return {'status': 'ok'}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = []; refs = []
    for i, name in enumerate(SOURCES):
        original = Path('repo/量化策略源代码') / name; copied = directory / 'source-reviews' / f'{strategy_id(name)}.source'
        text, encoding = read_source(copied)
        require(old[name]['review_status'] != '人工审查完成' and
            file_sha(original) == file_sha(copied) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        rows.append({'source_path': str(original), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'original_line_numbers_retained': True, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Dividend dates/round_lot and incremental/target orders; parser comparison only, no second ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changes')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'snapshot': SNAPSHOT, 'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch72 two complete dividend sources reviewed; original amounts/JY/pool/cap/time gates retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(ast.parse(text)) and row['original_line_numbers_retained'], 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


class Field:
    def __init__(self, name): self.name = name
    def in_(self, values): return ['in', self.name, list(values)]
    def __eq__(self, value): return ['eq', self.name, value]
    def __lt__(self, value): return ['lt', self.name, value]
    def __gt__(self, value): return ['gt', self.name, value]
    def __le__(self, value): return ['le', self.name, value]
    def desc(self): return ['desc', self.name]


class Query:
    def __init__(self, *fields): self.fields = [f.name for f in fields]; self.filters = []; self.sort = []; self.count = None
    def filter(self, *args): self.filters = list(args); return self
    def order_by(self, *args): self.sort = list(args); return self
    def limit(self, count): self.count = count; return self


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; scheduled = []; calls = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, kind, fn):
        try: fn()
        except kind as exc:
            details = {}; tb = exc.__traceback__
            while tb:
                if tb.tb_frame.f_code.co_name == 'controlReport':
                    details = {name: tb.tb_frame.f_locals.get(name) for name in ('statqs', 'one_year', 'two_year', 'three_year', 'market_cap')}
                tb = tb.tb_next
            record(case, error=type(exc).__name__, orders=orders.copy(), financial_locals=details)
        else: raise ValueError('Expected original failure: '+case)
    valuation = SimpleNamespace(code=Field('code'), market_cap=Field('market_cap'), pe_ratio=Field('pe_ratio'))
    income = SimpleNamespace(statDate=Field('statDate'), code=Field('income.code'))
    table = SimpleNamespace(bonus_amount_rmb=Field('bonus_amount_rmb'), report_date=Field('report_date'), code=Field('dividend.code'))
    g = SimpleNamespace(); context = SimpleNamespace(current_dt=dt.datetime(2024, 1, 2, 9, 30), previous_date=dt.date(2023, 12, 29),
        portfolio=SimpleNamespace(long_positions={}, positions={}, total_value=1000., available_cash=1000.))
    ns = selected(directory, 0, ['initialize', 'main', 'set_param', 'getBigStocks', 'controlReport', 'setSmallStocks', 'orderStock'],
        {'g': g, 'pd': pd, 'valuation': valuation, 'income': income, 'finance': SimpleNamespace(STK_XR_XD=table), 'query': Query,
            'run_monthly': lambda fn, *a, **k: scheduled.append([fn.__name__, list(a), k]),
            'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'set_order_cost': lambda *a, **k: None,
            'OrderCost': lambda **k: k, 'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)})
    with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.max_colwidth', 100): ns['initialize'](context)
    require(g.stock_num == 5, 'Original stock_num differs'); record('dividend70_monthly_five_and_no_explicit_slippage', scheduled=scheduled.copy())
    ns['get_index_stocks'] = lambda code: calls.append(code) or ['A', 'B']
    ns['get_fundamentals'] = lambda q, **k: calls.append(canonical({'query': q.__dict__, 'kwargs': k})) or pd.DataFrame()
    ns['getBigStocks'](); record('dividend70_HS300_PEgt0_current_fundamentals', calls=calls.copy())
    qcalls = []; dividends = pd.DataFrame({'bonus_amount_rmb': [600., 400., 200.],
        'report_date': [dt.date(y, 12, 31) for y in (2024, 2023, 2022)]})
    def finance_query(q): qcalls.append(canonical(q.__dict__)); return dividends.copy()
    ns['finance'].run_query = finance_query; ns['get_fundamentals'] = lambda q: pd.DataFrame({'market_cap': [1.]})
    report = pd.DataFrame({'statDate': ['2024-12-31'], 'code': ['A']})
    failure('dividend70_qualifying_native_append_AttributeError', AttributeError, lambda: ns['controlReport'](report))
    record('dividend70_query_report_date_is_not_availability', query=qcalls[-1], unit_equivalence_proved=False)
    for label, amounts in (('year1', [200., 400., 500.]), ('year2', [600., 160., 300.]),
            ('year3', [600., 400., 128.]), ('sum3', [400., 400., 200.])):
        dividends['bonus_amount_rmb'] = amounts
        require(ns['controlReport'](report).empty, 'Original strict threshold equality passes')
        record('dividend70_'+label+'_equality_fails', amounts=amounts)
    dividends = pd.DataFrame({'bonus_amount_rmb': [999999.], 'report_date': [dt.date(2024, 2, 1)]})
    require(ns['controlReport'](report).empty, 'February q0 unexpectedly counted')
    record('dividend70_report_month_floor_division_q0_excluded', result_empty=True)
    dividends['report_date'] = None
    failure('dividend70_missing_report_date_fill_zero_AttributeError', AttributeError, lambda: ns['controlReport'](report))
    for count in (10, 11, 19, 20, 100):
        values = pd.DataFrame({'code': [f'S{i:03}' for i in range(count)], 'gxl': np.arange(count),
            'gxl2': np.arange(count), 'gxl3': np.arange(count)})
        ns['setSmallStocks'](values)
        expected = 0 if count <= 10 else count-10 if count < 20 else 10
        require(len(g.bten) == expected and len(g.bfive) == min(expected, 5), 'Cascade count differs')
        record('dividend70_cascade_count_'+str(count), bten=g.bten.copy(), bfive=g.bfive.copy())
    ns['order_target_value'] = lambda *a: orders.append(['target_value', *a])
    ns['attribute_history'] = lambda *a: pd.DataFrame({'close': [100.]*35})
    ns['get_current_data'] = lambda: {'A': SimpleNamespace(last_price=100.)}
    g.bten = ['A']; g.bfive = []; context.portfolio.available_cash = 100.; orders.clear()
    ns['orderStock'](context); require(not orders, 'Half-budget equality unexpectedly buys')
    record('dividend70_cash_equality_blocked_at_price_mean_equality', orders=[])
    context.portfolio.available_cash = 101.; ns['orderStock'](context)
    require(orders == [['target_value', 'A', 200.]], 'Target budget differs')
    record('dividend70_cash_above_half_intends_full_fifth', orders=orders.copy())
    context.portfolio.long_positions = {'A': SimpleNamespace()}; orders.clear()
    def sell_immediate(stock, value): orders.append(['target_value', stock, value]); del context.portfolio.long_positions[stock]
    ns['order_target_value'] = sell_immediate
    failure('dividend70_immediate_sell_mutates_keys_RuntimeError', RuntimeError, lambda: ns['orderStock'](context))
    rg = SimpleNamespace(mavalue=10, buy_value=10000, add_percent=-.1, sell_percent=.16, security=['A'])
    context.portfolio.positions = {}; orders.clear(); current = {'A': SimpleNamespace(is_st=False)}
    data = {'A': SimpleNamespace(close=95., mavg=lambda *a: 100.)}
    rn = selected(directory, 1, ['judgeAndBuy2', 'judgeAndSell2', 'handle_data', 'choosePool', 'DividendRatio2', 'DividendRatio'],
        {'g': rg, 'pd': pd, 'valuation': valuation, 'query': Query, 'log': SimpleNamespace(info=lambda *a: None),
            'get_current_data': lambda: current, 'order_value': lambda *a: orders.append(['increment_value', *a]),
            'order_target': lambda *a: orders.append(['target', *a]), 'record': lambda **k: None})
    failure('dividend12_unheld_plain_dict_KeyError', KeyError, lambda: rn['judgeAndBuy2'](context, data, 'A'))
    position = SimpleNamespace(today_amount=0, closeable_amount=0, avg_cost=100., init_time=context.current_dt, value=100.)
    context.portfolio.positions = {'A': position}; orders.clear(); rn['judgeAndBuy2'](context, data, 'A')
    require(not orders, '95% equality unexpectedly buys'); record('dividend12_price_95percent_equality_no_buy', orders=[])
    data['A'].close = 94.99; rn['judgeAndBuy2'](context, data, 'A'); record('dividend12_below_95percent_fixed_10000', orders=orders.copy())
    position.closeable_amount = 100; data['A'].close = 90.; orders.clear(); rn['judgeAndBuy2'](context, data, 'A')
    require(not orders, '-10% equality unexpectedly adds'); record('dividend12_loss10_equality_no_add', orders=[])
    data['A'].close = 89.; rn['judgeAndBuy2'](context, data, 'A'); record('dividend12_loss_above10_adds_10000', orders=orders.copy())
    position.today_amount = 100; orders.clear(); rn['judgeAndBuy2'](context, data, 'A')
    require(not orders, 'Today100 unexpectedly adds'); record('dividend12_today100_blocks_add', orders=[])
    current['A'].is_st = True; position.today_amount = 1; rn['judgeAndSell2'](context, data, 'A')
    require(not orders, 'ST sold after same-day buy'); record('dividend12_today_positive_blocks_ST_sell', orders=[])
    position.today_amount = 0; rn['judgeAndSell2'](context, data, 'A'); record('dividend12_ST_target_zero', orders=orders.copy())
    current['A'].is_st = False; orders.clear(); data['A'].close = 116.; rn['judgeAndSell2'](context, data, 'A')
    require(not orders, '16% equality unexpectedly sells'); record('dividend12_profit16_equality_no_sell', orders=[])
    data['A'].close = 116.1; rn['judgeAndSell2'](context, data, 'A'); record('dividend12_profit_above16_target_zero', orders=orders.copy())
    data['A'].mavg = lambda *a: 1000000.; data['A'].close = 105.; orders.clear(); rn['judgeAndSell2'](context, data, 'A')
    require(not orders, 'Unused sell MA influences signal'); record('dividend12_sell_MA_unused', orders=[])
    data['A'].close = 89.; orders.clear(); rn['handle_data'](context, data)
    require(orders == [['increment_value', 'A', 10000]]*2, 'Repeated None intents differ')
    record('dividend12_None_buy_repeated_pool_and_positions', orders=orders.copy())
    position.avg_cost = 0.; orders.clear()
    failure('dividend12_zero_cost_ZeroDivisionError', ZeroDivisionError, lambda: rn['judgeAndBuy2'](context, data, 'A'))
    picks = pd.DataFrame({'DividendRatio': [0.04]*11, 'SecuCode': [str(i).zfill(6) for i in range(11)]})
    index_calls = []; rn['get_index_stocks'] = lambda code: index_calls.append(code) or ['SH_UNUSED']
    rn['normalize_code'] = lambda code: code; rn['DividendRatio2'] = lambda *a: picks.copy()
    rn['choosePool'](context); require(len(rg.security) == 10 and '000010' not in rg.security, 'Original first10 differs')
    record('dividend12_unused_Shanghai_pool_04_equality_first10', selected=rg.security.copy(), unused_index=index_calls.copy())
    picks = picks.iloc[:9]; failure('dividend12_under10_choosePool_KeyError', KeyError, lambda: rn['choosePool'](context))
    calls.clear(); rn['get_fundamentals'] = lambda q, **k: calls.append(canonical({'query': q.__dict__, 'kwargs': k})) or pd.DataFrame({'code': ['SZ_A']})
    rn['DividendRatio'] = lambda values, *a: values
    rn_original = selected(directory, 1, ['DividendRatio2'], rn); picked = rn_original['DividendRatio2'](context.previous_date, 10)
    record('dividend12_PE10_25_totalcap_desc100_all_market', query=calls[-1], list_to_dividend=picked)
    inner = SimpleNamespace(InnerCode=Field('InnerCode'), SecuCode=Field('SecuCode'), SecuCategory=Field('SecuCategory'))
    dividend = SimpleNamespace(InnerCode=Field('InnerCode'), TradingDay=Field('TradingDay'), DividendRatio=Field('DividendRatio'))
    def jy_query(q):
        if 'SecuCode' in q.fields: return pd.DataFrame({'InnerCode': [1], 'SecuCode': ['000001']})
        return pd.DataFrame({'InnerCode': [1], 'TradingDay': [context.previous_date], 'DividendRatio': [0.04]})
    rn.update(jy=SimpleNamespace(SecuMain=inner, LC_DIndicesForValuation=dividend, run_query=jy_query),
        get_trade_days=lambda **k: [context.previous_date])
    selected(directory, 1, ['DividendRatio'], rn)
    failure('dividend12_native_DataFrame_sort_AttributeError', AttributeError, lambda: rn['DividendRatio'](['000001.XSHE'], context.previous_date, 10))
    require(len({r['case'] for r in rows}) == len(rows), 'Duplicate diagnostic cases')
    return canonical({'cases': rows, 'not_a_backtest': True, 'synthetic_fixture': True, 'platform_equivalent': False,
        'real_historical_trade_windows': 0, 'limits': ['Explicit finance/query/position/current-price stubs; no actual historical ranking or fills',
            'No compatibility shim applied to original append/sort, no financial amount unit, JY field or immediate-order equivalence claimed']})


def study(root, directory):
    binding(root, directory)
    with redirect_stdout(StringIO()): result = diagnostics(directory)
    save(directory / 'diagnostics.json', result)
    return archive(root, 'Batch72 original quarterly thresholds/cascade/repeated orders/compatibility diagnoses archived; no trades')


def numeric_PE(frame, trees):
    frame = frame[['date', 'instrument', 'pe_ttm', 'is_trading']].copy()
    dates = pd.to_datetime(frame.date)
    frame['date'] = dates.astype('datetime64[ms]')
    require(dates.eq(frame.date).all(), 'Daily date precision loss')
    require(not frame.duplicated(['date', 'instrument']).any() and frame.date.notna().all(), 'Invalid daily identities')
    excluded = int(frame.date.gt('2026-09-29').sum()); frame = frame[frame.date.le('2026-09-29')].sort_values(['date', 'instrument']).reset_index(drop=True)
    comparisons = []
    for tree in trees:
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Attribute) and n.left.attr == 'pe_ratio']
        require(all(isinstance(n.left.value, ast.Name) and n.left.value.id == 'valuation' and len(n.ops) == len(n.comparators) == 1 and
            isinstance(n.comparators[0], ast.Constant) for n in nodes), 'Unsafe PE expression')
        comparisons.append(nodes)
    require(len(comparisons[0]) == 1 and isinstance(comparisons[0][0].ops[0], ast.Gt) and comparisons[0][0].comparators[0].value == 0 and
        len(comparisons[1]) == 2 and {(type(n.ops[0]), n.comparators[0].value) for n in comparisons[1]} == {(ast.Lt, 25), (ast.Gt, 10)}, 'Original PE rules differ')
    for nodes, column in zip(comparisons, ('source70_PE_gt0', 'source12_PE_10_25'), strict=True):
        mask = np.ones(len(frame), dtype=bool)
        for node in nodes:
            mask &= eval(compile(ast.Expression(node), '<original-batch72-PE>', 'eval'), {'valuation': SimpleNamespace(pe_ratio=frame.pe_ttm)})
        frame[column] = mask
    require(np.array_equal(frame.source70_PE_gt0, [float(v) > 0 for v in frame.pe_ttm]) and
        np.array_equal(frame.source12_PE_10_25, [10 < float(v) < 25 for v in frame.pe_ttm]), 'Independent PE comparison differs')
    return frame, {'stored_rows': len(frame), 'instruments': int(frame.instrument.nunique()), 'dates': int(frame.date.nunique()),
        'first': str(frame.date.min().date()) if len(frame) else None, 'last': str(frame.date.max().date()) if len(frame) else None,
        'source70_passes': int(frame.source70_PE_gt0.sum()), 'source12_passes': int(frame.source12_PE_10_25.sum()),
        'excluded_after_cutoff': excluded, 'not_a_backtest': True, 'original_signal_reproduced': False,
        'limits': ['Variable supplier coverage, not full historical market/HS300', 'No original income report, total cap rank or JY dividend ratio supplied']}


def numeric_means(price, calendar, tree):
    nodes = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'thirty_mean' for t in n.targets)]
    require(len(nodes) == 1 and ast.unparse(nodes[0]) == "stock_hitory['close'][-30:].mean()", 'Original mean expression differs')
    expression = compile(ast.Expression(nodes[0]), '<original-batch72-history-mean>', 'eval')
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.eq(True), 'date']))
    require(days.is_unique and days.is_monotonic_increasing and not price.duplicated(['date', 'instrument']).any(), 'Duplicate daily/calendar rows')
    records = []; paused = blocked = 0; difference = 0.
    for stock, frame in price.groupby('instrument', sort=True):
        frame = frame.sort_values('date').copy(); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.isin(days).all() and frame.is_trading.notna().all(), 'Unknown dates or trading status')
        absent = (~days.isin(frame.date)).astype(int).cumsum(); active = frame[frame.is_trading].reset_index(drop=True)
        paused += int((~frame.is_trading).sum()); positions = days.get_indexer(active.date)
        adjusted = active.close_adj.to_numpy(dtype=float)
        for end in range(34, len(active)):
            row = active.iloc[end]; basis = float(row.back_factor); window = adjusted[end-34:end+1]
            if absent[positions[end]]-absent[positions[end-34]] or not math.isfinite(basis) or basis <= 0 or not np.isfinite(window).all() or not (window > 0).all():
                blocked += 1; continue
            values = window/basis; mean = float(eval(expression, {'stock_hitory': pd.DataFrame({'close': values})}))
            independent = math.fsum(float(v) for v in values[-30:])/30
            require(math.isclose(mean, independent, rel_tol=1e-12, abs_tol=1e-12), 'Independent historical mean differs')
            difference = max(difference, abs(mean-independent))
            records.append({'history_end': row.date.strftime('%Y-%m-%d'), 'instrument': stock, 'hist35_last30_mean': mean})
    return pd.DataFrame(records, columns=['history_end', 'instrument', 'hist35_last30_mean']), {
        'windows': len(records), 'known_pause_rows_excluded': paused, 'invalid_or_unknown_windows_blocked': blocked,
        'max_independent_numeric_difference': difference, 'not_a_backtest': True,
        'limits': ['Ten accepted stock histories, not original HS300/selected dividend candidates',
            'Dynamic price anchoring and known-pause exclusion are explicit arithmetic inputs; platform history/09:30 last_price not proved',
            'No latest close used as original current price; no monthly picks, targets, fills, cash, fees or NAV']}


def corp_inputs(root, directory=None):
    store = Store(root); state = store.state(SNAPSHOT); frame = store.load_state(state, 'corp_actions')
    partitions = [{'file': str(store.root / e['file']), 'file_sha256': file_sha(store.root / e['file']), 'fingerprint': e['sha'], 'rows': e['rows']}
        for _, e in sorted(state['tables']['corp_actions'].items())]
    bound = {'snapshot': SNAPSHOT, 'partitions': partitions, 'loaded_fingerprint': fingerprint(frame), 'rows': len(frame)}
    if directory is not None: require(bound == read(directory / 'corp-input-binding.json'), 'Existing actions changed')
    return frame, bound


def prepare(root, directory):
    outputs = ('corp-action-input.parquet', 'corp-input-binding.json', 'calendar-input.parquet',
        'supplier-PE-components.parquet', 'historical-mean-components.parquet', 'component-research.json')
    require(not any((directory / name).exists() for name in outputs), 'Component output exists; use a fresh recovery directory')
    binding(root, directory); validate_sources(directory)
    frame, bound = corp_inputs(root); frame.to_parquet(directory / 'corp-action-input.parquet', index=False); save(directory / 'corp-input-binding.json', bound)
    calendar = Store(root).load_state(Store(root).state(SNAPSHOT), 'calendar'); calendar.to_parquet(directory / 'calendar-input.parquet', index=False)
    pe, pe_summary = numeric_PE(pd.read_parquet(read(directory / 'input-binding.json')['supplier_PE_file']), [source_tree(directory, i) for i in range(2)])
    price = pd.read_parquet(directory / 'price-input.parquet'); means, mean_summary = numeric_means(price, calendar, source_tree(directory, 0))
    pe.to_parquet(directory / 'supplier-PE-components.parquet', index=False); means.to_parquet(directory / 'historical-mean-components.parquet', index=False)
    save(directory / 'component-research.json', {'PE': pe_summary, 'history_mean': mean_summary, 'not_a_backtest': True, 'new_trading_backtests': 0})
    return archive(root, 'Batch72 full supplier PE and long ten-stock historical means archived; original dividend trades gated')


def expected_request(case):
    symbol = QUERIES[case]['parameters']['symbol']
    if case.startswith('sina-'): return 'GET', f'https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{symbol}.phtml', {}
    return 'POST', 'https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139', {'scode': symbol}


def parquet_dates(frame):
    frame = frame.copy()
    for name in frame:
        if pd.api.types.is_datetime64_any_dtype(frame[name]):
            original = frame[name]; converted = original.astype('datetime64[ms]')
            require((original.eq(converted) | (original.isna() & converted.isna())).all(), 'Parser date precision loss')
            frame[name] = converted
        elif frame[name].dtype == object:
            original = frame[name]; converted = original.where(original.notna(), None)
            require(original.notna().eq(converted.notna()).all() and
                original[original.notna()].eq(converted[original.notna()]).all(), 'Parser object values changed')
            frame[name] = converted
    return frame


def worker(directory, case):
    folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=False); query = QUERIES[case]
    row = {'case': case, 'query': query, 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        expected_method, expected_url, expected_params = expected_request(case)
        require(not row['attempts'] and method.upper() == expected_method and url == expected_url and (kwargs.get('params') or {}) == expected_params, 'Unexpected or repeated request')
        row['attempts'].append({'method': method.upper(), 'url': url, 'parameters': expected_params})
        session.trust_env = False; kwargs['timeout'] = (8, 10); kwargs['allow_redirects'] = False
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as output: output.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        import akshare as ak
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API/settings changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr):
            frame = parquet_dates(getattr(ak, query['function'])(**query['parameters']))
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    finally: save(folder / 'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); rows = []
    for case in QUERIES:
        require(not (directory / 'probes' / case).exists(), 'Case already attempted')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch72', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired as exc:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'stdout': str(exc.stdout), 'stderr': str(exc.stderr), 'timeout_seconds': 45})
            folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': 'Worker45s deadline; request completion unproved',
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / 'probes' / case / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'not_a_backtest': True, 'published': False})
    return archive(root, 'Batch72 six bounded dividend input attempts archived; no publication or original signal substitution')


def parse_probes(directory):
    import akshare as ak
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API/settings changed')
    results = read(directory / 'probe-results.json')['results']; require({r['case'] for r in results} == set(QUERIES) and len(results) == len(QUERIES), 'Probe scope differs')
    profiles = []; actions = pd.read_parquet(directory / 'corp-action-input.parquet')
    for row in results:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe binding differs')
        for response in row['wire_responses']: require(file_sha(response['file']) == response['sha256'], 'Response changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial response changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Parsed/request proof changed')
            attempts = []
            def replay(session, method, url, **kwargs):
                expected_method, expected_url, expected_params = expected_request(case)
                require(not attempts and method.upper() == expected_method and url == expected_url and (kwargs.get('params') or {}) == expected_params, 'Unexpected replay request')
                attempts.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()):
                frame = parquet_dates(getattr(ak, QUERIES[case]['function'])(**QUERIES[case]['parameters']))
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(attempts) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Replay profile differs')
            date_column = '公告日期' if case.startswith('sina-') else '实施方案公告日期'
            values = pd.to_datetime(frame[date_column]) if len(frame) else pd.Series([], dtype='datetime64[ms]')
            profile.update(rows=len(frame), columns=list(frame), native_parser_match=True,
                first_announcement=values.min().strftime('%Y-%m-%d') if values.notna().any() else None,
                last_announcement=values.max().strftime('%Y-%m-%d') if values.notna().any() else None,
                invalid_announcement_rows=int(values.isna().sum()), after_cutoff_announcements=int(values.gt('2026-09-29').sum()))
        else: require(row.get('error'), 'Failed probe lacks reason'); profile['error'] = row['error']
        symbol = QUERIES[case]['parameters']['symbol']; stock = symbol+('.SZ' if symbol.startswith('0') else '.SH')
        existing = actions[actions.instrument.eq(stock)]
        profile['existing_action_rows'] = len(existing)
        profile['missing_original_fields'] = ['bonus_amount_rmb', 'historically_available_income_statDate', 'JY_daily_DividendRatio', 'daily_total_cap']
        profiles.append(profile)
    return {'profiles': profiles, 'existing_action_rows': len(actions), 'existing_action_columns': list(actions),
        'new_admitted_rows': 0, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'limits': ['Native parser and exact saved values only; response version not historical availability proof',
            'Per-share amounts/dates and report descriptions do not prove total dividend amounts, JY daily yields or full historical pools',
            'No forward fill, interpolation, current-share multiplication or fee/NAV implementation added']}


def audit(root, directory):
    binding(root, directory); save(directory / 'dividend-input-audit.json', parse_probes(directory))
    return archive(root, 'Batch72 saved dividend parsers/availability profiles verified; amounts/JY and trading gates unchanged')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **k): raise ValueError('Batch72 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); dep = inventory(root); corp, _ = corp_inputs(root, directory); supplements = parse_probes(directory)
        calendar = Store(root).load_state(Store(root).state(SNAPSHOT), 'calendar')
        pe, pe_summary = numeric_PE(pd.read_parquet(read(directory / 'input-binding.json')['supplier_PE_file']), [source_tree(directory, i) for i in range(2)])
        means, mean_summary = numeric_means(pd.read_parquet(directory / 'price-input.parquet'), calendar, source_tree(directory, 0))
        cat = offline_catalog(root, directory)
    for frame, name in ((corp, 'corp-action-input.parquet'), (calendar, 'calendar-input.parquet'),
            (pe, 'supplier-PE-components.parquet'), (means, 'historical-mean-components.parquet')):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / name), check_exact=True)
    summary = {'PE': pe_summary, 'history_mean': mean_summary, 'not_a_backtest': True, 'new_trading_backtests': 0}
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and dep == read(directory / 'dependency-inventory.json') and
        supplements == read(directory / 'dividend-input-audit.json') and summary == read(directory / 'component-research.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'evidence_sha256': {name: file_sha(directory / name) for name in
            ('diagnostics.json', 'supplier-PE-components.parquet', 'historical-mean-components.parquet', 'component-research.json', 'dividend-input-audit.json')}, 'not_a_backtest': True})
    return archive(root, 'Batch72 forbidden-network original diagnostics/PE/mean/native dividend parses and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch72.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch72_research.py',
            'tests/unit/test_batch71_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
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
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory / 'baseline.json')['published'], 'Offline/publication changed')
    corp_inputs(root, directory); protection = protect(root, directory)
    progress = archive(root, 'Batch72 dividend source/PE/mean research accepted; original amount/JY/pool/cap/timing trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch72-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory / 'diagnostics.json'), 'component': read(directory / 'component-research.json'),
        'supplements': read(directory / 'probe-results.json'), 'input_audit': read(directory / 'dividend-input-audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'prepare', 'worker', 'probe', 'audit', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
