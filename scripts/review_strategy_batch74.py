"""Freeze FOF/two-eight source evidence and verify opt-in external-flow ledger."""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import datetime as dt
from fractions import Fraction
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
from scripts import review_strategy_batch73 as previous
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch66 import legacy_comparison, reproduce_legacy

DIRECTORY = Path('data/staging/strategies-batch74/20261008-FOF-two-eight')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2021年度精选策略/14.FOF养老成长基金-v2.0.txt', '2022年度精选策略/1.易方达二八轮动测试.txt')
SOURCE_SHA = ('c2ef8cbfe24740ccd493e59f7ced9bf3c658359adc52a7591441527468a8d848',
              '383eb75ad9102bca609a0280b5c0a5a2c362317e65bc8d9c1080bde4172f2f43')
FUNDS = ('515520.XSHG', '161907.XSHE', '512890.XSHG', '515300.XSHG', '510050.XSHG',
    '159916.XSHE', '512910.XSHG', '510310.XSHG', '512260.XSHG', '512090.XSHG',
    '515000.XSHG', '512040.XSHG', '510900.XSHG', '501021.XSHG', '513050.XSHG', '518880.XSHG')
QUERIES = {f'index-{symbol}': {'function': 'stock_zh_index_daily_em', 'parameters':
    {'symbol': symbol, 'start_date': '19900101', 'end_date': '20260929'}, 'market': 0 if symbol.startswith('sz') else 1}
    for symbol in ('sh000002', 'sh000016', 'sz399006', 'sh000012')}
for instrument in (*FUNDS, '163210.XSHE'):
    if instrument == '510310.XSHG': continue
    symbol = instrument[:6]
    lof = symbol.startswith(('16', '501'))
    QUERIES['fund-'+symbol] = {'function': 'fund_lof_hist_em' if lof else 'fund_etf_hist_em',
        'parameters': {'symbol': symbol, 'period': 'daily', 'start_date': '19900101', 'end_date': '20260929', 'adjust': ''},
        'market': 0 if instrument.endswith('XSHE') else 1}
REFERENCES = previous.REFERENCES[2:]
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch74.py', 'tests/unit/test_batch74_research.py',
    'src/observe/ledger/book.py', 'tests/unit/test_external_cash_flows.py', 'scripts/verify_strategy_batch66.py',
    'scripts/probe_strategy_batch74_fallback.py'}))
CORE = {'baseline.json', 'preflight.json', 'input-binding.json', 'catalog-before.json', 'book.py.before',
    'strategy_catalog.py.before', 'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json',
    'diagnostics.json', 'reused-ETF.json', 'ETF-reference-components.parquet', 'component-research.json',
    'probe-results.json', 'input-audit.json', 'offline-verification.json', 'offline-catalog.json',
    'ledger-validation.json', 'legacy-strategy-reproduction.json', 'external-flow-red.log',
    'diagnostics-expanded.json', 'sina-fallback/input-binding.json', 'sina-fallback/existing-apis.json',
    'sina-fallback/results.json', 'sina-fallback/audit.json', 'index-reference-components.parquet', 'index-component-research.json',
    'archive-guard-red.log', 'ledger-driver-before-archive-guard.py', 'ledger-scope-revalidation.json',
    'book-tested-before-whitespace.py', 'ledger-scope-revalidation-final.json', 'offline-verification-final.json',
    'check-0.log', 'check-1.log', 'check-2.log', 'ledger-scope-revalidation-final-amended.json',
    'dependency-inventory-final.json', 'offline-final-inventory-blocked.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-08-batch73-verification.json')
    proof = read(previous.DIRECTORY / 'final-binding-verification.json')
    require(read(receipt)['status'] == proof['status'] == 'ok' and file_sha(receipt) == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch73 changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'accepted_directory': str(previous.DIRECTORY), 'receipt_sha256': file_sha(receipt), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    return result


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, namespace, *, nested=False):
    tree = source_tree(directory, number)
    scope = ast.walk(tree) if nested else tree.body
    nodes = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names) and
        all(previous.safe_function(n) for n in nodes), 'Unsafe/missing selected function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch74-definitions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    code = {}
    for name in sorted({q['function'] for q in QUERIES.values()}):
        fn = getattr(ak, name); source = inspect.getsource(fn)
        code[name] = {'signature': str(inspect.signature(fn)), 'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'functions': code,
        'policy': '20 serial candidate-history calls; each once, explicit source-suffix market hint, no market discovery, retry, redirect or proxy;8/10s,worker45s',
        'limits': ['Market hint is not independent identity/listing/vintage evidence', 'Unadjusted candidates are not original fq=pre inputs or executable index securities']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'funds': list(FUNDS), 'bond': '163210.XSHE', 'reference': '000002.XSHG',
        'rotation_active_codes': ['000016.XSHG', '399006.XSHE', '000012.XSHG'],
        'original_fund_inputs_admitted': 0, 'original_trades_admitted': 0, 'ledger_sha256': file_sha('src/observe/ledger/book.py'),
        'ledger_external_flows_opt_in': True, 'new_trading_backtests': 0,
        'gates': ['Fund identity/listing/120 money/250 adjusted history/events/state/fees incomplete',
            'FOF handle_data callback frequency and historical pandas fill semantics unproved',
            '000002.XSHG must not become000002.XSHE stock; active rotation indices not silently mapped to ETFs',
            '11 records span ten returns;15:00 callbacks and incremental order_value semantics retained',
            'External-flow component cannot admit a strategy with missing inputs'], 'not_a_backtest': True}


def start(root, directory):
    bound = binding(root); require(bound == read(directory / 'preflight.json'), 'Preflight changed')
    save(directory / 'input-binding.json', bound)
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for i, name in enumerate(SOURCES):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        shutil.copyfile(path, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'original_line_numbers_retained': True, 'review': REVIEWS[name], 'newly_reviewed': True})
    refs = []; (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / (path.parent.name+'-'+path.name); shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'External-flow units reference; all execution and fees remain in sole Book'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder / 'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    receipt66 = Path('docs/handoff/2026-10-08-batch66-verification.json')
    payload = 'data/staging/strategy-dependencies/20261008-batch66-etf-quotes/em_raw/table.parquet'
    row = next(r for r in read(receipt66)['payload_references'] if r['path'] == payload)
    require(file_sha(payload) == row['sha256'] and Path(payload).stat().st_size == row['bytes'], 'Prior ETF changed')
    save(directory / 'reused-ETF.json', {'receipt': str(receipt66), 'receipt_sha256': file_sha(receipt66), 'payload': row,
        'strict_usable': False, 'policy': 'Exact batch66 candidate reused; never replace with fresh510310 download'})
    return archive(root, 'Batch74 full FOF/two-eight sources reviewed; opt-in external-flow development and original input gates retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for i, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / SOURCES[i]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[i] and
            row['review'] == REVIEWS[SOURCES[i]] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(ast.parse(text)) and row['original_line_numbers_retained'], 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def diagnostics(directory):
    rows = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, error, fn):
        try: fn()
        except error as exc: record(case, exception=type(exc).__name__, detail=str(exc)[:250])
        else: raise AssertionError('Expected original failure: '+case)
    cash = []
    ns = selected(directory, 0, ['initializeStockDict', 'getStockRSIRatio', 'getLastestTransactTime', 'stopLoss', 'delNewStock', 'getAvgMoney'],
        {'np': np, 'dt': dt, 'inout_cash': lambda amount, **kw: cash.append([amount, kw]),
         'isnan': math.isnan, 'get_security_info': lambda s: SimpleNamespace(start_date=dt.date(2000, 1, 1))})
    ctx = SimpleNamespace(current_dt=dt.datetime(2026, 9, 24), initPriceDict={}, positionDict={}, transactionRecord={})
    ns['attribute_history'] = lambda *a, **kw: {'money': np.full(120, 2e6)}
    for _ in range(2): ns['initializeStockDict'](ctx)
    require(tuple(ctx.pool) == FUNDS and ctx.stockDict[FUNDS[-1]] == 4 and sum(v == 10 for v in ctx.stockDict.values()) == 15, 'Pool differs')
    record('each_callback_deposits400', calls=cash, pool=ctx.pool, weights=ctx.stockDict, bond=ctx.bondStock, reference=ctx.referStock)
    record('undefined_ratio_returns50', value=ns['getStockRSIRatio']('A'))
    ctx.portfolio = SimpleNamespace(positions={})
    failure('implicit_datetime_missing', NameError, lambda: ns['getLastestTransactTime'](ctx))
    ns['datetime'] = dt
    for days_ago, weekday in ((29, 4), (30, 4), (31, 5)):
        ctx.current_dt = dt.datetime(2026, 9, 24) + dt.timedelta(days=weekday-4)
        ctx.portfolio.positions = {'A': SimpleNamespace(transact_time=ctx.current_dt-dt.timedelta(days=days_ago))}
        record(f'last_trade_{days_ago}d_weekday{weekday}', eligible=ns['getLastestTransactTime'](ctx))
    ctx.current_dt = dt.datetime(2026, 9, 24)
    cutoff = ctx.current_dt.date()-dt.timedelta(days=300)
    ns['get_security_info'] = lambda s: SimpleNamespace(start_date=cutoff if s == 'equal' else cutoff-dt.timedelta(days=1))
    record('listing_strict300d', retained=ns['delNewStock'](ctx, {'equal': 10, 'older': 10}))
    for rate in (2., 3., 4., -3.):
        ns['attribute_history'] = lambda *a, r=rate, **kw: pd.DataFrame({'close': [100., 100.+r]})
        record('absolute_stoploss_'+str(rate), value=ns['stopLoss']('A'))
    daily = selected(directory, 0, ['fun_getdailyreturn'], {'history': lambda *a, **kw:
        pd.DataFrame({'A': [100., 101.]}, index=pd.to_datetime(['2020-01-03', '2020-01-06']))}, nested=True)
    failure('pandas3_fillna_method_incompatible', TypeError, lambda: daily['fun_getdailyreturn']('A', '1d', 250))
    risk = selected(directory, 0, ['calstockRiskVaR', 'calstockRiskES'], {'context': ctx, 'np': np, 'isnan': math.isnan}, nested=True)
    for case, vals in (('short_fixed_denominator', [-.1, -.05, .1]), ('constant_positive_ES', [.01]*250), ('zero_ES', [0.]*250)):
        risk['fun_getdailyreturn'] = lambda *a, v=vals: np.array(v)
        record(case, ES=risk['calstockRiskES']('A'), VaR=risk['calstockRiskVaR']('A'))
    g = SimpleNamespace(); orders = []; schedules = []
    ns2 = selected(directory, 1, ['set_params', 'initialize', 'yfdebld_signal', 'trade', 'trade_bond'],
        {'g': g, 'set_backtest': lambda: None, 'run_monthly': lambda *a: schedules.append([a[1], a[2]]),
         'monthly': lambda c: None, 'log': SimpleNamespace(info=lambda *a: None),
         'order_target_value': lambda s, v, *a: orders.append(['target', s, v]),
         'order_value': lambda s, v, *a: orders.append(['increment', s, v])})
    ns2['initialize'](None)
    record('active_indices_and_1500_schedule', codes=g.fund_list, schedule=schedules, N=g.N, ratio=g.ratio)
    for case, pair in (('equal_positive', (11., 11.)), ('winner1', (12., 11.)), ('winner2', (11., 12.)), ('both_negative', (8., 9.))):
        ns2['attribute_history'] = lambda s, *a, p=pair, **kw: {'close': np.linspace(10., p[0 if s.startswith('000016') else 1], 11)}
        record('signal_'+case, value=ns2['yfdebld_signal'](None, g.fund_list, g.N, 0))
    ns2['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.linspace(10., 12., 11)})
    failure('pandas_integer_labels_fail', KeyError, lambda: ns2['yfdebld_signal'](None, g.fund_list, g.N, 0))
    portfolio = SimpleNamespace(total_value=10000., available_cash=3000., positions={s: object() for s in g.fund_list})
    for signal in (1, 3):
        orders.clear(); ns2['trade'](SimpleNamespace(portfolio=portfolio), signal, g.fund_list, .8)
        ns2['trade_bond'](SimpleNamespace(portfolio=portfolio), g.fund_list, 1)
        record(f'orders_signal{signal}_incremental', orders=list(orders), mocked_immediate_fills=False)
    return {'cases': rows, 'count': len(rows), 'scope': 'Original AST function bodies; injected deterministic synthetic inputs, no broker/messages', 'not_a_backtest': True}


def expanded_diagnostics(directory):
    result = diagnostics(directory); rows = result['cases']
    ns = selected(directory, 0, ['caltradeStockRatio', 'tradeStockDict'],
        {'np': np, 'isnan': math.isnan, 'power': np.power, 'getStockRSI': lambda s: 20,
         'variance': lambda s: 0, 'getPremiumRate': lambda *a: 0, 'getStockName': lambda s: s, 'record': lambda **kw: None})
    stocks = list('ABCDEF'); growth = dict.fromkeys(stocks+['REF'], 0.)
    # Fluent synthetic returns explicitly exercise the original ranking body, not pandas equivalence.
    class SyntheticReturns:
        def __init__(self, value): self.values = np.array([-value]*5+[0.]*245)
        def resample(self, *a): return self
        def last(self): return self
        def pct_change(self): return self
        def fillna(self, **kw): return self
    losses = dict(zip(stocks, (.01, .02, .03, .04, .05, .06), strict=True)); losses['REF'] = .1
    ns['history'] = lambda lag, freq, field, stock, **kw: SyntheticReturns(losses[stock])
    ns['attribute_history'] = lambda stock, n, *a, **kw: {'close': np.array([100.]) if n == 21 else np.array([100.*(1+growth[stock]/100)])}
    def context():
        return SimpleNamespace(pool=list(stocks), confidenceLevel=.02, referenceCycle=250, minPosition=0,
            stockDict=dict.fromkeys(stocks, 10), transactionRecord=dict.fromkeys(stocks, 0), stockCount=5,
            referStock='REF', Transitional=True, initPriceDict=dict.fromkeys(stocks, 0), maxRange=30, consolidation=5,
            Multiple=1, trackStock='', minRatio=.1, maxRatio=.3, positionDict=dict.fromkeys(stocks, 0),
            ratioMessage='', bondStock='BOND', portfolio=SimpleNamespace(positions={}, inout_cash=0, available_cash=10000, total_value=10000))
    c = context(); ratio = ns['caltradeStockRatio'](c)
    rows.append({'case': 'ranking_adjustment_after_caps', 'ratios': ratio, 'positions': dict(c.transactionRecord)})
    old = dict(c.transactionRecord); losses['A'] = .02; c.minPosition = 0
    ns['caltradeStockRatio'](c)
    rows.append({'case': 'state_smoothing80_20', 'previous_A': old['A'], 'new_raw_A': 300., 'smoothed_A': c.transactionRecord['A']})
    losses['A'] = .01; c = context(); c.portfolio.positions['F'] = SimpleNamespace(price=10, avg_cost=10)
    rows.append({'case': 'marginal_old_holding_preserved', 'ratios': ns['caltradeStockRatio'](c), 'ratio_message': c.ratioMessage})
    c = context(); growth['A'] = 30.
    locked = ns['caltradeStockRatio'](c); lock = c.initPriceDict['A']
    growth['A'] = 5.; c.minPosition = 0; ns['caltradeStockRatio'](c); equal = c.initPriceDict['A']
    growth['A'] = 4.99; c.minPosition = 0; unlocked = ns['caltradeStockRatio'](c)
    rows.append({'case': 'growth_lock_strict_unlock', 'at30': lock, 'at5': equal, 'below5': c.initPriceDict['A'],
        'locked_ratio_A': locked['A'], 'unlocked_ratio_A': unlocked['A']})
    ns['history'] = lambda lag, freq, field, stock, **kw: {s: np.array([10.]) for s in (stock if isinstance(stock, list) else [stock])}
    orders = []; ns['order_target_value'] = lambda s, v: orders.append([s, v])
    c = SimpleNamespace(portfolio=SimpleNamespace(total_value=10000., portfolio_value=10000., positions={}),
        positionDict={'A': 0}, minAmount=2000, doneTrade=False)
    for quota in (1000., 1499., 1500., 2500.):
        orders.clear(); c.doneTrade=False; ns['tradeStockDict'](c, {'A': quota/10000})
        rows.append({'case': 'rounded_delta_'+str(int(quota)), 'orders': list(orders), 'doneTrade': c.doneTrade})
    c.portfolio.positions['A'] = SimpleNamespace(total_amount=100, avg_cost=10)
    try: ns['tradeStockDict'](c, {'A': 0})
    except TypeError as exc: rows.append({'case': 'numpy_single_array_float_incompatible', 'exception': type(exc).__name__, 'detail': str(exc)[:250]})
    else: raise AssertionError('Current NumPy no longer rejects original single-array float; re-review evidence')
    result.update(count=len(rows), expanded=True)
    return result


def expand(root, directory):
    binding(root, directory); validate_sources(directory)
    save(directory / 'diagnostics-expanded.json', expanded_diagnostics(directory))
    return archive(root, 'Batch74 original ranking/smoothing/lock/rounding and NumPy compatibility evidence expanded; initial archive retained')


def numeric_ETF(directory):
    doc = read(directory / 'reused-ETF.json'); row = doc['payload']
    require(file_sha(doc['receipt']) == doc['receipt_sha256'] and file_sha(row['path']) == row['sha256'], 'Reused ETF changed')
    f = pd.read_parquet(row['path']); dates = pd.to_datetime(f['日期']); close = f['收盘'].to_numpy(float)
    require(len(f) == 3286 and dates.is_monotonic_increasing and not dates.duplicated().any() and
        np.isfinite(close).all() and (close > 0).all(), 'ETF profile changed')
    series = pd.Series(close, index=dates); output = []
    risk = selected(directory, 0, ['calstockRiskES'], {'context': SimpleNamespace(referenceCycle=250, confidenceLevel=.02), 'isnan': math.isnan}, nested=True)
    for k in range(249, len(f)):
        window = series.iloc[k-249:k+1].resample('D').last()
        # Both interpretations are reference components; original legacy fill behavior is unproved.
        for policy in ('explicit_forward_fill', 'no_forward_fill'):
            daily = (window.ffill() if policy == 'explicit_forward_fill' else window).pct_change(fill_method=None).fillna(0).to_numpy()
            risk['fun_getdailyreturn'] = lambda *a, values=daily: values
            value = float(risk['calstockRiskES']('A'))
            manual = -math.fsum(sorted(daily)[:5])/5
            require(math.isclose(value, manual, rel_tol=0, abs_tol=1e-15), 'ES arithmetic differs')
            output.append({'date': dates.iloc[k].date(), 'policy': policy, 'calendar_rows': len(daily), 'ES': value,
                'independent_ES': manual, 'growth21_pct': (close[k]/close[k-20]-1)*100})
    out = pd.DataFrame(output)
    return out, {'candidate': '510310.XSHG', 'rows': len(f), 'first': str(dates.iloc[0].date()), 'last': str(dates.iloc[-1].date()),
        'windows_per_interpretation': len(f)-249, 'component_rows': len(out), 'ES_max_abs_difference': float((out.ES-out.independent_ES).abs().max()),
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_signal': False,
        'limits': ['Raw historical candidate only; event-adjusted original fq=pre unproved',
            'Two explicit pandas fill interpretations; no implicit economic compatibility repair',
            'Single ETF components are not16-fund ranking, weights or strategy returns']}


def study(root, directory):
    binding(root, directory); validate_sources(directory)
    require(not any((directory / n).exists() for n in ('diagnostics.json', 'ETF-reference-components.parquet', 'component-research.json')), 'Study archive exists')
    save(directory / 'diagnostics.json', diagnostics(directory))
    frame, summary = numeric_ETF(directory)
    with (directory / 'ETF-reference-components.parquet').open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory / 'component-research.json', summary)
    return archive(root, 'Batch74 original function failures/boundaries and full-history ETF ES interpretations archived; no original trades')


def numeric_indices(directory):
    fallback = directory / 'sina-fallback'; results = read(fallback / 'results.json')['results']
    cases = ('index-sh000016', 'index-sz399006'); frames = []; inputs = []; missing = []
    for case in cases:
        row = next(r for r in results if r['case'] == case)
        if row['status'] != 'success': missing.append(case); continue
        require(file_sha(row['file']) == row['sha256'], 'Index candidate changed')
        frame = pd.read_parquet(row['file']); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.is_monotonic_increasing and not frame.date.duplicated().any(), 'Index dates not ordered/unique')
        inputs.append({'case': case, 'file': row['file'], 'sha256': row['sha256'], 'rows': len(frame),
            'after_cutoff': int((frame.date > pd.Timestamp('2026-09-29')).sum())})
        frames.append(frame[frame.date <= pd.Timestamp('2026-09-29')][['date', 'close']].rename(columns={'close': case}))
    if missing:
        return pd.DataFrame(columns=['date', 'signal', 'independent_signal']), {'status': 'blocked', 'missing': missing, 'inputs': inputs, 'not_a_backtest': True}
    joined = frames[0].merge(frames[1], on='date', how='inner', validate='one_to_one').sort_values('date').reset_index(drop=True)
    ns = selected(directory, 1, ['yfdebld_signal'], {'log': SimpleNamespace(info=lambda *a: None)})
    pool = ['000016.XSHG', '000016.XSHG', '399006.XSHE', '399006.XSHE', '000012.XSHG', '000012.XSHG']
    output = []; blocked = []
    for k in range(10, len(joined)):
        ends = [f.date.searchsorted(joined.date.iloc[k], side='right') for f in frames]
        histories = [f.iloc[end-11:end] for f, end in zip(frames, ends, strict=True)]
        if any(len(h) != 11 or h.date.tolist() != joined.date.iloc[k-10:k+1].tolist() for h in histories):
            blocked.append(str(joined.date.iloc[k].date())); continue
        windows = [h[c].to_numpy(float) for h, c in zip(histories, cases, strict=True)]
        if not all(np.isfinite(w).all() and (w > 0).all() for w in windows):
            blocked.append(str(joined.date.iloc[k].date())); continue
        values = {pool[0]: windows[0], pool[2]: windows[1]}
        ns['attribute_history'] = lambda s, *a, **kw: {'close': values[s]}
        signal = ns['yfdebld_signal'](None, pool, 11, 0)
        returns = [Fraction(str(w[-1]))/Fraction(str(w[0]))-1 for w in windows]
        diff = returns[0]-returns[1]
        independent = 1 if diff > 0 and returns[0] > 0 else 2 if diff < 0 and returns[1] > 0 else 3
        output.append({'date': joined.date.iloc[k].date(), 'signal': signal, 'independent_signal': independent,
            'return50': windows[0][-1]/windows[0][0]-1, 'return_growth': windows[1][-1]/windows[1][0]-1,
            'independent_return50': float(returns[0]), 'independent_return_growth': float(returns[1])})
    out = pd.DataFrame(output, columns=['date', 'signal', 'independent_signal', 'return50', 'return_growth',
        'independent_return50', 'independent_return_growth'])
    return out, {'status': 'ok', 'inputs': inputs, 'joined_rows': len(joined), 'windows': len(out), 'blocked_windows': len(blocked),
        'blocked_dates': blocked, 'first': str(out.date.iloc[0]) if len(out) else None, 'last': str(out.date.iloc[-1]) if len(out) else None,
        'signal_differences': int((out.signal != out.independent_signal).sum()),
        'return_max_abs_difference': max(float((out[c]-out['independent_'+c]).abs().max()) for c in ('return50', 'return_growth')) if len(out) else 0.,
        'signal_counts': {str(k): int(v) for k, v in out.signal.value_counts().items()},
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_signal': False,
        'limits': ['Vendor historical index reference only; platform versions/default filling unproved',
            'Common-date complete11-row windows use explicit arrays; current pandas original label failure retained separately',
            'Daily signal coverage is not monthly15:00 trading, ETF mapping or portfolio return']}


def index_study(root, directory):
    binding(root, directory)
    require(not any((directory / n).exists() for n in ('index-reference-components.parquet', 'index-component-research.json')), 'Index component archive exists')
    frame, summary = numeric_indices(directory)
    with (directory / 'index-reference-components.parquet').open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory / 'index-component-research.json', summary)
    return archive(root, 'Batch74 all available two-eight original-index reference windows checked independently with exact rational arithmetic; no trades')


def expected_request(case):
    q = QUERIES[case]; p = q['parameters']; symbol = p['symbol']; index = case.startswith('index-')
    params = {'secid': f"{q['market']}.{symbol[2:] if index else symbol}", 'fields1': 'f1,f2,f3,f4,f5' if index else 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58' if index else 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116',
        'klt': '101', 'fqt': '0', 'beg': p['start_date'], 'end': p['end_date']}
    if not index: params['ut'] = '7eea3edcaed734bea9cbfc24409ed989'
    return 'https://push2his.eastmoney.com/api/qt/stock/kline/get', params


def invoke(case):
    import akshare as ak
    q = QUERIES[case]; fn = getattr(ak, q['function']); module = inspect.getmodule(fn)
    with ExitStack() as stack:
        if q['function'] == 'fund_etf_hist_em': stack.enter_context(patch.object(module, 'get_market_id', return_value=q['market']))
        if q['function'] == 'fund_lof_hist_em': stack.enter_context(patch.object(module, '_fund_lof_code_id_map_em', return_value={q['parameters']['symbol']: q['market']}))
        return fn(**q['parameters'])


def worker(directory, case):
    folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        endpoint, params = expected_request(case)
        require(not row['attempts'] and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == params, 'Unexpected/repeated request')
        row['attempts'].append({'method': 'GET', 'url': url, 'parameters': params})
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); require(not (directory / 'probes').exists(), 'Probe batch already started')
    rows = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch74', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'timeout_seconds': 45})
            folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s worker deadline; request completion unproved',
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / 'probes' / case / 'result.json'))
        print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    save(directory / 'probe-results.json', {'results': rows, 'not_a_backtest': True, 'published': False})
    return archive(root, 'Batch74 bounded fund/index candidate-history attempts archived; no original input release')


def parse_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API changed')
    results = read(directory / 'probe-results.json')['results']; require(len(results) == len(QUERIES) and {r['case'] for r in results} == set(QUERIES), 'Probe scope changed')
    profiles = []
    for row in results:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe binding changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial file changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            attempts = []
            def replay(session, method, url, **kwargs):
                endpoint, params = expected_request(case)
                require(not attempts and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == params, 'Unexpected replay request')
                attempts.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Probe proof changed')
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(attempts) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Parse differs')
            date_col = 'date' if 'date' in frame else '日期'; close_col = 'close' if 'close' in frame else '收盘'
            if len(frame):
                dates = pd.to_datetime(frame[date_col]); close = pd.to_numeric(frame[close_col])
                profile.update(rows=len(frame), first=str(dates.min().date()), last=str(dates.max().date()),
                    unique_dates=dates.nunique(), positive_finite_closes=int((np.isfinite(close) & (close > 0)).sum()))
            profile['native_parser_match'] = True
        else: require(row.get('error'), 'Failed request lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'new_original_input_rows': 0, 'new_trading_backtests': 0, 'not_a_backtest': True,
        'limits': ['Latest retrieved unadjusted histories do not prove original platform data/vintage/identity',
            'Fund events, start dates, daily states and fees incomplete; rotation indices not admitted for trading']}


def audit(root, directory):
    binding(root, directory); save(directory / 'input-audit.json', parse_probes(directory))
    return archive(root, 'Batch74 raw candidate coverage and original input gates audited')


def validated_final_inventory(directory, actual, before):
    scope = read(directory / 'ledger-scope-revalidation-final-amended.json')
    frozen = read(directory / 'dependency-inventory.json'); tested = directory / 'book-tested-before-whitespace.py'
    tree = ast.dump(ast.parse(Path('src/observe/ledger/book.py').read_text(encoding='utf-8')), include_attributes=False)
    tested_tree = ast.dump(ast.parse(tested.read_text(encoding='utf-8')), include_attributes=False)
    require(scope['status'] == 'ok' and scope['implementation_sha256'] == before and scope['book_ast_unchanged'] and
        scope['tested_book_sha256'] == file_sha(tested) == frozen['ledger_sha256'] and
        scope['book_ast_sha256'] == hashlib.sha256(tree.encode()).hexdigest() and tree == tested_tree and
        actual['ledger_sha256'] == before['src/observe/ledger/book.py'], 'Final inventory lacks tested Book proof')
    expected = dict(frozen); expected['ledger_sha256'] = actual['ledger_sha256']
    require(expected == actual, 'Final inventory differs beyond Book whitespace SHA')
    return expected


def offline(root, directory, *, final=False):
    from scripts import probe_strategy_batch74_fallback as fallback
    target = directory / ('offline-verification-final.json' if final else 'offline-verification.json')
    require(not target.exists(), 'Offline verification archive exists')
    if final: require(not (directory / 'dependency-inventory-final.json').exists(), 'Final inventory archive exists')
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch74 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = expanded_diagnostics(directory); dep = inventory(root); supplements = parse_probes(directory); frame, summary = numeric_ETF(directory)
        extra = fallback.verify(directory / 'sina-fallback')
        index_frame, index_summary = numeric_indices(directory)
        if final:
            validate_catalog(root, directory); cat = read(directory / 'offline-catalog.json')
        else: cat = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / 'ETF-reference-components.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(index_frame, pd.read_parquet(directory / 'index-reference-components.parquet'), check_exact=True)
    expected_dep = validated_final_inventory(directory, dep, before) if final else read(directory / 'dependency-inventory.json')
    require(before == implementation() and diag == read(directory / 'diagnostics-expanded.json') and dep == expected_dep and
        supplements == read(directory / 'input-audit.json') and summary == read(directory / 'component-research.json') and
        extra == read(directory / 'sina-fallback/audit.json') and index_summary == read(directory / 'index-component-research.json'), 'Offline evidence differs')
    if not final: save(directory / 'offline-catalog.json', cat)
    validate_catalog(root, directory)
    if final: save(directory / 'dependency-inventory-final.json', dep)
    save(target, {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True,
        'catalog_policy': 'Previously forbidden-network rebuilt byte-identical catalog revalidated' if final else 'Rebuilt forbidden-network'})
    return archive(root, 'Batch74 forbidden-network source diagnostics/ETF arithmetic/native parsers and three catalogs match0')


def offline_final(root, directory): return offline(root, directory, final=True)


def ledger_checks(root, directory):
    binding(root, directory); before = implementation()
    command = [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_external_cash_flows.py',
        *('tests/unit/'+n for n in ('test_ledger.py', 'test_conversion.py', 'test_portfolio_loop.py', 'test_offline_integration.py', 'test_dividend_entitlements.py')),
        'tests/integration/test_conversion_execution.py', 'tests/integration/test_public_run.py',
        'tests/integration/test_strategy_scope.py', 'tests/integration/test_dividend_entitlement_execution.py']
    log = directory / 'ledger-pytest.log'
    with log.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    require(result.returncode == 0 and before == implementation(), 'Ledger check failed/implementation changed')
    legacy = legacy_comparison(directory)
    reproduction = reproduce_legacy(directory)
    require(before == implementation(), 'Ledger reproduction implementation changed')
    save(directory / 'ledger-validation.json', {'status': 'ok', 'command': command, 'returncode': result.returncode,
        'log': str(log), 'sha256': file_sha(log), 'legacy_comparison': legacy, 'legacy_reproduction': reproduction,
        'implementation_sha256': before, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'scope': 'Opt-in Book flow component; existing stock three-cost regression is not a new strategy'})
    return archive(root, 'Batch74 opt-in external-flow unit NAV/atomicity and legacy three-cost forbidden-network reproduction verified')


def ledger_scope(root, directory, *, final=False, amended=False):
    binding(root, directory); ledger = read(directory / 'ledger-validation.json'); current = implementation()
    old_driver = directory / 'ledger-driver-before-archive-guard.py'
    require(file_sha(old_driver) == ledger['implementation_sha256']['scripts/review_strategy_batch74.py'], 'Tested driver snapshot differs')
    changed = sorted(name for name, sha in ledger['implementation_sha256'].items() if current[name] != sha)
    allowed = {'scripts/review_strategy_batch74.py', 'tests/unit/test_batch74_research.py'}
    extra = {}
    if final:
        tested_book = directory / 'book-tested-before-whitespace.py'
        require(file_sha(tested_book) == ledger['implementation_sha256']['src/observe/ledger/book.py'], 'Tested Book snapshot differs')
        old_tree = ast.dump(ast.parse(tested_book.read_text(encoding='utf-8')), include_attributes=False)
        new_tree = ast.dump(ast.parse(Path('src/observe/ledger/book.py').read_text(encoding='utf-8')), include_attributes=False)
        require(old_tree == new_tree, 'Tested Book semantics changed')
        allowed.add('src/observe/ledger/book.py')
        extra = {'tested_book_sha256': file_sha(tested_book), 'book_ast_unchanged': True,
            'book_ast_sha256': hashlib.sha256(new_tree.encode()).hexdigest()}
    require(set(current) == set(ledger['implementation_sha256']) and set(changed) <= allowed, 'Tested ledger/runtime changed')
    def function_body(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ledger_checks']
        require(len(functions) == 1, 'Ledger verification body missing')
        return ast.dump(functions[0], include_attributes=False)
    require(function_body(old_driver) == function_body(Path(__file__)), 'Ledger test/reproduction workflow changed')
    require(ledger['status'] == 'ok' and file_sha(ledger['log']) == ledger['sha256'] and
        ledger['legacy_reproduction']['result']['reproduction'] == read(directory / 'legacy-strategy-reproduction.json')['result']['reproduction'],
        'Original ledger verification proof changed')
    target = 'ledger-scope-revalidation-final-amended.json' if amended else 'ledger-scope-revalidation-final.json' if final else 'ledger-scope-revalidation.json'
    save(directory / target, {'status': 'ok', 'ledger_proof_sha256': file_sha(directory / 'ledger-validation.json'),
        'tested_driver_sha256': file_sha(old_driver), 'ledger_checks_ast_unchanged': True, 'changed_research_files': changed,
        'implementation_sha256': current, **extra,
        'scope': 'Research archive guard/test and, in final scope, one Book CRLF blank line changed after148 tests/reproduction; tested Book AST and ledger test function identical'})
    return {'status': 'ok', 'changed_research_files': changed}


def ledger_scope_final(root, directory): return ledger_scope(root, directory, final=True, amended=True)


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch74.py',
        'scripts/probe_strategy_batch74_fallback.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch74_research.py', 'tests/unit/test_batch73_research.py',
            'tests/unit/test_external_cash_flows.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    folder = directory / 'final-checks'; folder.mkdir(exist_ok=False)
    for k, command in enumerate(commands()):
        path = folder / f'check-{k}.log'
        with path.open('x') as stream: process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': process.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(process.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification-final.json'); ledger = read(directory / 'ledger-validation.json')
    scope = read(directory / 'ledger-scope-revalidation-final-amended.json'); current = implementation()
    book_tree = ast.dump(ast.parse(Path('src/observe/ledger/book.py').read_text(encoding='utf-8')), include_attributes=False)
    require(scope['status'] == 'ok' and scope['ledger_proof_sha256'] == file_sha(directory / 'ledger-validation.json') and
        scope['tested_driver_sha256'] == file_sha(directory / 'ledger-driver-before-archive-guard.py') and scope['ledger_checks_ast_unchanged'] and
        scope['tested_book_sha256'] == file_sha(directory / 'book-tested-before-whitespace.py') and scope['book_ast_unchanged'] and
        scope['book_ast_sha256'] == hashlib.sha256(book_tree.encode()).hexdigest() and
        set(scope['changed_research_files']) <= {'scripts/review_strategy_batch74.py', 'tests/unit/test_batch74_research.py', 'src/observe/ledger/book.py'}, 'Ledger scope proof differs')
    require(all(current[name] == sha for name, sha in ledger['implementation_sha256'].items() if name not in scope['changed_research_files']), 'Tested ledger/runtime changed')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == current == off['implementation_sha256'] == scope['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and ledger['status'] == 'ok' and
        ledger['legacy_reproduction']['result']['reproduction']['result'] == 'match' and
        Store(root).published() == read(directory / 'baseline.json')['published'], 'Offline/ledger/publication changed')
    protection = protect(root, directory)
    progress = archive(root, 'Batch74 FOF/two-eight source research and opt-in external-flow ledger accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch74-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'ledger': ledger, 'ledger_scope': scope, 'diagnostics': read(directory / 'diagnostics-expanded.json'),
        'inventory': read(directory / 'dependency-inventory-final.json'),
        'component': read(directory / 'component-research.json'), 'index_component': read(directory / 'index-component-research.json'), 'primary': read(directory / 'probe-results.json'),
        'primary_audit': read(directory / 'input-audit.json'), 'fallback': read(directory / 'sina-fallback/audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'new_trading_backtests': 0, 'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'study', 'expand', 'index_study', 'worker', 'probe', 'audit', 'offline', 'offline_final',
        'ledger_checks', 'ledger_scope', 'ledger_scope_final', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
