"""Freeze EBIT/EV and market-PE rules, diagnostics and bounded input evidence."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import importlib.util
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import socket
import statistics
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
from scripts import review_strategy_batch72 as previous
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch73/20261008-ebit-market-PE')
SNAPSHOT = previous.SNAPSHOT
ACCEPTED = previous.DIRECTORY.with_name(previous.DIRECTORY.name+'-review-correction')
RECEIPT = Path('docs/handoff/2026-10-08-batch72-verification.json')
OLD_PE = Path('data/staging/strategies-batch12/20261005-daily-candidates')
SOURCES = ('2021年度精选策略/77.超强单因子策略（EBIT,EV）.txt',
    '2021年度精选策略/53.基于大盘PE标准差偏离度的聪明基金定投策略.txt')
SOURCE_SHA = ('77fc7abe51eadecde96db59ea4d4043e126815c61e1d6b2f8c3033bfc81e98be',
    '4b49483b6ffd92e9596da5a8eeb4faf836bf8df029798fb644848783d1bc1300')
REFERENCES = (('repo/akshare', 'akshare/stock_feature/stock_a_pe_and_pb.py'),
    ('repo/akshare', 'akshare/stock/stock_summary.py'),
    ('repo/rqalpha', 'rqalpha/portfolio/__init__.py'), ('repo/rqalpha', 'rqalpha/portfolio/account.py'))
QUERIES = {**{f'sse-{day}': {'function': 'stock_sse_deal_daily', 'parameters': {'date': day}}
    for day in ('20050105', '20200102', '20260929')},
    'szse-20260929': {'function': 'stock_szse_summary', 'parameters': {'date': '20260929'}}}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch73.py', 'tests/unit/test_batch73_research.py'}))
CORE = {'baseline.json', 'preflight.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'dependency-inventory.json', 'existing-apis.json', 'diagnostics.json',
    'reused-market-PE.json', 'supplier-market-PE-components.parquet', 'component-research.json',
    'probe-results.json', 'market-input-audit.json', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and
        file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Accepted batch72 changed')
    previous.binding(root, ACCEPTED)
    result = {'snapshot': SNAPSHOT, 'accepted_directory': str(ACCEPTED), 'receipt_file': str(RECEIPT),
        'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'checked_state_sha256': file_sha(ACCEPTED / 'checked-state.json'), 'not_a_backtest': True}
    if directory is not None:
        require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    return result


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def safe_function(node):
    return isinstance(node, ast.FunctionDef) and not node.decorator_list and not node.returns and not node.type_params and \
        all(a.annotation is None for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)) and \
        all(a is None or a.annotation is None for a in (node.args.vararg, node.args.kwarg)) and \
        all(isinstance(d, ast.Constant) for d in node.args.defaults) and \
        all(d is None or isinstance(d, ast.Constant) for d in node.args.kw_defaults)


def selected(directory, number, names, namespace):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names), 'Missing/duplicate selected definitions')
    for node in nodes:
        if isinstance(node, ast.FunctionDef): require(safe_function(node), 'Unsafe selected function')
        else:
            require(not node.bases and not node.keywords and not node.decorator_list and not node.type_params and
                all(safe_function(n) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and
                    isinstance(n.value.value, str)) for n in node.body), 'Unsafe selected class')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch73-definitions>', 'exec'), namespace)
    return namespace


def api_evidence():
    import akshare as ak
    code = {}
    for name in ('stock_market_pe_lg', 'stock_sse_deal_daily', 'stock_szse_summary'):
        fn = getattr(ak, name); source = inspect.getsource(fn)
        code[name] = {'signature': str(inspect.signature(fn)), 'source': source,
            'sha256': hashlib.sha256(source.encode()).hexdigest()}
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'functions': code,
        'jqdatasdk_installed': importlib.util.find_spec('jqdatasdk') is not None,
        'policy': 'Reuse old Legulegu responses by SHA; four exchange requests, each once, no redirects/retries/proxy,8/10s,parent45s; no headers archived',
        'limits': ['Exchange summaries do not prove JoinQuant code322002 mapping/PE method or historical availability',
            'Legulegu sparse observations are not original daily records; no interpolation or index-PE substitution']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); fields = store.load_state(state, 'financial_fields')
    candidates = fields[fields.astype(str).apply(lambda s: s.str.contains('EBIT|金融资产|金融负债|息税|市值', regex=True)).any(axis=1)]
    columns = ['table', 'column', 'standard_name', 'original_label', 'unit', 'joinquant_candidate', 'joinquant_equivalence', 'time_evidence']
    ledger = Path('src/observe/ledger/book.py'); tree = ast.parse(ledger.read_text())
    methods = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'financial_field_rows': len(fields), 'reference_field_candidates': candidates[columns].fillna('').to_dict('records'),
        'strict_original_factor_admitted_rows': 0, 'original_market_PE_admitted_rows': 0,
        'ledger_sha256': file_sha(ledger), 'ledger_methods': methods, 'not_a_backtest': True,
        'gates': ['EBIT/EV requires exact four factor definitions/units/vintages and previous-day SSE+SZSE index membership',
            'Annual market values and individual balance fields are not original daily factors; strict financial rows remain0',
            'Code322002 identity/pe_average definition and daily original vintages unproved; historical sparse Legulegu records are reference only',
            '510300 ETF event/state/reference-price coverage and original callbacks/fund fees remain unproved',
            'Book has no public external-flow/unit-NAV path; monthly5000 cannot be dropped or counted as profit',
            'Initialize does not schedule53; initial after_code_changed invocation and label0 query result must be proved']}


def start(root, directory):
    bound = binding(root); require(read(directory / 'preflight.json')['receipt_sha256'] == bound['receipt_sha256'], 'Preflight changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    save(directory / 'input-binding.json', bound)
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []; refs = []
    for i, name in enumerate(SOURCES):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == SOURCE_SHA[i], 'Source reviewed/changed')
        shutil.copyfile(path, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': SOURCE_SHA[i],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'original_line_numbers_retained': True, 'review': REVIEWS[name], 'newly_reviewed': True})
    (directory / 'references').mkdir()
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = directory / 'references' / (path.parent.name+'-'+path.name)
        shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Market summary parsing and external-flow unit-NAV reference; no second ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current} == set(old) and
        {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder / 'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch73 full EBIT_EV and market PE sources reviewed; original factor/PE/ETF/flow gates retained')


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


def diagnostics(directory):
    validate_sources(directory); rows = []; orders = []; scheduled = []; calls = []; flows = []; costs = []
    def record(case, **values): rows.append({'case': case, **values})
    def failure(case, error, fn):
        try: fn()
        except error as exc: record(case, error=type(exc).__name__, orders=orders.copy())
        else: raise ValueError('Expected original failure: '+case)
    log = SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)
    context = SimpleNamespace(previous_date=dt.date(2023, 12, 29), current_dt=dt.datetime(2024, 1, 2, 9, 30),
        portfolio=SimpleNamespace(positions={}, total_value=100000., positions_value=30000., available_cash=70000.))
    schedule = lambda mode: lambda fn, *a, **kw: scheduled.append([mode, fn.__name__, list(a), kw])
    common = {'np': np, 'pd': pd, 'log': log, 'query': previous.Query,
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None,
        'set_order_cost': lambda value, **kw: costs.append([value, kw]), 'OrderCost': lambda **kw: kw,
        'run_monthly': schedule('monthly'), 'run_daily': schedule('daily'), 'run_weekly': schedule('weekly')}
    g = SimpleNamespace(); table = SimpleNamespace(code=previous.Field('code'), name=previous.Field('name'), start_date=previous.Field('start_date'))
    ns = selected(directory, 0, ['initialize', 'get_EBIT_EV', 'before_market_open', 'market_open', 'after_market_close'],
        {**common, 'g': g, 'finance': SimpleNamespace(STK_LIST=table),
            'order_target_value': lambda i, v: orders.append(['target_value', i, v]),
            'order_target': lambda i, v: orders.append(['target', i, v])})
    ns['initialize'](context); record('ebit_monthly_schedule_and_costs', scheduled=scheduled.copy(), costs=costs.copy(), months=g.Transfer_date)
    stock_data = pd.DataFrame({'code': ['A', 'B', 'C', 'D', 'E', 'F'], 'name': ['ST-A', 'B', 'C', 'D', 'E', 'F'],
        'start_date': [dt.date(2024, 1, 2)]*6})
    ns['finance'].run_query = lambda q: calls.append({'fields': q.fields, 'filters': q.filters}) or stock_data.copy()
    factor_rows = {'market_cap': [10., 1., 1., 0., 1., np.nan], 'financial_liability': [0.]*6,
        'financial_assets': [0., 2., 1., 0., 0., 0.], 'EBIT': [1., -2., 2., 0., -1., 1.]}
    def factors(**kwargs):
        calls.append(kwargs); return {name: pd.DataFrame([values], columns=list('ABCDEF')) for name, values in factor_rows.items()}
    ns['get_factor_values'] = factors
    data = ns['get_EBIT_EV'](list('ABCDEF'), context.previous_date)
    record('ebit_negative_EV_negative_EBIT_and_infinite_rank', rank=data.index.tolist(),
        positive_infinity_first=bool(np.isposinf(data.iloc[0]['EBIT/EV'])), negative_EV_ratio=float(data.loc['B', 'EBIT/EV']),
        missing_codes=data.index[data['EBIT/EV'].isna()].tolist(), query=calls.copy(), ST_new_listing_retained='A' in data.index)
    ns['get_factor_values'] = lambda **kw: {'market_cap': pd.DataFrame([[1]], columns=['A'])}
    failure('ebit_missing_factor_KeyError', KeyError, lambda: ns['get_EBIT_EV'](['A'], context.previous_date))
    ns['get_index_stocks'] = lambda code, **kw: calls.append({'index': code, **kw}) or (['A', 'B'] if code == '000001.XSHG' else ['B', 'C'])
    ns['get_EBIT_EV'] = lambda **kw: calls.append(kw) or pd.DataFrame(index=['B', 'B', 'A', 'C'])
    g.entry_dates = {'A': 'old'}; calls.clear(); ns['before_market_open'](context)
    record('ebit_index_concat_duplicate_targets_reset_entry_dates', security=g.security, buy_list=g.buy_list,
        entry_dates=g.entry_dates.copy(), calls=calls.copy())
    current = {c: SimpleNamespace(paused=c in ('B', 'Y', 'Z'), day_open=30.) for c in ('A', 'B', 'C', 'X', 'Y', 'Z')}
    ns['get_current_data'] = lambda: current; p = context.portfolio
    g.buy_list = ['A', 'B', 'C', 'Z']; g.entry_dates = {c: None for c in g.buy_list}; p.positions = {'B': object(), 'X': object(), 'Y': object()}
    orders.clear(); ns['market_open'](context)
    record('ebit_held_paused_target_and_unheld_paused_skipped', orders=orders.copy(), entry_dates=g.entry_dates.copy())
    g.buy_list = ['A', 'A']; p.positions = {}; orders.clear(); ns['market_open'](context)
    record('ebit_duplicate_unfilled_orders_and_entry_registration', orders=orders.copy(), entry_date=g.entry_dates['A'])
    g.buy_list = []; p.positions = {'X': object(), 'Y': object()}; orders.clear(); ns['market_open'](context)
    record('ebit_empty_list_sells_only_unpaused', orders=orders.copy())
    g.buy_list = ['A']; p.positions = {}; p.total_value = 1.; orders.clear(); ns['market_open'](context)
    record('ebit_zero_lot_still_registers_entry', orders=orders.copy(), entry_date=g.entry_dates['A'])
    p.total_value = 100000.; current['A'].day_open = 0.; orders.clear()
    failure('ebit_zero_open_ZeroDivisionError', ZeroDivisionError, lambda: ns['market_open'](context))
    scheduled.clear(); costs.clear(); g = SimpleNamespace()
    table = SimpleNamespace(name='STK_EXCHANGE_TRADE_INFO', date=previous.Field('date'), exchange_code=previous.Field('exchange_code'))
    data_holder = {'frame': pd.DataFrame({'date': [dt.date(2023, 12, 29), dt.date(2023, 12, 28)], 'pe_average': [10., 100.], 'turnover_ratio': [.1, .2]})}
    def query(q): calls.append({'fields': q.fields, 'filters': q.filters, 'sort': q.sort, 'count': q.count}); return data_holder['frame'].head(q.count).copy()
    pe = selected(directory, 1, ['initialize', 'after_code_changed', 'before_market_open', 'market_open', 'after_market_close',
        'code_run_weekly', 'wages_monthily', 'QuantTiming'], {**common, 'g': g, 'finance': SimpleNamespace(STK_EXCHANGE_TRADE_INFO=table, run_query=query),
        'FixedSlippage': lambda v: v, 'set_slippage': lambda v: calls.append({'slippage': v}),
        'inout_cash': lambda value, **kw: flows.append([value, kw]), 'get_trades': lambda: {}, 'record': lambda **kw: None,
        'order_target_value': lambda i, v: orders.append(['target_value', i, v])})
    pe['initialize'](context); record('PE_initialize_no_schedule_no_security_weeks', scheduled=scheduled.copy(),
        has_security=hasattr(g, 'security'), has_weeks=hasattr(g, 'weeks'), has_stdH=hasattr(g, 'stdH'), costs=costs.copy())
    orders.clear(); failure('PE_trade_before_change_AttributeError', AttributeError, lambda: pe['market_open'](context))
    pe['after_code_changed'](context); record('PE_after_change_five_callbacks', scheduled=scheduled.copy(), security=g.security, weeks=g.weeks)
    calls.clear(); qtt = pe['QuantTiming'](context.previous_date, g.security); values = qtt.Q(2500); high, low = qtt.std()
    record('PE_query_code322002_desc2500_log10_sample_std', query=calls[0], V=values.V.tolist(), high=float(high), low=float(low), now=float(qtt.now()))
    data_holder['frame'].index = [9, 8]; failure('PE_now_label0_KeyError', KeyError, qtt.now)
    data_holder['frame'] = pd.DataFrame({'date': [context.previous_date]*3, 'pe_average': [0., -1., np.nan], 'turnover_ratio': [1.]*3})
    with np.errstate(divide='ignore', invalid='ignore'): values = qtt.Q(2500)
    record('PE_nonpositive_missing_log10_unprotected', negative_infinity=bool(np.isneginf(values.V.iloc[0])), nan_values=int(values.V.isna().sum()))
    data_holder['frame'] = pd.DataFrame({'date': [context.previous_date], 'pe_average': [10.], 'turnover_ratio': [1.]})
    high, low = qtt.std(); record('PE_single_row_sample_std_nan', high_is_nan=bool(pd.isna(high)), low_is_nan=bool(pd.isna(low)))
    cases = [('high_equality', 1.5, 1.5, 1.2), ('low_equality', 1.2, 1.5, 1.2),
        ('above_high', 1.65, 1.5, 1.2), ('below_low', 1.08, 1.5, 1.2),
        ('deep_low_clamp1', .6, 1.5, 1.2), ('extreme_high_negative_target', 4., 1.5, 1.2),
        ('NaN_normal50', float('nan'), 1.5, 1.2)]
    for case, now, high, low in cases:
        g.now_pe, g.stdH, g.stdL = now, high, low
        weeks = qtt.how_many_weeks(); record('PE_weeks_'+case, weeks=float(weeks))
        pe['QuantTiming'] = lambda *a, _now=now: SimpleNamespace(now=lambda: _now, how_many_weeks=lambda: weeks)
        orders.clear(); pe['market_open'](context); record('PE_target_'+case, orders=orders.copy())
    g.now_pe, g.stdH = 1., 0.
    failure('PE_zero_high_ZeroDivisionError', ZeroDivisionError, qtt.how_many_weeks)
    g.now_pe, g.stdH, g.stdL = -1., 1., 0.
    failure('PE_zero_low_ZeroDivisionError', ZeroDivisionError, qtt.how_many_weeks)
    pe['wages_monthily'](context); record('PE_monthly_5000_external_flow', flows=flows)
    result = {'cases': rows, 'synthetic_fixture': True, 'not_a_backtest': True, 'platform_equivalent': False,
        'real_historical_trade_windows': 0, 'new_trading_backtests': 0}
    return json.loads(json.dumps(result, default=str, allow_nan=False))


def reused_binding(directory):
    receipt = read('docs/handoff/2026-10-05-batch12-verification.json'); records = []
    for case in ('market_pe_sh', 'market_pe_sz'):
        original = OLD_PE / 'probes' / case / 'result.json'; row = read(original)
        accepted = next(r['result'] for r in receipt['probes']['results'] if r['endpoint'] == case)
        require(row == accepted and row['status'] == 'success' and not row['strict_usable'] and not row['published'], 'Old PE proof differs')
        require(file_sha(row['raw_file']) == row['raw_sha256'] and
            all(file_sha(r['file']) == r['sha256'] for r in row['wire_responses']), 'Old PE bytes changed')
        copied = directory / f'{case}.parquet'
        records.append({'case': case, 'result_file': str(original), 'result_sha256': file_sha(original),
            'raw_file': row['raw_file'], 'raw_sha256': row['raw_sha256'], 'copy': str(copied),
            'wire_responses': [{'file': r['file'], 'sha256': r['sha256'], 'status_code': r['status_code']} for r in row['wire_responses']],
            'rows': row['rows'], 'strict_usable': False})
    return {'receipt_file': 'docs/handoff/2026-10-05-batch12-verification.json',
        'receipt_sha256': file_sha('docs/handoff/2026-10-05-batch12-verification.json'), 'records': records,
        'not_a_backtest': True, 'no_new_download': True}


def numeric_PE(directory):
    bound = reused_binding(directory); require(bound == read(directory / 'reused-market-PE.json'), 'Reuse binding changed')
    ns = selected(directory, 1, ['QuantTiming'], {'np': np}); frames = []; profiles = []; maximum = 0.
    for row in bound['records']:
        require(file_sha(row['copy']) == row['raw_sha256'], 'PE copy changed')
        frame = pd.read_parquet(row['copy']).copy(); require(len(frame) == row['rows'], 'PE rows changed')
        frame['date'] = pd.to_datetime(frame['日期']); frame['pe_average'] = pd.to_numeric(frame['平均市盈率'], errors='raise')
        require(frame.date.notna().all() and not frame.date.duplicated().any(), 'Unknown/duplicate PE dates')
        beyond = int(frame.date.gt('2026-09-29').sum()); frame = frame[frame.date.le('2026-09-29')].sort_values('date').reset_index(drop=True)
        pe = frame.pe_average.to_numpy(float); require(np.isfinite(pe).all() and (pe > 0).all(), 'Invalid reference PE')
        class SavedTiming(ns['QuantTiming']):
            def Q(self, count):
                data = frame[frame.date.le(self.date)].tail(count).iloc[::-1].copy()
                data['V'] = data.pe_average.map(np.log10); return data
        records = []
        for day, value in zip(frame.date, pe, strict=True):
            qtt = SavedTiming(day, 'reference-only'); sample = [math.log10(float(v)) for v in pe[:len(records)+1][-2500:]]
            hi, lo = qtt.std(); median = statistics.median(sample); std = statistics.stdev(sample) if len(sample) > 1 else float('nan')
            expected = (median+.05*std, median-.25*std)
            for actual, independent in zip((hi, lo), expected, strict=True):
                require((pd.isna(actual) and math.isnan(independent)) or math.isclose(actual, independent, rel_tol=1e-12, abs_tol=1e-12), 'Independent threshold differs')
                if math.isfinite(actual): maximum = max(maximum, abs(float(actual)-independent))
            records.append({'case': row['case'], 'date': day, 'pe_average': float(value), 'V': math.log10(float(value)),
                'observations': len(sample), 'stdH': float(hi), 'stdL': float(lo)})
        frames.append(pd.DataFrame(records)); profiles.append({'case': row['case'], 'stored_rows': row['rows'], 'used_rows': len(frame),
            'after_cutoff': beyond, 'first': str(frame.date.min().date()), 'last': str(frame.date.max().date()),
            'distinct_months': int(frame.date.dt.to_period('M').nunique()), 'max_gap_calendar_days': int(frame.date.diff().dt.days.max()),
            'original_daily_series': False})
    result = pd.concat(frames, ignore_index=True); result['date'] = result.date.astype('datetime64[ms]')
    return result, {'profiles': profiles, 'rows': len(result), 'max_independent_threshold_difference': maximum,
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_signal_reproduced': False,
        'limits': ['Sparse vendor observations, not daily code322002; no fill or frequency substitution',
            'Original std formula checked at each saved observation; weekly carry/trades not reconstructed',
            'Class Q replaced by explicit saved reference rows, not a JoinQuant query reproduction; now label0 unchanged in diagnostics']}


def study(root, directory):
    binding(root, directory)
    require(not any((directory / name).exists() for name in ('diagnostics.json', 'reused-market-PE.json',
        'market_pe_sh.parquet', 'market_pe_sz.parquet', 'supplier-market-PE-components.parquet', 'component-research.json')), 'Study outputs exist')
    diag = diagnostics(directory); reused = reused_binding(directory)
    for row in reused['records']: shutil.copyfile(row['raw_file'], row['copy'])
    save(directory / 'reused-market-PE.json', reused); frame, summary = numeric_PE(directory)
    frame.to_parquet(directory / 'supplier-market-PE-components.parquet', index=False)
    save(directory / 'diagnostics.json', diag); save(directory / 'component-research.json', summary)
    return archive(root, 'Batch73 original EBIT_EV/PE diagnostics and all saved sparse PE formulas archived; no trades')


def expected_request(case):
    date = QUERIES[case]['parameters']['date']; day = '-'.join([date[:4], date[4:6], date[6:]])
    if case.startswith('sse-'):
        return 'https://query.sse.com.cn/commonQuery.do', {'sqlId': 'COMMON_SSE_SJ_GPSJ_CJGK_MRGK_C',
            'PRODUCT_CODE': '01,02,03,11,17', 'type': 'inParams', 'SEARCH_DATE': day}
    return 'http://www.szse.cn/api/report/ShowReport', {'SHOWTYPE': 'xlsx', 'CATALOGID': '1803_sczm',
        'TABKEY': 'tab1', 'txtQueryDate': day, 'random': '0.39339437497296137'}


def worker(directory, case):
    folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=False); query = QUERIES[case]
    row = {'case': case, 'query': query, 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        endpoint, parameters = expected_request(case)
        require(not row['attempts'] and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == parameters, 'Unexpected/repeated request')
        row['attempts'].append({'method': 'GET', 'url': url, 'parameters': parameters})
        session.trust_env = False; kwargs['timeout'] = (8, 10); kwargs['allow_redirects'] = False
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        import akshare as ak
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr):
            frame = getattr(ak, query['function'])(**query['parameters'])
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); require(not (directory / 'probes').exists() and not (directory / 'probe-results.json').exists(), 'Probes already attempted')
    rows = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch73', 'worker', '--directory', str(directory), '--case', case]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'timeout_seconds': 45})
            folder = directory / 'probes' / case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s worker deadline, request completion unproved',
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / 'probes' / case / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'not_a_backtest': True, 'published': False})
    return archive(root, 'Batch73 four bounded exchange-history input requests retained; no original PE substitution')


def parse_probes(directory):
    import akshare as ak
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed API changed')
    results = read(directory / 'probe-results.json')['results']; require(len(results) == len(QUERIES) and
        {r['case'] for r in results} == set(QUERIES), 'Probe scope changed')
    profiles = []
    for row in results:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe binding changed')
        for response in row['wire_responses']: require(file_sha(response['file']) == response['sha256'], 'Wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial file changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Probe proof changed')
            attempts = []
            def replay(session, method, url, **kwargs):
                endpoint, parameters = expected_request(case)
                require(not attempts and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == parameters, 'Unexpected replay request')
                attempts.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()):
                frame = getattr(ak, QUERIES[case]['function'])(**QUERIES[case]['parameters'])
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(attempts) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Parse scope differs')
            profile.update(rows=len(frame), columns=list(frame), native_parser_match=True)
        else: require(row.get('error'), 'Failed request lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'new_original_input_rows': 0, 'new_trading_backtests': 0, 'not_a_backtest': True,
        'limits': ['Exchange point samples cannot establish code322002 daily factor/method/vintage equivalence',
            'Raw SSE dates retained; native parser drops report-date row so requested date is not independently verified',
            'SZSE summary parser has no average-PE field; original query/factor/ETF/inflow gates unchanged']}


def audit(root, directory):
    binding(root, directory); save(directory / 'market-input-audit.json', parse_probes(directory))
    return archive(root, 'Batch73 native exchange parsers and failures audited; original daily PE/factor/ETF inputs not admitted')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch73 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); dep = inventory(root); supplements = parse_probes(directory); frame, summary = numeric_PE(directory)
        cat = offline_catalog(root, directory)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory / 'supplier-market-PE-components.parquet'), check_exact=True)
    require(before == implementation() and diag == read(directory / 'diagnostics.json') and dep == read(directory / 'dependency-inventory.json') and
        supplements == read(directory / 'market-input-audit.json') and summary == read(directory / 'component-research.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', cat); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch73 forbidden-network original diagnostics/sparse PE/native exchange parses and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch73.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch73_research.py', 'tests/unit/test_batch72_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
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
    protection = protect(root, directory)
    progress = archive(root, 'Batch73 EBIT_EV/market PE source and sparse-reference formula research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-08-batch73-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory / 'diagnostics.json'), 'component': read(directory / 'component-research.json'),
        'supplements': read(directory / 'probe-results.json'), 'input_audit': read(directory / 'market-input-audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'original_strategy_complete': False, 'formal_strategy_acceptance': False})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'study', 'worker', 'probe', 'audit', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
