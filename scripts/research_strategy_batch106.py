"""Frozen ETF85/VA69 source and native-operand study; never an ETF trading engine."""
import argparse
import ast
from contextlib import contextmanager, ExitStack, redirect_stdout
import datetime as dt
from decimal import Decimal
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import duckdb
import numpy as np
import pandas as pd
import requests

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch106/20261011-monthly-slots-value-average-v1')
PARENT = Path('docs/handoff/2026-10-11-batch105-final-review.json')
PARENT_SHA = '6be36344b6ebffd8257fb732d622d2077efbc51a7c7956bdd34a76877859d793'
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
SOURCES = ('聚宽2025年精选/85稳健型ETF策略.txt',
    '2024年度精选策略2/69.场内基金定投价值平均增强策略，年化30%+.txt')
SOURCE_SHA = ('dcae7848a99b7cdffb7a2d99564da319bbe662f0328c5424e7f1e3852b275876',
    '8af59e10a1c1c8810f91bdeb2d76cdc017e8812c058f0fb324aea47cc9d444ee')
REGISTRIES = (
    ('data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight/probe-results.json',
        ('fund-510880', 'fund-159905', 'fund-159949', 'fund-511880', 'index-sz399324')),
    ('data/staging/strategies-batch77/20261009-DCA-index-valuation/probe-results.json', ('fund-511010',)),
    ('data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probe-results.json', ('fund-513100',)),
    ('data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/results.json', ('fund-518880',)))
FILES = ('scripts/research_strategy_batch106.py', 'tests/unit/test_batch106_research.py')
OUTPUTS = ('diagnostics.json', 'value-average-operands.parquet', 'annual-return-operands.parquet', 'summary.json')


def literal(number, namespace=None, directory=DIRECTORY):
    require(number in (0, 1), 'Unknown Batch106 source')
    source = directory/'sources'/f'{number}.source'
    require(file_sha(source) == SOURCE_SHA[number], 'Original source identity changed')
    tree = ast.parse(read_source(source)[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (5 if number == 0 else 12), 'Original definition count changed')
    for n in nodes:
        require(not n.decorator_list and not n.returns and not n.type_params and
            all(a.annotation is None for a in (*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs)) and
            all(isinstance(d, ast.Constant) for d in n.args.defaults) and
            all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults), 'Unsafe definition')
    ns = {'pd': pd, 'np': np, 'g': SimpleNamespace(), 'dt': dt,
        'log': SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None)}
    ns.update(namespace or {})
    # The platform imports, SMTP, record/send_message and top-level code are never run.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def initialize(number, directory=DIRECTORY):
    calls = []
    ns = literal(number, {'set_option': lambda *a: calls.append(['option', list(a)]),
        'set_benchmark': lambda *a: calls.append(['benchmark', list(a)]),
        'set_slippage': lambda *a: calls.append(['slippage', list(a)]), 'FixedSlippage': lambda v: v,
        'OrderCost': lambda **k: k, 'set_order_cost': lambda *a, **k: calls.append(['cost', list(a), k]),
        'run_daily': lambda f, **k: calls.append(['daily', f.__name__, k]),
        'run_monthly': lambda f, *a, **k: calls.append(['monthly', f.__name__, list(a), k])}, directory)
    context = SimpleNamespace(current_dt=dt.datetime(2022,1,4),
        portfolio=SimpleNamespace(available_cash=100000., cash=100000., total_value=100000., positions={}))
    with redirect_stdout(StringIO()): ns['initialize'](context)
    return ns, context, calls


def slot_case(signals=(0, 0, 0, 0), zf=0, *, held=False, reject=False, cash_after_sale=20000., missing_math=False, directory=DIRECTORY):
    ns, context, schedule = initialize(0, directory); orders = []; queries = []
    funds = ('159949.XSHE', '513100.XSHG', '510880.XSHG', '511010.XSHG', '518880.XSHG', '511880.XSHG')
    ns.update(year_start=lambda c: 0, get_zf=lambda c: zf,
        get_macd_M=lambda symbol, date: dict(zip(('000300.XSHG', '513100.XSHG', '159949.XSHE', '518880.XSHG'), signals, strict=True))[symbol],
        get_current_data=lambda: dict.fromkeys(funds, SimpleNamespace(last_price=10.)))
    if not missing_math: ns['math'] = math
    if held:
        context.portfolio.total_value = 120000.; context.portfolio.cash = context.portfolio.available_cash = 1000.
        ns['g'].stock_fund = ['159949.XSHE', '510880.XSHG', '518880.XSHG', '511010.XSHG', '511880.XSHG']
        ns['g'].total_amount = [100, 200, 300, 400, 500]
    def order(symbol, amount):
        orders.append([symbol, amount])
        # Explicit externally injected feedback, never a second cash/fill implementation.
        if amount < 0 and not reject: context.portfolio.cash = cash_after_sale
        queries.append({'symbol': symbol, 'amount': amount, 'injected_cash': context.portfolio.cash})
        return None
    ns['order'] = order
    ns['on_start'](context)
    return {'stocks': ns['g'].stock_fund, 'slot_amounts': ns['g'].total_amount, 'orders': orders,
        'feedback': queries, 'schedule': schedule, 'orders_are_intents_only': True}


def value_case(name, directory=DIRECTORY):
    ns, context, schedule = initialize(1, directory); orders = []; g = ns['g']
    g.init_status = True; g.sold_price = 10000.; g.bought_price = 0.
    context.portfolio.positions[g.security] = SimpleNamespace(total_amount=1000, value=20000., price=20., avg_cost=10.)
    ns.update(get_current_data=lambda: {g.security: SimpleNamespace(last_price=20.)},
        order_value=lambda s, v: orders.append([s, v]), refresh_trade_price=lambda *a: None,
        max_increase=lambda d: (90., .2), max_increase1y=lambda d: 0.)
    context.previous_date = dt.date(2022, 1, 3); context.current_dt = dt.datetime(2022, 1, 4)
    if name == 'initial': return {'security': g.security, 'index': g.index, 'money': g.money, 'schedule': schedule}
    if name == 'initial_70_percent':
        g.init_status = False; ns['init_position'](context)
    elif name == 'half_reentry':
        g.take_profit_days = 1; ns['init_position'](context)
    elif name in ('profit_equal40', 'profit_equal9', 'profit_exact90', 'profit_over90'):
        g.r, g.down = (40., .091) if name == 'profit_equal40' else (40.01, .09) if name == 'profit_equal9' else (40.01, .091)
        if name == 'profit_over90': ns['max_increase'] = lambda d: (90.01, .2)
        ns['take_profit'](context)
    elif name.startswith('yearly_'):
        ns['max_increase1y'] = lambda d: 75. if name == 'yearly_equal75' else 75.01
        ns['take_profit'](context)
    elif name == 'cash_gt1000':
        g.money = 1000.; g.last_month_value = 22000.; g.count = 1
        context.portfolio.available_cash = 1000.; ns['target_mv'](context)
    elif name == 'buy_cap5':
        g.money = 1000.; g.last_month_value = 40000.; g.count = 1; ns['target_mv'](context)
    elif name == 'sell_cap5':
        g.money = 1000.; g.last_month_value = 1000.; g.count = 1; ns['target_mv'](context)
    elif name == 'profit_month_skip':
        g.count = 1; g.take_profit_flag = 1; ns['target_mv'](context)
    elif name == 'missing_position':
        context.portfolio.positions = {}; ns['after_trading_end'](context)
    elif name == 'removed_numpy_irr':
        context.run_params = SimpleNamespace(end_date=context.current_dt); ns['compute_profit'](context)
    else: raise ValueError('Unknown value case')
    return {'orders': orders, 'init_status': g.init_status, 'money': g.money,
        'take_profit_days': g.take_profit_days, 'take_profit_flag': g.take_profit_flag,
        'last_month_value': g.last_month_value, 'investments': g.investments.tolist(), 'orders_are_intents_only': True}


def diagnostics(directory=DIRECTORY):
    rows = []
    def case(name, callback):
        with redirect_stdout(StringIO()), warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter('always')
            try: value = callback(); error = None
            except Exception as exc: value = None; error = {'type': type(exc).__name__, 'message': str(exc)}
        rows.append({'case': name, 'value': value, 'error': error, 'warnings': [str(w.message) for w in notes]})
    for signals, name in [((0,0,1,0), 'growth_first'), ((0,1,0,0), 'nasdaq_second'), ((1,0,0,0), 'duplicate_dividend'), ((0,0,0,0), 'gold_fallback')]:
        case('slots_'+name, lambda signals=signals: slot_case(signals, directory=directory))
    case('slots_strict_minus6', lambda: slot_case(zf=-6, directory=directory))
    case('slots_above_minus6', lambda: slot_case(zf=-5.99, directory=directory))
    case('slots_rejected_orders_still_record_quantity', lambda: slot_case(reject=True, directory=directory))
    case('slots_switch_uses_all_injected_cash', lambda: slot_case((0,1,0,0), held=True, directory=directory))
    case('slots_rejected_sale_still_reassigns', lambda: slot_case((0,1,0,0), held=True, reject=True, directory=directory))
    case('slots_implicit_math_not_imported', lambda: slot_case(missing_math=True, directory=directory))
    def annual_date():
        ns = literal(0, {'get_trade_days': lambda **k: [dt.date(2022, 1, 4)]}, directory)
        return [ns['year_start'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(d))) for d in ('2022-01-04', '2022-02-04', '2022-01-05')]
    case('slots_year_first_day_not_week', annual_date)
    def macd_call():
        calls = []
        ns = literal(0, {'MACD': lambda s, **k: calls.append([s, k]) or ({}, {}, {s: 1.})}, directory)
        return {'value': ns['get_macd_M']('000300.XSHG', dt.datetime(2022,1,4).isoformat()), 'calls': calls}
    case('slots_original_monthly_MACD_contract', macd_call)
    def zf_failure():
        ns = literal(0, {'datetime': dt, 'attribute_history': lambda *a: pd.DataFrame({'close': [10.]*25}, index=pd.date_range('2021-12-01', periods=25))}, directory)
        return ns['get_zf'](SimpleNamespace(current_dt=dt.datetime(2022,1,4)))
    case('slots_original_series_negative_label_fails', zf_failure)
    for name in ('initial', 'initial_70_percent', 'half_reentry', 'profit_equal40', 'profit_equal9',
        'profit_exact90', 'profit_over90', 'yearly_equal75', 'yearly_over75', 'cash_gt1000',
        'buy_cap5', 'sell_cap5', 'profit_month_skip', 'missing_position', 'removed_numpy_irr'):
        case('value_'+name, lambda name=name: value_case(name, directory))
    def strict_float_count():
        def price(*a, **k):
            require(isinstance(k['count'], int), 'Controlled platform count requires int; actual platform validation unknown')
        ns = literal(1, {'get_price': price}, directory); ns['g'].index = '399324.XSHE'
        return ns['max_increase']('2022-01-03')
    case('value_original_float_count375', strict_float_count)
    return {'case_count': len(rows), 'cases': rows, 'economic_rules_changed': False,
        'new_trading_backtests': 0, 'monthly_MACD_computed': False, 'formal_source_research_acceptance': False}


@contextmanager
def network_blocked():
    def forbidden(*a, **k): raise AssertionError('Batch106 offline network forbidden')
    with ExitStack() as stack:
        for obj, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'), (socket.socket, 'connect'),
            (socket.socket, 'connect_ex'), (requests.sessions.Session, 'request')):
            stack.enter_context(patch.object(obj, name, forbidden))
        yield


def selection(catalog):
    rows = []
    major = {'get_fundamentals', 'get_fundamentals_continuously', 'run_query', 'get_factor_values',
        'get_all_securities', 'get_index_stocks', 'get_industry_stocks', 'get_concept_stocks',
        'get_money_flow', 'get_billboard_list', 'get_dominant_future', 'get_ticks', 'read_file'}
    for r in catalog:
        if r['review_status'] == '人工审查完成': continue
        p = Path('repo/量化策略源代码')/r['path']; require(file_sha(p) == r['bytes_sha256'], 'Remaining source changed')
        text = read_source(p)[0]; start = r.get('code_start_line') or 1; calls = []; error = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); t = ast.parse('\n'*(start-1)+'\n'.join(text.splitlines()[start-1:]))
            calls = sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(t)
                if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))})
        except SyntaxError as exc: error = {'line': exc.lineno, 'message': exc.msg}
        rows.append({'path': r['path'], 'bytes_sha256': r['bytes_sha256'], 'lines': len(text.splitlines()),
            'code_start_line': start, 'duplicate_of': r['duplicate_of'], 'syntax_error': error,
            'dependency_calls': sorted(set(calls)&major), 'order_calls': [s for s in calls if s.startswith('order')],
            'all_call_names': calls, 'selected_for_full_review': r['path'] in SOURCES})
    require(len(rows) == 399, 'Parent remaining-source count differs')
    return {'remaining_sources': len(rows), 'rows': rows,
        'scope': 'Static AST calls include unreachable definitions; not a reachability or tradability proof',
        'parse_failure_policy': 'Preserved for manual review; never treated as dependency-free',
        'full_source_reviews_this_stage': list(SOURCES)}


def candidates(directory):
    out = {}; bindings = {}
    for name, cases in REGISTRIES:
        doc = read(name)
        for case in cases:
            rows = [r for r in doc['results'] if r['case'] == case]; require(len(rows) == 1, 'Candidate identity missing')
            r = rows[0]; require(r['status'] == 'success' and file_sha(r['file']) == r['sha256'], 'Candidate changed')
            for w in r['wire_responses']: require(file_sha(w['file']) == w['sha256'], 'Wire bytes changed')
            raw = pd.read_parquet(r['file']); require(len(raw) == r['rows'], 'Candidate row count changed')
            f = pd.DataFrame({'date': pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d'), 'close': pd.to_numeric(raw.close, errors='raise').astype(float)})
            require(f.date.is_monotonic_increasing and not f.date.duplicated().any() and f.date.notna().all() and
                np.isfinite(f.close).all() and (f.close > 0).all(), 'Invalid candidate dates/prices')
            f = f.loc[f.date.le('2026-09-29')].reset_index(drop=True)
            target = directory/'inputs'/(case+'.parquet'); target.parent.mkdir(exist_ok=True); f.to_parquet(target, index=False)
            out[case] = f
            bindings[case] = {'registry': name, 'registry_sha256': file_sha(name), 'candidate': r,
                'input': str(target), 'input_sha256': file_sha(target), 'rows': len(f),
                'first': f.date.iloc[0], 'last': f.date.iloc[-1], 'future_excluded': len(raw)-len(f), 'strict_usable': False}
    return out, bindings


def implementation_files():
    root = Path.cwd().resolve(); names = set(FILES)|{'pyproject.toml', 'uv.lock'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name: continue
        try: relative = Path(name).resolve().relative_to(root)
        except ValueError: continue
        if relative.parts[0] in ('src', 'scripts') and relative.suffix == '.py': names.add(relative.as_posix())
    return {name: file_sha(name) for name in sorted(names)}


def prepare(directory=DIRECTORY):
    require(not directory.exists(), 'Immutable Batch106 stage exists')
    require(file_sha(PARENT) == PARENT_SHA and file_sha('data/snapshots/'+SNAPSHOT+'.json') == SNAPSHOT_SHA, 'Parent/snapshot changed')
    parent = read(PARENT); policy = read('configs/research_metadata.json')
    require(parent['status'] == 'ok' and parent['formal_source_research_acceptance'] and
        policy['accepted_checkpoint'] == parent['progress']['checkpoint'] and policy['accepted_receipt'] == str(PARENT), 'Parent is not formally accepted')
    for name, sha in parent['evidence_sha256'].items(): require(file_sha(name) == sha, 'Parent bound evidence changed: '+name)
    catalog_path = Path('data/catalog/strategies')/parent['progress']['catalog_id']/'catalog.json'
    require(file_sha(catalog_path) == parent['progress']['catalog_sha256'], 'Parent catalog changed')
    selected = selection(read(catalog_path))
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', directory, 'Batch106 full85/69 review and remaining399 AST selection;formal296/trading12/parents87 unchanged')
        save(directory/'selection.json', selected); rows = []
        (directory/'sources').mkdir()
        for number, name in enumerate(SOURCES):
            p = Path('repo/量化策略源代码')/name; require(file_sha(p) == SOURCE_SHA[number], 'Selected source changed')
            target = directory/'sources'/f'{number}.source'; shutil.copyfile(p, target)
            text, encoding = read_source(target); t = ast.parse(text)
            rows.append({'path': name, 'sha256': file_sha(target), 'encoding': encoding, 'lines': len(text.splitlines()),
                'definitions': [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno, 'ast': ast.dump(n, include_attributes=False)}
                    for n in t.body if isinstance(n, ast.FunctionDef)], 'formal_acceptance': False})
        save(directory/'source-review.json', {'sources': rows, 'original_imports_not_executed': True})
        _, bindings = candidates(directory)
        store = Store('data'); state = store.state(SNAPSHOT)
        require(state['tables'] == store.published()['tables'], 'Publication and frozen snapshot differ')
        symbols = ('000300.SH', '399324.SZ', '159905.SZ', '159949.SZ', '510880.SH', '511880.SH', '511010.SH', '513100.SH', '518880.SH')
        counts = {}
        with duckdb.connect() as con:
            for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
                paths = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
                col = 'index' if table == 'index_1d' else 'instrument'; result = dict.fromkeys(symbols, 0)
                if paths:
                    result.update({s: int(n) for s,n in con.execute(f'SELECT "{col}", count(*) FROM read_parquet(?,union_by_name=true) WHERE "{col}" IN ({",".join("?" for _ in symbols)}) GROUP BY "{col}"', [paths, *symbols]).fetchall()})
                counts[table] = result
        calendar = store.load_state(state, 'calendar'); calpath = directory/'inputs/calendar.parquet'; calendar.to_parquet(calpath, index=False)
        save(directory/'dependencies.json', {'candidate_bindings': bindings, 'active_table_counts': counts,
            'calendar_sha256': file_sha(calpath), 'calendar_source_partitions': list(state['tables']['calendar'].values()),
            'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA, 'published_batch': state['batch_id'],
            'original_fund_execution_admitted': False, 'monthly_MACD_seed_or_sampling_proven': False,
            'no_missing_inputs_replaced_by_latest_downloads': True, 'strict_financial_usable_rows': 0})
        files = implementation_files()
        for name in files:
            target = directory/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        save(directory/'prepared.json', {'parent': str(PARENT), 'parent_sha256': PARENT_SHA, 'source_sha256': list(SOURCE_SHA),
            'implementation_sha256': files, 'catalog': str(catalog_path), 'catalog_sha256': file_sha(catalog_path),
            'formal_source_research_acceptance': False, 'formal_trading_acceptance': False})
    return {'status': 'prepared', 'directory': str(directory), 'selected_sources': len(rows), 'candidate_inputs': len(bindings)}


def runtime_binding(directory):
    d = read(directory/'prepared.json')
    for name, sha in d['implementation_sha256'].items():
        require(file_sha(name) == file_sha(directory/'implementation'/name) == sha, 'Frozen implementation changed: '+name)
    require(file_sha(PARENT) == d['parent_sha256'] and file_sha(d['catalog']) == d['catalog_sha256'], 'Parent/catalog changed')
    deps = read(directory/'dependencies.json')
    require(file_sha(directory/'inputs/calendar.parquet') == deps['calendar_sha256'], 'Calendar operands changed')
    for r in deps['candidate_bindings'].values():
        require(file_sha(r['input']) == r['input_sha256'] and file_sha(r['registry']) == r['registry_sha256'] and
            file_sha(r['candidate']['file']) == r['candidate']['sha256'], 'Native operands/binding changed')
        for w in r['candidate']['wire_responses']: require(file_sha(w['file']) == w['sha256'], 'Wire binding changed')


def operands(frame, directory=DIRECTORY):
    """Original extrema on candidate native rows; integral-float count acceptance is explicit."""
    series = pd.Series(frame.close.to_numpy(), index=pd.to_datetime(frame.date)); dates = list(frame.date); rows = []
    ns = literal(1, directory=directory); ns['g'].index = '399324.XSHE'
    position = {v: k for k,v in enumerate(series.index)}
    def price(symbol, *, end_date, frequency, fields, count, panel):
        require(symbol == '399324.XSHE' and frequency == '1d' and fields == 'close' and panel is False, 'Operand API contract changed')
        require(count in (250, 375., 625.) and int(count) == count, 'Unknown original count')
        k = position[pd.Timestamp(end_date)]; require(k+1 >= count, 'Incomplete native operand window')
        return series.iloc[k+1-int(count):k+1].to_frame('close')
    ns['get_price'] = price; values = series.to_numpy(); skipped = 0
    for k in range(249, len(frame)):
        window = values[k-249:k+1]; peak = k-249+int(np.argmax(window)); mx = float(max(window)); mn = float(min(values[peak:k+1]))
        r1 = float(ns['max_increase1y'](series.index[k])); expected1 = (mx/mn-1)*100
        require(r1 == expected1, 'Independent one-year extrema differ')
        row = {'date': dates[k], 'year': int(dates[k][:4]), 'rate1y': r1, 'yearly_gt75': r1>75,
            'two_horizons_complete': False, 'rate0': np.nan, 'position_ratio': np.nan, 'ratio_lt0318': False,
            'peak_date': None, 'prior_min_date': None}
        if k >= 374:
            w = values[k-374:k+1]; peak2 = k-374+int(np.argmax(w))
            if peak2 >= 624:
                m0 = float(min(values[peak2:k+1])); m1 = float(min(values[peak2-624:peak2+1])); max0 = float(max(w))
                if max0 > m1:
                    a,b = ns['max_increase'](series.index[k]); ea = max0/m1*100-100; eb = 1-(max0-m0)/(max0-m1)
                    require(a == ea and b == eb, 'Independent two-horizon extrema differ')
                    row.update(two_horizons_complete=True, rate0=float(a), position_ratio=float(b), ratio_lt0318=bool(b<.318),
                        peak_date=dates[peak2], prior_min_date=dates[peak2-624+int(np.argmin(values[peak2-624:peak2+1]))])
                else: skipped += 1
            else: skipped += 1
        else: skipped += 1
        rows.append(row)
    return pd.DataFrame(rows), skipped


def annual_operands(frame, calendar):
    """Explicit scalar arithmetic reference; original Series[-1] failure is kept separately."""
    series = pd.Series(frame.close.to_numpy(), index=frame.date); days = pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d').tolist()
    rows = []; skipped = 0
    for k,day in enumerate(days):
        if not k or day[:7] == days[k-1][:7]: continue
        previous = days[k-1]; prior_year = str(int(day[:4])-1); prior = [d for d in days if d.startswith(prior_year)]
        history = series.loc[series.index <= previous]
        if len(history)<25 or previous not in series.index or not prior or prior[-1] not in series.index:
            skipped += 1; continue
        old = prior[-1]; last = float(history.iloc[-1]); base = float(series.loc[old]); value = round((last-base)*100/base, 2)
        decimal_value = float(round((Decimal(str(last))-Decimal(str(base)))*100/Decimal(str(base)),2))
        rows.append({'decision_date': day, 'previous_date': previous, 'prior_year_last_date': old,
            'last_close': last, 'prior_year_close': base, 'annual_pct_round2': value, 'strict_gt_minus6': value>-6,
            'decimal_round2_reference': decimal_value, 'decimal_threshold_difference': (decimal_value>-6)!=(value>-6)})
    return pd.DataFrame(rows), skipped


def compute(directory, label):
    runtime_binding(directory); output = directory/label; output.mkdir(exist_ok=False)
    with network_blocked():
        diag = diagnostics(directory); save(output/'diagnostics.json', diag)
        component, skipped = operands(pd.read_parquet(directory/'inputs/index-sz399324.parquet'), directory)
        component.to_parquet(output/'value-average-operands.parquet', index=False)
        annual, annual_skipped = annual_operands(pd.read_parquet(directory/'inputs/fund-510880.parquet'), pd.read_parquet(directory/'inputs/calendar.parquet'))
        annual.to_parquet(output/'annual-return-operands.parquet', index=False)
        summary = {'status': 'native_operands_and_original_diagnostics_only', 'diagnostic_cases': diag['case_count'],
            'one_year_windows': len(component), 'two_horizon_windows': int(component.two_horizons_complete.sum()),
            'incomplete_two_horizons_or_zero_range': skipped, 'first': component.date.iloc[0], 'last': component.date.iloc[-1],
            'independent_extrema_differences': 0, 'annual_scalar_reference_rows': len(annual), 'annual_skipped': annual_skipped,
            'annual_decimal_threshold_differences': int(annual.decimal_threshold_difference.sum()),
            'by_year': component.groupby('year').agg(rows=('date','size'), complete_two_horizons=('two_horizons_complete','sum')).reset_index().to_dict('records'),
            'original_MACD_computed': False, 'complete_original_signals': 0, 'new_trading_backtests': 0,
            'original_69_count_type': '375.0/625.0; explicit integral-float operand adapter, platform acceptance unproven',
            'original_85_get_zf_reproduced': False, 'ETF_adjustment_events_and_intraday_execution_proven': False,
            'formal_source_research_acceptance': False, 'library_goal_complete': False}
        save(output/'summary.json', summary)
    return summary


def verify(directory=DIRECTORY):
    runtime_binding(directory)
    tests = read(directory/'test-results.json')
    require(tests['status'] == 'ok' and tests['local_tests'] == tests['isolated_tests'] == 24 and
        all(file_sha(p) == sha for p,sha in tests['command_sha256'].items()), 'Required local/isolated validation missing')
    require(all(file_sha(directory/'compute'/n) == file_sha(directory/'replay'/n) for n in OUTPUTS), 'Offline outputs differ')
    first = pd.read_parquet(directory/'compute/value-average-operands.parquet')
    second = pd.read_parquet(directory/'compute/annual-return-operands.parquet')
    with network_blocked():
        reference, _ = operands(pd.read_parquet(directory/'inputs/index-sz399324.parquet'), directory)
        annual, _ = annual_operands(pd.read_parquet(directory/'inputs/fund-510880.parquet'), pd.read_parquet(directory/'inputs/calendar.parquet'))
    pd.testing.assert_frame_equal(first, reference, check_exact=True); pd.testing.assert_frame_equal(second, annual, check_exact=True)
    for name in SOURCES: require(file_sha(Path('repo/量化策略源代码')/name) == SOURCE_SHA[SOURCES.index(name)], 'Original changed')
    old = read(directory/'prepared.json'); require(read('data/catalog/strategies/latest.json')['catalog_id'] == Path(old['catalog']).parent.name, 'Stage accidentally published a catalog')
    protection = protect('data', directory)
    result = {'status': 'ok_component_stage', 'summary': read(directory/'compute/summary.json'), 'output_files': len(OUTPUTS),
        'fresh_network_blocked_processes': 2, 'offline_differences': 0, 'saved_exact_readback': True,
        'protection': protection, 'tests': tests, 'source_sha256': list(SOURCE_SHA), 'stage_only': True,
        'formal_source_research_acceptance': False, 'formal_trading_acceptance': False, 'new_trading_backtests': 0,
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
        'next_step': 'Review slot/fill and MACD policy; restore/prove fund events and exact execution; formal source acceptance separately'}
    save(directory/'checked.json', result)
    return {k:v for k,v in result.items() if k!='evidence_sha256'}


def validate_tests(directory=DIRECTORY):
    runtime_binding(directory); isolated = directory/'isolated-validation'; isolated.mkdir(exist_ok=False)
    frozen = read(directory/'prepared.json')['implementation_sha256']
    for name,sha in frozen.items():
        target = isolated/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory/'implementation'/name, target); require(file_sha(target) == sha, 'Isolated copy differs')
    for k,name in enumerate(SOURCES):
        target = isolated/'repo/量化策略源代码'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory/'sources'/f'{k}.source', target)
    script = """import json, pathlib
from scripts import research_strategy_batch106 as study
import observe.data.store, observe.strategy_catalog
import pytest
root=pathlib.Path.cwd().resolve()
modules=[study,observe.data.store,observe.strategy_catalog]
for module in modules:
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), module.__file__
print(json.dumps({'actual_imports':[str(pathlib.Path(m.__file__).resolve()) for m in modules]}),flush=True)
with study.network_blocked():
    raise SystemExit(pytest.main(['-q','tests/unit/test_batch106_research.py']))
"""
    commands = {}
    for label,cwd in (('local',Path.cwd()), ('isolated',isolated.resolve())):
        env = dict(os.environ); env['PYTHONPATH'] = str(cwd/'src')+os.pathsep+str(cwd)
        child = subprocess.run([sys.executable,'-c',script], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        path = directory/(label+'-tests.json')
        save(path, {'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr,
            'cwd': str(cwd), 'network_blocked': True, 'test_scope': 'tests/unit/test_batch106_research.py'})
        commands[str(path)] = file_sha(path)
        require(child.returncode == 0 and '24 passed' in child.stdout, 'Copied/local tests failed; preserve evidence')
    result = {'status':'ok', 'local_tests':24, 'isolated_tests':24, 'actual_imports_checked':3,
        'implementation_copies':len(frozen), 'command_sha256':commands}
    save(directory/'test-results.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare','compute','verify','run','test'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY); parser.add_argument('--label', default='compute'); args = parser.parse_args()
    if args.action == 'prepare': result = prepare(args.directory)
    elif args.action == 'compute': result = compute(args.directory, args.label)
    elif args.action == 'verify': result = verify(args.directory)
    elif args.action == 'test': result = validate_tests(args.directory)
    else:
        commands = []
        for label in ('compute','replay'):
            command = [sys.executable, '-m', 'scripts.research_strategy_batch106', 'compute', '--directory', str(args.directory), '--label', label]
            child = subprocess.run(command, capture_output=True, text=True, timeout=180)
            record = {'command': command, 'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr}
            save(args.directory/(label+'-command.json'), record); commands.append(record)
            require(child.returncode == 0, 'Fresh offline process failed; preserve failed directory')
        result = {'status': 'offline_processes_completed', 'commands': len(commands)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
