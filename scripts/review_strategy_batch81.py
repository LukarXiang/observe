"""Freeze stock/bond switching and drawdown states without fabricating tick paths."""
import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import datetime as dt
from fractions import Fraction
import hashlib
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import re
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
from scripts import review_strategy_batch80 as previous
from scripts import review_strategy_batch76 as batch76
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch24 import INDEX_RECEIPT, INDEX_TERMINAL, index_sample
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch59 import index_operand
from scripts.review_strategy_batch73 import safe_function
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch81/20261009-stock-bond-drawdown')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略2/57.别人恐惧我贪婪3——年化20%的股债组合.txt',
    '2024年度精选策略2/76.ETF-控制回撤性能拉满（国债ETF增强).txt')
SOURCE_SHA = ('50b529f7a2898c26c299d327e61ce1b48b3f50fa4ba63dc9eec0e8a82cb7a3ce',
    '4f41bed4afc75de12c83299fcda6e35311015be749d62efa8de9098e88780dbc')
STOCKS = ('510300.XSHG', '510220.XSHG', '513500.XSHG', '513100.XSHG')
FUNDS = ('512100.XSHG', '511010.XSHG', *STOCKS)
QUERIES = {f'fund-{s}': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh'+s}}
    for s in ('512100', '510220', '513500')}
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/rqalpha', 'rqalpha/portfolio/account.py'), ('repo/backtrader', 'backtrader/analyzers/drawdown.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch81.py',
    'tests/unit/test_batch81_research.py', 'scripts/review_strategy_batch24.py'}))
TABLES = ('index-risk-components.parquet', 'yesterday-drop-components.parquet', 'fund-price-reference.parquet')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'reused-inputs.json', 'index-input.parquet', 'index-input-binding.json', 'dependency-inventory.json',
    'existing-apis.json', 'source-reviews/review.json', 'catalog-impact.json', 'probe-results.json',
    'candidate-audit.json', 'diagnostics.json', 'component-research.json', 'offline-verification.json',
    'offline-catalog.json', 'preflight-failure-binding.json', 'development-tick-regression/failure.json',
    'development-specialized-failure/failure.json', *TABLES}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch80-verification.json')
    review = Path('docs/handoff/2026-10-09-batch80-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted80 binding changed')
    for name in ('scripts/review_strategy_batch80.py', 'tests/unit/test_batch80_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted80 entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def reused_inputs():
    prior = previous.reused_inputs()['inputs']
    rows = {s: prior[s] for s in ('511010.XSHG', '513100.XSHG')}
    rows['510300.XSHG'] = {**batch76.reused_inputs()['inputs']['fund-510300'], 'status': 'success',
        'status_basis': 'Accepted76 registry checked payload and rows; original20 has no status key'}
    receipt = read(INDEX_RECEIPT); terminal = read(INDEX_TERMINAL)
    original = next(r for r in receipt['probes']['results'] if r['endpoint'] == 'bs_000001')
    require(terminal == original['result'] and terminal['status'] == 'success' and
        any(r['file'] == str(INDEX_TERMINAL) and r['sha256'] == file_sha(INDEX_TERMINAL)
            for r in original['evidence_files']), 'Accepted Shanghai index registry changed')
    require(terminal['parameters'] == {'code': 'sh.000001', 'start': '2005-01-05',
        'end': '2026-09-29', 'adjustflag': '3'} and file_sha(terminal['raw_file']) == terminal['raw_sha256'], 'Shanghai index changed')
    rows['000001.XSHG'] = {'file': terminal['raw_file'], 'sha256': terminal['raw_sha256'],
        'rows': len(pd.read_parquet(terminal['raw_file'])), 'origin_receipt': str(INDEX_RECEIPT),
        'origin_receipt_sha256': file_sha(INDEX_RECEIPT), 'status': 'success'}
    for row in rows.values():
        require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'] and
            len(pd.read_parquet(row['file'])) == row['rows'], 'Reused payload changed')
        for wire in row.get('wire_responses', []): require(file_sha(wire['file']) == wire['sha256'], 'Prior response changed')
    return {'inputs': rows, 'new_requests': 0, 'strict_usable': False, 'not_a_backtest': True}


def api_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_sina; source = inspect.getsource(fn); module = inspect.getmodule(fn)
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES,
        'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
        'decoder': module.hk_js_decode, 'decoder_sha256': hashlib.sha256(module.hk_js_decode.encode()).hexdigest(),
        'policy': 'Three serial native GETs,8/10s and parent45s; no redirect/proxy/retry; no headers or credentials archived',
        'limits': ['Daily candidates do not prove tick, identity, events, adjustments, execution or default fund fees']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); instruments = [s[:6]+'.SH' for s in FUNDS]
    tables = ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')
    return {'snapshot': SNAPSHOT, 'funds': list(FUNDS),
        'registered_fund_rows': {t: len(store.load_state(state, t, filters=[('instrument', 'in', instruments)])) for t in tables},
        'index_intraday_rows': len(store.load_state(state, 'bars_5m', filters=[('instrument', 'in', ['000001.SH', '000300.SH'])])),
        'original_input_rows_admitted': 0, 'not_a_backtest': True,
        'gates': ['57 original14:55 Shanghai index tick and two prior closes/platform Series indexing unproved',
            '76 native60-index rows available; drawdown requires real portfolio equity and close-only highwater',
            'Six fund identity/events/reference prices/state/adjustment and9:30/14:55 execution unproved',
            'Both sources configure stock fees only; actual default fund fees/slippage unproved',
            '57 delayed/partial sale cash and day counter after failed buy;76 debt-first target submission retained'],
        'old_missing_fund_evidence': {'512100': 'data/staging/strategies-batch35/20261006-bbi-damped-v21/probe-results.json',
            '510220': 'data/staging/strategies-batch43/20261007-candle-weekly-margin/probe-results.json',
            '513500': 'data/staging/strategies-batch51/20261007-weekly-etf-emotion/dependency-inventory.json'},
        'acquisition_policy': 'Reuse accepted inputs; three unfilled native fund histories only; no substitution of current close for14:55 tick'}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted evidence changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    for repository, name in REFERENCES: require((Path(repository)/name).is_file(), 'Reference missing')
    reused = reused_inputs(); frame, operand = index_operand(root); dep = inventory(root)
    checkpoint(root, directory, 'Batch81 baseline; accepted daily inputs checked; tick and fund fees still missing')
    save(directory/'input-binding.json', bound); save(directory/'reused-inputs.json', reused)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    path = directory/'index-input.parquet'
    with path.open('xb') as stream: frame.to_parquet(stream, index=False)
    save(directory/'index-input-binding.json', {**operand, 'frozen_sha256': file_sha(path)})
    save(directory/'dependency-inventory.json', dep); save(directory/'existing-apis.json', api_evidence())
    failure = directory.with_name(directory.name+'-preflight-failure-01')
    save(directory/'preflight-failure-binding.json', {str(p): file_sha(p) for p in sorted(failure.iterdir()) if p.is_file()})
    return {'status': 'ok', 'reused_files': len(reused['inputs']), 'index_rows': len(frame), 'not_a_backtest': True}


def catalog_impact(old, current):
    require(len(old) == len(current) == 695 and set(old) == {r['path'] for r in current}, 'Catalog scope changed')
    rows = []; impacted = []
    for item in current:
        prior = old[item['path']]; fields = sorted(k for k in set(prior) | set(item) if prior.get(k) != item.get(k))
        text, _ = read_source(Path('repo/量化策略源代码')/item['path'])
        candidate = bool(re.search(r'\bget_current_tick\s*\(', text))
        expected_apis = prior['apis'].copy()
        if candidate:
            require('get_current_tick' in item['apis'], 'Tick candidate lost')
            require([a for a in item['apis'] if a != 'get_current_tick'] == expected_apis, 'Unrelated APIs changed')
            impacted.append(item['path'])
        else: require(item['apis'] == expected_apis, 'Unrelated APIs changed')
        if item['path'] not in SOURCES:
            require(set(fields) <= {'apis', 'status', 'gaps'} and (not fields or candidate), 'Unrelated catalog change')
            if prior['review_status'] == '人工审查完成':
                require(set(fields) <= {'apis'}, 'Existing manual rules/status changed')
        def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rows.append({'path': item['path'], 'before_sha256': digest(prior), 'after_sha256': digest(item),
            'changed_fields': fields, 'tick_text_candidate': candidate,
            'tick_ast_called': 'get_current_tick' in (item['called_functions'] or []),
            'differences': {k: {'before': prior.get(k), 'after': item.get(k)} for k in fields}})
    return {'rows': rows, 'total_sources': len(rows), 'tick_candidates': impacted,
        'changed_sources': sum(bool(r['changed_fields']) for r in rows), 'newly_reviewed': list(SOURCES),
        'policy': 'All695 compared; existing manual rules/status and source fingerprints retained; text API candidates include comments'}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    (directory/'references').mkdir(exist_ok=False)
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    impact = catalog_impact(old, current); save(directory/'catalog-impact.json', impact)
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch81 two complete sources frozen; all695 tick detection impact archived; no trades admitted')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/SOURCES[number]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['review'] == REVIEWS[SOURCES[number]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    for name, sha in read(directory/'preflight-failure-binding.json').items(): require(file_sha(name) == sha, 'Preflight failure changed')
    failure = read(directory/'development-tick-regression/failure.json')
    require(failure['returncode'] == 1, 'Tick regression red missing')
    for name, sha in failure['binding'].items(): require(file_sha(name) == sha, 'Tick regression evidence changed')
    specialized = directory/'development-specialized-failure/failure.json'
    if specialized.exists():
        failure = read(specialized); require(failure['returncode'] == 1, 'Specialized failure missing')
        for name, sha in failure['binding'].items(): require(file_sha(name) == sha, 'Specialized failure evidence changed')
    old = {r['path']: r for r in read(doc['previous_catalog_file'])}
    require(catalog_impact(old, read(Path(doc['catalog']['output'])/'catalog.json')) == read(directory/'catalog-impact.json'), 'Catalog impact changed')


def namespace(directory, number):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    nodes = [n for n in ast.parse(read_source(source)[0]).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (8 if number else 4) and all(safe_function(n) for n in nodes), 'Unsafe/missing definitions')
    ns = {'g': SimpleNamespace(), 'np': np, 'pd': pd, 'print': lambda *a: None,
        'log': SimpleNamespace(info=lambda *a: None), 'set_benchmark': lambda *a: None,
        'set_option': lambda *a: None, 'OrderCost': lambda **kw: kw, 'set_order_cost': lambda *a, **kw: None,
        'run_daily': lambda *a, **kw: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<batch81-frozen-definitions>', 'exec'), ns)
    return ns


def context(cash=10000., holdings=None, total=None):
    positions = {s: SimpleNamespace(value=v) for s, v in (holdings or {}).items()}
    return SimpleNamespace(current_dt=dt.datetime(2020, 1, 6, 14, 55), previous_date=dt.date(2020, 1, 3),
        portfolio=SimpleNamespace(positions=positions, available_cash=cash, cash=cash,
            total_value=total if total is not None else cash+math.fsum(p.value for p in positions.values())))


def order_stub(ns, ctx, mode):
    require(mode in ('immediate', 'delayed', 'partial', 'failed'), 'Unknown synthetic fill mode')
    orders = []; events = []
    def order(stock, target):
        p = ctx.portfolio; current = p.positions.get(stock, SimpleNamespace(value=0.)).value
        orders.append([stock, target]); event = {'stock': stock, 'target': target, 'cash_before': p.available_cash,
            'current_value': current, 'accepted': False}; events.append(event)
        if mode in ('delayed', 'failed'): return None
        diff = (target-current)*(.5 if mode == 'partial' else 1.)
        if diff > p.available_cash: return None
        p.available_cash -= diff; p.cash = p.available_cash
        value = current+diff; event['accepted'] = True
        if value == 0: p.positions.pop(stock, None)
        else: p.positions[stock] = SimpleNamespace(value=value)
        return SimpleNamespace(status='synthetic')
    ns['order_target_value'] = order
    return orders, events


def controlled57(directory, *, closes=(1000., 980.), tick=960., holdings=None, cash=100.,
                 day=1, button=0, mode='delayed', steps=1, series=False):
    ns = namespace(directory, 0); ctx = context(cash, holdings); ns['initialize'](ctx); ns['market_open'](ctx)
    ns['g'].day = day; ns['g'].buttun = button
    value = pd.Series(closes, index=pd.date_range('2020-01-02', periods=len(closes))) if series else np.asarray(closes, float)
    ns['attribute_history'] = lambda *a, **kw: {'close': value}
    ns['get_current_tick'] = lambda *a: SimpleNamespace(current=tick)
    orders, events = order_stub(ns, ctx, mode); error = None; states = []
    for _ in range(steps):
        try: ns['market_1455'](ctx)
        except (KeyError, IndexError, ZeroDivisionError) as exc: error = type(exc).__name__; break
        states.append({'day': ns['g'].day, 'button': ns['g'].buttun, 'cash': ctx.portfolio.available_cash})
    return {'orders': orders, 'events': events, 'states': states, 'error': error,
        'final_values': {s: p.value for s, p in ctx.portfolio.positions.items()}, 'mode': mode,
        'history_shape': 'datetime-indexed Series' if series else 'explicit ndarray operand',
        'synthetic_state_only': True, 'not_a_backtest': True}


def controlled76(directory, values, *, total=10000., highwater=10000., status=0, cash=10000., holdings=None, mode='delayed'):
    ns = namespace(directory, 1); ctx = context(cash, holdings, total); ns['initialize'](ctx)
    ns['g'].max_net = highwater; ns['g'].drawback_status = status
    ns['get_bars'] = lambda *a, **kw: {'close': values}
    orders, events = order_stub(ns, ctx, mode); ns['before_market_open'](ctx); ns['market_open'](ctx)
    return {'status': ns['g'].drawback_status, 'bad_sign': ns['g'].bad_sign,
        'stock_cash': ctx.stock_cash, 'debt_cash': ctx.debt_cash, 'orders': orders, 'events': events,
        'final_cash': ctx.portfolio.available_cash, 'final_values': {s: p.value for s, p in ctx.portfolio.positions.items()},
        'total': total, 'highwater': highwater, 'mode': mode, 'synthetic_state_only': True, 'not_a_backtest': True}


def risk_arithmetic(values):
    require(len(values) == 60 and all(math.isfinite(float(v)) and v > 0 for v in values), 'Invalid60-row price window')
    means = [math.fsum(float(v) for v in values[a:b])/19 for a, b in ((40, 59), (20, 39), (0, 19))]
    peak9 = max(float(v) for v in values[50:59]); peak19 = max(float(v) for v in values[40:59]); last = float(values[59])
    trend = means[0] < means[1] < means[2]; fall9 = last < .92*peak9; fall19 = last < .85*peak19
    return {'mean_recent19': means[0], 'mean_middle19': means[1], 'mean_old19': means[2],
        'max_previous9': peak9, 'max_previous19': peak19, 'last': last,
        'trend_declining': trend, 'fall_from9': fall9, 'fall_from19': fall19,
        'bad_sign': int(trend or fall9 or fall19)}


def diagnostics(directory):
    validate_sources(directory); rows = []
    for number in range(2):
        ns = namespace(directory, number); costs = []; schedules = []; ctx = context()
        ns['set_order_cost'] = lambda v, **kw: costs.append([v, kw])
        ns['run_daily'] = lambda fn, **kw: schedules.append([fn.__name__, kw])
        ns['initialize'](ctx)
        rows.append({'case': f'{number}_defaults', 'costs': costs, 'schedules': schedules,
            'slippage_explicit': False, 'fund_cost_explicit': False, 'initial_state': vars(ns['g']).copy()})
    for label, kwargs in (
        ('initial_no_signal', {'closes': (1000., 1000.), 'tick': 1000.}),
        ('series_label_error', {'series': True}), ('missing_close', {'closes': (1000.,)}),
        ('zero_close', {'closes': (0., 0.)}),
        ('y_equal', {'closes': (1000., 985.), 'tick': 965.}),
        ('y_above', {'closes': (1000., np.nextafter(985., np.inf)), 'tick': 965.}),
        ('t_equal', {'closes': (1000., 500.), 'tick': 493.}),
        ('t_above', {'closes': (1000., 500.), 'tick': np.nextafter(493., np.inf)}),
        ('repeat_held', {'holdings': {'512100.XSHG': 1000.}, 'day': 7, 'button': 1}),
        ('entry_and_exit_same_callback', {'day': 19, 'holdings': {'511010.XSHG': 1000.}, 'mode': 'immediate'}),
        ('exit_without_trigger', {'closes': (1000., 1000.), 'tick': 1000., 'day': 19, 'button': 1,
            'holdings': {'512100.XSHG': 1000.}, 'mode': 'immediate'}),
        ('entry_19_callbacks', {'steps': 19, 'mode': 'immediate'})):
        rows.append({'case': '57_'+label, **controlled57(directory, **kwargs)})
    for mode in ('immediate', 'delayed', 'partial', 'failed'):
        rows.append({'case': '57_switch_'+mode, **controlled57(directory, holdings={'511010.XSHG': 1000.}, mode=mode)})
    safe = np.linspace(100., 110., 60)
    for initial in range(3):
        for label, total in (('below95', 9499.), ('equal95', 9500.), ('equal975', 9750.),
            ('above975', 9751.), ('equal99', 9900.), ('above99', 9901.)):
            rows.append({'case': f'76_state{initial}_{label}', **controlled76(directory, safe, total=total, status=initial)})
    for label, values in (('declining_means', np.linspace(120., 100., 60)),
        ('peak9_equal', np.r_[np.full(59, 100.), 92.]),
        ('peak9_below', np.r_[np.full(59, 100.), np.nextafter(92., -np.inf)]),
        ('peak19_equal', np.r_[np.full(40, 50.), 100., np.full(18, 50.), 85.]),
        ('peak19_below', np.r_[np.full(40, 50.), 100., np.full(18, 50.), np.nextafter(85., -np.inf)])):
        row = controlled76(directory, values, status=2, total=9500.)
        require(row['bad_sign'] == risk_arithmetic(values)['bad_sign'], 'Independent synthetic risk differs')
        rows.append({'case': '76_'+label, **row})
    for mode in ('immediate', 'delayed', 'partial', 'failed'):
        rows.append({'case': '76_stock_to_debt_'+mode, **controlled76(directory, np.linspace(120., 100., 60),
            cash=0., holdings=dict.fromkeys(STOCKS, 2500.), mode=mode)})
    ns = namespace(directory, 1); ctx = context(); ns['initialize'](ctx); ns['get_bars'] = lambda *a, **kw: {'close': safe}
    ns['before_market_open'](ctx); snapshots = []
    for value in (10000., 11000., 10900.):
        ctx.portfolio.total_value = value; before = ns['g'].max_net; ns['after_market_close'](ctx)
        snapshots.append({'close_equity': value, 'max_before': before, 'max_after': ns['g'].max_net})
    rows.append({'case': '76_highwater_after_close', 'states': snapshots, 'synthetic_state_only': True})
    # Sentinels expose the actual skipped boundary rows without changing source slices.
    windows = []
    class Window:
        def __init__(self, indices): self.indices = indices
        def mean(self): windows.append({'kind': 'mean', 'indices': self.indices}); return 100.
        def max(self): windows.append({'kind': 'max', 'indices': self.indices}); return 100.
    class Operand:
        def __getitem__(self, key): return Window(list(range(60))[key]) if isinstance(key, slice) else 100.
    ns['get_bars'] = lambda *a, **kw: {'close': Operand()}; ctx.portfolio.total_value = 10000.; ns['cash_management'](ctx)
    rows.append({'case': '76_exact_slices', 'windows': windows, 'synthetic_state_only': True})
    return {'cases': rows, 'not_a_backtest': True,
        'operand_policy': 'Explicit ndarray compatibility reference only; datetime Series error retained; no tick/portfolio NAV substitution'}


def expected_request(case):
    symbol = QUERIES[case]['parameters']['symbol']
    return f'https://finance.sina.com.cn/realstock/company/{symbol}/hisdata_klc2/klc_kl.js'


def invoke(case):
    import akshare as ak
    q = QUERIES[case]; return getattr(ak, q['function'])(**q['parameters'])


def worker(directory, case):
    folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(not row['attempts'] and method.upper() == 'GET' and url == expected_request(case) and
            kwargs.get('params') is None, 'Unexpected/repeated request')
        row['attempts'].append({'method': 'GET', 'url': url, 'parameters': None})
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder/'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path),
            'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'Installed API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder/'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); validate_sources(directory)
    require(not (directory/'probe-results.json').exists(), 'Probe archive exists'); before = implementation(); rows = []
    for case in QUERIES:
        require(not (directory/'probes'/case).exists(), 'Probe already attempted')
        cmd = [sys.executable, '-m', 'scripts.review_strategy_batch81', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            save(directory/f'worker-{case}.json', {'command': cmd, 'returncode': process.returncode,
                'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/f'worker-{case}.json', {'command': cmd, 'status': 'timeout', 'timeout_seconds': 45})
            folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'result.json').exists():
                save(folder/'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout',
                    'attempts': [], 'wire_responses': [], 'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                    'error': '45s deadline; completion unproved', 'strict_usable': False, 'published': False,
                    'api_sha256': file_sha(directory/'existing-apis.json')})
        rows.append(read(directory/'probes'/case/'result.json')); print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    require(before == implementation(), 'Implementation changed during probes')
    save(directory/'probe-results.json', {'results': rows, 'implementation_sha256': before,
        'original_input_rows_admitted': 0, 'not_a_backtest': True})
    save(directory/'candidate-audit.json', verify_candidates(directory))
    return archive(root, 'Batch81 three bounded native fund requests and parser evidence archived; tick/equity inputs unfilled')


def verify_candidates(directory):
    require(api_evidence() == read(directory/'existing-apis.json'), 'Installed API changed')
    rows = read(directory/'probe-results.json')['results']; profiles = []
    require(len(rows) == len(QUERIES) and {r['case'] for r in rows} == set(QUERIES), 'Probe scope changed')
    for row in rows:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory/'existing-apis.json'), 'Query binding changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial evidence changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(len(row['attempts']) == len(row['wire_responses']) == 1 and
                row['attempts'][0] == {'method': 'GET', 'url': expected_request(case), 'parameters': None} and
                file_sha(row['file']) == row['sha256'], 'Native parse binding changed')
            calls = []
            def replay(session, method, url, **kwargs):
                require(not calls and method.upper() == 'GET' and url == expected_request(case) and kwargs.get('params') is None, 'Unexpected replay')
                calls.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes()
                response.encoding = wire['encoding']; response.url = url; response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(calls) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Parse scope changed')
            dates = pd.to_datetime(frame.date)
            require(dates.notna().all() and not dates.duplicated().any(), 'Invalid native dates')
            profile.update(rows=len(frame), first=str(dates.min().date()) if len(frame) else None,
                last=str(dates.max().date()) if len(frame) else None,
                post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29')).sum()), native_parser_match=True)
        else:
            require(row.get('error'), 'Failure lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'not_a_backtest': True}


def index_frames(directory):
    require(read(directory/'reused-inputs.json') == reused_inputs(), 'Reused registry changed')
    path = directory/'index-input.parquet'; bound = read(directory/'index-input-binding.json')
    require(file_sha(path) == bound['frozen_sha256'], 'Frozen index changed')
    frame = pd.read_parquet(path); dates = frame.date.astype(str).tolist()
    require(len(dates) == bound['rows'] == 5280 and len(set(dates)) == len(dates) and dates == sorted(dates), 'Index dates changed')
    shanghai = read(directory/'reused-inputs.json')['inputs']['000001.XSHG']
    original = pd.read_parquet(shanghai['file']); shanghai_frame = index_sample(original, dates)
    return frame, shanghai_frame


def fund_frames(directory):
    rows = read(directory/'reused-inputs.json')['inputs']; inputs = {s: rows[s] for s in ('511010.XSHG', '513100.XSHG', '510300.XSHG')}
    for row in read(directory/'probe-results.json')['results']:
        if row['status'] == 'success': inputs[row['case'].removeprefix('fund-')+'.XSHG'] = row
    profiles = []; references = []
    for stock, row in inputs.items():
        require(file_sha(row['file']) == row['sha256'], 'Fund payload changed')
        frame = pd.read_parquet(row['file']).copy(); frame.date = pd.to_datetime(frame.date)
        require(frame.date.notna().all() and not frame.date.duplicated().any() and len(frame) == row['rows'], 'Fund date/row scope changed')
        for field in ('open', 'high', 'low', 'close'): frame[field] = pd.to_numeric(frame[field], errors='raise')
        frame = frame.sort_values('date'); a = frame[['open', 'high', 'low', 'close']].to_numpy(float)
        valid = np.isfinite(a).all(axis=1) & (a > 0).all(axis=1) & (a[:, 1] >= a[:, 2]) & (a[:, 0] >= a[:, 2]) & (a[:, 0] <= a[:, 1]) & (a[:, 3] >= a[:, 2]) & (a[:, 3] <= a[:, 1])
        ratio = frame.close/frame.close.shift(1); jumps = frame.loc[(ratio-1).abs() > .3]
        profiles.append({'instrument': stock, 'file': row['file'], 'sha256': row['sha256'], 'rows': len(frame),
            'first': str(frame.date.min().date()), 'last': str(frame.date.max().date()),
            'post_cutoff_rows': int((frame.date > pd.Timestamp('2026-09-29')).sum()),
            'invalid_ohlc_dates': [str(d.date()) for d in frame.loc[~valid, 'date']],
            'raw_close_jumps_over30pct': [{'date': str(r.date.date()), 'close': float(r.close),
                'ratio_to_previous_native_row': float(ratio.loc[i])} for i, r in jumps.iterrows()],
            'strict_usable': False, 'jump_policy': 'Native data retained; no corporate-action inference or smoothing'})
        for r in frame.loc[(frame.date <= pd.Timestamp('2026-09-29')) & valid].itertuples():
            references.append({'instrument': stock, 'history_end': str(r.date.date()), 'close_reference': float(r.close),
                'original_intraday_price': False, 'not_a_portfolio_state': True})
    return pd.DataFrame(references), profiles


def components(directory):
    validate_sources(directory); index, shanghai = index_frames(directory); ns = namespace(directory, 1); ctx = context()
    ns['initialize'](ctx); values = index.close.to_numpy(float); risk = []; drops = []; maxima = defaultdict(float)
    for k in range(59, len(values)):
        operand = values[k-59:k+1]; independent = risk_arithmetic(operand)
        ns['get_bars'] = lambda *a, v=operand, **kw: {'close': v}; ns['cash_management'](ctx)
        original_bad = ns['g'].bad_sign; require(original_bad == independent['bad_sign'], 'Independent original risk differs')
        original_values = dict(zip(('mean_recent19', 'mean_middle19', 'mean_old19', 'max_previous9', 'max_previous19'),
            (operand[-20:-1].mean(), operand[-40:-21].mean(), operand[-60:-41].mean(), operand[-10:-1].max(), operand[-20:-1].max()), strict=True))
        for key, value in original_values.items(): maxima[key] = max(maxima[key], previous.previous.close_enough(float(value), independent[key], key))
        risk.append({'history_end': str(index.date.iloc[k]),
            'next_decision_date': str(index.date.iloc[k+1]) if k+1 < len(index) else None,
            **independent, 'original_bad_sign': original_bad, 'full_source_cash_management_executed': True,
            'real_drawdown_state_available': False})
    close = shanghai.close.to_numpy(float)
    for k in range(1, len(close)):
        a, b = close[k-1:k+1]; actual = float((b-a)/a)
        independent = float((Fraction(str(b))-Fraction(str(a)))/Fraction(str(a)))
        maxima['yesterday_drop'] = max(maxima['yesterday_drop'], previous.previous.close_enough(actual, independent, 'yesterday_drop'))
        require((actual <= -.015) == (independent <= -.015), 'Independent yesterday threshold differs')
        drops.append({'history_end': str(shanghai.date.iloc[k]),
            'next_decision_date': str(shanghai.date.iloc[k+1]) if k+1 < len(close) else None,
            'prior_close': float(a), 'last_close': float(b), 'yesterday_drop': actual, 'independent_drop': independent,
            'yesterday_condition': actual <= -.015, 'original14_55_tick_available': False,
            'complete_entry_signal_available': False})
    funds, profiles = fund_frames(directory); risk_table = pd.DataFrame(risk); drop_table = pd.DataFrame(drops)
    summary = {'risk_components': previous.coverage(risk_table), 'yesterday_drop_components': previous.coverage(drop_table),
        'fund_price_reference': previous.coverage(funds), 'profiles': profiles,
        'original_cash_management_calls': len(risk), 'original_risk_differences': 0, 'yesterday_condition_differences': 0,
        'risk_bad_days': int(risk_table.bad_sign.sum()), 'yesterday_condition_days': int(drop_table.yesterday_condition.sum()),
        'max_abs_difference': dict(maxima), 'complete57_entry_signals': 0, 'real76_equity_trajectories': 0,
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_input_rows_admitted': 0,
        'limits': ['60-row original price condition only; synthetic portfolio state is not original NAV or a trade path',
            '57 yesterday condition only; no daily-close substitution for14:55 tick',
            'Latest native row has no next decision in frozen calendar; retained as arithmetic reference only',
            'Fund raw jumps retained without inferred events/adjustments; no portfolio return']}
    return (risk_table, drop_table, funds), summary


def study(root, directory):
    require(all(not (directory/n).exists() for n in ('diagnostics.json', 'component-research.json', *TABLES)), 'Study archive exists')
    binding(root, directory); before = implementation(); diag = diagnostics(directory); tables, summary = components(directory)
    require(before == implementation(), 'Implementation changed during study')
    for name, table in zip(TABLES, tables, strict=True):
        with (directory/name).open('xb') as stream: table.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', {**summary, 'implementation_sha256': before})
    return archive(root, 'Batch81 longest60-row risk and yesterday-drop arithmetic archived; no original tick/equity trajectory')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch81 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); tables, summary = components(directory); candidates = verify_candidates(directory)
        dep = inventory(root); catalog = offline_catalog(root, directory)
    for name, table in zip(TABLES, tables, strict=True): pd.testing.assert_frame_equal(table, pd.read_parquet(directory/name), check_exact=True)
    require(before == implementation() and diag == read(directory/'diagnostics.json') and
        {**summary, 'implementation_sha256': before} == read(directory/'component-research.json') and
        candidates == read(directory/'candidate-audit.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch81 forbidden-network diagnostics/three arithmetic tables/native parsers/inventory/three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch81.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch81_research.py', 'tests/unit/test_batch80_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


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
    summary = read(directory/'component-research.json')
    require(summary['original_risk_differences'] == summary['yesterday_condition_differences'] == 0 and
        summary['original_cash_management_calls'] == summary['risk_components']['rows'] > 0 and
        summary['complete57_entry_signals'] == summary['real76_equity_trajectories'] == summary['new_trading_backtests'] == 0, 'Independent original risk or trade scope differs')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch81 stock/bond source and longest risk/reference research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch81-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'diagnostics': read(directory/'diagnostics.json'), 'components': summary, 'inventory': read(directory/'dependency-inventory.json'),
        'candidates': read(directory/'candidate-audit.json'), 'catalog_impact': read(directory/'catalog-impact.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'worker', 'probe', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
