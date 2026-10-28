"""Freeze reversal rotation and fixed-weight fund rules without inventing trades."""
import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stdout
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
from scripts import review_strategy_batch79 as previous
from scripts import review_strategy_batch76 as batch76
from scripts import review_strategy_batch77 as batch77
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch80/20261009-reversal-fixed-weights')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略1/38.ETF核心资产轮动-添油加醋.txt', '聚宽2025年精选/90固收+.txt')
SOURCE_SHA = ('0e1415cfcb53c25a34c6b25e1480744b59a046b593332c93c4231335e64a826a',
    '8855383f94fa8a5b28a370263c31f0e1b02fdc0f272e0a602cce7636d5b187e7')
POOL = previous.POOLS[0]
WEIGHTS = {'510880.XSHG': .08, '513100.XSHG': .08, '518880.XSHG': .14, '511010.XSHG': .70}
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch80.py', 'tests/unit/test_batch80_research.py'}))
REFERENCES = previous.REFERENCES
TABLES = ('score-components.parquet', 'beta-components.parquet', 'rank-components.parquet', 'fixed-weight-price-reference.parquet')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py', 'reused-inputs.json',
    'dependency-inventory.json', 'source-reviews/review.json', 'diagnostics.json', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json', 'development-failure-specialized/binding.json',
    'development-failure-window-cache/binding.json', *TABLES}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch79-verification.json')
    review = Path('docs/handoff/2026-10-09-batch79-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted79 binding changed')
    for name in ('scripts/review_strategy_batch79.py', 'tests/unit/test_batch79_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted79 entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def reused_inputs():
    prior = previous.reused_inputs()['inputs']; result = {s: prior[s] for s in POOL}
    for batch, instrument in ((batch76, '510880.XSHG'), (batch77, '511010.XSHG')):
        path = batch.DIRECTORY/'probe-results.json'
        require(file_sha(path) == read(batch.DIRECTORY/'checked-state.json')['evidence_sha256']['probe-results.json'], 'Prior registry changed')
        rows = [r for r in read(path)['results'] if r['case'] == 'fund-'+instrument[:6]]
        require(len(rows) == 1, 'Prior scope changed'); result[instrument] = rows[0]
    for row in result.values():
        require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'] and
            len(pd.read_parquet(row['file'])) == row['rows'], 'Reused payload changed')
        for wire in row.get('wire_responses', []): require(file_sha(wire['file']) == wire['sha256'], 'Prior response changed')
    return {'inputs': result, 'new_requests': 0, 'strict_usable': False, 'not_a_backtest': True}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    funds = [s[:6]+('.SZ' if s.endswith('XSHE') else '.SH') for s in set(POOL) | set(WEIGHTS)]
    return {'snapshot': SNAPSHOT, 'registered_fund_rows': {t: len(store.load_state(state, t, filters=[('instrument', 'in', funds)]))
        for t in ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')},
        'original_input_rows_admitted': 0, 'not_a_backtest': True,
        'gates': ['Six fund identity/events/reference prices/state/adjustment/fee versions and9:30/9:35 execution unproved',
            '38 implicit math/mean/std platform injection and default Position/immediate cash unproved',
            '90 daily closes do not establish9:35 prices, holdings, order failures or partial fills'],
        'existing_minute_probe': {'file': str(previous.DIRECTORY/'probe-results.json'),
            'sha256': file_sha(previous.DIRECTORY/'probe-results.json'), 'scope': '15994960m; not any80 original intraday input'},
        'acquisition_policy': 'Reuse accepted native daily files; no redownload; prior recent-minute interface cannot prove long9:35 history'}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted evidence changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    reused = reused_inputs(); dep = inventory(root)
    checkpoint(root, directory, 'Batch80 baseline; six accepted native fund inputs reused')
    save(directory/'input-binding.json', bound); save(directory/'reused-inputs.json', reused)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    save(directory/'dependency-inventory.json', dep)
    return {'status': 'ok', 'funds_reused': len(reused['inputs']), 'not_a_backtest': True}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(); rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    (directory/'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch80 two complete sources frozen; beta229 and fixed initial targets retained; research in progress')


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
    for name in ('development-failure-specialized', 'development-failure-window-cache'):
        failure = directory/name/'binding.json'
        if failure.exists():
            for name, sha in read(failure).items(): require(file_sha(name) == sha, 'Development failure changed')


def namespace(directory, number, injected=True):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    nodes = [n for n in ast.parse(read_source(source)[0]).body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == (2 if number else 5) and all(safe_function(n) for n in nodes), 'Unsafe/missing definitions')
    ns = {'g': SimpleNamespace(), 'np': np, 'pd': pd,
        'log': SimpleNamespace(set_level=lambda *a: None), 'set_benchmark': lambda *a: None,
        'set_option': lambda *a: None, 'set_slippage': lambda *a: None, 'FixedSlippage': lambda v: v,
        'set_order_cost': lambda *a, **kw: None, 'OrderCost': lambda **kw: kw, 'run_daily': lambda *a: None}
    if injected: ns.update(math=math, mean=np.mean, std=np.std)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<batch80-frozen-definitions>', 'exec'), ns)
    ns['initialize'](None); return ns


def synthetic_orders(directory, number, holdings, cash, prices=None, mode='delayed', spread=1., slope=1.1, beta=1.):
    """Run original order intent against explicit synthetic callback state, without NAV."""
    ns = namespace(directory, number); orders = []; events = []
    pos = defaultdict(lambda: SimpleNamespace(total_amount=0, value=0.))
    pos.update({s: SimpleNamespace(total_amount=100, value=v) for s, v in holdings.items()})
    total = cash+math.fsum(holdings.values()); portfolio = SimpleNamespace(positions=pos, cash=cash, available_cash=cash, total_value=total)
    ctx = SimpleNamespace(portfolio=portfolio)
    def order(stock, value):
        orders.append([stock, value]); events.append({'stock': stock, 'target': value, 'cash_before': portfolio.cash, 'total_before': portfolio.total_value})
        if mode == 'delayed': return None
        factor = .5 if mode == 'partial' else 1.
        current = pos[stock].value; diff = (value-current)*factor
        if diff > portfolio.cash: return None
        portfolio.cash -= diff; portfolio.available_cash = portfolio.cash
        pos[stock].value += diff
        if pos[stock].value == 0: del pos[stock]
        # Deliberately perturb total_value to verify frozen callback targets in90.
        portfolio.total_value += 1.
        return SimpleNamespace(status='synthetic')
    ns['order_target_value'] = order
    if number:
        ns['get_current_data'] = lambda: {s: SimpleNamespace(last_price=v) for s, v in (prices or dict.fromkeys(WEIGHTS, 1.)).items()}
    else:
        ns['get_rank'] = lambda pool: pd.DataFrame({'score': [spread, 0., 0., 0.]}, index=POOL)
        ns['getBeta'] = lambda *a: beta
        ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'low': np.arange(18)+10., 'high': slope*(np.arange(18)+10.)+3.})
    error = None
    try: ns['trade'](ctx)
    except (KeyError, ZeroDivisionError) as exc: error = type(exc).__name__
    return {'orders': orders, 'events': events, 'initial_total': total, 'final_cash': portfolio.cash,
        'final_synthetic_values': {s: p.value for s, p in pos.items()},
        'mode': mode, 'error': error, 'synthetic_state_only': True, 'not_a_backtest': True}


def diagnostics(directory):
    validate_sources(directory); rows = []
    for number in range(2):
        ns = namespace(directory, number); costs = []; slips = []; schedules = []
        ns['set_order_cost'] = lambda value, **kw: costs.append([value, kw]); ns['set_slippage'] = slips.append
        ns['run_daily'] = lambda fn, time: schedules.append([fn.__name__, time]); ns['initialize'](None)
        rows.append({'case': f'{number}_defaults', 'costs': costs, 'slippage': slips, 'schedules': schedules,
            'pool': dict(ns['g'].weights) if number else list(ns['g'].etf_pool)})
    ns = namespace(directory, 0, injected=False)
    ns['attribute_history'] = lambda s, count, *a, **kw: pd.DataFrame({'close': np.exp(np.arange(count)*.001),
        'low': np.arange(count)+10., 'high': np.arange(count)+11.})
    for name, args, missing in (('get_rank', [POOL], 'math'), ('countBeta', [None, POOL[0]], 'mean'), ('countBeta', [None, POOL[0]], 'std')):
        try: ns[name](*args)
        except NameError as exc: rows.append({'case': '38_missing_'+missing, 'exception': type(exc).__name__, 'missing': exc.name})
        else: raise AssertionError('Missing platform injection unexpectedly succeeded')
        if missing == 'mean': ns['mean'] = np.mean
    for label, value in (('c_equal_lower', .1), ('c_below_lower', .09), ('c_above_lower', .11),
        ('c_equal_upper', 15.), ('c_above_upper', 15.1), ('c_below_upper', 14.9)):
        rows.append({'case': '38_'+label, **synthetic_orders(directory, 0, {}, 1000., spread=value)})
    for label, mode, holdings in (('same_holding', 'immediate', {POOL[0]: 1000.}),
        ('switch_immediate', 'immediate', {POOL[1]: 1000.}), ('switch_delayed', 'delayed', {POOL[1]: 1000.}),
        ('switch_partial', 'partial', {POOL[1]: 1000.})):
        rows.append({'case': '38_'+label, **synthetic_orders(directory, 0, holdings, 100., mode=mode)})
    rows.append({'case': '38_slope_below', **synthetic_orders(directory, 0, {POOL[0]: 1000.}, 100., slope=.9)})
    ns = namespace(directory, 0); windows = []
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'low': np.arange(250), 'high': np.arange(250)})
    ns['np'] = SimpleNamespace(polyfit=lambda x, y, n: (windows.append([int(x.iloc[0]), int(x.iloc[-1])]) or [float(x.iloc[0]), 0.]))
    result = float(ns['countBeta'](None, POOL[0]))
    rows.append({'case': '38_beta_windows', 'windows': windows, 'value': result, 'synthetic_slope': True})
    for mode in ('immediate', 'delayed', 'partial'):
        rows.append({'case': '90_sell_cash_'+mode, **synthetic_orders(directory, 1, {'511010.XSHG': 10000.}, 0., mode=mode)})
        rows.append({'case': '90_empty_'+mode, **synthetic_orders(directory, 1, {}, 10000., mode=mode)})
    # Other funds are exactly on target; only510880 may generate an order.
    for label, current, price, cash in (('deviation_equal', 680., 1., 1000.), ('deviation_above', 679., 1., 1000.),
        ('shares_equal', 600., 2., 1000.), ('shares_above', 599., 2., 1000.),
        ('cash_equal', 600., 1., 100.), ('cash_above', 600., 1., 101.)):
        holdings = {s: 10000*w for s, w in WEIGHTS.items()}; holdings['510880.XSHG'] = current
        # Freeze total10000 independently of supplied holdings/cash to isolate original boundaries.
        ns = namespace(directory, 1); orders = []
        ctx = SimpleNamespace(portfolio=SimpleNamespace(total_value=10000., cash=cash,
            positions={s: SimpleNamespace(value=v) for s, v in holdings.items()}))
        ns['get_current_data'] = lambda: {s: SimpleNamespace(last_price=price if s == '510880.XSHG' else 1.) for s in WEIGHTS}
        ns['order_target_value'] = lambda s, v: orders.append([s, v]); ns['trade'](ctx)
        rows.append({'case': '90_'+label, 'orders': orders, 'synthetic_state_only': True})
    rows.append({'case': '90_zero_price', **synthetic_orders(directory, 1, {}, 10000., prices=dict.fromkeys(WEIGHTS, 0.))})
    rows.append({'case': '90_missing_price', **synthetic_orders(directory, 1, {}, 10000., prices={'511010.XSHG': 1.})})
    return {'cases': rows, 'not_a_backtest': True, 'injection_policy': 'Explicit math and NumPy population mean/std; original platform equivalence unproved'}


def frames(directory):
    result = {}; profiles = []
    require(reused_inputs() == read(directory/'reused-inputs.json'), 'Input registry changed')
    for stock, row in read(directory/'reused-inputs.json')['inputs'].items():
        frame = pd.read_parquet(row['file']).copy(); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.notna().all() and not frame.date.duplicated().any(), 'Invalid native dates')
        for field in ('open', 'close', 'low', 'high'): frame[field] = pd.to_numeric(frame[field], errors='raise')
        a = frame[['open', 'high', 'low', 'close']].to_numpy(float)
        bad = (~np.isfinite(a).all(axis=1)) | (a <= 0).any(axis=1) | (a[:, 1] < a[:, 2]) | (a[:, 0] < a[:, 2]) | (a[:, 0] > a[:, 1]) | (a[:, 3] < a[:, 2]) | (a[:, 3] > a[:, 1])
        ordered = frame.sort_values('date'); ratio = ordered.close/ordered.close.shift(1)
        jumps = ordered.loc[(ratio-1).abs() > .3, ['date', 'close']]
        profiles.append({'instrument': stock, 'rows': len(frame), 'first': str(frame.date.min().date()), 'last': str(frame.date.max().date()),
            'post_cutoff_rows': int((frame.date > pd.Timestamp('2026-09-29')).sum()), 'strict_usable': False,
            'invalid_ohlc_dates': [str(v.date()) for v in frame.loc[bad, 'date']],
            'raw_close_jumps_over30pct': [{'date': str(row.date.date()), 'close': float(row.close),
                'ratio_to_previous_native_row': float(ratio.loc[i])} for i, row in jumps.iterrows()],
            'jump_policy': 'No event/adjustment inference, no smoothing or correction'})
        frame = frame.loc[frame.date <= pd.Timestamp('2026-09-29')].sort_values('date').set_index('date')
        result[stock] = frame
    return result, profiles


def coverage(frame):
    return {'rows': len(frame), 'first_history_end': frame.history_end.min() if len(frame) else None,
        'last_history_end': frame.history_end.max() if len(frame) else None,
        'years': {str(year): len(group) for year, group in frame.groupby(frame.history_end.str[:4])} if len(frame) else {}}


def components(directory):
    validate_sources(directory); native, profiles = frames(directory); ns = namespace(directory, 0)
    scores = []; betas = []; ranks = []; blocked = []; by_stock = {}; beta_by_stock = {}; maxima = defaultdict(float)
    # Single-fund original get_rank computes the identical loop; full four-fund calls follow below.
    for stock in POOL:
        frame = native[stock]; by_stock[stock] = {}; beta_by_stock[stock] = {}; slopes20 = {}; independent20 = {}
        for k in range(17, len(frame)):
            # The historical20-window cache does not depend on the current18-window gate.
            if k >= 19:
                window20 = frame.iloc[k-19:k+1]; a = window20[['low', 'high']].to_numpy(float)
                if np.isfinite(a).all() and (a > 0).all() and np.ptp(a[:, 0]) > 0:
                    slopes20[k] = float(np.polyfit(window20.low, window20.high, 1)[0])
                    independent20[k] = previous.independent_slope(window20.low.to_numpy(float), window20.high.to_numpy(float))
                else: blocked.append([stock, str(frame.index[k].date()), 'invalid_or_constant_low20'])
            short = frame.iloc[k-17:k+1]; prices = short[['low', 'high']].to_numpy(float)
            if not np.isfinite(prices).all() or not (prices > 0).all() or np.ptp(prices[:, 0]) == 0:
                blocked.append([stock, str(frame.index[k].date()), 'invalid_or_constant_low18']); continue
            slope18 = float(np.polyfit(short.low, short.high, 1)[0])
            other18 = previous.independent_slope(short.low.to_numpy(float), short.high.to_numpy(float))
            maxima['slope18'] = max(maxima['slope18'], previous.close_enough(slope18, other18, 'slope18'))
            if k >= 249:
                history = frame.iloc[k-249:k+1]; indices = range(k-230, k-1)
                if all(i in slopes20 for i in indices):
                    values = [slopes20[i] for i in indices]; other = [independent20[i] for i in indices]
                    actual = float(np.mean(values)-2*np.std(values)); mean = math.fsum(other)/len(other)
                    independent = mean-2*math.sqrt(math.fsum((v-mean)**2 for v in other)/len(other))
                    maxima['beta'] = max(maxima['beta'], previous.close_enough(actual, independent, 'beta'))
                    require((slope18 > actual) == (other18 > independent), 'Independent beta decision differs')
                    beta_by_stock[stock][frame.index[k]] = (actual, independent, slope18, other18, history)
                    betas.append({'instrument': stock, 'history_end': str(frame.index[k].date()), 'beta': actual,
                        'independent_beta': independent, 'slope18': slope18, 'independent_slope18': other18,
                        'pass': slope18 > actual, 'windows': len(values), 'last_beta_window_end': str(frame.index[k-2].date())})
        for k in range(199, len(frame)):
            history = frame.iloc[k-199:k+1]; p = history.close.to_numpy(float)
            if not np.isfinite(p).all() or not (p > 0).all() or np.ptp(p) == 0 or np.ptp(p[-25:]) == 0:
                blocked.append([stock, str(frame.index[k].date()), 'invalid_or_constant_close']); continue
            ns['attribute_history'] = lambda s, width, *a, f=history, **kw: f.iloc[-width:][['close']].copy()
            actual = float(ns['get_rank']([stock]).iloc[0].score)
            short = previous.previous.independent(p[-25:], False)[0]; long = previous.previous.independent(p, False)[0]
            independent = short-long/6
            maxima['score'] = max(maxima['score'], previous.close_enough(actual, independent, 'reversal_score'))
            by_stock[stock][frame.index[k]] = (actual, independent, history)
            scores.append({'instrument': stock, 'history_end': str(frame.index[k].date()), 'score': actual,
                'independent_score': independent, 'short_score': short, 'long_score': long})
    common = sorted(set.intersection(*(set(by_stock[s]) for s in POOL)))
    full_beta_calls = 0
    for date in common:
        items = [by_stock[s][date] for s in POOL]
        if any(tuple(v[2].index) != tuple(items[0][2].index) for v in items[1:]):
            blocked.append(['ALL', str(date.date()), 'unequal_native_200_dates']); continue
        ns['attribute_history'] = lambda s, width, *a, **kw: by_stock[s][date][2].iloc[-width:][['close']].copy()
        original = ns['get_rank'](list(POOL)); rank = list(original.index)
        independent_rank = sorted(POOL, key=lambda s: by_stock[s][date][1], reverse=True)
        require(rank == independent_rank, 'Independent full rank differs')
        c = float(original.score.max()-original.score.min()); other_c = max(v[1] for v in items)-min(v[1] for v in items)
        eligible = .1 < c < 15; require(eligible == (.1 < other_c < 15), 'Independent spread gate differs')
        target = None; reason = 'spread_rejected'; full_beta = False
        if eligible:
            winner = rank[0]; operand = beta_by_stock[winner].get(date)
            if operand is None:
                blocked.append([winner, str(date.date()), 'missing_or_invalid_beta250']); reason = 'beta_blocked'
            else:
                actual, independent, slope, other_slope, history = operand
                ns['attribute_history'] = lambda s, width, *a, f=history, **kw: f.iloc[-width:][['high', 'low']].copy()
                beta = float(ns['countBeta'](None, winner)); full_beta_calls += 1; full_beta = True
                previous.close_enough(beta, actual, 'original_countBeta'); previous.close_enough(beta, independent, 'independent_countBeta')
                passed = slope > beta; require(passed == (other_slope > independent), 'Original RSRS gate differs')
                target = winner if passed else None; reason = 'winner_passed' if passed else 'slope_rejected'
        ranks.append({'history_end': str(date.date()), 'rank': '|'.join(rank), 'independent_rank': '|'.join(independent_rank),
            'spread': c, 'independent_spread': other_c, 'eligible': eligible, 'target': target, 'reason': reason,
            'original_full_rank_executed': True, 'original_beta_with_injected_mean_std': full_beta,
            'not_a_portfolio_state': True})
    fixed = []
    shared = sorted(set.intersection(*(set(native[s].index) for s in WEIGHTS)))
    for date in shared:
        prices = {s: float(native[s].loc[date, 'close']) for s in WEIGHTS}
        if not all(math.isfinite(v) and v > 0 for v in prices.values()):
            blocked.append(['90_ALL', str(date.date()), 'invalid_close_reference']); continue
        for stock, weight in WEIGHTS.items():
            fixed.append({'history_end': str(date.date()), 'instrument': stock, 'close_reference': prices[stock],
                'weight': weight, 'shares_per_10000_target_reference': 10000*weight/prices[stock],
                'original9_35_price': False, 'not_a_portfolio_state': True})
    tables = (pd.DataFrame(scores), pd.DataFrame(betas), pd.DataFrame(ranks), pd.DataFrame(fixed))
    summary = {'scores': coverage(tables[0]), 'beta': coverage(tables[1]), 'ranks': coverage(tables[2]),
        'fixed_weight_daily_reference': coverage(tables[3]), 'profiles': profiles, 'blocked': blocked,
        'original_full_rank_calls': len(ranks), 'original_injected_beta_calls': full_beta_calls,
        'rank_differences': 0, 'spread_differences': 0, 'beta_decision_differences': 0,
        'max_abs_difference': dict(maxima), 'target_reference_counts': tables[2].reason.value_counts().to_dict(),
        'not_a_backtest': True, 'new_trading_backtests': 0, 'original_input_rows_admitted': 0,
        'limits': ['Explicit math/NumPy mean/std injection; platform equivalent functions unproved',
            'Daily close targets are unit notional references;90 original9:35 order trajectories0',
            'Raw native jumps retained without event proof; no synthetic NAV or strategy returns']}
    return tables, summary


def study(root, directory):
    outputs = ('diagnostics.json', 'component-research.json', *TABLES)
    require(all(not (directory/n).exists() for n in outputs), 'Study archive exists')
    binding(root, directory); before = implementation(); diag = diagnostics(directory); tables, summary = components(directory)
    require(before == implementation(), 'Implementation changed during study')
    for name, table in zip(TABLES, tables, strict=True):
        with (directory/name).open('xb') as stream: table.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', {**summary, 'implementation_sha256': before})
    return archive(root, 'Batch80 longest reversal/beta and fixed-weight daily references archived; trading inputs not admitted')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch80 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); tables, summary = components(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    for name, table in zip(TABLES, tables, strict=True): pd.testing.assert_frame_equal(table, pd.read_parquet(directory/name), check_exact=True)
    require(before == implementation() and diag == read(directory/'diagnostics.json') and
        {**summary, 'implementation_sha256': before} == read(directory/'component-research.json') and
        dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch80 forbidden-network diagnostics/four reference tables/inventory/three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch80.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch80_research.py', 'tests/unit/test_batch79_research.py',
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
    require(summary['rank_differences'] == summary['spread_differences'] == summary['beta_decision_differences'] == 0 and
        summary['original_full_rank_calls'] == summary['ranks']['rows'] > 0 and summary['original_injected_beta_calls'] > 0, 'Independent ranking/beta gate differs')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch80 reversal/fixed-weight source and longest reference research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch80-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'diagnostics': read(directory/'diagnostics.json'), 'components': summary, 'inventory': read(directory/'dependency-inventory.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True,
        'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
