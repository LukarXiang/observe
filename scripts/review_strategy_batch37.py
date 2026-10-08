"""Freeze RPS/LOF/chase rules and verify available operators, never trading NAV."""
import argparse
import ast
from contextlib import redirect_stdout
from decimal import Decimal
import hashlib
import inspect
import io
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

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch32 as prices
from scripts.review_strategy_batch32 import source_tree
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = prices.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch32/20261006-bluechip-trix-boll')
RECEIPT = Path('docs/handoff/2026-10-06-batch32-verification.json')
LATEST_RECEIPT = Path('docs/handoff/2026-10-06-batch36-verification.json')
SOURCES = ('2021年度精选策略/91.欧奈尔 RPS选股战法回测.txt',
    '2021年度精选策略/92.敢于直接实盘的—八仙过海V2.txt',
    '2022年度精选策略/60.2019年3月以来的前日涨停昨日未涨停今日大涨的追涨策略.txt')
FILES = tuple(sorted(set(prices.FILES) | {'scripts/review_strategy_batch37.py', 'tests/unit/test_batch37_research.py'}))
REFERENCES = (('repo/backtrader', 'backtrader/indicators/percentchange.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_sina.py'))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'existing-apis.json',
    'price-input.parquet', 'input-analysis.json', 'component-research.json', 'diagnostics.json',
    'probe-results.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json',
    'source-reviews/review.corrected.json', 'source-clarification.json'}
QUERIES = {
    'lof161903': {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sz161903'},
        'limit': 'Single LOF daily prices do not prove eight-fund events, statuses or14:30 execution'},
    'minute600519': {'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600519', 'period': '1',
        'start_date': '2021-01-04 09:30:00', 'end_date': '2021-01-04 15:00:00', 'adjust': ''},
        'limit': 'Recent minute endpoint cannot establish whole-pool historical09:34/09:35/14:25 prices'},
    'pool000002': {'function': 'index_stock_cons_csindex', 'parameters': {'symbol': '000002'},
        'limit': 'Current Shanghai A-share membership is not dated platform000002 plus399106 pools'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); latest = read(LATEST_RECEIPT)
    require(receipt['status'] == latest['status'] == 'ok' and receipt['snapshot'] == latest['snapshot'] == SNAPSHOT, 'Accepted receipt differs')
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Published snapshot differs')
    rows = {}
    for name in ('price-input.parquet', 'input-analysis.json', 'input-binding.json'):
        path = ACCEPTED / name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted price evidence changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'latest_receipt_file': str(LATEST_RECEIPT), 'latest_receipt_sha256': file_sha(LATEST_RECEIPT),
        'upstream': prices.binding(root, ACCEPTED), 'files': rows, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); source = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': source, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()})
    return {'version': ak.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    bound = binding(root); apis = api_evidence()
    checkpoint(root, directory, 'Batch37 RPS/LOF/chase source research started')
    save(directory / 'input-binding.json', bound); save(directory / 'existing-apis.json', apis)
    old_path = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path), 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only percent-change window and LOF API comparison; not platform substitutions'})
    search = ['rg', '-n', '^def (n_day_chg_xiaoyu|situation_filter_xiaoyu_ma)', 'repo', '-g', '*.py', '-g', '*.txt']
    found = subprocess.run(search, capture_output=True, text=True); require(found.returncode in (0, 1), found.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'dependency_search': {'command': search, 'returncode': found.returncode, 'matches': found.stdout.splitlines()},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch37 three complete sources frozen; wizard/fund/intraday dependencies remain unavailable')


def reviewed(directory):
    doc = read(directory / 'source-reviews/review.corrected.json'); correction = read(directory / 'source-clarification.json')
    require(correction['original_sha256'] == file_sha(directory / 'source-reviews/review.json') and
        correction['corrected_sha256'] == file_sha(directory / 'source-reviews/review.corrected.json'), 'Review correction changed')
    original = read(directory / 'source-reviews/review.json')
    allowed = {'catalog', 'sources'}
    require(all(doc[key] == value for key, value in original.items() if key not in allowed), 'Correction changed unrelated evidence')
    for a, b in zip(original['sources'], doc['sources'], strict=True):
        require(all(a[key] == b[key] for key in a if key != 'review'), 'Correction changed source provenance')
    return doc


def clarify(root, directory):
    binding(root, directory); original = read(directory / 'source-reviews/review.json')
    require([r['review'] for r in original['sources']] != [REVIEWS[n] for n in SOURCES], 'No source review correction needed')
    doc = json.loads(json.dumps(original))
    for name, row in zip(SOURCES, doc['sources'], strict=True): row['review'] = REVIEWS[name]
    doc['catalog'] = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.corrected.json', doc)
    save(directory / 'source-clarification.json', {'original_sha256': file_sha(directory / 'source-reviews/review.json'),
        'corrected_sha256': file_sha(directory / 'source-reviews/review.corrected.json'),
        'reason': 'Initial claim that all five sequential weights fail equality1 was incorrect; original sequence produces exactly1.0 and full allocation',
        'original_strategy_changed': False, 'original_archive_preserved': True})
    validate_sources(directory)
    return archive(root, 'Batch37 corrected LOF all-five weight claim; initial review retained with explicit SHA chain')


def validate_sources(directory):
    doc = reviewed(directory)
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 3, 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory)
    path = directory / 'price-input.parquet'; require(not path.exists(), 'Input copy exists')
    shutil.copyfile(bound['files'][path.name]['file'], path); frame = pd.read_parquet(path)
    profile = read(ACCEPTED / 'input-analysis.json')
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'file': str(path), 'sha256': file_sha(path),
        'rows': len(frame), 'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max(),
        'pool': profile['pool'], 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Ten verified stocks only for arithmetic samples, not original index pools or fund inputs',
            'Explicit old positional index adapter; known suspended rows excluded, no current intraday price supplied']})
    return archive(root, 'Batch37 accepted ten-stock bytes frozen for RPS120 and three-row pattern arithmetic only')


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); path = directory / 'price-input.parquet'
    require(doc['snapshot'] == SNAPSHOT and doc['file'] == str(path) and doc['not_a_backtest'] is True and
        doc['platform_equivalent'] is False and file_sha(path) == doc['sha256'], 'Input binding differs')
    frame = pd.read_parquet(path)
    require(len(frame) == doc['rows'] and list(frame.columns) == doc['columns'] and frame.date.min() == doc['first'] and
        frame.date.max() == doc['last'] and set(frame.instrument) == set(doc['pool']) and not frame.duplicated(['date', 'instrument']).any(), 'Input profile differs')
    frame = frame[frame.is_trading].copy()
    require(frame.adjustment_status.eq('usable').all() and np.isfinite(frame[['close_adj', 'back_factor']]).all().all() and
        frame[['close_adj', 'back_factor']].gt(0).all().all(), 'Unknown price operands')
    return frame.sort_values(['instrument', 'date']).reset_index(drop=True), doc['rows'] - len(frame)


def ratio_expression(directory):
    tree = source_tree(directory, 0)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rps_select')
    expr = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'increase' for t in n.targets))
    return compile(ast.Expression(expr), '<original-rps-ratio>', 'eval'), hashlib.sha256(ast.dump(expr).encode()).hexdigest()


def compute(directory):
    validate_sources(directory); frame, excluded = inputs(directory); expr, sha = ratio_expression(directory)
    ns = {'get_current_data': lambda: {}, 'get_current_data_unused': True}
    pattern_sha = selected(directory, 2, ('high_limit_filter',), ns)
    result = {'not_a_backtest': True, 'platform_equivalent': False, 'excluded_known_paused_rows': excluded,
        'source_ast_sha256': {'ratio': sha, 'pattern': pattern_sha}, 'stocks': []}
    for instrument, group in frame.groupby('instrument', sort=True):
        group = group.reset_index(drop=True); values = group.close_adj.to_numpy(); factors = group.back_factor.to_numpy()
        rps = []; events = []; boundaries = []; max_diff = 0.
        for k in range(2, len(values)):
            close = values[k-2:k+1] / factors[k]
            ns['attribute_history'] = lambda *a, close=close, **kw: pd.DataFrame({'close': close})
            actual = bool(ns['high_limit_filter'](['operator_sample']))
            first = float(Decimal(str(close[1])) / Decimal(str(close[0])) - 1)
            second = float(Decimal(str(close[2])) / Decimal(str(close[1])) - 1)
            expected = first > .095 and not second > .095
            if actual != expected: boundaries.append({'kind': 'pattern', 'date': group.date.iloc[k], 'close': close.tolist(), 'actual': actual, 'reference': expected})
            events.append(actual)
            if k >= 119:
                window = values[k-119:k+1] / factors[k]
                ratio = float(eval(expr, {'rps_data': {'close': window}}))
                reference = float(Decimal(str(window[-1])) / Decimal(str(window[0])) - 1)
                max_diff = max(max_diff, abs(ratio-reference)); rps.append(ratio)
                if (ratio > .5) != (reference > .5):
                    boundaries.append({'kind': 'rps', 'date': group.date.iloc[k], 'first_close': float(window[0]), 'last_close': float(window[-1]), 'actual': ratio, 'reference': reference})
        result['stocks'].append({'instrument': instrument, 'traded_rows': len(group), 'rps_windows': len(rps),
            'rps_first_end_date': group.date.iloc[119], 'last_end_date': group.date.iloc[-1], 'rps_max_difference': max_diff,
            'rps_sha256': hashlib.sha256(np.asarray(rps, dtype='<f8').tobytes()).hexdigest(), 'pattern_windows': len(events),
            'pattern_true_count': sum(events), 'pattern_sha256': hashlib.sha256(bytes(events)).hexdigest(), 'boundaries': boundaries})
    return result


def diagnostics(directory):
    from datetime import datetime
    cases = []; log = SimpleNamespace(info=lambda *a: None)
    g = SimpleNamespace(rps_period=120, filter_over_increase_percent=.5, min_rps=85)
    ns = {'g': g, 'get_bars': lambda stock, **kw: {'close': np.linspace(100., 140., 120)}}
    selected(directory, 0, ('rps_select', 'buy', 'sell', 'get_security_universe'), ns)
    g.watch_list = [f's{k:03d}' for k in range(100)]
    found = ns['rps_select'](None, g.watch_list)
    cases.append({'case': 'rps_stable_ties_no_ten_cap', 'selected': found, 'count': len(found), 'lowest_selected_rps': 85.})
    ns['get_bars'] = lambda stock, **kw: {'close': [100., 160. if stock == 'a' else 150.]}
    g.watch_list = ['a', 'b']; g.min_rps = 0
    cases.append({'case': 'rps_exclusion_after_rank_strict_half', 'selected': ns['rps_select'](None, g.watch_list)})
    orders = []; ns.update(log=log, order_value=lambda *a: orders.append(list(a))); g.max_stock_num = 5
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={}, available_cash=1000.))
    ns['buy'](ctx, 'a'); ctx.portfolio.positions = {'a': None}; ns['buy'](ctx, 'a'); ns['buy'](ctx, 'b')
    cases.append({'case': 'rps_slots_and_duplicate', 'orders': orders})
    try: ns['sell'](ctx, 'a')
    except NameError as exc: cases.append({'case': 'missing_wizard_sell', 'error': str(exc)})
    else: raise AssertionError('Missing wizard unexpectedly supplied')
    ns.update(get_index_stocks=lambda *a: ['b', 'a', 'a'])
    cases.append({'case': 'rps_sorted_unique_pool', 'pool': ns['get_security_universe'](None, ['000300.XSHG'], [])})

    ns = {'g': SimpleNamespace(ETFList=[], lag=60), 'pd': pd, 'get_current_data': lambda: {}}
    selected(directory, 1, ('get_signal', 'ETFtrade'), ns)
    try: ns['get_signal'](None)
    except ZeroDivisionError as exc: cases.append({'case': 'lof_empty_pool', 'error': str(exc)})
    else: raise AssertionError('Empty pool fault disappeared')
    ns['g'].ETFList = ['fund']; ns.update(attribute_history=lambda *a, **kw: {'close': np.ones(60)},
        get_current_data=lambda: {'fund': SimpleNamespace(last_price=2.)})
    try: ns['get_signal'](None)
    except AttributeError as exc: cases.append({'case': 'lof_removed_append', 'error': str(exc), 'weight_before_error': ns['g'].cang['fund'], 'sum_before_error': ns['g'].allCang})
    else: raise AssertionError('Removed append fault disappeared')
    weights = [.3, .25, .2, .15, .1]; index = 0
    for weight in weights: index += weight
    cases.append({'case': 'lof_all_five_float_branch', 'index': index, 'equal_one': index == 1, 'original_weight_one_fund': 1. if index == 1 else index*.3})
    orders = []; g = SimpleNamespace(signal='BUY', empty_keep_stock='511880.XSHG', buy=['fund'], cang={'fund': .3}, allCang=.3)
    ns.update(g=g, log=log, record=lambda **kw: None, order_target=lambda *a: orders.append(['target', *a]),
        order_target_value=lambda *a: orders.append(['value', *a]))
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={'fund': None}, available_cash=700., total_value=1000., returns=0.))
    ns['ETFtrade'](ctx); cases.append({'case': 'lof_adjust_and_unconditional_cash_target', 'orders': orders.copy()})
    g.signal = 'CLEAR'; orders.clear(); ns['ETFtrade'](ctx)
    cases.append({'case': 'lof_clear_then_cash_target', 'orders': orders.copy()})

    orders = []; calls = []; ns = {'g': SimpleNamespace(stocks=['a', 'b']), 'log': log,
        'get_current_data': lambda: {x: SimpleNamespace(last_price=106., high_limit=110.) for x in ('a', 'b')},
        'attribute_history': lambda *a, **kw: pd.DataFrame({'close': [100.]}),
        'order_value': lambda *a: orders.append(list(a))}
    selected(directory, 2, ('market_buy', 'market_sell', 'high_limit_filter', 'get_fullist'), ns)
    ctx = SimpleNamespace(current_dt=datetime(2021, 1, 4, 9, 35), portfolio=SimpleNamespace(total_value=1000., positions={}))
    ns['market_buy'](ctx); cases.append({'case': 'chase_equal_return_overwrites_first', 'orders': orders.copy()})
    orders.clear(); ns['get_current_data'] = lambda: {'a': SimpleNamespace(last_price=106., high_limit=110.), 'b': SimpleNamespace(last_price=107., high_limit=110.)}
    ns['market_buy'](ctx); cases.append({'case': 'chase_multiple_total_value_orders', 'orders': orders.copy()})
    ns['get_current_data'] = lambda: {}; ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': [100., 110., 110.]}, index=pd.date_range('2021-01-01', periods=3))
    try: ns['high_limit_filter'](['a'])
    except KeyError as exc: cases.append({'case': 'chase_datetime_integer_index_fault', 'error': str(exc)})
    else: raise AssertionError('Legacy integer index fault disappeared')
    for label, values in [('strict_first_boundary', [100., 109.5, 109.5]), ('second_equal_boundary', [100., 110., 120.45]), ('not_true_limit', [100., 109.6, 109.6])]:
        ns['attribute_history'] = lambda *a, values=values, **kw: pd.DataFrame({'close': values, 'high_limit': [110., 120.45, 130.]})
        cases.append({'case': f'chase_{label}', 'selected': ns['high_limit_filter'](['a'])})
    ns['get_index_stocks'] = lambda code: calls.append(code) or ['600001.XSHG', '300001.XSHE', '688001.XSHG']
    cases.append({'case': 'chase_pool_keeps_duplicates_and_star', 'pool': ns['get_fullist'](None), 'requests': calls})
    return {'not_a_backtest': True, 'cases': cases, 'limits': ['Synthetic requests/orders only; no fills, fees or NAV',
        'Original functions selected without platform imports; no messages or corrected trading models executed']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed')
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch37 RPS120/three-row arithmetic and original ranking/fund/chase defects frozen')


def worker(root, directory, endpoint):
    import akshare as ak
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API differs')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'),
        'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kw):
        require(len(row['wire']) < 10, 'Response cap exceeded'); session.trust_env = False; kw['timeout'] = (8, 10)
        response = original(session, method, url, **kw); path = folder / f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status': response.status_code})
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()):
            frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample')
        path = raw.save(root, 'batch37_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}],
            limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['published'] is False, 'Probe binding differs')
        for item in [*row['files'], *row['wire']]: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] == 'sample':
            require(row['strict_usable'] is False and row['limit'] == query['limit'] and len(row['files']) == 1, 'Unproven sample admitted')
            item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(0 < len(frame) == item['rows'] <= 100000 and list(frame.columns) == item['columns'], 'Sample profile differs')
        else: require(row['status'] in ('failed', 'timeout') and not row['files'] and row.get('error'), 'Failed probe admitted data')
        rows.append(row)
    return rows


def probe(root, directory):
    binding(root, directory)
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch37', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch37 LOF/minute/pool supplement attempts frozen; no publication')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch37 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result)
    require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch37 forbidden-network operator/diagnostic/catalog byte match')


def offline_catalog(root, directory):
    proof = Path(root) / 'catalog/strategies/implementation-evidence.json'
    require(file_sha(proof) == file_sha(directory / 'implementation-evidence.before.json'), 'Implementation registry changed')
    folder = directory / 'catalog-offline'; folder.mkdir(exist_ok=False)
    copied = folder / 'catalog/strategies/implementation-evidence.json'; copied.parent.mkdir(parents=True)
    shutil.copyfile(proof, copied); repeated = catalog_strategies(folder, 'repo/量化策略源代码')
    original = reviewed(directory)['catalog']; require(repeated['catalog_id'] == original['catalog_id'], 'Offline catalog identity differs')
    rows = []
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        a = Path(original['output']) / name; b = Path(repeated['output']) / name
        require(file_sha(a) == file_sha(b), 'Offline catalog bytes differ')
        rows.append({'name': name, 'original_file': str(a), 'repeated_file': str(b), 'sha256': file_sha(a)})
    return {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'catalog_id': repeated['catalog_id'],
        'files': rows, 'implementation_evidence_sha256': file_sha(proof)}


def validate_catalog(root, directory):
    doc = read(directory / 'offline-catalog.json'); original = reviewed(directory)['catalog']
    require(doc['result'] == 'match' and doc['differences'] == 0 and doc['socket_network_disabled'] is True and
        doc['catalog_id'] == original['catalog_id'] and len(doc['files']) == 3 and
        {r['name'] for r in doc['files']} == {'catalog.json', 'summary.json', 'catalog.parquet'}, 'Offline catalog scope differs')
    require(doc['implementation_evidence_sha256'] == file_sha(Path(root) / 'catalog/strategies/implementation-evidence.json') ==
        file_sha(directory / 'implementation-evidence.before.json'), 'Implementation registry differs')
    for row in doc['files']:
        a = Path(original['output']) / row['name']; b = directory / 'catalog-offline/catalog/strategies' / doc['catalog_id'] / row['name']
        require(row['original_file'] == str(a) and row['repeated_file'] == str(b) and file_sha(a) == row['sha256'] == file_sha(b), 'Offline catalog bytes/path differ')
    require(read(Path(root) / 'catalog/strategies/latest.json')['catalog_id'] == doc['catalog_id'], 'Latest catalog differs')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch37.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch37_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch32_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalog(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked code changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked code copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); validate_catalog(root, directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch37 RPS/chase arithmetic accepted; original wizard/fund/intraday dependencies remain blocked')
    save(Path('docs/handoff/2026-10-06-batch37-verification.json'), {'status': 'ok', 'reviews': reviewed(directory),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'clarify', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch37/20261006-rps-lof-chase')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
