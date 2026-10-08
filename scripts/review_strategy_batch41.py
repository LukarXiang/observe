"""Freeze ETF stop/weighted momentum/opening rules and historical finance API evidence."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import io
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

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.review_strategy_batch32 import offline_catalog, source_tree, validate_catalog
from scripts.review_strategy_batch39 import inputs
from scripts import review_strategy_batch40 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch40/20261006-futures-mixed-grid')
RECEIPT = Path('docs/handoff/2026-10-06-batch40-verification.json')
SOURCES = ('聚宽2025年精选/22带上止损的核心资产轮动才更安心，低回撤，高收益率.txt',
    '聚宽2025年精选/69“稳定摸狗策略”学习笔记.txt',
    '2024年度精选策略2/83.开盘幅度决定日内方向.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/ols.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'),
    ('repo/akshare', 'akshare/index/index_stock_zh.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch41.py', 'tests/unit/test_batch41_research.py'}))
CORE = previous.CORE | {'catalog-before.json', 'strategy_catalog.before.py', 'red-regression.json', 'api-scan.json', 'dependency-inventory.json'}
QUERIES = {
    'fund_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '510180', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'One fund raw daily sample does not prove all four funds, previous dynamic pre-adjustment versions or events/states'},
    'safe_fund_daily': {'function': 'fund_etf_hist_em', 'parameters': {'symbol': '511880', 'period': 'daily',
        'start_date': '20050101', 'end_date': '20260929', 'adjust': ''},
        'limit': 'Safe fund sample is not the four ranked funds, historical fund execution rules or event ledger'},
    'index_open': {'function': 'stock_zh_index_daily_em', 'parameters': {'symbol': 'sz399303',
        'start_date': '20050101', 'end_date': '20260929'},
        'limit': 'Daily index OHLC does not prove two original09:30 current-data arrival semantics or fund fills'}}
EXPECTED_SOURCE_SHA = ('c16bb1d73b4d7f6aa007272426bca766b828452d410890dfd74497a3f4299dc3',
    'aa703d37a720815758f28a581a1b28ae312a502153f6232af907cd79e17f9790',
    '4e1a8afa7de836dce45d54dbe552c5d7347ca028eed7740a73d903434a305aa6')


def implementation(): return {name: file_sha(name) for name in FILES}


def preflight(root, directory):
    checkpoint(root, directory, 'Batch41 historical finance API and three ETF source reviews started')
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_dependencies.py', '--tb=no']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
        'code_sha256': file_sha(directory / 'strategy_catalog.before.py'),
        'test_sha256': file_sha('tests/unit/test_catalog_dependencies.py')})
    require(result.returncode == 1 and '4 failed, 25 passed' in result.stdout, 'Expected historical API regression differs')
    return {'status': 'ok', 'regression': '4 failed, 25 passed'}



def validate_scan(directory):
    before = read(directory / 'catalog-before.json')
    old = Path(before['directory']) / 'catalog.json'
    red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '4 failed, 25 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'strategy_catalog.before.py') and
        red['test_sha256'] == file_sha('tests/unit/test_catalog_dependencies.py'), 'Red regression binding differs')
    scan_path = directory / 'api-scan.json'
    if scan_path.exists():
        doc = read(scan_path); path = Path(doc['catalog']['output']) / 'catalog.json'
        require(doc['old_catalog_file'] == str(old) and file_sha(old) == doc['old_catalog_sha256'] and
            file_sha(path) == doc['catalog_sha256'], 'Frozen scan catalogs changed')
        prior = {r['path']: r for r in read(old)}; rows = read(path); changes = []
        require(len(prior) == len(rows) == 695 and {r['path'] for r in rows} == set(prior), 'Frozen scan source set differs')
        for row in rows:
            previous_row = prior[row['path']]
            require(all(row[k] == v for k, v in previous_row.items() if k not in ('apis', 'gaps', 'status')), 'Frozen scan source/manual metadata differs')
            delta = {k: {'before': previous_row[k], 'after': row[k]} for k in ('apis', 'gaps', 'status') if row[k] != previous_row[k]}
            if delta: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
        require(changes == doc['changes'] and doc['not_a_backtest'] is True, 'Frozen scan changes differ')
    return old


def scan(root, directory):
    old_path = validate_scan(directory); old = {r['path']: r for r in read(old_path)}
    result = catalog_strategies(root, 'repo/量化策略源代码'); path = Path(result['output']) / 'catalog.json'
    rows = read(path); changes = []
    require(len(rows) == len(old) == 695 and {r['path'] for r in rows} == set(old), 'Source set differs')
    for row in rows:
        prior = old[row['path']]
        require(all(row[k] == v for k, v in prior.items() if k not in ('apis', 'gaps', 'status')), 'Scan changed source/manual metadata')
        delta = {k: {'before': prior[k], 'after': row[k]} for k in ('apis', 'gaps', 'status') if row[k] != prior[k]}
        if delta: changes.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
    save(directory / 'api-scan.json', {'catalog': result, 'old_catalog_file': str(old_path), 'old_catalog_sha256': file_sha(old_path),
        'catalog_sha256': file_sha(path), 'changes': changes, 'not_a_backtest': True,
        'limits': ['Static calls can be in comments, strings or inactive functions; not proof of active financial signals']})
    return {'changed_sources': len(changes), 'catalog_id': result['catalog_id'],
        'progress': archive(root, 'Batch41 historical finance API correction across695 sources frozen')}


def start(root, directory):
    validate_scan(directory); bound = binding(root); apis = api_evidence()
    scan_doc = read(directory / 'api-scan.json')
    old_path = Path(scan_doc['catalog']['output']) / 'catalog.json'
    require(file_sha(old_path) == scan_doc['catalog_sha256'], 'API scan changed')
    save(directory / 'input-binding.json', bound); save(directory / 'existing-apis.json', apis)
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name, sha in zip(SOURCES, EXPECTED_SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path) == sha, 'Source changed/reviewed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only OLS/fund/index interface comparison; original kernels and unique ledger retained'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path),
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch41 three full ETF sources frozen; missing fund and opening dependencies retained')


def inventory(root, directory):
    store = Store(root); state = store.state(SNAPSHOT)
    funds = ['518880.SH', '513100.SH', '159915.SZ', '510180.SH', '511880.SH']
    indices = ['399300.SZ', '399303.SZ']; requested = funds + indices; rows = {}
    for name in ('instruments', 'minute_universe', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'adj_coverage'):
        frame = store.load_state(state, name, filters=[('instrument', 'in', requested)])
        rows[name] = {s: int(frame.instrument.eq(s).sum()) for s in requested}
    frame = store.load_state(state, 'index_1d', filters=[('index', 'in', indices)])
    rows['index_1d'] = {s: int(frame['index'].eq(s).sum()) for s in indices}
    return {'snapshot': SNAPSHOT, 'requested_instruments': requested, 'table_rows': rows,
        'not_a_backtest': True, 'limits': ['000300 daily math sample is not the original two indices or four ranked funds',
            'Raw daily prices alone do not prove historical pre-adjustment cache, company actions or09:30 states']}


def score_kernel(directory):
    fn = next(n for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name == 'get_rank')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    nodes = loop.body[1:-1]
    require(len(nodes) == 6 and all(isinstance(n, ast.Assign) for n in nodes), 'Original score structure differs')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(module, '<original-unweighted-score-only>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def original_scores(directory, values, kernel=None):
    x = np.asarray(values, dtype=float)
    require(len(x) == 25 and np.isfinite(x).all() and min(x) > 0, 'Invalid25-row score operands')
    code, _ = kernel or score_kernel(directory)
    ns = {'df': pd.DataFrame({'close': x}), 'np': np, 'math': math}
    exec(code, ns)
    weighted = {'g': SimpleNamespace(m_days=25), 'np': np, 'math': math,
        'attribute_history': lambda *a, **kw: pd.DataFrame({'close': x})}
    selected(directory, 1, ('MOM',), weighted)
    return [float(ns['score']), float(weighted['MOM']('arithmetic-sample'))]


def reference_scores(values):
    y = [math.log(float(v)) for v in values]; n = len(y); x = list(range(n)); out = []
    require(n == 25, 'Reference requires25rows')
    for weighted in (False, True):
        w = [1 + k / (n - 1) if weighted else 1. for k in x]
        q = [v*v for v in w]; total = math.fsum(q)
        mx = math.fsum(a*b for a, b in zip(q, x, strict=True)) / total
        my = math.fsum(a*b for a, b in zip(q, y, strict=True)) / total
        slope = math.fsum(a*(b-mx)*(c-my) for a, b, c in zip(q, x, y, strict=True)) / math.fsum(a*(b-mx)**2 for a, b in zip(q, x, strict=True))
        intercept = my - slope * mx; mean = math.fsum(y)/n
        residual = math.fsum(a*(c-slope*b-intercept)**2 for a, b, c in zip(w, x, y, strict=True))
        denominator = math.fsum(a*(c-mean)**2 for a, c in zip(w, y, strict=True))
        out.append((math.exp(slope)**250 - 1) * (1-residual/denominator))
    return out


def compute(directory):
    validate_sources(directory); frame, _ = inputs(directory); code = score_kernel(directory)
    require(len(frame) >= 25, 'Incomplete score history')
    values = frame.close.to_numpy(dtype=float); maxima = [0., 0.]; boundaries = []; examples = []
    for k in range(24, len(frame)):
        sample = values[k-24:k+1]; actual = original_scores(directory, sample, code); expected = reference_scores(sample)
        require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Nonfinite real score sample')
        for j, (a, b) in enumerate(zip(actual, expected, strict=True)):
            maxima[j] = max(maxima[j], abs(a-b))
            require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), 'Score arithmetic differs')
        if (0 < actual[1] <= 5) != (0 < expected[1] <= 5):
            boundaries.append({'date': str(frame.date.iloc[k]), 'original': actual[1], 'reference': expected[1]})
        if k == 24 or k == len(frame)-1: examples.append({'date': str(frame.date.iloc[k]), 'scores': actual, 'reference': expected})
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False, 'instrument': '000300.SH',
        'windows_per_formula': len(frame)-24, 'first': str(frame.date.iloc[24]), 'last': str(frame.date.iloc[-1]),
        'max_score_differences': maxima, 'weighted_score_boundaries': boundaries, 'examples': examples,
        'source_sha256': [r['source_sha256'] for r in read(directory / 'source-reviews/review.json')['sources']],
        'unweighted_ast_sha256': code[1],
        'reference': 'Centered fsum regression: fitting w squared, residual and denominator w once, unweighted y mean',
        'limits': ['Daily-index arithmetic only; no ETF ranking, order fills, NAV or cost performance computed']}


def diagnostic_namespace(directory, number, names):
    ns = {'g': SimpleNamespace(), 'np': np, 'pd': pd, 'math': math,
        'log': SimpleNamespace(info=lambda *a: None), 'record': lambda **kw: None}
    selected(directory, number, names, ns); return ns


def diagnostics(directory):
    rows = []
    def add(name, **kw): rows.append({'case': name, **deepcopy(kw)})
    ns = diagnostic_namespace(directory, 0, ('initial_etf_info', 'update_etf_table', 'evaluate_etf_worth', 'trade'))
    g = ns['g']; g.etf_pool = ['fund']; requests_seen = []
    def history(*a, **kw):
        requests_seen.append([list(a), kw])
        value = [100.]*100 if a[1] == 100 else [50.]
        return pd.DataFrame({'close': value}, index=range(-len(value), 0))
    ns['attribute_history'] = history; g.etf_info = ns['initial_etf_info'](g.etf_pool, 100)
    ns['update_etf_table'](None)
    add('stop_cache_old_pre_scale_not_reanchored', cache_length=len(g.etf_info['fund']),
        first=g.etf_info['fund'][0], last=g.etf_info['fund'][-1], requests=requests_seen)
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': [50.]}, index=pd.to_datetime(['2020-01-02']))
    try: ns['update_etf_table'](None)
    except KeyError as exc: add('stop_update_date_integer_fault', error=f'{type(exc).__name__}: {exc}')
    rank_ns = diagnostic_namespace(directory, 0, ('get_rank',)); rank_ns['g'].m_days = 25
    rank_ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(100., 125.)})
    records = []
    rank_ns['record'] = lambda **kw: records.append({key: {'type': type(value).__name__, 'values': value.to_dict()} for key, value in kw.items()})
    rank_ns['get_rank'](['518880.XSHG', '513100.XSHG', '159915.XSHE', '510180.XSHG'])
    add('stop_record_passes_series_to_platform', records=records)
    for last in (96., 96.00000001, 95.):
        g.etf_info['fund'] = [100.]*99+[last]
        add('stop_yesterday_boundary_'+str(last), last=last, worth=int(ns['evaluate_etf_worth']('fund', 1)))
    g.etf_info['fund'] = [100.]*100
    add('stop_recovery_mean_equality', worth=int(ns['evaluate_etf_worth']('fund', 0)))
    calls = []; ns['order_target_value'] = lambda *a: calls.append(list(a))
    for name, holdings, target, pre, selected_worth, held_worth in (
        ('stop_same_selected_no_same_call_rebuy', ['old'], 'old', None, 1, 0),
        ('stop_empty_below_mean_no_restore', [], 'old', 'old', 0, 1),
        ('stop_empty_mean_restore', [], 'old', 'old', 1, 1),
        ('stop_switch_ignores_selected_mean_gate', ['old'], 'new', None, 0, 1),
        ('stop_only_first_holding_sold', ['old1', 'old2'], 'new', None, 0, 1)):
        calls.clear(); g.etf_pre = pre; ns['get_rank'] = lambda *a, t=target: [t]
        ns['evaluate_etf_worth'] = lambda *a, s=selected_worth, h=held_worth: h if a[1] else s
        ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=dict.fromkeys(holdings), available_cash=100.))
        ns['trade'](ctx); add(name, orders=calls, previous_fund=g.etf_pre, pre_call_holdings=holdings,
            limit='Order stub records intentions; cash/holdings not mutated, no fills simulated')
    ns = diagnostic_namespace(directory, 1, ('get_rank', 'trade'))
    g = ns['g']; g.etf_pool = ['negative', 'zero', 'middle', 'five', 'above']; g.m_days = 25
    scores = dict(zip(g.etf_pool, [-1., 0., 1., 5., 5.000001], strict=True)); ns['MOM'] = lambda s: scores[s]
    add('weighted_rank_strict_lower_inclusive_upper', ranks=ns['get_rank'](g.etf_pool), scores=scores)
    ns['MOM'] = lambda s: -1.; add('weighted_all_negative_safe_fund', ranks=ns['get_rank'](g.etf_pool))
    calls = []; ns['order_target_value'] = lambda *a: calls.append(list(a)); ns['get_rank'] = lambda *a: ['target']
    for name, holdings in (('weighted_held_target_not_rebalanced', {'target': SimpleNamespace(total_amount=100)}),
        ('weighted_rejected_sell_blocks_new_buy', {'old': SimpleNamespace(total_amount=100)})):
        calls.clear(); ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=holdings, available_cash=100.))
        ns['trade'](ctx); add(name, orders=calls, holdings=list(holdings), limit='Synthetic rejected/nonmutating order stub')
    ns = diagnostic_namespace(directory, 1, ('MOM',)); ns.pop('math')
    ns['g'].m_days = 25; ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(100., 125.)})
    try: ns['MOM']('fund')
    except NameError as exc: add('weighted_math_requires_platform_injection', error=f'{type(exc).__name__}: {exc}')
    ns['math'] = math; ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.ones(25)})
    with np.errstate(invalid='ignore', divide='ignore'):
        score = ns['MOM']('fund')
    add('weighted_constant_score_nonfinite_excluded', finite=bool(np.isfinite(score)), eligible=bool(0 < score <= 5))
    ns = diagnostic_namespace(directory, 2, ('market_open',)); ns['g'].security = '510180.XSHG'
    calls = []; requests_seen = []
    ns['order_value'] = lambda *a, **kw: calls.append({'args': list(a), 'kwargs': kw})
    ns['order_target_value'] = lambda *a: calls.append({'args': list(a), 'kwargs': {}})
    for name, a, b, held in (('open_equal_one_empty_no_buy', 1., 1., False),
        ('open_equal_above_one_buy', 1.01, 1.01, False),
        ('open_beats_other_below_one_buy', .99, .98, False),
        ('open_below_other_below_one_sell', .98, .99, True),
        ('open_equal_below_one_held_keep', .99, .99, True),
        ('open_above_other_held_keep', 1.01, 1., True)):
        calls.clear(); requests_seen.clear()
        def opening_history(*args, **kw):
            requests_seen.append([list(args), kw]); return pd.DataFrame({'close': [100.]})
        ns['attribute_history'] = opening_history
        ns['get_current_data'] = lambda: {'399300.XSHE': SimpleNamespace(day_open=100*a),
            '399303.XSHE': SimpleNamespace(day_open=100*b), '510180.XSHG': SimpleNamespace(day_open=1.)}
        ctx = SimpleNamespace(current_dt=datetime(2020, 1, 2, 9, 30),
            portfolio=SimpleNamespace(available_cash=100., long_positions={'fund': object()} if held else {}))
        ns['market_open'](ctx); add(name, ratios=[a,b], held=held, orders=calls, requests=requests_seen)
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': [100.]}, index=pd.to_datetime(['2020-01-01']))
    try: ns['market_open'](ctx)
    except KeyError as exc: add('open_original_date_integer_fault', error=f'{type(exc).__name__}: {exc}')
    return {'not_a_backtest': True, 'platform_equivalent': False, 'cases': rows,
        'limits': ['Original functions only, explicit sample labels and injected math; not a compatibility-fixed trading variant',
            'Synthetic opening ratios/order intentions are not historical prices, fills or cash-ledger evidence']}


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch41.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch41_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch40_research.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def binding(root, directory=None):
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch40 differs')
    rows = {}
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = ACCEPTED / name; require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Accepted input changed')
        rows[name] = {'file': str(path), 'sha256': file_sha(path)}
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'upstream': previous.binding(root, ACCEPTED), 'files': rows, 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding differs')
    return result

def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, query in QUERIES.items():
        fn = getattr(ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'queries': QUERIES, 'apis': rows, 'numpy': np.__version__, 'pandas': pd.__version__}

def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')

def prepare(root, directory):
    bound = binding(root, directory); validate_sources(directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input archive exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame.columns), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'platform_equivalent': False,
        'profiles': profiles, 'component_instrument': '000300.SH',
        'limits': ['Daily-index25-row arithmetic samples only; not original fund ranking or opening-ratio inputs',
            'No fund cache or platform opening feed reconstructed; no fills, costs or NAV simulated']})
    save(directory / 'dependency-inventory.json', inventory(root, directory))
    return archive(root, 'Batch41 actual fund/two-index inventory and accepted arithmetic operands frozen')

def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory)
    with redirect_stdout(io.StringIO()): diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch41 original ETF scores and stop/weighted/open state diagnoses frozen')

def worker(root, directory, endpoint):
    import akshare as ak
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe API differs')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False); socket.setdefaulttimeout(8)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'failed', 'published': False, 'files': [], 'wire': []}
    original = requests.sessions.Session.request
    def request(session, method, url, **kw):
        require(len(row['wire']) < 10, 'Response cap exceeded'); session.trust_env = False; kw['timeout'] = (8, 10)
        response = original(session, method, url, **kw); path = folder / f'response-{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status': response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()): frame = getattr(ak, query['function'])(**query['parameters'])
        require(0 < len(frame) <= 100000, 'Empty/excessive provider sample'); path = raw.save(root, 'batch41_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}], limit=query['limit'], strict_usable=False)
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
        command = [sys.executable, '-m', 'scripts.review_strategy_batch41', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch41 existing fund/index supplementation attempts frozen; no publication')

def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise AssertionError('Batch41 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        result = compute(directory); diagnostic = diagnostics(directory); catalog = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline components differ')
    save(directory / 'offline-catalog.json', catalog); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    validate_catalog(root, directory); return archive(root, 'Batch41 forbidden-network arithmetic/diagnostic/catalog byte match')

def checks(root, directory):
    before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked code changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True); require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
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
    binding(root, directory); validate_catalog(root, directory); validate_sources(directory); validate_scan(directory)
    require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    with redirect_stdout(io.StringIO()): require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch41 ETF stop/weighted/open arithmetic accepted; original trade dependencies remain missing')
    save(Path('docs/handoff/2026-10-06-batch41-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'),
        'protection': protection, 'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'scan', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch41/20261006-etf-stop-weighted-open')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
