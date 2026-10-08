"""Freeze weekly ETF rotation and volume-emotion rules without invented trades."""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
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
import talib

from observe.data import raw
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch20 import backend
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch47 import native
from scripts import review_strategy_batch50 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch50/20261007-macro-rules')
RECEIPT = Path('docs/handoff/2026-10-07-batch50-verification.json')
SOURCES = ('2021年度精选策略/22.“开弓”ETF轮动模型——改.txt',
    '2021年度精选策略/30.ETF宽基轮动修改版-1.0.txt',
    '2021年度精选策略/81.无杠杆，稳定盈利的etf轮动，06年开始3000%收益.txt')
SOURCE_SHA = ('94ebb573284feffd37e6f3eceac23eeca7e3b3773746d5b9cbe7bc7ac534aa8f',
    '6448ae64f0343d555e7fc857850e149cf1fb8b0cdf666f74920bc778ac54b427',
    '541947cc376a3275237ef4a1f18c0eb6cb1a7a7d05b8fe76c416229125b109bc')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/basicops.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_sina.py'), ('repo/akshare', 'akshare/fund/fund_etf_em.py'),
    ('repo/akshare', 'akshare/index/index_zh_em.py'))
QUERIES = {'fund_daily': ('fund_etf_hist_sina', {'symbol': 'sz159901'}),
    'fund_minute': ('fund_etf_hist_min_em', {'symbol': '159901', 'period': '5', 'adjust': '',
        'start_date': '2026-09-28 09:30:00', 'end_date': '2026-09-29 15:00:00'}),
    'emotion_index': ('index_zh_a_hist', {'symbol': '399632', 'period': 'daily', 'start_date': '20050101', 'end_date': '20260929'})}
FUNDS = tuple(sorted({'510050.SH', '510500.SH', '159901.SZ', '159902.SZ', '159915.SZ', '513100.SH', '513500.SH',
    '518880.SH', '510300.SH', '159920.SZ', '511010.SH', '510180.SH', '510880.SH', '159905.SZ', '511880.SH'}))
INDICES = ('000300.SH', '000905.SH', '399006.SZ', '399632.SZ', '000016.SH', '000010.SH', '000015.SH', '399324.SZ', '399001.SZ')
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch51.py', 'tests/unit/test_batch51_research.py', 'scripts/review_strategy_batch20.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py', 'source-reviews/review.json',
    'existing-apis.json', 'input-analysis.json', 'dependency-inventory.json', 'raw-discovery.json', 'index-input.parquet',
    'calendar-input.parquet', 'gem-raw.parquet', 'fund-raw.parquet', 'component-research.json', 'diagnostics.json',
    'probe-results.json', 'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
CORE |= {'review_strategy_batch51.initial.py', 'initial-study-failure.json', 'ast-red-regression.log'}
CORE |= {'ma-boundary-diagnosis.json', 'study-retry.log'}
ANCHOR = '2026-09-29'


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'], 'Batch50 changed')
    files = {}; old = read('docs/handoff/2026-10-07-batch47-verification.json')
    for name in ('index-input.parquet', 'calendar-input.parquet'):
        path = Path('data/staging/strategies-batch47/20261007-candidate-momentum') / name
        require(file_sha(path) == old['checks']['evidence_sha256'][name], 'Accepted operand changed')
        files[name] = {'file': str(path), 'sha256': file_sha(path)}
    index_path = Path('data/staging/strategies-batch12/20261005-daily-candidates/probes/bs_399006/result.json')
    idx = read(index_path); require(idx['status'] == 'success' and idx['strict_usable'] is False and idx['published'] is False and
        idx['parameters'] == {'code': 'sz.399006', 'start': '2005-01-05', 'end': ANCHOR, 'adjustflag': '3'} and
        idx['raw_sha256'] == file_sha(idx['raw_file']) == 'b17838d931f8c8a0ec30e51f1a2557cf0c379374514bbebc3a39103427bf64ed', 'Old GEM index changed')
    files['gem-raw.parquet'] = {'file': idx['raw_file'], 'sha256': idx['raw_sha256'], 'origin': str(index_path), 'origin_sha256': file_sha(index_path)}
    fund_path = Path('data/staging/strategies-batch20/20261005-macd-patterns/probes/etf510300/result.json')
    fund = read(fund_path); item = fund['files'][0]
    require(fund['status'] == 'success' and fund['published'] is False and len(fund['files']) == 1 and item['parameters'] == {'symbol': 'sh510300'} and
        item['sha256'] == file_sha(item['file']) == '40f3a5a24e70d1fc90517139d73982b08fdec9ee35a777a79e78ae959abb59ed', 'Old ETF operand changed')
    for row in fund['wire_responses']: require(file_sha(row['file']) == row['sha256'], 'Old ETF response changed')
    files['fund-raw.parquet'] = {'file': item['file'], 'sha256': item['sha256'], 'origin': str(fund_path), 'origin_sha256': file_sha(fund_path)}
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'files': files, 'not_a_backtest': True, 'anchor': ANCHOR}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root); checkpoint(root, directory, 'Batch51 weekly ETF/volume-emotion source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def api_evidence():
    import akshare as ak
    rows = []
    for endpoint, (name, params) in QUERIES.items():
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'endpoint': endpoint, 'function': name, 'parameters': params, 'signature': str(inspect.signature(fn)),
            'source': code, 'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'apis': rows, 'indicator': backend()}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); shutil.copyfile(path, copied); tree = ast.parse(text)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True,
            'function_ast': {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Read-only EMA seed and existing fund/index interfaces; no replacement of original scoring/state rules'})
    save(directory / 'existing-apis.json', api_evidence())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch51 three full weekly/broad ETF and volume-emotion sources frozen')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json'); require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source/rule changed')
        text, encoding = read_source(Path(row['source_copy'])); tree = ast.parse(text)
        require(row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and row['function_ast'] ==
            {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}, 'Function evidence changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for row in doc['references']: require(file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Selected source changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def selected(directory, number, names, ns):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Selected functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-ETF-function>', 'exec'), ns)


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); counts = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'fund_actions', 'fund_coverage', 'adj_factors', 'adj_coverage', 'instrument_status'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', list(FUNDS))])
        counts[table] = {s: int(frame.instrument.eq(s).sum()) if 'instrument' in frame else 0 for s in FUNDS}
    frame = store.load_state(state, 'index_1d', filters=[('index', 'in', list(INDICES))])
    return {'snapshot': SNAPSHOT, 'funds': list(FUNDS), 'indices': list(INDICES), 'fund_rows': counts,
        'index_rows': {s: int(frame['index'].eq(s).sum()) for s in INDICES}, 'registered_tables': sorted(state['tables']),
        'not_a_backtest': True, 'limits': ['Published snapshot counts only; separately bound raw files are not complete intraday/event inputs']}


def parse_daily(frame, anchor=ANCHOR):
    out = frame[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    out['date'] = pd.to_datetime(out.date, errors='raise').dt.strftime('%Y-%m-%d')
    for name in ('open', 'high', 'low', 'close', 'volume'):
        out[name] = pd.to_numeric(out[name], errors='raise')
    require(not out.date.duplicated().any() and np.isfinite(out.iloc[:, 1:].to_numpy(dtype=float)).all() and
        out[['open', 'high', 'low', 'close']].astype(float).gt(0).all().all() and out.volume.astype(float).ge(0).all(), 'Invalid daily operands')
    out = out[out.date.le(anchor)].sort_values('date').reset_index(drop=True)
    require(len(out) > 0, 'Empty daily operands')
    for name in ('open', 'high', 'low', 'close', 'volume'): out[name] = out[name].astype(float)
    return out


def prepare(root, directory):
    bound = binding(root, directory); profiles = {}
    for name, row in bound['files'].items():
        path = directory / name; require(not path.exists(), 'Input exists'); shutil.copyfile(row['file'], path)
        frame = pd.read_parquet(path); profiles[name] = {'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}
    calendar = pd.read_parquet(directory / 'calendar-input.parquet'); dates = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    open_dates = set(dates[calendar.is_open]); samples = []
    for name in ('index-input.parquet', 'gem-raw.parquet', 'fund-raw.parquet'):
        original = pd.read_parquet(directory / name); frame = parse_daily(original)
        require(set(frame.date) <= open_dates, 'Unknown calendar dates in arithmetic operands')
        expected = {d for d in open_dates if frame.date.min() <= d <= frame.date.max()}
        samples.append({'file': name, 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max(),
            'rows_after_anchor_excluded': len(original) - len(frame), 'missing_calendar_dates': sorted(expected - set(frame.date)), 'not_a_backtest': True})
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'profiles': profiles, 'samples': samples, 'anchor': ANCHOR,
        'not_a_backtest': True, 'platform_equivalent': False, 'strict_usable': False,
        'limits': ['EOD/index and unadjusted provider ETF values are arithmetic operands, not original intraday current_price/include_now bars',
            '510300 raw does not supply15 funds, original dynamic adjustment, distributions or execution; no pool substitution',
            'All old raw bytes retained; Sep30 beyond frozen anchor is explicitly excluded from arithmetic only']})
    command = ['rg', '--files', 'data/raw']; result = subprocess.run(command, capture_output=True, text=True, check=True)
    paths = [Path(p) for p in result.stdout.splitlines() if ('etf' in p or 'fund' in p) and p.endswith('.parquet')]
    discovered = []
    for path in paths:
        frame = pd.read_parquet(path); discovered.append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
    save(directory / 'raw-discovery.json', {'command': command, 'files': discovered, 'not_a_backtest': True,
        'limits': ['Path-based ETF/fund search only; empty BaoStock responses,510310 other fund and requested-date NAV lists do not fill original pool']})
    save(directory / 'dependency-inventory.json', inventory(root))
    return archive(root, 'Batch51 verified old GEM/510300 raw and missing15-fund/index dependencies frozen')


def weekly_score(values):
    values = list(map(float, values)); require(len(values) == 9 and min(values) > 0 and all(map(math.isfinite, values)), 'Invalid weekly operands')
    return math.fsum((values[-1]/values[-1-lag]-1)*weight for lag, weight in zip((1, 2, 3, 4, 8), (.4, .2, .15, .2, .05), strict=True))


def ema_reference(values, n):
    values = list(map(float, values)); require(len(values) >= n and n >= 2 and all(map(math.isfinite, values)), 'Invalid EMA operands')
    result = [math.nan]*(n-1); value = math.fsum(values[:n])/n; result.append(value); alpha = 2/(n+1)
    for current in values[n:]: value += alpha*(current-value); result.append(value)
    return result


def emotion_reference(ratios, lag=6):
    require(len(ratios) >= 30 and lag >= 1, 'Invalid emotion operands')
    for k in range(30):
        if ratios[-1] >= 0:
            if ratios[-1-k] < 0: return 1 if k >= 3 else 0
        elif ratios[-1-k] >= 0: return -1 if k >= lag else 0
    return None


def kernels(directory):
    tree = source_tree(directory, 0); weekly_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'chenk_stocks')
    loop = next(n for n in weekly_fn.body if isinstance(n, ast.For)); branch = next(n for n in loop.body if isinstance(n, ast.If))
    weekly = branch.body[:-1]; require([n.targets[0].id for n in weekly] == ['wzf1', 'wzf2', 'wzf3', 'wzf4', 'wzf5', 'cp'], 'Weekly expressions changed')
    trade = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trade'); prefix = []
    for node in trade.body:
        if isinstance(node, ast.If): break
        require(isinstance(node, ast.Assign), 'Unexpected trade prefix'); prefix.append(node)
    require(prefix[-1].targets[0].id == 'code1_type1_bad', 'Trade condition block differs')
    return (compile(ast.Module(body=weekly, type_ignores=[]), '<original-weekly-score>', 'exec'),
        compile(ast.Module(body=prefix, type_ignores=[]), '<original-trade-conditions>', 'exec'),
        {'weekly': hashlib.sha256(ast.dump(branch).encode()).hexdigest(), 'trade_prefix': hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[])).encode()).hexdigest()})


def momentum_kernels(tree):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_signal')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    expressions = {n.targets[0].id: n.value for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    code = {name: compile(ast.Expression(expressions[name]), '<original-momentum-expression>', 'eval')
        for name in ('cp_increase', 'ma_n1', 'pre_price')}
    return code, hashlib.sha256(ast.dump(loop).encode()).hexdigest()


def compute(directory):
    validate_sources(directory); settings = backend(); require(settings == read(directory / 'existing-apis.json')['indicator'] and
        settings['wrapper'] == '0.8.1' and settings['compatibility'] == 0 and settings['ema_unstable'] == 0, 'TA-Lib backend changed')
    for name, row in read(directory / 'input-binding.json')['files'].items(): require(file_sha(directory / name) == row['sha256'], 'Input bytes changed')
    weekly_code, trade_code, ast_sha = kernels(directory); groups = []; boundaries = []
    momentum_code, ast_sha['momentum'] = momentum_kernels(source_tree(directory, 1))
    for name in ('index-input.parquet', 'gem-raw.parquet', 'fund-raw.parquet'):
        frame = parse_daily(pd.read_parquet(directory / name)); values = frame.close.to_numpy(); volume = frame.volume.to_numpy()
        weekly = frame.groupby(pd.to_datetime(frame.date).dt.to_period('W-FRI'), sort=True).tail(1).reset_index(drop=True)
        weekly_digest = hashlib.sha256(); weekly_error = 0.
        for k in range(8, len(weekly)):
            part = weekly.close.iloc[k-8:k+1].to_numpy(); ns = {'close_sec1w': part}; exec(weekly_code, ns)
            actual = float(ns['cp']); expected = weekly_score(part); weekly_error = max(weekly_error, abs(actual-expected))
            weekly_digest.update(json.dumps({'date': weekly.date.iloc[k], 'value': actual}, sort_keys=True).encode())
        require(weekly_error < 1e-12, 'Weekly arithmetic differs')
        trade_digest = hashlib.sha256(); ema_error = 0.; flag_counts = {n: 0 for n in ('code1_strong', 'code1_weak', 'code1_type1_bad', 'code1_type2_bad')}
        env = {'talib': talib, 'g': SimpleNamespace(codes='arithmetic_only')}
        for k in range(119, len(frame)):
            part = values[k-119:k+1]; env['get_bars'] = lambda *a, _part=part, **kw: {'close': _part}
            exec(trade_code, env); reference = ema_reference(part, 12)
            ema_error = max(ema_error, float(np.nanmax(np.abs(env['ema12']-reference))))
            m = {n: math.fsum(part[-n:])/n for n in (5, 10, 12, 20, 120)}
            pre5 = math.fsum(part[-6:-1])/5; pre12 = math.fsum(part[-13:-1])/12
            bias = (part[-1]-m[12])/m[12]*100; prev_bias = (part[-2]-pre12)/pre12*100
            flags = {'code1_strong': bool(m[20] < m[10] < m[5] < part[-1] and bias>prev_bias and reference[-1]>=reference[-2] and
                part[-1]/m[5]<1.1 and (part[-1]>m[120] or (part[-1]<m[120] and m[5]>pre5))),
                'code1_weak': bool(part[-1]/m[5]<.88), 'code1_type2_bad': bool(part[-1]<part[-2]),
                'code1_type1_bad': bool((part[-1]<m[10] and part[-2]<m[10]) or part[-1]/m[5]>=1.1 or
                    (reference[-1]<reference[-2] and reference[-1]/reference[-2]<.997))}
            actual = {n: bool(env[n]) for n in flags}; flag_counts = {n: flag_counts[n]+int(actual[n]) for n in flags}
            if actual != flags: boundaries.append({'file': name, 'date': frame.date.iloc[k], 'original': actual, 'stable_reference': flags,
                'original_ma': {n: float(env[f'ma{n}']) for n in (5, 10, 12, 20, 120)}, 'reference_ma': m,
                'ema_last': [float(env['ema12'][-1]), float(env['ema12'][-2])], 'reference_ema_last': [reference[-1], reference[-2]]})
            trade_digest.update(json.dumps({'date': frame.date.iloc[k], 'flags': actual, 'ema_last': float(env['ema12'][-1])}, sort_keys=True).encode())
        require(ema_error < 1e-9, 'EMA recursion differs')
        momentum_digest = hashlib.sha256(); momentum_error = 0.
        for k in range(20, len(frame)):
            prior = values[k-20:k]; current = values[k]
            env = {'current_price': current, 'close_data': {'close': prior}, 'g': SimpleNamespace(lag2=10, ma_threshold=.9)}
            original_return = eval(momentum_code['cp_increase'], env)
            env['close_data'] = {'close': prior[-10:]}; original_ma = eval(momentum_code['ma_n1'], env)
            env['ma_n1'] = original_ma; original_difference = eval(momentum_code['pre_price'], env)
            reference_ma = (current+math.fsum(prior[-9:]))/10; momentum_error = max(momentum_error, abs(original_ma-reference_ma))
            original_gate = bool(original_return>2 and original_difference>0)
            reference_gate = bool(original_return>2 and current-reference_ma*.9>0)
            if original_gate != reference_gate: boundaries.append({'file': name, 'date': frame.date.iloc[k],
                'operator': 'momentum', 'original': original_gate, 'stable_reference': reference_gate,
                'original_ma': float(original_ma), 'reference_ma': float(reference_ma)})
            momentum_digest.update(json.dumps({'date': frame.date.iloc[k], 'return20': float(original_return),
                'ma10': float(original_ma), 'eligible': original_gate}, sort_keys=True).encode())
        require(momentum_error < 1e-9, 'Current-price MA arithmetic differs')
        emotion_digest = hashlib.sha256(); emotion_counts = {}; ns = {'talib': talib, 'g': SimpleNamespace(target_market='arithmetic_only', lag0=7, lag=6)}
        selected(directory, 2, ['EmotionMonitor'], ns); emotion_error = 0.
        for k in range(99, len(frame)):
            part = volume[k-99:k+1]; ns['attribute_history'] = lambda *a, _part=part, **kw: pd.DataFrame({'volume': _part})
            actual = ns['EmotionMonitor'](None); means = talib.MA(part, 7)
            expected_means = np.array([math.nan]*6+[math.fsum(part[j-6:j+1])/7 for j in range(6, 100)])
            emotion_error = max(emotion_error, float(np.nanmax(np.abs(means-expected_means)/np.maximum(np.abs(expected_means), 1.))))
            with np.errstate(divide='ignore', invalid='ignore'): expected = emotion_reference(part/expected_means-1)
            if actual != expected: boundaries.append({'file': name, 'date': frame.date.iloc[k], 'operator': 'emotion', 'original': actual, 'stable_reference': expected})
            key = str(actual); emotion_counts[key] = emotion_counts.get(key, 0)+1
            emotion_digest.update(json.dumps({'date': frame.date.iloc[k], 'state': actual, 'rate': native(ns['g'].emotion_rate)}, sort_keys=True).encode())
        require(emotion_error < 1e-12, 'Volume SMA arithmetic differs')
        groups.append({'file': name, 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max(),
            'weekly_windows': len(weekly)-8, 'weekly_sha256': weekly_digest.hexdigest(), 'weekly_max_error': weekly_error,
            'trade_condition_windows': len(frame)-119, 'flags': flag_counts, 'trade_sha256': trade_digest.hexdigest(), 'ema_max_error': ema_error,
            'momentum_windows': len(frame)-20, 'momentum_sha256': momentum_digest.hexdigest(), 'current_ma_max_error': momentum_error,
            'emotion_windows': len(frame)-99, 'emotion_sha256': emotion_digest.hexdigest(), 'emotion_states': emotion_counts, 'volume_ma_max_relative_error': emotion_error})
    return native({'snapshot': SNAPSHOT, 'not_a_backtest': True, 'original_strategy_complete': False, 'strategy_results': [],
        'backend': settings, 'source_sha256': SOURCE_SHA, 'kernel_ast_sha256': ast_sha, 'groups': groups, 'boundaries': boundaries,
        'limits': ['Verified EOD row arithmetic only; original15-fund pools, intraday prices/events and economic state continuity absent',
            'Neither ETF raw nor index arithmetic supplies original EMA/weekly platform bars or a trading benchmark',
            'No portfolio, fill, fee or NAV calculated; independent reference differences remain separate evidence']})


def diagnostics(directory):
    validate_sources(directory); log = SimpleNamespace(info=lambda *a: None); rows = []
    calls = []; g = SimpleNamespace(); ns = {'pd': pd, 'np': np, 'talib': talib, 'g': g, 'log': log, 'print': lambda *a: None,
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'get_security_info': lambda s: SimpleNamespace(code=s, display_name=s, start_date='not_observed')}
    selected(directory, 2, ['set_params', 'set_variables', 'set_backtest', 'initialize'], ns)
    log.set_level = lambda *a: None; ns['run_daily'] = lambda fn, **kw: calls.append({'callback': fn.__name__, **kw})
    ns['ETFtrade1'] = lambda *a: None; ns['ETFtrade2'] = lambda *a: None; ns['initialize'](None)
    require(len(g.ETFList) == 7 and [r['time'] for r in calls] == ['11:30', '14:40'], 'Original pool/schedule differs')
    rows.append({'case': 'seven_funds_and_static_schedule', 'pool': g.ETFList, 'callbacks': calls, 'signal_at_initialize': g.signal})
    selected(directory, 2, ['EmotionMonitor'], ns)
    for case, volume in (('constant_volume_no_cross', np.ones(100)*100), ('zero_volume_no_cross', np.zeros(100))):
        ns['attribute_history'] = lambda *a, _v=volume, **kw: pd.DataFrame({'volume': _v})
        with np.errstate(divide='ignore', invalid='ignore'): state = ns['EmotionMonitor'](None)
        require(state is None, 'Original no-cross None differs'); rows.append({'case': case, 'state': state, 'rate': native(g.emotion_rate)})
    def missing(*a, **kw): raise ValueError('synthetic missing volume')
    ns['attribute_history'] = missing; require(ns['EmotionMonitor'](None) == 1, 'Original fallback differs')
    rows.append({'case': 'missing_volume_fail_open', 'state': 1})
    ns['EmotionMonitor'] = lambda *a: 0; selected(directory, 2, ['get_signal'], ns)
    prices = {s: 110.+i for i, s in enumerate(g.ETFList.values())}
    ns['attribute_history'] = lambda *a, **kw: {'close': np.ones(13)*100}
    ns['get_current_data'] = lambda: {s: SimpleNamespace(last_price=p) for s, p in prices.items()}
    result = ns['get_signal'](None); require(result == 'BUY' and g.buy == [max(prices, key=prices.get)] and g.target_market == g.IdxList[min(prices, key=prices.get)], 'Original one-fund/worst-index differs')
    rows.append({'case': 'unreachable_multi_fund_branches', 'state': result, 'selected': list(g.buy), 'emotion_index': g.target_market})
    prices = {s: 100.1 for s in g.ETFList.values()}; g.buy = []; g.last = []
    ns['get_current_data'] = lambda: {s: SimpleNamespace(last_price=p) for s, p in prices.items()}
    result = ns['get_signal'](None); rows.append({'case': 'point_one_decimal_float_boundary', 'state': result, 'computed_return': float(g.df.iloc[0, 2])})
    weekly_ns = {'pd': pd, 'g': SimpleNamespace(stocks=['negative_A', 'negative_B']), 'get_bars': lambda s, **kw: {'close': np.linspace(100., 90. if s=='negative_A' else 80., 9)}}
    selected(directory, 0, ['chenk_stocks'], weekly_ns); weekly_ns['chenk_stocks'](None)
    require(weekly_ns['g'].codes == 'negative_A', 'Original negative weekly winner differs'); rows.append({'case': 'negative_weekly_scores_still_ranked', 'selected': weekly_ns['g'].codes})
    weekly_ns['get_bars'] = lambda *a, **kw: {'close': np.ones(8)}
    try: weekly_ns['chenk_stocks'](None)
    except IndexError as exc: rows.append({'case': 'no_nine_week_candidate', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected empty ranking error')
    close = 100.+40.*np.linspace(0., 1., 120)**2; position = SimpleNamespace(avg_cost=100.)
    for immediate in (False, True):
        orders = []; positions = {'old': position}; context = SimpleNamespace(portfolio=SimpleNamespace(positions=positions, available_cash=1000.))
        trade_ns = {'talib': talib, 'g': SimpleNamespace(codes='new'), 'log': log, 'get_bars': lambda *a, **kw: {'close': close}}
        def sell(stock, quantity):
            orders.append(['order_target', stock, quantity])
            if immediate: positions.pop(stock, None)
        trade_ns['order_target'] = sell; trade_ns['order_value'] = lambda *a: orders.append(['order_value', *a])
        selected(directory, 0, ['trade'], trade_ns); trade_ns['trade'](context)
        require(len(orders) == (2 if immediate else 1), 'Original same-day state dependency differs')
        rows.append({'case': 'original_sale_state_visibility', 'immediate_mock_position_removal': immediate, 'mock_orders': orders})
    momentum_g = SimpleNamespace(lag1=20, lag2=10, ma_threshold=.9, mtm_threshold=2,
        ETFList=np.array([[f'signal_{i}', f'fund_{i}'] for i in range(7)]))
    momentum_ns = {'pd': pd, 'np': np, 'g': momentum_g, 'log': log, 'print': lambda *a: None}
    selected(directory, 1, ['get_signal', 'ETFtrade1', 'ETFtrade2', 'sell_the_stocks', 'buy_the_stocks'], momentum_ns)
    momentum_ns['attribute_history'] = lambda s, n, *a, **kw: {'close': np.ones(n)*100.}
    momentum_ns['get_current_data'] = lambda: {f'signal_{i}': SimpleNamespace(last_price=103.+i) for i in range(7)}
    momentum_ns['ETFtrade1'](None); winners = [f'fund_{i}' for i in (6, 5, 4, 3)]
    require(momentum_g.buy_list == winners and np.allclose(momentum_g.buy_list_weights.values, np.array([9., 8., 7., 6.])/30.), 'Original top4 weighted ranks differ')
    orders = []; momentum_ns['order_target_value'] = lambda *a: orders.append(list(a))
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={'old': object()}, portfolio_value=3000.))
    momentum_ns['ETFtrade2'](context)
    require(orders[0] == ['old', 0] and [o[0] for o in orders[1:]] == winners and
        np.allclose([o[1] for o in orders[1:]], [900., 800., 700., 600.]), 'Original weighted target intents differ')
    rows.append({'case': 'momentum_top4_return_weights', 'selected': winners, 'weights': momentum_g.buy_list_weights.tolist(), 'mock_orders': orders})
    momentum_ns['get_current_data'] = lambda: {f'signal_{i}': SimpleNamespace(last_price=100.) for i in range(7)}
    scalar, weights = momentum_ns['get_signal'](None)
    require(isinstance(scalar, np.str_) and weights.tolist() == [1.], 'Original bond scalar differs')
    momentum_ns['ETFtrade1'](None); require(momentum_g.buy_list == ['511010.XSHG'], 'Original scalar normalization differs')
    rows.append({'case': 'momentum_bond_scalar_normalization', 'scalar_type': type(scalar).__name__, 'buy_list': momentum_g.buy_list})
    selected(directory, 2, ['ETFtrade2'], ns)
    for value in (150., float(np.nextafter(150., math.inf))):
        ns['g'] = deepcopy(g); ns['g'].signal = 'BUY'; ns['g'].buy = ['A', 'B']; ns['g'].clear = []
        positions = {s: SimpleNamespace(value=value, security=s) for s in ('A', 'B')}; orders = []
        context = SimpleNamespace(portfolio=SimpleNamespace(positions=positions, total_value=200., returns=0., available_cash=0.))
        ns['order_target_value'] = lambda *a: orders.append(list(a)); ns['order_value'] = lambda *a: orders.append(list(a))
        ns['ETFtrade2'](context)
        require(len(orders) == (0 if value==150. else 4), 'Original strict1.5 repeated intents differ')
        rows.append({'case': 'emotion_strict_rebalance_repeated_intents', 'synthetic_position_value': value, 'mock_orders': orders,
            'synthetic_state': 'Two held candidates exercise original inner loop; not reachable from seven-fund single-winner signal'})
    return native({'cases': rows, 'not_a_backtest': True, 'platform_equivalent': False,
        'limits': ['Mock functions are explicit synthetic inputs; state visibility cases do not prove platform fill/settlement timing']})


def boundary_diagnosis(directory):
    result = read(directory / 'component-research.json'); rows = []
    for boundary in result['boundaries']:
        require(boundary['file'] == 'fund-raw.parquet' and boundary['date'] == '2021-10-15' and
            boundary['original']['code1_strong'] is True and boundary['stable_reference']['code1_strong'] is False,
            'Unexplained boundary requires separate diagnosis')
        input_path = directory / boundary['file']; bound = read(directory / 'input-binding.json')['files'][boundary['file']]
        require(file_sha(input_path) == bound['sha256'], 'Boundary operand changed')
        frame = parse_daily(pd.read_parquet(input_path)); k = frame.index[frame.date.eq(boundary['date'])].item()
        values = frame.close.to_numpy()[k-119:k+1]
        env = {'talib': talib, 'g': SimpleNamespace(codes='arithmetic_only'), 'get_bars': lambda *a, **kw: {'close': values}}
        exec(kernels(directory)[1], env)
        binary = [Fraction.from_float(float(v)) for v in values]
        decimal = [Fraction(str(v)) for v in values]
        exact_binary = [sum(binary[-5:])/5, sum(binary[-6:-1])/5]
        exact_decimal = [sum(decimal[-5:])/5, sum(decimal[-6:-1])/5]
        original = [float(env['ma5']), float(env['ma5_r1'])]
        stable = [math.fsum(values[-5:])/5, math.fsum(values[-6:-1])/5]
        require(exact_binary[0] == exact_binary[1] and exact_decimal[0] == exact_decimal[1] and
            original[0] > original[1] and stable[0] == stable[1] and values[-1] < env['ma120'] and
            bool(env['code1_strong']) == boundary['original']['code1_strong'], 'Boundary cause differs')
        rows.append({'file': str(input_path), 'input_sha256': file_sha(input_path), 'date': boundary['date'],
            'window120': values.tolist(), 'window120_float_hex': [float(v).hex() for v in values],
            'ma5_windows': [values[-5:].tolist(), values[-6:-1].tolist()],
            'original_ma5_current_previous': original, 'fsum_ma5_current_previous': stable,
            'exact_binary_current_previous': list(map(str, exact_binary)), 'exact_decimal_current_previous': list(map(str, exact_decimal)),
            'exact_binary_difference': str(exact_binary[0]-exact_binary[1]), 'exact_decimal_difference': str(exact_decimal[0]-exact_decimal[1]),
            'source_sha256': SOURCE_SHA[0], 'original_strong': True, 'stable_reference_strong': False,
            'cause': 'MA5 current and previous contain identical values in different order; NumPy reduction rounds previous lower by one ULP',
            'not_an_original_strategy_trade': True})
    return {'component_sha256': file_sha(directory / 'component-research.json'), 'rows': rows,
        'not_a_backtest': True, 'original_kernel_retained': True,
        'limits': ['510300 is an explicitly labelled arithmetic operand outside source22 active pool; one formula flag difference is not an order/fill difference']}


def diagnose(root, directory):
    binding(root, directory); validate_sources(directory)
    save(directory / 'ma-boundary-diagnosis.json', boundary_diagnosis(directory))
    return archive(root, 'Batch51 one reordered-MA5 float boundary independently diagnosed; original result retained')


def study(root, directory):
    binding(root, directory); save(directory / 'component-research.json', compute(directory)); save(directory / 'diagnostics.json', diagnostics(directory))
    return archive(root, 'Batch51 exact weekly/EMA/volume and original state defects archived; no trades')


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False); name, params = QUERIES[endpoint]
    row = {'endpoint': endpoint, 'function': name, 'parameters': params, 'status': 'failed', 'files': [], 'wire_responses': [],
        'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}; original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire_responses']) < 8, 'Request cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'response-{len(row["wire_responses"]):03d}.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API changed')
        import akshare as ak
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, name)(**params)
        require(not (Path(root) / 'raw/etf_rotation_probe_batch51' / endpoint / f'{directory.name}.parquet').exists(), 'Raw exists')
        path = raw.save(root, 'etf_rotation_probe_batch51', endpoint, directory.name, frame)
        row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
        row['status'] = 'success' if len(frame) else 'empty'
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def validate_probes(directory):
    require(json.loads(json.dumps(api_evidence())) == read(directory / 'existing-apis.json'), 'API evidence changed')
    rows = read(directory / 'probe-results.json')['results']; require([r['endpoint'] for r in rows] == list(QUERIES), 'Probe scope changed')
    for row in rows:
        require(row == read(directory / 'probes' / row['endpoint'] / 'result.json') and row['published'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['function'] == QUERIES[row['endpoint']][0] and
            row['parameters'] == QUERIES[row['endpoint']][1] and row['status'] in ('success', 'empty', 'failed', 'timeout'), 'Probe binding changed')
        for item in row['wire_responses']+row['files']: require(file_sha(item['file']) == item['sha256'], 'Probe file changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw count differs'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] and list(frame) == item['columns'] and bool(len(frame)) == (row['status']=='success'), 'Probe raw profile differs')
        else: require(not row['files'] and row.get('error'), 'Failed probe raw/error differs')
    return rows


def probe(root, directory):
    rows = []; logs = directory / 'logs'; logs.mkdir()
    for endpoint in QUERIES:
        folder = directory / 'probes' / endpoint; require(not folder.exists(), 'Probe exists')
        command = [sys.executable, '-m', 'scripts.review_strategy_batch51', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (logs / f'{endpoint}.stdout').open('x') as out, (logs / f'{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (folder / 'result.json').exists(): save(folder / 'result.json', {'endpoint': endpoint, 'function': QUERIES[endpoint][0],
                    'parameters': QUERIES[endpoint][1], 'status': 'timeout', 'error': 'Parent deadline45s', 'published': False,
                    'files': [], 'wire_responses': [], 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(folder / 'result.json'))
    save(directory / 'probe-results.json', {'results': rows, 'published': False}); validate_probes(directory)
    return archive(root, 'Batch51 existing ETF daily/minute and missing-index supplements archived')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch51 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(io.StringIO()):
        component = compute(directory); diag = diagnostics(directory); boundary = boundary_diagnosis(directory); catalog = offline_catalog(root, directory)
    save(directory / 'component-offline.json', component)
    require(before == implementation() and file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json') and
        diag == read(directory / 'diagnostics.json') and boundary == read(directory / 'ma-boundary-diagnosis.json'), 'Offline outputs changed')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'sha256': file_sha(directory / 'component-offline.json'), 'not_a_backtest': True})
    return archive(root, 'Batch51 forbidden-network original component/state/catalog byte match accepted')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch51.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch51_research.py', 'tests/unit/test_batch50_research.py',
         'tests/unit/test_batch43_research.py', 'tests/unit/test_catalog_bars.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
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
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline proof changed')
    validate_probes(directory); require(inventory(root) == read(directory / 'dependency-inventory.json') and compute(directory) == read(directory / 'component-research.json') and
        diagnostics(directory) == read(directory / 'diagnostics.json') and boundary_diagnosis(directory) == read(directory / 'ma-boundary-diagnosis.json'), 'Recomputed research changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch51 weekly ETF/volume-emotion research accepted; original funds and intraday inputs missing')
    save(Path('docs/handoff/2026-10-07-batch51-verification.json'), {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'progress': progress,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'prepare', 'study', 'diagnose', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch51/20261007-weekly-etf-emotion')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
