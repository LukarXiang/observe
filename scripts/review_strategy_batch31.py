"""Freeze CAPM/weekly RSRS components and financial API dependency corrections."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
import hashlib
import importlib.util
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tokenize
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats

from observe.data import raw
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts import review_strategy_batch27 as stock_binding
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261006-145049-e06c'
RECEIPT = Path('docs/handoff/2026-10-06-batch30-verification.json')
ACCEPTED = Path('data/staging/strategies-batch30/20261006-kama-adhesion-pullback')
STOCK_ACCEPTED = Path('data/staging/strategies-batch27/20261006-value-ma-turtle')
SOURCES = ('2020年度精选策略/71 【投资学】CAPM单因子回归模型+ROE股票池（含止损）.txt',
    '2021年度精选策略/88.基于动量因子的ETF轮动加上RSRS择时.txt',
    '2023年度精选策略/83.韶华研究之一，布林突破+均线金叉，四年五倍.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/ols.py'),
    ('repo/baostock', 'baostock/demo/demo_profit_data.py'),
    ('repo/akshare', 'akshare/stock_fundamental/stock_finance_sina.py'))
FILES = tuple(sorted(set(stock_binding.FILES) | {'scripts/review_strategy_batch31.py',
    'tests/unit/test_batch31_research.py', 'tests/unit/test_catalog_dependencies.py'}))
CORE = {'baseline.json', 'catalog-before.json', 'red-regression.json', 'code-before/strategy_catalog.py', 'api-scan.json',
    'input-binding.json', 'source-reviews/review.json', 'existing-apis.json', 'input-analysis.json', 'price-input.parquet',
    'index-input.parquet', 'calendar-input.parquet', 'component-research.json', 'diagnostics.json', 'probe-results.json',
    'component-offline.json', 'offline-verification.json', 'offline-catalog.json'}
QUERIES = {'roe': {'provider': 'baostock', 'function': 'query_profit_data',
        'parameters': {'code': 'sh.600519', 'year': 2020, 'quarter': 1},
        'limit': 'Quarterly final ROE/publication sample is not platform ROE vintage or the whole original pool'},
    'holder': {'provider': 'akshare', 'function': 'stock_circulate_stock_holder', 'parameters': {'symbol': '600519'},
        'limit': 'May provide only five holders; announcement dates do not prove original top10 table revisions'},
    'etf': {'provider': 'akshare', 'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh510050'},
        'limit': 'One fund price sample does not supply seven funds/events/states or seven index histories'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def preflight(root, directory):
    checkpoint(root, directory, 'Batch31 CAPM/weekly RSRS/finance dependency correction started')
    latest = read(Path(root) / 'catalog/strategies/latest.json'); folder = Path(latest['directory'])
    save(directory / 'catalog-before.json', {'catalog_id': latest['catalog_id'],
        'files_sha256': {str(folder / n): file_sha(folder / n) for n in ('catalog.json', 'summary.json', 'catalog.parquet')}})
    copied = directory / 'code-before'; copied.mkdir()
    shutil.copyfile('src/observe/strategy_catalog.py', copied / 'strategy_catalog.py')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_dependencies.py', '--tb=no']
    result = subprocess.run(command, capture_output=True, text=True)
    save(directory / 'red-regression.json', {'command': command, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'code_sha256': file_sha(copied / 'strategy_catalog.py'),
        'test_sha256': file_sha('tests/unit/test_catalog_dependencies.py')})
    require(result.returncode == 1 and '10 failed, 11 passed' in result.stdout, 'Expected API dependency regression not reproduced')
    return {'status': 'ok', 'regression': '10 failed, 11 passed on original catalog code'}


def validate_scan(directory):
    before = read(directory / 'catalog-before.json')
    require(all(file_sha(p) == sha for p, sha in before['files_sha256'].items()), 'Old catalog changed')
    red = read(directory / 'red-regression.json')
    require(red['returncode'] == 1 and '10 failed, 11 passed' in red['stdout'] and
        red['code_sha256'] == file_sha(directory / 'code-before/strategy_catalog.py') and
        red['test_sha256'] == file_sha('tests/unit/test_catalog_dependencies.py'), 'Red regression binding differs')
    return next(p for p in before['files_sha256'] if Path(p).name == 'catalog.json')


def scan(root, directory):
    old_file = validate_scan(directory); old = {r['path']: r for r in read(old_file)}
    result = catalog_strategies(root, 'repo/量化策略源代码'); path = Path(result['output']) / 'catalog.json'; rows = read(path)
    require(len(rows) == len(old) == 695 and {r['path'] for r in rows} == set(old), 'Source set differs')
    changed = []
    for row in rows:
        prior = old[row['path']]
        require(all(row[k] == v for k, v in prior.items() if k not in ('apis', 'gaps', 'status')), 'API scan changed manual/source metadata')
        delta = {k: {'before': prior[k], 'after': row[k]} for k in ('apis', 'gaps', 'status') if row[k] != prior[k]}
        if delta: changed.append({'path': row['path'], 'source_sha256': row['bytes_sha256'], 'changes': delta})
    save(directory / 'api-scan.json', {'status': 'ok', 'catalog': result, 'catalog_file': str(path), 'catalog_sha256': file_sha(path),
        'changes': changed, 'not_a_backtest': True, 'strategy_results': [],
        'limits': ['Text call candidates include comments and strings; qualified API names do not prove platform injection',
            'Generic finance table gap is distinct from accounting ROE/financial statements']})
    return {'catalog': result['catalog_id'], 'changed_sources': len(changed),
        'progress': archive(root, 'Batch31 valuation/finance API dependency correction frozen across695 original sources')}


def binding(root, directory=None):
    receipt = read(RECEIPT); store = Store(root); state = store.state(SNAPSHOT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted batch30 differs')
    price = ACCEPTED / 'price-input.parquet'; baseline = ACCEPTED / 'baseline.json'
    require(file_sha(price) == receipt['checks']['evidence_sha256']['price-input.parquet'] and
        file_sha(baseline) == receipt['checks']['evidence_sha256']['baseline.json'], 'Accepted prices/baseline changed')
    prior = read(baseline)['published']['tables']; rows = []
    for table in ('index_1d', 'calendar'):
        require(state['tables'][table] == prior[table], 'Accepted index/calendar references changed')
        for part, entry in sorted(state['tables'][table].items()):
            path = store.root / entry['file']; before = file_sha(path); frame = pd.read_parquet(path)
            require(len(frame) == entry['rows'] and fingerprint(frame) == entry['sha'] and file_sha(path) == before,
                'Index/calendar content differs from snapshot')
            keys = ['date', 'index'] if table == 'index_1d' else ['date']
            require(not frame[keys].isna().any().any() and not frame.duplicated(keys).any(), 'Index/calendar keys invalid')
            rows.append({'table': table, 'part': part, 'entry': entry, 'file': str(path), 'sha256': before})
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'price_file': str(price), 'price_sha256': file_sha(price), 'partitions': rows,
        'stock_binding': stock_binding.binding(root, STOCK_ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    return result


def api_evidence():
    import akshare as ak
    import baostock as bs
    rows = []
    for name, query in QUERIES.items():
        fn = getattr(bs if query['provider'] == 'baostock' else ak, query['function']); code = inspect.getsource(fn)
        rows.append({'endpoint': name, 'function': query['function'], 'signature': str(inspect.signature(fn)),
            'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'akshare': ak.__version__, 'baostock': bs.__version__, 'queries': QUERIES, 'apis': rows}


def start(root, directory):
    validate_scan(directory); scanned = read(directory / 'api-scan.json')
    require(file_sha(scanned['catalog_file']) == scanned['catalog_sha256'] and
        read(Path(root) / 'catalog/strategies/latest.json')['catalog_id'] == scanned['catalog']['catalog_id'], 'API scan changed')
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old = {r['path']: r for r in read(scanned['catalog_file'])}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == file_sha(path), 'Source reviewed/changed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': file_sha(path),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'OLS conventions, quarterly ROE and holder date/schema availability; not platform-equivalent substitution'})
    search = ['rg', '--files', 'repo', '-g', '*bug_eye*', '-g', '*bug_brain*', '-g', '*technical_analysis*']
    result = subprocess.run(search, capture_output=True, text=True); require(result.returncode in (0, 1), result.stderr)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': scanned['catalog_file'], 'previous_catalog_sha256': scanned['catalog_sha256'], 'snapshot': SNAPSHOT,
        'dependency_search': {'command': search, 'returncode': result.returncode, 'matches': result.stdout.splitlines()},
        'installed_modules': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqlib', 'jqfactor')},
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch31 three complete CAPM/weekly-RSRS/eye-brain reviews frozen; originals blocked where inputs absent')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Reviewed previous catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference set differs')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def prepare(root, directory):
    binding(root, directory); validate_sources(directory); store = Store(root); state = store.state(SNAPSHOT)
    require(not any((directory / n).exists() for n in ('price-input.parquet', 'index-input.parquet', 'calendar-input.parquet')), 'Inputs exist')
    shutil.copyfile(ACCEPTED / 'price-input.parquet', directory / 'price-input.parquet')
    index = store.load_state(state, 'index_1d').sort_values('date').reset_index(drop=True)
    calendar = store.load_state(state, 'calendar').sort_values('date').reset_index(drop=True)
    index['date'] = pd.to_datetime(index.date).dt.strftime('%Y-%m-%d'); calendar['date'] = pd.to_datetime(calendar.date).dt.strftime('%Y-%m-%d')
    require(set(index['index']) == {'000300.SH'}, 'Original CAPM/RSRS benchmark not covered')
    dates = calendar[calendar.is_open & calendar.date.between(index.date.min(), index.date.max())].date.tolist()
    require(index.date.tolist() == dates, 'Index has missing/extra verified trading dates')
    index.to_parquet(directory / 'index-input.parquet', index=False); calendar.to_parquet(directory / 'calendar-input.parquet', index=False)
    profiles = {}
    for name in ('price', 'index', 'calendar'):
        path = directory / f'{name}-input.parquet'; frame = pd.read_parquet(path)
        profiles[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'first': frame.date.min(), 'last': frame.date.max()}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'inputs': profiles, 'pool': list(stock_binding.INSTRUMENTS),
        'index': '000300.SH', 'trading_dates': dates, 'not_a_backtest': True, 'platform_equivalent': False,
        'capm_policy': 'Each completed end-date pairs121 traded stock rows with latest121 market rows by position; differing dates explicitly counted, not realigned',
        'rsrs_clock': 'Third verified trading session of each ISO week, previous verified close as input; platform holiday/runtime equivalence unproved',
        'limits': ['Ten stocks are not original ROE top1000 pool', 'One index is not seven-index ranking or seven-ETF trading',
            'RangeIndex adapters model positional library samples; original dated-Series/platform compatibility remains unproved']})
    return archive(root, 'Batch31 long verified stock/index/calendar components frozen without changing dates or original strategies')


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); frames = {}
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and doc['pool'] == list(stock_binding.INSTRUMENTS) and
        doc['index'] == '000300.SH' and set(doc['inputs']) == {'price', 'index', 'calendar'}, 'Input scope differs')
    for name, row in doc['inputs'].items():
        path = directory / f'{name}-input.parquet'; require(str(path) == row['file'] and file_sha(path) == row['sha256'], 'Input bytes changed')
        frame = pd.read_parquet(path)
        require(len(frame) == row['rows'] and frame.date.min() == row['first'] and frame.date.max() == row['last'], 'Input profile differs')
        keys = ['date', 'instrument'] if name == 'price' else ['date']
        require(not frame.duplicated(keys).any() and not frame[keys].isna().any().any(), 'Input keys invalid'); frames[name] = frame
    require(frames['index'].date.tolist() == doc['trading_dates'] and set(frames['index']['index']) == {'000300.SH'} and
        set(frames['price'].instrument) == set(doc['pool']), 'Input universe/calendar differs')
    return frames


def fragment(directory, number, name):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    lines = read_source(Path(row['source_copy']))[0].splitlines(keepends=True)
    starts = [k for k, line in enumerate(lines) if line.startswith(f'def {name}(')]
    require(len(starts) == 1, 'Missing/ambiguous source function'); start = starts[0]; text = ''.join(lines[start:]); depth = 0; entered = False; end = len(lines) - start
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.INDENT: depth += 1; entered = True
        elif token.type == tokenize.DEDENT:
            depth -= 1
            if entered and depth == 0: end = token.start[0] - 1; break
    return ''.join(lines[start:start + end])


def regression_expression(directory):
    tree = ast.parse(fragment(directory, 0, 'get_signal'))
    nodes = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'stockalpha' for t in ast.walk(n.targets[0]))]
    require(len(nodes) == 1, 'Missing/ambiguous CAPM regression')
    return compile(ast.Expression(nodes[0]), '<original-capm-regression>', 'eval'), hashlib.sha256(ast.dump(nodes[0]).encode()).hexdigest()


def reference_ols(x, y):
    x = list(map(float, x)); y = list(map(float, y)); n = len(x)
    require(n == len(y) and n > 1 and np.isfinite(x).all() and np.isfinite(y).all(), 'Invalid OLS inputs')
    mx = math.fsum(x) / n; my = math.fsum(y) / n
    xx = math.fsum((v - mx) ** 2 for v in x); yy = math.fsum((v - my) ** 2 for v in y)
    require(xx > 0 and yy > 0, 'Degenerate OLS input')
    xy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)); slope = xy / xx
    return my - slope * mx, slope, xy * xy / (xx * yy)


def capm_namespace(directory):
    ns = {'np': np, 'stats': stats, 'g': SimpleNamespace(rf=.04 / 252, num=16, days=121, index='000300.XSHG', if_trade=True, feasible_stocks=[])}
    sha = selected(directory, 0, ('price2ret', 'get_signal', 'sell_and_buy_stocks', 'set_feasible_stocks', 'set_slip_fee'), ns)
    return ns, sha


def capm_components(directory, price, index):
    ns, sha = capm_namespace(directory); regression, reg_sha = regression_expression(directory)
    dates = index.date.tolist(); lookup = {d: k for k, d in enumerate(dates)}; market = index.close.to_numpy(); groups = []; histories = {}; examples = []
    require(np.isfinite(market).all() and (market > 0).all(), 'Invalid market close')
    for stock, frame in price.groupby('instrument', sort=True):
        frame = frame.sort_values('date'); paused = int((~frame.is_trading).sum()); frame = frame[frame.is_trading].reset_index(drop=True); histories[stock] = frame
        require(frame.adjustment_status.eq('usable').all() and np.isfinite(frame[['close_adj', 'back_factor']]).all().all() and
            frame[['close_adj', 'back_factor']].gt(0).all().all(), 'Unknown stock adjustment/price')
        count = misaligned = 0; error = np.zeros(3); digest = hashlib.sha256()
        for k in range(120, len(frame)):
            day = frame.date.iloc[k]; ix = lookup[day]
            if ix < 120: continue
            window = frame.iloc[k - 120:k + 1]; x = window.close_adj.to_numpy() / frame.back_factor.iloc[k]; m = market[ix - 120:ix + 1]
            stockreturn = ns['price2ret'](pd.Series(x)); marketreturn = ns['price2ret'](pd.Series(m))
            actual = eval(regression, ns, {'stockreturn': stockreturn, 'marketreturn': marketreturn})
            sr = [b / a - 1 - .04 / 252 for a, b in zip(x[:-1], x[1:], strict=True)]
            mr = [b / a - 1 - .04 / 252 for a, b in zip(m[:-1], m[1:], strict=True)]
            alpha, beta, r2 = reference_ols(mr, sr); values = [float(actual.intercept), float(actual.slope), float(actual.rvalue ** 2)]
            error = np.maximum(error, np.abs(np.asarray(values) - [alpha, beta, r2])); same = window.date.tolist() == dates[ix - 120:ix + 1]
            if not same and len(examples) < 10: examples.append({'stock': stock, 'end_date': day, 'stock_first': window.date.iloc[0], 'market_first': dates[ix - 120],
                'different_pairs': sum(a != b for a, b in zip(window.date, dates[ix - 120:ix + 1], strict=True))})
            misaligned += int(not same); count += 1
            digest.update(json.dumps({'date': day, 'coefficients': values, 'same_dates': same}, sort_keys=True).encode())
        require(error.max() < 1e-10, 'CAPM component mismatch')
        groups.append({'stock': stock, 'windows': count, 'misaligned_date_windows': misaligned, 'known_paused_rows_excluded': paused,
            'max_absolute_differences': dict(zip(('alpha', 'beta', 'r2'), error.tolist())), 'sha256': digest.hexdigest()})
    holder = {}; ns['attribute_history'] = lambda stock, count, unit, fields: pd.DataFrame({'close': holder[stock][-count:]})
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions={}, portfolio_value=100000.)); digest = hashlib.sha256(); checks = 0; max_error = 0.; boundaries = []
    for ix in range(120, len(index), 10):
        day = dates[ix]; holder['000300.XSHG'] = market[ix - 120:ix + 1]; stocks = []; alphas = []
        for stock in stock_binding.INSTRUMENTS:
            frame = histories[stock]; window = frame[frame.date.le(day)].tail(121)
            if len(window) != 121: continue
            x = window.close_adj.to_numpy() / window.back_factor.iloc[-1]; holder[stock] = x; stocks.append(stock)
            sr = [b / a - 1 - .04 / 252 for a, b in zip(x[:-1], x[1:], strict=True)]
            m = holder['000300.XSHG']; mr = [b / a - 1 - .04 / 252 for a, b in zip(m[:-1], m[1:], strict=True)]
            alphas.append((stock, reference_ols(mr, sr)[0]))
        if len(stocks) < 2: continue
        ns['g'].feasible_stocks = stocks; ns['g'].if_trade = True; sold, targets = ns['get_signal'](ctx)
        ordered = sorted(alphas, key=lambda a: a[1])[-16:]; numbers = [a for _, a in ordered]; span = max(numbers) - min(numbers)
        require(span > 0 and not sold, 'Degenerate CAPM research allocation')
        raw_weights = [(max(numbers) - a) / span for a in numbers]; total = math.fsum(raw_weights)
        expected = {s: 100000 * w / total for (s, _), w in zip(ordered, raw_weights, strict=True)}
        delta = max(abs(float(targets[s]) - v) / 100000 for s, v in expected.items()); max_error = max(max_error, delta)
        if list(targets) != list(expected): boundaries.append({'date': day, 'original': list(targets), 'reference': list(expected)})
        require(np.isfinite(list(targets.values())).all() and delta < 1e-10, 'CAPM allocation mismatch')
        digest.update(json.dumps({'date': day, 'targets': [(s, float(v)) for s, v in targets.items()]}, sort_keys=True).encode()); checks += 1
    return {'source_ast_sha256': sha, 'regression_ast_sha256': reg_sha, 'groups': groups, 'date_mismatch_examples': examples,
        'cross_sections': checks, 'weight_max_absolute_difference': max_error, 'rank_boundaries': boundaries, 'allocation_sha256': digest.hexdigest(),
        'limits': ['Positional date mismatches retained; not corrected into a new CAPM rule', 'Ten-stock allocations are component checks, not ROE selections/orders/NAV']}


def weekly_clock(index):
    stamps = pd.to_datetime(index.date); iso = stamps.dt.isocalendar(); groups = {}
    for k, (year, week) in enumerate(zip(iso.year, iso.week, strict=True)): groups.setdefault((int(year), int(week)), []).append(k)
    return [{'execution_date': index.date.iloc[rows[2]], 'price_end': index.date.iloc[rows[2] - 1], 'end_index': rows[2] - 1}
        for rows in groups.values() if len(rows) >= 3 and rows[2] >= 618]


def reference_z(values):
    x = list(map(float, values)); mean = math.fsum(x) / len(x); variance = math.fsum((v - mean) ** 2 for v in x) / len(x)
    require(variance > 0, 'Zero RSRS reference variance'); return (x[-1] - mean) / math.sqrt(variance)


def rsrs_namespace(directory, index):
    holder = {'end': len(index) - 1}
    def history(stock, count, unit, fields):
        require(stock == '000300.XSHG' and unit == '1d' and count in (15, 18, 23, 618), 'Original index request differs')
        require(holder['end'] >= count - 1, 'Insufficient original index window')
        return index.iloc[holder['end'] - count + 1:holder['end'] + 1][fields].reset_index(drop=True).copy()
    ns = {'np': np, 'math': math, 'attribute_history': history, 'g': SimpleNamespace(), 'log': SimpleNamespace(set_level=lambda *a: None)}
    sha = selected(directory, 1, ('initial_config', 'initial_slope_series', 'get_ols', 'get_zscore', 'get_signal', 'get_socre'), ns)
    return ns, holder, sha


def rsrs_components(directory, index):
    clock = weekly_clock(index); require(clock, 'Weekly RSRS clock empty'); ns, holder, sha = rsrs_namespace(directory, index)
    seed = clock[0]['end_index']; holder['end'] = seed; ns['initial_config'](); initial = index.iloc[seed - 617:seed + 1].reset_index(drop=True)
    reference = [reference_ols(initial.low.iloc[k:k + 18], initial.high.iloc[k:k + 18])[1] for k in range(600)][:-1]
    require(len(ns['g'].slope_series) == len(reference) == 599, 'Original RSRS initial count differs')
    seed_error = max(abs(float(a) - b) for a, b in zip(ns['g'].slope_series, reference, strict=True)); max_error = 0.; slope_error = 0.; boundaries = []; rows = []; digest = hashlib.sha256()
    for row in clock:
        k = row['end_index']; holder['end'] = k; signal = ns['get_signal'](); window = index.iloc[k - 17:k + 1]
        _, slope, r2 = reference_ols(window.low, window.high); reference.append(slope); score = reference_z(reference[-600:]) * slope * r2
        today = math.fsum(map(float, index.close.iloc[k - 19:k + 1])) / 20; before = math.fsum(map(float, index.close.iloc[k - 22:k - 2])) / 20
        expected = 'BUY' if score > .7 and today > before else 'SELL' if score < -.7 and today < before else 'KEEP'
        _, actual_slope, actual_r2 = ns['get_ols'](window.low, window.high); actual_score = ns['get_zscore'](ns['g'].slope_series[-600:]) * actual_slope * actual_r2
        max_error = max(max_error, abs(float(actual_score) - score)); slope_error = max(slope_error, abs(float(actual_slope) - slope))
        if signal != expected: boundaries.append({**row, 'original': signal, 'reference': expected, 'score': float(actual_score), 'reference_score': score})
        value = {**row, 'signal': signal, 'score': float(actual_score), 'history_length': len(ns['g'].slope_series)}; rows.append(value)
        digest.update(json.dumps(value, sort_keys=True).encode())
    require(seed_error < 1e-10 and max_error < 1e-10 and slope_error < 1e-10, 'RSRS component mismatch')
    scores = []; momentum_error = 0.; momentum_boundaries = []; mdigest = hashlib.sha256()
    for k in range(14, len(index)):
        holder['end'] = k; value = float(ns['get_socre']('000300.XSHG')); y = np.log(index.close.iloc[k - 14:k + 1].to_numpy()); regression = stats.linregress(np.arange(15), y)
        expected = math.expm1(float(regression.slope) * 250) * float(regression.rvalue ** 2); momentum_error = max(momentum_error, abs(value - expected))
        if (value > 0) != (expected > 0): momentum_boundaries.append({'date': index.date.iloc[k], 'original': value, 'reference': expected})
        mdigest.update(json.dumps({'date': index.date.iloc[k], 'score': value}, sort_keys=True).encode()); scores.append(value)
    require(momentum_error < 1e-10, 'Momentum component mismatch')
    return {'source_ast_sha256': sha, 'weekly_windows': len(rows), 'first': rows[0], 'last': rows[-1],
        'seed': {'input_rows': 618, 'slope_count': 599, 'last_used_date': initial.date.iloc[615], 'unused_last_dates': initial.date.iloc[616:].tolist(),
            'max_absolute_difference': seed_error}, 'rsrs_max_absolute_difference': max_error, 'slope_max_absolute_difference': slope_error,
        'comparison_boundaries': boundaries, 'signal_counts': {s: sum(r['signal'] == s for r in rows) for s in ('BUY', 'SELL', 'KEEP')},
        'weekly_sha256': digest.hexdigest(), 'momentum': {'windows': len(scores), 'max_absolute_difference': momentum_error,
            'positive': sum(v > 0 for v in scores), 'comparison_boundaries': momentum_boundaries, 'sha256': mdigest.hexdigest()},
        'clock': 'Third verified ISO-week trading session/previous close; original holiday/runtime equivalence unproved',
        'limits': ['Only HS300 numeric component; not seven-index pool or ETF returns', '599 daily initial slopes mixed with weekly appends exactly retained']}


def compute(directory):
    validate_sources(directory); data = inputs(directory)
    require(np.isfinite(data['index'][['open', 'high', 'low', 'close']]).all().all() and data['index'][['open', 'high', 'low', 'close']].gt(0).all().all(), 'Invalid index OHLC')
    return {'snapshot': SNAPSHOT, 'not_a_backtest': True, 'original_strategy_complete': False, 'strategy_results': [],
        'input_sha256': {n: file_sha(directory / f'{n}-input.parquet') for n in data},
        'backend': {'numpy': np.__version__, 'pandas': pd.__version__, 'linregress_source_sha256': hashlib.sha256(inspect.getsource(stats.linregress).encode()).hexdigest(),
            'polyfit_source_sha256': hashlib.sha256(inspect.getsource(np.polyfit).encode()).hexdigest()},
        'capm': capm_components(directory, data['price'], data['index']), 'weekly_rsrs': rsrs_components(directory, data['index'])}


def capm_weight_case(directory, alphas, positions=None):
    ns, _ = capm_namespace(directory); values = iter(alphas); names = [f's{k:02d}' for k in range(len(alphas))]
    ns['g'].feasible_stocks = names
    ns['stats'] = SimpleNamespace(linregress=lambda *a: (1., next(values), 1., 0., 0.))
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.arange(121.) + 100})
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=positions or {}, portfolio_value=100000.))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            sold, bought = ns['get_signal'](ctx); error = None
        except (ZeroDivisionError, ValueError) as exc: sold, bought = {}, {}; error = f'{type(exc).__name__}: {exc}'
    return {'stocks': names, 'sell_targets': {s: float(v) for s, v in sold.items()},
        'buy_targets': {s: float(v) if np.isfinite(v) else 'nonfinite' for s, v in bought.items()},
        'error': error, 'warnings': [str(w.message) for w in caught]}, ns, ctx


def technical_case(directory, today, yesterday):
    now = date(2022, 1, 4); requests_made = []
    ns = {'datetime': SimpleNamespace(timedelta=timedelta)}
    for name in ('TRIX', 'MACD', 'EMV', 'KD'):
        def indicator(stock, name=name, **kw):
            requests_made.append({'indicator': name, 'check_date': str(kw['check_date']), 'parameters': {k: v for k, v in kw.items() if k != 'check_date'}})
            values = today[name] if kw['check_date'] == now else yesterday[name]
            return tuple({stock: value} for value in values)
        ns[name] = indicator
    selected(directory, 2, ('Technical_signal',), ns)
    return {'signals': list(ns['Technical_signal']('component', now)), 'requests': requests_made}


def diagnostics(directory):
    cases = []; ns, sha0 = capm_namespace(directory)
    for name, alphas in (('capm_top16_inverse_alpha', list(map(np.float64, range(1, 21)))), ('capm_empty_pool', []),
        ('capm_numpy_single', [np.float64(1)]), ('capm_numpy_equal', [np.float64(1), np.float64(1)]), ('capm_python_single', [1.])):
        value, _, _ = capm_weight_case(directory, alphas); cases.append({'case': name, **value})
    positions = {'s04': SimpleNamespace(sellable_amount=100, last_sale_price=100., total_amount=1000)}
    value, local, ctx = capm_weight_case(directory, list(map(np.float64, range(1, 21))), positions)
    orders = []; local['order_target_value'] = lambda *a: orders.append(list(a)); local['sell_and_buy_stocks'](ctx, {}, {'s04': 12500.})
    cases.append({'case': 'capm_sellable_value_and_rejected_trade_flag', 'sell_targets': value['sell_targets'], 'orders': orders, 'if_trade': local['g'].if_trade})
    try: ns['price2ret'](pd.Series([1., 2., 3.], index=pd.date_range('2022-01-01', periods=3)))
    except KeyError as exc: cases.append({'case': 'capm_dated_series_compatibility', 'error': f'KeyError: {exc}'})
    else: raise AssertionError('Dated Series unexpectedly accepted legacy integer labels')
    try: ast.parse(fragment(directory, 0, 'stop'))
    except SyntaxError as exc: cases.append({'case': 'capm_python2_stop_compilation', 'error': exc.msg, 'tradable_bond': False})
    else: raise AssertionError('Original Python2 stop unexpectedly parsed')
    ns.update(isnan=np.isnan, get_current_data=lambda: {}, get_price=lambda *a, **kw: {'paused': pd.DataFrame({'short': [0]})},
        attribute_history=lambda *a, **kw: pd.DataFrame({'close': [10., 11.]}))
    approved = ns['set_feasible_stocks'](['short'], 121, SimpleNamespace(current_dt=datetime(2022, 1, 4)))
    cases.append({'case': 'capm_short_history_passes_eligibility', 'approved': approved, 'provided_rows': 2, 'requested_rows': 121})
    fees = []; ns.update(datetime=SimpleNamespace(datetime=datetime), FixedSlippage=lambda v: v, set_slippage=lambda *a: None,
        PerTrade=lambda **kw: kw, set_commission=lambda value: fees.append(value), log=SimpleNamespace(info=lambda *a: None))
    for year in (2008, 2010, 2012, 2014): ns['set_slip_fee'](SimpleNamespace(current_dt=datetime(year, 1, 2)))
    cases.append({'case': 'capm_original_fee_branches', 'fees': fees})
    frame = pd.DataFrame({'date': pd.date_range('2005-01-01', periods=618).strftime('%Y-%m-%d'),
        'close': np.arange(618.) + 100., 'low': np.arange(618.) + 100., 'high': np.arange(618.) + 101.})
    ns1, _, sha1 = rsrs_namespace(directory, frame); pairs = []
    def mark(x, y): pairs.append([int(x.index[0]), int(x.index[-1])]); return (0., float(x.iloc[0]), 1.)
    ns1['get_ols'] = mark; ns1['initial_config']()
    cases.append({'case': 'rsrs_initial_loop_tail', 'slopes': len(ns1['g'].slope_series), 'first_pair': pairs[0],
        'last_called_pair': pairs[-1], 'last_retained_pair': pairs[-2], 'unused_retained_input_rows': [616, 617]})
    trade_ns = {'g': SimpleNamespace(stock_num=2), 'print': lambda *a: None}
    trade_sha = selected(directory, 1, ('trade', 'change_position'), trade_ns)
    for name, held, pool, signal in (('keep_empty_can_buy', {}, tuple(f'e{k}' for k in range(5)), 'KEEP'),
        ('rank3_can_remain', {'e3': object()}, tuple(f'e{k}' for k in range(5)), 'KEEP'),
        ('rank4_replaces', {'e4': object()}, tuple(f'e{k}' for k in range(5)), 'KEEP'),
        ('sell_clears', {'e0': object()}, ('e0', 'e1'), 'SELL'), ('empty_pool_clears', {'old': object()}, (), 'KEEP')):
        orders = []; trade_ns.update(get_stock_pool=lambda pool=pool: pool, get_signal=lambda signal=signal: signal,
            order_target_value=lambda *a: orders.append(list(a)))
        trade_ns['trade'](SimpleNamespace(portfolio=SimpleNamespace(positions=held, available_cash=900.)))
        cases.append({'case': f'rsrs_{name}', 'orders': orders})
    ns1, holder, _ = rsrs_namespace(directory, frame); ns1['g'].momentum_day = 15; del ns1['math']
    try: ns1['get_socre']('000300.XSHG')
    except NameError as exc: cases.append({'case': 'momentum_missing_math_injection', 'error': str(exc)})
    else: raise AssertionError('Original momentum missing math unexpectedly succeeded')
    ns1['math'] = math; holder['end'] = 617
    constant = frame.copy(); constant[['close', 'low', 'high']] = [10., 9., 11.]; degenerate, _, _ = rsrs_namespace(directory, constant); degenerate['g'].momentum_day = 15
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); score = degenerate['get_socre']('000300.XSHG'); ols = degenerate['get_ols'](np.ones(18), np.ones(18)); z = degenerate['get_zscore']([1.] * 600)
    cases.append({'case': 'rsrs_degenerate_inputs_not_replaced', 'momentum_finite': bool(np.isfinite(score)), 'r2_finite': bool(np.isfinite(ols[2])),
        'z_finite': bool(np.isfinite(z)), 'warnings': [str(w.message) for w in caught]})
    equal = {'TRIX': [10., 10.], 'MACD': [10., 10., 0.], 'EMV': [10., 10.], 'KD': [10., 10.]}
    cases.append({'case': 'technical_equalities_choose_first_golden_branch', **technical_case(directory, equal, equal)})
    t = {'TRIX': [9.98, 10.], 'MACD': [9.98, 10., 0.], 'EMV': [9., 10.], 'KD': [9., 10.]}
    y = {'TRIX': [9.94, 10.], 'MACD': [9.94, 10., 0.], 'EMV': [9., 10.], 'KD': [9., 10.]}
    cases.append({'case': 'trix_near_cross_unreachable_macd_reachable', **technical_case(directory, t, y)})
    ns2 = {'pd': pd, 'BytesIO': io.BytesIO, 'datetime': SimpleNamespace(date=date), 'print': lambda *a: None,
        'log': SimpleNamespace(info=lambda *a: None), 'g': SimpleNamespace(buylist=['a', 'b', 'c'], selllist=[])}
    sha2 = selected(directory, 2, ('before_market_open', 'market_open', 'market_close', 'after_close_brain'), ns2)
    today = datetime(2022, 1, 4, 14, 55); ctx = SimpleNamespace(current_dt=today, portfolio=SimpleNamespace(positions={}, available_cash=0.))
    ns2.update(get_trade_days=lambda **kw: [date(2022, 1, 3), date(2022, 1, 4)], read_file=lambda name: (_ for _ in ()).throw(FileNotFoundError(name)))
    try: ns2['before_market_open'](ctx)
    except FileNotFoundError as exc: cases.append({'case': 'brain_file_not_initialized', 'error': str(exc)})
    else: raise AssertionError('Missing original brain file silently initialized')
    orders = []; ns2['g'].buylist = ['a', 'b', 'c']; ns2['g'].selllist = []
    ns2.update(get_current_data=lambda: {s: SimpleNamespace(paused=False) for s in ('a', 'b', 'c')},
        order_target_value=lambda *a: orders.append(list(a)), write_file=lambda *a, **kw: None)
    ns2['market_open'](SimpleNamespace(current_dt=today, portfolio=SimpleNamespace(positions={s: object() for s in ('a', 'b', 'c')}, available_cash=60000.)))
    cases.append({'case': 'open_all_candidates_and_list_mutation', 'orders': orders, 'remaining_buylist': ns2['g'].buylist.copy()})
    for name, price, days in (('minus10_exact', 90., 2), ('minus10_below', 89.99, 2), ('plus50_exact', 150., 2),
        ('plus50_above', 150.01, 2), ('minus5_binary_boundary', 95., 4)):
        orders = []; ns2['g'].buylist = []; ns2.update(order_target=lambda *a: orders.append(list(a)) or object(), get_trade_days=lambda *a, days=days, **kw: list(range(days)))
        ctx.portfolio.positions = {'a': SimpleNamespace(avg_cost=100., price=price, value=1000., init_time=datetime(2022, 1, 1))}
        ns2['market_close'](ctx)
        cases.append({'case': f'eye_brain_{name}', 'ret': price / 100 - 1, 'sold': bool(orders)})
    eye = pd.DataFrame([['2022-01-04', 'a', 60, 100, 1, 30, 10], ['2022-01-03', 'b', 60, 100, 1, 30, 10]],
        columns=['date', 'code', 'cir_m', 'pe', 'pb', 'cyf', 'f10']); writes = []
    ns2.update(read_file=lambda name: eye.to_csv(index=False).encode(), get_current_data=lambda: {s: SimpleNamespace(last_price=10.) for s in ('a', 'b')},
        get_price=lambda *a, **kw: pd.DataFrame({'close': np.linspace(100., 120., 30)}), write_file=lambda *a, **kw: writes.append(list(a)))
    ctx.portfolio.positions = {}; ns2['after_close_brain'](ctx)
    cases.append({'case': 'brain_only_today_eye_not_comment_five_days', 'buylist': ns2['g'].buylist, 'writes': writes})
    return {'not_a_backtest': True, 'source_ast_sha256': [sha0, sha1, trade_sha, sha2], 'cases': cases,
        'limits': ['Synthetic library samples and stub order intentions only; no fills/NAV/cost or alternate ledger',
            'Source economic rules and Python2 stop are not rewritten']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study implementation changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch31 long CAPM/weekly mixed-seed RSRS/momentum and original defect diagnostics frozen')


def worker(root, directory, endpoint):
    import akshare as ak
    import baostock as bs
    from observe.data.sources.baostock import BaoStock
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
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(io.StringIO()):
            if query['provider'] == 'baostock':
                with BaoStock(root).session():
                    result = bs.query_profit_data(**query['parameters']); values = []
                    require(result.error_code == '0', f'Provider error: {result.error_code}/{result.error_msg}')
                    while result.next(): values.append(result.get_row_data()); require(len(values) <= 100, 'ROE response cap exceeded')
                    frame = pd.DataFrame(values, columns=result.fields)
            else: frame = getattr(ak, query['function'])(**query['parameters'])
        require(len(frame) and len(frame) <= 100000, 'Empty/excessive provider sample')
        path = raw.save(root, 'batch31_dependency_probe', endpoint, directory.name, frame)
        row.update(status='sample', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame.columns)}],
            limit=query['limit'], strict_usable=False)
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Probe implementation differs'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['published'] is False, 'Probe binding differs')
        for item in [*row['files'], *row['wire']]: require(file_sha(item['file']) == item['sha256'], 'Probe artifact changed')
        if row['status'] == 'sample':
            require(row['strict_usable'] is False and row['limit'] == query['limit'] and len(row['files']) == 1, 'Unproven sample admitted')
            item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(len(frame) == item['rows'] > 0 and list(frame.columns) == item['columns'], 'Sample profile differs')
        else: require(row['status'] in ('failed', 'timeout') and not row['files'] and row.get('error'), 'Failed probe accepted data')
        rows.append(row)
    return rows


def probe(root, directory):
    binding(root, directory)
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch31', 'worker', '--root', root, '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists(): save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query,
                    'api_sha256': file_sha(directory / 'existing-apis.json'), 'status': 'timeout', 'published': False, 'files': [], 'wire': [], 'error': 'Child exceeded45s'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch31 ROE/announced holders/ETF input supplementation attempts archived; no publication')


def offline_catalog(root, directory):
    rows = []; proof_source = Path(root) / 'catalog/strategies/implementation-evidence.json'
    require(file_sha(proof_source) == file_sha(directory / 'implementation-evidence.before.json'), 'Implementation registry changed')
    for name, catalog in (('api', read(directory / 'api-scan.json')['catalog']), ('final', read(directory / 'source-reviews/review.json')['catalog'])):
        folder = directory / f'catalog-offline-{name}'; folder.mkdir(exist_ok=False)
        target = folder / 'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True); shutil.copyfile(proof_source, target)
        values = {k: v for k, v in REVIEWS.items() if name == 'final' or k not in SOURCES}
        with patch.dict(REVIEWS, values, clear=True): result = catalog_strategies(folder, 'repo/量化策略源代码')
        require(result['catalog_id'] == catalog['catalog_id'], 'Offline catalog identity differs'); pairs = []
        for file in ('catalog.json', 'summary.json', 'catalog.parquet'):
            original = Path(catalog['output']) / file; repeated = Path(result['output']) / file
            require(file_sha(original) == file_sha(repeated), 'Offline catalog bytes differ')
            pairs.append({'name': file, 'original_file': str(original), 'repeated_file': str(repeated), 'sha256': file_sha(original)})
        rows.append({'stage': name, 'catalog_id': catalog['catalog_id'], 'files': pairs})
    return {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_evidence_sha256': file_sha(proof_source), 'catalogs': rows}


def validate_catalogs(root, directory):
    validate_scan(directory); off = read(directory / 'offline-catalog.json')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['implementation_evidence_sha256'] == file_sha(Path(root) / 'catalog/strategies/implementation-evidence.json') == file_sha(directory / 'implementation-evidence.before.json'), 'Offline catalogs/evidence differ')
    require(len(off['catalogs']) == 2 and {r['stage'] for r in off['catalogs']} == {'api', 'final'}, 'Offline catalog stages differ')
    for row in off['catalogs']:
        original = read(directory / ('api-scan.json' if row['stage'] == 'api' else 'source-reviews/review.json'))['catalog']
        require(row['catalog_id'] == original['catalog_id'] and len(row['files']) == 3 and
            {p['name'] for p in row['files']} == {'catalog.json', 'summary.json', 'catalog.parquet'}, 'Offline catalog scope differs')
        for item in row['files']:
            a = Path(original['output']) / item['name']; b = directory / f'catalog-offline-{row["stage"]}/catalog/strategies' / row['catalog_id'] / item['name']
            require(item['original_file'] == str(a) and item['repeated_file'] == str(b) and file_sha(a) == item['sha256'] == file_sha(b), 'Offline catalog binding differs')
    final = read(directory / 'source-reviews/review.json')['catalog']
    require(read(Path(root) / 'catalog/strategies/latest.json')['catalog_id'] == final['catalog_id'], 'Latest catalog changed')


def offline(root, directory):
    before = implementation(); binding(root, directory)
    def denied(*a, **kw): raise AssertionError('Batch31 forbidden-network recheck attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        result = compute(directory); diagnostic = diagnostics(directory); catalogs = offline_catalog(root, directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result); require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline component bytes differ')
    save(directory / 'offline-catalog.json', catalogs); save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0,
        'socket_network_disabled': True, 'sha256': file_sha(directory / 'component-offline.json'), 'implementation_sha256': before, 'not_a_backtest': True})
    validate_catalogs(root, directory); return archive(root, 'Batch31 forbidden-network original components and two full catalogs match byte-for-byte')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch31.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch31_research.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/unit/test_batch17_archive.py', 'tests/unit/test_signal_slots.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_catalogs(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
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
    binding(root, directory); validate_catalogs(root, directory); require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed components differ')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch31 original CAPM/weekly-RSRS/momentum components and API corrections accepted; original trading gaps retained')
    save(Path('docs/handoff/2026-10-06-batch31-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'scan': read(directory / 'api-scan.json'), 'checks': checked, 'offline': off,
        'offline_catalog': read(directory / 'offline-catalog.json'), 'protection': protection, 'probes': read(directory / 'probe-results.json'),
        'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'scan', 'start', 'prepare', 'study', 'worker', 'probe', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch31/20261006-capm-weekly-rsrs-finance')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
