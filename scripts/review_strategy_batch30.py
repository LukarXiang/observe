"""Freeze exact MA/KAMA components; absent pools and intraday execution remain blocked."""
import argparse
import ast
import datetime
import hashlib
import inspect
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
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch21 import selected
from scripts import review_strategy_batch29 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch29/20261006-wizard-smart-northbound')
RECEIPT = Path('docs/handoff/2026-10-06-batch29-verification.json')
SOURCES = ('2020年度精选策略/82 次新+小市值+KAMA择时 轮动.txt',
    '2021年度精选策略/41.均线黏合，突破前三十个交易日最高点选股法.txt',
    '2023年度精选策略/70.回踩均线搏反弹，21年来70%，无未来（测试第一弹）.txt')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/kama.py'),
    ('repo/backtrader', 'backtrader/resamplerfilter.py'), ('repo/akshare', 'akshare/stock_feature/stock_hist_em.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch30.py', 'tests/unit/test_batch30_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'source-reviews/review.json', 'input-analysis.json', 'price-input.parquet',
    'five-minute-input.parquet', 'ten-minute-input.parquet', 'minute-coverage.json', 'component-research.json', 'diagnostics.json',
    'existing-apis.json', 'probe-results.json', 'component-offline.json', 'offline-verification.json', 'minute-source-verification.json',
    'minute-tail-coverage.json'}
MINUTE_STOCK = '600519.SH'
QUERIES = {'minute1': {'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600519', 'period': '1',
        'start_date': '2022-01-04 09:30:00', 'end_date': '2022-01-04 15:00:00', 'adjust': ''},
        'limit': 'Latest-five-day endpoint then date filter, not original 09:59 or purchase-time historical minute inputs'},
    'minute5': {'function': 'stock_zh_a_hist_min_em', 'parameters': {'symbol': '600519', 'period': '5',
        'start_date': '2022-01-04 09:30:00', 'end_date': '2022-01-04 15:00:00', 'adjust': ''},
        'limit': 'Provider recent five-minute sample cannot prove platform ten-minute unfinished-bar semantics'},
    'turnover': {'function': 'stock_zh_a_hist', 'parameters': {'symbol': '600519', 'period': 'daily',
        'start_date': '20220104', 'end_date': '20220106', 'adjust': '', 'timeout': 10},
        'limit': 'One-stock final turnover is not original all-stock vintage or circulating market cap'}}


def implementation(): return {p: file_sha(p) for p in FILES}


def api_evidence():
    import akshare as ak
    rows = []
    for name in sorted({q['function'] for q in QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        rows.append({'function': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'version': ak.__version__, 'apis': rows, 'queries': QUERIES}


def binding(root, directory=None):
    receipt = read(RECEIPT); store = Store(root); state = store.state(SNAPSHOT)
    require(receipt['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT, 'Accepted receipt differs')
    price = ACCEPTED / 'price-input.parquet'
    require(file_sha(price) == receipt['checks']['evidence_sha256']['price-input.parquet'], 'Accepted prices changed')
    baseline = ACCEPTED / 'baseline.json'
    require(file_sha(baseline) == receipt['checks']['evidence_sha256']['baseline.json'], 'Accepted baseline changed')
    accepted_tables = read(baseline)['published']['tables']
    require(all(state['tables'][t] == accepted_tables[t] for t in ('bars_5m', 'minute_universe')), 'Accepted minute references changed')
    partitions = []
    for table in ('bars_5m', 'minute_universe'):
        for part, entry in sorted(state['tables'][table].items()):
            path = store.root / entry['file']
            partitions.append({'table': table, 'part': part, 'entry': entry, 'file': str(path), 'sha256': file_sha(path)})
    result = {'snapshot': SNAPSHOT, 'receipt_file': str(RECEIPT), 'receipt_sha256': file_sha(RECEIPT),
        'price_file': str(price), 'price_sha256': file_sha(price), 'stock_binding': previous.binding(root, ACCEPTED),
        'minute_partitions': partitions, 'minute_stock': MINUTE_STOCK, 'not_a_backtest': True}
    if directory is not None:
        require(result == read(directory / 'input-binding.json'), 'Batch30 input binding changed')
        if (directory / 'minute-source-verification.json').exists(): validate_minute_verification(directory, result)
    return result


def verify_minute_frame(frame, entry, table):
    require(len(frame) == entry['rows'], 'Minute partition row count differs')
    actual = fingerprint(frame); accepted = actual; restored = []
    if actual != entry['sha']:
        candidate = frame.copy()
        for name in frame.select_dtypes(include=['datetime', 'datetimetz']).columns:
            if frame[name].dt.unit != 'ms': continue
            seconds = frame[name].dt.as_unit('s')
            if seconds.dt.as_unit('ms').equals(frame[name]): candidate[name] = seconds; restored.append(name)
        require(restored and fingerprint(candidate) == entry['sha'], 'Minute partition fingerprint differs')
        accepted = fingerprint(candidate)
    keys = ['bar_end', 'instrument'] if table == 'bars_5m' else ['year', 'instrument']
    require(not frame[keys].isna().any().any() and not frame.duplicated(keys).any(), 'Minute partition keys invalid')
    return {'actual_rows': len(frame), 'canonical_fingerprint': actual, 'accepted_fingerprint': accepted, 'legacy_seconds_columns': restored}


def verify_minute_sources(directory, bound):
    rows = []
    for row in bound['minute_partitions']:
        before = file_sha(row['file']); require(before == row['sha256'], 'Minute source changed before content verification')
        frame = pd.read_parquet(row['file']); proof = verify_minute_frame(frame, row['entry'], row['table'])
        require(file_sha(row['file']) == before, 'Minute source changed during content verification')
        rows.append({**row, **proof})
    save(directory / 'minute-source-verification.json', {'snapshot': SNAPSHOT, 'input_binding_sha256': file_sha(directory / 'input-binding.json'),
        'partitions': rows, 'status': 'ok', 'not_a_backtest': True})
    validate_minute_verification(directory, bound)


def validate_minute_verification(directory, bound):
    doc = read(directory / 'minute-source-verification.json')
    require(doc['status'] == 'ok' and doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and
        doc['input_binding_sha256'] == file_sha(directory / 'input-binding.json') and len(doc['partitions']) == len(bound['minute_partitions']), 'Minute source proof scope differs')
    for row, original in zip(doc['partitions'], bound['minute_partitions'], strict=True):
        require(all(row[k] == v for k, v in original.items()) and row['actual_rows'] == original['entry']['rows'] and
            row['accepted_fingerprint'] == original['entry']['sha'] and
            (row['canonical_fingerprint'] == row['accepted_fingerprint'] or bool(row['legacy_seconds_columns'])), 'Minute source proof differs')


def start(root, directory):
    checkpoint(root, directory, 'Batch30 exact KAMA/MA adhesion/pullback research started')
    save(directory / 'input-binding.json', binding(root)); save(directory / 'existing-apis.json', api_evidence())
    old_path = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    old = {r['path']: r for r in read(old_path)}; folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    for name in SOURCES:
        path = Path('repo/量化策略源代码') / name; copied = folder / f'{strategy_id(name)}.source'
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'], 'Source reviewed/changed')
        shutil.copyfile(path, copied); text, encoding = read_source(path)
        rows.append({'source_path': str(path), 'source_sha256': file_sha(path), 'source_copy': str(copied),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repo, name in REFERENCES:
        path = Path(repo) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'KAMA efficiency/smoothing formula only; Backtrader mean seed is not TA-Lib seed; session resampling and API range limits'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(directory / 'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(old_path), 'previous_catalog_sha256': file_sha(old_path), 'snapshot': SNAPSHOT,
        'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch30 three complete source reviews frozen; original trading dependencies retained')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and len(doc['sources']) == len(SOURCES), 'Source scope differs')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(SOURCES, doc['sources'], strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Reviewed source changed')
    require(len(doc['references']) == len(REFERENCES), 'References missing')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo) / name) and file_sha(row['file']) == row['sha256'] == file_sha(row['copy']), 'Reference changed')


def function(directory, number, name):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == row['source_sha256'], 'Selected source changed')
    nodes = [n for n in ast.parse(read_source(Path(row['source_copy']))[0]).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'Missing/ambiguous source function'); return nodes[0]


def compiled(nodes):
    module = ast.Module(body=nodes, type_ignores=[])
    return compile(module, '<frozen-original-component>', 'exec'), hashlib.sha256(ast.dump(module).encode()).hexdigest()


def technical_kernel(directory):
    fn = function(directory, 1, 'get_stocks_tobuy')
    first = next(k for k, n in enumerate(fn.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'close')
    return compiled([fn.body[0], *fn.body[first:-1]])


def pullback_kernel(directory):
    fn = function(directory, 2, 'select_ticks')
    loops = [n for n in fn.body if isinstance(n, ast.For) and any(isinstance(x, ast.Name) and x.id == 'tdf' for x in ast.walk(n))]
    require(len(loops) == 1, 'Missing/ambiguous pullback loop')
    score = function(directory, 2, 'get_stocks_score')
    body = [n for n in score.body if isinstance(n, ast.For) and any(isinstance(x, ast.Name) and x.id == 'box_ratio' for x in ast.walk(n))]
    require(len(body) == 1, 'Missing/ambiguous score loop')
    return compiled(loops), compiled(body)


def minute_grid(day):
    day = pd.Timestamp(day).normalize()
    return pd.DatetimeIndex([day + pd.Timedelta(minutes=m) for m in (*range(575, 691, 5), *range(785, 901, 5))])


def aggregate_ten(frame, trading_dates):
    require(not frame.duplicated(['bar_end', 'instrument']).any() and set(frame.instrument) <= {MINUTE_STOCK}, 'Minute scope/duplicates differ')
    dates = {str(d): k for k, d in enumerate(trading_dates)}; rows = []; rejected = []; prior = None; segment = -1
    for day, group in frame.groupby(frame.bar_end.dt.strftime('%Y-%m-%d'), sort=True):
        group = group.sort_values('bar_end')
        reason = None
        if day not in dates: reason = 'not_verified_trading_date'
        elif not pd.DatetimeIndex(group.bar_end).equals(minute_grid(day)): reason = 'incomplete_or_noncanonical_48_bar_session'
        elif not np.isfinite(group[['open', 'high', 'low', 'close', 'volume', 'amount', 'back_factor']]).all().all(): reason = 'nonfinite_ohlc_or_factor'
        elif not group[['open', 'high', 'low', 'close', 'back_factor']].gt(0).all().all() or not group[['volume', 'amount']].ge(0).all().all(): reason = 'invalid_price_volume_or_factor'
        elif not ((group.high >= group[['open', 'close']].max(axis=1)) & (group.low <= group[['open', 'close']].min(axis=1))).all(): reason = 'invalid_ohlc'
        elif group.back_factor.nunique() != 1 or not group.adjustment_status.eq('usable').all(): reason = 'unknown_or_conflicting_adjustment'
        if reason:
            rejected.append({'date': day, 'rows': len(group), 'reason': reason}); continue
        if prior is None or dates[day] != prior + 1: segment += 1
        prior = dates[day]; values = group.to_dict('records')
        for k in range(0, 48, 2):
            a, b = values[k:k + 2]
            rows.append({'bar_end': b['bar_end'], 'instrument': MINUTE_STOCK, 'open': a['open'], 'high': max(a['high'], b['high']),
                'low': min(a['low'], b['low']), 'close': b['close'], 'volume': a['volume'] + b['volume'], 'amount': a['amount'] + b['amount'],
                'back_factor': b['back_factor'], 'close_adj': b['close'] * b['back_factor'], 'segment': segment})
    out = pd.DataFrame(rows)
    missing = sorted(set(dates) - set(frame.bar_end.dt.strftime('%Y-%m-%d')))
    return out, {'accepted_days': len(rows) // 24, 'ten_minute_rows': len(rows), 'rejected_days': rejected, 'absent_verified_dates': missing,
        'segments': segment + 1, 'session_policy': 'Only complete 48-bar sessions; pair within morning/afternoon; missing trading days break warmup',
        'not_a_backtest': True, 'platform_equivalent': False}


def prepare(root, directory):
    bound = binding(root, directory); store = Store(root); state = store.state(SNAPSHOT)
    require(not any((directory / n).exists() for n in ('price-input.parquet', 'five-minute-input.parquet', 'ten-minute-input.parquet')), 'Prepared inputs exist')
    if not (directory / 'minute-source-verification.json').exists(): verify_minute_sources(directory, bound)
    price = pd.read_parquet(ACCEPTED / 'price-input.parquet'); pool = list(previous.previous.prior.INSTRUMENTS)
    extra = store.load_state(state, 'bars_1d', columns=['date', 'instrument', 'open', 'volume', 'close', 'is_trading'], filters=[('instrument', 'in', pool)])
    extra['date'] = pd.to_datetime(extra.date).dt.strftime('%Y-%m-%d')
    merged = price.merge(extra, on=['date', 'instrument'], suffixes=('', '_raw'), validate='one_to_one')
    require(len(merged) == len(price) and merged.is_trading.eq(merged.is_trading_raw).all(), 'Daily input scope changed')
    pd.testing.assert_series_equal(merged.close, merged.close_raw, check_names=False)
    price = merged.drop(columns=['close_raw', 'is_trading_raw']); price['open_adj'] = price.open * price.back_factor
    price.to_parquet(directory / 'price-input.parquet', index=False)
    minute = store.load_state(state, 'bars_5m', filters=[('instrument', '=', MINUTE_STOCK)]).sort_values('bar_end').reset_index(drop=True)
    require(len(minute) and not minute.duplicated(['bar_end', 'instrument']).any(), 'Minute sample empty/duplicate')
    minute['date'] = minute.bar_end.dt.strftime('%Y-%m-%d')
    stock = price[price.instrument.eq(MINUTE_STOCK)].sort_values('date')
    minute = minute.merge(stock[['date', 'instrument', 'back_factor', 'adjustment_status']], on=['date', 'instrument'], how='left', validate='many_to_one')
    dates = stock[stock.is_trading & stock.date.between(minute.date.min(), minute.date.max())].date.tolist()
    ten, coverage = aggregate_ten(minute, dates); require(len(ten), 'No complete ten-minute component sessions')
    minute.to_parquet(directory / 'five-minute-input.parquet', index=False); ten.to_parquet(directory / 'ten-minute-input.parquet', index=False)
    coverage.update(snapshot=SNAPSHOT, instrument=MINUTE_STOCK, five_minute_rows=len(minute), verified_trading_dates=dates,
        input_sha256=file_sha(directory / 'five-minute-input.parquet'), output_sha256=file_sha(directory / 'ten-minute-input.parquet'))
    save(directory / 'minute-coverage.json', coverage)
    info = {}
    for name in ('price', 'five-minute', 'ten-minute'):
        path = directory / f'{name}-input.parquet'; frame = pd.read_parquet(path); col = 'date' if name == 'price' else 'bar_end'
        info[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'first': str(frame[col].min()), 'last': str(frame[col].max())}
    save(directory / 'input-analysis.json', {'snapshot': SNAPSHOT, 'inputs': info, 'pool': pool, 'minute_stock': MINUTE_STOCK,
        'not_a_backtest': True, 'daily_policy': 'Verified traded rows only; each 250/260 window anchored to final raw price; platform pause-fill equivalence unproved',
        'minute_policy': 'Completed ten-minute bars only, original 09:59 context uses last completed09:50; unfinished platform bars unproved',
        'limits': ['Ten stocks do not replace historical pools', 'One-stock ten-minute component is not IPO pool or original minute trading',
            'No substitution for historical circulating market cap, turnover universe, stock names or minute peak']})
    return archive(root, 'Batch30 accepted long daily and completed ten-minute component inputs frozen')


def inputs(directory):
    doc = read(directory / 'input-analysis.json'); frames = {}
    validate_minute_verification(directory, read(directory / 'input-binding.json'))
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] and doc['minute_stock'] == MINUTE_STOCK and
        doc['pool'] == list(previous.previous.prior.INSTRUMENTS) and set(doc['inputs']) == {'price', 'five-minute', 'ten-minute'}, 'Input scope differs')
    for name, row in doc['inputs'].items():
        path = directory / f'{name}-input.parquet'; require(str(path) == row['file'] and file_sha(path) == row['sha256'], 'Component input changed')
        frame = pd.read_parquet(path); col = 'date' if name == 'price' else 'bar_end'
        require(len(frame) == row['rows'] and str(frame[col].min()) == row['first'] and str(frame[col].max()) == row['last'] and
            not frame.duplicated([col, 'instrument']).any(), 'Input profile differs'); frames[name] = frame
    return frames


def reference_kama(values, period=312):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) >= period + 2 and np.isfinite(x).all() and (x > 0).all(), 'Invalid KAMA input')
    out = np.full(len(x), np.nan); value = float(x[period - 1]); slow = 2 / 31; fast_minus_slow = 2 / 3 - slow
    for k in range(period, len(x)):
        direction = abs(float(x[k] - x[k - period])); volatility = math.fsum(abs(float(x[j] - x[j - 1])) for j in range(k - period + 1, k + 1))
        efficiency = 1. if volatility <= direction or volatility < 1e-14 else direction / volatility
        smooth = (efficiency * fast_minus_slow + slow) ** 2; value += (float(x[k]) - value) * smooth; out[k] = value
    return out


def minute_tail(directory, data):
    stock = data['price'][data['price'].instrument.eq(MINUTE_STOCK) & data['price'].is_trading].sort_values('date')
    first = str(data['five-minute'].bar_end.min())[:10]; last = str(data['five-minute'].bar_end.max())[:10]
    return {'snapshot': SNAPSHOT, 'instrument': MINUTE_STOCK, 'not_a_backtest': True, 'platform_equivalent': False,
        'price_sha256': file_sha(directory / 'price-input.parquet'), 'five_minute_sha256': file_sha(directory / 'five-minute-input.parquet'),
        'minute_first': first, 'minute_last': last, 'frozen_verified_daily_last': stock.date.max(),
        'missing_tail_dates': stock[stock.date.gt(last)].date.tolist(),
        'limit': 'Zero absent dates in original coverage applies only between its first/last minute dates, not the frozen daily endpoint'}


def coverage(root, directory):
    binding(root, directory); data = inputs(directory)
    save(directory / 'minute-tail-coverage.json', minute_tail(directory, data))
    return archive(root, 'Batch30 minute coverage tail16 verified trading days missing; initial interval proof retained unchanged')


def reference_means(values, periods):
    return {p: np.array([math.fsum(map(float, values[k - p + 1:k + 1])) / p if k >= p - 1 else np.nan for k in range(len(values))]) for p in periods}


def adhesion_reference(close, opening, means, k):
    row = {'close': float(close[k]), 'open': float(opening[k]), 'highest': float(close[k - 11]), 'lowest': float(close[k - 11]), 'close250': means[250][k]}
    for offset, suffix in ((0, ''), (9, '_1'), (18, '_2')):
        row.update({f'close{p}{suffix}': means[p][k - offset] for p in (13, 34, 55)})
    passed = row['close'] > row['highest'] and row['close13'] > row['close34'] > row['close55'] and row['close'] >= row['open']
    for suffix in ('_1', '_2'):
        a, b, c = (row[f'close{p}{suffix}'] for p in (13, 34, 55))
        passed &= abs((b - a) / a) <= .03 and abs((c - b) / b) <= .03 and abs((c - a) / a) <= .03
    return row, bool(passed)


def pullback_reference(close, opening, means, k):
    six = slice(k - 5, k + 1); a, b, c = (means[p] for p in (21, 55, 120))
    passed = sum(a[six] > .98 * b[six]) >= 5 and sum(b[six] > .98 * c[six]) >= 5 and sum(close[six] > .98 * b[six]) >= 5
    passed &= close[k - 1] <= opening[k - 1] and close[k - 1] <= b[k - 1] and close[k] >= b[k] and b[k] >= b[k - 1]
    box = 0.; array = 0.
    for j in range(k - 20, k):
        values = [float(means[p][j]) for p in (13, 21, 55)]; median = sorted(values)[1]
        box += math.sqrt(math.fsum(((v - median) / median) ** 2 for v in values) / 3)
    for j in range(k - 4, k + 1):
        a, b, c = (float(means[p][j]) for p in (13, 21, 55))
        if a >= b >= c: array += (b - c) / c + (a - b) / b
    return bool(passed), box - array


def daily_components(directory, price):
    technical, tech_sha = technical_kernel(directory); (filters, filter_sha), (scoring, score_sha) = pullback_kernel(directory)
    helper = {'talib': talib, 'np': np}
    helper_sha = selected(directory, 2, ('get_ma', 'get_bigger_than_val_counter', 'get_std_percentage', 'get_avg_array'), helper)
    groups = []; boundaries = []; paused = 0
    for instrument, frame in price.groupby('instrument', sort=True):
        frame = frame.sort_values('date'); paused += int((~frame.is_trading).sum()); frame = frame[frame.is_trading].reset_index(drop=True)
        require(frame.adjustment_status.eq('usable').all() and np.isfinite(frame[['close_adj', 'open_adj', 'high_adj', 'low_adj', 'back_factor', 'volume']]).all().all(), 'Unknown daily input')
        x = frame.close_adj.to_numpy(); opening = frame.open_adj.to_numpy(); factors = frame.back_factor.to_numpy(); means = reference_means(x, (13, 21, 34, 55, 120, 250))
        close_holder = {}; ns = {'pd': pd, 'np': np, 'stock_list1': ['component'],
            'g': SimpleNamespace(previous_buylist={}), 'context': SimpleNamespace(current_dt='2000-01-01')}
        ns['get_price'] = lambda stocks, **kw: {kw['fields']: close_holder[kw['fields']].tail(kw['count']).copy()}
        windows41 = windows70 = buys41 = passes70 = positive70 = 0; mean_max = score_max = 0.; digest = hashlib.sha256()
        for k in range(249, len(frame)):
            scale = factors[k]; close_holder.update(close=pd.DataFrame({'component': x[k - 249:k + 1] / scale}),
                open=pd.DataFrame({'component': [opening[k] / scale]}))
            exec(technical, ns); actual = ns['All'].loc['component'].to_dict(); reference, expected = adhesion_reference(x, opening, means, k)
            reference = {n: v / scale for n, v in reference.items()}
            mean_max = max(mean_max, max(abs(actual[n] - v) / max(1., abs(v)) for n, v in reference.items()))
            signal = not ns['stock_list'].empty; windows41 += 1; buys41 += int(signal)
            if signal != expected: boundaries.append({'source': '41', 'instrument': instrument, 'date': frame.date.iloc[k], 'original': signal, 'reference': expected,
                'original_values': actual, 'reference_values': reference})
            digest.update(json.dumps({'date': frame.date.iloc[k], '41': actual, 'signal41': signal}, sort_keys=True).encode())
            if k < 259: continue
            window = frame.iloc[k - 259:k + 1]; df = pd.DataFrame({'code': 'component', 'close': window.close_adj.to_numpy() / scale,
                'open': window.open_adj.to_numpy() / scale, 'high': window.high_adj.to_numpy() / scale, 'low': window.low_adj.to_numpy() / scale, 'volume': window.volume.to_numpy()})
            local = dict(helper, df=df, data_frame=df, tick_list=['component'], filter_list=[], score={}, avg_score={})
            exec(filters, local); exec(scoring, local); passed = not local['filter_list']; score = float(local['score']['component'])
            expected_pass, expected_score = pullback_reference(x, opening, means, k); score_max = max(score_max, abs(score - expected_score))
            windows70 += 1; passes70 += int(passed); positive70 += int(score > 0)
            if passed != expected_pass or (score > 0) != (expected_score > 0):
                boundaries.append({'source': '70', 'instrument': instrument, 'date': frame.date.iloc[k], 'original_filter': passed,
                    'reference_filter': expected_pass, 'original_score': score, 'reference_score': expected_score})
            digest.update(json.dumps({'date': frame.date.iloc[k], 'filter70': passed, 'score70': score}, sort_keys=True).encode())
        require(mean_max < 1e-10 and score_max < 1e-10, 'Daily component numeric mismatch')
        groups.append({'instrument': instrument, 'windows41': windows41, 'windows70': windows70, 'technical_passes41': buys41,
            'technical_passes70': passes70, 'positive_scores70': positive70, 'mean_max_relative_difference': mean_max,
            'score_max_absolute_difference': score_max, 'component_sha256': digest.hexdigest()})
    return {'groups': groups, 'known_paused_rows_excluded': paused, 'comparison_boundaries': boundaries,
        'source_ast_sha256': {'technical41': tech_sha, 'filter70': filter_sha, 'score70': score_sha, 'helpers70': helper_sha},
        'limits': ['Technical passes and positive scores are not pool selections, trades or returns', 'Stable-sum comparator is diagnostic only; source calculation retained']}


def kama_components(directory, ten):
    holder = {}; ns = {'g': SimpleNamespace(kama_days=13, return_radio=.999)}
    def price(security, **kw):
        require(security == MINUTE_STOCK and kw['frequency'] == '10m' and kw['count'] == 360 and kw['fields'] == ['close'], 'Original KAMA request differs')
        return pd.DataFrame({'close': holder['close']})
    def engine(values, **kw):
        holder['actual'] = talib.KAMA(values, **kw); return holder['actual']
    ns.update(get_price=price, tb=SimpleNamespace(KAMA=engine)); sha = selected(directory, 0, ('get_kama_single',), ns)
    rows = []; maximum = 0.; boundaries = []; digest = hashlib.sha256(); rejected = 0
    for segment, group in ten.groupby('segment', sort=True):
        group = group.sort_values('bar_end').reset_index(drop=True)
        for k in group.index[(group.bar_end.dt.hour == 9) & (group.bar_end.dt.minute == 50)]:
            if k < 359: rejected += 1; continue
            x = group.close_adj.iloc[k - 359:k + 1].to_numpy() / group.back_factor.iloc[k]; holder['close'] = x
            end = group.bar_end.iloc[k]; context = SimpleNamespace(current_dt=end.normalize() + pd.Timedelta(hours=9, minutes=59))
            signal = int(ns['get_kama_single'](context, MINUTE_STOCK)); actual = holder['actual']; reference = reference_kama(x)
            error = float(np.max(np.abs(actual[312:] - reference[312:]) / np.maximum(1., np.abs(reference[312:])))); maximum = max(maximum, error)
            ratio = float(actual[-1] / actual[-2]); expected_ratio = float(reference[-1] / reference[-2]); expected = -1 if expected_ratio < .999 else 1 if expected_ratio > 1.01 else 0
            if signal != expected: boundaries.append({'bar_end': str(end), 'original': signal, 'reference': expected, 'ratio': ratio, 'reference_ratio': expected_ratio})
            row = {'bar_end': str(end), 'segment': int(segment), 'signal': signal, 'ratio': ratio, 'relative_error': error}; rows.append(row)
            digest.update(actual[312:].tobytes()); digest.update(json.dumps(row, sort_keys=True).encode())
    require(rows and maximum < 1e-10, 'KAMA component numeric mismatch')
    return {'source_ast_sha256': sha, 'windows': len(rows), 'finite_values': len(rows) * 48, 'first': rows[0]['bar_end'], 'last': rows[-1]['bar_end'],
        'insufficient_contiguous_windows': rejected, 'max_relative_difference': maximum, 'comparison_boundaries': boundaries,
        'signal_counts': {str(v): sum(r['signal'] == v for r in rows) for v in (-1, 0, 1)}, 'component_sha256': digest.hexdigest(),
        'seed_policy': 'TA-Lib seed at close[311], efficiency312, fast2/slow30; recomputed for each360-row window',
        'decision_policy': 'Last completed09:50 bar only for09:59 context; platform unfinished-bar semantics unproved'}


def compute(directory):
    validate_sources(directory); data = inputs(directory)
    if (directory / 'minute-tail-coverage.json').exists():
        require(minute_tail(directory, data) == read(directory / 'minute-tail-coverage.json'), 'Minute tail coverage differs')
    require(talib.get_compatibility() == 0 and talib.get_unstable_period('KAMA') == 0, 'Indicator settings differ')
    ten, coverage = aggregate_ten(data['five-minute'], read(directory / 'minute-coverage.json')['verified_trading_dates'])
    pd.testing.assert_frame_equal(ten, data['ten-minute'])
    saved = read(directory / 'minute-coverage.json')
    require(all(saved[k] == v for k, v in coverage.items()) and saved['input_sha256'] == file_sha(directory / 'five-minute-input.parquet') and
        saved['output_sha256'] == file_sha(directory / 'ten-minute-input.parquet'), 'Minute coverage binding differs')
    return {'not_a_backtest': True, 'original_strategy_complete': False, 'snapshot': SNAPSHOT,
        'input_sha256': {n: file_sha(directory / f'{n}-input.parquet') for n in data},
        'talib': {'python': talib.__version__, 'c_core': talib.__ta_version__.decode(), 'compatibility': 0, 'KAMA_unstable': 0},
        'daily': daily_components(directory, data['price']), 'kama': kama_components(directory, data['ten-minute']),
        'minute_coverage': coverage, 'strategy_results': []}


def diagnostics(directory):
    cases = []; ns = {'tb': talib, 'g': SimpleNamespace(kama_days=13, return_radio=.999),
        'get_price': lambda *a, **kw: pd.DataFrame({'close': np.ones(20)}), 'order_target_value': lambda *a: None}
    sha0 = selected(directory, 0, ('get_kama_single', 'market_end', 'handle_data'), ns)
    context = SimpleNamespace(current_dt=datetime.datetime(2022, 1, 4, 9, 59), portfolio=SimpleNamespace(positions={}))
    cases.append({'case': 'short_kama_returns_neutral', 'signal': int(ns['get_kama_single'](context, MINUTE_STOCK))})
    ns['get_price'] = lambda *a, **kw: pd.DataFrame({'close': np.full(360, np.nan)})
    cases.append({'case': 'missing_kama_returns_neutral', 'signal': int(ns['get_kama_single'](context, MINUTE_STOCK))})
    selected_calls = []; ns.update(select_stocks=lambda *a: selected_calls.append('select') or [], adjust_position=lambda *a: None)
    ns['g'].days_counter = 0; ns['g'].buy_period = 2
    ns['handle_data'](context, None); ns['market_end'](context); ns['handle_data'](context, None)
    cases.append({'case': 'two_day_clock', 'selection_calls': len(selected_calls), 'counter': ns['g'].days_counter})
    ns1 = {'pd': pd, 'np': np, 'print': lambda *a: None, 'log': SimpleNamespace(info=lambda *a: None), 'get_stocks_tobuy': lambda *a: [],
        'get_price': lambda *a, **kw: {'close': pd.DataFrame({'a': np.arange(16.) + 100})},
        'g': SimpleNamespace(previous_buylist={'old': 30, 'new': 29}, stock_tosell=['stale'])}
    sha1 = selected(directory, 1, ('after_trading_end', 'buy_stock', 'sell_stock'), ns1)
    ns1['after_trading_end'](SimpleNamespace(current_dt=context.current_dt, portfolio=SimpleNamespace(positions={'a': object()})))
    cases.append({'case': 'empty_sell_does_not_clear', 'sell_list': ns1['g'].stock_tosell.copy(), 'previous_buylist': ns1['g'].previous_buylist.copy()})
    orders = []; ns1.update(order_target_value=lambda *a: orders.append(list(a)))
    ns1['g'].stock_tosell = []; ns1['g'].my_stock_today = ['candidate']
    portfolio = SimpleNamespace(positions={str(k): object() for k in range(6)}, total_value=600., available_cash=100.)
    ns1['buy_stock'](SimpleNamespace(portfolio=portfolio))
    cases.append({'case': 'six_positions_still_submit_and_mark', 'orders': orders, 'marked': ns1['g'].previous_buylist.get('candidate') == 0})
    ns2 = {'np': np, 'talib': talib, 'log': SimpleNamespace(info=lambda *a: None)}
    sha2 = selected(directory, 2, ('get_bigger_than_val_counter', 'get_avg_array', 'get_std_percentage', 'stop_loss', 'trade'), ns2)
    cases.append({'case': 'five_of_six_with_two_percent_tolerance', 'count': int(ns2['get_bigger_than_val_counter']([99, 97, 99, 99, 99, 99], 6, [100] * 6))})
    cases.append({'case': 'ascending_ma_has_zero_array_score', 'value': float(ns2['get_avg_array']([1, 2, 3]))})
    cases.append({'case': 'median_rms_not_standard_deviation', 'value': float(ns2['get_std_percentage']([1, 2, 6]))})
    for name, price, peak, days in (('strict_minus7_boundary', 93., 100., 2), ('drawdown_current_price_denominator', 119., 122.65, 2),
        ('first_day_losing', 99., 100., 1), ('third_day_below5', 104., 105., 3), ('empty_peak_still_age_exit', 104., np.nan, 3)):
        orders = []; ns2.update(get_price=lambda *a, peak=peak, **kw: pd.DataFrame({'high': [peak]}), hold_days=lambda *a, days=days: days,
            order_target=lambda *a: orders.append(list(a)))
        p = SimpleNamespace(avg_cost=100., price=price, init_time=context.current_dt)
        ns2['stop_loss'](SimpleNamespace(portfolio=SimpleNamespace(positions={'a': p}), previous_date=datetime.date(2022, 1, 3)))
        cases.append({'case': name, 'sold': bool(orders), 'orders': orders})
    orders = []; ns2.update(stop_loss=lambda *a: None, order_value=lambda *a, **kw: orders.append(list(a)))
    ns2['g'] = SimpleNamespace(security=['already_held', 'new']); portfolio = SimpleNamespace(positions={'already_held': object()}, available_cash=100.)
    ns2['trade'](SimpleNamespace(portfolio=portfolio))
    cases.append({'case': 'target_existing_stock_can_add', 'orders': orders})
    return {'not_a_backtest': True, 'source_ast_sha256': [sha0, sha1, sha2], 'cases': cases,
        'limits': ['Stub orders record original intents only; no fills, NAV, fees or alternate ledger']}


def study(root, directory):
    binding(root, directory); before = implementation(); result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation(), 'Study code changed'); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diagnostic)
    return archive(root, 'Batch30 exact long MA and completed ten-minute KAMA components plus source defects archived')


def worker(root, directory, endpoint):
    import akshare as ak
    require(read(directory / 'existing-apis.json') == api_evidence(), 'Probe API changed')
    query = QUERIES[endpoint]; folder = directory / 'probe' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(directory / 'existing-apis.json'),
        'status': 'failed', 'published': False, 'files': [], 'wire': []}; original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(row['wire']) < 10, 'Probe response cap reached'); session.trust_env = False; kwargs['timeout'] = (8, 10)
        response = original(session, method, url, **kwargs); path = folder / f'{len(row["wire"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire'].append({'file': str(path), 'sha256': file_sha(path), 'url': response.url, 'status_code': response.status_code}); return response
    try:
        with patch.object(requests.sessions.Session, 'request', request): frame = getattr(ak, query['function'])(**query['parameters'])
        path = raw.save(root, 'batch30_dependency_probe', endpoint, directory.name, frame)
        row.update(status='success' if len(frame) else 'empty', files=[{'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)}])
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1500]
    save(directory / f'probe-{endpoint}.json', row); return row


def validate_probes(directory):
    require(read(directory / 'existing-apis.json') == api_evidence(), 'API evidence changed'); rows = []
    for endpoint, query in QUERIES.items():
        row = read(directory / f'probe-{endpoint}.json')
        require(row['endpoint'] == endpoint and row['query'] == query and row['published'] is False and
            row['api_sha256'] == file_sha(directory / 'existing-apis.json') and row['status'] in ('failed', 'timeout', 'empty', 'success'), 'Probe binding differs')
        for item in row['wire']: require(file_sha(item['file']) == item['sha256'], 'Probe wire changed')
        if row['status'] in ('success', 'empty'):
            require(len(row['files']) == 1, 'Probe raw missing'); item = row['files'][0]; frame = pd.read_parquet(item['file'])
            require(file_sha(item['file']) == item['sha256'] and len(frame) == item['rows'] and list(frame) == item['columns'] and
                bool(len(frame)) == (row['status'] == 'success'), 'Probe raw changed')
        else: require(not row['files'] and row.get('error'), 'Failed probe accepted raw/no error')
        rows.append(row)
    return rows


def probe(root, directory):
    for endpoint, query in QUERIES.items():
        command = [sys.executable, '-m', 'scripts.review_strategy_batch30', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        with (directory / f'probe-{endpoint}.stdout').open('x') as out, (directory / f'probe-{endpoint}.stderr').open('x') as err:
            try: subprocess.run(command, stdout=out, stderr=err, timeout=45, check=True)
            except subprocess.TimeoutExpired:
                if not (directory / f'probe-{endpoint}.json').exists():
                    save(directory / f'probe-{endpoint}.json', {'endpoint': endpoint, 'query': query, 'status': 'timeout', 'files': [], 'wire': [],
                        'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'error': 'Parent deadline45s; partial unaccepted wire files may remain'})
    save(directory / 'probe-results.json', {'results': validate_probes(directory), 'published': False})
    return archive(root, 'Batch30 historical minute/turnover supplementation attempts archived without publication')


def offline(root, directory):
    before = implementation(); binding(root, directory)
    require((directory / 'minute-tail-coverage.json').is_file(), 'Minute tail coverage missing')
    def denied(*a, **kw): raise AssertionError('Batch30 offline attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied): result = compute(directory); diagnostic = diagnostics(directory)
    require(before == implementation() and diagnostic == read(directory / 'diagnostics.json'), 'Offline code/diagnostics differ')
    save(directory / 'component-offline.json', result)
    require(file_sha(directory / 'component-offline.json') == file_sha(directory / 'component-research.json'), 'Offline bytes differ')
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'sha256': file_sha(directory / 'component-offline.json'), 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch30 forbidden-network MA/KAMA components match byte-for-byte')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch30.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch30_research.py', 'tests/unit/test_batch29_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_signal_slots.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_probes(directory); before = implementation(); rows = []
    require(all((directory / n).is_file() for n in CORE), 'Core evidence missing')
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Frozen code exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': rows}


def finish(root, directory):
    checked = read(directory / 'checked-state.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 for r in checked['commands']) and
        CORE <= checked['evidence_sha256'].keys(), 'Checked code/commands/core differ')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Checked evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    off = read(directory / 'offline-verification.json')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and
        off['socket_network_disabled'] is True and off['sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json'), 'Offline evidence changed')
    binding(root, directory); require(read(directory / 'probe-results.json') == {'results': validate_probes(directory), 'published': False}, 'Probe summary differs')
    require(compute(directory) == read(directory / 'component-research.json') and diagnostics(directory) == read(directory / 'diagnostics.json'), 'Recomputed component differs')
    protection = protect(root, directory); progress = archive(root, 'Batch30 three complete reviews/long MA/completed10m KAMA/offline accepted; original trading gaps retained')
    save(Path('docs/handoff/2026-10-06-batch30-verification.json'), {'status': 'ok', 'reviews': read(directory / 'source-reviews/review.json'),
        'progress': progress, 'snapshot': SNAPSHOT, 'checks': checked, 'offline': off, 'protection': protection,
        'probes': read(directory / 'probe-results.json'), 'not_a_backtest': True, 'original_strategy_complete': False})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'prepare', 'study', 'coverage', 'probe', 'worker', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch30/20261006-kama-adhesion-pullback')
    parser.add_argument('--endpoint', choices=list(QUERIES)); args = parser.parse_args()
    result = worker(args.root, Path(args.directory), args.endpoint) if args.action == 'worker' else globals()[args.action](args.root, Path(args.directory))
    print(json.dumps(result, ensure_ascii=False, default=str))
