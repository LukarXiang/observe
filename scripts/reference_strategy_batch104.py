"""Frozen raw-price/volume formula references for short80/limit78, without a stock pool."""
import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from scripts import research_strategy_batch104 as study
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-raw-formula-reference-v3')
PARENT = Path('docs/handoff/2026-10-11-batch103-final-review.json')
OLD = Path('data/staging/strategies-batch97/20261011-components-v1')
OLD_PREPARED_SHA = '5c789f0997afc0d091e62c0af25442d09b363c9a831e22cd6c076a91bd557b9e'
ORIGINAL_PRICE_SHA = '38449170568a9ecbf9858fd03364f497f5600dbde9aedbad9f094fa374073bb2'
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
OUTPUTS = ('diagnostics.json', 'short-MA-reference.parquet', 'relative-position-reference.parquet', 'summary.json')


def nested():
    outer = next(n for n in study.tree(0).body if isinstance(n, ast.FunctionDef) and n.name == 'stock_filter1')
    nodes = [n for n in outer.body if isinstance(n, ast.FunctionDef)]
    require(len(nodes) == 4 and all(study.safe_definition(n) for n in nodes), 'Nested definition contract differs')
    ns = {}; exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-short80-nested-MA>', 'exec'), ns)
    return ns


def independent_means(values, lengths):
    """Pandas mean skips NaN; preserve that behavior explicitly in the independent formula."""
    def mean(array):
        valid = [float(v) for v in array if not pd.isna(v)]
        return math.fsum(valid)/len(valid) if valid else math.nan
    return [mean(values[-n:]) for n in lengths]+[mean(values[-n-1:-1]) for n in lengths]


def short_conditions(close, volume):
    p5, p10, p20, p30, q5, q10, q20, q30 = close
    v5, v10, w5, w10 = volume
    return [q5 > q10 and p5 < p10, p20 > q20 and p30 > q30, w5 > w10 and v5*1.2 > v10]


def compute(prices):
    require({'date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'is_trading'} <= set(prices), 'Raw operand fields missing')
    require(not prices.duplicated(['date', 'instrument']).any(), 'Duplicate raw operand')
    original = nested(); ns = study.literal(1); holder = {}; short_rows = []; relative_rows = []; profiles = {}
    def quote(*a, **kwargs):
        frame = holder['frame']
        if kwargs['fields'] == ['close']: return frame[['close']].tail(1)
        require(kwargs['fields'] == ['high', 'low'], 'Unexpected original price request')
        return frame[['high', 'low']].tail(kwargs['count'])
    ns['get_price'] = quote
    for symbol, raw in prices.groupby('instrument', sort=True):
        raw = raw.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d')
        close = raw.close.to_numpy(); volume = raw.volume.to_numpy()
        numeric = raw[['close', 'volume', 'high', 'low']].to_numpy()
        missing = raw[['close', 'high', 'low']].isna().any(axis=1)
        require(dates.is_monotonic_increasing and not np.isinf(numeric).any() and
            np.isfinite(volume).all() and (volume >= 0).all() and
            not (missing & raw.is_trading).any() and raw.loc[missing,'volume'].eq(0).all(),
            'Invalid native numeric operand; missing prices only on zero-volume nontrading rows')
        for end in range(30, len(raw)):
            window = raw.iloc[end-30:end+1]
            p5, p10, q5, q10 = original['caculate_close_ma_5_10'](window.close)
            p20, p30, q20, q30 = original['caculate_close_ma_20_30'](window.close)
            values = [p5, p10, p20, p30, q5, q10, q20, q30]
            amounts = list(original['caculate_volume_ma'](window.volume))
            direct = independent_means(close[end-30:end+1], (5, 10, 20, 30))
            direct_volume = independent_means(volume[end-30:end+1], (5, 10))
            original_conditions = short_conditions(values, amounts)
            direct_conditions = short_conditions(direct, direct_volume)
            short_rows.append({'instrument': symbol, 'window_first': dates.iloc[end-30], 'window_last': dates.iloc[end],
                'close_means': values, 'volume_means': amounts, 'independent_close_means': direct,
                'independent_volume_means': direct_volume, 'original_conditions': original_conditions,
                'independent_conditions': direct_conditions, 'condition_disagreements': sum(a != b for a,b in zip(original_conditions, direct_conditions, strict=True)),
                'original_all_conditions': all(original_conditions), 'independent_all_conditions': all(direct_conditions),
                'nontrading_rows': int((~window.is_trading).sum()), 'platform_adjustment_or_pause_semantics_proven': False})
        for length, function in ((15, 'get_relative_position_stocks'), (30, 'get_relative_position_stocks2')):
            for end in range(length-1, len(raw)):
                window = raw.iloc[end-length+1:end+1]; holder['frame'] = window
                selected = ns[function](dates.iloc[end], length, [symbol])
                highs = [float(v) for v in window.high if not pd.isna(v)]
                lows = [float(v) for v in window.low if not pd.isna(v)]
                maximum = max(highs) if highs else math.nan; minimum = min(lows) if lows else math.nan
                denominator = maximum-minimum; last = float(window.close.iloc[-1])
                ratio = (last-minimum)/denominator if denominator else math.nan
                upper = .3 if length == 15 else .5
                accepted = 0. <= ratio <= upper
                relative_rows.append({'instrument': symbol, 'length': length, 'window_first': dates.iloc[end-length+1],
                    'window_last': dates.iloc[end], 'maximum_high': maximum, 'minimum_low': minimum, 'close': last,
                    'independent_ratio': ratio, 'original_upper_bound': upper,
                    'original_selected': bool(selected), 'independent_selected': accepted,
                    'disagreement': bool(selected) != accepted, 'zero_range': denominator == 0,
                    'nontrading_rows': int((~window.is_trading).sum()), 'platform_adjustment_or_pause_semantics_proven': False})
        profiles[symbol] = {'rows': len(raw), 'first': dates.iloc[0], 'last': dates.iloc[-1],
            'short31_windows': max(0, len(raw)-30), 'relative15_windows': max(0, len(raw)-14),
            'relative30_windows': max(0, len(raw)-29), 'nontrading_rows': int((~raw.is_trading).sum()),
            'native_missing_price_rows': int(missing.sum())}
    ma = pd.DataFrame(short_rows); relative = pd.DataFrame(relative_rows)
    summary = {'profiles': profiles, 'short31_windows': len(ma), 'relative_position_windows': len(relative),
        'short_condition_disagreements': int(ma.condition_disagreements.sum()),
        'short_all_condition_disagreements': int(ma.original_all_conditions.ne(ma.independent_all_conditions).sum()),
        'relative_position_disagreements': int(relative.disagreement.sum()), 'zero_range_windows': int(relative.zero_range.sum()),
        'source_close_volume_inputs_are_native_raw_not_dynamic_adjusted': True,
        'native_missing_prices_preserved': True,
        'historical_original_pool_proven': False, 'complete_original_pool_signals': 0, 'new_trading_backtests': 0,
        'original_strategy_complete': False, 'network_blocked': True,
        'short_year_windows': ma.window_last.str[:4].value_counts().sort_index().to_dict()}
    # Keep JSON keys simple and preserve the two lookback profiles separately.
    summary['relative_year_windows'] = {str(n): group.window_last.str[:4].value_counts().sort_index().to_dict()
        for n, group in relative.groupby('length')}
    return ma, relative, summary


def prepare():
    require(not DIRECTORY.exists(), 'Immutable reference directory exists')
    parent = read(PARENT); require(parent['status'] == 'ok' and parent['progress']['manually_reviewed'] == 292, 'Formal parent differs')
    for name, sha in parent['evidence_sha256'].items(): require(file_sha(name) == sha, 'Formal parent evidence changed')
    store = Store('data'); state = store.state(SNAPSHOT)
    require(state['tables'] == store.published()['tables'] and file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA,
            'Actual snapshot/publication changed')
    require(file_sha(OLD/'prepared.json') == OLD_PREPARED_SHA, 'Archived operand manifest byte SHA differs')
    previous = read(OLD/'prepared.json'); require(previous['snapshot_sha256'] == SNAPSHOT_SHA and
        previous['tables']['bars_1d'] == state['tables']['bars_1d'], 'Historical raw operand graph differs')
    original_price = OLD/'prices.parquet'
    require(file_sha(original_price) == previous['operands_sha256']['prices.parquet'] == ORIGINAL_PRICE_SHA, 'Archived raw operand SHA differs')
    require(len(pd.read_parquet(original_price)) == 46781, 'Original projection row count differs')
    DIRECTORY.mkdir(parents=True); shutil.copyfile(original_price, DIRECTORY/'prices.parquet')
    files = sorted({p.as_posix() for folder in ('src/observe',) for p in Path(folder).rglob('*.py')} |
        {'scripts/reference_strategy_batch104.py', 'scripts/research_strategy_batch104.py',
         'scripts/research_strategy_batch101.py', 'scripts/research_strategy_batch102.py',
         'scripts/recover_strategy_batch94_reference.py', 'scripts/review_strategy_batch18.py',
         'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py',
         'scripts/archive_strategy_progress.py', 'scripts/verify_strategy_batch104_stage.py',
         'scripts/probe_strategy_batch104.py', 'scripts/check_strategy_batch104_expressions.py',
         'tests/unit/test_batch104_research.py',
         'tests/unit/test_batch104_reference.py', 'pyproject.toml', 'uv.lock'})
    project = Path.cwd().resolve()
    imported = {Path(module.__file__).resolve().relative_to(project).as_posix()
        for module in tuple(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).resolve().is_relative_to(project)
        and Path(module.__file__).resolve().relative_to(project).as_posix().startswith(('scripts/', 'src/observe/'))}
    files = sorted(set(files)|imported)
    for name in files:
        destination = DIRECTORY/'implementation'/name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, destination)
    paths = [str(PARENT), str(OLD/'prepared.json'), str(original_price), str(DIRECTORY/'prices.parquet'),
             'data/staging/strategies-batch104/20261011-raw-reference-failure/failure.json',
             'data/staging/strategies-batch104/20261011-raw-reference-failure/red-tests.xml',
             'data/staging/strategies-batch104/20261011-raw-reference-failure/red-tests.log',
             'data/staging/strategies-batch104/20261011-relative-threshold-failure/failure.json',
             'data/staging/strategies-batch104/20261011-relative-threshold-failure/red-regression.json',
             str(study.PREREAD/'preread.json'), str(study.PREREAD/'selection.json'),
             *(str(study.PREREAD/f'{n}.source') for n in (0,1))]
    save(DIRECTORY/'prepared.json', {'status': 'component_only', 'parent': str(PARENT), 'snapshot': SNAPSHOT,
        'snapshot_sha256': SNAPSHOT_SHA, 'evidence_sha256': {name: file_sha(name) for name in paths},
        'implementation_sha256': {name: file_sha(name) for name in files},
        'origin': 'Archived Batch97 snapshot-bound ten-stock raw projection;explicit operands,not original full stock pool;v3 preserves native NaN/skipna and correct second relative upper0.5;v1/v2 failures retained',
        'new_trading_backtests': 0, 'formal_source_research_acceptance': False})
    return {'status': 'prepared', 'frozen_files': len(files), 'price_rows': 46781, 'new_trading_backtests': 0}


def validate():
    doc = read(DIRECTORY/'prepared.json')
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Input changed: '+name)
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == sha == file_sha(DIRECTORY/'implementation'/name), 'Implementation changed: '+name)
    return doc


def worker(mode):
    validate(); require(mode in ('compute', 'replay'), 'Unknown worker mode')
    dest = DIRECTORY/mode; dest.mkdir(exist_ok=False)
    with network_blocked():
        diagnostics = study.diagnostics(); ma, relative, summary = compute(pd.read_parquet(DIRECTORY/'prices.parquet'))
        ma.to_parquet(dest/OUTPUTS[1], index=False); relative.to_parquet(dest/OUTPUTS[2], index=False)
        save(dest/OUTPUTS[0], diagnostics); save(dest/OUTPUTS[3], summary)
    return {'status': 'ok_component_only', **summary}


def run():
    validate(); require(not (DIRECTORY/'offline-verification.json').exists(), 'Immutable replay evidence exists')
    commands = []
    for mode in ('compute', 'replay'):
        command = [sys.executable, '-m', 'scripts.reference_strategy_batch104', mode]
        with (DIRECTORY/(mode+'.log')).open('xb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        commands.append({'command': command, 'returncode': result.returncode,
            'log': str(DIRECTORY/(mode+'.log')), 'log_sha256': file_sha(DIRECTORY/(mode+'.log'))})
        save(DIRECTORY/(mode+'-command.json'), commands[-1]); require(result.returncode == 0, 'Original reference worker failed')
    differences = [name for name in OUTPUTS if file_sha(DIRECTORY/'compute'/name) != file_sha(DIRECTORY/'replay'/name)]
    require(not differences, 'Actual replay bytes differ')
    save(DIRECTORY/'offline-verification.json', {'status': 'ok', 'differences': differences, 'commands': commands,
        'fresh_network_blocked_processes': 2, 'outputs_sha256': {str(DIRECTORY/'compute'/name): file_sha(DIRECTORY/'compute'/name) for name in OUTPUTS},
        'formal_trading_acceptance': False, 'new_trading_backtests': 0})
    progress = archive('data', 'Batch104 raw MA/relative-position references and46 diagnostics offline;formal292/trading12/parents87 unchanged')
    save(DIRECTORY/'progress.json', progress)
    return {'status': 'ok_stage_only', 'outputs': len(OUTPUTS), 'checkpoint': progress['checkpoint'], 'differences': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'run', 'compute', 'replay'))
    action = parser.parse_args().action
    result = globals()[action]() if action in ('prepare', 'run') else worker(action)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
