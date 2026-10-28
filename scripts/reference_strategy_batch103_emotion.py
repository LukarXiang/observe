"""Original ETF6 heat callback on native volume and declared float operands; never trades."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import talib

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from scripts import reference_strategy_batch103 as price
from scripts import research_strategy_batch103 as source
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch103/20261011-native-emotion-reference')
PHASE = Path('docs/handoff/2026-10-11-batch103-price-stage.json')
REGISTRIES = (price.REGISTRIES[0], price.REGISTRIES[2])
INDEX_CODES = ('000016.XSHG', '399006.XSHE', '399001.XSHE', '000010.XSHG',
               '000015.XSHG', '399324.XSHE', '000905.XSHG', '399673.XSHE')
TEST = 'tests/unit/test_batch103_emotion.py'
OUTPUTS = ('emotion-reference.parquet', 'dtype-diagnostics.json', 'summary.json')


def normalize_volume(raw):
    require({'date', 'volume'} <= set(raw.columns), 'Volume fields missing')
    dates = pd.to_datetime(raw.date, errors='raise').dt.strftime('%Y-%m-%d')
    values = raw.volume.to_numpy()
    require(dates.notna().all() and dates.is_monotonic_increasing and not dates.duplicated().any() and
        np.issubdtype(values.dtype, np.number) and np.isfinite(values).all() and (values >= 0).all(),
        'Native volume invalid')
    frame = pd.DataFrame({'date': dates, 'volume': raw.volume})
    retained = frame.loc[frame.date.le('2026-09-29')].reset_index(drop=True)
    require(len(retained) >= 100 and retained.volume.dtype == raw.volume.dtype, 'Native type/history changed')
    return retained, len(frame)-len(retained)


def operands():
    frames = {}; bindings = {}; evidence = {}
    for registry in REGISTRIES:
        doc = read(registry); evidence[registry] = file_sha(registry)
        for symbol in INDEX_CODES:
            prefix = 'sz' if symbol.endswith('XSHE') else 'sh'; key = prefix+symbol[:6]
            rows = [r for r in doc['results'] if r.get('case') == 'index-'+key and r.get('status') == 'success']
            if not rows: continue
            require(len(rows) == 1 and symbol not in frames, 'Duplicate volume registration')
            row = rows[0]
            require(row['query'] == {'function': 'stock_zh_index_daily', 'parameters': {'symbol': key}} and
                file_sha(row['file']) == row['sha256'], 'Volume candidate API/bytes differ')
            raw = pd.read_parquet(row['file']); require(len(raw) == row['rows'], 'Native row count differs')
            frame, excluded = normalize_volume(raw); frames[symbol] = frame; evidence[row['file']] = row['sha256']
            for wire in row['wire_responses']:
                require(wire['status_code'] == 200 and file_sha(wire['file']) == wire['sha256'], 'Volume wire differs')
                evidence[wire['file']] = wire['sha256']
            bindings[symbol] = {'registry': registry, 'query': row['query'], 'file': row['file'], 'sha256': row['sha256'],
                'native_rows': len(raw), 'retained_rows': len(frame), 'future_excluded': excluded,
                'volume_dtype': str(raw.volume.dtype), 'strict_usable': False}
    require(set(frames) == set(INDEX_CODES), 'Selected eight registrations incomplete')
    store = Store('data'); state = store.state(price.SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(price.SNAPSHOT+'.json')) == price.SNAPSHOT_SHA and
        state['tables'] == store.published()['tables'], 'Actual publication changed')
    for entry in state['tables']['index_1d'].values():
        name = str(Path('data')/entry['file']); frame = pd.read_parquet(name)
        require(fingerprint(frame) == entry['sha'], 'Published index logical SHA differs')
        if entry.get('file_sha256'): require(file_sha(name) == entry['file_sha256'], 'Published index bytes differ')
        evidence[name] = file_sha(name)
    published = store.load_state(state, 'index_1d')
    frame, excluded = normalize_volume(published.loc[published['index'].eq('000300.SH')].reset_index(drop=True))
    require(len(frame) == 5280 and str(frame.volume.dtype) == 'float64', 'CSI300 operand identity differs')
    frames['000300.XSHG'] = frame
    bindings['000300.XSHG'] = {'source_kind': 'actual_published_index_projection', 'snapshot': price.SNAPSHOT,
        'retained_rows': len(frame), 'future_excluded': excluded, 'volume_dtype': str(frame.volume.dtype),
        'platform_attribute_history_dtype_proven': False, 'strict_usable': False}
    return frames, bindings, evidence


def callback(volume):
    """Pass input dtype unchanged; the original broad exception behavior stays intact."""
    ns = source.literal(0, {'attribute_history': lambda *a, **k: pd.DataFrame({'volume': volume})})
    ns['g'].target_market = 'explicit-volume-operand'; ns['g'].lag0 = 7; ns['g'].lag = 5; ns['g'].emotion_rate = 0.
    value = ns['EmotionMonitor'](SimpleNamespace())
    return value, ns['g'].emotion_rate


def independent(volume):
    values = np.asarray(volume, dtype=float)
    ratios = []
    for end in range(70, 100):
        mean = math.fsum(values[end-6:end+1])/7
        ratios.append(values[end]/mean-1 if mean != 0 else np.nan)
    latest = ratios[-1]; positive = latest >= 0; result = None; crossing = None
    for distance, prior in enumerate(reversed(ratios)):
        if (prior < 0) if positive else (prior >= 0):
            crossing = distance
            result = (1 if distance >= 3 else 0) if positive else (-1 if distance >= 5 else 0)
            break
    rate = float(round(latest*100, 2)) if math.isfinite(latest) else None
    return result, rate, crossing, float(latest)


def result_label(value): return 'None' if value is None else str(value)


def dtype_diagnostics():
    rows = []
    for dtype in ('int64', 'float64'):
        values = np.asarray([1000]*95+[100]*5, dtype=dtype); value, rate = callback(values)
        error = None
        try: talib.MA(values, 7)
        except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)}
        rows.append({'dtype': dtype, 'result': value, 'rate': rate, 'direct_TA_Lib_error': error})
    return {'rows': rows, 'float_conversion_is_reference_only': True,
            'platform_history_dtype_proven': False, 'economic_or_compatibility_fix_approved': False}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable emotion directory exists'); price.validate()
    phase = read(PHASE)
    require(phase['status'] == 'ok_stage_only' and not phase['formal_source_research_acceptance'], 'Stage parent differs')
    for name, sha in phase['evidence_sha256'].items(): require(file_sha(name) == sha, 'Stage evidence changed')
    frames, bindings, evidence = operands()
    files = sorted(set(price.implementation_files()) | {'scripts/reference_strategy_batch103_emotion.py', TEST})
    calendar = read(price.DIRECTORY/'dependencies.json')['calendar']
    with operation_lock('data', DATA_WRITER):
        DIRECTORY.mkdir(parents=True)
        for name in ('baseline.json', 'implementation-evidence.before.json'):
            shutil.copyfile(price.DIRECTORY/name, DIRECTORY/name)
        evidence[str(PHASE)] = file_sha(PHASE); evidence[str(DIRECTORY/'baseline.json')] = file_sha(DIRECTORY/'baseline.json')
        for name in files:
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
        for symbol, frame in frames.items():
            dest = DIRECTORY/'inputs'/(symbol+'.parquet'); dest.parent.mkdir(exist_ok=True); frame.to_parquet(dest, index=False)
            evidence[str(dest)] = file_sha(dest)
        dependencies = {'bindings': bindings, 'calendar': calendar, 'calendar_coverage_start': calendar[0],
            'raw_index_candidates': 8, 'published_index_projections': 1, 'dynamic_pool_references_available': 8,
            'initial_default_reference': '399001.XSHE',
            'dynamic_references_missing_in_selected_operands': ['399632.XSHE', '162605.XSHE', '161005.XSHE'],
            'absence_scope': 'Two selected original registries and actual CSI300 snapshot projection; not whole-machine absence',
            'native_dtype_preserved': True, 'explicit_float_reference_is_not_platform_input': True,
            'new_requests': 0, 'new_responses': 0, 'published': False}
        save(DIRECTORY/'dependencies.json', dependencies); evidence[str(DIRECTORY/'dependencies.json')] = file_sha(DIRECTORY/'dependencies.json')
        save(DIRECTORY/'prepared.json', {'parent': str(PHASE), 'source_sha256': list(source.SOURCE_SHA),
            'snapshot': price.SNAPSHOT, 'snapshot_sha256': price.SNAPSHOT_SHA, 'evidence_sha256': evidence,
            'implementation_sha256': {n: file_sha(n) for n in files}, 'new_trading_backtests': 0,
            'strict_usable': False, 'baseline_kind': 'byte-identical inherited price-stage baseline; fresh protection required at acceptance'})
        save(DIRECTORY/'progress-start.json', archive('data', 'Batch103 native volume study started;source/formal290 and trading12/parents87 unchanged'))
    return {'status': 'prepared', 'volume_operands': len(frames), 'frozen_files': len(files)}


def validate():
    doc = read(DIRECTORY/'prepared.json'); require(doc['new_trading_backtests'] == 0 and not doc['strict_usable'], 'Volume scope differs')
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Volume input changed: '+name)
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == sha, 'Volume implementation changed: '+name)
    return doc


def compute(frames, calendar):
    rows = []; profiles = {}; days = set(calendar)
    for symbol, frame in sorted(frames.items()):
        raw = frame.volume.to_numpy(); converted = raw.astype('float64')
        missing = sorted(d for d in days-set(frame.date) if frame.date.iloc[0] <= d <= frame.date.iloc[-1])
        ns = source.literal(0); holder = {}
        ns['attribute_history'] = lambda *a, **k: pd.DataFrame({'volume': holder['volume']})
        ns['g'].target_market = symbol; ns['g'].lag0 = 7; ns['g'].lag = 5
        for end in range(99, len(frame)):
            values = raw[end-99:end+1]; floating = converted[end-99:end+1]
            outcomes = []
            for operand in (values, floating):
                holder['volume'] = operand; ns['g'].emotion_rate = 0.
                value = ns['EmotionMonitor'](SimpleNamespace()); rate = float(ns['g'].emotion_rate)
                outcomes.append((value, rate if math.isfinite(rate) else None))
            original, native_rate = outcomes[0]; proxy, proxy_rate = outcomes[1]
            direct, direct_rate, crossing, ratio = independent(floating)
            first = frame.date.iloc[end-99]; last = frame.date.iloc[end]
            rows.append({'instrument': symbol, 'window_first': first, 'asof_history_last': last,
                'native_volume_dtype': str(raw.dtype), 'native_result': result_label(original), 'native_rate': native_rate,
                'float_reference_result': result_label(proxy), 'float_reference_rate': proxy_rate,
                'independent_result': result_label(direct), 'independent_rate': direct_rate,
                'independent_last_ratio': ratio, 'independent_crossing_distance': crossing,
                'float_state_disagreement': proxy != direct,
                'native_vs_float_state_disagreement': original != proxy,
                'missing_open_count': sum(first <= d <= last for d in missing),
                'calendar_complete_for_window': first >= calendar[0], 'platform_history_dtype_proven': False})
        profiles[symbol] = {'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
            'windows': max(0, len(frame)-99), 'dtype': str(raw.dtype), 'missing_known_open_dates': missing,
            'year_windows': frame.date.iloc[99:].str[:4].value_counts().sort_index().to_dict()}
    return pd.DataFrame(rows), profiles


def worker(mode):
    validate(); require(mode in ('compute', 'replay'), 'Unknown volume mode')
    dest = DIRECTORY/mode; require(not dest.exists(), 'Immutable volume worker output exists'); dest.mkdir()
    with network_blocked():
        deps = read(DIRECTORY/'dependencies.json')
        frames = {s: pd.read_parquet(DIRECTORY/'inputs'/(s+'.parquet')) for s in deps['bindings']}
        frame, profiles = compute(frames, deps['calendar']); frame.to_parquet(dest/OUTPUTS[0], index=False)
        save(dest/OUTPUTS[1], dtype_diagnostics())
        summary = {'windows': len(frame), 'profiles': profiles,
            'native_results': frame.native_result.value_counts().sort_index().to_dict(),
            'float_reference_results': frame.float_reference_result.value_counts().sort_index().to_dict(),
            'float_state_disagreements': int(frame.float_state_disagreement.sum()),
            'native_vs_float_state_disagreements': int(frame.native_vs_float_state_disagreement.sum()),
            'float_rate_disagreements': int(frame.float_reference_rate.fillna(1e100).ne(frame.independent_rate.fillna(1e100)).sum()),
            'gap_crossing_windows': int(frame.missing_open_count.gt(0).sum()),
            'windows_before_complete_calendar_coverage': int((~frame.calendar_complete_for_window).sum()),
            'network_blocked': True, 'platform_history_dtype_proven': False,
            'complete_original_dynamic_heat_signals': 0, 'full_original_pool_signals': 0, 'new_trading_backtests': 0}
        save(dest/OUTPUTS[2], summary)
    return summary


def run():
    validate(); require(not (DIRECTORY/'offline-verification.json').exists(), 'Immutable volume verification exists')
    commands = []
    for mode in ('compute', 'replay'):
        env = dict(os.environ); env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONHASHSEED'] = '103'
        result = subprocess.run([sys.executable, '-B', '-m', 'scripts.reference_strategy_batch103_emotion', 'worker', '--mode', mode],
            capture_output=True, text=True, env=env)
        log = DIRECTORY/(mode+'.log'); log.write_text(result.stdout+result.stderr)
        entry = {'mode': mode, 'returncode': result.returncode, 'log': str(log), 'log_sha256': file_sha(log)}
        save(DIRECTORY/(mode+'-command.json'), entry); commands.append(entry)
        require(result.returncode == 0, 'Volume worker failed; original evidence retained')
    differences = [n for n in OUTPUTS if file_sha(DIRECTORY/'compute'/n) != file_sha(DIRECTORY/'replay'/n)]
    doc = {'status': 'ok' if not differences else 'error', 'differences': differences,
        'fresh_network_blocked_processes': 2, 'commands': commands,
        'outputs_sha256': {str(DIRECTORY/'compute'/n): file_sha(DIRECTORY/'compute'/n) for n in OUTPUTS},
        'float_conversion_reference_only': True, 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', doc); require(not differences, 'Volume replay differs'); return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('prepare', 'run', 'worker', 'protect'))
    parser.add_argument('--mode', choices=('compute', 'replay')); args = parser.parse_args()
    if args.action == 'worker': result = worker(args.mode)
    elif args.action == 'protect': validate(); result = protect('data', DIRECTORY)
    else: result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
