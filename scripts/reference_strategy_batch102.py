"""Freeze macro48/ETF58 operands and longest numerical references, never trades."""
import argparse
from decimal import Decimal
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from scripts import research_strategy_batch102 as original
from scripts.prepare_strategy_batch101 import normalize
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.research_strategy_batch100 import SNAPSHOT, SNAPSHOT_SHA
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch102/20261011-macro-ETF-reference')
PARENT = Path('docs/handoff/2026-10-11-batch101-final-review.json')
PMI = 'data/raw/dependency_probe_batch12/macro_pmi/20261005-daily-candidates.parquet'
INDEX = 'data/staging/strategies-batch47/20261007-candidate-momentum/index-input.parquet'
OPERANDS = {PMI: '6121f0fb8f1dba5672b75c24d5d56346a401e0a1b330d9935a5d788430b7d895',
    INDEX: 'b37fb1497f5cc21b4c11a3cb7d0a823b91661a63ab3e94579c0d639105372928'}
REGISTRIES = (
    ('159949.XSHE', 'data/staging/strategies-batch76/20261009-dynamic-emotion-two-eight/probe-results.json'),
    ('511010.XSHG', 'data/staging/strategies-batch77/20261009-DCA-index-valuation/probe-results.json'),
    ('513100.XSHG', 'data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probe-results.json'),
    ('512100.XSHG', 'data/staging/strategies-batch81/20261009-stock-bond-drawdown/probe-results.json'),
    ('513500.XSHG', 'data/staging/strategies-batch81/20261009-stock-bond-drawdown/probe-results.json'))
FILES = ('scripts/research_strategy_batch102.py', 'scripts/reference_strategy_batch102.py',
    'tests/unit/test_batch102_research.py', 'tests/unit/test_batch102_reference.py')
OUTPUTS = ('diagnostics.json', 'PMI-reference.parquet', 'index-reference.parquet',
    'annual-reference.parquet', 'ETF-reference.parquet', 'summary.json')


def inventory():
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA and
        state['tables'] == store.published()['tables'], 'Actual publication/snapshot changed')
    symbols = sorted({s.replace('.XSHG', '.SH').replace('.XSHE', '.SZ') for s in
        (*original.POOL, '510300.XSHG', '511010.XSHG', '000300.XSHG')})
    counts = {}
    with duckdb.connect() as con:
        for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
            paths = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
            column = 'index' if table == 'index_1d' else 'instrument'; rows = dict.fromkeys(symbols, 0)
            if paths:
                marks = ','.join('?' for _ in symbols)
                rows.update(dict(con.execute(f'SELECT "{column}", count(*) FROM read_parquet(?, union_by_name=true) '
                    f'WHERE "{column}" IN ({marks}) GROUP BY "{column}"', [paths, *symbols]).fetchall()))
            counts[table] = rows
    calendar_bindings = {}
    for entry in state['tables']['calendar'].values():
        name = str(Path('data')/entry['file']); sha = file_sha(name)
        require(fingerprint(pd.read_parquet(name)) == entry['sha'], 'Calendar Store content changed')
        if entry.get('file_sha256'): require(sha == entry['file_sha256'], 'Calendar partition bytes changed')
        calendar_bindings[name] = sha
    days = store.load_state(state, 'calendar')
    calendar = pd.to_datetime(days.loc[days.is_open, 'date']).dt.strftime('%Y-%m-%d').tolist()
    require(calendar == sorted(set(calendar)), 'Calendar order differs')
    return {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
        'published_batch': store.published()['batch_id'], 'active_table_counts': counts,
        'calendar': [d for d in calendar if d <= '2026-09-29'],
        'fresh_calendar_byte_sha256_with_legacy_logical_SHA_verified': calendar_bindings, 'strict_usable': False}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable Batch102 output exists')
    parent = read(PARENT); policy = read('configs/research_metadata.json')
    require(parent['status'] == 'ok' and parent['formal_source_research_acceptance'] and
        parent['progress']['manually_reviewed'] == 288 and policy['accepted_receipt'] == str(PARENT) and
        policy['accepted_checkpoint'] == parent['progress']['checkpoint'], 'Formal Batch101 parent differs')
    for name, sha in parent['evidence_sha256'].items(): require(file_sha(name) == sha, 'Parent changed: '+name)
    for name, sha in OPERANDS.items(): require(file_sha(name) == sha, 'Existing operand changed')
    duplicate = Path('repo/量化策略源代码/2024年度精选策略1/21.一种宏观数据的中长线策略，年化15%，最大回撤9%.txt')
    require(file_sha(duplicate) == original.SOURCE_SHA[0], 'Exact macro duplicate differs')
    catalog = Path('data/catalog/strategies')/parent['progress']['catalog_id']/'catalog.json'
    records = {r['path']: r for r in read(catalog)}
    require(records[original.SOURCES[0]]['duplicate_of'] == records[duplicate.relative_to('repo/量化策略源代码').as_posix()]['strategy_id'],
        'Catalog duplicate identity differs')
    inputs = inventory(); bindings = {**OPERANDS, str(PARENT): file_sha(PARENT), str(catalog): file_sha(catalog),
        str(duplicate): file_sha(duplicate)}; frames = {}; candidates = {}
    for symbol, registry in REGISTRIES:
        doc = read(registry); rows = [r for r in doc['results'] if r['case'] == 'fund-'+symbol[:6]]
        require(len(rows) == 1, 'Candidate registration differs'); row = rows[0]
        prefix = 'sz' if symbol.endswith('XSHE') else 'sh'
        require(row['status'] == 'success' and row['query'] == {'function': 'fund_etf_hist_sina',
            'parameters': {'symbol': prefix+symbol[:6]}} and file_sha(row['file']) == row['sha256'], 'Candidate query/SHA differs')
        bindings[registry] = file_sha(registry); bindings[row['file']] = row['sha256']
        for wire in row['wire_responses']:
            require(file_sha(wire['file']) == wire['sha256'], 'Candidate wire differs'); bindings[wire['file']] = wire['sha256']
        raw = pd.read_parquet(row['file']); frame, excluded = normalize(raw)
        require(len(raw) == row['rows'], 'Registered native row count differs'); frames[symbol] = frame
        candidates[symbol] = {'registry': registry, 'file': row['file'], 'sha256': row['sha256'], 'query': row['query'],
            'native_rows': len(raw), 'retained_rows': len(frame), 'future_excluded': excluded,
            'first': frame.date.iloc[0], 'last': frame.date.iloc[-1], 'strict_usable': False}
    # A bounded registry scan records its exact files and scope, not whole-machine absence.
    scanned = sorted(Path('data/staging').glob('**/*results.json')); found = []
    for path in scanned:
        doc = read(path)
        for row in doc.get('results', []) if isinstance(doc, dict) else []:
            if isinstance(row, dict) and row.get('status') == 'success' and row.get('case', '').startswith('fund-'):
                if row['case'][5:] in {s[:6] for s in (*original.POOL, '511010.XSHG', '510300.XSHG')}:
                    found.append({'registry': str(path), 'case': row['case'], 'file': row.get('file'), 'sha256': row.get('sha256')})
    inputs.update(candidate_bindings=candidates, ETF_pool_entries=18, ETF_unique_pool=17,
        ETF_candidates_in_selected_registries=sorted(set(frames)&set(original.POOL)),
        ETF_missing_in_selected_registries=sorted(set(original.POOL)-set(frames)),
        absence_scope='Selected SHA-checked registrations and top-level success fund-case scan; not whole-machine absence')
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch102 source/input work started after formal288 parent;12 sources/87 strategy parents unchanged')
        for n in (0, 1):
            original.tree(n)
            for path in (original.PREREAD/f'{n}.source', Path('repo/量化策略源代码')/original.SOURCES[n]):
                bindings[str(path)] = original.SOURCE_SHA[n]
            dest = DIRECTORY/'sources'/f'{n}.source'; dest.parent.mkdir(exist_ok=True); shutil.copyfile(original.PREREAD/f'{n}.source', dest)
        for name in FILES:
            dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
        for symbol, frame in frames.items():
            dest = DIRECTORY/'inputs'/(symbol+'.parquet'); dest.parent.mkdir(exist_ok=True); frame.to_parquet(dest, index=False)
            bindings[str(dest)] = file_sha(dest)
        for name in OPERANDS:
            dest = DIRECTORY/'inputs'/('PMI.parquet' if name == PMI else 'index.parquet'); shutil.copyfile(name, dest); bindings[str(dest)] = file_sha(dest)
        save(DIRECTORY/'dependencies.json', inputs); bindings[str(DIRECTORY/'dependencies.json')] = file_sha(DIRECTORY/'dependencies.json')
        save(DIRECTORY/'registry-scan.json', {'files': {str(p): file_sha(p) for p in scanned}, 'matches': found,
            'scope': inputs['absence_scope'], 'new_requests': 0})
        save(DIRECTORY/'prepared.json', {'parent': str(PARENT), 'parent_sha256': file_sha(PARENT),
            'evidence_sha256': bindings, 'implementation_sha256': {n: file_sha(n) for n in FILES},
            'source_sha256': list(original.SOURCE_SHA), 'duplicate': {'source': original.SOURCES[0],
            'duplicate_of': records[original.SOURCES[0]]['duplicate_of'], 'same_bytes_as': str(duplicate), 'new_independent_strategy': False},
            'snapshot': SNAPSHOT, 'new_trading_backtests': 0, 'strict_usable': False})
    return {'status': 'prepared', 'candidate_funds': len(frames), 'not_a_backtest': True}


def probe():
    """At most two actual NBS calls, with timeout, no proxy/retry/redirect or invented cid."""
    import akshare as ak
    from curl_cffi import requests
    path = DIRECTORY/'NBS-probe'; require(not path.exists(), 'Immutable NBS probe exists'); path.mkdir()
    function = ak.macro_china_nbs_nation; installed = Path(inspect.getfile(function))
    ref = Path('repo/akshare/akshare/economic/macro_china_nbs.py')
    shutil.copyfile(installed, path/'installed-api.py'); shutil.copyfile(ref, path/'repo-api.py')
    calls = []; session = requests.Session(impersonate='chrome', trust_env=False)
    urls = [('https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData', None),
        ('https://data.stats.gov.cn/dg/website/publicrelease/web/external/new/queryIndexTreeAsync', {'pid': '', 'code': 1})]
    for i, (url, params) in enumerate(urls):
        row = {'method': 'GET', 'url': url, 'params': params, 'timeout_seconds': 10, 'redirects': False, 'retries': 0}
        try:
            response = session.get(url, params=params, timeout=10, allow_redirects=False)
            wire = path/f'{i}.response.bin'; wire.write_bytes(response.content)
            row.update(status='response', status_code=response.status_code, file=str(wire), sha256=file_sha(wire))
            if response.status_code != 200: calls.append(row); break
        except Exception as exc:
            row.update(status='failed', error=type(exc).__name__, message=str(exc)); calls.append(row); break
        calls.append(row)
    session.close()
    doc = {'status': 'blocked' if any(r['status'] == 'failed' or r.get('status_code') != 200 for r in calls) else 'directory_only',
        'akshare': ak.__version__, 'installed_api_sha256': file_sha(path/'installed-api.py'),
        'repo_api_sha256': file_sha(path/'repo-api.py'), 'repo_commit': subprocess.run(['git', '-C', 'repo/akshare', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip(), 'attempts': calls, 'new_data_samples': 0,
        'published': False, 'history_release_versions_proven': False,
        'next': 'Resolve genuine returned catalog ids before requesting indicators; no guessed cid and no further same-domain retries'}
    save(path/'results.json', doc); return doc


def validate():
    doc = read(DIRECTORY/'prepared.json')
    require(doc['source_sha256'] == list(original.SOURCE_SHA) and doc['new_trading_backtests'] == 0 and not doc['strict_usable'], 'Scope changed')
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Frozen input changed: '+name)
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == sha, 'Frozen implementation changed: '+name)
    return doc


def pmi_reference(raw):
    months = raw['月份'].str.replace('年', '-', regex=False).str.replace('月份', '', regex=False)
    frame = pd.DataFrame({'month': months, 'pmi': pd.to_numeric(raw['制造业-指数'])}).sort_values('month').reset_index(drop=True)
    require(frame.month.tolist() == pd.period_range(frame.month.iloc[0], frame.month.iloc[-1], freq='M').astype(str).tolist(), 'PMI gap/duplicate')
    fn = original.literal(0)['func']; rows = []; t = [float(x-6) for x in range(13)]
    denominator = math.fsum((x*x-14.)**2 for x in t)
    for end in range(12, len(frame)):
        y = frame.pmi.iloc[end-12:end+1].to_numpy(float)
        params, _ = original.op.curve_fit(fn, np.arange(13), y)
        slope = float(24*params[0]+params[1])
        a = math.fsum((x*x-14.)*v for x, v in zip(t, y, strict=True))/denominator
        b = math.fsum(x*v for x, v in zip(t, y, strict=True))/182.
        independent = 12*a+b
        require(math.isclose(slope, independent, abs_tol=2e-6, rel_tol=2e-6), 'Independent quadratic slope differs')
        rows.append({'window_first': frame.month.iloc[end-12], 'last_stat_month': frame.month.iloc[end],
            'slope': slope, 'independent_slope': independent, 'pmi_first': float(y[0]), 'pmi_last': float(y[-1]),
            'aggregate_condition': bool(slope > 0 and y[-1] >= 50 and y[0] <= y[-1]),
            'independent_condition': bool(independent > 0 and y[-1] >= 50 and y[0] <= y[-1]),
            'same_history_as_batch50': True, 'original_point_in_time_signal_proven': False})
    return pd.DataFrame(rows)


def index_reference(frame):
    ns = original.literal(0); prices = frame.close.to_numpy(float); rows = []; monthly = []; annual = []
    for end in range(len(frame)):
        if end < 119: continue
        current = {}; manual = {}
        for n in (5, 10, 20, 30, 120):
            ns['get_bars'] = lambda *a, count=n, stop=end, **k: pd.DataFrame({'close': prices[stop-count+1:stop+1]})
            current[n] = float(ns['day_move_average'](n, True)); manual[n] = math.fsum(prices[end-n+1:end+1])/n
            require(math.isclose(current[n], manual[n], abs_tol=1e-9, rel_tol=1e-12), 'Independent MA differs')
        previous30 = float(np.mean(prices[end-30:end])); independent30 = math.fsum(prices[end-30:end])/30
        rows.append({'asof_EOD': str(frame.date.iloc[end]), **{f'MA{n}': v for n, v in current.items()},
            **{f'independent_MA{n}': v for n, v in manual.items()}, 'previous_MA30': previous30,
            'MA_chain': bool(current[5] > current[10] > current[20] > current[30] > previous30),
            'independent_MA_chain': bool(manual[5] > manual[10] > manual[20] > manual[30] > independent30),
            'original1455_include_now_proven': False})
    dated = frame[['date', 'close']].copy(); dated['date'] = pd.to_datetime(dated.date)
    for _, group in dated.groupby(dated.date.dt.to_period('M'), sort=True): monthly.append(group.iloc[-1])
    monthly = pd.DataFrame(monthly).reset_index(drop=True)
    for end in range(108, len(monthly)):
        window = monthly.iloc[end-108:end+1].copy(); ns['get_bars'] = lambda *a, w=window, **k: w.copy(deep=True)
        result = ns['change_to_yeak_k'](); value = ns['year_move_average'](result)
        independent = {}
        for row in window.itertuples(index=False): independent[str(row.date.year)] = float(row.close)
        require(result == independent, 'Independent annual dictionary differs')
        total = sum((Decimal(str(v)) for v in independent.values()), Decimal('0'))/Decimal('10')
        require(value == float(total), 'Independent fixed10 denominator differs')
        annual.append({'last_observed_month_close': window.date.iloc[-1].strftime('%Y-%m-%d'),
            'annual_keys': len(result), 'year_points': json.dumps(result, sort_keys=True), 'fixed10_average': value,
            'input_month_rows': 109, 'original1455_include_now_proven': False})
    return pd.DataFrame(rows), pd.DataFrame(annual)


def etf_reference(frames, calendar):
    rows = []; profiles = {}; days = set(calendar)
    for symbol, frame in sorted(frames.items()):
        observed = set(frame.date)
        missing = sorted(d for d in days if frame.date.iloc[0] <= d <= frame.date.iloc[-1] and d not in observed)
        prices = frame.close.to_numpy(float)
        for end in range(19, len(frame)):
            values = prices[end-19:end+1]; result = original.reference_metrics({symbol: values}).iloc[0]
            long = math.fsum(values)/20; short = math.fsum(values[-5:])/5; inc = values[-1]/values[0]-1
            require(result['inc'] == inc and math.isclose(result.malong, long, abs_tol=1e-12) and
                math.isclose(result.mashort, short, abs_tol=1e-12), 'Independent ETF operands differ')
            crossed = [d for d in missing if frame.date.iloc[end-19] <= d <= frame.date.iloc[end]]
            rows.append({'instrument': symbol, 'window_first': frame.date.iloc[end-19], 'asof_native_close': frame.date.iloc[end],
                'inc19_intervals': float(result['inc']), 'MA20': float(result.malong), 'MA5': float(result.mashort),
                'independent_MA20': long, 'independent_MA5': short,
                'eligible_metric_reference': bool(result.mashort > result.malong and .05 < inc < .2),
                'independent_eligible': bool(short > long and .05 < inc < .2), 'missing_open_count': len(crossed),
                'original_attribute_history_proven': False})
        profiles[symbol] = {'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
            'windows': len(frame)-19, 'missing_open_dates': missing,
            'year_windows': frame.date.iloc[19:].str[:4].value_counts().sort_index().to_dict()}
    return pd.DataFrame(rows), profiles


def worker(mode):
    require(mode in ('compute', 'replay'), 'Unknown worker mode'); validate()
    dest = DIRECTORY/mode; require(not dest.exists(), 'Immutable reference worker output exists'); dest.mkdir()
    with network_blocked():
        pmi = pmi_reference(pd.read_parquet(DIRECTORY/'inputs/PMI.parquet'))
        index, annual = index_reference(pd.read_parquet(DIRECTORY/'inputs/index.parquet'))
        inputs = read(DIRECTORY/'dependencies.json')
        frames = {s: pd.read_parquet(DIRECTORY/'inputs'/(s+'.parquet')) for s in inputs['ETF_candidates_in_selected_registries']}
        etf, profiles = etf_reference(frames, inputs['calendar'])
        for name, frame in (('PMI', pmi), ('index', index), ('annual', annual), ('ETF', etf)): frame.to_parquet(dest/(name+'-reference.parquet'), index=False)
        doc = original.diagnostics(); save(dest/'diagnostics.json', doc)
        summary = {'PMI_windows': len(pmi), 'PMI_reused_history_not_new_samples': True,
            'PMI_condition_disagreements': int(pmi.aggregate_condition.ne(pmi.independent_condition).sum()),
            'PMI_max_slope_error': float((pmi.slope-pmi.independent_slope).abs().max()),
            'index_complete120_EOD_rows': len(index), 'index_strict_chain_disagreements': int(index.MA_chain.ne(index.independent_MA_chain).sum()),
            'annual109_month_windows': len(annual), 'ETF20_row_windows': len(etf), 'ETF_profiles': profiles,
            'ETF_gap_crossing_windows': int(etf.missing_open_count.gt(0).sum()),
            'ETF_eligibility_disagreements': int(etf.eligible_metric_reference.ne(etf.independent_eligible).sum()),
            'diagnostics': doc['case_count'], 'errors': {r['case']: r['error'] for r in doc['cases'] if r['error']},
            'network_blocked': True, 'full_original_pool_signals': 0, 'new_trading_backtests': 0,
            'original_strategy_complete': False, 'strict_usable': False}
        save(dest/'summary.json', summary)
    return summary


def run():
    require(not (DIRECTORY/'offline-verification.json').exists(), 'Immutable verification exists')
    commands = []
    for mode in ('compute', 'replay'):
        result = subprocess.run([sys.executable, '-m', 'scripts.reference_strategy_batch102', 'worker', '--mode', mode], capture_output=True, text=True)
        log = DIRECTORY/(mode+'.log'); log.write_text(result.stdout+result.stderr)
        commands.append({'mode': mode, 'returncode': result.returncode, 'log': str(log), 'log_sha256': file_sha(log)})
        save(DIRECTORY/(mode+'-command.json'), commands[-1]); require(result.returncode == 0, 'Reference worker failed; evidence preserved')
    differences = [name for name in OUTPUTS if file_sha(DIRECTORY/'compute'/name) != file_sha(DIRECTORY/'replay'/name)]
    doc = {'status': 'ok' if not differences else 'error', 'differences': differences, 'commands': commands,
        'outputs_sha256': {str(DIRECTORY/'compute'/n): file_sha(DIRECTORY/'compute'/n) for n in OUTPUTS},
        'fresh_network_blocked_processes': 2, 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', doc); require(not differences, 'Reference byte replay differs'); return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'worker', 'run', 'protect'))
    parser.add_argument('--mode', choices=('compute', 'replay')); args = parser.parse_args()
    result = worker(args.mode) if args.action == 'worker' else protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
