"""Bind the accepted parent and existing native candidates before Batch101 experiments."""
import argparse
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from scripts import research_strategy_batch100 as parent_study
from scripts import research_strategy_batch101 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch101/20261011-allweather-futures-source')
PARENT = Path('docs/handoff/2026-10-11-batch100-final-review.json')
REGISTRIES = (
    ('518880.XSHG', 'data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/results.json'),
    ('511010.XSHG', 'data/staging/strategies-batch77/20261009-DCA-index-valuation/probe-results.json'),
    ('513100.XSHG', 'data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probe-results.json'),
    ('513500.XSHG', 'data/staging/strategies-batch81/20261009-stock-bond-drawdown/probe-results.json'))
REUSED = Path('data/staging/strategies-batch74/20261008-FOF-two-eight/reused-ETF.json')
FILES = ('scripts/research_strategy_batch101.py', 'scripts/prepare_strategy_batch101.py',
         'tests/unit/test_batch101_research.py', 'tests/unit/test_batch101_inputs.py')


def normalize(raw):
    """Keep native close/date values; never rescale, fill gaps or replace downloads."""
    date = next((n for n in ('date', '日期') if n in raw), None)
    close = next((n for n in ('close', '收盘') if n in raw), None)
    require(date is not None and close is not None, 'Native candidate fields missing')
    frame = pd.DataFrame({'date': pd.to_datetime(raw[date]).dt.strftime('%Y-%m-%d'),
                          'close': pd.to_numeric(raw[close], errors='raise').astype(float)})
    require(frame.date.is_monotonic_increasing and not frame.date.duplicated().any() and
            frame.date.notna().all() and np.isfinite(frame.close).all() and (frame.close > 0).all(),
            'Native candidate order/price invalid')
    retained = frame.loc[frame.date.le('2026-09-29')].reset_index(drop=True)
    require(len(retained) > 0, 'No candidate before snapshot cutoff')
    return retained, len(frame)-len(retained)


def candidates():
    frames = {}; bindings = {}
    reuse = read(REUSED)
    require(file_sha(reuse['receipt']) == reuse['receipt_sha256'] and
            file_sha(reuse['payload']['path']) == reuse['payload']['sha256'], 'Legacy510310 candidate changed')
    native = pd.read_parquet(reuse['payload']['path']); frame, excluded = normalize(native)
    frames['510310.XSHG'] = frame
    bindings['510310.XSHG'] = {'registry': str(REUSED), 'registry_sha256': file_sha(REUSED),
        'file': reuse['payload']['path'], 'sha256': reuse['payload']['sha256'], 'native_rows': len(native),
        'future_excluded': excluded, 'source_kind': 'frozen_batch66_em_raw'}
    for symbol, name in REGISTRIES:
        registry = Path(name); doc = read(registry)
        rows = [r for r in doc['results'] if r['case'] == 'fund-'+symbol[:6]]
        require(len(rows) == 1, 'Legacy candidate identity differs')
        row = rows[0]
        require(row['status'] == 'success' and row['query'] == {'function': 'fund_etf_hist_sina',
            'parameters': {'symbol': 'sh'+symbol[:6]}} and file_sha(row['file']) == row['sha256'],
            'Legacy candidate byte/query differs')
        for wire in row['wire_responses']:
            require(file_sha(wire['file']) == wire['sha256'], 'Legacy candidate wire changed')
        native = pd.read_parquet(row['file']); frame, excluded = normalize(native)
        require(len(native) == row['rows'], 'Legacy native row count differs')
        frames[symbol] = frame
        bindings[symbol] = {'registry': name, 'registry_sha256': file_sha(registry),
            'file': row['file'], 'sha256': row['sha256'], 'wire_responses': row['wire_responses'],
            'query': row['query'], 'native_rows': len(native), 'future_excluded': excluded,
            'source_kind': 'frozen_sina_native_close'}
    return frames, bindings


def inventory():
    store = Store('data'); state = store.state(parent_study.SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(parent_study.SNAPSHOT+'.json')) == parent_study.SNAPSHOT_SHA and
            state['tables'] == store.published()['tables'], 'Actual snapshot/publication changed')
    symbols = tuple(s.replace('.XSHG', '.SH') for s in (*study.FUNDS, *study.INDICES)); counts = {}
    with duckdb.connect() as con:
        for table in ('instruments', 'index_1d', 'bars_1d', 'bars_5m', 'adj_factors', 'adj_coverage', 'corp_actions'):
            paths = [str(Path('data')/r['file']) for r in state['tables'].get(table, {}).values()]
            column = 'index' if table == 'index_1d' else 'instrument'; rows = dict.fromkeys(symbols, 0)
            if paths:
                marks = ','.join('?' for _ in symbols)
                rows.update({s: int(n) for s, n in con.execute(
                    f'SELECT "{column}", count(*) FROM read_parquet(?, union_by_name=true) '
                    f'WHERE "{column}" IN ({marks}) GROUP BY "{column}"', [paths, *symbols]).fetchall()})
            counts[table] = rows
    frames, bindings = candidates()
    calendar = store.load_state(state, 'calendar')
    days = set(pd.to_datetime(calendar.loc[calendar.is_open, 'date']).dt.strftime('%Y-%m-%d'))
    profiles = {}
    for symbol, frame in frames.items():
        expected = {d for d in days if frame.date.iloc[0] <= d <= frame.date.iloc[-1]}
        observed = set(frame.date)
        profiles[symbol] = {'rows': len(frame), 'first': frame.date.iloc[0], 'last': frame.date.iloc[-1],
            'calendar_missing': sorted(expected-observed), 'calendar_extra': sorted(observed-expected),
            'native120_row_reference_windows': max(0, len(frame)-119),
            'history_basis': 'candidate native observations; original history pause/adjustment semantics unproven',
            'strict_usable': False}
    return {'snapshot': parent_study.SNAPSHOT, 'snapshot_sha256': parent_study.SNAPSHOT_SHA,
        'published_batch': store.published()['batch_id'], 'active_table_counts': counts,
        'bars_1m_registered': 'bars_1m' in state['tables'],
        'futures_tables_registered': [n for n in state['tables'] if 'future' in n],
        'candidate_bindings': bindings, 'candidate_profiles': profiles,
        'funds_without_candidate_in_selected_registries': sorted(set(study.FUNDS)-set(frames)),
        'absence_scope': 'Selected SHA-checked registries, not an exhaustive whole-machine absence proof',
        'original_returns_or_fund_trades_admitted': 0, 'original_futures_trades_admitted': 0,
        'new_requests_responses_publications': 0,
        'DNS_failure_references': ['docs/tasks/2026-10-11-第九十六批原回调与输入核查-阶段记录.md',
            'docs/tasks/2026-10-10-第九十一批玉米价差与股指跨期-正式验收.md']}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable Batch101 output exists')
    parent = read(PARENT); policy = read('configs/research_metadata.json')
    require(parent['status'] == 'ok' and parent['formal_source_research_acceptance'] is True and
            parent['progress']['manually_reviewed'] == 286 and
            policy['accepted_checkpoint'] == parent['progress']['checkpoint'] and
            policy['accepted_receipt'] == str(PARENT), 'Batch100 formal parent not yet accepted')
    for name, sha in parent['evidence_sha256'].items():
        require(file_sha(name) == sha, 'Actual parent evidence changed: '+name)
    inputs = inventory(); frames, _ = candidates()
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch101 source/input study started after formal286 parent;trading12/parents87 unchanged')
        for number in (0, 1):
            study.tree(number)
            target = DIRECTORY/'sources'/f'{number}.source'; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(study.PREREAD/f'{number}.source', target)
        for name in FILES:
            target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        for symbol, frame in frames.items():
            target = DIRECTORY/'inputs'/(symbol+'.parquet'); target.parent.mkdir(exist_ok=True)
            frame.to_parquet(target, index=False)
        save(DIRECTORY/'dependencies.json', inputs)
        save(DIRECTORY/'prepared.json', {'parent': str(PARENT), 'parent_sha256': file_sha(PARENT),
            'snapshot': parent_study.SNAPSHOT, 'source_sha256': list(study.SOURCE_SHA),
            'implementation_sha256': {n: file_sha(n) for n in FILES},
            'dependencies_sha256': file_sha(DIRECTORY/'dependencies.json'),
            'input_sha256': {str(p): file_sha(p) for p in sorted((DIRECTORY/'inputs').glob('*.parquet'))},
            'strict_usable': False, 'new_trading_backtests': 0, 'economic_rule_changes': False})
    return {'status': 'prepared', 'candidate_funds': len(frames), 'not_a_backtest': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('inspect', 'prepare'))
    args = parser.parse_args()
    print(json.dumps(inventory() if args.action == 'inspect' else prepare(), ensure_ascii=False))
