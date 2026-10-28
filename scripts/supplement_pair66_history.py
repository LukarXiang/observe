"""Backtest all pre-2020 continuous eligible pair66 windows with at least200 sessions."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import file_sha
from observe.strategies import StrategyConfig
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.run_strategy_batch8 import signal_check
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-history-supplement/20261009-pair66-pre2020')
CONFIG = Path('data/staging/strategies-batch14/20261005-long-history/configs/pair_zscore_rotation_batch8-extended.yaml')
CONFIG_SHA = '1ac0eb44055b594584ae200d411ee75a5fb24b862cf14ee0c268ceda6dcde003'
SOURCE_SHA = '011bfb2aa689d471d949e1813c53089adce4125ae748b06ce227265a81c842fd'
SNAPSHOT = '20261006-145049-e06c'
BASELINE = Path('data/staging/strategies-batch90/20261009-limit-near-high-shortline-stage/baseline.json')
RECEIPT = Path('docs/handoff/2026-10-09-pair66-earlier-windows-verification.json')


def prepare(root, directory):
    policy = read('configs/research_metadata.json')
    require(next(d for d in policy['decisions'] if d['id'] == 'D-rotation66')['status'] == 'approved', 'Pair66 variant not approved')
    require(file_sha(CONFIG) == CONFIG_SHA, 'Original approved config changed')
    config = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    require(file_sha(config['source_path']) == SOURCE_SHA, 'Original pair66 source changed')
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published() == read(BASELINE)['published'] and store.published()['tables'] == state['tables'], 'Snapshot/publication/baseline differs')
    pair = config['execution_instruments']; filters = [('instrument', 'in', pair)]
    bars = store.load_state(state, 'bars_1d', filters=filters)
    factors = store.load_state(state, 'adj_factors', filters=filters)
    coverage = store.load_state(state, 'adj_coverage', filters=filters)
    actions = store.load_state(state, 'corp_actions', filters=filters)
    view = with_adjusted(bars, factors, coverage)
    calendar = sessions(store.load_state(state, 'calendar')); mask = np.ones(len(calendar), dtype=bool)
    for instrument in pair:
        frame = view[view.instrument.eq(instrument)].set_index('date').reindex(calendar)
        mask &= frame.is_trading.eq(True).to_numpy() & frame.adjustment_status.eq('usable').to_numpy() & np.isfinite(frame.close.to_numpy(float)) & \
            (frame.close.to_numpy(float) > 0) & np.isfinite(frame.close_adj.to_numpy(float)) & (frame.close_adj.to_numpy(float) > 0)
    rights = set(actions.loc[actions.rights_ratio.gt(0), 'ex_date'])
    mask &= np.array([day not in rights for day in calendar])
    heads = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    tails = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    candidates = []; selected = []
    for head, tail in zip(heads, tails, strict=True):
        if calendar[tail] >= dt.date(2020, 3, 5):
            continue
        length = int(tail-head+1); count = max(0, length-60)
        row = {'input_start': str(calendar[head]), 'end': str(calendar[tail]),
            'continuous_rows': length, 'post_warmup_sessions': count,
            'selected': count >= 200}
        if count:
            row['start'] = str(calendar[head+60])
        candidates.append(row)
        if row['selected']:
            selected.append(row)
    require([(r['start'], r['end'], r['post_warmup_sessions']) for r in selected] == [
        ('2012-08-21', '2013-09-12', 257), ('2013-12-18', '2014-12-05', 237),
        ('2017-12-19', '2019-03-29', 310)], 'Eligibility windows differ from read-only preflight')
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BASELINE, directory/'baseline.json')
    shutil.copyfile(CONFIG, directory/'approved-original-config.yaml')
    outputs = []
    for number, row in enumerate(selected, 1):
        updated = {**config, 'snapshot': SNAPSHOT, 'start': row['start'], 'end': row['end'], 'cache': False}
        cfg = StrategyConfig.model_validate(updated)
        require({k: updated[k] for k in config if k not in ('snapshot', 'start', 'end', 'cache')} ==
                {k: config[k] for k in config if k not in ('snapshot', 'start', 'end', 'cache')}, 'Economic parameters changed')
        path = directory/f'pair66-earlier-{number}.yaml'
        with path.open('x', encoding='utf-8') as out:
            yaml.safe_dump(cfg.model_dump(mode='json'), out, allow_unicode=True, sort_keys=False)
        outputs.append({**row, 'config_file': str(path), 'config_sha256': file_sha(path), 'label': path.stem})
    save(directory/'prepared-windows.json', {'snapshot': SNAPSHOT, 'source_sha256': SOURCE_SHA,
        'original_config_sha256': CONFIG_SHA, 'baseline_source': str(BASELINE), 'baseline_sha256': file_sha(BASELINE),
        'all_pre2020_candidates': candidates, 'selected': outputs,
        'selection': 'All earlier continuous strict-close windows with at least200 post60-warmup sessions; no outcome selection',
        'initial_state': 'Each independent window restarts1million cash and original signal state; NAVs not stitched',
        'parameters_changed': False, 'data_downloaded': False})
    return {'status': 'prepared', 'windows': outputs}


def validate(directory):
    doc = read(directory/'prepared-windows.json')
    require(file_sha(CONFIG) == file_sha(directory/'approved-original-config.yaml') == CONFIG_SHA and
            file_sha(BASELINE) == file_sha(directory/'baseline.json') == doc['baseline_sha256'], 'Prepared inputs changed')
    for row in doc['selected']:
        require(file_sha(row['config_file']) == row['config_sha256'], 'Frozen config changed')
        require(file_sha(yaml.safe_load(Path(row['config_file']).read_text(encoding='utf-8'))['source_path']) == SOURCE_SHA, 'Source changed')
    return doc


def selected_windows(directory, label=None):
    rows = validate(directory)['selected']
    if label is None:
        return rows
    selected = [row for row in rows if row['label'] == label]
    require(len(selected) == 1, 'Unknown frozen window label')
    return selected


def run(root, directory, label=None):
    return [run_one(root, directory, r['config_file'], batch_label='Pair66 earlier eligible windows')
            for r in selected_windows(directory, label)]


def verify(root, directory, label=None):
    return [verify_one(root, directory, r['label'], checker=signal_check,
            batch_label='Pair66 earlier eligible windows') for r in selected_windows(directory, label)]


def finish(root, directory):
    doc = validate(directory); rows = []
    for row in doc['selected']:
        name = directory/(row['label']+'-verification.json'); verified = read(name)
        require(verified['status'] == 'ok' and verified['reproduction']['reproduction']['result'] == 'match' and
                verified['reproduction']['reproduction']['differences'] == 0 and verified['original_unchanged'], 'Window not verified')
        require(verified['hand_check']['sessions_checked'] == row['post_warmup_sessions'], 'Signal coverage differs')
        report = verified['report']
        rows.append({**row, 'run_id': verified['run']['run_id'], 'reproduction_run_id': verified['reproduction']['run_id'],
            'validation_file': str(name), 'validation_sha256': file_sha(name),
            'hand_check': verified['hand_check'], 'period': report['period'], 'results': report['results'],
            'benchmark': report['benchmark'], 'source_sha256': verified['source_sha256']})
    fees = fee_checks(directory); protection = protect(root, directory)
    progress = archive(root, 'Approved pair66 all three earlier200+session continuous windows: three costs, independent state/Book checks and offline match0; no new source')
    save(RECEIPT, {'status': 'ok', 'nature': 'supplement_existing_approved_approximate_backtests',
        'prepared_file': str(directory/'prepared-windows.json'), 'prepared_sha256': file_sha(directory/'prepared-windows.json'),
        'windows': rows, 'fees': fees, 'protection': protection, 'progress': progress,
        'new_strategy_sources': 0, 'new_original_backtest_parent_runs': 3, 'new_reproduction_parent_runs': 3,
        'source_count_unchanged': 695, 'manually_reviewed_unchanged': 260,
        'original_strategy_complete': False, 'library_goal_complete': False,
        'limits': ['Approved dynamic-anchoring daily approximation, not original intraday/platform equivalence',
            'Independent empty initial state for each eligibility island; do not stitch with each other or existing2020-2026 tail',
            'Fixed stock pool, historical execution/fee approximations and nonindependent holdout caveats retained',
            'Book and original signal state reused; no fee/NAV replacement or pause filling']})
    return {'status': 'ok', 'receipt': str(RECEIPT), 'windows': rows, 'progress': progress}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--label', help='Resume one prepared run or verification without overwriting earlier windows')
    args = parser.parse_args()
    if args.label is not None and args.action not in ('run', 'verify'):
        parser.error('--label is only valid for run or verify')
    result = globals()[args.action](args.root, args.directory, args.label) if args.action in ('run', 'verify') \
        else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
