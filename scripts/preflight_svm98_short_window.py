"""Signals only for the previously inventoried, approved SVM98 early16-day island."""
import json
from pathlib import Path

import pandas as pd
import yaml

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import file_sha
from observe.strategies import StrategyConfig, _freeze_inputs, _generate_signals
from scripts.backtest_pair66_short_windows import ASSETS, ASSETS_SHA, bound
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.run_strategy_batch6 import signal_check
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-history-supplement/20261011-svm98-short-island-preflight')
CONFIG = Path('data/staging/strategies-batch14/20261005-long-history/configs/svm_lagged_shape_batch6-extended.yaml')
CONFIG_SHA = 'd605832c35490d9765b422adf1035e8a0eb6eb1acb3cab6646cee037c9acbf6c'
SOURCE_SHA = '0aba6027a1f433627ff7903c0b5a8df3094bde7013ca1231cc3b498d2efed572'
SNAPSHOT = '20261006-145049-e06c'
INVENTORY = Path('docs/handoff/2026-10-11-approved-window-inventory.json')


def main():
    require(not DIRECTORY.exists(), 'Immutable SVM preflight exists')
    bound(CONFIG, CONFIG_SHA); bound(ASSETS, ASSETS_SHA)
    original = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); bound(original['source_path'], SOURCE_SHA)
    policy = read('configs/research_metadata.json')
    require(next(d for d in policy['decisions'] if d['id'] == 'D-svm-shape')['status'] == 'approved', 'SVM approval missing')
    # The candidate was recorded before fitting or looking at this period's returns.
    inventory = read(INVENTORY)
    candidates = inventory['svm98']['unbacktested_positive_windows']
    require(len(candidates) == 1 and candidates[0]['start'] == '2008-05-28' and
        candidates[0]['end'] == '2008-06-19' and candidates[0]['post_warmup_sessions'] == 16, 'Inventoried SVM island differs')
    cfg = StrategyConfig.model_validate({**original, 'snapshot': SNAPSHOT, 'start': '2008-05-28', 'end': '2008-06-19', 'cache': False})
    require(all(cfg.model_dump(mode='json')[k] == StrategyConfig.model_validate(original).model_dump(mode='json')[k]
        for k in original if k not in ('snapshot', 'start', 'end', 'cache')), 'Economic parameters changed')
    store = Store('data'); state = store.state(SNAPSHOT); checksums = {r['path']: r['file_sha256'] for r in read(ASSETS)['assets']}
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    with network_blocked():
        data, used = _freeze_inputs(store, state, cfg, DIRECTORY)
        for entries in used.values():
            for entry in entries.values():
                path = 'data/'+entry['file']; require(entry['file_sha256'] == checksums[path], 'Input drift relative to archived SHA')
        _generate_signals(cfg, data, DIRECTORY)
        save(DIRECTORY/'config.json', {'kind': 'signal_preflight_only', 'config': cfg.model_dump(mode='json')})
        save(DIRECTORY/'subruns.json', {'backtests': [], 'scope': 'No Book/orders/NAV; independent checker order loop intentionally empty'})
        checks = signal_check('data', DIRECTORY)
        require(checks['sessions_checked'] == 16 and checks['fits_checked'] > 0, 'Signal/model check empty')
        view = with_adjusted(data['bars_1d'], data['adj_factors'], data['adj_coverage'])
        view['date'] = pd.to_datetime(view.date).dt.date
        calendar = sessions(data['calendar']); prices = view.set_index('date').reindex(calendar)
        models = read(DIRECTORY/'svm_models.json')
        for model in models:
            k = calendar.index(pd.Timestamp(model['decision_date']).date()); history = prices.iloc[k-251:k+1]
            require(len(history) == 252 and history.is_trading.eq(True).all() and
                history.adjustment_status.eq('usable').all(), 'Strict252-row fit history missing')
    intended = [r for r in checks['fits'] if r['prediction'] == 1 and str(r['execution_day']) <= str(cfg.end)]
    outputs = {str(p): file_sha(p) for p in DIRECTORY.iterdir() if p.is_file()}
    code = {n: file_sha(n) for n in ('scripts/preflight_svm98_short_window.py', 'scripts/run_strategy_batch6.py',
        'src/observe/strategy_svm.py', 'src/observe/strategies.py', 'src/observe/execution.py', 'src/observe/data/prices.py')}
    save(DIRECTORY/'preflight.json', {'status': 'ok', 'scope': 'approved_svm98_signal_and_model_preflight_only',
        'start': str(cfg.start), 'end': str(cfg.end), 'decision_days': 16, 'checks': checks,
        'potential_positive_target_decisions': intended, 'source_sha256': SOURCE_SHA, 'config_sha256': CONFIG_SHA,
        'snapshot': SNAPSHOT, 'snapshot_sha256': file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')),
        'inventory_sha256': file_sha(INVENTORY), 'input_sha_evidence': str(ASSETS), 'input_sha_evidence_sha256': ASSETS_SHA,
        'outputs_sha256': outputs, 'implementation_sha256': code, 'network_forbidden': True,
        'parameters_changed': False, 'new_trade_backtests': 0, 'orders_executed': 0,
        'limits': ['No actual open/liquidity/Book/fee/NAV acceptance or fresh-worker reproduction yet',
            'Independent fsum arithmetic and fresh SVC on archived samples; same sklearn, not platform equivalence',
            'Original five-session prediction feature lag and frozen back-adjusted input basis retained',
            'Strict training-price windows checked; target decisions are not confirmed fills; no stitched NAV',
            'This signal preflight uses an explicitly empty child list to omit transaction checks']})
    print(json.dumps({'status': 'ok', 'days': 16, 'fits': checks['fits_checked'], 'predictions': checks['prediction_counts'],
        'potential_positive_decisions': len(intended), 'new_trade_backtests': 0}))


if __name__ == '__main__': main()
