import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.data.store import Store
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from observe.replay import reproduce
from scripts.run_strategy_batch6 import signal_check
from tests.integration.helpers import bar, instruments, snapshot, tree_hash


def fixture(root):
    days = list(pd.bdate_range('2023-01-02', periods = 340).date)
    prices = [round(20 + 3 * np.sin(k / 12), 4) for k in range(len(days))]
    rows = [bar(d, '600085.SH', p, pre = prices[max(0, k - 1)], amount = 1e7 + k * 1000)
            for k, (d, p) in enumerate(zip(days, prices, strict = True))]
    coverage = pd.DataFrame([{'instrument': '600085.SH', 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1],
                             'has_start_basis': True, 'has_gap': False, 'confirmed_no_events': True}])
    sid = snapshot(root, rows, instruments('600085.SH'), sessions = days, coverage = coverage)
    source = root / 'source.txt'; source.write_text('# svm fixture\n')
    config = yaml.safe_load(Path('configs/strategies/svm_lagged_shape_batch6.yaml').read_text())
    config.update(snapshot = sid, source_path = str(source), start = days[252], end = days[-1], cache = False)
    return days, config


def test_svm_three_costs_weekly_schedule_frozen_training_and_offline_reproduction(tmp_path, monkeypatch):
    _, config = fixture(tmp_path); result = run_strategy(tmp_path, **config); output = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    checked = signal_check(tmp_path, output)
    assert checked['sessions_checked'] == 88 and checked['fits_checked'] > 10
    assert checked['samples_checked'] == checked['fits_checked'] * 225
    assert sum(c['orders_checked'] for c in checked['orders']) > 0
    manifest = json.loads((output / 'signals_manifest.json').read_text())
    assert 'svm_models.json' in manifest['files'] and 'svm_samples.parquet' in manifest['files']
    before = tree_hash(output)
    def forbidden(*a, **kw): raise AssertionError('Offline SVM attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(output) == before


def test_svm_future_prices_cannot_change_earlier_training_or_predictions(tmp_path):
    days, config = fixture(tmp_path); cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    first = tmp_path / 'signals'; first.mkdir(); _generate_signals(cfg, data, first)
    changed = data['bars_1d'].copy(); changed.loc[changed.date >= days[300], 'close'] *= 1.5
    second = tmp_path / 'future'; second.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': changed}, second)
    for name, column in (('factors', 'date'), ('svm_samples', 'decision_date')):
        a, b = (pd.read_parquet(p / f'{name}.parquet') for p in (first, second))
        pd.testing.assert_frame_equal(a[a[column] < days[300]].reset_index(drop = True), b[b[column] < days[300]].reset_index(drop = True))
    a, b = (json.loads((p / 'svm_models.json').read_text()) for p in (first, second))
    assert [v for v in a if v['decision_date'] < str(days[300])] == [v for v in b if v['decision_date'] < str(days[300])]


@pytest.mark.parametrize('change', ['gamma', 'lag', 'instrument', 'schedule', 'refill'])
def test_svm_configuration_rejects_unapproved_changes(tmp_path, change):
    _, config = fixture(tmp_path)
    if change == 'gamma': config['parameters']['svc_parameters']['gamma'] = 'auto'
    elif change == 'lag': config['parameters']['label_horizon'] = 1
    elif change == 'instrument': config['instrument'] = '000333.SZ'
    elif change == 'schedule': config['portfolio']['rebalance_session'] = 2
    else: config['portfolio']['refill_between_rebalance'] = True
    with pytest.raises(ValueError): StrategyConfig.model_validate(config)
