from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.data.store import Store
from observe.execution import InputBlocked
from observe.replay import reproduce, run_params
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from scripts.run_strategy_batch5 import signal_check
from tests.integration.helpers import bar, instruments, snapshot, tree_hash


def fixture(root):
    days = list(pd.bdate_range('2024-01-02', periods = 170).date)
    prices = [round(20 + 3 * np.sin(k / 12) + .01 * k, 4) for k in range(len(days))]
    rows = [bar(d, '600196.SH', p, pre = prices[max(0, k - 1)]) for k, (d, p) in enumerate(zip(days, prices, strict = True))]
    coverage = pd.DataFrame([{'instrument': '600196.SH', 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1],
                             'has_start_basis': True, 'has_gap': False, 'confirmed_no_events': True}])
    sid = snapshot(root, rows, instruments('600196.SH'), sessions = days, coverage = coverage)
    source = root / 'source.txt'; source.write_text('# multi ma fixture\n')
    config = yaml.safe_load(Path('configs/strategies/multi_ma_fixed_20k_batch5.yaml').read_text())
    config.update(snapshot = sid, source_path = str(source), start = days[32], end = days[-1], cache = False)
    return days, config


def test_multi_ma_original_branches_actual_holdings_three_costs_and_offline_reproduction(tmp_path, monkeypatch):
    _, config = fixture(tmp_path)
    result = run_strategy(tmp_path, **config); output = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    checked = signal_check(tmp_path, output)
    assert checked['sessions_checked'] == 138 and checked['flags']['bull'] > 0 and checked['flags']['bear'] > 0
    assert sum(c['orders_checked'] for c in checked['orders']) > 0
    before = tree_hash(output)
    def forbidden(*a, **kw): raise AssertionError('Offline run attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(output) == before


def test_multi_ma_future_invariance_pause_and_equal_means(tmp_path):
    days, config = fixture(tmp_path); cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    first = tmp_path / 'signals'; first.mkdir(); _generate_signals(cfg, data, first)
    factors = pd.read_parquet(first / 'factors.parquet').set_index('date')
    future = data['bars_1d'].copy(); future.loc[future.date >= days[100], 'close'] *= 1.5
    second = tmp_path / 'future'; second.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': future}, second)
    pd.testing.assert_frame_equal(factors.loc[:days[99]], pd.read_parquet(second / 'factors.parquet').set_index('date').loc[:days[99]])
    paused = data['bars_1d'].copy(); paused.loc[paused.date.eq(days[50]), 'is_trading'] = False
    third = tmp_path / 'paused'; third.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': paused}, third)
    frame = pd.read_parquet(third / 'factors.parquet').set_index('date')
    assert not frame.loc[days[50]:days[81], 'valid'].any() and frame.loc[days[82], 'valid']
    targets = pd.read_parquet(third / 'targets.parquet').set_index('decision_date')
    assert not targets.loc[days[50]:days[81], ['enter_when_empty', 'exit', 'skip_when_empty']].any().any()
    flat = tmp_path / 'flat'; flat.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': data['bars_1d'].assign(close = 20.)}, flat)
    neutral = pd.read_parquet(flat / 'factors.parquet')
    assert neutral.valid.all() and neutral.struggle.all()
    assert not neutral[['bull', 'bear', 'crossdown', 'crossup']].any().any()
    short = cfg.model_copy(update = {'start': days[30]})
    with pytest.raises(InputBlocked, match = '32'): _generate_signals(short, data, tmp_path / 'short')


@pytest.mark.parametrize('changes', [
    {'instrument': '000333.SZ'}, {'benchmark_kind': 'index'}, {'execution_instruments': None},
    {'parameters': {'windows': [5, 10, 20, 30], 'target_value': 1000000, 'struggle10_20': .003, 'struggle20_30': .002}},
    {'portfolio': {'construction': 'target_weights', 'buffer': 0, 'max_sell': None}},
])
def test_multi_ma_rejects_silent_rule_changes(tmp_path, changes):
    _, config = fixture(tmp_path)
    with pytest.raises(ValueError): StrategyConfig.model_validate({**config, **changes})


def test_conditional_replay_requires_rules_and_matching_frozen_mode(tmp_path):
    from observe.replay import score_source
    _, config = fixture(tmp_path); result = run_strategy(tmp_path, **config)
    cfg = run_params({'snapshot': config['snapshot'], 'portfolio': {'construction': 'conditional_values'}})['config']
    with pytest.raises(ValueError, match = 'rules'): score_source(cfg, cfg.snapshot, tmp_path)
    cfg = run_params({'snapshot': config['snapshot'], 'portfolio': {'construction': 'target_weights'}, 'execution_instruments': ['600196.SH'],
                      'scores': {'source': 'rules', 'run': result['run_id'], 'model': 'multi_ma_fixed_value_v1'}})['config']
    with pytest.raises(ValueError, match = '构建方式'): score_source(cfg, cfg.snapshot, tmp_path)
