import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.data.store import Store
from observe.execution import InputBlocked
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from tests.integration.helpers import tree_hash
from tests.integration.test_strategies import _fixture


def _boll_fixture(root):
    sid, days, config = _fixture(root); store = Store(root)
    bars = store.load('bars_1d', sid); one = bars.instrument.eq('000333.SZ')
    bars.loc[one, ['open', 'close', 'preclose', 'high', 'low']] = [10, 10, 10, 10.1, 9.9]
    for k, price in ((26, 12), (52, 8)):
        row = one & bars.date.eq(days[k])
        bars.loc[row, ['close', 'high', 'low']] = [price, max(price, 10) + .1, min(price, 10) - .1]
    bars.loc[one & bars.date.eq(days[27]), 'is_trading'] = False
    history = bars.loc[one, 'close'].where(bars.loc[one, 'is_trading']).ffill()
    bars.loc[one, 'preclose'] = history.shift(1).fillna(10).to_numpy()
    store.publish(store.write_batch({'bars_1d': {'2024': store.write_partition('bars_1d', '2024', bars)}}))
    config.update(snapshot = store.snapshot('bollinger fixture'), implementation = 'bollinger_breakout_corrected_v1',
                  instrument = '000333.SZ', parameters = {'boll_window': 20, 'boll_std_multiplier': 2})
    config['portfolio']['refill_between_rebalance'] = False
    return days, config


def test_bollinger_population_std_strict_boundaries_missing_and_state(tmp_path):
    days, config = _boll_fixture(tmp_path); cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    output = tmp_path / 'signals'; output.mkdir(); _generate_signals(cfg, data, output)
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    first = factors.loc[days[25]]
    assert first['std'] == 0 and first.middle == first.upper == first.lower == 10
    assert first.action == 'hold' and first.state == 0
    entry = factors.loc[days[26]]
    assert entry.middle == pytest.approx(10.1) and entry['std'] == pytest.approx(np.sqrt(.19))
    assert entry.upper == pytest.approx(10.1 + 2 * np.sqrt(.19))
    assert entry.action == 'buy' and entry.state == 1
    paused = factors.loc[days[27]]
    assert not paused.valid and paused.action == 'hold' and paused.state == 1
    neutral = factors.loc[days[48]]
    assert neutral['std'] == 0 and neutral.action == 'hold' and neutral.state == 1
    exit_ = factors.loc[days[52]]
    assert exit_.action == 'sell' and exit_.state == 0
    targets = pd.read_parquet(output / 'targets.parquet')
    assert targets.groupby('decision_date').weight.sum().eq(1).all()
    changed = data['bars_1d'].copy(); future = changed.date >= days[60]
    changed.loc[future, 'close'] *= 1.5
    second = tmp_path / 'future'; second.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': changed}, second)
    for name, column in (('factors', 'date'), ('targets', 'decision_date')):
        a, b = pd.read_parquet(output / f'{name}.parquet'), pd.read_parquet(second / f'{name}.parquet')
        pd.testing.assert_frame_equal(a[a[column] < days[60]], b[b[column] < days[60]])


def test_bollinger_uses_ledger_cost_scenarios_and_offline_reproduce(tmp_path, monkeypatch):
    _, config = _boll_fixture(tmp_path)
    def forbidden(*a, **kw): raise AssertionError('No network or fitting in a frozen strategy run')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    monkeypatch.setattr('observe.research._fit_predict', forbidden)
    result = run_strategy(tmp_path, **config); output = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    report = json.loads((output / 'report.json').read_text())
    assert 'ddof=0' in report['formula']
    assert report['results'][0]['trading']['filled_orders'] > 0
    assert verify_run(tmp_path, output)['status'] == 'ok'
    before = tree_hash(output); again = reproduce(tmp_path, output)
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(output) == before


def test_bollinger_parameters_and_history_are_explicit(tmp_path):
    days, config = _boll_fixture(tmp_path)
    with pytest.raises(ValueError, match = 'instrument'):
        StrategyConfig.model_validate({k: v for k, v in config.items() if k != 'instrument'})
    with pytest.raises(ValueError, match = '显式声明'):
        StrategyConfig.model_validate({**config, 'parameters': {}})
    with pytest.raises(ValueError, match = 'daily'):
        StrategyConfig.model_validate({**config, 'portfolio': {**config['portfolio'], 'rebalance_frequency': 'monthly'}})
    with pytest.raises(ValueError, match = '不接受布林'):
        StrategyConfig.model_validate({**config, 'implementation': 'ma10_ma20_v1'})
    old = StrategyConfig.model_validate({**config, 'implementation': 'ma10_ma20_v1', 'parameters': {}})
    assert set(old.parameters.model_dump()) == {'top_fraction', 'positive_only', 'short', 'long', 'buy_multiplier'}
    cfg = StrategyConfig.model_validate({**config, 'parameters': {'boll_window': 100, 'boll_std_multiplier': 2}})
    store = Store(tmp_path); data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    with pytest.raises(InputBlocked, match = '100'):
        _generate_signals(cfg, data, tmp_path / 'short')
    assert cfg.start == days[25]
