from pathlib import Path

import pandas as pd
import pytest

from observe.data.store import Store
from observe.strategies import RuleParameters, MultiMAParameters, StrategyConfig, _generate_signals, run_strategy
from observe.replay import reproduce
from scripts.run_strategy_batch14 import stable_check
from tests.integration.test_multi_ma_strategy import fixture
from tests.integration.test_strategies import _fixture


def test_old_parameters_keep_original_serialized_shape():
    assert 'mean_algorithm' not in RuleParameters().model_dump()
    params = MultiMAParameters(windows=(5, 10, 20, 30), target_value=20000, struggle10_20=.003, struggle20_30=.002)
    assert 'mean_algorithm' not in params.model_dump()


def test_ma5_stable_policy_holds_at_equal_close_boundary(tmp_path):
    sid, days, config = _fixture(tmp_path)
    config.update(implementation='ma5_ma10_price_v1', parameters={'short': 5, 'long': 10, 'buy_multiplier': 1,
        'mean_algorithm': 'window_fsum_v1'}, instrument='000333.SZ', execution_instruments=['000333.SZ'])
    cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    data = {t: store.load(t, sid) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    bars = data['bars_1d']; selected = bars.instrument.eq('000333.SZ'); bars.loc[selected, 'close'] = 4.3
    window = [5.866102110000001, 5.866102110000001, 5.824276070000001, 5.907928150000001, 5.866102110000001]
    for day, price in zip(days[35:40], window, strict=True): bars.loc[selected & bars.date.eq(day), 'close'] = price
    output = tmp_path / 'signals'; output.mkdir(); _generate_signals(cfg, data, output)
    frame = pd.read_parquet(output / 'factors.parquet').set_index('date')
    assert frame.loc[days[39], 'close_adj'] == frame.loc[days[39], 'ma_short']
    assert frame.loc[days[38], 'state'] == frame.loc[days[39], 'state'] == 1.


def test_stable_multi_mean_three_costs_and_offline_reproduction(tmp_path, monkeypatch):
    _, config = fixture(tmp_path); config['parameters']['mean_algorithm'] = 'window_fsum_v1'
    result = run_strategy(tmp_path, **config); output = Path(result['output'])
    proof = stable_check(tmp_path, output)
    assert result['status'] == 'success_limited' and proof['sessions_checked'] == 138
    assert proof['exact_decimal_windows_checked'] == 552 and proof['numerical_policy'] == 'window_fsum_v1'
    def forbidden(*args, **kwargs): raise AssertionError('Offline reproduction attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0


def test_stable_policy_rejects_unapproved_strategy(tmp_path):
    _, _, config = _fixture(tmp_path)
    config['implementation'] = 'bp_component_v1'; config['parameters']['mean_algorithm'] = 'window_fsum_v1'
    with pytest.raises(ValueError, match='数值算法'):
        StrategyConfig.model_validate(config)
