from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.data.store import Store
from observe.execution import InputBlocked
from observe.replay import reproduce
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from scripts.run_strategy_batch8 import signal_check
from tests.integration.helpers import bar, instruments, snapshot, tree_hash

PAIR = ['002415.SZ', '000651.SZ']


def fixture(root):
    days = list(pd.bdate_range('2024-01-02', periods = 155).date)
    prices = {PAIR[0]: [30 + 4 * np.sin(k / 12) + (3 if k > 90 else 0) for k in range(len(days))], PAIR[1]: [25.] * len(days)}
    rows = [bar(d, i, px, pre = prices[i][max(0, k - 1)]) for i in PAIR for k, (d, px) in enumerate(zip(days, prices[i], strict = True))]
    coverage = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1], 'has_start_basis': True,
                             'has_gap': False, 'confirmed_no_events': True} for i in PAIR])
    index = pd.DataFrame([{'date': d, 'index': '000300.SH', 'close': 100. + k} for k, d in enumerate(days)])
    sid = snapshot(root, rows, instruments(*PAIR), sessions = days, coverage = coverage, extra = {'index_1d': {'all': index}})
    source = root / 'source.txt'; source.write_text('# rotation fixture\n')
    cfg = yaml.safe_load(Path('configs/strategies/pair_zscore_rotation_batch8.yaml').read_text())
    cfg.update(snapshot = sid, source_path = str(source), start = days[60], end = days[-1], cache = False)
    return days, cfg


def test_rotation_three_costs_offline_and_original_readonly(tmp_path, monkeypatch):
    _, cfg = fixture(tmp_path); result = run_strategy(tmp_path, **cfg); out = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    fac = pd.read_parquet(out / 'factors.parquet'); assert fac.action.gt(0).any()
    assert signal_check(tmp_path, out)['sessions_checked'] == 95
    before = tree_hash(out)
    def forbidden(*a, **kw): raise AssertionError('Offline run attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0 and tree_hash(out) == before


def test_original_state_rounding_and_sample_std_future_invariance_and_missing(tmp_path):
    days, raw = fixture(tmp_path); cfg = StrategyConfig.model_validate(raw); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    out = tmp_path / 'signals'; out.mkdir(); _generate_signals(cfg, data, out); fac = pd.read_parquet(out / 'factors.parquet')
    wide = data['bars_1d'].pivot(index = 'date', columns = 'instrument', values = 'close'); held = 0
    for row in fac.itertuples():
        window = wide.loc[:row.date].tail(60); spread = window[PAIR[0]] - window[PAIR[1]]
        z = float(round((spread.iloc[-1] - spread.mean()) / spread.std(ddof = 1), 4)); action = 0
        if held != 1 and z <= -2: held = action = 1
        elif held != 2 and z >= 2: held = action = 2
        assert row.z_rounded == z and row.state == held and row.action == action
    future = data['bars_1d'].copy(); future.loc[future.date >= days[120], 'close'] *= 3
    later = tmp_path / 'future'; later.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': future}, later)
    pd.testing.assert_frame_equal(fac[fac.date < days[120]].reset_index(drop = True), pd.read_parquet(later / 'factors.parquet').query('date < @days[120]').reset_index(drop = True))
    invalid = data['bars_1d'].copy(); invalid.loc[invalid.date.eq(days[60]), 'close'] = np.nan
    with pytest.raises(InputBlocked, match = '60 complete'):
        _generate_signals(cfg, {**data, 'bars_1d': invalid}, tmp_path / 'missing')
    zero = data['bars_1d'].copy(); zero['close'] = 25.
    with pytest.raises(InputBlocked, match = 'zero_std'):
        _generate_signals(cfg, {**data, 'bars_1d': zero}, tmp_path / 'zero')


@pytest.mark.parametrize('change', [{'execution_instruments': PAIR[::-1]}, {'instrument': PAIR[1]}, {'parameters': {'history': 120}},
                                   {'portfolio': {'construction': 'target_weights', 'buffer': 0, 'max_sell': None}}])
def test_rotation_frozen_contract_rejects_changes(tmp_path, change):
    _, cfg = fixture(tmp_path)
    with pytest.raises(ValueError): StrategyConfig.model_validate({**cfg, **change})
