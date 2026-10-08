from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.data.store import Store
from observe.execution import InputBlocked
from observe.replay import reproduce, run_params, score_source
from observe.strategies import StrategyConfig, _generate_signals, run_strategy
from observe.strategy_rsi import POOL, UNIQUE_POOL, backend, generate
from scripts.run_strategy_batch7 import signal_check
from tests.integration.helpers import bar, instruments, snapshot, tree_hash

talib = pytest.importorskip('talib')


def fixture(root):
    days = list(pd.bdate_range('2024-01-02', periods = 90).date)
    prices = [round(30 + 4 * np.sin(k / 5), 4) for k in range(len(days))]
    rows = [bar(d, i, px, pre = prices[max(0, k - 1)]) for i in UNIQUE_POOL for k, (d, px) in enumerate(zip(days, prices, strict = True))]
    coverage = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1],
                             'has_start_basis': True, 'has_gap': False, 'confirmed_no_events': True} for i in UNIQUE_POOL])
    index = pd.DataFrame([{'date': d, 'index': '000300.SH', 'close': 100. + k} for k, d in enumerate(days)])
    sid = snapshot(root, rows, instruments(*UNIQUE_POOL), sessions = days, coverage = coverage, extra = {'index_1d': {'all': index}})
    source = root / 'source.txt'; source.write_text('# RSI fixture\n')
    config = yaml.safe_load(Path('configs/strategies/rsi_slots_corrected_batch7.yaml').read_text())
    config.update(snapshot = sid, source_path = str(source), start = days[65], end = days[-1], cache = False)
    return days, config


def test_three_costs_actual_slots_offline_reproduce_and_unchanged_original(tmp_path, monkeypatch):
    _, cfg = fixture(tmp_path); result = run_strategy(tmp_path, **cfg); out = Path(result['output'])
    assert result['status'] == 'success_limited'
    factors = pd.read_parquet(out / 'factors.parquet'); date_accesses = 0; original_getattr = pd.DataFrame.__getattr__
    def counted(frame, name):
        nonlocal date_accesses
        if name == 'date' and len(frame) == 61: date_accesses += 1
        return original_getattr(frame, name)
    with monkeypatch.context() as scope:
        scope.setattr(pd.DataFrame, '__getattr__', counted)
        assert signal_check(tmp_path, out)['sessions_checked'] == 25
    assert date_accesses < len(factors) * 8
    for child in result['subruns']['backtests']:
        import json
        orders = json.loads((Path(child['output']) / 'orders.json').read_text())
        assert orders and len([o for o in orders if o['side'] == 'buy']) == 9
        assert len({o['instrument'] for o in orders if o['side'] == 'buy'}) == 9
    before = tree_hash(out)
    def forbidden(*a, **kw): raise AssertionError('Offline run attempted network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0 and tree_hash(out) == before


def test_restarted_windows_skip_pauses_block_missing_and_future_invariance(tmp_path):
    days, raw = fixture(tmp_path); cfg = StrategyConfig.model_validate(raw); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    out = tmp_path / 'signals'; out.mkdir(); _generate_signals(cfg, data, out)
    first = pd.read_parquet(out / 'factors.parquet')
    altered = data['bars_1d'].copy(); altered.loc[altered.date >= days[80], 'close'] *= 2
    later = tmp_path / 'future'; later.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': altered}, later)
    pd.testing.assert_frame_equal(first[first.date < days[80]].reset_index(drop = True), pd.read_parquet(later / 'factors.parquet').query('date < @days[80]').reset_index(drop = True))
    paused = data['bars_1d'].copy(); paused.loc[paused.instrument.eq(POOL[0]) & paused.date.eq(days[64]), 'is_trading'] = False
    target = tmp_path / 'paused'; target.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': paused}, target)
    factor = pd.read_parquet(target / 'factors.parquet').query('instrument == @POOL[0]').iloc[0]
    assert factor.window_start == days[4] and factor.window_end == days[65]
    missing = paused[~(paused.instrument.eq(POOL[0]) & paused.date.eq(days[64]))]
    with pytest.raises(InputBlocked, match = 'explicit pause'):
        _generate_signals(cfg, {**data, 'bars_1d': missing}, tmp_path / 'missing')
    invalid = data['bars_1d'].copy(); invalid.loc[invalid.date.eq(days[64]), 'close'] = np.nan
    with pytest.raises(InputBlocked, match = 'missing or nonpositive'):
        _generate_signals(cfg, {**data, 'bars_1d': invalid}, tmp_path / 'invalid')
    view = data['bars_1d'].assign(close_adj = data['bars_1d'].close)
    factors, *_ = generate(view, days, [days[-1]])
    series = view[view.instrument.eq(POOL[0])].sort_values('date').close.to_numpy()
    assert factors[0]['rsi_raw'] == talib.RSI(series[-61:], timeperiod = 6)[-1]
    assert abs(factors[0]['rsi_raw'] - talib.RSI(series, timeperiod = 6)[-1]) > 1e-5


def test_backend_settings_block_without_reset():
    old = talib.get_unstable_period('RSI')
    try:
        talib.set_unstable_period('RSI', 1)
        with pytest.raises(InputBlocked, match = 'backend_changed'): backend()
        assert talib.get_unstable_period('RSI') == 1
    finally: talib.set_unstable_period('RSI', old)


@pytest.mark.parametrize('change', [{'execution_instruments': list(UNIQUE_POOL[:-1])}, {'parameters': {'pool': list(UNIQUE_POOL)}},
                                   {'portfolio': {'construction': 'target_weights', 'buffer': 0, 'max_sell': None}}])
def test_frozen_rule_changes_rejected(tmp_path, change):
    _, config = fixture(tmp_path)
    with pytest.raises(ValueError): StrategyConfig.model_validate({**config, **change})


def test_slots_require_rules_source(tmp_path):
    cfg = run_params({'snapshot': 'unused', 'portfolio': {'construction': 'signal_slots'}})['config']
    with pytest.raises(ValueError, match = 'rules'): score_source(cfg, cfg.snapshot, tmp_path)


def test_rsi_covered_tail_keeps_signals_and_blocks_insufficient_traded_rows(tmp_path):
    from observe.strategies import _freeze_inputs
    days, raw = fixture(tmp_path); store = Store(tmp_path); original = store.state(raw['snapshot'])
    earlier = pd.DataFrame({'date': pd.bdate_range('2023-01-02', '2023-12-29').date, 'is_open': True})
    calendar = pd.concat([earlier, store.load_state(original, 'calendar')], ignore_index=True)
    old_bars = store.load_state(original, 'bars_1d'); partial = old_bars.iloc[:0].copy()
    entry = store.write_partition('bars_1d', '2023', partial)
    entry['history_scope'] = {'policy': 'selected_instruments_v1', 'instruments': ['600519.SH'],
        'start': '2023-01-02', 'end': '2023-12-29'}
    bid = store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}, 'bars_1d': {'2023': entry}})
    store.publish(bid); sid = store.snapshot('RSI covered tail regression')
    baseline = tmp_path / 'baseline-signals'; baseline.mkdir()
    cfg = StrategyConfig.model_validate(raw)
    data = {t: store.load_state(original, t) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    _generate_signals(cfg, data, baseline)
    cfg = StrategyConfig.model_validate({**raw, 'snapshot': sid})
    output = tmp_path / 'covered-tail'; output.mkdir()
    data, _ = _freeze_inputs(store, store.state(sid), cfg, output)
    _generate_signals(cfg, data, output)
    pd.testing.assert_frame_equal(pd.read_parquet(baseline / 'factors.parquet'), pd.read_parquet(output / 'factors.parquet'))
    pd.testing.assert_frame_equal(pd.read_parquet(baseline / 'targets.parquet'), pd.read_parquet(output / 'targets.parquet'))
    cfg = StrategyConfig.model_validate({**raw, 'snapshot': sid, 'start': days[50]})
    short = tmp_path / 'too-short'; short.mkdir(); data, _ = _freeze_inputs(store, store.state(sid), cfg, short)
    with pytest.raises(InputBlocked, match='insufficient_history|rsi_missing_window'):
        _generate_signals(cfg, data, short)
