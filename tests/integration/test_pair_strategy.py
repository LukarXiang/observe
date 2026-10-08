import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.data.store import Store
from observe.evaluation.benchmarks import stock_price_levels
from observe.execution import InputBlocked
from observe.replay import reproduce
from observe.strategies import StrategyConfig, _generate_signals, _pair_transition, run_strategy
from scripts.run_strategy_batch4 import pair_signal_check
from tests.integration.helpers import action, bar, instruments, snapshot, tree_hash


def fixture(root, implementation = 'pair_yili_cmb_anchored_v1'):
    days = list(pd.bdate_range('2024-01-02', periods = 170).date)
    first = '600887.SH' if implementation == 'pair_yili_cmb_anchored_v1' else '603288.SH'; second = '600036.SH'
    rows = []
    for k, day in enumerate(days):
        a = 20 if k < 120 else 10
        b = round(22 + .03 * np.sin(k), 4)
        if k == 125: b = 24
        if k == 126: b = 22.1
        if k == 127: b = 20
        if k == 128: b = 22.1
        rows += [bar(day, first, a, pre = a), bar(day, second, b, pre = rows[-1]['close'] if rows else b)]
    adj = pd.DataFrame({'instrument': [first], 'ex_date': [days[120]], 'back_factor': [2.]})
    coverage = pd.DataFrame([{'instrument': i, 'status': 'complete' if i == first else 'no_events',
                             'verified_from': days[0], 'verified_through': days[-1], 'has_start_basis': True,
                             'has_gap': False, 'confirmed_no_events': i == second} for i in (first, second)])
    sid = snapshot(root, rows, instruments(first, second), sessions = days,
                   adj = adj, coverage = coverage, actions = [action(first, days[120], bonus = 1, listed = days[120])])
    source = root / 'source.txt'; source.write_text('# pair fixture\n')
    config = {'source_path': str(source), 'snapshot': sid, 'implementation': implementation, 'instrument': first,
              'execution_instruments': [first, second], 'start': days[125], 'end': days[-1], 'initial_cash': 100000,
              'parameters': {'instrument1': first, 'instrument2': second, 'test_days': 120, 'regression_ratio': 1, 'threshold': 1,
                             'price_basis': 'decision_close_anchor_v1'}, 'benchmark': second, 'benchmark_kind': 'stock',
              'universe': {'boards': ['main'], 'min_listed_sessions': 1, 'exclude_st': False, 'suspend_window': 1,
                           'max_suspended': 1, 'liquidity_window': 1, 'min_avg_amount': 0},
              'portfolio': {'construction': 'target_weights', 'n': 2, 'max_weight': 1, 'rebalance_every': 1, 'rebalance_frequency': 'daily',
                            'buffer': 0, 'max_sell': None, 'refill_between_rebalance': False},
              'execution': {'slippage': .001, 'liquidity_window': 2}}
    return days, config


@pytest.mark.parametrize('state,z,expected', [
    ('empty', 0, 'empty'), ('empty', 1, 'empty'), ('empty', -1, 'empty'), ('empty', 1.01, 'buy1'), ('empty', -1.01, 'buy2'),
    ('buy1', 0, 'buy1'), ('buy1', -.1, 'even'), ('buy1', -1, 'even'), ('buy1', -1.01, 'buy2'),
    ('buy2', -.1, 'buy2'), ('buy2', 0, 'even'), ('buy2', 1, 'even'), ('buy2', 1.01, 'buy1'),
    ('even', -.5, 'even'), ('even', .5, 'even'),
])
def test_pair_original_state_machine_boundaries(state, z, expected):
    assert _pair_transition(state, z) == expected


def test_pair_anchor_formula_global_factor_scale_and_future_invariance(tmp_path):
    days, config = fixture(tmp_path); cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    output = tmp_path / 'signals'; output.mkdir(); _generate_signals(cfg, data, output)
    factors = pd.read_parquet(output / 'factors.parquet').set_index('date')
    first = factors.loc[days[125]]
    assert first.close1 == 10 and first.close2 == 24 and first.spread == 14 and first.basis1 == 2
    assert first.state == 'buy1' and factors.loc[days[127], 'state'] == 'buy2'
    scaled = pd.concat([pd.DataFrame({'instrument': [cfg.instrument], 'ex_date': [days[0]], 'back_factor': [4.]}),
                        data['adj_factors'].assign(back_factor = 8.)], ignore_index = True)
    second = tmp_path / 'scaled'; second.mkdir(); _generate_signals(cfg, {**data, 'adj_factors': scaled}, second)
    other = pd.read_parquet(second / 'factors.parquet').set_index('date')
    pd.testing.assert_frame_equal(factors.drop(columns = 'basis1'), other.drop(columns = 'basis1'))
    changed = data['bars_1d'].copy(); changed.loc[changed.date >= days[140], 'close'] *= 1.5
    future = tmp_path / 'future'; future.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': changed}, future)
    other = pd.read_parquet(future / 'factors.parquet').set_index('date')
    pd.testing.assert_frame_equal(factors.loc[:days[139]], other.loc[:days[139]])
    missing = data['bars_1d'].copy(); missing.loc[missing.date.eq(days[126]) & missing.instrument.eq(cfg.instrument), 'is_trading'] = False
    third = tmp_path / 'paused'; third.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': missing}, third)
    paused = pd.read_parquet(third / 'factors.parquet').set_index('date')
    assert not paused.loc[days[126], 'valid'] and paused.loc[days[127], 'state'] == 'buy1'
    flat = data['bars_1d'].copy(); flat.loc[flat.instrument.eq(config['benchmark']), 'close'] = 22
    zero = tmp_path / 'zero'; zero.mkdir(); _generate_signals(cfg, {**data, 'bars_1d': flat}, zero)
    zero_factors = pd.read_parquet(zero / 'factors.parquet')
    assert zero_factors['std'].eq(0).all() and not zero_factors.valid.any() and zero_factors.state.eq('empty').all()


@pytest.mark.parametrize('implementation', ['pair_yili_cmb_anchored_v1', 'pair_haitian_cmb_anchored_v1'])
def test_pair_shared_ledger_original_sources_stock_benchmark_and_reproduce(tmp_path, monkeypatch, implementation):
    _, config = fixture(tmp_path, implementation)
    result = run_strategy(tmp_path, **config); output = Path(result['output'])
    assert result['status'] == 'success_limited' and len(result['subruns']['backtests']) == 3
    checked = pair_signal_check(tmp_path, output)
    assert checked['sessions_checked'] == 45
    report = json.loads((output / 'report.json').read_text())
    assert report['benchmark'][0]['index']['kind'] == 'stock'
    assert report['benchmark'][0]['metrics']['benchmark_missing_days'] == 0
    assert report['results'][0]['trading']['filled_orders'] > 0
    before = tree_hash(output)
    def forbidden(*a, **kw): raise AssertionError('Offline pair used network')
    monkeypatch.setattr('observe.data.sources.baostock.BaoStock.session', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0
    assert tree_hash(output) == before


def test_pair_scope_parameters_history_and_stock_benchmark_gaps(tmp_path):
    days, config = fixture(tmp_path); cfg = StrategyConfig.model_validate(config); store = Store(tmp_path)
    for changes in ({'execution_instruments': list(reversed(cfg.execution_instruments))}, {'benchmark_kind': 'index'},
                    {'benchmark': '000300.SH'}, {'parameters': {**config['parameters'], 'test_days': 60}}):
        with pytest.raises(ValueError): StrategyConfig.model_validate({**config, **changes})
    data = {t: store.load(t, cfg.snapshot) for t in ('calendar', 'instruments', 'bars_1d', 'adj_factors', 'adj_coverage')}
    short = cfg.model_copy(update = {'start': days[100]})
    with pytest.raises(InputBlocked, match = '120'):
        _generate_signals(short, data, tmp_path / 'short')
    levels, info = stock_price_levels(data['bars_1d'], data['adj_factors'], data['adj_coverage'], days, [days[119], days[120]], cfg.instrument)
    assert list(levels) == [20, 20, 20] and info['kind'] == 'stock'
    bars = data['bars_1d'][~(data['bars_1d'].date.eq(days[120]) & data['bars_1d'].instrument.eq(cfg.instrument))]
    levels, info = stock_price_levels(bars, data['adj_factors'], data['adj_coverage'], days, [days[119], days[120], days[121]], cfg.instrument)
    assert info['missing_levels'] == 1 and np.isnan(levels[2]) and levels[3] == 20
