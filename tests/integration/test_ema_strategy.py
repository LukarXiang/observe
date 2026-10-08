from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from observe.data.store import Store
from observe.execution import InputBlocked
from observe.ledger.book import Book, Position
from observe.portfolio import slot_value_orders
from observe.replay import reproduce
from observe.strategies import StrategyConfig, _freeze_inputs, run_strategy
from observe.strategy_ema import ANCHOR, POOL, backend, generate, slot_batch
from tests.integration.helpers import bar, instruments, snapshot, tree_hash
from tests.unit.test_portfolio_loop import D, ZERO, flat

talib = pytest.importorskip('talib')


def fixture(root):
    # Synthetic calendar has an anchor block and a later execution block.
    days = [*pd.bdate_range(ANCHOR, periods=70).date, *pd.bdate_range('2024-01-02', periods=45).date]
    prices = [round(30 + k * .03 + 2 * np.sin(k / 3), 4) for k in range(len(days))]
    rows = [bar(d, i, px, pre=prices[max(0, k - 1)]) for i in POOL for k, (d, px) in enumerate(zip(days, prices, strict=True))]
    master = instruments(*POOL); master['list_date'] = days[0]
    coverage = pd.DataFrame([{'instrument': i, 'status': 'no_events', 'verified_from': days[0], 'verified_through': days[-1],
        'has_start_basis': True, 'has_gap': False, 'confirmed_no_events': True} for i in POOL])
    index = pd.DataFrame([{'date': d, 'index': '000300.SH', 'close': 100. + k} for k, d in enumerate(days)])
    sid = snapshot(root, rows, master, sessions=days, coverage=coverage, extra={'index_1d': {'all': index}})
    source = root / 'source.txt'; source.write_text('# EMA fixture\n')
    cfg = yaml.safe_load(Path('configs/strategies/ema_slots_talib_batch19.yaml').read_text())
    cfg.update(snapshot=sid, source_path=str(source), start=days[70], end=days[-1])
    return days, cfg


def test_continuous_seed_pauses_missing_and_future_invariance(tmp_path):
    days, raw = fixture(tmp_path); store = Store(tmp_path)
    view = store.load('bars_1d', raw['snapshot']); view['close_adj'] = view.close
    listings = dict.fromkeys(POOL, ANCHOR)
    first = generate(view, days, days[70:80], listings)[0]
    modified = view.copy(); modified.loc[modified.date >= days[80], 'close_adj'] *= 2
    assert generate(modified, days, days[70:80], listings)[0] == first
    paused = view.copy(); paused.loc[paused.instrument.eq(POOL[0]) & paused.date.eq(days[65]), 'is_trading'] = False
    rows = generate(paused, days, [days[70]], listings)[0]
    series = paused[paused.instrument.eq(POOL[0]) & paused.is_trading & paused.date.le(days[70])].close_adj.to_numpy(float)
    assert rows[0]['ema60'] == talib.EMA(series, 60)[-1] and rows[0]['paused_rows'] == 1
    missing = paused[~(paused.instrument.eq(POOL[0]) & paused.date.eq(days[65]))]
    with pytest.raises(InputBlocked, match='ema_missing_history'): generate(missing, days, [days[70]], listings)
    with pytest.raises(InputBlocked, match='ema_insufficient_history'): generate(view, days, [days[59]], listings)
    with pytest.raises(InputBlocked, match='ema_history_anchor'): generate(view, days[1:], [days[70]], listings)


def test_slot_postfill_divisor_and_ledger_constraints():
    signals = [{'instrument': i, 'buy': k < 3, 'sell': k == 3, 'priority': k} for k, i in enumerate(POOL)]
    batch = slot_batch(signals); assert batch['buys'] == list(POOL[:3])
    book = Book(100000); book.positions[POOL[3]] = Position(qty=1000, last_price=10, today_buy=1000)
    quotes = {i: flat('A')[D[0]]['A'].copy() for i in POOL}
    gen = slot_value_orders(batch, book, quotes, ZERO, D[0], .25)
    order = next(gen); assert order['side'] == 'sell'
    assert book.execute(order, quotes[POOL[3]], D[0], ZERO)['status'] == 'rejected'
    order = next(gen); assert order['instrument'] == POOL[0] and order['amount'] == 100000 / (4 * 1.5)
    book.execute(order, quotes[POOL[0]], D[0], ZERO)
    order = next(gen); assert order['amount'] == book.cash / (3 * 1.5)
    with pytest.raises(ValueError): slot_batch(signals, list(reversed(POOL)))


def test_three_costs_backend_binding_full_history_and_offline_reproduce(tmp_path, monkeypatch):
    days, cfg = fixture(tmp_path); store = Store(tmp_path); spec = StrategyConfig.model_validate(cfg)
    dest = tmp_path / 'frozen'; dest.mkdir()
    data, _ = _freeze_inputs(store, store.state(cfg['snapshot']), spec, dest)
    assert data['bars_1d'].date.min() == ANCHOR
    result = run_strategy(tmp_path, **cfg); assert result['status'] == 'success_limited'
    out = Path(result['output']); assert (out / 'ema_backend.json').exists()
    factors = pd.read_parquet(out / 'factors.parquet'); assert len(factors) == 45 * 8
    assert factors.history_start.eq(ANCHOR).all() and factors.buy.any() and factors.sell.any()
    before = tree_hash(out)
    def forbidden(*a, **kw): raise AssertionError('Network attempted')
    monkeypatch.setattr('socket.socket.connect', forbidden)
    again = reproduce(tmp_path, result['run_id'])
    assert again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0 and tree_hash(out) == before
    assert len(result['subruns']['backtests']) == 3 and days[-1] == spec.end


def test_backend_change_blocks_without_reset():
    old = talib.get_unstable_period('EMA')
    try:
        talib.set_unstable_period('EMA', 1)
        with pytest.raises(InputBlocked, match='ema_backend_changed'): backend()
        assert talib.get_unstable_period('EMA') == 1
    finally: talib.set_unstable_period('EMA', old)


@pytest.mark.parametrize('section,key,value', [('parameters', 'cash_divisor', 1), ('parameters', 'history_anchor', '2010-01-01'),
    ('portfolio', 'participation', .05), ('execution', 'slippage', .001), ('execution', 'liquidity_override', 1e9), ('universe', 'exclude_st', True)])
def test_approved_economic_changes_rejected(tmp_path, section, key, value):
    _, cfg = fixture(tmp_path); cfg[section][key] = value
    with pytest.raises(ValueError): StrategyConfig.model_validate(cfg)
