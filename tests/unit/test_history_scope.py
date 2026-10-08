from datetime import date

import pandas as pd
import pytest

from observe.execution import InputBlocked, load


class HistoricalStore:
    def __init__(self): self.read_bars = False

    def load_state(self, state, table, **kwargs):
        if table == 'calendar':
            return pd.DataFrame({'date': pd.bdate_range('2020-12-01', '2021-02-01').date, 'is_open': True})
        if table == 'instruments': return pd.DataFrame({'instrument': ['600519.SH', '000333.SZ']})
        if table == 'bars_1d': self.read_bars = True
        return pd.DataFrame()


def historical_state():
    return {'tables': {'bars_1d': {
        '2020': {'history_scope': {'policy': 'selected_instruments_v1', 'instruments': ['600519.SH'],
                                  'start': '2020-01-01', 'end': '2020-12-31'}},
        '2021': {}}}}


@pytest.mark.parametrize('instruments', [None, ['000333.SZ'], ['600519.SH', '000333.SZ']])
def test_partial_history_blocks_global_or_uncovered_scope_before_read(instruments):
    store = HistoricalStore()
    with pytest.raises(InputBlocked, match='partial_market_history'):
        load(store, historical_state(), date(2021, 1, 4), date(2021, 2, 1), 20, instruments)
    assert not store.read_bars


@pytest.mark.parametrize('start,instruments', [(date(2021, 1, 4), ['600519.SH']), (date(2021, 2, 1), None)])
def test_selected_scope_and_later_full_market_are_usable(start, instruments):
    store = HistoricalStore(); load(store, historical_state(), start, date(2021, 2, 1), 2, instruments)
    assert store.read_bars


def test_unrecognized_scope_is_blocked():
    state = historical_state(); state['tables']['bars_1d']['2020']['history_scope']['policy'] = 'unknown'
    with pytest.raises(InputBlocked, match='invalid_history_scope'):
        load(HistoricalStore(), state, date(2020, 12, 10), date(2021, 2, 1))


def test_global_research_blocks_partial_history_before_loading_bars(tmp_path, monkeypatch):
    from observe.data.store import Store
    from observe.research import run_research
    from tests.integration.test_strategies import _fixture

    sid, days, _ = _fixture(tmp_path); store = Store(tmp_path); state = store.state(sid)
    for entry in state['tables']['bars_1d'].values():
        entry['history_scope'] = {'policy': 'selected_instruments_v1', 'instruments': ['000333.SZ'],
                                  'start': str(days[0]), 'end': str(days[-1])}
    original = Store.load_state
    monkeypatch.setattr(Store, 'state', lambda self, snapshot=None: state)
    def guarded(self, state, table, **kwargs):
        assert table != 'bars_1d', 'Coverage gate must run before loading the partial market'
        return original(self, state, table, **kwargs)
    monkeypatch.setattr(Store, 'load_state', guarded)
    result = run_research(tmp_path, snapshot=sid)
    assert result['status'] == 'blocked' and result['blocked'][0]['kind'] == 'partial_market_history'


def test_unbounded_traded_history_stops_at_latest_uncovered_partition():
    from observe.execution import covered_history_warmup, sessions
    store = HistoricalStore(); state = historical_state(); days = sessions(store.load_state(state, 'calendar'))
    start = date(2021, 2, 1)
    warmup = covered_history_warmup(state, days, start, start, ['000333.SZ'])
    assert warmup == sum(date(2021, 1, 1) <= d < start for d in days)
    load(store, state, start, start, warmup, ['000333.SZ'])
    assert store.read_bars
    assert covered_history_warmup(state, days, start, start, ['600519.SH']) == len(days)


@pytest.mark.parametrize('fault', ['decision_uncovered', 'unknown_scope'])
def test_traded_history_cannot_shorten_away_decision_gaps_or_unknown_scope(fault):
    from observe.execution import covered_history_warmup, sessions
    state = historical_state(); days = sessions(HistoricalStore().load_state(state, 'calendar'))
    if fault == 'unknown_scope': state['tables']['bars_1d']['2020']['history_scope']['policy'] = 'unknown'
    with pytest.raises(InputBlocked):
        covered_history_warmup(state, days, date(2020, 12, 15), date(2021, 2, 1), ['000333.SZ'])


def test_future_partial_scope_does_not_change_traded_history_loading():
    from observe.execution import covered_history_warmup, sessions
    state = historical_state(); days = sessions(HistoricalStore().load_state(state, 'calendar'))
    start = date(2021, 2, 1); before = covered_history_warmup(state, days, start, start, ['000333.SZ'])
    state['tables']['bars_1d']['2022'] = {'history_scope': {'policy': 'selected_instruments_v1', 'instruments': ['600519.SH'],
        'start': '2022-01-01', 'end': '2022-12-31'}}
    assert covered_history_warmup(state, days, start, start, ['000333.SZ']) == before


@pytest.mark.parametrize('listing,allowed', [(date(2021, 1, 1), True), (date(2020, 12, 31), False), (None, False)])
def test_partial_scope_allows_only_proven_prelisting_absence(listing, allowed):
    from observe.execution import check_history_scope
    args = (historical_state(), date(2020, 12, 1), date(2021, 2, 1), ['000333.SZ'])
    if allowed:
        check_history_scope(*args, listings={'000333.SZ': listing})
    else:
        with pytest.raises(InputBlocked, match='partial_market_history'):
            check_history_scope(*args, listings={'000333.SZ': listing})
    with pytest.raises(InputBlocked, match='partial_market_history'):
        check_history_scope(*args)
