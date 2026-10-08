from datetime import date

import pandas as pd
import pytest

from observe.data.store import Store
from observe.execution import InputBlocked, build, load
from observe.ledger import Book


def test_frozen_store_conversion_reaches_book_without_stock_candidate_override(tmp_path):
    day = date(2024, 9, 23); following = date(2024, 9, 24); instrument = '510310.SH'
    store = Store(tmp_path)
    tables = {
        'calendar': pd.DataFrame({'date': [day, following], 'is_open': [True, True]}),
        'instruments': pd.DataFrame([{'instrument': instrument, 'kind': 'etf', 'board': 'main', 'list_date': date(2013, 3, 25)}]),
        'bars_1d': pd.DataFrame([{'instrument': instrument, 'date': d, 'is_trading': True,
                                'open': 4, 'close': 4, 'preclose': 4, 'amount': 1000000} for d in (day, following)]),
        'corp_actions': pd.DataFrame([{'instrument': instrument, 'ex_date': day, 'convert_to': instrument,
                                      'convert_ratio': .49977589, 'convert_rounding': 'ceil'}]),
    }
    parts = {table: {'all': store.write_partition(table, 'all', frame)} for table, frame in tables.items()}
    store.publish(store.write_batch(parts)); snapshot = store.snapshot('synthetic conversion regression')
    inputs = build(load(store, store.state(snapshot), day, following, instruments = [instrument]),
                   day, following, liquidity_window = 1)
    assert all(not candidates for candidates in inputs.candidates.values())
    book = Book(0, inputs.calendar); position = book.pos(instrument)
    position.qty, position.cost, position.last_price = 1000, 2, 2
    book.start_day(day, inputs.actions[day], inputs.market[day])
    assert position.qty == 500 and position.sellable == 500
    assert position.cost * position.qty == 2000
    assert book.close_day(day, inputs.market[day])['equity'] == 2000
    book.start_day(following, inputs.actions.get(following, []), inputs.market[following])
    assert position.qty == 500
    tables['corp_actions'].loc[0, 'convert_to'] = '510300.SH'
    with pytest.raises(InputBlocked, match = 'conversion_destination_missing'):
        build(tables, day, following, liquidity_window = 1)


def test_conversion_with_cash_event_is_not_silently_discarded():
    from observe.execution import _action
    with pytest.raises(InputBlocked, match = 'invalid_conversion'):
        _action({'instrument': '510310.SH', 'ex_date': date(2024, 9, 23), 'convert_to': '510310.SH',
                 'convert_ratio': .5, 'cash_per_share': .1})
