"""Approved source55 approximation: continuous TA-Lib EMA and original cash slots."""
from datetime import date

import numpy as np
import pandas as pd

from .execution import InputBlocked

POOL = ('300014.SZ', '300059.SZ', '300168.SZ', '300253.SZ', '300274.SZ', '000651.SZ', '000858.SZ', '600809.SH')
PERIODS = (2, 25, 60)
ANCHOR = date(2005, 1, 5)


def backend():
    try: import talib
    except ImportError as exc:
        raise InputBlocked([{'kind': 'ema_backend_missing', 'detail': 'Requires locked indicators extra in the existing environment'}]) from exc
    info = {'wrapper_version': talib.__version__, 'c_version': talib.__ta_version__.decode().split(' ')[0],
            'compatibility': talib.get_compatibility(), 'ema_unstable_period': talib.get_unstable_period('EMA')}
    if info != {'wrapper_version': '0.8.1', 'c_version': '0.8.1', 'compatibility': 0, 'ema_unstable_period': 0}:
        raise InputBlocked([{'kind': 'ema_backend_changed', 'detail': str(info)}])
    return talib, info


def generate(view, calendar, days, listings):
    talib, info = backend()
    if not calendar or calendar[0] != ANCHOR or calendar != sorted(set(calendar)):
        raise InputBlocked([{'kind': 'ema_history_anchor', 'detail': 'Requires verified calendar starting 2005-01-05'}])
    if not days or any(day not in calendar for day in days) or set(listings) != set(POOL):
        raise InputBlocked([{'kind': 'ema_signal_scope', 'detail': 'Requires all eight stocks and covered decision days'}])
    tables = {}
    for instrument in POOL:
        listing = listings[instrument]
        if not isinstance(listing, date): raise InputBlocked([{'kind': 'ema_listing_unknown', 'instrument': instrument}])
        span = [d for d in calendar if max(ANCHOR, listing) <= d <= days[-1]]
        rows = view[view.instrument.eq(instrument) & view.date.le(days[-1])].set_index('date').sort_index()
        if rows.index.duplicated().any(): raise ValueError('EMA input dates duplicate')
        rows = rows.reindex(span)
        if not rows.is_trading.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise InputBlocked([{'kind': 'ema_missing_history', 'instrument': instrument, 'detail': 'Every post-listing session needs an explicit trade or pause record'}])
        traded = rows[rows.is_trading.eq(True)]
        prices = traded.close_adj.to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise InputBlocked([{'kind': 'ema_invalid_price', 'instrument': instrument}])
        table = pd.DataFrame(index=span)
        for period in PERIODS:
            table[f'ema{period}'] = pd.Series(talib.EMA(prices, timeperiod=period), index=traded.index).reindex(span).ffill()
        table['traded_rows'] = rows.is_trading.astype(int).cumsum()
        table['paused_rows'] = (~rows.is_trading).astype(int).cumsum()
        table['history_start'] = span[0] if span else None
        tables[instrument] = table.reindex(calendar)
    factors, scores, targets, coverage = [], [], [], []
    for day in days:
        position = calendar.index(day)
        if position < 1: raise InputBlocked([{'kind': 'ema_prior_day_missing', 'date': str(day)}])
        previous = calendar[position - 1]; buys = sells = 0
        for index, instrument in enumerate(POOL):
            table = tables[instrument]
            current, prior = table.loc[day], table.loc[previous]
            values = [current[f'ema{n}'] for n in PERIODS] + [prior[f'ema{n}'] for n in PERIODS]
            if not np.isfinite(values).all():
                raise InputBlocked([{'kind': 'ema_insufficient_history', 'date': str(day), 'instrument': instrument,
                    'detail': 'Requires current and prior EMA60 after full continuous history seed'}])
            a, b, c, old_a, old_b, _ = values
            buy = bool(a > b and old_a < old_b and b > c)
            sell = bool(a < b and old_a > old_b and b > c)
            buys += buy; sells += sell
            factors.append({'date': day, 'instrument': instrument, 'ema2': a, 'ema25': b, 'ema60': c,
                'prior_ema2': old_a, 'prior_ema25': old_b, 'prior_ema60': values[-1], 'prior_date': previous,
                'history_start': current.history_start, 'traded_rows': int(current.traded_rows),
                'paused_rows': int(current.paused_rows), 'pool_index': index, 'buy': buy, 'sell': sell})
            scores.append({'decision_date': day, 'instrument': instrument, 'score': float(-index)})
            targets.append({'decision_date': day, 'instrument': instrument, 'buy': buy, 'sell': sell, 'priority': index, 'rebalance': True})
        coverage.append({'date': day, 'candidates': len(POOL), 'finite_factors': len(POOL), 'buy_entries': buys,
            'sell_signals': sells, 'max_positions': 5, 'cash_divisor': 1.5})
    return factors, scores, targets, coverage, info


def slot_batch(rows, pool=POOL):
    if list(pool) != list(POOL): raise ValueError('EMA requires original eight-stock order')
    signals = {}
    for row in rows:
        if set(row) != {'instrument', 'buy', 'sell', 'priority'} or row['instrument'] in signals or row['instrument'] not in POOL:
            raise ValueError('Invalid EMA slot signal')
        if type(row['buy']) is not bool or type(row['sell']) is not bool or row['buy'] and row['sell']:
            raise ValueError('Invalid EMA slot flags')
        if type(row['priority']) is not int or row['priority'] != POOL.index(row['instrument']): raise ValueError('EMA pool priority differs')
        signals[row['instrument']] = row
    if set(signals) != set(POOL): raise ValueError('EMA slot signals must cover eight stocks')
    return {'pool': list(POOL), 'buys': [i for i in POOL if signals[i]['buy']], 'sells': [i for i in POOL if signals[i]['sell']],
        'max_positions': 5, 'cash_divisor': 1.5, 'order_guard_policy': 'ledger_only_v1'}
