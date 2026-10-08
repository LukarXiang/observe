"""Source40: restarted TA-Lib RSI windows and frozen slot signals."""
import numpy as np

from .execution import InputBlocked

POOL = ('603799.SH', '300750.SZ', '601633.SH', '603659.SH', '002594.SZ', '603259.SH',
        '601012.SH', '000661.SZ', '600763.SH', '300359.SZ', '300347.SZ', '300014.SZ',
        '300661.SZ', '300073.SZ', '002050.SZ', '002714.SZ', '601888.SH', '002407.SZ',
        '002456.SZ', '300782.SZ', '000333.SZ', '002088.SZ', '600660.SH', '002597.SZ',
        '002821.SZ', '600276.SH', '600196.SH', '002371.SZ', '300595.SZ', '300750.SZ',
        '600309.SH', '002352.SZ', '300357.SZ', '300009.SZ', '300702.SZ', '002595.SZ',
        '300036.SZ', '300037.SZ', '601058.SH', '601677.SH', '601222.SH', '002286.SZ')
UNIQUE_POOL = tuple(dict.fromkeys(POOL))


def backend():
    try: import talib
    except ImportError as exc:
        raise InputBlocked([{'kind': 'rsi_backend_missing', 'detail': 'Install locked indicators extra in the existing platform environment'}]) from exc
    info = {'wrapper_version': talib.__version__, 'c_version': talib.__ta_version__.decode().split(' ')[0],
            'compatibility': talib.get_compatibility(), 'rsi_unstable_period': talib.get_unstable_period('RSI')}
    if info != {'wrapper_version': '0.8.1', 'c_version': '0.8.1', 'compatibility': 0, 'rsi_unstable_period': 0}:
        raise InputBlocked([{'kind': 'rsi_backend_changed', 'detail': str(info)}])
    return talib, info


def flags(raw):
    if not np.isfinite(raw): raise ValueError('RSI must be finite')
    value = int(raw)
    return value, 15 < value < 25, value > 85 or value < 10


def generate(view, calendar, days):
    talib, info = backend()
    tables = {}
    for instrument in UNIQUE_POOL:
        rows = view[view.instrument.eq(instrument)].set_index('date').sort_index()
        if rows.index.duplicated().any(): raise ValueError('RSI input dates duplicate')
        tables[instrument] = rows.reindex(calendar)
    factors, scores, targets, coverage = [], [], [], []
    for day in days:
        buys = sells = 0
        for instrument in UNIQUE_POOL:
            known = tables[instrument].loc[:day]
            window = known[known.is_trading.eq(True)].tail(61)
            if len(window) != 61 or not known.loc[window.index[0]:day, 'is_trading'].map(lambda flag: isinstance(flag, (bool, np.bool_))).all():
                raise InputBlocked([{'kind': 'rsi_missing_window', 'date': str(day), 'instrument': instrument, 'detail': 'Requires 61 traded rows and explicit pause evidence; missing rows are not skipped'}])
            prices = window.close_adj.to_numpy(float)
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise InputBlocked([{'kind': 'rsi_invalid_window', 'date': str(day), 'instrument': instrument, 'detail': 'Required prices missing or nonpositive; no substitute RSI'}])
            raw = float(talib.RSI(prices, timeperiod = 6)[-1])
            if not np.isfinite(raw): raise InputBlocked([{'kind': 'rsi_nonfinite', 'date': str(day), 'instrument': instrument}])
            value, buy, sell = flags(raw); buys += buy * POOL.count(instrument); sells += sell
            first = POOL.index(instrument)
            factors.append({'date': day, 'instrument': instrument, 'rsi_raw': raw, 'rsi_int': value, 'window_start': window.index[0],
                            'window_end': window.index[-1], 'window_rows': 61, 'buy': buy, 'sell': sell,
                            'first_pool_index': first, 'pool_occurrences': POOL.count(instrument)})
            scores.append({'decision_date': day, 'instrument': instrument, 'score': float(-value)})
            targets.append({'decision_date': day, 'instrument': instrument, 'buy': buy, 'sell': sell, 'priority': value, 'rebalance': True})
        coverage.append({'date': day, 'candidates': len(UNIQUE_POOL), 'finite_factors': len(UNIQUE_POOL),
                         'buy_entries': buys, 'sell_signals': sells, 'max_positions': 9})
    return factors, scores, targets, coverage, info


def slot_batch(rows, pool = POOL):
    expected = {'instrument', 'buy', 'sell', 'priority'}
    signals = {}
    for row in rows:
        if set(row) != expected or row['instrument'] in signals: raise ValueError('Invalid or duplicate slot signal')
        if type(row['buy']) is not bool or type(row['sell']) is not bool or type(row['priority']) is not int:
            raise ValueError('Slot flags must be bool and priority integer')
        if row['buy'] and row['sell'] or not 0 <= row['priority'] <= 100: raise ValueError('Invalid slot thresholds')
        if (row['buy'], row['sell']) != flags(row['priority'])[1:]: raise ValueError('Slot flags differ from frozen RSI thresholds')
        signals[row['instrument']] = row
    if set(signals) != set(pool): raise ValueError('Slot signals must cover the full frozen pool')
    buys = sorted((i for i in pool if signals[i]['buy']), key = lambda i: signals[i]['priority'])
    return {'pool': list(pool), 'buys': buys, 'sells': [i for i in dict.fromkeys(pool) if signals[i]['sell']], 'max_positions': 9}
