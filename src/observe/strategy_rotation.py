"""Source66's rounded sample zscore and signal-driven cash rotation."""
import numpy as np

from .execution import InputBlocked


def generate(wide, bases, days, pair):
    first, second = pair; held = 0
    factors, scores, targets, coverage = [], [], [], []
    for day in days:
        k = wide['close_adj'].index.get_loc(day)
        prices = wide['close_adj'][pair].iloc[k - 59:k + 1].to_numpy(float)
        basis = [float(bases.at[day, i]) for i in pair]
        if len(prices) != 60 or not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(basis).all() or min(basis) <= 0:
            raise InputBlocked([{'kind': 'rotation_missing_window', 'date': str(day), 'detail': 'Requires 60 complete paired sessions and current adjustment basis; no filling'}])
        anchored = prices / np.array(basis)
        spread = anchored[:, 0] - anchored[:, 1]; middle = float(spread.mean()); std = float(spread.std(ddof = 1))
        if not np.isfinite(std) or std <= 0:
            raise InputBlocked([{'kind': 'rotation_zero_std', 'date': str(day), 'detail': 'Original zscore undefined; no constant substitute'}])
        raw = float((spread[-1] - middle) / std); z = float(round(np.float64(raw), 4)); previous = held; action = 0
        if held != 1 and z <= -2: held = action = 1
        elif held != 2 and z >= 2: held = action = 2
        factors.append({'date': day, 'instrument': first, 'window_start': wide['close_adj'].index[k - 59], 'window_end': day,
                        'basis1': basis[0], 'basis2': basis[1], 'spread': float(spread[-1]), 'mean': middle, 'std': std,
                        'z_raw': raw, 'z_rounded': z, 'previous_state': previous, 'state': held, 'action': action})
        scores += [{'decision_date': day, 'instrument': first, 'score': -z}, {'decision_date': day, 'instrument': second, 'score': z}]
        targets += [{'decision_date': day, 'instrument': i, 'buy': action == n, 'sell': action != 0 and action != n, 'rebalance': action != 0}
                    for n, i in enumerate(pair, 1)]
        coverage.append({'date': day, 'candidates': 2, 'finite_factors': 1, 'state': held, 'switch': action != 0})
    return factors, scores, targets, coverage


def rotation_batch(rows, pair):
    if len(rows) != 2 or {r['instrument'] for r in rows} != set(pair): raise ValueError('Rotation requires exactly the frozen pair')
    buys, sells = [], []
    for row in rows:
        if set(row) != {'instrument', 'buy', 'sell'} or type(row['buy']) is not bool or type(row['sell']) is not bool or row['buy'] and row['sell']:
            raise ValueError('Invalid cash rotation flags')
        if row['buy']: buys.append(row['instrument'])
        if row['sell']: sells.append(row['instrument'])
    if len(buys) != len(sells) or len(buys) > 1: raise ValueError('Rotation must pair one sell with one buy or hold')
    return {'buy': buys[0] if buys else None, 'sell': sells[0] if sells else None}
