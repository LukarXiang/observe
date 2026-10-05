"""按冻结交易日历安排调仓；周/月交易在选定交易日开盘，决策在前一交易日收盘。"""
from collections import defaultdict

import pandas as pd


def rebalance_dates(dates, calendar, frequency = 'sessions', every = 1, session = 1):
    dates = [pd.Timestamp(d).date() for d in dates]; calendar = sorted({pd.Timestamp(d).date() for d in calendar})
    if frequency not in ('sessions', 'daily', 'weekly', 'monthly'): raise ValueError('未知调仓频率')
    if type(every) is not int or every < 1 or type(session) is not int or session < 1: raise ValueError('调仓间隔与期内交易日序号必须为正整数')
    if frequency == 'sessions': return set(dates[::every])
    if frequency == 'daily': return set(dates)
    groups = defaultdict(list)
    for day in calendar:
        key = day.isocalendar()[:2] if frequency == 'weekly' else (day.year, day.month)
        groups[key].append(day)
    previous = dict(zip(calendar[1:], calendar)); chosen = set()
    for members in groups.values():
        if len(members) >= session and members[session - 1] in previous: chosen.add(previous[members[session - 1]])
    return chosen & set(dates)
