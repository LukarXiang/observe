"""股票池：只用决策日及以前的信息；原因代码固定；上市天数按交易日历计。"""
from datetime import date, timedelta

import pandas as pd

from observe.universe import build_universe

CFG = {'min_listed_sessions': 5, 'suspend_window': 5, 'max_suspended': 2, 'liquidity_window': 3, 'min_avg_amount': 1000.0}
# 12 个交易日，第 6 与第 7 个之间有 10 天长假
DAYS = [date(2024, 1, 2) + timedelta(days = k) for k in range(5)] + [date(2024, 1, 17) + timedelta(days = k) for k in range(7)]


def row(d, i, trading = True, st = False, amount = 5000.0): return {'date': d, 'instrument': i, 'is_trading': trading, 'is_st': st, 'amount': amount if trading else 0.0}


def inst(*rows): return pd.DataFrame([{'instrument': i, 'kind': k, 'list_date': ld, 'delist_date': pd.NaT, 'board': b} for i, k, ld, b in rows])


def reasons(u, i): return dict(zip(u.loc[u.instrument == i, 'decision_date'], u.loc[u.instrument == i, 'reason']))


def test_reason_codes_follow_point_in_time_rules():
    A, ST, SUS, THIN, GEM = '600001.SH', '600002.SH', '600003.SH', '600004.SH', '300001.SZ'
    old = date(2010, 1, 4)
    rows = [row(d, A) for d in DAYS] + [row(d, GEM) for d in DAYS] + [row(d, THIN, amount = 500.0) for d in DAYS]
    rows += [row(d, ST, st = 4 <= k <= 6) for k, d in enumerate(DAYS)]                         # 第 5–7 个交易日 ST，之后摘帽
    rows += [row(d, SUS, trading = not (5 <= k <= 7)) for k, d in enumerate(DAYS)]             # 第 6–8 个交易日停牌
    u = build_universe(pd.DataFrame(rows), inst(*[(i, 'stock', old, 'main') for i in (A, ST, SUS, THIN)], (GEM, 'stock', old, 'gem')), DAYS, {**CFG, 'min_listed_sessions': 1})
    assert [reasons(u, A)[d] for d in DAYS[:2]] == ['no_data'] * 2 and set(reasons(u, A)[d] for d in DAYS[2:]) == {None}   # 前两天流动性回看不足
    assert set(reasons(u, GEM).values()) == {'board_off'}
    assert set(reasons(u, THIN).values()) - {'no_data'} == {'illiquid'}
    st = reasons(u, ST); assert [st[d] for d in DAYS[3:9]] == [None, 'st', 'st', 'st', None, None]
    sus = reasons(u, SUS)
    assert sus[DAYS[4]] is None                                 # 次日停牌不影响当天候选
    assert sus[DAYS[5]] is None and sus[DAYS[7]] == 'suspended_long' and sus[DAYS[10]] is None
    assert (u.universe_version.nunique() == 1)


def test_listed_sessions_are_counted_on_the_trading_calendar():
    NEW = '600009.SH'
    rows = [row(d, NEW) for d in DAYS[2:]]
    u = build_universe(pd.DataFrame(rows), inst((NEW, 'stock', DAYS[2], 'main')), DAYS, {**CFG, 'liquidity_window': 1})
    r = reasons(u, NEW)
    # 上市日 1/4 起：跨越长假后自然日已过 15 天，但交易日只有 1/4、1/5、1/6、1/17 = 4 个 → 仍 too_new；第 5 个交易日 1/18 满足
    assert r[DAYS[5]] == 'too_new' and r[DAYS[6]] is None


def test_listing_before_calendar_start_is_a_conservative_lower_bound():
    # 上市日早于交易日历起点：上市交易日数只能从日历起点算下界，前 4 天保守记为 too_new（研究区间须从日历第 min_listed_sessions 个交易日之后开始）
    A = '600001.SH'
    u = build_universe(pd.DataFrame([row(d, A) for d in DAYS]), inst((A, 'stock', date(2010, 1, 4), 'main')), DAYS, {**CFG, 'liquidity_window': 1})
    assert [reasons(u, A)[d] for d in DAYS[:4]] == ['too_new'] * 4 and set(reasons(u, A)[d] for d in DAYS[4:]) == {None}


def test_only_stocks_with_a_bar_that_day_are_members_and_delisting_is_respected():
    OLD, IDX = '600010.SH', '000300.SH'
    rows = [row(d, OLD) for d in DAYS[:6]] + [row(d, IDX) for d in DAYS]
    meta = pd.concat([inst((OLD, 'stock', date(2010, 1, 4), 'main')).assign(delist_date = pd.Timestamp(DAYS[5])), inst((IDX, 'index', date(2005, 1, 4), 'index'))])
    u = build_universe(pd.DataFrame(rows), meta, DAYS, CFG)
    assert set(u.instrument) == {OLD} and reasons(u, OLD)[DAYS[5]] == 'not_listed' and DAYS[6] not in reasons(u, OLD)
