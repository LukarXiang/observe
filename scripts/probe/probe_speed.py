"""吞吐实测：估算全量建库耗时。BaoStock 与通达信分开跑，BaoStock 全机只能一个会话。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_speed.py bs|tdx"""
import sys, time, logging
import pandas as pd
from _common import Probe

p = Probe('speed', pause = 0)


def bs_speed():
    import baostock as bs
    from probe_baostock import q, k
    bs.login()
    days = q(bs.query_trade_dates('2024-03-01', '2024-03-31')); days = days[days.is_trading_day == '1'].calendar_date.tolist()[:10]
    t = time.perf_counter(); rows = [len(q(bs.query_daily_history_k_AStock(d))) for d in days]; a = time.perf_counter() - t
    p.res['bs_astock_10days'] = {'sec': round(a, 1), 'per_day_sec': round(a / 10, 2), 'rows': rows}; print(p.res['bs_astock_10days'], flush = True)
    codes = ['sh.600004', 'sh.600009', 'sh.600010', 'sh.600011', 'sh.600015', 'sz.000002', 'sz.000063', 'sz.000100', 'sz.000157', 'sz.000333']
    t = time.perf_counter(); rows = [len(k(c, '2005-01-01', '2026-09-24')) for c in codes]; a = time.perf_counter() - t
    p.res['bs_daily_10stocks_2005on'] = {'sec': round(a, 1), 'per_stock_sec': round(a / 10, 2), 'rows': rows}; print(p.res['bs_daily_10stocks_2005on'], flush = True)
    t = time.perf_counter(); rows = [len(k(c, '2025-09-01', '2026-08-31', '5')) for c in codes[:3]]; a = time.perf_counter() - t
    p.res['bs_min5_3stocks_1y'] = {'sec': round(a, 1), 'per_stock_year_sec': round(a / 3, 2), 'rows': rows}; print(p.res['bs_min5_3stocks_1y'], flush = True)
    d = k('sh.600000', '2019-10-01', '2020-01-31', '5'); p.res['bs_min5_first_date'] = {'first': d.date.min() if len(d) else None, 'rows': len(d)}; print(p.res['bs_min5_first_date'], flush = True)
    bs.logout()


def tdx_speed():
    from mootdx.quotes import Quotes
    from probe_tdx import all_trans, n, HOST
    logging.getLogger('mootdx').setLevel(logging.ERROR)
    c = Quotes.factory(market = 'std', server = HOST, timeout = 10)
    days = pd.bdate_range('2024-03-01', periods = 40).strftime('%Y%m%d')
    t = time.perf_counter(); rows = sum(n(c.minutes(symbol = '600000', date = d)) for d in days); a = time.perf_counter() - t
    p.res['tdx_minutes_40req'] = {'sec': round(a, 1), 'per_req_sec': round(a / 40, 3), 'rows': rows}; print(p.res['tdx_minutes_40req'], flush = True)
    t = time.perf_counter(); rows = [len(all_trans(c, '600000', d)) for d in days[:10]]; a = time.perf_counter() - t
    p.res['tdx_trans_10days'] = {'sec': round(a, 1), 'per_stock_day_sec': round(a / 10, 3), 'rows': rows}; print(p.res['tdx_trans_10days'], flush = True)
    # 最早日期：按月找 2004–2006 年第一个有分时的交易日
    first = None
    for d in pd.bdate_range('2004-01-01', '2007-01-31'):
        if d.day not in (5, 15, 25): continue
        if n(c.minutes(symbol = '600000', date = d.strftime('%Y%m%d'))): first = d.strftime('%Y%m%d'); break
    p.res['tdx_minutes_first_hit'] = {'first_hit_among_5_15_25': first}; print(p.res['tdx_minutes_first_hit'], flush = True)
    for code, d in (('000418', '20180105'), ('000418', '20120105'), ('600087', '20120105'), ('002450', '20180105')):
        p.res[f'tdx_delisted_{code}_{d}'] = {'minutes': n(c.minutes(symbol = code, date = d)), 'trans': n(c.transactions(symbol = code, date = d, start = 0, offset = 10))}
        print(code, d, p.res[f'tdx_delisted_{code}_{d}'], flush = True)
    c.close()


if __name__ == '__main__':
    p.name = sys.argv[1]; p.dir = p.dir.parent / f'speed_{p.name}'; p.dir.mkdir(exist_ok = True)
    (bs_speed if p.name == 'bs' else tdx_speed)(); p.dump()
