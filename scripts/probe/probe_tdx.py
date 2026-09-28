"""通达信行情服务器（mootdx）实测：服务器可用性、K 线接口、历史分时/逐笔最早日期、逐笔合成 5 分钟线与 BaoStock 对账、除权除息。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_tdx.py"""
import time, logging
import pandas as pd, baostock as bs
from mootdx.quotes import Quotes
from mootdx.consts import HQ_HOSTS
from _common import Probe, profile

logging.getLogger('mootdx').setLevel(logging.ERROR)
p = Probe('tdx', pause = 0.1)
HOST = ('123.125.108.14', 7709)
n = lambda x: 0 if x is None else len(x)  # mootdx 空结果可能是 None 或空表


def scan():
    ok = []
    for _, ip, port in HQ_HOSTS:
        t = time.perf_counter()
        try:
            c = Quotes.factory(market = 'std', server = (ip, port), timeout = 4); a = c.client
            ok.append({'ip': ip, 'ms': int((time.perf_counter() - t) * 1000), 'bars': len(a.get_security_bars(9, 1, '600000', 0, 10) or []),
                       'minutes': len(a.get_history_minute_time_data(1, '600000', 20260924) or []), 'quotes': len(a.get_security_quotes([(1, '600000')]) or [])}); c.close()
        except Exception as e: ok.append({'ip': ip, 'err': str(e)[:60]})
    good = [x for x in ok if x.get('minutes')]
    return {'n_hosts': len(HQ_HOSTS), 'n_minutes_ok': len(good), 'n_bars_ok': sum(1 for x in ok if x.get('bars')), 'n_quotes_ok': sum(1 for x in ok if x.get('quotes')), 'good': good[:10]}


def earliest(c, code):
    has = lambda d: n(c.minutes(symbol = code, date = d)) > 0
    yrs = [y for y in range(2003, 2008) if has(f'{y}0110') or has(f'{y}0615')]
    if not yrs: return {'earliest': None}
    y = yrs[0]; months = [m for m in range(1, 13) if has(f'{y - 1}{m:02d}15')]
    days = pd.bdate_range(f'{y - 1}-01-01', f'{y}-02-01'); first = next((d.strftime('%Y%m%d') for d in days if has(d.strftime('%Y%m%d'))), None)
    return {'first_year_hit': y, 'prev_year_months_hit': months, 'first_day_hit': first, 'trans_same_day': n(c.transactions(symbol = code, date = first, start = 0, offset = 10)) if first else None}


def all_trans(c, code, date):
    out, start = [], 0
    while True:
        x = c.transactions(symbol = code, date = date, start = start, offset = 2000)
        if x is None or not len(x): break
        out.append(x); start += len(x)
        if len(x) < 2000: break
    return pd.concat(out[::-1], ignore_index = True) if out else pd.DataFrame()


def trans_to_5m(t, date):
    """逐笔合成 5 分钟 K 线：09:25 集合竞价并入第一根，按区间结束时刻标记（与 BaoStock 一致）"""
    t = t[t.time.astype(str) <= '15:00']  # 15:00 后为盘后交易，不进连续竞价 K 线
    hm = t.time.astype(str); ts = pd.to_datetime(date + ' ' + hm)
    end = ts.dt.floor('5min') + pd.Timedelta('5min')  # 分钟标签 HH:MM 含该分钟 00-59 秒
    end = end.mask(hm < '09:30', pd.Timestamp(date + ' 09:35')).mask(hm == '11:30', pd.Timestamp(date + ' 11:30')).mask(hm == '15:00', pd.Timestamp(date + ' 15:00'))
    g = t.assign(end = end).groupby('end')
    return pd.DataFrame({'open': g.price.first(), 'high': g.price.max(), 'low': g.price.min(), 'close': g.price.last(), 'vol_lots': g.vol.sum(), 'n': g.size()}).reset_index()


def reconcile(c, code, date):
    t = all_trans(c, code, date); p.save(f'trans_{code}_{date}', t)
    if not len(t): return {'trans_rows': 0}
    m = trans_to_5m(t, f'{date[:4]}-{date[4:6]}-{date[6:]}')
    bs.login(); r = bs.query_history_k_data_plus(f"{'sh' if code[0] == '6' else 'sz'}.{code}", 'time,open,high,low,close,volume', start_date = f'{date[:4]}-{date[4:6]}-{date[6:]}', end_date = f'{date[:4]}-{date[4:6]}-{date[6:]}', frequency = '5', adjustflag = '3').get_data(); bs.logout()
    if not len(r): return {'trans_rows': len(t), 'bars_from_trans': len(m), 'baostock_rows': 0}
    r['end'] = pd.to_datetime(r.time.str[:12], format = '%Y%m%d%H%M'); r[['open', 'high', 'low', 'close', 'volume']] = r[['open', 'high', 'low', 'close', 'volume']].astype(float)
    j = m.merge(r, on = 'end', how = 'outer', suffixes = ('_tdx', '_bs')); p.save(f'reconcile5m_{code}_{date}', j)
    both = j.dropna(subset = ['close_tdx', 'close_bs'])
    return {'after_close_rows': int((t.time.astype(str) > '15:00').sum()), 'trans_rows': len(t), 'trans_time_range': [str(t.time.iloc[0]), str(t.time.iloc[-1])], 'bars_from_trans': len(m), 'baostock_rows': len(r), 'matched': len(both),
            'close_eq_pct': round(float((both.close_tdx.round(2) == both.close_bs.round(2)).mean() * 100), 1), 'open_eq_pct': round(float((both.open_tdx.round(2) == both.open_bs.round(2)).mean() * 100), 1),
            'high_eq_pct': round(float((both.high_tdx.round(2) == both.high_bs.round(2)).mean() * 100), 1), 'low_eq_pct': round(float((both.low_tdx.round(2) == both.low_bs.round(2)).mean() * 100), 1),
            'vol_ratio_median(bs_shares/tdx_lots)': round(float((both.volume / both.vol_lots).median()), 2), 'day_vol_tdx_lots': float(t.vol.sum()), 'day_vol_bs_shares': float(r.volume.sum())}


def minutes_check(c, code, date):
    m = c.minutes(symbol = code, date = date); p.save(f'minutes_{code}_{date}', m); return profile(m)


def throughput(c, code, days_n = 40):
    days = pd.bdate_range('2024-03-01', periods = days_n).strftime('%Y%m%d'); t = time.perf_counter(); rows = sum(n(c.minutes(symbol = code, date = d)) for d in days)
    a = time.perf_counter() - t; t = time.perf_counter(); tr = sum(len(all_trans(c, code, d)) for d in days[:10]); b = time.perf_counter() - t
    return {'minutes_req': days_n, 'minutes_sec': round(a, 1), 'minutes_rows': rows, 'trans_days': 10, 'trans_sec': round(b, 1), 'trans_rows': tr}


if __name__ == '__main__':
    import sys
    if 'noscan' not in sys.argv: p.run('scan_hosts', scan)
    c = Quotes.factory(market = 'std', server = HOST, timeout = 10)
    for code in ('600000', '000001'): p.run(f'earliest_{code}', earliest, c, code)
    p.run('minutes_600000_20260924', minutes_check, c, '600000', '20260924')
    p.run('minutes_000418_20180105', minutes_check, c, '000418', '20180105')  # 已退市股的历史分时
    for code, d in (('600000', '20260924'), ('000001', '20260924'), ('600000', '20150105'), ('300750', '20200303')): p.run(f'reconcile5m_{code}_{d}', reconcile, c, code, d)
    for code in ('600519', '000418', '300750'):
        p.run(f'xdxr_{code}', lambda s: (lambda x: (p.save(f'xdxr_{s}', x), {**profile(x), 'category': x.category.value_counts().to_dict() if len(x) else {}})[1])(c.xdxr(symbol = s)), code)
    p.run('finance_600519', lambda: profile(c.finance(symbol = '600519')))
    p.run('throughput_600000', throughput, c, '600000')
    c.close(); p.dump()
