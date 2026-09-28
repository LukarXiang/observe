"""BaoStock 实测：历史深度、字段、单位、退市/停牌/ST、5 分钟起点、公司行动、财务公布日、吞吐。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_baostock.py"""
import time
import baostock as bs, pandas as pd
from _common import Probe, profile

DAY_F = 'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST'
MIN_F = 'date,time,code,open,high,low,close,volume,amount,adjustflag'
SAMPLE = {'sh.600000': '沪主板老股', 'sz.000001': '深主板老股', 'sz.300750': '创业板', 'sh.688981': '科创板',
          'sz.000418': '已退市(吸收合并)', 'sh.600087': '已退市(强制)', 'sz.002450': '已退市(康得新)',
          'sh.600519': '高分红', 'sh.000300': '指数沪深300', 'sh.000001': '上证指数'}
p = Probe('baostock')


def q(rs):
    """逐行读取结果集：baostock 0.9.4 的 get_data() 翻页时调用 pandas 2.0 已删除的 DataFrame.append"""
    if rs.error_code != '0': raise RuntimeError(f'{rs.error_code} {rs.error_msg}')
    rows = []
    while (rs.error_code == '0') & rs.next(): rows.append(rs.get_row_data())
    if rs.error_code != '0': raise RuntimeError(f'分页中断 {rs.error_code} {rs.error_msg}')
    return pd.DataFrame(rows, columns = rs.fields)


def k(code, start, end, freq = 'd', adj = '3', fields = None):
    return q(bs.query_history_k_data_plus(code, fields or (DAY_F if freq == 'd' else MIN_F), start_date = start, end_date = end, frequency = freq, adjustflag = adj))


def calendar():
    d = q(bs.query_trade_dates('1990-01-01', '2026-12-31')); o = d[d.is_trading_day == '1']
    return {'first_open': o.calendar_date.min(), 'last_row': d.calendar_date.max(), 'n_open': len(o)}


def all_stock(day):
    d = q(bs.query_all_stock(day)); p.save(f'all_stock_{day}', d)
    pre = d.code.str[:6].value_counts().to_dict()
    return {'rows': len(d), 'prefix': pre, 'has_000418': bool((d.code == 'sz.000418').any()), 'has_600087': bool((d.code == 'sh.600087').any()), 'cols': list(d.columns)}


def basic():
    d = q(bs.query_stock_basic()); p.save('stock_basic', d)
    return {'rows': len(d), 'type': d.type.value_counts().to_dict(), 'status': d.status.value_counts().to_dict(),
            'delisted_with_outDate': int((d.outDate != '').sum()), 'bj_rows': int(d.code.str.startswith('bj').sum()),
            'sample': d[d.code.isin(SAMPLE)].to_dict('records')}


def daily_full(code):
    d = k(code, '1990-01-01', '2026-09-28'); p.save(f'daily_{code}', d)
    r = profile(d, 'date')
    if len(d):
        r['tradestatus0'] = int((d.tradestatus == '0').sum()); r['isST1'] = int((d.isST == '1').sum())
        r['vol0'] = int((pd.to_numeric(d.volume, errors = 'coerce').fillna(0) == 0).sum())
        x = d[d.tradestatus == '1'].tail(200); vw = pd.to_numeric(x.amount) / pd.to_numeric(x.volume) / pd.to_numeric(x.close)
        r['amount_over_vol_close_median'] = round(float(vw.median()), 4) if len(x) else None  # ≈1 说明成交量单位为股
        if (d.tradestatus == '0').any(): r['susp_row_example'] = d[d.tradestatus == '0'].head(1).to_dict('records')
    return r


def adj_compare(code):
    a = {f: k(code, '2015-01-01', '2026-09-28', adj = f, fields = 'date,close') for f in '123'}
    m = a['3'].rename(columns = {'close': 'raw'}).merge(a['2'].rename(columns = {'close': 'qfq'}), on = 'date').merge(a['1'].rename(columns = {'close': 'hfq'}), on = 'date')
    return {'rows': len(m), 'first': m.head(1).to_dict('records'), 'last': m.tail(1).to_dict('records')}


def min_earliest(code, freq = '5'):
    """逐年 1 月第一周试取，找分钟线最早有数据的年份，再逐月细化"""
    hit = None
    for y in range(1999, 2027):
        if len(k(code, f'{y}-01-01', f'{y}-01-10', freq)): hit = y; break
    if hit is None: return {'earliest_year': None}
    months = [m for m in range(1, 13) if len(k(code, f'{hit - 1}-{m:02d}-01', f'{hit - 1}-{m:02d}-10', freq))] if hit > 1999 else []
    d = k(code, f'{hit - 1}-{(months[0] if months else 12):02d}-01', f'{hit}-01-31', freq)
    return {'earliest_year_jan_hit': hit, 'prev_year_months_hit': months, 'first_date': d.date.min() if len(d) else None}


def min_sample(code, start, end, freq = '5', key = None):
    d = k(code, start, end, freq); p.save(key or f'min{freq}_{code}_{start}', d); r = profile(d, 'time')
    if len(d): c = d.groupby('date').size(); r['bars_per_day'] = c.value_counts().to_dict(); r['first_time'] = d.time.iloc[0]; r['last_time'] = d.time.iloc[-1]
    return r


def whole_market(day):
    d = q(bs.query_daily_history_k_AStock(day)); p.save(f'astock_{day}', d); r = profile(d)
    if len(d): r.update(prefix = d.code.str[:6].value_counts().to_dict(), tradestatus0 = int((d.tradestatus == '0').sum()), isST1 = int((d.isST == '1').sum()))
    return r


def adj_factor_day(day):
    d = q(bs.query_daily_adjust_factor(day)); return {'rows': len(d), 'head': d.head(3).to_dict('records')}


def dividends(code, years):
    out = [q(bs.query_dividend_data(code = code, year = str(y), yearType = 'report')) for y in years]
    d = pd.concat(out, ignore_index = True); p.save(f'dividend_{code}', d); return profile(d)


def adj_factor(code):
    d = q(bs.query_adjust_factor(code = code, start_date = '1990-01-01', end_date = '2026-09-28')); p.save(f'adjfactor_{code}', d); return profile(d, 'dividOperateDate')


def profit(code):
    out = [q(bs.query_profit_data(code = code, year = y, quarter = qq)) for y in (2008, 2016, 2024, 2025) for qq in (1, 2, 3, 4)]
    d = pd.concat(out, ignore_index = True); p.save(f'profit_{code}', d); return profile(d, 'statDate')


def industry():
    d = q(bs.query_stock_industry()); p.save('industry', d); return {'rows': len(d), 'updateDate': d.updateDate.unique()[:3].tolist(), 'classification': d.industryClassification.unique()[:3].tolist()}


def throughput():
    t = time.perf_counter(); d = k('sz.000001', '1990-01-01', '2026-09-28'); a = time.perf_counter() - t
    t = time.perf_counter(); m = k('sz.000001', '2025-09-01', '2026-09-24', '5'); b = time.perf_counter() - t
    return {'daily_full_rows': len(d), 'daily_full_sec': round(a, 2), 'min5_1y_rows': len(m), 'min5_1y_sec': round(b, 2)}


if __name__ == '__main__':
    lg = bs.login(); p.res['_meta'].update(login = f'{lg.error_code} {lg.error_msg}', client = bs.__version__)
    p.run('calendar', calendar)
    for day in ('2005-01-04', '2015-06-01', '2026-09-24'): p.run(f'all_stock_{day}', all_stock, day)
    p.run('stock_basic', basic)
    for c in SAMPLE: p.run(f'daily_{c}', daily_full, c)
    p.run('adj_compare_sh.600519', adj_compare, 'sh.600519')
    for c in ('sh.600000', 'sz.300750', 'sh.000300'): p.run(f'min5_earliest_{c}', min_earliest, c)
    p.run('min1_earliest_sh.600000', lambda: {'rows_2026': len(k('sh.600000', '2026-09-24', '2026-09-24', '1'))})
    p.run('min5_sh.600000_2026-09', min_sample, 'sh.600000', '2026-09-01', '2026-09-24')
    p.run('min5_sz.000418_2019', min_sample, 'sz.000418', '2019-01-02', '2019-01-31')
    p.run('min5_sh.600000_suspend_check', min_sample, 'sh.600000', '2020-01-02', '2020-01-10')
    for f in ('15', '30', '60'): p.run(f'min{f}_sh.600000', min_sample, 'sh.600000', '2026-09-22', '2026-09-24', f)
    for day in ('1995-01-03', '2000-01-04', '2006-01-04', '2010-01-04', '2015-07-08', '2026-09-24'): p.run(f'astock_{day}', whole_market, day)
    for day in ('2026-09-24', '2015-06-01'): p.run(f'adjf_day_{day}', adj_factor_day, day)
    p.run('dividend_sh.600519', dividends, 'sh.600519', range(2010, 2026))
    p.run('adjfactor_sh.600519', adj_factor, 'sh.600519')
    p.run('profit_sh.600519', profit, 'sh.600519')
    p.run('industry', industry)
    p.run('throughput', throughput)
    bs.logout(); p.dump()
