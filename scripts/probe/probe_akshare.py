"""AKShare 实测：东方财富/新浪/腾讯三个上游的日线与分钟线深度、字段单位、退市股可得性、证券清单、分红、日历、指数。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_akshare.py"""
import akshare as ak, pandas as pd
from _common import Probe, profile

EN = {'日期': 'date', '时间': 'time', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'}
p = Probe('akshare', pause = 1.0)


def em_daily(code, adj = '', start = '19900101'):
    d = ak.stock_zh_a_hist(symbol = code, period = 'daily', start_date = start, end_date = '20260928', adjust = adj); p.save(f'em_daily_{code}_{adj or "raw"}', d)
    r = profile(d.rename(columns = EN), 'date')
    if len(d) and '成交量' in d:
        x = d.tail(200); r['amount_over_vol_close_median'] = round(float((x['成交额'] / x['成交量'] / x['收盘']).median()), 2)  # ≈100 说明成交量单位为手
    return r


def sina_daily(sym, adj = ''):
    d = ak.stock_zh_a_daily(symbol = sym, start_date = '19900101', end_date = '20260928', adjust = adj); p.save(f'sina_daily_{sym}_{adj or "raw"}', d)
    r = profile(d, 'date')
    if len(d): x = d.tail(200); r['amount_over_vol_close_median'] = round(float((x.amount / x.volume / x.close).median()), 2)
    return r


def tx_daily(sym):
    d = ak.stock_zh_a_hist_tx(symbol = sym, start_date = '19900101', end_date = '20260928'); p.save(f'tx_daily_{sym}', d); return profile(d, 'date')


def em_min(code, period, start):
    d = ak.stock_zh_a_hist_min_em(symbol = code, start_date = f'{start} 09:30:00', end_date = '2026-09-28 15:00:00', period = period, adjust = ''); p.save(f'em_min{period}_{code}', d)
    d2 = d.rename(columns = EN); r = profile(d2, 'time')
    if len(d2):
        day = d2.time.astype(str).str[:10]; r['bars_per_day'] = day.value_counts().value_counts().head(5).to_dict(); r['n_days'] = int(day.nunique())
        r['open_zero'] = int((pd.to_numeric(d2.open, errors = 'coerce') == 0).sum())
    return r


def sina_min(sym, period):
    d = ak.stock_zh_a_minute(symbol = sym, period = period, adjust = ''); p.save(f'sina_min{period}_{sym}', d)
    r = profile(d, 'day');
    if len(d): r['n_days'] = int(d.day.astype(str).str[:10].nunique())
    return r


def lst(name, fn, **kw):
    d = fn(**kw); p.save(name, d); return {'rows': len(d), 'cols': list(d.columns), 'head': d.head(2).astype(str).to_dict('records')}


def dividend(name, fn, **kw):
    d = fn(**kw); p.save(name, d); return profile(d)


def calendar():
    d = ak.tool_trade_date_hist_sina(); s = d.trade_date.astype(str); return {'rows': len(d), 'first': s.min(), 'last': s.max(), 'has_2026-09-25': bool((s == '2026-09-25').any())}


if __name__ == '__main__':
    p.res['_meta']['client'] = ak.__version__
    p.run('calendar_sina', calendar)
    for c in ('600000', '000001', '300750', '688981', '000418', '002450', '600087'): p.run(f'em_daily_{c}', em_daily, c)
    p.run('em_daily_600519_qfq', em_daily, '600519', 'qfq'); p.run('em_daily_600519_raw', em_daily, '600519')
    for s in ('sh600000', 'sz000418'): p.run(f'sina_daily_{s}', sina_daily, s)
    p.run('sina_daily_sh600519_qfq', sina_daily, 'sh600519', 'qfq')
    for s in ('sh600000', 'sz000418'): p.run(f'tx_daily_{s}', tx_daily, s)
    for per in ('5', '1'): p.run(f'em_min{per}_600000', em_min, '600000', per, '2015-01-01')
    p.run('em_min5_000418', em_min, '000418', '5', '2019-01-01')
    for per in ('5', '1'): p.run(f'sina_min{per}_sh600000', sina_min, 'sh600000', per)
    p.run('list_a_code_name', lst, 'list_a_code_name', ak.stock_info_a_code_name)
    p.run('list_sh_delist', lst, 'list_sh_delist', ak.stock_info_sh_delist)
    p.run('list_sz_delist', lst, 'list_sz_delist', ak.stock_info_sz_delist, symbol = '终止上市公司')
    p.run('list_bj', lst, 'list_bj', ak.stock_info_bj_name_code)
    p.run('list_st_em', lst, 'list_st_em', ak.stock_zh_a_st_em)
    p.run('list_stop_em', lst, 'list_stop_em', ak.stock_zh_a_stop_em)
    p.run('change_name_000418', lst, 'change_name_000418', ak.stock_info_change_name, symbol = '000418')
    p.run('div_fhps_em_600519', dividend, 'div_fhps_em_600519', ak.stock_fhps_detail_em, symbol = '600519')
    p.run('div_sina_600519', dividend, 'div_sina_600519', ak.stock_history_dividend_detail, symbol = '600519', indicator = '分红')
    p.run('div_cninfo_600519', dividend, 'div_cninfo_600519', ak.stock_dividend_cninfo, symbol = '600519')
    p.run('index_em_000300', lambda: profile(ak.index_zh_a_hist(symbol = '000300', period = 'daily').rename(columns = EN), 'date'))
    p.run('index_sina_sh000300', lambda: profile(ak.stock_zh_index_daily(symbol = 'sh000300'), 'date'))
    bj = p.res.get('list_bj', {}).get('head') or []
    if bj: code = str(list(bj[0].values())[0]); p.run(f'em_daily_bj_{code}', em_daily, code)
    p.dump()
