"""存储占用实测：用通达信真实逐笔合成 1/5/15/30/60 分钟 K 线，按项目存法写 Parquet，量每行字节数，再乘 25 年的「股票 × 交易日」数。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_storage.py"""
import io, time, logging, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from mootdx.quotes import Quotes
from probe_tdx import all_trans, HOST
from _common import OUT

logging.getLogger('mootdx').setLevel(logging.ERROR)
CODES = ['600000', '600519', '601318', '600036', '600887', '601988', '600104', '600309', '603288', '605499',
         '000001', '000002', '000333', '000651', '000858', '002415', '002594', '002475', '003816', '001979',
         '300750', '300059', '300015', '300124', '301236', '688981', '688111', '688599', '688008', '688036',
         '600010', '600028', '601857', '600048', '600585', '600690', '600900', '601888', '603259', '600276',
         '000063', '000100', '000725', '000776', '002027', '002230', '002352', '002714', '300274', '300760',
         '601012', '601166', '601398', '601668', '601899', '603501', '603986', '000568', '000938', '002142']
STOCK_DAYS = 16.52e6          # 2001–2025 年在市「股票 × 交易日」（由 BaoStock 上市退市日期推算，含停牌日，偏上限）
BARS = {'1m': 240, '5m': 48, '15m': 16, '30m': 8, '60m': 4, '1d': 1}


def bars_1m(t, date, sz):
    t = t[t.time.astype(str) <= '15:00'].reset_index(drop = True); hm = t.time.astype(str); ts = pd.to_datetime(date + ' ' + hm)
    end = ts + pd.Timedelta('1min')
    if sz: end = end.mask((hm != hm.shift()) & (hm > '09:30'), ts)
    end = end.mask(hm < '09:30', pd.Timestamp(date + ' 09:31')).mask(hm == '11:30', pd.Timestamp(date + ' 11:30')).mask(hm == '15:00', pd.Timestamp(date + ' 15:00'))
    p, v = t.price.astype(float), t.vol.astype(float) * 100
    g = pd.DataFrame({'end': end, 'p': p, 'v': v, 'a': p * v}).groupby('end')
    return pd.DataFrame({'open': g.p.first(), 'high': g.p.max(), 'low': g.p.min(), 'close': g.p.last(), 'volume': g.v.sum(), 'amount': g.a.sum()}).reset_index()


def resample(m, freq):
    """按开盘后第几分钟编号（上午 1–120、下午 121–240）分组，保证 60 分钟线在 10:30/11:30/14:00/15:00 结束"""
    n = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '1d': 240}[freq]
    day = m.end.dt.normalize(); am = m.end.dt.strftime('%H:%M') <= '11:30'
    idx = np.where(am, (m.end - day - pd.Timedelta('9h30min')).dt.total_seconds() // 60, 120 + (m.end - day - pd.Timedelta('13h')).dt.total_seconds() // 60)
    k = np.ceil(idx / n) * n
    end = day + pd.to_timedelta(np.where(k <= 120, 570 + k, 780 + k - 120), unit = 'min')
    g = m.assign(end = end).groupby(['instrument', 'end'])
    return pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last(), 'volume': g.volume.sum(), 'amount': g.amount.sum()}).reset_index()


def size(df, compact):
    df = df.sort_values(['instrument', 'end']).reset_index(drop = True)
    if compact:  # 价格按「分」存 int32、代码字典编码、成交量 int64、成交额按「分」int64
        t = pa.table({'instrument': pa.array(df.instrument).dictionary_encode(), 'end': pa.array(df.end, pa.timestamp('s')),
                      **{c: pa.array(np.round(df[c] * 100).astype('int32')) for c in ('open', 'high', 'low', 'close')},
                      'volume': pa.array(df.volume.astype('int64')), 'amount': pa.array(np.round(df.amount * 100).astype('int64'))})
    else:
        t = pa.table({'instrument': pa.array(df.instrument), 'end': pa.array(df.end, pa.timestamp('ns')), **{c: pa.array(df[c].astype('float64')) for c in ('open', 'high', 'low', 'close', 'volume', 'amount')}})
    b = io.BytesIO(); pq.write_table(t, b, compression = 'zstd'); return b.tell() / len(df)


if __name__ == '__main__':
    c = Quotes.factory(market = 'std', server = HOST, timeout = 10)
    days = [d.strftime('%Y%m%d') for d in pd.bdate_range('2024-03-01', '2024-03-29')][:20]
    ms, ticks, t0 = [], [], time.perf_counter()
    for code in CODES:
        for d in days:
            t = all_trans(c, code, d)
            if not len(t): continue
            ticks.append(t.assign(instrument = code, date = d)); ms.append(bars_1m(t, f'{d[:4]}-{d[4:6]}-{d[6:]}', code[0] != '6').assign(instrument = code))
    c.close(); m1 = pd.concat(ms, ignore_index = True); tk = pd.concat(ticks, ignore_index = True)
    sd = m1.groupby('instrument').end.apply(lambda s: s.dt.date.nunique()).sum()
    res = {'stock_days_sampled': int(sd), 'fetch_sec': round(time.perf_counter() - t0, 1), 'ticks_per_stock_day': round(len(tk) / sd)}
    for f in ('1m', '5m', '15m', '30m', '60m', '1d'):
        df = resample(m1, f); res[f] = {'rows_per_stock_day': round(len(df) / sd, 1), 'bytes_per_row_plain': round(size(df, False), 1), 'bytes_per_row_compact': round(size(df, True), 1)}
    b = io.BytesIO(); pq.write_table(pa.Table.from_pandas(tk.assign(price = tk.price.astype('float32'), vol = tk.vol.astype('int32'), buyorsell = tk.buyorsell.astype('int8'))[['instrument', 'date', 'time', 'price', 'vol', 'buyorsell']], preserve_index = False), b, compression = 'zstd')
    res['tick'] = {'rows_per_stock_day': res['ticks_per_stock_day'], 'bytes_per_row_compact': round(b.tell() / len(tk), 1)}
    csv = m1.head(200000).to_csv(index = False).encode(); res['1m']['bytes_per_row_csv'] = round(len(csv) / 200000, 1)
    res['1m']['bytes_per_row_in_pandas'] = round(m1.memory_usage(deep = True).sum() / len(m1), 1)
    for f, v in res.items():
        if isinstance(v, dict):
            n = STOCK_DAYS * BARS.get(f, v['rows_per_stock_day']); v['rows_25y_billion'] = round(n / 1e9, 3)
            for k in ('plain', 'compact', 'csv', 'in_pandas'):
                if f'bytes_per_row_{k}' in v: v[f'GB_25y_{k}'] = round(n * v[f'bytes_per_row_{k}'] / 1e9, 1)
    print(json.dumps(res, ensure_ascii = False, indent = 1))
    (OUT / 'storage.json').write_text(json.dumps(res, ensure_ascii = False, indent = 1), encoding = 'utf-8')
