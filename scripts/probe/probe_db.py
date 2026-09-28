"""存储与查询实测：按项目真实规模造模拟数据（日线约 1300 万行按年分区、5 分钟约 7900 万行按月分区），
测 DuckDB 直接查 Parquet 的典型查询耗时，以及 pandas 整表读取的耗时与内存。数据写在 data/bench/（不入库，测完可删）。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/probe/probe_db.py [build]"""
import sys, time, json, shutil
import numpy as np, pandas as pd, duckdb, pyarrow as pa, pyarrow.parquet as pq
from _common import ROOT, OUT

BENCH = ROOT / 'data' / 'bench'
rng = np.random.default_rng(7)


def codes(n): return [f'{600000 + i:06d}.SH' if i < n // 2 else f'{i:06d}.SZ' for i in range(n)]


def write(df, path):
    path.parent.mkdir(parents = True, exist_ok = True)
    pq.write_table(pa.Table.from_pandas(df.sort_values(['instrument', df.columns[1]]), preserve_index = False), path, compression = 'zstd', row_group_size = 1_000_000)


def build():
    shutil.rmtree(BENCH, ignore_errors = True)
    days = pd.bdate_range('2005-01-04', '2026-09-24'); ins = np.array(codes(5500)); t0 = time.perf_counter()
    for y in range(2005, 2027):
        d = days[days.year == y]; n_live = int(1300 + (y - 2005) * 190); I = np.repeat(ins[:n_live], len(d)); D = np.tile(d.values, n_live)
        close = np.round(np.abs(10 + rng.standard_normal(len(I)).cumsum() * 0.02) + 1, 2)
        df = pd.DataFrame({'instrument': I, 'date': D, 'open': close, 'high': np.round(close * 1.01, 2), 'low': np.round(close * 0.99, 2), 'close': close,
                           'preclose': close, 'volume': rng.lognormal(14, 1, len(I)).astype('int64'), 'amount': np.round(close * 1e6, 2),
                           'turnover': rng.random(len(I)) / 50, 'is_trading': True, 'is_st': False})
        write(df, BENCH / 'bars_1d' / f'{y}.parquet')
    print('1d built', round(time.perf_counter() - t0, 1), 's', flush = True)
    m5 = pd.date_range('2020-01-01 09:35', '2020-01-01 11:30', freq = '5min').append(pd.date_range('2020-01-01 13:05', '2020-01-01 15:00', freq = '5min'))
    off = (m5 - m5.normalize()).values; ins5 = ins[:1000]; t0 = time.perf_counter()
    for mo in pd.period_range('2020-01', '2026-09', freq = 'M'):
        d = days[(days.year == mo.year) & (days.month == mo.month)]
        ts = (d.values[:, None] + off[None, :]).ravel(); I = np.repeat(ins5, len(ts)); T = np.tile(ts, len(ins5))
        p = np.round(np.abs(10 + rng.standard_normal(len(I)).cumsum() * 0.002) + 1, 2)
        df = pd.DataFrame({'instrument': I, 'bar_end': T, 'open': p, 'high': np.round(p * 1.002, 2), 'low': np.round(p * 0.998, 2), 'close': p,
                           'volume': rng.lognormal(11, 1, len(I)).astype('int64'), 'amount': np.round(p * 1e5, 2)})
        write(df, BENCH / 'bars_5m' / f'{mo}.parquet')
    print('5m built', round(time.perf_counter() - t0, 1), 's', flush = True)


def timed(con, name, sql, res):
    t = time.perf_counter(); r = con.execute(sql).fetchall(); s = time.perf_counter() - t
    t = time.perf_counter(); con.execute(sql).fetchall(); s2 = time.perf_counter() - t
    res[name] = {'first_sec': round(s, 3), 'warm_sec': round(s2, 3), 'rows': len(r)}; print(name, res[name], flush = True)


if __name__ == '__main__':
    if 'build' in sys.argv or not BENCH.exists(): build()
    d1, m5 = (BENCH / 'bars_1d' / '*.parquet').as_posix(), (BENCH / 'bars_5m' / '*.parquet').as_posix()
    size = lambda p: round(sum(f.stat().st_size for f in (BENCH / p).glob('*.parquet')) / 1e9, 2)
    con = duckdb.connect(); res = {'1d_rows': con.execute(f"select count(*) from '{d1}'").fetchone()[0], '5m_rows': con.execute(f"select count(*) from '{m5}'").fetchone()[0],
                                   '1d_GB': size('bars_1d'), '5m_GB': size('bars_5m')}
    print(res, flush = True)
    timed(con, '1d 单只全历史（前端 K 线）', f"select * from '{d1}' where instrument = '600100.SH' order by date", res)
    timed(con, '1d 某日全市场（横截面）', f"select * from '{d1}' where date = '2024-03-15'", res)
    timed(con, '1d 全市场 20 日均成交额（窗口计算）', f"select instrument, date, avg(amount) over (partition by instrument order by date rows 19 preceding) from '{d1}' where date >= '2024-01-01'", res)
    timed(con, '5m 单只一年（前端分钟 K 线）', f"select * from '{m5}' where instrument = '600100.SH' and bar_end >= '2025-01-01' and bar_end < '2026-01-01'", res)
    timed(con, '5m 某日全池（横截面）', f"select * from '{m5}' where bar_end >= '2025-03-14' and bar_end < '2025-03-15'", res)
    timed(con, '5m 一年全池算日内特征（实现波动、尾盘收益）', f"""
        with r as (select instrument, cast(bar_end as date) d, bar_end, close, ln(close / lag(close) over (partition by instrument, cast(bar_end as date) order by bar_end)) lr
                   from '{m5}' where bar_end >= '2025-01-01' and bar_end < '2026-01-01')
        select instrument, d, sqrt(sum(lr * lr)) rv, sum(case when strftime(bar_end, '%H:%M') > '14:30' then lr else 0 end) tail_ret from r group by instrument, d""", res)
    t = time.perf_counter(); df = pd.read_parquet(BENCH / 'bars_1d', columns = ['instrument', 'date', 'open', 'close', 'volume', 'amount', 'turnover'])
    res['pandas 读全部日线（7 列）'] = {'sec': round(time.perf_counter() - t, 2), 'rows': len(df), 'GB_in_memory': round(df.memory_usage(deep = True).sum() / 1e9, 2)}
    print('pandas 读全部日线', res['pandas 读全部日线（7 列）'], flush = True); del df
    t = time.perf_counter(); df = pd.read_parquet(BENCH / 'bars_5m' / '2025-03.parquet')
    res['pandas 读一个月 5 分钟全池'] = {'sec': round(time.perf_counter() - t, 2), 'rows': len(df), 'GB_in_memory': round(df.memory_usage(deep = True).sum() / 1e9, 2)}
    print('pandas 读一个月 5m', res['pandas 读一个月 5 分钟全池'], flush = True)
    (OUT / 'db_bench.json').write_text(json.dumps(res, ensure_ascii = False, indent = 1), encoding = 'utf-8')
