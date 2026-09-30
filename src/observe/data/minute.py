"""外部分钟线导入（模块 10、决策 17）：日压缩包里的 1 分钟 CSV → 逐文件校验 → 与日线对账 → 合成 5 分钟 → bars_5m 按月分区。

来源是用户提供的文件（目录/年/月/YYYYMMDD.zip，个别是 7z），格式随日期变化：表头有三种、CSV 多数没有时间列、文件名有五种写法、
个别日期多套一层日期文件夹或每只证券存了两份。这里按表头名解析，时间戳按行位置生成（09:31..11:30、13:01..15:00，区间结束时刻），
校验不过的「证券 × 交易日」整条剔除并记入审计问题，不修补。
只导入分钟股票池（上一年成交额前 N 的主板股票）：范围小、对账口径清楚。"""
import io, os, re, shutil, subprocess, tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .locks import DATA_WRITER, operation_lock
from .store import Store, _atomic_json

SOURCE = 'external_1m'
NAME = re.compile(r'(?:^|/)(?:(sh|sz|bj))?(\d{6})(?:\.(sh|sz|bj))?\.csv$')
COLS = ['open', 'high', 'low', 'close', 'amount', 'volume']
MINUTES = np.array([571 + i for i in range(120)] + [781 + i for i in range(120)])                 # 自午夜起的分钟数：09:31..11:30、13:01..15:00
TIME_STR = [f'{m // 60:02d}:{m % 60:02d}' for m in MINUTES]
START_STR = [f'{m // 60:02d}:{m % 60:02d}' for m in list(range(571, 690)) + list(range(780, 901))]          # 20251201-03 的新格式：时间列是分钟起点（09:31..11:29 共 119 行、13:00..15:00 共 121 行）
BAR_OFFSETS = pd.to_timedelta(MINUTES[4::5], unit = 'm')                                          # 48 根 5 分钟线的结束时刻：09:35 .. 15:00
TOL = 0.01                                                                                        # 分钟合计与日线成交量 / 成交额的相对容差
VOL_FLOOR, AMT_FLOOR = 500, 5000                                                                  # 分钟成交量按手取整、成交额有浮点误差，小额日再给一点绝对容差
SEVENZIP = ('7z', '7za', 'C:/Program Files/7-Zip/7z.exe', 'C:/Program Files (x86)/7-Zip/7z.exe')


def issue(level, rule, day, instrument = None, detail = ''): return {'level': level, 'rule': rule, 'date': day, 'instrument': instrument, 'detail': detail}


# 分钟股票池 -------------------------------------------------------------------
def build_universe(bars, top = 800, min_days = 120):
    """按年生成名单：上一自然年沪深主板全部交易日成交额的算术平均（停牌日不计）前 top 名、交易日不少于 min_days。
    上一年数据必须完整（首个交易日在 1 月中旬前、末个交易日在 12 月下旬后），否则该年不生成，而不是用残缺年份凑数。"""
    b = bars[['date', 'instrument', 'amount', 'is_trading', 'board']].copy(); b['date'] = pd.to_datetime(b.date)
    b = b[b.is_trading.astype(bool) & (b.board == 'main')]
    rows = []
    for y in sorted(b.date.dt.year.unique()):
        prev = b[b.date.dt.year == y - 1]
        if not len(prev): continue
        if prev.date.min() > pd.Timestamp(y - 1, 1, 15) or prev.date.max() < pd.Timestamp(y - 1, 12, 20): continue
        g = prev.groupby('instrument').agg(avg_amount = ('amount', 'mean'), traded_days = ('amount', 'size'))
        g = g[g.traded_days >= min_days].sort_values(['avg_amount'], ascending = False, kind = 'stable').head(top)
        g['rank'] = np.arange(1, len(g) + 1); g['year'] = y; g['generated_asof'] = prev.date.max().date()
        rows.append(g.reset_index())
    cols = ['year', 'instrument', 'rank', 'avg_amount', 'traded_days', 'generated_asof']
    return pd.concat(rows, ignore_index = True)[cols] if rows else pd.DataFrame(columns = cols)


def plan_days(sessions, years, start = None, end = None, warm = 20):
    """{交易日: 该日需要的名单年份集合}。下载起点 = 名单年份首个交易日前 warm 个交易日（预热），预热期用即将生效的名单；
    跨年的预热期同时保留上一年名单，保证上一年名单里的证券不断档。"""
    sessions = sorted(sessions); first = {}
    for i, d in enumerate(sessions): first.setdefault(d.year, i)
    warm_from = {y: sessions[max(first[y] - warm, 0)] for y in years if y in first}
    lo = start or (min(warm_from.values()) if warm_from else None); hi = end or (sessions[-1] if sessions else None)
    plan = {}
    for d in sessions:
        if lo is None or d < lo or d > hi: continue
        ys = {d.year} if d.year in years else set()
        if d.year + 1 in warm_from and d >= warm_from[d.year + 1]: ys.add(d.year + 1)
        if ys: plan[d] = ys
    return plan


# 读取 -------------------------------------------------------------------------
def archive_path(source, day):
    for ext in ('zip', '7z'):
        p = os.path.join(source, f'{day:%Y}', f'{day:%m}', f'{day:%Y%m%d}.{ext}')
        if os.path.exists(p): return p
    return None


def find_7z(explicit = None):
    for c in filter(None, (explicit, os.environ.get('OBSERVE_7Z'), *SEVENZIP)):
        found = shutil.which(c) or (c if os.path.isfile(c) else None)
        if found: return found
    raise RuntimeError('该日压缩包是 7z 格式，需要 7-Zip：用 --sevenzip 指定 7z.exe 路径，或设置环境变量 OBSERVE_7Z')


def _key(name):
    """压缩包内文件名 → (市场小写, 六位代码)；不是证券 csv 返回 None"""
    m = NAME.search(name)
    if not m: return None
    parts = name.split('/'); mk = parts[-2] if len(parts) > 1 and parts[-2] in ('sh', 'sz', 'bj') else (m.group(1) or m.group(3))
    return (mk, m.group(2)) if mk else None


def read_members(path, keep, sevenzip = None):
    """keep: {(市场小写, 代码): 证券}。返回 {证券: csv 字节}；同一证券出现多份时取第一份（实测内容相同）。
    扩展名是 .zip 但内容是 7z 的文件也按魔数识别。"""
    out = {}
    with open(path, 'rb') as h: magic = h.read(4)
    if magic == b'PK\x03\x04':
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                k = None if i.is_dir() else _key(i.filename)
                if k in keep and keep[k] not in out: out[keep[k]] = z.read(i)
        return out
    exe = find_7z(sevenzip); tmp = tempfile.mkdtemp(prefix = 'minute7z-')
    try:
        subprocess.run([exe, 'x', '-y', f'-o{tmp}', path], check = True, capture_output = True)
        for dp, _, fns in os.walk(tmp):
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), tmp).replace(os.sep, '/'); k = _key(rel)
                if k in keep and keep[k] not in out:
                    with open(os.path.join(dp, fn), 'rb') as h: out[keep[k]] = h.read()
    finally: shutil.rmtree(tmp, ignore_errors = True)
    return out


def parse_file(raw):
    """按表头名解析；表头缺必要列报错；只有表头没有数据返回 None"""
    df = pd.read_csv(io.BytesIO(raw), na_values = [''])
    df.columns = [str(c).strip() for c in df.columns]
    if not set(COLS) <= set(df.columns): raise ValueError(f'表头缺少必要列：{list(df.columns)}')
    return df if len(df) else None


def check(df):
    """返回剔除原因，通过返回 None"""
    if df is None: return 'empty_file'
    if len(df) != 240: return 'bad_row_count'
    x = df[COLS]
    for c in COLS:
        if not pd.api.types.is_numeric_dtype(x[c]): return 'non_numeric'
    if x.isna().any().any(): return 'has_nan'
    if 'time' in df.columns and df.time.astype(str).str[:5].tolist() not in (TIME_STR, START_STR): return 'time_mismatch'
    if (x[['open', 'high', 'low', 'close']] <= 0).any().any() or (x.volume < 0).any() or (x.amount < 0).any(): return 'bad_value'
    if ((x.high < x.low - 1e-9) | (x.open > x.high + 1e-9) | (x.open < x.low - 1e-9) | (x.close > x.high + 1e-9) | (x.close < x.low - 1e-9)).any(): return 'ohlc_inconsistent'
    return None


def bar_groups(df):
    """每行属于第几根 5 分钟线（0..47）。默认按位置每 5 行一组（标签是分钟结束时刻）；
    时间列是分钟起点的变体（20251201-03，经与 BaoStock 累计成交量核对：09:35 含 4 行、11:30 含 119 行、13:05 含 5 行）按时间分组：
    行的结束时刻 = 起点 + 1 分钟，向上取整到 5 分钟，晚于 15:00 的（收盘集合竞价）并入 15:00 那根。"""
    if 'time' in df.columns and df.time.astype(str).str[:5].tolist() == START_STR:
        m = np.array([int(t[:2]) * 60 + int(t[3:5]) for t in START_STR]) + 1; end = np.minimum(-(-m // 5) * 5, 900)
        return np.unique(end, return_inverse = True)[1]
    return np.arange(240) // 5


def to_5m(df):
    """240 行 1 分钟 → 48 根 5 分钟（结束时刻 09:35 .. 15:00，中午与收盘各自对齐、不跨午休）。价格取两位小数（来源里有单精度浮点噪声）"""
    g = bar_groups(df); a = df[COLS].to_numpy(dtype = float); out = np.empty((48, 6))
    for k in range(48):
        x = a[g == k]; out[k] = (x[0, 0], x[:, 1].max(), x[:, 2].min(), x[-1, 3], x[:, 5].sum(), x[:, 4].sum())
    out[:, :4] = np.round(out[:, :4], 2); return out


# 单日处理（可在子进程运行）---------------------------------------------------------
def process_day(task):
    """task: (日期, 压缩包路径, {证券: (日线成交量, 日线成交额, 是否交易)}, 7z 路径)。返回 (5 分钟数据, 审计问题, 统计)"""
    day, path, wanted, sevenzip = task; ds = str(day)
    keep = {(i.split('.')[1].lower(), i.split('.')[0]): i for i in wanted}
    issues = []; frames = {}
    files = read_members(path, keep, sevenzip)
    for inst, (dv, da, trading) in wanted.items():
        if not trading:
            if inst in files:
                try:
                    df = parse_file(files[inst])
                    if df is not None and float(df.volume.fillna(0).sum()) > 0: issues.append(issue('warn', 'bars_on_suspended_day', ds, inst, '日线标为停牌但分钟线有成交'))
                except Exception: pass
            continue
        if inst not in files: issues.append(issue('warn', 'missing_file', ds, inst, '日线有交易但压缩包里没有该证券')); continue
        try: df = parse_file(files[inst]); why = check(df)
        except Exception as exc: df, why = None, 'unreadable'; issues.append(issue('warn', 'unreadable_file', ds, inst, str(exc)[:120])); continue
        if why: issues.append(issue('warn', f'excluded_{why}', ds, inst, '整条剔除')); continue
        frames[inst] = df
    def unit_scale(col, k):
        """整日该列与日线的比值中位数恒为 0.01：来源把单位缩小了 100 倍（20251201-03 的成交量是手、成交额是手 × 价格），整日乘 100"""
        r = [float(df[col].sum()) / wanted[i][k] for i, df in frames.items() if wanted[i][k] > 0]
        return 100.0 if len(r) >= 20 and 0.0099 <= float(np.median(r)) <= 0.0101 else 1.0
    sv, sa = unit_scale('volume', 0), unit_scale('amount', 1)
    if sv != 1 or sa != 1:
        issues.append(issue('warn', 'unit_rescaled', ds, None, '分钟' + '、'.join(n for n, k in (('成交量', sv), ('成交额', sa)) if k != 1) + '整日是日线的 0.01 倍，已乘 100'))
    rows = []
    for inst, df in frames.items():
        dv, da, _ = wanted[inst]; five = to_5m(df); five[:, 4] *= sv; five[:, 5] *= sa
        v, a = five[:, 4].sum(), five[:, 5].sum()
        if dv > 0 and abs(v - dv) > max(TOL * dv, VOL_FLOOR) or da > 0 and abs(a - da) > max(TOL * da, AMT_FLOOR):
            issues.append(issue('warn', 'excluded_daily_mismatch', ds, inst, f'成交量比 {v / dv if dv else float("nan"):.4f}，成交额比 {a / da if da else float("nan"):.4f}')); continue
        t = pd.Timestamp(day) + BAR_OFFSETS
        rows.append(pd.DataFrame({'bar_end': t, 'instrument': inst, 'open': five[:, 0], 'high': five[:, 1], 'low': five[:, 2], 'close': five[:, 3],
                                  'volume': five[:, 4].round().astype('int64'), 'amount': five[:, 5], 'source': SOURCE}))
    out = pd.concat(rows, ignore_index = True) if rows else pd.DataFrame(columns = ['bar_end', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount', 'source'])
    return out, issues, {'expected': sum(1 for v in wanted.values() if v[2]), 'kept': len(rows)}


# 导入 -------------------------------------------------------------------------
def _merge_month(store, part, new, run_days):
    old = store.load('bars_5m', parts = [part])
    if len(old): old = old[~pd.to_datetime(old.bar_end).dt.normalize().isin(run_days)]
    return store.write_partition('bars_5m', part, pd.concat([old, new], ignore_index = True) if len(old) else new)


def import_minute(root, source, start = None, end = None, top = 800, workers = 4, sevenzip = None, log = None):
    """导入 [start, end] 的分钟股票池 5 分钟线并发布。默认区间 = 各年名单的预热起点 .. 日线最后一天。
    校验不过的「证券 × 交易日」剔除并记入 data/minute_audits/<批次>.issues.csv；没有阻断项即发布，批次里同时带 minute_universe。"""
    say = log or (lambda *a: None); store = Store(root); start = pd.Timestamp(start).date() if start else None; end = pd.Timestamp(end).date() if end else None
    with operation_lock(root, DATA_WRITER):
        pub = store.published()
        if not pub['batch_id']: raise RuntimeError('还没有已发布的日线，无法确定分钟股票池')
        bars = store.load('bars_1d', columns = ['date', 'instrument', 'volume', 'amount', 'is_trading', 'board']); cal = store.load('calendar')
        bars['date'] = pd.to_datetime(bars.date).dt.date
        uni = build_universe(bars, top = top)
        if not len(uni): raise RuntimeError('日线历史不足一整年，无法生成分钟股票池')
        sessions = [d for d in pd.to_datetime(cal[cal.is_open].date).dt.date]
        pools = {int(y): set(g.instrument) for y, g in uni.groupby('year')}
        plan = plan_days(sessions, set(pools), start, end)
        if not plan: raise RuntimeError('区间内没有需要导入的交易日')
        daily = {d: g for d, g in bars[bars.date.isin(plan)].groupby('date')}
        issues, parts, kept, expected = [], {}, 0, 0
        months = sorted({(d.year, d.month) for d in plan})
        with ProcessPoolExecutor(workers) if workers > 1 else _Inline() as pool:
            for ym in months:
                days = [d for d in plan if (d.year, d.month) == ym]; tasks = []
                for d in days:
                    path = archive_path(source, d)
                    if path is None: issues.append(issue('warn', 'missing_day', str(d), None, '交易日历有该日但没有分钟线压缩包')); continue
                    want = set().union(*(pools[y] for y in plan[d])); g = daily.get(d)
                    if g is None: continue
                    g = g[g.instrument.isin(want)]
                    tasks.append((d, path, {r.instrument: (float(r.volume), float(r.amount), bool(r.is_trading)) for r in g.itertuples()}, sevenzip))
                frames = []
                for out, iss, st in pool.map(process_day, tasks):
                    if len(out): frames.append(out)
                    issues += iss; kept += st['kept']; expected += st['expected']
                if frames:
                    part = f'{ym[0]}{ym[1]:02d}'; parts.setdefault('bars_5m', {})[part] = _merge_month(store, part, pd.concat(frames, ignore_index = True), pd.to_datetime(days))
                    say(f'{part}: {parts["bars_5m"][part]["rows"]} 行')
        if 'bars_5m' not in parts: raise RuntimeError('没有任何证券通过校验，未写入')
        parts['minute_universe'] = {'all': store.write_partition('minute_universe', 'all', uni)}
        cur = pub['tables']
        if all(cur.get(t, {}).get(p, {}).get('file') == v['file'] for t, ps in parts.items() for p, v in ps.items()):
            return {'batch_id': pub['batch_id'], 'status': 'no_change', 'kept': kept, 'expected': expected}
        iss = pd.DataFrame(issues, columns = ['level', 'rule', 'date', 'instrument', 'detail'])
        bid = store.write_batch(parts, note = f'minute {min(plan)}..{max(plan)} pool_top={top} source={SOURCE}')
        d = Path(root) / 'minute_audits'; d.mkdir(parents = True, exist_ok = True); tmp = d / f'{bid}.issues.tmp'; iss.to_csv(tmp, index = False); os.replace(tmp, d / f'{bid}.issues.csv')
        counts = {f'{l}/{r}': int(n) for (l, r), n in iss.groupby(['level', 'rule']).size().items()} if len(iss) else {}
        _atomic_json(d / f'{bid}.json', {'batch_id': bid, 'source': SOURCE, 'range': [str(min(plan)), str(max(plan))], 'top': top, 'stock_days_expected': expected, 'stock_days_kept': kept, 'issues': counts})
        blocked = iss[iss.level == 'block'] if len(iss) else iss
        if len(blocked): store.reject(bid, f'{len(blocked)} 条阻断级问题'); status = 'rejected'
        else: store.publish(bid); status = 'published'
    return {'batch_id': bid, 'status': status, 'range': [str(min(plan)), str(max(plan))], 'days': len(plan), 'stock_days_expected': expected, 'stock_days_kept': kept,
            'rows': sum(v['rows'] for v in parts['bars_5m'].values()), 'issues': counts}


class _Inline:
    """workers = 1 时不开子进程（测试与调试）"""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, it): return map(fn, it)

