"""数据源探测公共工具：计时、异常收集、DataFrame 画像、结果落盘（data/probe/<源>/）。"""
import json, time, pathlib, datetime as dt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'probe'


def profile(df, time_col = None, price_cols = ('open', 'high', 'low', 'close')):
    """行数、列、时间范围、空值、OHLC 零值与非正值计数、前两行样例"""
    if df is None: return {'rows': None}
    r = {'rows': len(df), 'cols': list(map(str, df.columns))}
    if not len(df): return r
    if time_col and time_col in df: s = df[time_col].astype(str); r.update(first = s.min(), last = s.max(), n_unique_time = int(s.nunique()))
    na = df.isna().sum(); r['na'] = {k: int(v) for k, v in na[na > 0].items()}
    px = [c for c in price_cols if c in df]
    if px:
        v = df[px].apply(pd.to_numeric, errors = 'coerce')
        r['px_nonpos'] = {c: int((v[c] <= 0).sum()) for c in px if (v[c] <= 0).any()}
        r['px_nan'] = {c: int(v[c].isna().sum()) for c in px if v[c].isna().any()}
    r['head'] = df.head(2).astype(str).to_dict('records')
    return r


class Probe:
    def __init__(self, name, pause = 0.3):
        self.name, self.pause, self.res, self.dir = name, pause, {}, OUT / name
        self.dir.mkdir(parents = True, exist_ok = True)
        self.res['_meta'] = {'started_at': dt.datetime.now().astimezone().isoformat(timespec = 'seconds')}

    def run(self, key, fn, *a, **kw):
        t0 = time.perf_counter()
        try: r = fn(*a, **kw); r = r if isinstance(r, dict) else {'value': r}; r = {'ok': True, **r}
        except Exception as e: r = {'ok': False, 'err': f'{type(e).__name__}: {e}'[:400]}
        r['sec'] = round(time.perf_counter() - t0, 2); self.res[key] = r
        brief = {k: v for k, v in r.items() if k not in ('head', 'cols')}
        print(f'[{self.name}] {key}: {json.dumps(brief, ensure_ascii = False, default = str)[:600]}', flush = True)
        time.sleep(self.pause); return r

    def save(self, key, df):
        if df is not None and len(df): df.astype(str).to_parquet(self.dir / f'{key}.parquet', index = False)

    def dump(self):
        self.res['_meta']['finished_at'] = dt.datetime.now().astimezone().isoformat(timespec = 'seconds')
        p = self.dir / 'result.json'; p.write_text(json.dumps(self.res, ensure_ascii = False, indent = 1, default = str), encoding = 'utf-8')
        print('saved', p)
