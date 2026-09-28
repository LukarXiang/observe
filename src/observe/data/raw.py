"""原始记录留底：原样保存（字段全为字符串），每次请求写一行日志（模块 10）。"""
import hashlib, json, time
from datetime import datetime
from pathlib import Path
import pandas as pd


def save(root, upstream, dataset, key, df):
    payload = df.astype(str)
    digest = hashlib.sha256(pd.util.hash_pandas_object(payload, index = False).to_numpy().tobytes()).hexdigest()[:12]
    base = Path(root) / 'raw' / upstream / dataset / f'{key}.parquet'; base.parent.mkdir(parents = True, exist_ok = True)
    if not base.exists(): path = base
    else:
        old = pd.read_parquet(base).astype(str)
        old_digest = hashlib.sha256(pd.util.hash_pandas_object(old, index = False).to_numpy().tobytes()).hexdigest()[:12]
        if old_digest == digest: return base
        path = base.with_name(f'{key}__{digest}.parquet')
    if not path.exists(): payload.to_parquet(path, index = False)
    return path


def log(root, source, upstream, endpoint, params, status, rows = None, error = None, sec = None):
    path = Path(root) / 'raw' / 'requests.jsonl'; path.parent.mkdir(parents = True, exist_ok = True)
    rec = {'at': datetime.now().isoformat(timespec = 'seconds'), 'source': source, 'upstream': upstream, 'endpoint': endpoint, 'params': params, 'status': status, 'rows': rows, 'error': error, 'sec': sec}
    with path.open('a', encoding = 'utf-8') as h: h.write(json.dumps(rec, ensure_ascii = False, default = str) + '\n')


def retry(fn, tries = 3, waits = (2, 4, 8), sleep = time.sleep):
    """只对网络类错误重试；其余错误直接抛出"""
    for k in range(tries):
        try: return fn()
        except (ConnectionError, TimeoutError, OSError):
            if k == tries - 1: raise
            sleep(waits[min(k, len(waits) - 1)])
