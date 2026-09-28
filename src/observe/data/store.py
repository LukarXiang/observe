"""分区存储、批次发布、快照与清理（模块 10、决策 19）。

目录：std/<表>/<分区>__<指纹8位>.parquet；batches/<批次>.json；PUBLISHED.json；snapshots/<快照>.json；pins/<任务>.json
分区写入后不改；同内容同指纹同文件。研究只读已发布状态或快照。"""
import hashlib, json, os, secrets, tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

KEYS = {'calendar': ['date'], 'instruments': ['instrument'], 'bars_1d': ['date', 'instrument'], 'adj_factors': ['instrument', 'ex_date'],
        'corp_actions': ['instrument', 'ex_date'], 'shares': ['instrument', 'date'], 'index_1d': ['date', 'index'], 'bars_5m': ['bar_end', 'instrument']}


def _now(): return datetime.now().strftime('%Y%m%d-%H%M%S')


def _atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents = True, exist_ok = True)
    fd, tmp = tempfile.mkstemp(prefix = f'.{path.name}.', dir = path.parent)
    try:
        with os.fdopen(fd, 'w', encoding = 'utf-8') as h: json.dump(data, h, ensure_ascii = False, indent = 1, default = str); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _read_json(path): return json.loads(Path(path).read_text(encoding = 'utf-8'))


def fingerprint(df):
    h = hashlib.sha256('|'.join(map(str, df.columns)).encode()); h.update(pd.util.hash_pandas_object(df, index = False).to_numpy().tobytes()); return h.hexdigest()


class Store:
    def __init__(self, root): self.root = Path(root)

    # 分区 ---------------------------------------------------------------------
    def write_partition(self, table, part, df):
        """按主键去重排序后写入；返回 {'file','sha','rows'}。内容相同则复用已有文件"""
        key = KEYS[table]; df = df.drop_duplicates(key, keep = 'last').sort_values(key).reset_index(drop = True)
        if df[key].isna().any().any(): raise ValueError(f'{table}/{part}: 主键含空值')
        sha = fingerprint(df); rel = f'std/{table}/{part}__{sha[:8]}.parquet'; path = self.root / rel
        if not path.exists():
            path.parent.mkdir(parents = True, exist_ok = True); tmp = path.with_suffix('.tmp'); df.to_parquet(tmp, index = False); os.replace(tmp, path)
        return {'file': rel, 'sha': sha, 'rows': len(df)}

    # 发布 ---------------------------------------------------------------------
    @property
    def published_path(self): return self.root / 'PUBLISHED.json'

    def published(self): return _read_json(self.published_path) if self.published_path.exists() else {'batch_id': None, 'tables': {}}

    def write_batch(self, parts, note = ''):
        """parts: {表: {分区: 写入结果}}。批次只列本次变化的分区，base 记录它基于哪个已发布批次"""
        bid = f'{_now()}-{secrets.token_hex(2)}'; m = {'batch_id': bid, 'base': self.published()['batch_id'], 'status': 'pending', 'note': note, 'tables': parts}
        _atomic_json(self.root / 'batches' / f'{bid}.json', m); return bid

    def publish(self, bid):
        """把批次合并进已发布状态并原子替换 PUBLISHED.json；调用方须持有 data-writer 锁"""
        path = self.root / 'batches' / f'{bid}.json'; m = _read_json(path); cur = self.published()
        if m['status'] != 'pending': raise ValueError(f'批次 {bid} 状态为 {m["status"]}，不能发布')
        if m['base'] != cur['batch_id']: raise RuntimeError(f'批次 {bid} 基于 {m["base"]}，但当前已发布 {cur["batch_id"]}：请重新生成批次')
        tables = {t: dict(v) for t, v in cur['tables'].items()}
        for t, parts in m['tables'].items(): tables.setdefault(t, {}).update(parts)
        _atomic_json(self.published_path, {'batch_id': bid, 'published_at': _now(), 'tables': tables})
        m['status'] = 'published'; _atomic_json(path, m); return bid

    def reject(self, bid, reason):
        path = self.root / 'batches' / f'{bid}.json'; m = _read_json(path); m.update(status = 'rejected', reason = reason); _atomic_json(path, m)

    # 快照 ---------------------------------------------------------------------
    def snapshot(self, note = ''):
        cur = self.published()
        if not cur['batch_id']: raise RuntimeError('还没有已发布的数据')
        sid = f'{_now()}-{secrets.token_hex(2)}'; _atomic_json(self.root / 'snapshots' / f'{sid}.json', {**cur, 'snapshot_id': sid, 'note': note}); return sid

    def state(self, snapshot = None): return _read_json(self.root / 'snapshots' / f'{snapshot}.json') if snapshot else self.published()

    def load(self, table, snapshot = None, columns = None, parts = None):
        entries = self.state(snapshot)['tables'].get(table, {})
        files = [self.root / v['file'] for k, v in sorted(entries.items()) if parts is None or k in parts]
        if not files: return pd.DataFrame(columns = columns)
        return pd.concat([pd.read_parquet(f, columns = columns) for f in files], ignore_index = True)

    # 清理 ---------------------------------------------------------------------
    def pin(self, job_id, snapshot = None):
        """运行中的任务固定其输入；结束时 unpin"""
        _atomic_json(self.root / 'pins' / f'{job_id}.json', self.state(snapshot)); return job_id

    def unpin(self, job_id): (self.root / 'pins' / f'{job_id}.json').unlink(missing_ok = True)

    def protected(self):
        refs = [self.published()] + [_read_json(p) for d in ('snapshots', 'pins') for p in (self.root / d).glob('*.json')]
        refs += [m for m in (_read_json(p) for p in (self.root / 'batches').glob('*.json')) if m['status'] == 'pending']
        return {v['file'] for r in refs for parts in r['tables'].values() for v in parts.values()}

    def gc(self, apply = False):
        """默认只列出；apply = True 才删除。保护：已发布、全部快照、运行中任务固定的输入、待发布批次"""
        keep = self.protected(); files = sorted(p.relative_to(self.root).as_posix() for p in (self.root / 'std').rglob('*.parquet'))
        drop = [f for f in files if f not in keep]
        if apply:
            for f in drop: (self.root / f).unlink()
        return drop
