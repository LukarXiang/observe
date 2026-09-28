import hashlib
import json
from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root): self.root = Path(root)

    def write_partition(self, table, partition, frame):
        path = self.root / table / f"{partition}.parquet"; path.parent.mkdir(parents = True, exist_ok = True)
        if path.exists(): raise FileExistsError(path)
        frame.to_parquet(path, index = False); return path

    def read(self, table, filters = None):
        paths = sorted((self.root / table).glob("*.parquet"))
        if not paths: return pd.DataFrame()
        out = pd.concat((pd.read_parquet(path) for path in paths), ignore_index = True)
        for column, value in filters or []: out = out[out[column].isin(value if isinstance(value, (list, set, tuple)) else [value])]
        return out

    def fingerprint(self, paths):
        digest = hashlib.sha256()
        for path in sorted(map(Path, paths)): digest.update(path.read_bytes())
        return digest.hexdigest()

    def snapshot(self, snapshot_id, paths, target):
        manifest = {"snapshot_id": snapshot_id, "files": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]}
        target = Path(target); target.parent.mkdir(parents = True, exist_ok = True); target.write_text(json.dumps(manifest, ensure_ascii = False, indent = 2), encoding = "utf-8"); return manifest
