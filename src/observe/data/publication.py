import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishedState:
    batch_id: str
    tables: dict

    def as_dict(self):
        return {"batch_id": self.batch_id, "tables": self.tables}


class BatchPublisher:
    def __init__(self, root):
        self.root = Path(root); self.batch_dir = self.root / "batches"; self.state_path = self.root / "PUBLISHED.json"

    def write_batch(self, batch_id, tables):
        self.batch_dir.mkdir(parents = True, exist_ok = True)
        manifest = {"batch_id": batch_id, "tables": tables}
        path = self.batch_dir / f"{batch_id}.json"
        if path.exists(): raise FileExistsError(path)
        self._atomic_json(path, manifest)
        return path

    def publish(self, batch_id):
        manifest_path = self.batch_dir / f"{batch_id}.json"
        if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
        if manifest.get("batch_id") != batch_id or not manifest.get("tables"):
            raise ValueError("invalid batch manifest")
        self._atomic_json(self.state_path, manifest)
        return PublishedState(batch_id, manifest["tables"])

    def read(self):
        if not self.state_path.exists(): return None
        data = json.loads(self.state_path.read_text(encoding = "utf-8"))
        return PublishedState(data["batch_id"], data["tables"])

    @staticmethod
    def _atomic_json(path, data):
        path.parent.mkdir(parents = True, exist_ok = True)
        fd, name = tempfile.mkstemp(prefix = f".{path.name}.", dir = path.parent)
        try:
            with os.fdopen(fd, "w", encoding = "utf-8") as handle:
                json.dump(data, handle, ensure_ascii = False, indent = 2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name): os.unlink(name)
