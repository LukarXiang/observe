import json
from datetime import UTC, datetime
from pathlib import Path


def append_request(root, source, upstream, endpoint, params, status, rows, error = None):
    path = Path(root) / "raw" / "requests.jsonl"; path.parent.mkdir(parents = True, exist_ok = True)
    record = {"source": source, "upstream": upstream, "endpoint": endpoint, "params": params, "requested_at": datetime.now(UTC).isoformat(), "status": status, "rows": rows, "error": error}
    with path.open("a", encoding = "utf-8") as handle: handle.write(json.dumps(record, ensure_ascii = False) + "\n")
