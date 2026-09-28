from contextlib import contextmanager
from pathlib import Path

try:
    from filelock import FileLock, Timeout
except ImportError:  # keeps offline development usable before uv sync
    FileLock = None; Timeout = TimeoutError


@contextmanager
def operation_lock(root, name, timeout = 0):
    path = Path(root) / "locks" / f"{name}.lock"; path.parent.mkdir(parents = True, exist_ok = True)
    if FileLock is not None:
        lock = FileLock(str(path))
        try:
            with lock.acquire(timeout = timeout): yield lock
        except Timeout as exc:
            raise RuntimeError(f"lock occupied: {path}") from exc
        return
    marker = path.with_suffix(path.suffix + ".held")
    try:
        marker.touch(exist_ok = False); yield marker
    except FileExistsError as exc:
        raise RuntimeError(f"lock occupied: {path}") from exc
    finally:
        marker.unlink(missing_ok = True)
