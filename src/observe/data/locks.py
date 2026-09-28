"""跨进程互斥：命令行、Web 任务、独立脚本共用的操作系统文件锁（决策 19）。进程退出时由操作系统释放，拿不到立即失败。"""
from contextlib import contextmanager
import json, os, tempfile
from pathlib import Path

from filelock import FileLock, Timeout

BAOSTOCK, DATA_WRITER = 'baostock', 'data-writer'


@contextmanager
def operation_lock(root, name, timeout = 0):
    path = Path(root) / 'locks' / f'{name}.lock'; path.parent.mkdir(parents = True, exist_ok = True)
    owner = path.with_suffix('.owner')
    token = json.dumps({'pid': os.getpid()})
    lock = FileLock(str(path), timeout = timeout)
    try: lock.acquire()
    except Timeout as exc: raise RuntimeError(f'{name} 锁被占用：{path}') from exc
    try:
        # owner is diagnostic metadata only. It is written after the OS lock
        # has been acquired and never participates in ownership decisions.
        fd, tmp = tempfile.mkstemp(prefix = f'.{owner.name}.', dir = owner.parent)
        with os.fdopen(fd, 'w', encoding = 'utf-8') as h: h.write(token); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, owner)
        yield lock
    finally:
        owner.unlink(missing_ok = True); lock.release()
