"""跨进程互斥：命令行、Web 任务、独立脚本共用的操作系统文件锁（决策 19）。进程退出时由操作系统释放，拿不到立即失败。"""
from contextlib import contextmanager
import os, json
from pathlib import Path

from filelock import FileLock, Timeout

BAOSTOCK, DATA_WRITER = 'baostock', 'data-writer'


@contextmanager
def operation_lock(root, name, timeout = 0):
    path = Path(root) / 'locks' / f'{name}.lock'; path.parent.mkdir(parents = True, exist_ok = True)
    owner = path.with_suffix('.owner')
    token = json.dumps({'pid': os.getpid()})
    try:
        fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w', encoding = 'utf-8') as h: h.write(token)
    except FileExistsError:
        try: pid = json.loads(owner.read_text(encoding = 'utf-8')).get('pid')
        except (OSError, ValueError, TypeError): pid = None
        alive = False
        if pid:
            try: os.kill(int(pid), 0); alive = True
            except OSError: alive = False
        if not alive:
            owner.unlink(missing_ok = True)
            fd = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding = 'utf-8') as h: h.write(token)
        else: raise RuntimeError(f'{name} 锁被占用：{path}')
    lock = FileLock(str(path), timeout = timeout)
    try: lock.acquire()
    except Timeout as exc:
        owner.unlink(missing_ok = True); raise RuntimeError(f'{name} 锁被占用：{path}') from exc
    try: yield lock
    finally:
        lock.release(); owner.unlink(missing_ok = True)
