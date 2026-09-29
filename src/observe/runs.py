"""实验目录（模块 18）：独占创建、不覆盖已有产物；运行状态先写 running，结束时提交终态；复现源目录只读。"""
import hashlib, json, secrets
from datetime import datetime
from pathlib import Path

from .data.store import _atomic_json

TERMINAL = ('success', 'success_limited', 'blocked', 'failed')


class RunDirError(ValueError): pass


def create_run_dir(runs_root, output = None, tag = ''):
    """显式 output 必须不存在；默认目录 = 时间 + 配置指纹 + 随机后缀，靠 mkdir 独占创建，冲突就换后缀"""
    if output is not None:
        out = Path(output)
        try: out.mkdir(parents = True, exist_ok = False)
        except FileExistsError: raise RunDirError(f'输出目录已存在，拒绝写入：{out}') from None
        return out
    runs_root = Path(runs_root); runs_root.mkdir(parents = True, exist_ok = True)
    for _ in range(50):
        out = runs_root / '-'.join(x for x in (f'{datetime.now():%Y%m%d-%H%M%S}', tag, secrets.token_hex(2)) if x)
        try: out.mkdir(); return out
        except FileExistsError: continue
    raise RunDirError(f'无法在 {runs_root} 下创建新的实验目录')


def ensure_outside(source, output):
    """复现输出不能等于源目录，也不能落在源目录内部；按解析后的真实路径判断（相对路径、链接都先展开）"""
    if output is None: return
    src, out = Path(source).resolve(), Path(output).resolve()
    if out == src or src in out.parents: raise RunDirError(f'复现输出 {out} 与源实验目录 {src} 重叠，源目录只读')


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value): _atomic_json(path, value)


def write_table(out, name, rows):
    """核心产物统一写成按行记录的 JSON，键排序、日期转字符串，便于逐值比较和哈希"""
    write_json(Path(out) / f'{name}.json', canonical(rows))


def canonical(x):
    return json.loads(json.dumps(x, ensure_ascii = False, sort_keys = True, default = str))


class RunStatus:
    """status.json：创建目录后立即写 running；每完成一个阶段记一次；结束写终态。失败目录保留已完成阶段与错误。"""
    def __init__(self, out, run_id, kind = 'run', **info):
        self.path = Path(out) / 'status.json'
        self.data = {'run_id': run_id, 'kind': kind, 'status': 'running', 'started_at': datetime.now().isoformat(timespec = 'seconds'), 'stages': {}, **info}
        self._write()

    def _write(self): write_json(self.path, self.data)

    def stage(self, name, state = 'done', **info):
        self.data['stages'][name] = {'state': state, **info}; self._write()

    def finish(self, status, **info):
        if status not in TERMINAL: raise ValueError(f'unknown run status {status}')
        self.data.update(status = status, finished_at = datetime.now().isoformat(timespec = 'seconds'), **info); self._write()
