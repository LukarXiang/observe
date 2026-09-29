"""实验目录（模块 18）：独占创建、不覆盖已有产物；运行状态先写 running，结束时提交终态；复现源目录只读；核心产物逐值比较。

状态（文件、函数返回值、命令退出码、队列任务共用一套定义）：
  success          程序跑完，没有证据限制
  success_limited  程序跑完，但有数据或执行上的证据限制（limitations 非空）
  blocked          输入或账本遇到无法正确处理的情况；有限制也不会改成 success_limited
  mismatch         复现重跑的核心产物与源实验不一致
  failed           程序出错（异常）
"""
import hashlib, importlib.metadata, json, math, platform, secrets, subprocess
from datetime import datetime
from pathlib import Path

from .data.store import _atomic_json

TERMINAL = ('success', 'success_limited', 'blocked', 'mismatch', 'failed')
EXIT_CODES = {'success': 0, 'success_limited': 0, 'failed': 1, 'mismatch': 2, 'blocked': 3}
REPO = Path(__file__).resolve().parents[2]


class RunDirError(ValueError): pass


def environment():
    """代码与依赖版本：复现时只检测并报告差异，不自动切换工作区"""
    def git(*args):
        try: return subprocess.run(['git', *args], cwd = REPO, capture_output = True, text = True, timeout = 10).stdout.strip()
        except (OSError, subprocess.SubprocessError): return None
    lock = REPO / 'uv.lock'; dirty = git('status', '--porcelain', '--untracked-files=no')
    packages = {}
    for p in ('pandas', 'numpy', 'pyarrow', 'pydantic'):
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    return {'git_commit': git('rev-parse', 'HEAD') or None, 'git_dirty': None if dirty is None else bool(dirty),
            'lock_sha256': file_sha(lock) if lock.exists() else None, 'python': platform.python_version(), 'packages': packages}


def drift(recorded, current):
    return {k: {'recorded': recorded.get(k), 'current': current.get(k)} for k in sorted(set(recorded) | set(current)) if recorded.get(k) != current.get(k)}


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


# 复现比较 ---------------------------------------------------------------------------------------------
KEYS = {'scores': ('decision_date', 'instrument'), 'orders': ('exec_date', 'order_id'), 'fills': ('fill_id',), 'cash_events': None,
        'equity': ('date',), 'positions_daily': ('date', 'instrument'), 'receivables': ('date', 'pay_date'), 'limitations': None}
STATUS_FIELDS = ('status', 'blocked', 'assumptions', 'issues', 'limitations', 'rules_used_unverified', 'summary')


def _same(a, b, abs_tol, rel_tol):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)): return a == b
    if math.isnan(a) or math.isnan(b): return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= abs_tol + rel_tol * abs(b)


def _rows(table, rows):
    key = KEYS.get(table)
    if key is None: return {(k,): r for k, r in enumerate(rows)}, ('row',)
    return {tuple(r.get(k) for k in key): r for r in rows}, key


def compare_tables(expected, actual, abs_tol = 1e-9, rel_tol = 0.0, limit = 200):
    """expected / actual: {表名: 行列表或字典}。按稳定主键对齐逐字段比较；返回 (差异明细, 每表摘要)"""
    diffs, summary = [], {}
    for table in sorted(set(expected) | set(actual)):
        e, a = expected.get(table), actual.get(table); n = 0
        if isinstance(e, dict) or isinstance(a, dict):
            e, a = e or {}, a or {}
            for f in sorted(set(e) | set(a)):
                if not _same(e.get(f), a.get(f), abs_tol, rel_tol): n += 1; diffs.append({'table': table, 'key': {}, 'field': f, 'expected': e.get(f), 'actual': a.get(f)})
            summary[table] = {'differences': n}; continue
        if e is None and a is None: summary[table] = {'differences': 0}; continue
        if e is None or a is None:
            diffs.append({'table': table, 'key': {}, 'field': None, 'kind': 'missing_table', 'expected': e is not None, 'actual': a is not None})
            summary[table] = {'differences': 1}; continue
        (er, key), (ar, _) = _rows(table, e), _rows(table, a)
        for k in sorted(set(er) | set(ar), key = lambda x: tuple(map(str, x))):
            kd = dict(zip(key, k))
            if k not in ar or k not in er:
                n += 1; diffs.append({'table': table, 'key': kd, 'field': None, 'kind': 'only_in_expected' if k not in ar else 'only_in_actual'}); continue
            x, y = er[k], ar[k]
            for f in sorted(set(x) | set(y)):
                if not _same(x.get(f), y.get(f), abs_tol, rel_tol): n += 1; diffs.append({'table': table, 'key': kd, 'field': f, 'expected': x.get(f), 'actual': y.get(f)})
        summary[table] = {'rows_expected': len(e), 'rows_actual': len(a), 'differences': n}
    return diffs[:limit], summary


def compare_frames(expected, actual, keys, abs_tol = 1e-9, rel_tol = 0.0, limit = 20):
    """大表按主键对齐逐列比较（数值按容差，其他按相等，缺失与缺失视为相同）。返回 (差异样例, 摘要)"""
    import numpy as np, pandas as pd
    keys = list(keys); m = expected.merge(actual, on = keys, how = 'outer', suffixes = ('__e', '__a'), indicator = True)
    summary = {'rows_expected': len(expected), 'rows_actual': len(actual), 'only_in_expected': int((m._merge == 'left_only').sum()),
               'only_in_actual': int((m._merge == 'right_only').sum()), 'columns': {}}
    diffs = [{'key': canonical(dict(zip(keys, r[:-1]))), 'kind': 'only_in_expected' if r[-1] == 'left_only' else 'only_in_actual'}
             for r in m.loc[m._merge != 'both', keys + ['_merge']].head(limit).itertuples(index = False)]
    both = m[m._merge == 'both']
    for c in [c for c in expected.columns if c not in keys]:
        if f'{c}__a' not in both: summary['columns'][c] = -1; diffs.append({'field': c, 'kind': 'missing_column'}); continue
        x, y = both[f'{c}__e'], both[f'{c}__a']
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float); same = np.isclose(xv, yv, atol = abs_tol, rtol = rel_tol) | (np.isnan(xv) & np.isnan(yv))
        else: same = ((x == y) | (x.isna() & y.isna())).to_numpy(bool)
        bad = np.flatnonzero(~same)
        if len(bad): summary['columns'][c] = int(len(bad))
        for k in bad[:max(0, limit - len(diffs))]:
            r = both.iloc[k]; diffs.append({'key': canonical({n: r[n] for n in keys}), 'field': c, 'expected': canonical(r[f'{c}__e']), 'actual': canonical(r[f'{c}__a'])})
    summary['differences'] = summary['only_in_expected'] + summary['only_in_actual'] + sum(abs(v) for v in summary['columns'].values())
    return diffs, summary
