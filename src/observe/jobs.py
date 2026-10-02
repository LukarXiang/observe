"""任务队列（模块 19）：SQLite 记录，单个工作进程顺序执行，每个任务在独立子进程中运行、日志单独落盘。
写数据的互斥不靠「先查没有任务」，而靠任务函数内部的文件锁（决策 19）。"""
import json, os, secrets, sqlite3, subprocess, sys, time
from contextlib import contextmanager
from threading import Thread
from filelock import FileLock, Timeout
from datetime import datetime
from pathlib import Path

# KINDS 保留数据库兼容性；SUPPORTED 才是界面和新提交允许执行的能力。
KINDS = ('data_update', 'data_audit', 'data_index', 'snapshot', 'gc', 'factor_eval', 'run_experiment', 'backtest_variant', 'reproduce', 'research', 'minute_import', 'paired', 'experiment')
SUPPORTED = frozenset(KINDS)
STATUS = ('queued', 'running', 'success', 'success_limited', 'partial', 'blocked', 'mismatch', 'failed', 'cancelled', 'interrupted')   # 回放与复现沿用 runs.py 的运行状态
SCHEMA = '''create table if not exists jobs (job_id text primary key, kind text not null, params text not null, status text not null,
            created_at text, started_at text, finished_at text, pid integer, result text, error text, retry_of text, log_path text)'''


def _now(): return datetime.now().isoformat(timespec = 'seconds')


def alive(pid):
    """Windows 上 os.kill(pid, 0) 会结束进程，不能用；这里查询进程退出码（259 = 仍在运行）"""
    if not pid: return False
    if os.name != 'nt':
        try: os.kill(pid, 0); return True
        except PermissionError: return True
        except ProcessLookupError: return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))               # PROCESS_QUERY_LIMITED_INFORMATION
    if not h: return False
    code = ctypes.c_ulong(); ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
    return bool(ok) and code.value == 259


class Jobs:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents = True, exist_ok = True); self.path = self.root / 'jobs.sqlite'
        with self._db() as c: c.execute(SCHEMA)

    def _db(self):
        c = sqlite3.connect(self.path, timeout = 10, isolation_level = None); c.row_factory = sqlite3.Row; return c

    def submit(self, kind, params = None, retry_of = None):
        if kind not in KINDS: raise ValueError(f'未知任务种类 {kind}')
        params = dict(params or {})
        if kind == 'data_index':
            from .data.indices import IndexUpdateConfig
            params = IndexUpdateConfig.model_validate(params).model_dump(mode = 'json')
        if kind == 'experiment':
            from .experiments import experiment_params
            p = experiment_params(params); params = {**p['config'].model_dump(mode = 'json'), 'output': p['output']}
        if kind == 'factor_eval':
            from .experiments import FactorEvalConfig
            output = params.pop('output', None); params = {**FactorEvalConfig.model_validate(params).model_dump(mode = 'json'), 'output': output}
        if kind == 'backtest_variant':
            from .experiments import VariantConfig
            if set(params) - {'parent', 'config', 'output'} or not params.get('parent'): raise ValueError('组合变体任务需要 parent，仅接受 parent / config / output')
            params['config'] = VariantConfig.model_validate(params.get('config') or {}).model_dump(mode = 'json', exclude_unset = True)
        jid = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"; log = self.root / 'jobs' / f'{jid}.log'
        with self._db() as c:
            c.execute('insert into jobs (job_id, kind, params, status, created_at, retry_of, log_path) values (?,?,?,?,?,?,?)',
                      (jid, kind, json.dumps(params or {}, ensure_ascii = False), 'queued', _now(), retry_of, str(log)))
            if kind not in SUPPORTED:
                c.execute('update jobs set status = ?, finished_at = ?, error = ? where job_id = ?', ('failed', _now(), f'任务种类 {kind} 尚未实现', jid))
        return jid

    def get(self, jid):
        with self._db() as c: r = c.execute('select * from jobs where job_id = ?', (jid,)).fetchone()
        return dict(r) if r else None

    def list(self, limit = 50):
        with self._db() as c: return [dict(r) for r in c.execute('select * from jobs order by created_at desc, job_id desc limit ?', (limit,))]

    def claim(self):
        """原子领取唯一运行席位；PID 先记录领取者，启动事务内交接给子进程"""
        with self._db() as c:
            c.execute('begin immediate')
            if c.execute("select 1 from jobs where status = 'running' limit 1").fetchone():
                c.execute('commit'); return None
            r = c.execute("select job_id from jobs where status = 'queued' order by created_at, job_id limit 1").fetchone()
            if not r: c.execute('commit'); return None
            c.execute("update jobs set status = 'running', started_at = ?, pid = ? where job_id = ?", (_now(), os.getpid(), r['job_id'])); c.execute('commit')
        return r['job_id']

    def finish(self, jid, status, result = None, error = None, *, pid = None):
        if status not in STATUS or status in ('queued', 'running', 'cancelled'): raise ValueError(f'非法完成状态 {status}')
        with self._db() as c:
            return c.execute("update jobs set status = ?, finished_at = ?, result = ?, error = ? where job_id = ? and status = 'running' and pid = ?",
                             (status, _now(), json.dumps(result, ensure_ascii = False, default = str) if result is not None else None, error, jid, os.getpid() if pid is None else pid)).rowcount == 1

    def cancel(self, jid):
        with self._db() as c: return c.execute("update jobs set status = 'cancelled', finished_at = ? where job_id = ? and status = 'queued'", (_now(), jid)).rowcount == 1

    def retry(self, jid):
        j = self.get(jid)
        if j is None: raise ValueError(f'任务不存在：{jid}')
        if j['status'] not in ('failed', 'interrupted', 'partial', 'blocked', 'mismatch', 'cancelled'): raise ValueError(f'任务 {jid} 状态为 {j["status"]}，不能重试')
        return self.submit(j['kind'], json.loads(j['params']), retry_of = jid)

    def _execution_lock(self):
        path = self.root / 'locks' / 'job-execution.lock'
        path.parent.mkdir(parents = True, exist_ok = True)
        return FileLock(path)

    @contextmanager
    def execution(self, jid):
        """仅启动事务指定的子进程可执行；锁覆盖整个任务及完成状态写入。"""
        with self._execution_lock():
            # 写事务屏障：普通 SELECT 在 WAL/回滚日志下可能看到交接前的 PID。
            with self._db() as c:
                c.execute('begin immediate')
                row = c.execute('select * from jobs where job_id = ?', (jid,)).fetchone()
                job = dict(row) if row else None
                c.execute('commit')
            if job is None or job['status'] != 'running' or job['pid'] != os.getpid():
                raise ValueError(f'任务 {jid} 未授权当前进程执行')
            yield job

    def recover(self):
        """不触碰仍在运行的子进程；失去执行者的任务只标记中断，不自动重跑。"""
        lock = self._execution_lock()
        try: lock.acquire(timeout = 0)
        except Timeout: return []
        try:
            with self._db() as c:
                c.execute('begin immediate')
                dead = [r['job_id'] for r in c.execute("select job_id, pid from jobs where status = 'running'") if not alive(r['pid'])]
                for j in dead:
                    c.execute("update jobs set status = 'interrupted', finished_at = ?, error = ? where job_id = ? and status = 'running'",
                              (_now(), '任务执行进程已退出；可显式重试', j))
                c.execute('commit')
            return dead
        finally: lock.release()

    def _watch(self, jid, process):
        code = process.wait()
        self.finish(jid, 'failed' if code else 'interrupted',
                    error = f'子进程退出码 {code} 且未写入完成状态', pid = process.pid)

    def run_next(self, wait = True):
        """领取后在事务内启动并交接 PID；恢复及子进程校验只能看到交接前或交接后。"""
        self.recover()
        jid = self.claim()
        if jid is None: return None
        try:
            with self._db() as c:
                c.execute('begin immediate')
                job = c.execute("select * from jobs where job_id = ? and status = 'running' and pid = ?", (jid, os.getpid())).fetchone()
                if job is None: c.execute('commit'); return jid
                log = Path(job['log_path']); log.parent.mkdir(parents = True, exist_ok = True)
                with log.open('a', encoding = 'utf-8') as h:
                    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'}
                    p = subprocess.Popen([sys.executable, '-m', 'observe.cli', '--root', str(self.root), 'jobs', 'exec', jid], stdout = h, stderr = subprocess.STDOUT, env = env)
                c.execute("update jobs set pid = ? where job_id = ?", (p.pid, jid))
                c.execute('commit')
        except Exception as exc:
            self.finish(jid, 'failed', error = f'{type(exc).__name__}: {exc}')
            return jid
        if wait: self._watch(jid, p)
        else: Thread(target = self._watch, args = (jid, p), daemon = True).start()
        return jid

    def worker(self, idle = 2.0, once = False):
        """once 排空当前可执行队列；其他执行者占用席位时返回。"""
        while True:
            if self.run_next() is None:
                if once: return
                time.sleep(idle)
