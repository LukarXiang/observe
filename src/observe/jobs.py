"""任务队列（模块 19）：SQLite 记录，单个工作进程顺序执行，每个任务在独立子进程中运行、日志单独落盘。
写数据的互斥不靠「先查没有任务」，而靠任务函数内部的文件锁（决策 19）。"""
import json, os, secrets, sqlite3, subprocess, sys, time
from datetime import datetime
from pathlib import Path

# KINDS 保留数据库兼容性；SUPPORTED 才是界面和新提交允许执行的能力。
KINDS = ('data_update', 'data_audit', 'snapshot', 'gc', 'factor_eval', 'run_experiment', 'backtest_variant', 'reproduce')
SUPPORTED = frozenset(('data_update', 'data_audit', 'snapshot', 'gc', 'run_experiment', 'reproduce'))
STATUS = ('queued', 'running', 'success', 'partial', 'failed', 'cancelled', 'interrupted')
SCHEMA = '''create table if not exists jobs (job_id text primary key, kind text not null, params text not null, status text not null,
            created_at text, started_at text, finished_at text, pid integer, result text, error text, retry_of text, log_path text)'''


def _now(): return datetime.now().isoformat(timespec = 'seconds')


def alive(pid):
    """Windows 上 os.kill(pid, 0) 会结束进程，不能用；这里查询进程退出码（259 = 仍在运行）"""
    if not pid: return False
    if os.name != 'nt':
        try: os.kill(pid, 0); return True
        except OSError: return False
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
        """原子地领取最早的排队任务；多个工作进程也不会领到同一个"""
        with self._db() as c:
            c.execute('begin immediate')
            r = c.execute("select job_id from jobs where status = 'queued' order by created_at, job_id limit 1").fetchone()
            if not r: c.execute('commit'); return None
            c.execute("update jobs set status = 'running', started_at = ? where job_id = ?", (_now(), r['job_id'])); c.execute('commit')
        return r['job_id']

    def finish(self, jid, status, result = None, error = None):
        with self._db() as c:
            c.execute('update jobs set status = ?, finished_at = ?, result = ?, error = ? where job_id = ?', (status, _now(), json.dumps(result, ensure_ascii = False, default = str) if result is not None else None, error, jid))

    def cancel(self, jid):
        with self._db() as c: return c.execute("update jobs set status = 'cancelled', finished_at = ? where job_id = ? and status = 'queued'", (_now(), jid)).rowcount == 1

    def retry(self, jid):
        j = self.get(jid)
        if j['status'] not in ('failed', 'interrupted', 'partial', 'cancelled'): raise ValueError(f'任务 {jid} 状态为 {j["status"]}，不能重试')
        return self.submit(j['kind'], json.loads(j['params']), retry_of = jid)

    def recover(self):
        """服务启动时：状态为 running 但进程已不存在的任务改为 interrupted（数据任务可从已提交进度续跑）"""
        with self._db() as c:
            dead = [r['job_id'] for r in c.execute("select job_id, pid from jobs where status = 'running'") if not alive(r['pid'])]
            for j in dead: c.execute("update jobs set status = 'interrupted', finished_at = ? where job_id = ?", (_now(), j))
        return dead

    def run_next(self, wait = True):
        """领取一个任务，在子进程中执行 `python -m observe.cli jobs exec <id>`；返回任务编号"""
        jid = self.claim()
        if jid is None: return None
        log = Path(self.get(jid)['log_path']); log.parent.mkdir(parents = True, exist_ok = True)
        try:
            with log.open('a', encoding = 'utf-8') as h:
                env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
                p = subprocess.Popen([sys.executable, '-m', 'observe.cli', '--root', str(self.root), 'jobs', 'exec', jid], stdout = h, stderr = subprocess.STDOUT, env = env)
        except Exception as exc:
            self.finish(jid, 'failed', error = f'{type(exc).__name__}: {exc}')
            return jid
        with self._db() as c: c.execute('update jobs set pid = ? where job_id = ?', (p.pid, jid))
        if wait:
            code = p.wait(); current = self.get(jid)
            if current and current['status'] == 'running':
                self.finish(jid, 'failed' if code else 'interrupted', error = f'子进程退出码 {code} 且未写入完成状态')
        return jid

    def worker(self, idle = 2.0, once = False):
        self.recover()
        while True:
            if self.run_next() is None:
                if once: return
                time.sleep(idle)
