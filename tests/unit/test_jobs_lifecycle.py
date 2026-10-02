"""队列生命周期：真实进程互斥、执行身份、恢复及终态保护。"""
import os
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.jobs import Jobs


def until(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(0.02)
    raise AssertionError('等待子进程状态超时')


HOLD = '''
from observe.jobs import Jobs
from pathlib import Path
import sys, time
q = Jobs(sys.argv[1])
jid = q.claim()
with q.execution(jid):
    Path(sys.argv[2]).write_text(jid)
    while not Path(sys.argv[3]).exists(): time.sleep(0.02)
    q.finish(jid, 'success')
'''


def test_processes_share_one_slot_and_recover_only_dead_owner(tmp_path):
    q = Jobs(tmp_path); ids = {q.submit('snapshot') for _ in range(2)}
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    p = subprocess.Popen([sys.executable, '-c', HOLD, str(tmp_path), str(ready), str(release)])
    try:
        until(ready.exists); jid = ready.read_text()
        assert jid in ids
        assert q.recover() == []
        assert q.claim() is None and q.run_next() is None
        assert not q.finish(jid, 'failed', error = '非拥有者不得完成')
        p.terminate(); p.wait(timeout = 10)
        assert q.recover() == [jid]
        assert q.get(jid)['status'] == 'interrupted'
        assert q.claim() in ids - {jid}
    finally:
        release.touch()
        if p.poll() is None: p.terminate()
        p.wait(timeout = 10)


def test_claim_before_spawn_is_not_interrupted_and_terminal_is_immutable(tmp_path):
    q = Jobs(tmp_path); jid = q.submit('snapshot')
    assert q.claim() == jid and q.get(jid)['pid'] == os.getpid()
    assert q.recover() == []
    assert q.finish(jid, 'success', {'evidence': 1})
    before = q.get(jid)
    assert not q.finish(jid, 'failed', error = '迟到的监控回写')
    assert q.get(jid) == before
    with pytest.raises(ValueError): q.finish(jid, 'running')


@pytest.mark.parametrize('state', ['queued', 'running', 'cancelled', 'success'])
def test_cli_exec_cannot_bypass_worker_or_reexecute(tmp_path, state):
    q = Jobs(tmp_path); jid = q.submit('snapshot')
    if state == 'cancelled': q.cancel(jid)
    elif state in ('running', 'success'):
        q.claim()
        if state == 'success': q.finish(jid, 'success')
    before = q.get(jid)
    p = subprocess.run([sys.executable, '-m', 'observe.cli', '--root', str(tmp_path), 'jobs', 'exec', jid], capture_output = True, text = True, timeout = 15)
    assert p.returncode != 0 and '未授权当前进程执行' in p.stderr
    assert q.get(jid) == before
    assert not (tmp_path / 'snapshots').exists()


def test_missing_retry_is_validation_error_in_api(tmp_path):
    with pytest.raises(ValueError, match = '任务不存在'): Jobs(tmp_path).retry('missing')
    response = TestClient(create_app(tmp_path)).post('/api/jobs/missing/retry')
    assert response.status_code == 400 and '任务不存在' in response.json()['detail']


def test_async_crash_is_reaped_without_restarting_worker(tmp_path, monkeypatch):
    q = Jobs(tmp_path); jid = q.submit('snapshot'); popen = subprocess.Popen
    child = []
    def crash(*args, **kwargs):
        p = popen([sys.executable, '-c', 'import os; os._exit(7)'], **kwargs)
        child.append(p); return p
    monkeypatch.setattr('observe.jobs.subprocess.Popen', crash)
    assert q.run_next(wait = False) == jid
    until(lambda: q.get(jid)['status'] == 'failed')
    assert '退出码 7' in q.get(jid)['error'] and child[0].returncode == 7


def test_spawn_failure_releases_slot(tmp_path, monkeypatch):
    q = Jobs(tmp_path); jid = q.submit('snapshot')
    def fail(*args, **kwargs): raise OSError('spawn failed')
    monkeypatch.setattr('observe.jobs.subprocess.Popen', fail)
    assert q.run_next() == jid
    assert q.get(jid)['status'] == 'failed'
    next_id = q.submit('snapshot'); assert q.claim() == next_id


def test_handoff_transaction_blocks_child_until_pid_is_committed(tmp_path):
    q = Jobs(tmp_path); jid = q.submit('snapshot'); q.claim()
    ready = tmp_path / 'executed'
    code = '''
from observe.jobs import Jobs
from pathlib import Path
import sys
q = Jobs(sys.argv[1])
with q.execution(sys.argv[2]):
    Path(sys.argv[3]).touch()
    q.finish(sys.argv[2], 'success')
'''
    with q._db() as c:
        c.execute('begin immediate')
        p = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), jid, str(ready)])
        c.execute('update jobs set pid = ? where job_id = ?', (p.pid, jid))
        c.execute('commit')
    assert p.wait(timeout = 15) == 0
    assert ready.exists() and q.get(jid)['status'] == 'success'


def test_recovery_cannot_overwrite_concurrent_completion(tmp_path):
    q = Jobs(tmp_path); jid = q.submit('snapshot'); q.claim()
    with q.execution(jid):
        found = []
        thread = threading.Thread(target = lambda: found.extend(Jobs(tmp_path).recover()))
        thread.start(); thread.join(timeout = 10)
        assert not thread.is_alive() and found == []
        q.finish(jid, 'success')
    assert q.recover() == [] and q.get(jid)['status'] == 'success'


def test_worker_death_does_not_interrupt_live_child(tmp_path):
    q = Jobs(tmp_path); jid = q.submit('snapshot')
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    child_script = tmp_path / 'child.py'
    child_script.write_text('''
from observe.jobs import Jobs
from pathlib import Path
import sys, time
q = Jobs(sys.argv[1])
with q.execution(sys.argv[2]):
    Path(sys.argv[3]).touch()
    while not Path(sys.argv[4]).exists(): time.sleep(0.02)
    q.finish(sys.argv[2], 'success')
''')
    launcher = '''
import subprocess, sys
from observe.jobs import Jobs
popen = subprocess.Popen
def start(args, **kwargs):
    return popen([sys.executable, sys.argv[2], sys.argv[1], args[-1], sys.argv[3], sys.argv[4]], **kwargs)
subprocess.Popen = start
Jobs(sys.argv[1]).run_next()
'''
    p = subprocess.Popen([sys.executable, '-c', launcher, str(tmp_path), str(child_script), str(ready), str(release)])
    try:
        until(ready.exists)
        p.terminate(); p.wait(timeout = 10)
        q.submit('snapshot')
        assert q.recover() == [] and q.run_next() is None
        release.touch()
        until(lambda: q.get(jid)['status'] == 'success')
    finally:
        release.touch()
        if p.poll() is None: p.terminate()
        p.wait(timeout = 10)
