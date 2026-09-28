"""批次发布与跨进程文件锁（决策 19）。"""
import json
import subprocess
import sys

import pytest

from observe.data.locks import operation_lock
from observe.data.publication import BatchPublisher


def test_batch_is_published_atomically(tmp_path):
    p = BatchPublisher(tmp_path); p.write_batch('b1', {'bars_1d': ['bars_1d/2024.parquet'], 'adj_factors': ['adj.parquet'], 'corp_actions': ['ca.parquet']})
    assert p.read() is None                                                          # 写了批次不等于发布
    assert p.publish('b1').batch_id == 'b1' and json.loads((tmp_path / 'PUBLISHED.json').read_text())['tables']['adj_factors']
    with pytest.raises(FileExistsError): p.write_batch('b1', {'bars_1d': ['x']})


def test_lock_is_exclusive_across_processes(tmp_path):
    code = f"import time; from observe.data.locks import operation_lock\nwith operation_lock(r'{tmp_path}', 'baostock'):\n    print('held', flush = True); time.sleep(3)"
    child = subprocess.Popen([sys.executable, '-c', code], stdout = subprocess.PIPE, text = True)
    try:
        assert child.stdout.readline().strip() == 'held'
        with pytest.raises(RuntimeError, match = '锁被占用'):
            with operation_lock(tmp_path, 'baostock'): pass
    finally: child.wait()
    with operation_lock(tmp_path, 'baostock'): pass                                   # 占用进程退出后可再次获得
