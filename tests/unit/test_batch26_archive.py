from pathlib import Path
from types import SimpleNamespace

import pytest

from observe.runs import file_sha, write_json
from scripts import recover_strategy_batch26 as batch


def prepared(tmp_path, monkeypatch):
    implementation = tmp_path / 'implementation.py'; implementation.write_text('version = 1\n')
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(batch, 'FILES', ('implementation.py',))
    directory = tmp_path / 'archive'; directory.mkdir()
    copy = directory / 'implementation-checked/implementation.py'; copy.parent.mkdir(); copy.write_bytes(implementation.read_bytes())
    impl_hashes = {'implementation.py': file_sha(implementation)}
    for name in batch.CORE | set(batch.offline_names(impl_hashes)): write_json(directory / name, {})
    rows = []
    for k, command in enumerate(batch.required_commands()):
        path = directory / f'{k}.log'; path.write_text('passed\n')
        rows.append({'command': command, 'returncode': 0, 'log': str(path), 'log_sha256': file_sha(path)})
    checked = {'status': 'passed', 'commands': rows, 'implementation_sha256': {'implementation.py': file_sha(implementation)},
        'evidence_sha256': {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()}}
    return directory, implementation, checked


@pytest.mark.parametrize('defect', ['failed', 'commands', 'core', 'implementation', 'evidence', 'log'])
def test_finish_rejects_unchecked_or_changed_evidence_before_binding(tmp_path, monkeypatch, defect):
    directory, implementation, checked = prepared(tmp_path, monkeypatch)
    if defect == 'failed': checked['commands'][1]['returncode'] = 1
    if defect == 'commands':
        for row in checked['commands']: row['command'] = ['true']
    if defect == 'core': checked['evidence_sha256'] = {}
    if defect == 'implementation': implementation.write_text('version = 2\n')
    if defect == 'evidence': write_json(directory / 'component-research.json', {'changed': True})
    if defect == 'log': Path(checked['commands'][0]['log']).write_text('changed\n')
    write_json(directory / 'checked-state.json', checked)
    def reached(*args): raise AssertionError('Gate should fail before binding')
    monkeypatch.setattr(batch, 'binding', reached)
    with pytest.raises(ValueError, match='Required|Checked|changed'): batch.finish(tmp_path, directory)


def test_failed_checks_keep_logs_without_freezing_success(tmp_path, monkeypatch):
    directory, _, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, 'binding', lambda *args: {})
    def run(command, stdout, **kwargs): stdout.write('one failure\n'); return SimpleNamespace(returncode=1)
    monkeypatch.setattr(batch.subprocess, 'run', run)
    with pytest.raises(ValueError, match='Check failed'): batch.checks(tmp_path, directory)
    assert not (directory / 'checked-state.json').exists()
    log = next(directory.glob('checks-*/*.log')); assert 'one failure' in log.read_text()


def test_offline_worker_code_drift_is_rejected_before_receipt(tmp_path, monkeypatch):
    directory, implementation, _ = prepared(tmp_path, monkeypatch)
    _, receipt_name = batch.offline_names({'implementation.py': file_sha(implementation)})
    before = (directory / receipt_name).read_bytes()
    def run(*args, **kwargs): implementation.write_text('version = 2\n'); return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, 'run', run)
    with pytest.raises(ValueError, match='Offline implementation changed'): batch.offline(tmp_path, directory, final=True)
    assert (directory / receipt_name).read_bytes() == before
