"""Prep gates refuse altered ancestry, hollow replay and noncanonical data paths."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts import verify_strategy_batch95 as batch
from scripts.review_strategy_batch18 import save


def test_alternate_root_refused_before_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'ancestry', lambda: pytest.fail('must reject before reading'))
    with pytest.raises(ValueError, match='canonical'): batch.prepare(tmp_path, batch.DIRECTORY)


def test_data_symlink_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); target = Path('payload'); target.mkdir(); Path('data').symlink_to(target)
    with pytest.raises(ValueError, match='canonical'): batch.require_root('data')


@pytest.mark.parametrize('kind', ['empty', 'subset', 'extra', 'live', 'copy'])
def test_complete_implementation_and_copy_required(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(batch, 'FILES', ('driver.py', 'tests.py'))
    directory = Path('acceptance'); hashes = {}
    for n in batch.FILES:
        Path(n).write_text('original\n'); target = directory/'implementation'/n
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(n).read_bytes())
        hashes[n] = batch.file_sha(n)
    assert batch.implementation(directory, hashes) == hashes
    wrong = dict(hashes)
    if kind == 'empty': wrong = {}
    elif kind == 'subset': wrong.pop('tests.py')
    elif kind == 'extra': wrong['unknown.py'] = '0'*64
    elif kind == 'live': Path('driver.py').write_text('changed')
    else: (directory/'implementation/driver.py').write_text('changed')
    with pytest.raises(ValueError): batch.implementation(directory, wrong)


def test_bound_rejects_changed_bytes_and_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); path = Path('input.json'); path.write_text('{"rows": 1}')
    sha = batch.file_sha(path); assert json.loads(batch.bound(path, sha)) == {'rows': 1}
    Path('linked.json').symlink_to(path)
    with pytest.raises(ValueError, match='symlink'): batch.bound('linked.json', sha)
    path.write_text('{"rows": 2}')
    with pytest.raises(ValueError, match='Bound bytes'): batch.bound(path, sha)


def test_ancestry_uses_frozen_catalog_without_live_hash(monkeypatch):
    names = []; actual = batch.bound
    def observed(path, sha):
        names.append(str(path)); return actual(path, sha)
    monkeypatch.setattr(batch, 'bound', observed)
    evidence = batch.ancestry()
    assert 'src/observe/strategy_catalog.py' not in names
    assert any(n.endswith('implementation-checked/src/observe/strategy_catalog.py') for n in evidence)
    assert len(evidence) > 50


@pytest.mark.parametrize('bad', ['rows', 'mean', 'trading', 'subset', 'command', 'bytes', 'unrelated_log'])
def test_replay_rejects_hollow_or_changed_proof(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path); directory = Path('acceptance'); directory.mkdir(); hashes = {'code': 'f'*64}
    monkeypatch.setattr(batch, 'validate', lambda d: hashes)
    log = directory/'replay.log'; log.write_text('worker executed\n'); replay = directory/'replay'; replay.mkdir()
    outputs = {}; frozen = Path('frozen-stage'); frozen.mkdir()
    monkeypatch.setattr(batch.stage, 'DIRECTORY', frozen)
    for n in batch.stage.OUTPUTS:
        if n.endswith('.parquet'):
            count = 22044 if n.startswith('native') else 4662
            pd.DataFrame({'value': range(count)}).to_parquet(replay/n, index=False)
        else:
            save(replay/n, {'count': 108} if n == 'diagnostics.json' else {'new_trading_backtests': 0,
                'formal_source_research_acceptance': False, 'original_strategy_complete': False})
        shutil.copyfile(replay/n, frozen/n); outputs[n] = batch.file_sha(replay/n)
    result = {'status': 'match', 'differences': 0, 'new_trading_backtests': 0, 'scope': [4662, 4662, 22044, 108],
        'network_forbidden': True, 'formal_source_research_acceptance': False, 'formal_trading_acceptance': False,
        'implementation_sha256': hashes, 'file_sha256': outputs}
    command = {'command': [batch.sys.executable, '-B', '-m', 'scripts.verify_strategy_batch95', 'worker'],
        'returncode': 0, 'log': str(log), 'sha256': batch.file_sha(log)}
    save(directory/'replay.json', result); save(directory/'replay-command.json', command)
    assert batch.validate_replay(directory) == result
    wrong = deepcopy(result)
    if bad == 'rows': wrong['scope'][0] = 0
    elif bad == 'mean': wrong['network_forbidden'] = False
    elif bad == 'trading': wrong['formal_trading_acceptance'] = True
    elif bad == 'subset': wrong['file_sha256'].pop(batch.stage.OUTPUTS[-1])
    elif bad == 'command': command['command'] = ['true']
    elif bad == 'unrelated_log':
        other = Path('unrelated.log'); other.write_text('different successful command\n')
        command['log'] = str(other); command['sha256'] = batch.file_sha(other)
    else: (replay/batch.stage.OUTPUTS[0]).write_text('changed')
    (directory/'replay.json').write_text(json.dumps(wrong))
    (directory/'replay-command.json').write_text(json.dumps(command))
    with pytest.raises(ValueError): batch.validate_replay(directory)


def test_worker_refuses_existing_output_before_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('acceptance'); (directory/'replay').mkdir(parents=True)
    monkeypatch.setenv('OBSERVE_BATCH95_WORKER', '1'); monkeypatch.setattr(batch.sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(batch, 'validate', lambda d: {})
    monkeypatch.setattr(batch.stage, 'compute', lambda d: pytest.fail('must refuse before computing'))
    with pytest.raises(ValueError, match='already exists'): batch.worker(directory)


def test_worker_requires_fresh_process_flags(tmp_path, monkeypatch):
    monkeypatch.delenv('OBSERVE_BATCH95_WORKER', raising=False)
    monkeypatch.setattr(batch, 'validate', lambda d: pytest.fail('must reject before reading'))
    with pytest.raises(ValueError, match='Fresh worker'): batch.worker(tmp_path)


def test_environment_removes_test_selection_and_import_overrides(monkeypatch):
    for n in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONPATH'): monkeypatch.setenv(n, 'override')
    env = batch.environment()
    assert not set(('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONPATH')) & set(env)
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_complete_refuses_publication_changed_while_waiting_for_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('preparation'); directory.mkdir()
    monkeypatch.setattr(batch, 'RECEIPT', Path('receipt.json'))
    frozen = {'published': {'batch_id': 'old'}}; save(directory/'input-inventory.json', frozen)
    current = [deepcopy(frozen)]
    monkeypatch.setattr(batch, 'require_root', lambda r: None)
    monkeypatch.setattr(batch, 'validate', lambda d: {})
    monkeypatch.setattr(batch, 'validate_replay', lambda d: {})
    monkeypatch.setattr(batch, 'input_inventory', lambda r: deepcopy(current[0]))
    @contextmanager
    def lock(*args):
        current[0] = {'published': {'batch_id': 'legally-appended-index'}}; yield
    monkeypatch.setattr(batch, 'operation_lock', lock)
    monkeypatch.setattr(batch, 'protect', lambda r, d: {})
    monkeypatch.setattr(batch, 'archive', lambda *a: {'manually_reviewed': 273,
        'locally_backtested_sources': 12, 'local_strategy_runs': 77})
    with pytest.raises(ValueError, match='[Ii]nventory changed'): batch.complete_preparation('data', directory)
    assert not batch.RECEIPT.exists()


def test_actual_runtime_network_helper_change_rejected(tmp_path, monkeypatch):
    source_root = Path.cwd(); monkeypatch.chdir(tmp_path); directory = Path('preparation'); manifest = {}
    names = {*batch.FILES, 'scripts/recover_strategy_batch94_reference.py'}
    for n in names:
        target = Path(n); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source_root/n, target)
    for n in batch.FILES:
        copy = directory/'implementation'/n; copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(n, copy); manifest[n] = batch.file_sha(n)
    batch.implementation(directory, manifest)
    Path('scripts/recover_strategy_batch94_reference.py').write_text('def network_blocked(): return None\n')
    with pytest.raises(ValueError): batch.implementation(directory, manifest)


def test_self_hashed_non_parquet_replay_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('preparation'); directory.mkdir(); hashes = {'code': 'f'*64}
    monkeypatch.setattr(batch, 'validate', lambda d: hashes); replay = directory/'replay'; replay.mkdir()
    outputs = {}; frozen = Path('frozen-stage'); frozen.mkdir()
    monkeypatch.setattr(batch.stage, 'DIRECTORY', frozen)
    for n in batch.stage.OUTPUTS:
        if n.endswith('.parquet'):
            count = 22044 if n.startswith('native') else 4662
            pd.DataFrame({'value': range(count)}).to_parquet(frozen/n, index=False)
            (replay/n).write_text('not numerical output')
        else:
            save(frozen/n, {'count': 108} if n == 'diagnostics.json' else {'new_trading_backtests': 0,
                'formal_source_research_acceptance': False, 'original_strategy_complete': False})
            shutil.copyfile(frozen/n, replay/n)
        outputs[n] = batch.file_sha(replay/n)
    result = {'status': 'match', 'differences': 0, 'scope': [4662, 4662, 22044, 108], 'network_forbidden': True,
        'implementation_sha256': hashes, 'file_sha256': outputs, 'new_trading_backtests': 0,
        'formal_source_research_acceptance': False, 'formal_trading_acceptance': False}
    log = directory/'replay.log'; log.write_text('pretend worker log\n')
    save(directory/'replay.json', result)
    save(directory/'replay-command.json', {'command': [batch.sys.executable, '-B', '-m',
        'scripts.verify_strategy_batch95', 'worker'], 'returncode': 0, 'log': str(log), 'sha256': batch.file_sha(log)})
    with pytest.raises((ValueError, OSError)): batch.validate_replay(directory)


def test_self_hashed_valid_parquet_with_changed_values_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('preparation'); directory.mkdir(); hashes = {'code': 'f'*64}
    monkeypatch.setattr(batch, 'validate', lambda d: hashes); replay = directory/'replay'; replay.mkdir()
    frozen = Path('frozen-stage'); frozen.mkdir(); monkeypatch.setattr(batch.stage, 'DIRECTORY', frozen)
    outputs = {}
    for n in batch.stage.OUTPUTS:
        if n.endswith('.parquet'):
            count = 22044 if n.startswith('native') else 4662
            pd.DataFrame({'value': range(count)}).to_parquet(frozen/n, index=False)
            pd.DataFrame({'value': [0]*count}).to_parquet(replay/n, index=False)
        else:
            save(frozen/n, {'count': 108} if n == 'diagnostics.json' else {'new_trading_backtests': 0,
                'formal_source_research_acceptance': False, 'original_strategy_complete': False})
            shutil.copyfile(frozen/n, replay/n)
        outputs[n] = batch.file_sha(replay/n)
    log = directory/'replay.log'; log.write_text('pretend worker log\n')
    save(directory/'replay.json', {'status': 'match', 'differences': 0, 'scope': [4662, 4662, 22044, 108],
        'network_forbidden': True, 'implementation_sha256': hashes, 'file_sha256': outputs,
        'new_trading_backtests': 0, 'formal_source_research_acceptance': False, 'formal_trading_acceptance': False})
    save(directory/'replay-command.json', {'command': [batch.sys.executable, '-B', '-m',
        'scripts.verify_strategy_batch95', 'worker'], 'returncode': 0, 'log': str(log), 'sha256': batch.file_sha(log)})
    with pytest.raises(AssertionError, match='DataFrame'): batch.validate_replay(directory)


def test_preparation_inventory_tamper_refused_before_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('preparation'); directory.mkdir()
    monkeypatch.setattr(batch, 'DIRECTORY', directory)
    hashes = {}
    for n in batch.BASELINE_FILES:
        (directory/n).write_text('{}'); hashes[n] = batch.file_sha(directory/n)
    save(directory/'preparation-binding.json', hashes)
    (directory/'input-inventory.json').write_text('{"claimed_rows": 100}')
    monkeypatch.setattr(batch, 'ancestry', lambda: pytest.fail('must reject before ancestry'))
    with pytest.raises(ValueError, match='Bound bytes'): batch.validate(directory)


def test_complete_refuses_replay_changed_while_waiting_for_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('preparation'); directory.mkdir()
    monkeypatch.setattr(batch, 'RECEIPT', Path('receipt.json')); save(directory/'input-inventory.json', {'stable': True})
    monkeypatch.setattr(batch, 'require_root', lambda r: None); monkeypatch.setattr(batch, 'validate', lambda d: {})
    current = [{'file_sha256': {'replay.parquet': 'old'}}]
    monkeypatch.setattr(batch, 'validate_replay', lambda d: deepcopy(current[0]))
    monkeypatch.setattr(batch, 'input_inventory', lambda r: {'stable': True})
    @contextmanager
    def lock(*args):
        current[0] = {'file_sha256': {'replay.parquet': 'changed'}}; yield
    monkeypatch.setattr(batch, 'operation_lock', lock)
    monkeypatch.setattr(batch, 'protect', lambda *a: pytest.fail('must refuse before protecting'))
    with pytest.raises(ValueError, match='Replay changed'): batch.complete_preparation('data', directory)
    assert not batch.RECEIPT.exists()
