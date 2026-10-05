"""冻结产物核验：逐项错误、引用图、迁移、路径边界及只读约束。"""
import json
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import main
from observe.integrity import verify_run
from observe.runs import RunRegistry, file_sha, write_json


def sealed(root, name, kind = 'run', subruns = None):
    out = root / 'runs' / name; out.mkdir(parents = True)
    write_json(out / 'status.json', {'run_id': name, 'kind': kind, 'status': 'success'})
    write_json(out / 'config.json', {'kind': kind})
    (out / 'scores.json').write_text('[{"score":1}]')
    if subruns is not None: write_json(out / 'subruns.json', subruns)
    reseal(out)
    return out


def reseal(out):
    status = json.loads((out / 'status.json').read_text())
    write_json(out / 'manifest.json', {**status, 'files': {p.name: file_sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})


def ref(out, **overrides):
    return {'run_id': out.name, 'output': str(out), 'status': 'success', 'manifest_sha256': file_sha(out / 'manifest.json'), **overrides}


def codes(report): return {i['code'] for i in report['issues']}


def test_tree_and_shared_child_are_checked_once_without_writes(tmp_path, monkeypatch):
    child = sealed(tmp_path, 'child'); parent = sealed(tmp_path, 'parent', 'experiment', {'research': ref(child), 'backtests': [ref(child)], 'benchmarks': []})
    before = {p: (file_sha(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs): raise AssertionError('核验不能写库、训练或创建产物')
    monkeypatch.setattr(RunRegistry, '_db', forbidden)
    monkeypatch.setattr('observe.models.RidgeModel.fit', forbidden)
    monkeypatch.setattr('observe.data.store.Store.__init__', forbidden)
    r = verify_run(tmp_path, parent)
    assert r['status'] == 'ok' and r['summary']['runs'] == 2 and r['summary']['references'] == 2
    assert r['summary']['unique_files_hashed'] == 9
    assert {p: (file_sha(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before
    assert verify_run(tmp_path, parent, recursive = False)['summary']['runs'] == 1


def test_all_missing_and_modified_files_are_reported(tmp_path):
    out = sealed(tmp_path, 'broken'); (out / 'scores.json').write_text('[]'); (out / 'config.json').unlink()
    r = verify_run(tmp_path, out)
    assert r['status'] == 'error' and {'hash_mismatch', 'missing_file'} <= codes(r)


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside', 'nested/../../outside', 'nested\\outside'])
def test_manifest_paths_cannot_escape_run(tmp_path, name):
    out = sealed(tmp_path, 'paths'); m = json.loads((out / 'manifest.json').read_text()); m['files'][name] = 'a' * 64
    write_json(out / 'manifest.json', m)
    assert 'invalid_artifact' in codes(verify_run(tmp_path, out))


def test_symlink_to_external_file_is_not_read(tmp_path, monkeypatch):
    out = sealed(tmp_path, 'links'); external = tmp_path / 'external'; external.write_text('private')
    try: (out / 'link').symlink_to(external)
    except OSError as e: pytest.skip(f'当前系统不允许创建符号链接：{e}')            # Windows 非管理员且未开开发者模式
    m = json.loads((out / 'manifest.json').read_text(encoding = 'utf-8')); m['files']['link'] = file_sha(external); write_json(out / 'manifest.json', m)
    import observe.integrity as module
    original = module.file_sha
    def guarded(path):
        assert Path(path) != external
        return original(path)
    monkeypatch.setattr(module, 'file_sha', guarded)
    assert 'invalid_artifact' in codes(verify_run(tmp_path, out))


def test_resealed_child_does_not_match_parent_reference(tmp_path):
    child = sealed(tmp_path, 'child'); parent = sealed(tmp_path, 'parent', 'experiment', {'research': ref(child), 'backtests': []})
    (child / 'scores.json').write_text('[]'); reseal(child)
    r = verify_run(tmp_path, parent)
    assert r['status'] == 'error' and 'reference_hash_mismatch' in codes(r)
    assert 'hash_mismatch' not in codes(r)


def test_untrusted_references_are_not_followed(tmp_path):
    child = sealed(tmp_path, 'child'); parent = sealed(tmp_path, 'parent', 'experiment', {'research': ref(child)})
    write_json(parent / 'subruns.json', {'research': ref(child, run_id = 'different')})
    r = verify_run(tmp_path, parent)
    assert 'untrusted_references' in codes(r) and r['summary']['runs'] == 1


def test_missing_child_and_identity_mismatch_are_both_reported(tmp_path):
    child = sealed(tmp_path, 'child'); parent = sealed(tmp_path, 'parent', 'experiment', {'research': ref(child, run_id = 'wrong'), 'backtests': [{'run_id': 'missing'}]})
    assert {'reference_identity_mismatch', 'unresolved_reference'} <= codes(verify_run(tmp_path, parent))


def test_migrated_nested_child_without_registry(tmp_path):
    child = sealed(tmp_path, 'child'); parent = sealed(tmp_path, 'parent', 'experiment', {'research': ref(child, output = r'D:\old\runs\child'), 'backtests': []})
    (parent / 'research').mkdir(); child.rename(parent / 'research' / 'child')
    assert verify_run(tmp_path, parent)['status'] == 'ok'
    assert not (tmp_path / 'runs/registry.sqlite').exists()


def test_legacy_pair_without_reference_hash_is_warning(tmp_path):
    child = sealed(tmp_path, 'child'); legacy = ref(child); legacy.pop('manifest_sha256')
    parent = sealed(tmp_path, 'parent', 'paired', {'arms': {'base': legacy, 'extended': legacy}, 'backtests': []})
    r = verify_run(tmp_path, parent)
    assert r['status'] == 'warning' and codes(r) == {'unpinned_reference'}


def test_cycles_terminate_and_are_errors(tmp_path):
    parent = sealed(tmp_path, 'parent', 'paired', {'arms': {'base': {'run_id': 'parent'}}, 'backtests': []})
    r = verify_run(tmp_path, parent)
    assert 'reference_cycle' in codes(r) and r['summary']['runs'] == 1


@pytest.mark.parametrize('status, expected', [('running', 'warning'), ('failed', 'warning'), ('success', 'error'), ('blocked', 'error')])
def test_unsealed_run_is_never_ok(tmp_path, status, expected):
    out = tmp_path / 'runs/unsealed'; out.mkdir(parents = True)
    write_json(out / 'status.json', {'run_id': out.name, 'status': status})
    assert verify_run(tmp_path, out)['status'] == expected


@pytest.mark.parametrize('manifest', ['{', '[]', '{"files":{}}', '{"files":[]}'])
def test_malformed_manifest_is_reported(tmp_path, manifest):
    out = sealed(tmp_path, 'invalid'); (out / 'manifest.json').write_text(manifest)
    assert verify_run(tmp_path, out)['status'] == 'error'


def test_readonly_registry_resolution(tmp_path, monkeypatch):
    out = sealed(tmp_path / 'external', 'registered'); reg = RunRegistry(tmp_path / 'runs')
    reg.record(out, json.loads((out / 'status.json').read_text()))
    before = file_sha(reg.path), reg.path.stat().st_mtime_ns
    original = sqlite3.connect
    def readonly(*args, **kwargs):
        assert args[0].endswith('?mode=ro') and kwargs['uri']
        return original(*args, **kwargs)
    monkeypatch.setattr('observe.runs.sqlite3.connect', readonly)
    assert verify_run(tmp_path, 'registered')['status'] == 'ok'
    assert (file_sha(reg.path), reg.path.stat().st_mtime_ns) == before


def test_api_and_cli_agree_and_have_explicit_exit_codes(tmp_path, capsys):
    out = sealed(tmp_path, 'good'); client = TestClient(create_app(tmp_path))
    assert main(['--root', str(tmp_path), 'runs', 'verify', out.name]) is None
    cli = json.loads(capsys.readouterr().out)
    assert client.get('/api/runs/good/verify').json() == cli
    assert client.get('/api/runs/missing/verify').status_code == 404
    (out / 'scores.json').write_text('[]')
    with pytest.raises(SystemExit) as exc: main(['--root', str(tmp_path), 'runs', 'verify', out.name])
    assert exc.value.code == 1 and json.loads(capsys.readouterr().out)['status'] == 'error'
    assert client.get('/api/runs/good/verify').json()['status'] == 'error'
    (out / 'manifest.json').unlink(); write_json(out / 'status.json', {'run_id': out.name, 'status': 'running'})
    with pytest.raises(SystemExit) as exc: main(['--root', str(tmp_path), 'runs', 'verify', out.name, '--shallow'])
    assert exc.value.code == 2 and json.loads(capsys.readouterr().out)['status'] == 'warning'


@pytest.mark.parametrize('kind, subruns', [('experiment', {}), ('experiment', {'research': None, 'backtests': []}),
                                          ('paired', {'arms': {}, 'backtests': []}), ('paired', {'arms': {'base': {}}, 'backtests': []})])
def test_successful_aggregate_cannot_omit_required_children(tmp_path, kind, subruns):
    out = sealed(tmp_path, 'incomplete', kind, subruns)
    assert 'missing_references' in codes(verify_run(tmp_path, out))


def test_manifest_identity_and_untracked_controls_are_reported(tmp_path):
    out = sealed(tmp_path, 'identity'); m = json.loads((out / 'manifest.json').read_text())
    m['run_id'] = 'other'; del m['files']['config.json']; write_json(out / 'manifest.json', m)
    assert {'identity_mismatch', 'untracked_control'} <= codes(verify_run(tmp_path, out))


def test_files_changed_after_hashing_are_reported(tmp_path, monkeypatch):
    out = sealed(tmp_path, 'changing')
    import observe.integrity as module
    original = module.file_sha
    def change_later(path):
        result = original(path)
        if Path(path).name == 'manifest.json': (out / 'scores.json').write_text('[]')
        return result
    monkeypatch.setattr(module, 'file_sha', change_later)
    assert 'changed_during_check' in codes(verify_run(tmp_path, out))


def test_empty_status_cannot_pass_even_when_hash_matches(tmp_path):
    out = sealed(tmp_path, 'empty'); write_json(out / 'status.json', {}); reseal(out)
    assert {'invalid_identity', 'invalid_status'} <= codes(verify_run(tmp_path, out))
