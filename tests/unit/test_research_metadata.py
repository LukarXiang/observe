import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('metadata_export', Path(__file__).parents[2] / 'scripts/export_research_metadata.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_export(root):
    b = m.Export(root)
    project = {'current': {'checkpoint': 'working'}, 'accepted': {'checkpoint': 'accepted'},
               'indexes': {'assets': 'data/metadata/assets/index.json', 'runs': 'data/metadata/runs/index.json'}}
    b.put('project-state.json', project)
    b.put('evidence/implementation-files.json', [])
    b.put('assets/index.json', {'assets': [{'path': 'data/std/example.parquet', 'file_sha256': 'a' * 64}], 'missing_snapshots': []})
    b.put('runs/index.json', [])
    return b, project, 'status\n'


def test_git_only_and_partial_payload(tmp_path):
    m.publish(*fixture_export(tmp_path))
    assert m.verify(tmp_path)['status'] == 'ok'
    result = m.verify(tmp_path, payload=True)
    assert result['status'] == 'blocked' and result['payload_verified'] is False
    assert result['issues'][0]['problem'] == 'missing_payload'


def test_idempotent_and_old_state_immutable(tmp_path):
    first = m.publish(*fixture_export(tmp_path))
    assert m.publish(*fixture_export(tmp_path)) == first
    old_index = json.loads((tmp_path / 'data/metadata/index.json').read_bytes())
    old_bytes = m.safe(tmp_path, old_index['project']).read_bytes()
    b, p, page = fixture_export(tmp_path)
    p['current']['checkpoint'] = 'next'
    b.files['project-state.json'] = m.encoded(p)
    m.publish(b, p, page)
    assert m.safe(tmp_path, old_index['project']).read_bytes() == old_bytes
    assert m.verify(tmp_path)['status'] == 'ok'


@pytest.mark.parametrize('path', ['../escape', '/tmp/escape', 'C:\\escape', 'data/../../escape', ''])
def test_reject_unsafe_paths(tmp_path, path):
    with pytest.raises(ValueError): m.safe(tmp_path, path)


def test_reject_symlink(tmp_path):
    (tmp_path / 'real').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'real', target_is_directory=True)
    with pytest.raises(ValueError): m.safe(tmp_path, 'link/file')


def test_input_change_keeps_index(tmp_path):
    m.publish(*fixture_export(tmp_path))
    original = (tmp_path / 'data/metadata/index.json').read_bytes()
    source = tmp_path / 'source.json'
    source.write_text('{}')
    b, p, page = fixture_export(tmp_path)
    b.read('source.json')
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Input changed'): m.publish(b, p, page)
    with pytest.raises(ValueError, match='Input changed'): b.read('source.json')
    assert (tmp_path / 'data/metadata/index.json').read_bytes() == original


def test_immutable_conflict_keeps_index(tmp_path):
    b, p, page = fixture_export(tmp_path)
    b.put('catalogs/frozen/catalog.json', [])
    b.originals.add('catalogs/frozen/catalog.json')
    m.publish(b, p, page)
    original = (tmp_path / 'data/metadata/index.json').read_bytes()
    b.files['catalogs/frozen/catalog.json'] = m.encoded([1])
    with pytest.raises(ValueError, match='Immutable metadata conflict'): m.publish(b, p, page)
    assert (tmp_path / 'data/metadata/index.json').read_bytes() == original


def test_index_cannot_redirect_outside_manifest(tmp_path):
    m.publish(*fixture_export(tmp_path))
    path = tmp_path / 'data/metadata/index.json'
    value = json.loads(path.read_bytes())
    value['implementation_files'] = 'data/metadata/foreign.json'
    path.write_bytes(m.encoded(value))
    with pytest.raises(ValueError, match='outside manifest'): m.verify(tmp_path)


def test_validation_binds_run_and_reproduction():
    item = {'run_id': 'r', 'report': {'v': 1}, 'reproduction': {'run_id': 'replay'}}
    doc = {'status': 'ok', 'strategies': [item]}
    assert m.validated_report(doc, 'r', {'v': 1}, 'replay')
    assert not m.validated_report(doc, 'other', {'v': 1}, 'replay')
    assert not m.validated_report(doc, 'r', {'v': 1}, 'other')
    single = {'status': 'ok', 'run': {'run_id': 'r'}, 'report': {'v': 1}, 'reproduction': {'run_id': 'replay'}}
    assert m.validated_report(single, 'r', {'v': 1}, 'replay')


def test_daily_coverage_keeps_scope_without_exporting_payload():
    rows = [{'date': f'day-{i}', 'valid': bool(i % 2)} for i in range(1000)]
    result = m.coverage_summary(rows)
    assert result['rows'] == 1000
    assert result['first'] == rows[0] and result['last'] == rows[-1]
    assert len(m.encoded(result)) < 500
    assert m.coverage_summary(None) is None


def test_failed_install_leaves_no_partial_target(tmp_path, monkeypatch):
    target = tmp_path / 'immutable.json'
    def fail(*args): raise OSError('interrupted')
    monkeypatch.setattr(m.os, 'link', fail)
    with pytest.raises(OSError): m.install(target, b'complete')
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_selected_run_requires_snapshot_and_only_used_partitions(tmp_path):
    b, p, page = fixture_export(tmp_path)
    base = 'data/runs/run/'
    payloads = {base + 'config.json': {'snapshot_id': 'snap'},
                base + 'data_manifest.json': {'used': {'bars': {'one': {'file': 'std/used.parquet'}}},
                                               'tables': {'bars': {'unused': {'file': 'std/unused.parquet'}}}},
                base + 'manifest.json': {}}
    assets = []
    for name, value in payloads.items():
        data = m.encoded(value)
        path = m.safe(tmp_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        assets.append({'path': name, 'file_sha256': m.digest(data)})
    used = tmp_path / 'data/std/used.parquet'
    used.parent.mkdir(parents=True)
    used.write_bytes(b'used')
    assets.append({'path': 'data/std/used.parquet', 'file_sha256': m.digest(b'used')})
    snap = tmp_path / 'data/snapshots/snap.json'
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b'{}')
    assets.append({'path': 'data/snapshots/snap.json', 'file_sha256': m.digest(b'{}')})
    audit = tmp_path / 'data/audits/latest.json'
    audit.parent.mkdir(parents=True)
    audit.write_bytes(b'{}')
    assets.append({'path': 'data/audits/latest.json', 'file_sha256': m.digest(b'{}'), 'kind': 'audit_control'})
    b.files['assets/index.json'] = m.encoded({'assets': assets, 'missing_snapshots': []})
    b.files['runs/index.json'] = m.encoded([{'run_id': 'run'}])
    m.publish(b, p, page)
    assert m.verify(tmp_path, payload=True, run_id='run')['status'] == 'ok'
    audit.unlink()
    assert m.verify(tmp_path, payload=True, run_id='run')['issues'] == [{'path': 'data/audits/latest.json', 'problem': 'missing_payload'}]
    audit.write_bytes(b'{}')
    snap.unlink()
    result = m.verify(tmp_path, payload=True, run_id='run')
    assert result['status'] == 'blocked'
    assert result['issues'] == [{'path': 'data/snapshots/snap.json', 'problem': 'missing_payload'}]


def test_comparison_only_legacy_run_is_not_silently_omitted(tmp_path):
    out = tmp_path / 'data/runs/legacy'
    out.mkdir(parents=True)
    (out / 'status.json').write_bytes(m.encoded({'kind': 'paired', 'status': 'success', 'reproduce_of': 'original'}))
    (out / 'comparison.json').write_bytes(m.encoded({'result': 'match', 'differences': []}))
    b = m.Export(tmp_path)
    m.export_runs(b, [], [], {})
    assert len(b.run_index) == 1 and b.run_index[0]['kind'] == 'paired'
    result = json.loads(b.files['runs/legacy/summary.json'])
    assert result['configuration_available'] is False
    assert result['result_scope'] == 'status_or_comparison_only'
    assert result['comparison']['result'] == 'match'
    assert result['reproduce_of'] == 'original'
