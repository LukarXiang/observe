import json

import pytest

from scripts.prepare_handoff import collect, record, safe, verify


def fixture(root):
    def write(name, value):
        path = root / name; path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(value), encoding = 'utf-8')
    write('data/std/calendar/old.parquet', ['old']); write('data/std/calendar/new.parquet', ['new'])
    write('data/PUBLISHED.json', {'batch_id': 'new', 'tables': {'calendar': {'all': {'file': 'std/calendar/new.parquet'}}}})
    write('data/snapshots/old.json', {'tables': {'calendar': {'all': {'file': 'std/calendar/old.parquet'}}}})
    write('data/runs/example/status.json', {'status': 'success'})
    write('data/runs/registry.sqlite', {'machine_specific': True})
    write('data/runs/.DS_Store', {'finder': True})
    return {'published_batch': 'new', 'include': ['data/PUBLISHED.json', 'data/runs'], 'runs': ['example'], 'snapshots': ['old']}


def test_handoff_keeps_snapshot_partition_union_and_excludes_machine_state(tmp_path):
    plan = fixture(tmp_path); names = collect(tmp_path, plan)
    assert names == ['data/PUBLISHED.json', 'data/runs/example/status.json', 'data/snapshots/old.json', 'data/std/calendar/new.parquet', 'data/std/calendar/old.parquet']
    (tmp_path / 'data/std/calendar/old.parquet').unlink()
    with pytest.raises(FileNotFoundError): collect(tmp_path, plan)


def test_handoff_rejects_changed_published_state(tmp_path):
    plan = fixture(tmp_path); plan['published_batch'] = 'different'
    with pytest.raises(ValueError, match = '发布批次已变化'): collect(tmp_path, plan)


def test_handoff_detects_missing_and_corrupt_files_and_checks_financial_inputs(tmp_path):
    plan = fixture(tmp_path); entries = [record(tmp_path, name) for name in collect(tmp_path, plan)]
    raw = tmp_path / 'annual.csv'; raw.write_text('raw financials')
    manifest = {'schema': 1, 'project_files': [], 'files': entries, 'financial_inputs': [record(tmp_path, 'annual.csv')]}
    assert verify(tmp_path, manifest)['status'] == 'ok'
    assert verify(tmp_path, manifest, True)['checked_files'] == len(entries) + 1
    raw.write_text('changed financials')
    assert verify(tmp_path, manifest)['status'] == 'ok'
    assert verify(tmp_path, manifest, True)['issues'] == [{'path': 'annual.csv', 'problem': 'hash_mismatch'}]
    (tmp_path / 'data/std/calendar/old.parquet').unlink()
    (tmp_path / 'data/std/calendar/new.parquet').write_text('corrupt')
    assert {i['problem'] for i in verify(tmp_path, manifest)['issues']} == {'missing', 'hash_mismatch'}


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside', 'data/../../outside', 'data\\outside', '.'])
def test_handoff_rejects_paths_outside_project(tmp_path, name):
    with pytest.raises(ValueError): safe(tmp_path, name)


def test_handoff_rejects_symlinked_source(tmp_path):
    outside = tmp_path.parent / 'outside.json'; outside.write_text('private')
    (tmp_path / 'linked').symlink_to(outside)
    with pytest.raises(ValueError): safe(tmp_path, 'linked')
