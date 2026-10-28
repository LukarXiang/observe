"""Hand-calculated crowd boundaries and source-loading rejection paths, without payloads."""
import hashlib
import json

import pytest

from scripts import research_strategy_batch97 as batch


def test_crowd_threshold_includes_the_next_rank():
    result = batch.independent_crowd(range(1,21))
    assert result == {'threshold':19.,'selected':2,'rows':20,'total':210.,'top_money':39.,'rounded':18.6}


def test_crowd_ties_are_all_included():
    assert batch.independent_crowd([5]*20)['selected'] == 20
    assert batch.independent_crowd([5]*20)['rounded'] == 100.


def test_small_pool_selects_one_instead_of_zero():
    assert batch.independent_crowd([1,3,2])['rounded'] == 50.


def test_zero_total_is_undefined():
    assert batch.independent_crowd([0]*20)['rounded'] is None


def test_empty_money_population_is_rejected():
    with pytest.raises(ValueError,match='Empty'): batch.independent_crowd([])


def source_fixture(tmp_path,monkeypatch,first):
    # Nineteen inert definitions satisfy the inventory but exercise definition-time safety.
    text = first+'\n'+ '\n'.join(f'def inert{i}():\n    return {i}' for i in range(18))
    path = tmp_path/'source.txt'; path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(batch,'SOURCE_SHA',(digest,batch.SOURCE_SHA[1]))
    return path


def test_loader_does_not_run_top_level_statements(tmp_path,monkeypatch):
    path = source_fixture(tmp_path,monkeypatch,"raise RuntimeError('top level forbidden')\ndef ordinary(n: int):\n    return n")
    ns = batch.definitions(path,0,{})
    assert ns['ordinary'](7) == 7


@pytest.mark.parametrize('definition',[
    'def ordinary(x=side_effect()):\n    return x',
    '@side_effect()\ndef ordinary():\n    return 0',
    'def ordinary(x: side_effect()):\n    return x',
    'def ordinary() -> side_effect():\n    return 0',
    'def ordinary(*x: side_effect()):\n    return x',
    'def ordinary(*, x=side_effect()):\n    return x',
])
def test_loader_rejects_definition_time_execution(tmp_path,monkeypatch,definition):
    path = source_fixture(tmp_path,monkeypatch,definition)
    with pytest.raises(ValueError): batch.definitions(path,0,{'side_effect':lambda:pytest.fail('must never execute')})


def test_loader_rejects_source_byte_drift(tmp_path,monkeypatch):
    path = source_fixture(tmp_path,monkeypatch,'def ordinary():\n    return 1')
    path.write_text(path.read_text()+'\n# changed')
    with pytest.raises(ValueError,match='identity'): batch.definitions(path,0,{})


def test_loader_rejects_inventory_drift(tmp_path,monkeypatch):
    path = source_fixture(tmp_path,monkeypatch,'# missing first function')
    with pytest.raises(ValueError,match='inventory'): batch.definitions(path,0,{})


def test_loader_rejects_unknown_source_number(tmp_path):
    with pytest.raises(ValueError,match='identity'): batch.definitions(tmp_path/'missing',2,{})


def test_partition_scope_cannot_shrink():
    with pytest.raises(ValueError,match='scope'): batch.verify_entries('.',{})


def test_partition_inventories_cannot_be_empty():
    with pytest.raises(ValueError,match='Empty'):
        batch.verify_entries('.',{n:{} for n in ('bars_1d','calendar','instruments')})


def partition_fixture(tmp_path,monkeypatch):
    tables = {}; assets = []
    for table in ('bars_1d','calendar','instruments'):
        path = tmp_path/(table+'.parquet'); path.write_bytes(b'archived '+table.encode())
        tables[table] = {'part':{'file':path.name,'sha':'different-logical-content-fingerprint','rows':1}}
        assets.append({'path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    index = tmp_path/'assets.json'; index.write_text(json.dumps({'assets':assets}))
    monkeypatch.setattr(batch,'ASSETS',index)
    monkeypatch.setattr(batch,'ASSETS_SHA',hashlib.sha256(index.read_bytes()).hexdigest())
    return tables


def test_byte_sha_does_not_use_logical_fingerprint(tmp_path,monkeypatch):
    tables = partition_fixture(tmp_path,monkeypatch)
    result = batch.verify_entries(tmp_path,tables)
    assert len(result) == 3
    assert all(r['logical_fingerprint'] != r['file_sha256'] for r in result.values())


def test_byte_drift_is_rejected_even_when_manifest_is_unchanged(tmp_path,monkeypatch):
    tables = partition_fixture(tmp_path,monkeypatch)
    (tmp_path/'bars_1d.parquet').write_bytes(b'drift')
    with pytest.raises(ValueError,match='byte SHA'): batch.verify_entries(tmp_path,tables)
