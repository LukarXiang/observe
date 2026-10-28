"""Reject financial provenance mutations and incomplete formal research checks."""
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from observe.strategy_catalog import REVIEWS
from scripts import verify_strategy_batch93 as batch


def test_frozen_stage_scope_and_nontrading_identity():
    bound = batch.binding()
    assert bound['source_sha256'] == list(batch.stage.SOURCE_SHA)
    assert bound['not_a_backtest'] and bound['new_trading_backtests'] == 0
    assert len(batch.stage.source_tree(0).body) > 11


@pytest.mark.parametrize('name', ['financial-operands.parquet', 'operand-inventory.json', 'diagnostics.json'])
def test_changed_stage_evidence_is_rejected(name, monkeypatch):
    original = batch.file_sha; target = batch.stage.DIRECTORY/name
    monkeypatch.setattr(batch, 'file_sha', lambda p: 'changed' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='Stage evidence changed'): batch.binding()


def test_changed_stage_scope_is_rejected(monkeypatch):
    original = batch.read
    monkeypatch.setattr(batch, 'read', lambda p: {**original(p), 'new_trading_backtests': 1}
        if Path(p) == batch.STAGE_CHECKPOINT else original(p))
    with pytest.raises(ValueError, match='Stage scope changed'): batch.binding()


def test_adapter_rejects_unexpected_directory(tmp_path):
    with pytest.raises(ValueError, match='Unexpected stage directory'): batch.frozen_operand_binding(tmp_path)


@pytest.mark.parametrize('name', ['baseline.json', 'input-binding.json'])
def test_adapter_preserves_upstream_evidence(name, monkeypatch):
    target = batch.previous.DIRECTORY/name; original = batch.stage.file_sha
    monkeypatch.setattr(batch.stage, 'file_sha', lambda p: 'changed' if Path(p) == target else original(p))
    with pytest.raises(ValueError, match='Accepted92 evidence changed'): batch.frozen_operand_binding(batch.stage.DIRECTORY)


def test_catalog_review_preserves_original_rules_without_trades():
    ff, three = [REVIEWS[name] for name in batch.stage.SOURCES]
    for review in (ff, three):
        assert review['status'] == '待数据' and review['implementations'] == []
        assert review['original_strategy_complete'] is False
    assert ff['rules']['selected_fields'] == batch.stage.SCORE_FIELDS
    assert ff['rules']['max_window'] == 1 and ff['rules']['paused_rows'] == 22
    assert three['rules']['buy_cash_divisor'] == three['rules']['target_count'] == 50
    assert three['rules']['winsorize_quantiles'] == [.05, .93]


def test_full_junit_rejects_subset_duplicates_and_skips(tmp_path):
    cases = batch.expected_test_cases(); xml = tmp_path/'tests.xml'
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests=str(len(cases)), failures='0', errors='0', skipped='0')
    for name, case in cases: ET.SubElement(suite, 'testcase', classname=name, name=case)
    ET.ElementTree(root).write(xml)
    assert batch.test_coverage(xml)['complete']
    original = dict(suite[1].attrib); suite[1].attrib.update(suite[0].attrib)
    ET.ElementTree(root).write(xml); assert not batch.test_coverage(xml)['complete']
    suite[1].attrib = original; ET.SubElement(suite[0], 'skipped')
    ET.ElementTree(root).write(xml); assert not batch.test_coverage(xml)['complete']
    suite.remove(suite[0]); suite.set('tests', str(len(cases)-1))
    ET.ElementTree(root).write(xml); assert not batch.test_coverage(xml)['complete']


def test_selection_environment_is_removed(monkeypatch):
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k absent')
    monkeypatch.setenv('PYTEST_PLUGINS', 'selection_plugin')
    env = batch.test_environment()
    assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
    assert 'addopts=' in batch.test_command()


def test_unrelated_baseline_changes_are_rejected(tmp_path, monkeypatch):
    helper = tmp_path/'helper.py'; helper.write_text('original')
    baseline = {'code': [{'path': str(helper), 'sha256': batch.file_sha(helper)}]}
    helper.write_text('modified')
    monkeypatch.setattr(batch, 'read', lambda _: baseline)
    with pytest.raises(ValueError, match='Unrelated baseline implementation changed'): batch.validate_baseline(tmp_path)


def test_repeated_catalog_is_rejected_before_writes(tmp_path, monkeypatch):
    (tmp_path/'source-reviews').mkdir(); monkeypatch.setattr(batch, 'DIRECTORY', tmp_path)
    monkeypatch.setattr(batch, 'binding', lambda *_: pytest.fail('Must reject before validation or publication'))
    with pytest.raises(ValueError, match='Catalog phase already archived'): batch.catalog('data', tmp_path)


def test_original_binding_capture_prevents_recursion(monkeypatch):
    monkeypatch.setattr(batch.stage, 'parent_binding', batch.frozen_operand_binding)
    bound = batch.frozen_operand_binding(batch.stage.DIRECTORY)
    assert bound['sha256'] == batch.stage.PARENT_SHA


def test_required_command_contract_rejects_empty_or_substituted(tmp_path):
    log = tmp_path/'check.log'; log.write_text('success')
    commands = [{'command': c, 'returncode': 0, 'log': str(log), 'sha256': batch.file_sha(log)}
        for c in batch.check_commands()]
    batch.validate_commands({'commands': commands})
    for invalid in ([], commands[:2], [*commands[:2], {**commands[2], 'command': ['true']}]):
        with pytest.raises(ValueError, match='Required command scope changed'):
            batch.validate_commands({'commands': invalid})


def test_stage_control_mutation_before_acceptance_is_rejected(tmp_path, monkeypatch):
    old = {'published': {'batch_id': 'frozen'}, 'controls': {'runs/old/result.json': 'original'}, 'partitions': {}}
    current = {**old, 'controls': {'runs/old/result.json': 'changed before new baseline'}}
    monkeypatch.setattr(batch, 'read', lambda p: old if Path(p) == batch.stage.DIRECTORY/'baseline.json' else current)
    with pytest.raises(ValueError, match='Stage-to-acceptance old assets changed'): batch.stage_ancestry(tmp_path)
    current = {**old, 'controls': {**old['controls'], 'runs/new/config.json': 'new-before-preflight'}}
    with pytest.raises(ValueError, match='Stage-to-acceptance old assets changed'): batch.stage_ancestry(tmp_path)


def test_new_run_inventory_is_rejected_before_acceptance_publication(tmp_path, monkeypatch):
    config = tmp_path/'runs/old/config.json'; config.parent.mkdir(parents=True); config.write_text('{}')
    nested = tmp_path/'runs/old/variants/base/config.json'; nested.parent.mkdir(parents=True); nested.write_text('{}')
    monkeypatch.setattr(batch, 'read', lambda _: {'controls': {'runs/old/config.json': 'sha',
        'runs/old/variants/base/config.json': 'nested-sha'}})
    batch.validate_run_inventory(tmp_path, tmp_path)
    extra = tmp_path/'runs/new/config.json'; extra.parent.mkdir(); extra.write_text('{}')
    with pytest.raises(ValueError, match='Run inventory changed before acceptance publication'):
        batch.validate_run_inventory(tmp_path, tmp_path)


def test_isolation_candidate_must_match_tested_manifest():
    current = {n: batch.file_sha(n) for n in batch.FILES}
    proof = {'candidate_implementation_sha256': current}; copied = {'files': dict(current)}
    batch.validate_copied_candidate(proof, copied)
    copied['files'][batch.FILES[0]] = 'old-isolated-code'
    with pytest.raises(ValueError, match='Isolated candidate differs from tested copies'):
        batch.validate_copied_candidate(proof, copied)


def test_independent_numeric_missing_and_nonfinite_boundaries():
    numbers, bad = batch.independent_number(pd.Series([' null ', '.', None, '--', '1e3', '-2.5', 'inf', 'bad', ' 0 ']))
    assert np.isnan(numbers[:4]).all() and numbers[4:6].tolist() == [1000., -2.5]
    assert np.isnan(numbers[6:8]).all() and numbers[8] == 0
    assert bad.tolist() == [False]*6+[True, True, False]


def test_partition_manifest_set_and_identity_are_exact():
    stage = batch.stage; state = batch.read(Path('data/snapshots')/f'{stage.SNAPSHOT}.json')
    bound = batch.read(stage.DIRECTORY/'input-binding.json')
    batch.validate_partition_manifest(bound, state)
    for rows in (bound['partitions'][:-1], [*bound['partitions'][:-1], bound['partitions'][0]],
                 [{**bound['partitions'][0], 'snapshot_logical_sha': 'wrong'}, *bound['partitions'][1:]]):
        with pytest.raises(ValueError, match='Financial partition set or identity changed'):
            batch.validate_partition_manifest({**bound, 'partitions': rows}, state)


def test_alternate_root_rejected_before_any_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'binding', lambda *_: pytest.fail('Root must be rejected before binding'))
    with pytest.raises(ValueError, match='Only canonical data root'):
        batch.preflight(tmp_path/'data', tmp_path/'acceptance')


def test_candidate_publication_copies_validated_bytes_without_live_regeneration(tmp_path, monkeypatch):
    root = tmp_path/'data'; directory = tmp_path/'acceptance'; candidate_root = directory/'candidate'
    folder = candidate_root/'catalog/strategies/frozen'; folder.mkdir(parents=True)
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'): (folder/name).write_bytes(name.encode())
    latest = root/'catalog/strategies/latest.json'; latest.parent.mkdir(parents=True)
    batch.save(latest, {'catalog_id': 'old'})
    batch.save(directory/'catalog-before.json', {'catalog_id': 'old'})
    candidate = {'catalog_id': 'frozen', 'output': str(folder)}
    monkeypatch.setattr(batch, 'catalog_strategies', lambda *_: pytest.fail('Do not regenerate live source catalog'))
    result = batch.publish_candidate(root, directory, candidate)
    assert result['output'] == str(root/'catalog/strategies/frozen')
    assert batch.read(latest)['catalog_id'] == 'frozen'
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        assert (Path(result['output'])/name).read_bytes() == (folder/name).read_bytes()


def test_candidate_publication_rejects_concurrent_navigation_change(tmp_path):
    root = tmp_path/'data'; directory = tmp_path/'acceptance'; directory.mkdir()
    latest = root/'catalog/strategies/latest.json'; latest.parent.mkdir(parents=True)
    batch.save(latest, {'catalog_id': 'other-editor'})
    batch.save(directory/'catalog-before.json', {'catalog_id': 'old'})
    with pytest.raises(ValueError, match='Active catalog changed before candidate publication'):
        batch.publish_candidate(root, directory, {'catalog_id': 'frozen', 'output': 'unused'})
    assert not (root/'catalog/strategies/frozen').exists()


def test_progress_catalog_identity_rejects_same_count_competitor(tmp_path):
    directory = tmp_path/'acceptance'; root = tmp_path/'data'
    output = root/'catalog/strategies/frozen'; output.mkdir(parents=True)
    (output/'catalog.json').write_text('frozen catalog bytes')
    sha = batch.file_sha(output/'catalog.json')
    latest = root/'catalog/strategies/latest.json'
    batch.save(latest, {'catalog_id': 'frozen', 'directory': str(output), 'sha256': sha})
    batch.save(directory/'source-reviews/review.json', {'catalog': {'catalog_id': 'frozen', 'output': str(output)}})
    batch.save(directory/'offline-catalog.json', {'files': [{'name': 'catalog.json', 'sha256': sha}]})
    progress = {'catalog_id': 'frozen', 'catalog_sha256': sha, 'manually_reviewed': 270,
        'locally_backtested_sources': 12, 'local_strategy_runs': 75}
    batch.validate_progress_catalog(root, directory, progress)
    for altered in ({**progress, 'catalog_id': 'competing'}, {**progress, 'catalog_sha256': 'competing-SHA'}):
        with pytest.raises(ValueError, match='Final progress catalog identity differs'):
            batch.validate_progress_catalog(root, directory, altered)
    batch.write_json(latest, {'catalog_id': 'competing', 'directory': str(output), 'sha256': sha})
    with pytest.raises(ValueError, match='Final catalog navigation differs'):
        batch.validate_progress_catalog(root, directory, progress)
