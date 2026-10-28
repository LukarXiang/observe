"""Reject changed provenance, missing test coverage and invented index trades."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from observe.strategy_catalog import REVIEWS
from scripts import verify_strategy_batch92 as batch


def test_frozen_stage_scope_and_nontrading_identity():
    bound = batch.binding()
    assert bound['source_sha256'] == batch.stage.SOURCE_SHA
    assert bound['not_a_backtest'] and bound['new_trading_backtests'] == 0
    assert len(batch.stage.definitions(batch.stage.tree())) == 7


@pytest.mark.parametrize('name', ['index-input.parquet', 'independent-full-signal-audit.json', 'demo-coverage-audit.json'])
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
    with pytest.raises(ValueError, match='Accepted91 evidence changed'): batch.frozen_operand_binding(batch.stage.DIRECTORY)


def test_catalog_review_preserves_original_rules_without_trades():
    review = REVIEWS[batch.SOURCE_NAME]
    assert review['status'] == '非交易研究脚本' and review['implementations'] == []
    assert review['rules']['score_rows'] == 60 and review['rules']['quantile_rows'] == 20
    assert review['rules']['lower_quantile'] == .05 and review['rules']['upper_quantile'] == .85
    assert review['rules']['MA_rows'] == 5
    assert review['rules']['first_flag'] == 0 and review['rules']['exit_priority']
    assert review['original_strategy_complete'] is False


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
    monkeypatch.setattr(batch.stage, 'binding', batch.frozen_operand_binding)
    bound = batch.frozen_operand_binding(batch.stage.DIRECTORY)
    assert bound['source_sha256'] == batch.stage.SOURCE_SHA and bound['not_a_backtest']


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
