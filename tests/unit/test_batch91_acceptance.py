"""Acceptance guards reject changed evidence while preserving source-only status."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from observe.strategy_catalog import REVIEWS
from scripts import verify_strategy_batch91 as batch


def test_five_sources_and_stage_binding_are_real_and_nonempty():
    rows = batch.source_checks(); bound = batch.binding()
    assert len(rows) == 5
    assert [r['complete_lines_read'] for r in rows] == [101, 166, 166, 166, 166]
    assert bound['not_a_backtest']
    assert bound['new_requests'] == 0


@pytest.mark.parametrize('name', ['C-input.parquet', 'sources-frozen.json', 'candidate-coverage-audit.json',
    'notebook-compatibility-review.json'])
def test_changed_stage_inputs_are_rejected(name, monkeypatch):
    original = batch.file_sha
    target = batch.stage.DIRECTORY/name
    monkeypatch.setattr(batch, 'file_sha', lambda path: 'changed' if Path(path) == target else original(path))
    with pytest.raises(ValueError, match='stage evidence changed'):
        batch.binding()


def test_source_reviews_do_not_admit_candidate_trades():
    corn = REVIEWS[batch.stage.SOURCES[0]]
    assert corn['status'] == '非交易研究脚本'
    assert corn['implementations'] == []
    assert corn['rules']['trading'] == '无订单/持仓/资金曲线，不由标题或z越界构造交易规则'
    for source in batch.stage.SOURCES[1:]:
        row = REVIEWS[source]
        assert row['status'] == '待数据'
        assert row['implementations'] == []
        assert row['rules']['cash_divisor'] == 1.2
        assert row['rules']['history_rows'] == 300
        assert row['rules']['time_threshold'] == 'hour+minute/60>=9.55，即9:33'
        assert row['rules']['same_AST_sources'] == list(batch.stage.SOURCES[1:])


def test_extra_evidence_replays_under_forbidden_network():
    result = batch.network_disabled(batch.supplemental_replay)
    assert result['status'] == 'match' and result['differences'] == 0
    assert result['synthetic_coefficient_error'] == 'KeyError'
    assert result['future_engine_reference_files'] == 2
    assert [r['common_rows'] for r in result['candidate_ranges']] == [2859, 1076]
    assert all(r['missing_open_reference_dates'] == ['2015-02-25', '2015-10-08', '2019-04-22']
        for r in result['candidate_ranges'])
    assert result['actual_statsmodels_executed'] is False


def test_binding_adapter_accepts_only_the_frozen_stage_directory():
    with pytest.raises(ValueError, match='Unexpected stage directory'):
        batch.frozen_operand_binding(Path('unexpected-stage'))


def test_stage_scope_change_is_rejected_before_replay(monkeypatch):
    original = batch.read
    def changed(path):
        doc = original(path)
        return {**doc, 'synthetic_cases': 0} if Path(path) == batch.STAGE_RECEIPT else doc
    monkeypatch.setattr(batch, 'read', changed)
    with pytest.raises(ValueError, match='Stage scope changed'):
        batch.binding()


@pytest.mark.parametrize('classname', ['tests.unit.test_batch91_acceptance', 'unrelated.tests'])
def test_partial_isolated_junit_is_rejected_before_checks(tmp_path, monkeypatch, classname):
    # An exit-zero run with one selected test cannot prove the intended eight-file suite.
    for name in batch.CORE:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    log = tmp_path/'tests.log'; log.write_text('1 passed, all others deselected')
    junit = tmp_path/'tests.xml'
    junit.write_text(f'<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0">'
        f'<testcase classname="{classname}" name="selected"/></testsuite></testsuites>')
    proof = {'status': 'passed', 'returncode': 0, 'skipped': 0, 'passed': 1,
        'command': batch.test_command(), 'log': str(log), 'log_sha256': batch.file_sha(log),
        'JUnit_file': str(junit), 'JUnit_sha256': batch.file_sha(junit),
        'candidate_implementation_sha256': {n: batch.file_sha(n) for n in batch.FILES}}
    monkeypatch.setattr(batch, 'validate_sources', lambda _: None)
    monkeypatch.setattr(batch, 'validate_catalog', lambda *_: None)
    monkeypatch.setattr(batch, 'read', lambda _: proof)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match='Isolated verification missing or stale'):
        batch.checks('data', tmp_path)


def test_pytest_selection_environment_is_removed(monkeypatch):
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k nonexistent --ignore=tests/unit/test_batch91_stage.py')
    monkeypatch.setenv('PYTEST_PLUGINS', 'external_selection_plugin')
    env = batch.test_environment()
    assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
    assert '-o' in batch.test_command() and 'addopts=' in batch.test_command()


def test_complete_junit_requires_every_intended_file_and_no_duplicate_cases(tmp_path):
    total = sum(batch.EXPECTED_TEST_FILES.values())
    xml = tmp_path/'complete.xml'
    root = ET.Element('testsuites')
    suite = ET.SubElement(root, 'testsuite', tests=str(total), errors='0', failures='0', skipped='0')
    for name, case in batch.expected_test_cases(): ET.SubElement(suite, 'testcase', classname=name, name=case)
    ET.ElementTree(root).write(xml)
    result = batch.test_coverage(xml)
    assert result['complete'] and result['passed'] == total and result['files'] == batch.EXPECTED_TEST_FILES
    suite[1].set('name', suite[0].get('name'))
    ET.ElementTree(root).write(xml)
    assert batch.test_coverage(xml)['complete'] is False


def test_baseline_helper_change_is_rejected(tmp_path, monkeypatch):
    helper = tmp_path/'helper.py'; helper.write_text('original')
    baseline = {'code': [{'path': str(helper), 'sha256': batch.file_sha(helper)}]}
    helper.write_text('changed after isolation')
    monkeypatch.setattr(batch, 'read', lambda _: baseline)
    with pytest.raises(ValueError, match='Unrelated baseline implementation changed'):
        batch.validate_baseline(tmp_path)
    (tmp_path/'source-reviews').mkdir()
    def unexpected_publish(*_args):
        raise AssertionError('Repeated catalog must fail before generation or publication')
    monkeypatch.setattr(batch, 'catalog_strategies', unexpected_publish)
    with pytest.raises(ValueError, match='Catalog phase already archived'):
        batch.catalog('data', tmp_path)


@pytest.mark.parametrize('name', ['baseline.json', 'input-binding.json'])
def test_adapter_preserves_upstream_accepted_evidence_checks(monkeypatch, name):
    target = Path('data/staging/strategies-batch90/20261010-intraday-source-acceptance')/name
    original = batch.file_sha
    changed = lambda path: 'changed' if Path(path) == target else original(path)
    monkeypatch.setattr(batch, 'file_sha', changed)
    monkeypatch.setattr(batch.stage, 'file_sha', changed)
    with pytest.raises(ValueError, match='Accepted90 evidence changed'):
        batch.frozen_operand_binding(batch.stage.DIRECTORY)


def test_adapter_catalog_replay_keeps_original_binding_callable():
    result = batch.network_disabled(lambda: batch.replay_values(True))
    assert result['status'] == 'match' and result['differences'] == 0
    assert result['synthetic_cases'] == 158 and result['candidate_rows_overlapping_ranges'] == 3935
