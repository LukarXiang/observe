import pytest

from observe.runs import file_sha, write_json
from scripts.archive_strategy_batch11_blocked import finish, settlement_evidence, supplemental_checks
import scripts.archive_strategy_batch11_blocked as archive_module


def test_changed_tested_code_refuses_acceptance_before_running_simulation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = tmp_path / 'implementation.py'; code.write_text('x = 1\n')
    directory = tmp_path / 'batch'; short = directory / 'final-short-verifications'; short.mkdir(parents = True)
    write_json(short / 'implementation-sha256.json', {'implementation.py': file_sha(code)})
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}], 'implementation_sha256': {'implementation.py': 'outdated-test-fingerprint'}})
    with pytest.raises(ValueError, match = 'Tested implementation changed'): finish('data', directory)
    assert not (tmp_path / 'docs/handoff/2026-10-05-batch11-verification.json').exists()


def test_changed_official_raw_response_refuses_settlement_evidence(tmp_path):
    response = tmp_path / 'announcement.pdf'; response.write_bytes(b'original response')
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, {'responses': [{'path': str(response), 'sha256': file_sha(response)}]})
    response.write_bytes(b'changed response')
    with pytest.raises(ValueError, match = 'Settlement raw response changed'): settlement_evidence(manifest)


def test_explicit_network_failure_is_preserved_without_a_response_file(tmp_path):
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, {'responses': [{'error': 'SSLError: failed TLS handshake'}]})
    result = settlement_evidence(manifest)
    assert result['raw_responses_verified'] == 0 and result['failed_attempts_preserved'] == 1
    assert result['published'] is False


def test_missing_response_without_failure_record_is_rejected(tmp_path):
    manifest = tmp_path / 'manifest.json'; write_json(manifest, {'responses': [{}]})
    with pytest.raises(ValueError, match = 'Missing response without recorded failure'): settlement_evidence(manifest)


def test_supplemental_checks_reject_changed_script(tmp_path):
    source = tmp_path / 'checked.py'; source.write_text('x = 1\n')
    files = [archive_module.__file__, 'tests/unit/test_batch11_archive.py', 'scripts/research_delisting_evidence.py', str(source)]
    write_json(tmp_path / 'supplemental-checks.json', {
        'commands': [{'returncode': 0}], 'implementation_sha256': {name: file_sha(name) for name in files}})
    source.write_text('x = 2\n')
    with pytest.raises(ValueError, match = 'Supplemental tested implementation changed'): supplemental_checks(tmp_path)


def test_supplemental_checks_require_archive_test_and_research_fingerprints(tmp_path):
    write_json(tmp_path / 'supplemental-checks.json', {'commands': [{'returncode': 0}], 'implementation_sha256': {}})
    with pytest.raises(ValueError, match = 'Supplemental checked files missing'): supplemental_checks(tmp_path)
