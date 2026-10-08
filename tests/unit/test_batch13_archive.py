from pathlib import Path

import pytest

from observe.runs import file_sha, write_json
import scripts.review_strategy_batch13 as archive_module


@pytest.mark.parametrize('fault', ['changed_input', 'missing_input', 'original_sha'])
def test_offline_evidence_refused_before_freezing(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'batch'; directory.mkdir()
    code_paths = ['scripts/review_strategy_batch13.py', 'src/observe/strategy_catalog.py', 'tests/unit/test_catalog_dependencies.py']
    for name in code_paths:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'code\n')
    monkeypatch.setattr(archive_module, '__file__', str(tmp_path / code_paths[0]))
    source = tmp_path / code_paths[0]
    record = {'source_path': str(source), 'source_copy': str(source), 'source_sha256': file_sha(source)}
    write_json(directory / 'source-reviews/review.json', {'sources': [record] * len(archive_module.SOURCES)})
    original_input = directory / 'input.parquet'; original_input.write_bytes(b'original input')
    repeated_input = directory / 'repeated.parquet'; repeated_input.write_bytes(original_input.read_bytes())
    original = directory / 'trend-research.json'
    repeated = directory / 'repeated.json'
    write_json(original, {'not_a_backtest': True, 'input_file': str(original_input), 'input_sha256': file_sha(original_input)})
    write_json(repeated, {'input_file': str(repeated_input), 'input_sha256': file_sha(repeated_input)})
    write_json(directory / 'trend-offline-verification.json', {'status': 'match', 'differences': 0, 'original_unchanged': True,
        'original_sha256': 'incorrect' if fault == 'original_sha' else file_sha(original),
        'recomputed_file': str(repeated), 'recomputed_sha256': file_sha(repeated), 'input_sha256': file_sha(original_input)})
    evidence_names = ['source-reviews/review.json', 'trend-research.json', 'trend-offline-verification.json']
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}],
        'implementation_sha256': {name: file_sha(tmp_path / name) for name in code_paths},
        'evidence_sha256': {name: file_sha(directory / name) for name in evidence_names}})
    write_json(directory / 'baseline.json', {'published': {}})
    monkeypatch.setattr(archive_module, 'Store', lambda root: type('StoreStub', (), {'published': lambda self: {}})())
    protected = []
    monkeypatch.setattr(archive_module, 'protect', lambda *args: protected.append(True) or {})
    monkeypatch.setattr(archive_module, 'archive', lambda *args: {'output': 'checkpoint'})
    if fault == 'changed_input': repeated_input.write_bytes(b'tampered')
    elif fault == 'missing_input': repeated_input.unlink()
    with pytest.raises((ValueError, FileNotFoundError), match='Recomputed input|Original research|repeated.parquet'):
        archive_module.finish('data', directory)
    assert not protected
    assert not (directory / 'implementation-final').exists()
    assert not Path('docs/handoff/2026-10-05-batch13-verification.json').exists()
