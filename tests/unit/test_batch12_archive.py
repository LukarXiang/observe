from pathlib import Path

import pytest

from observe.runs import file_sha, write_json
import scripts.review_strategy_batch12 as archive_module


@pytest.fixture
def archive_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    helper = tmp_path / 'scripts/review_strategy_batch12.py'
    catalog = tmp_path / 'src/observe/strategy_catalog.py'
    for path in (helper, catalog):
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b'preserved implementation\n')
    monkeypatch.setattr(archive_module, '__file__', str(helper))
    directory = tmp_path / 'batch'; directory.mkdir()
    api = directory / 'existing-apis.json'; api.write_bytes(b'[]\n')
    write_json(directory / 'source-reviews/review.json', {'sources': [], 'api_evidence_sha256': file_sha(api)})
    write_json(directory / 'probe-results.json', {'results': [
        {'endpoint': name, 'evidence_files': [], 'result': {}} for name in archive_module.PROBES]})
    official = directory / 'official'; official.write_bytes(b'official documentation\n')
    write_json(directory / 'input-analysis.json', {'official_get_bars': {
        'copy': str(official), 'original_file': str(official), 'sha256': file_sha(official),
        'excerpt': str(official), 'excerpt_sha256': file_sha(official)}})
    write_json(directory / 'baseline.json', {'published': {'batch_id': 'unchanged'}})
    monkeypatch.setattr(archive_module, 'Store', lambda root: type('StoreStub', (), {
        'published': lambda self: {'batch_id': 'unchanged'}})())
    protected = []
    monkeypatch.setattr(archive_module, 'protect', lambda *args: protected.append(True) or {})
    monkeypatch.setattr(archive_module, 'archive', lambda *args: {'output': 'checkpoint'})

    def checks(extra=None, absolute=True):
        names = [helper, catalog]
        if extra is not None: names.append(extra)
        write_json(directory / 'checks.json', {'commands': [{'returncode': 0}],
            'implementation_sha256': {str(p if absolute or p == extra else p.relative_to(tmp_path)): file_sha(p)
                                      for p in names}})
    return directory, helper, catalog, api, protected, checks


@pytest.mark.parametrize('absolute', [True, False])
def test_implementation_paths_freeze_inside_archive(archive_inputs, absolute):
    directory, helper, catalog, _, protected, checks = archive_inputs
    checks(absolute=absolute)
    originals = {path: path.read_bytes() for path in (helper, catalog)}
    archive_module.finish('data', directory)
    for path, content in originals.items():
        assert path.read_bytes() == content
        assert (directory / 'implementation-final' / path.relative_to(Path.cwd())).read_bytes() == content
    assert protected == [True]


@pytest.mark.parametrize('relative', [True, False])
def test_outside_implementation_refused_before_mutation(archive_inputs, tmp_path, relative):
    directory, _, _, _, protected, checks = archive_inputs
    outside = tmp_path.parent / f'{tmp_path.name}-outside.py'; outside.write_bytes(b'outside\n')
    checks(extra=Path('..') / outside.name if relative else outside)
    with pytest.raises(ValueError, match='outside repository'):
        archive_module.finish('data', directory)
    assert not protected
    assert not (directory / 'implementation-final').exists()
    assert outside.read_bytes() == b'outside\n'


@pytest.mark.parametrize('missing', [True, False])
def test_api_evidence_required_before_acceptance(archive_inputs, missing):
    directory, _, _, api, protected, checks = archive_inputs
    checks()
    if missing: api.unlink()
    else: api.write_bytes(b'changed\n')
    with pytest.raises((ValueError, FileNotFoundError), match='API evidence|existing-apis'):
        archive_module.finish('data', directory)
    assert not protected
    assert not (directory / 'implementation-final').exists()
