from pathlib import Path

import pytest

from observe.runs import file_sha, write_json
from scripts.review_strategy_batch17 import verify_offline_scan


@pytest.mark.parametrize('fault', ['missing_pair', 'duplicate_pair', 'unexpected_path', 'changed_original', 'changed_recomputed', 'changed_scan'])
def test_offline_receipt_rejects_missing_changed_or_redirected_catalog_files(tmp_path, fault):
    directory = tmp_path / 'stage'; directory.mkdir()
    original = tmp_path / 'catalog'; original.mkdir()
    repeated = directory / 'catalog-offline/catalog/strategies/test-catalog'; repeated.mkdir(parents=True)
    scanned = {'catalog': {'output': str(original), 'catalog_id': 'test-catalog'}}
    write_json(directory / 'scan.json', scanned)
    pairs = []
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        before = original / name; after = repeated / name
        before.write_bytes(b'frozen catalog'); after.write_bytes(before.read_bytes())
        pairs.append({'original_file': str(before), 'recomputed_file': str(after), 'sha256': file_sha(before)})
    receipt = {'status': 'match', 'differences': 0, 'scan_sha256': file_sha(directory / 'scan.json'), 'files': pairs}
    verify_offline_scan(directory, scanned, receipt)
    if fault == 'missing_pair': pairs.pop()
    elif fault == 'duplicate_pair': pairs[-1] = pairs[0]
    elif fault == 'unexpected_path': pairs[1]['recomputed_file'] = str(original / 'summary.json')
    elif fault == 'changed_original': Path(pairs[1]['original_file']).write_bytes(b'changed summary')
    elif fault == 'changed_recomputed': Path(pairs[-1]['recomputed_file']).write_bytes(b'changed parquet')
    else: write_json(directory / 'scan.json', {'catalog': {'catalog_id': 'changed'}})
    with pytest.raises(ValueError, match='file set incomplete|file path differs|file changed|verification differs'):
        verify_offline_scan(directory, scanned, receipt)
