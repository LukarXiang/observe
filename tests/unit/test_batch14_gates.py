from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts.probe_strategy_batch14 import checked_artifacts, probe
from scripts.run_strategy_batch14 import check_references, config_record, retain_old_rows
import scripts.run_strategy_batch14 as batch14


@pytest.mark.parametrize('fault', ['old_bars', 'old_factors', 'new_2021', 'new_nonyear'])
def test_historical_extension_rejects_old_reference_changes_and_overlapping_keys(fault):
    old = {'tables': {'bars_1d': {'2021': {'file': 'old'}}, 'adj_factors': {'all': {'file': 'factors'}},
                      'calendar': {'all': {'file': 'cal'}}, 'index_1d': {'all': {'file': 'index'}}}}
    current = deepcopy(old)
    if fault == 'old_bars': current['tables']['bars_1d']['2021']['file'] = 'changed'
    elif fault == 'old_factors': current['tables']['adj_factors']['all']['file'] = 'changed'
    elif fault == 'new_2021': current['tables']['bars_1d']['2022'] = {'file': 'new'}
    else: current['tables']['bars_1d']['prefix'] = {'file': 'new'}
    with pytest.raises(ValueError): check_references(old, current)


def test_disjoint_historical_entries_allowed_and_old_cells_protected():
    old = {'tables': {'bars_1d': {'2021': {'file': 'old'}}}}; current = deepcopy(old)
    current['tables']['bars_1d']['2005'] = {'file': 'new'}
    assert check_references(old, current) == ['2005']
    original = pd.DataFrame({'date': [2], 'close': [10.]})
    retained = pd.DataFrame({'date': [1, 2], 'close': [8., 10.]})
    retain_old_rows(original, retained, ['date'])
    retained.loc[1, 'close'] = 11.
    with pytest.raises(AssertionError): retain_old_rows(original, retained, ['date'])


@pytest.mark.parametrize('fault', ['api', 'raw', 'retry_binding'])
def test_probe_evidence_tampering_is_refused(tmp_path, fault):
    api = tmp_path / 'api.json'; api.write_bytes(b'api')
    raw = tmp_path / 'raw.parquet'; pd.DataFrame({'date': ['2005-01-05']}).to_parquet(raw)
    probes = tmp_path / 'probe-results.json'
    write_json(probes, {'api_file': str(api), 'api_sha256': file_sha(api), 'results': [{'result': {'files': [
        {'dataset': 'calendar', 'file': str(raw), 'sha256': file_sha(raw), 'rows': 1, 'columns': ['date']} ]}}]})
    if fault == 'api': api.write_bytes(b'changed')
    elif fault == 'raw': pd.DataFrame({'date': ['2006-01-05']}).to_parquet(raw)
    else: write_json(tmp_path / 'retry-results.json', {'original_file': str(probes), 'original_sha256': 'wrong', 'results': []})
    with pytest.raises(ValueError, match='evidence changed|Raw probe changed'): checked_artifacts(tmp_path)


def test_retry_keeps_first_pass_and_selects_new_raw_evidence(tmp_path):
    api = tmp_path / 'api.json'; api.write_bytes(b'api')
    probes = tmp_path / 'probe-results.json'
    write_json(probes, {'api_file': str(api), 'api_sha256': file_sha(api), 'results': [{'result': {'status': 'failed', 'files': []}}]})
    before = probes.read_bytes(); raw = tmp_path / 'raw.parquet'
    pd.DataFrame({'close': [10.]}).to_parquet(raw)
    write_json(tmp_path / 'retry-results.json', {'original_file': str(probes), 'original_sha256': file_sha(probes),
        'results': [{'result': {'files': [{'dataset': 'bars', 'file': str(raw), 'sha256': file_sha(raw), 'rows': 1, 'columns': ['close']}]}}]})
    artifacts, evidence = checked_artifacts(tmp_path)
    assert artifacts['bars'][1].close.tolist() == [10.] and len(evidence) == 2
    assert Path(artifacts['bars'][0]['probe_file']).name == 'retry-results.json'
    assert probes.read_bytes() == before


@pytest.mark.parametrize('fault', ['source', 'original_config'])
def test_prepared_source_and_original_config_changes_block_execution(tmp_path, fault):
    source = tmp_path / 'source.txt'; source.write_bytes(b'original source')
    original = tmp_path / 'original.yaml'; original.write_bytes(b'original config')
    prepared = tmp_path / 'prepared.yaml'; prepared.write_text(f'source_path: {source}\n', encoding='utf-8')
    write_json(tmp_path / 'prepared-configs.json', {'configs': [{'label': 'test', 'config_file': str(prepared),
        'config_sha256': file_sha(prepared), 'source_sha256': file_sha(source), 'original_config_file': str(original),
        'original_config_sha256': file_sha(original)}]})
    (source if fault == 'source' else original).write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed|Original config changed'):
        config_record(tmp_path, 'test')


def test_interrupted_api_evidence_is_never_overwritten(tmp_path):
    write_json(tmp_path / 'baseline.json', {})
    api = tmp_path / 'existing-api.json'; api.write_bytes(b'original API evidence')
    with pytest.raises(FileExistsError): probe(tmp_path, tmp_path)
    assert api.read_bytes() == b'original API evidence'


@pytest.mark.parametrize('fault', ['empty_implementation', 'missing_implementation', 'empty_evidence', 'missing_evidence'])
def test_finish_requires_complete_checked_file_sets(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path); directory = tmp_path / 'stage'; directory.mkdir()
    implementations = {}
    for name in batch14.IMPLEMENTATION_FILES:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'implementation')
        implementations[name] = file_sha(path)
    evidence = {}
    for name in batch14.CORE_EVIDENCE:
        path = directory / name; path.write_bytes(b'{}'); evidence[name] = file_sha(path)
    if fault == 'empty_implementation': implementations = {}
    elif fault == 'missing_implementation': implementations.pop('src/observe/research.py')
    elif fault == 'empty_evidence': evidence = {}
    else: evidence.pop('publication.json')
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}],
        'implementation_sha256': implementations, 'evidence_sha256': evidence})
    protected = []; monkeypatch.setattr(batch14, 'protect', lambda *a: protected.append(True))
    with pytest.raises(ValueError, match='Missing required'):
        batch14.finish(tmp_path, directory)
    assert not protected and not (directory / 'implementation-final').exists()


@pytest.fixture
def final_archive_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = tmp_path / 'stage'; directory.mkdir()
    root = tmp_path / 'data'
    for rid in ('original', 'repeated'):
        folder = root / 'runs' / rid; folder.mkdir(parents=True)
        write_json(folder / 'status.json', {'run_id': rid, 'status': 'success_limited', 'kind': 'strategy'})
        write_json(folder / 'config.json', {'kind': 'strategy'})
        (folder / 'result.txt').write_bytes(b'original result')
        write_json(folder / 'manifest.json', {'run_id': rid, 'status': 'success_limited',
            'files': {name: file_sha(folder / name) for name in ('status.json', 'config.json', 'result.txt')}})
    implementations = {}
    for name in batch14.IMPLEMENTATION_FILES:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'implementation')
        implementations[name] = file_sha(path)
    original_cfg = tmp_path / 'original.yaml'; original_cfg.write_bytes(b'config')
    configs = [{'label': f'{label}-extended', 'original_config_file': str(original_cfg),
                'original_config_sha256': file_sha(original_cfg), 'source_sha256': 'source'} for label in batch14.CONFIGS]
    for name in batch14.CORE_EVIDENCE: write_json(directory / name, {})
    write_json(directory / 'publication.json', {'batch_id': 'batch', 'snapshot': 'snapshot'})
    write_json(directory / 'prepared-configs.json', {'configs': configs})
    write_json(directory / 'stable-configs.json', {'configs': []})
    run = {'status': 'success_limited', 'run_id': 'original'}
    verified = {'status': 'ok', 'run': run, 'original_unchanged': True, 'source_sha256': 'source',
        'reproduction': {'run_id': 'repeated', 'reproduction': {'result': 'match', 'differences': 0}},
        'hand_check': {}, 'report': {'period': {}, 'results': [], 'benchmark': []}}
    for record in configs:
        for suffix, doc in [('run', run), ('freeze', {}), ('verification', verified)]:
            write_json(directory / f'{record["label"]}-{suffix}.json', doc)
    evidence = {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}],
        'implementation_sha256': implementations, 'evidence_sha256': evidence})
    monkeypatch.setattr(batch14, 'checked_artifacts', lambda *a: ({}, []))
    monkeypatch.setattr(batch14, 'config_record', lambda *a: {})
    monkeypatch.setattr(batch14, 'Store', lambda *a: type('StoreStub', (), {
        'published': lambda self: {'batch_id': 'batch', 'tables': {}},
        'state': lambda self, snapshot: {'tables': {}}})())
    return root, directory


@pytest.mark.parametrize('tampered', ['original', 'repeated'])
def test_finish_rechecks_new_run_artifacts_after_prior_verification(final_archive_setup, tampered):
    root, directory = final_archive_setup
    (root / 'runs' / tampered / 'result.txt').write_bytes(b'tampered after verification')
    with pytest.raises(ValueError, match='Final original run integrity failed|Final reproduction integrity failed'):
        batch14.finish(root, directory)
    assert not (directory / 'implementation-final').exists()


@pytest.mark.parametrize('fault', ['claimed_reproduction', 'wrong_manifest', 'wrong_source', 'no_state_difference', 'wrong_output'])
def test_pending_numeric_policy_cannot_claim_verified_result(final_archive_setup, fault):
    root, directory = final_archive_setup
    (directory / 'ma5_ma10_long_batch4-extended-verification.json').unlink()
    run_path = root / 'runs/original'
    run = {'status': 'success_limited', 'run_id': 'original', 'output': str(run_path)}
    write_json(directory / 'ma5_ma10_long_batch4-extended-run.json', run)
    pending = {'status': 'blocked_pending_numeric_policy', 'run_id': 'original', 'source_sha256': 'source',
        'original_unchanged': True, 'reproduction_claimed': False, 'state_difference_count': 3,
        'original_manifest_sha256': file_sha(run_path / 'manifest.json')}
    if fault == 'claimed_reproduction': pending['reproduction_claimed'] = True
    elif fault == 'wrong_manifest': pending['original_manifest_sha256'] = 'wrong'
    elif fault == 'wrong_source': pending['source_sha256'] = 'wrong'
    elif fault == 'no_state_difference': pending['state_difference_count'] = 0
    else:
        other = root / 'runs/repeated'; run['output'] = str(other)
        pending['original_manifest_sha256'] = file_sha(other / 'manifest.json')
        write_json(directory / 'ma5_ma10_long_batch4-extended-run.json', run)
    write_json(directory / 'ma5-numeric-boundary-blocked.json', pending)
    checks = batch14.read(directory / 'checks.json')
    checks['evidence_sha256'] = {p.name: file_sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'checks.json'}
    write_json(directory / 'checks.json', checks)
    with pytest.raises(ValueError, match='Numeric diagnosis does not bind original run|Pending numeric run output mismatch'):
        batch14.finish(root, directory)
    assert not (directory / 'implementation-final').exists()


@pytest.mark.parametrize('fault', ['old_algorithm', 'cash', 'start', 'missing_decimal', 'partial_decimal'])
def test_stable_proof_must_bind_actual_config_and_complete_decimal_windows(tmp_path, fault):
    import yaml
    from observe.strategies import StrategyConfig
    from tests.integration.test_multi_ma_strategy import fixture
    _, config = fixture(tmp_path)
    config['parameters']['mean_algorithm'] = 'window_fsum_v1'
    expected = StrategyConfig.model_validate(config).model_dump(mode='json')
    config_file = tmp_path / 'stable.yaml'; config_file.write_text(yaml.safe_dump(expected), encoding='utf-8')
    actual = deepcopy(expected); proof = {'hand_check': {'sessions_checked': 138, 'exact_decimal_windows_checked': 552,
        'numerical_policy': 'window_fsum_v1'}}
    if fault == 'old_algorithm': actual['parameters'].pop('mean_algorithm')
    elif fault == 'cash': actual['initial_cash'] = 1_000_000
    elif fault == 'start': actual['start'] = '2021-03-01'
    elif fault == 'missing_decimal': proof['hand_check'].pop('exact_decimal_windows_checked')
    else: proof['hand_check']['exact_decimal_windows_checked'] = 4
    output = tmp_path / 'run'; output.mkdir(); write_json(output / 'config.json', {'config': actual})
    with pytest.raises(ValueError, match='Stable actual config differs|Stable Decimal coverage incomplete'):
        batch14.check_verified_stable({'config_file': str(config_file), 'sessions': 138}, output, proof)
