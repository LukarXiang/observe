from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from observe.runs import file_sha, write_json
from observe.strategies import StrategyConfig
import scripts.run_strategy_batch15 as batch
from scripts.probe_strategy_batch15 import MISSING, custody, probe
from scripts.run_strategy_batch15 import merge_year, publication_references


def test_new_year_retains_old_cells_and_rejects_duplicate_or_overlapping_raw_rows():
    old = pd.DataFrame({'date': [1, 2], 'instrument': ['old', 'old'], 'close': [3., float('nan')]})
    incoming = pd.DataFrame({'date': [1, 2], 'instrument': ['new', 'new'], 'close': [4., 5.]})
    merged = merge_year(old, incoming)
    pd.testing.assert_frame_equal(merged[merged.instrument.eq('old')].reset_index(drop=True), old)
    with pytest.raises(ValueError, match='overlaps'): merge_year(old, old)
    with pytest.raises(ValueError, match='duplicates'): merge_year(old, pd.concat([incoming, incoming]))


@pytest.mark.parametrize('fault', ['other_year', 'index', 'year_keys', 'one_year'])
def test_history_publication_cannot_replace_unapproved_tables_or_years(fault):
    old = {'tables': {'bars_1d': {y: {'file': y} for y in ('2018', '2019', '2020', '2021')}, 'index_1d': {'all': {'file': 'index'}}}}
    new = deepcopy(old)
    for year in ('2019', '2020'): new['tables']['bars_1d'][year]['file'] += '-expanded'
    assert publication_references(old, new) == {'2019', '2020'}
    if fault == 'other_year': new['tables']['bars_1d']['2018']['file'] = 'changed'
    elif fault == 'index': new['tables']['index_1d']['all']['file'] = 'changed'
    elif fault == 'year_keys': new['tables']['bars_1d'].pop('2021')
    else: new['tables']['bars_1d']['2020'] = old['tables']['bars_1d']['2020']
    with pytest.raises(ValueError): publication_references(old, new)


def test_probe_custody_checks_sources_and_never_overwrites_interrupted_api_evidence(tmp_path, monkeypatch):
    import scripts.probe_strategy_batch15 as probes
    source = tmp_path / 'source.txt'; source.write_bytes(b'source')
    config = tmp_path / 'original.yaml'; config.write_bytes(b'config')
    monkeypatch.setattr(probes, 'CONFIG', config)
    write_json(tmp_path / 'precheck.json', {'config_sha256': file_sha(config), 'source_file': str(source), 'source_sha256': file_sha(source)})
    api = tmp_path / 'existing-api.json'; api.write_bytes(b'interrupted original API evidence')
    with pytest.raises(FileExistsError): probe(tmp_path, tmp_path)
    assert api.read_bytes() == b'interrupted original API evidence'
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'): custody(tmp_path)
    assert len(MISSING) == 39 and len(set(MISSING)) == 39


@pytest.mark.parametrize('fault', ['cash', 'slippage', 'participation', 'snapshot', 'start'])
def test_extended_config_preserves_approved_economics_and_binds_interval(tmp_path, monkeypatch, fault):
    original = yaml.safe_load(Path('configs/strategies/rsi_slots_corrected_batch7.yaml').read_text())
    original_path = tmp_path / 'original.yaml'; original_path.write_text(yaml.safe_dump(original), encoding='utf-8')
    monkeypatch.setattr(batch, 'CONFIG', original_path)
    approved = {'source_sha256': 'source', 'config_sha256': file_sha(original_path)}
    monkeypatch.setattr(batch, 'custody', lambda directory: approved)
    config = StrategyConfig.model_validate(original).model_dump(mode='json')
    config.update(snapshot='new', start='2019-09-18', cache=False)
    if fault == 'cash': config['initial_cash'] *= 2
    elif fault == 'slippage': config['execution']['slippage'] *= 2
    elif fault == 'participation': config['portfolio']['participation'] *= 2
    elif fault == 'snapshot': config['snapshot'] = 'wrong'
    else: config['start'] = '2021-04-01'
    path = tmp_path / 'extended.yaml'; path.write_text(yaml.safe_dump(config), encoding='utf-8')
    write_json(tmp_path / 'prepared-config.json', {'config_file': str(path), 'config_sha256': file_sha(path),
        'source_sha256': 'source', 'original_config_sha256': approved['config_sha256'], 'snapshot': 'new', 'start': '2019-09-18', 'end': config['end']})
    with pytest.raises(ValueError, match='economic config differs|interval differs'): batch.config_record(tmp_path)


@pytest.mark.parametrize('fault', ['missing_baseline', 'changed_baseline'])
def test_final_checks_must_bind_full_protection_baseline(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path); directory = tmp_path / 'stage'; directory.mkdir()
    monkeypatch.setattr(batch, 'FILES', {'implementation.py'})
    code = tmp_path / 'implementation.py'; code.write_bytes(b'checked code')
    for name in batch.CORE | {'baseline.json'}: write_json(directory / name, {})
    evidence = {p.name: file_sha(p) for p in directory.iterdir()}
    if fault == 'missing_baseline': evidence.pop('baseline.json')
    else: write_json(directory / 'baseline.json', {'controls': {}, 'partitions': {}})
    write_json(directory / 'checks.json', {'commands': [{'returncode': 0}], 'implementation_sha256': {'implementation.py': file_sha(code)},
        'evidence_sha256': evidence})
    def too_late(*args): raise AssertionError('Missing baseline proof reached downstream input loading')
    monkeypatch.setattr(batch, 'analyzed_inputs', too_late)
    with pytest.raises(ValueError, match='sets incomplete|evidence changed'): batch.finish(tmp_path, directory)


@pytest.mark.parametrize('fault', ['old_snapshot', 'wrong_session_count'])
def test_extension_must_use_publication_snapshot_and_actual_calendar_count(fault):
    days = list(pd.bdate_range('2019-09-18', periods=10).date)
    record = {'snapshot': 'new', 'start': str(days[0]), 'end': str(days[-1]), 'sessions': 10}
    if fault == 'old_snapshot': record['snapshot'] = 'old-self-consistent-snapshot'
    else: record['sessions'] = 11
    with pytest.raises(ValueError, match='publication snapshot differs|session count differs'):
        batch.check_prepared_interval(record, {'snapshot': 'new'}, days)
