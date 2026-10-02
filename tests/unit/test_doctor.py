"""只读诊断：损坏输入、切分一致性、可选依赖和 CLI / API 同口径。"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from observe.api.app import create_app
from observe.cli import main
from observe.data.store import Store, _atomic_json, fingerprint
from observe.doctor import doctor
from observe.research import run_research
from tests.integration.helpers import tree_hash
from tests.integration.test_research import make, cfg


@pytest.fixture
def ready(tmp_path):
    sid, days, factors = make(tmp_path, n_days = 55)
    config = cfg(sid, factors); config['benchmarks'] = {'enabled': False}
    return tmp_path, sid, days, config


def check(result, name): return [c for c in result['checks'] if c['name'] == name]


def test_read_only_and_no_network_or_training(ready, monkeypatch):
    from observe.models import RidgeModel
    from observe.data.sources.baostock import BaoStock
    root, sid, days, config = ready; before = tree_hash(root)
    def forbidden(*args, **kwargs): raise AssertionError('预检不得联网或训练')
    monkeypatch.setattr(RidgeModel, 'fit', forbidden); monkeypatch.setattr(BaoStock, 'session', forbidden)
    result = doctor(root, config = config, verify_files = True)
    assert result['status'] == 'warning' and result['summary']['error'] == 0 and result['plan']['windows'] > 0
    assert result['snapshot_id'] == sid and result['tables']['bars_1d']['rows'] == 55 * 40
    assert check(result, 'calendar')[0]['missing_dates']  # 合成夹具仅含开市日，缺口不能假定为休市
    assert tree_hash(root) == before


def test_split_plan_agrees_with_research_pipeline(ready):
    root, sid, days, config = ready
    result = doctor(root, config = config)
    run = run_research(root, **{k: v for k, v in config.items() if k != 'benchmarks'})
    from pathlib import Path
    plan = pd.read_parquet(Path(run['output']) / 'split_plan.parquet')
    assert result['plan']['windows'] == len(plan)
    assert result['plan']['holdout_start'] == str(plan.holdout_start.iloc[0])
    assert result['plan']['test_start'] == str(plan.test_start.min())
    assert result['plan']['test_end'] == str(plan.test_end.max())


def test_missing_root_is_not_created(tmp_path):
    root = tmp_path / 'missing'; result = doctor(root)
    assert result['status'] == 'error' and not root.exists()
    assert check(result, 'snapshot')[0]['status'] == 'error'


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'columns', 'rows', 'escape'])
def test_partition_problems_are_reported(ready, tmp_path, kind):
    root, sid, days, config = ready; store = Store(root); state = store.state(sid)
    part = next(iter(state['tables']['bars_1d'].values())); path = root / part['file']
    if kind == 'missing': path.unlink()
    elif kind == 'corrupt': path.write_bytes(b'not parquet')
    elif kind == 'columns': pd.read_parquet(path).drop(columns = 'open').to_parquet(path, index = False)
    elif kind == 'rows': part['rows'] += 1
    else: part['file'] = '../outside.parquet'
    _atomic_json(root / 'snapshots' / f'{sid}.json', state)
    before = tree_hash(root); result = doctor(root, config = config)
    assert result['status'] == 'error'
    assert check(result, 'partitions:bars_1d')[0]['status'] == 'error'
    assert result['plan'] is None and tree_hash(root) == before


def test_deep_verification_detects_same_size_content_change(ready):
    root, sid, days, config = ready; state = Store(root).state(sid)
    path = root / next(iter(state['tables']['bars_1d'].values()))['file']
    bars = pd.read_parquet(path); bars.loc[0, 'close'] += .01; bars.to_parquet(path, index = False)
    quick = doctor(root, config = config)
    assert check(quick, 'partitions:bars_1d')[0]['status'] == 'ok'
    deep = doctor(root, config = config, verify_files = True)
    assert deep['status'] == 'error' and '指纹' in check(deep, 'partitions:bars_1d')[0]['detail']


def test_insufficient_holdout_and_requested_history_are_errors(ready):
    root, sid, days, config = ready
    short = doctor(root, config = {**config, 'split': {'train': 20, 'valid': 10, 'test': 5, 'holdout': 60}})
    assert short['status'] == 'error' and '最终留出' in check(short, 'research_plan')[0]['detail']
    early = doctor(root, config = {**config, 'start': '2007-01-01'})
    assert check(early, 'requested_range')[0]['status'] == 'error'


def test_empty_or_unknown_calendar_is_not_treated_as_closed(ready):
    root, sid, days, config = ready; store = Store(root); state = store.state(sid)
    path = root / state['tables']['calendar']['all']['file']
    cal = pd.read_parquet(path); cal['is_open'] = cal.is_open.astype(object); cal.loc[0, 'is_open'] = None
    cal.to_parquet(path, index = False)
    result = doctor(root, config = config)
    assert check(result, 'calendar')[0]['status'] == 'error' and result['plan'] is None


def test_corrupt_config_manifest_and_snapshot_identity_are_errors(ready):
    root, sid, days, config = ready
    path = root / 'invalid.yaml'; path.write_text('split: [')
    assert check(doctor(root, sid, path), 'config')[0]['status'] == 'error'
    assert check(doctor(root, config = {**config, 'unknown': 3}), 'config')[0]['status'] == 'error'
    state_path = root / 'snapshots' / f'{sid}.json'; state = Store(root).state(sid)
    state['snapshot_id'] = 'wrong'; _atomic_json(state_path, state)
    assert check(doctor(root, sid), 'snapshot')[0]['status'] == 'error'
    state_path.write_text('{')
    assert doctor(root, sid)['status'] == 'error'
    assert doctor(root, '../PUBLISHED')['status'] == 'error'


def test_optional_lightgbm_is_checked_only_when_requested(ready, monkeypatch):
    from observe import models
    root, sid, days, config = ready
    def unavailable(): raise RuntimeError('缺少 libomp')
    monkeypatch.setattr(models, 'lightgbm', unavailable)
    assert not check(doctor(root, config = config), 'lightgbm')
    result = doctor(root, config = {**config, 'models': {**config['models'], 'lgbm': [{}]}})
    assert check(result, 'lightgbm')[0]['status'] == 'error' and 'libomp' in check(result, 'lightgbm')[0]['detail']


def test_missing_benchmark_is_a_limitation(ready):
    root, sid, days, config = ready
    result = doctor(root, config = {**config, 'benchmarks': {'enabled': True}})
    assert result['summary']['error'] == 0 and check(result, 'price_benchmark')[0]['status'] == 'warning'
    assert check(result, 'price_benchmark')[0]['missing_return_days'] > 0


def test_cli_api_and_core_match_without_changing_data(ready, capsys):
    root, sid, days, config = ready; path = root / 'doctor.json'; path.write_text(json.dumps(config))
    client = TestClient(create_app(root)); before = tree_hash(root)
    expected = doctor(root, config = config)
    main(['--root', str(root), 'doctor', '--config', str(path)])
    assert json.loads(capsys.readouterr().out) == expected
    assert client.post('/api/doctor', json = {'config': config}).json() == expected
    assert client.get('/api/doctor', params = {'snapshot': sid}).json() == doctor(root, sid)
    assert client.post('/api/doctor', json = {'unknown': True}).status_code == 422
    with pytest.raises(SystemExit) as error: main(['--root', str(root), 'doctor', '--snapshot', 'missing'])
    assert error.value.code == 1 and json.loads(capsys.readouterr().out)['status'] == 'error'
    assert tree_hash(root) == before


def test_absent_core_dependency_is_reported_before_data_access(tmp_path, monkeypatch):
    import importlib.metadata
    original = importlib.metadata.version
    def version(name):
        if name == 'pandas': raise importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(importlib.metadata, 'version', version)
    root = tmp_path / 'unused'; result = doctor(root)
    assert result['status'] == 'error' and check(result, 'dependencies')[0]['versions']['pandas'] is None
    assert not root.exists()


def test_baseline_minute_contract_and_missing_pool(ready):
    root, sid, days, config = ready
    missing = doctor(root, config = {**config, 'models': {**config['models'], 'baseline_factor': 'absent'}})
    assert check(missing, 'factors')[0]['status'] == 'error'
    minute = doctor(root, config = {**config, 'minute_pool': True})
    assert check(minute, 'research_plan')[0]['status'] == 'error' and 'minute_universe' in check(minute, 'research_plan')[0]['detail']


def test_missing_session_detected_without_full_price_read(ready):
    root, sid, days, config = ready; store = Store(root); state = store.state(sid)
    part = next(iter(state['tables']['bars_1d'].values())); path = root / part['file']
    bars = pd.read_parquet(path); bars = bars[bars.date != days[15]]; bars.to_parquet(path, index = False)
    part['rows'] = len(bars); _atomic_json(root / 'snapshots' / f'{sid}.json', state)
    result = doctor(root, config = config)
    assert check(result, 'daily_coverage')[0]['missing_dates'] == [str(days[15])]
    assert result['status'] == 'error'


def test_broken_native_dependency_returns_structured_report(tmp_path, monkeypatch):
    import importlib
    original = importlib.import_module
    def load(name, *args, **kwargs):
        if name == 'numpy': raise OSError('native library unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', load)
    result = doctor(tmp_path)
    assert result['status'] == 'error'
    assert check(result, 'runtime_imports')[0]['failures'] == [{'package': 'numpy', 'error': 'OSError: native library unavailable'}]


def test_all_bad_partitions_are_listed_and_not_silently_skipped(ready):
    root, sid, days, config = ready; store = Store(root); state = store.state(sid)
    state['tables']['bars_1d'] = {'first': {'file': 'missing1.parquet', 'rows': 1, 'sha': 'a'}, 'second': {'file': 'missing2.parquet', 'rows': 1, 'sha': 'b'}}
    _atomic_json(root / 'snapshots' / f'{sid}.json', state)
    result = doctor(root, config = config, verify_files = True)
    assert {x['partition'] for x in check(result, 'partitions:bars_1d')[0]['failures']} == {'first', 'second'}
    assert result['tables']['bars_1d'] == {'partitions': 2, 'verified_partitions': 0, 'rows': None}


def test_legacy_timestamp_unit_compatibility_never_truncates_time(ready):
    root, sid, days, config = ready; store = Store(root); state = store.state(sid)
    frame = pd.DataFrame({'bar_end': pd.to_datetime(['2023-01-02 09:35:00']).as_unit('s'), 'instrument': ['600001.SH']})
    path = root / 'legacy.parquet'; frame.to_parquet(path, index = False)
    assert str(pd.read_parquet(path).bar_end.dtype) == 'datetime64[ms]'
    state['tables']['bars_5m'] = {'202301': {'file': 'legacy.parquet', 'rows': 1, 'sha': fingerprint(frame)}}
    _atomic_json(root / 'snapshots' / f'{sid}.json', state); before = tree_hash(root)
    result = doctor(root, config = config, verify_files = True)
    item = check(result, 'partitions:bars_5m')[0]
    assert item['status'] == 'warning' and item['representation_adjustments'][0]['columns'] == ['bar_end']
    assert tree_hash(root) == before
    altered = pd.read_parquet(path); altered['bar_end'] += pd.Timedelta(milliseconds = 1); altered.to_parquet(path, index = False)
    assert check(doctor(root, config = config, verify_files = True), 'partitions:bars_5m')[0]['status'] == 'error'


def test_new_partition_timestamp_fingerprint_matches_serialized_content(tmp_path):
    store = Store(tmp_path)
    frame = pd.DataFrame({'bar_end': pd.to_datetime(['2023-01-02 09:35:00']).as_unit('s'), 'instrument': ['600001.SH']})
    entry = store.write_partition('bars_5m', '202301', frame)
    loaded = pd.read_parquet(tmp_path / entry['file'])
    assert entry['sha'] == fingerprint(loaded)
    assert store.write_partition('bars_5m', '202301', loaded) == entry
    assert store.write_partition('bars_5m', '202301', frame) == entry
    assert str(frame.bar_end.dtype) == 'datetime64[s]'  # 调用者原表不变
