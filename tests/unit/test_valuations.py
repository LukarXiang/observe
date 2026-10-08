import json

import numpy as np
import pandas as pd
import pytest

from observe.data.store import Store
from observe.data.valuations import FIELDS, attach_valuations, import_valuations, load_valuations, normalize, query_valuations, valuation_lookback
from observe.execution import InputBlocked
from observe.factors.expr import parse
from observe.runs import file_sha, write_json


def bind_raw(store, frames):
    entries = []
    folder = store.root / 'raw/baostock/daily_market'; folder.mkdir(parents=True, exist_ok=True)
    for day, frame in frames:
        path = folder / f'{day}.parquet'; frame.to_parquet(path, index=False)
        entries.append({'file': str(path), 'date': str(day), 'sha256': file_sha(path), 'raw_rows': len(frame), 'columns': list(frame)})
    manifest = store.root / 'binding.json'; write_json(manifest, {'source_files': entries, 'missing_dates': []})
    return manifest, file_sha(manifest)


def fixture(root):
    store = Store(root); days = list(pd.bdate_range('2024-01-02', periods=3).date)
    calendar = pd.DataFrame({'date': days, 'is_open': True})
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}}))
    frames = [(day, pd.DataFrame({'date': [str(day)] * 3, 'code': ['sh.600001', 'sz.000001', 'sh.600002'],
        'tradestatus': ['1', '1', '0'], 'psTTM': ['-2', '', '8'], 'pcfNcfTTM': ['11', '-9', '']})) for day in days]
    return store, days, frames


def test_full_recovery_separate_fields_immutable_receipt_and_strict_default(tmp_path):
    store, days, frames = fixture(tmp_path); old = store.snapshot(); manifest, sha = bind_raw(store, frames)
    result = import_valuations(store.root, manifest, sha); sid = store.snapshot()
    assert result['status'] == 'published' and result['rows'] == 9
    assert result['missing'] == {'ps_ttm': 3, 'pcf_ncf_ttm': 3} and result['negative'] == {'ps_ttm': 3, 'pcf_ncf_ttm': 3}
    values = store.load('valuations_1d')
    one = values[values.instrument.eq('600001.SH')]
    assert one.ps_ttm.tolist() == [-2] * 3 and one.pcf_ncf_ttm.tolist() == [11] * 3
    assert not values.strict_usable.any() and values.source_manifest_sha256.eq(sha).all()
    assert 'valuations_1d' not in store.state(old)['tables']
    before = store.published_path.read_bytes(); receipt_before = open(result['receipt'], 'rb').read()
    assert import_valuations(store.root, manifest, sha)['status'] == 'unchanged'
    assert before == store.published_path.read_bytes() and receipt_before == open(result['receipt'], 'rb').read()
    strict = query_valuations(store.root, sid, days[0], days[-1], ['600001.SH'])
    assert strict['data'] == [] and strict['coverage']['strict_excluded'] == 3
    final = query_valuations(store.root, sid, days[0], days[-1], ['600001.SH'], mode='provider_final')
    assert final['data'][0]['ps_ttm'] == -2 and final['data'][0]['pcf_ncf_ttm'] == 11
    with pytest.raises(InputBlocked, match='valuation_policy'):
        load_valuations(store, store.state(sid), days[0], days[-1], FIELDS, None)
    with pytest.raises(InputBlocked, match='valuation_missing'):
        load_valuations(store, store.state(old), days[0], days[-1], FIELDS, 'provider_final')


@pytest.mark.parametrize('kind', ['numeric', 'infinite', 'duplicate', 'code', 'date', 'status', 'schema'])
def test_invalid_late_source_never_publishes(tmp_path, kind):
    store, days, frames = fixture(tmp_path); raw = frames[-1][1]
    if kind == 'numeric': raw.loc[0, 'psTTM'] = 'broken'
    if kind == 'infinite': raw.loc[0, 'pcfNcfTTM'] = 'inf'
    if kind == 'duplicate': raw.loc[1, 'code'] = raw.loc[0, 'code']
    if kind == 'code': raw.loc[0, 'code'] = '600001'
    if kind == 'date': raw.loc[0, 'date'] = str(days[0])
    if kind == 'status': raw.loc[0, 'tradestatus'] = ''
    if kind == 'schema': raw.drop(columns=['pcfNcfTTM'], inplace=True)
    manifest, sha = bind_raw(store, frames); before = store.published_path.read_bytes()
    with pytest.raises(ValueError): import_valuations(store.root, manifest, sha)
    assert before == store.published_path.read_bytes()
    assert list((store.root / 'raw/valuation_imports').glob('*/attempt-*/failed.json'))
    assert not list((store.root / 'raw/valuation_imports').glob('*/receipt.json'))


def test_bound_sources_manifest_profiles_and_missing_calendar_rejected(tmp_path):
    store, days, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames)
    with pytest.raises(ValueError, match='Manifest SHA256'): import_valuations(store.root, manifest, '0' * 64)
    raw = frames[0][1]; raw.loc[0, 'psTTM'] = '3'; raw.to_parquet(store.root / f'raw/baostock/daily_market/{days[0]}.parquet', index=False)
    before = store.published_path.read_bytes()
    with pytest.raises(ValueError, match='Raw SHA256'): import_valuations(store.root, manifest, sha)
    assert before == store.published_path.read_bytes()
    manifest, sha = bind_raw(store, frames[:1] + frames[-1:])
    with pytest.raises(ValueError, match='trading calendar'): import_valuations(store.root, manifest, sha)


def test_mutation_after_import_rejected_without_rewriting_receipt(tmp_path):
    store, days, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames)
    result = import_valuations(store.root, manifest, sha); before = store.published_path.read_bytes()
    receipt = open(result['receipt'], 'rb').read()
    path = store.root / f'raw/baostock/daily_market/{days[0]}.parquet'
    frame = pd.read_parquet(path); frame.loc[0, 'psTTM'] = '9'; frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Raw SHA256'): import_valuations(store.root, manifest, sha)
    assert before == store.published_path.read_bytes() and receipt == open(result['receipt'], 'rb').read()


def test_interrupted_publication_resumes_prepared_receipt(tmp_path, monkeypatch):
    store, _, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames); before = store.published_path.read_bytes()
    original = Store.publish
    with monkeypatch.context() as context:
        def fail(*args): raise RuntimeError('interrupted publish')
        context.setattr(Store, 'publish', fail)
        with pytest.raises(RuntimeError, match='interrupted'): import_valuations(store.root, manifest, sha)
    assert before == store.published_path.read_bytes() and Store.publish is original
    receipt = next((store.root / 'raw/valuation_imports').glob('*/receipt.json')); frozen = receipt.read_bytes()
    result = import_valuations(store.root, manifest, sha)
    assert result['status'] == 'unchanged' and len(store.load('valuations_1d')) == 9 and receipt.read_bytes() == frozen


def test_sidecar_missing_rows_blanks_and_pause_mask(tmp_path):
    store, days, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames); import_valuations(store.root, manifest, sha)
    values, _ = load_valuations(store, store.published(), days[0], days[-1], FIELDS, 'provider_final')
    bars = values[['date', 'instrument', 'is_trading']].copy()
    merged, detail = attach_valuations(bars, values, FIELDS, days[0], days[-1])
    assert merged.loc[~merged.is_trading, list(FIELDS)].isna().all().all()
    assert merged.loc[merged.instrument.eq('000001.SZ'), 'ps_ttm'].isna().all()
    assert detail['missing_values']['ps_ttm'] == 3
    with pytest.raises(InputBlocked, match='valuation_rows_missing'):
        attach_valuations(bars, values[values.instrument.ne('600001.SH')], FIELDS, days[0], days[-1])
    changed = values.copy(); changed.loc[changed.instrument.eq('600001.SH'), 'is_trading'] = False
    with pytest.raises(InputBlocked, match='valuation_trading_mismatch'):
        attach_valuations(bars, changed, FIELDS, days[0], days[-1])
    with pytest.raises(ValueError, match='shadows'):
        attach_valuations(merged, values, FIELDS, days[0], days[-1])


def test_default_query_validates_coverage_and_partition_identity(tmp_path):
    store, days, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames); import_valuations(store.root, manifest, sha)
    calendar = store.load('calendar'); extra = pd.Timestamp('2024-01-05').date()
    calendar.loc[len(calendar)] = [extra, True]
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}}))
    with pytest.raises(InputBlocked, match='valuation_coverage'):
        query_valuations(store.root, store.snapshot(), days[0], extra, ['600001.SH'])
    state = store.published(); path = store.root / state['tables']['valuations_1d']['202401']['file']
    changed = pd.read_parquet(path); changed.loc[0, 'ps_ttm'] = 99; changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='partition differs'):
        load_valuations(store, state, days[0], days[-1], FIELDS, 'provider_final')


def test_normalize_null_zero_signed_and_store_duplicates(tmp_path):
    day = pd.Timestamp('2024-01-02').date()
    raw = pd.DataFrame({'date': [str(day)] * 4, 'code': [f'sh.60000{k}' for k in range(1, 5)], 'tradestatus': '1',
        'psTTM': ['0', '-0.5', None, '  '], 'pcfNcfTTM': ['-1', '0', '2e3', None]})
    values = normalize(raw, 'a' * 64, 'b' * 64, day)
    assert values.ps_ttm.iloc[:2].tolist() == [0, -.5] and np.isnan(values.ps_ttm.iloc[2:]).all()
    assert values.pcf_ncf_ttm.iloc[:3].tolist() == [-1, 0, 2000]
    with pytest.raises(ValueError, match='禁止静默覆盖'):
        Store(tmp_path).write_partition('valuations_1d', '202401', pd.concat([values, values]))


def test_manifest_outside_archive_and_raw_profile_fail(tmp_path):
    store, _, frames = fixture(tmp_path); manifest, _ = bind_raw(store, frames)
    doc = json.loads(manifest.read_text()); doc['source_files'][0]['file'] = str(tmp_path / 'outside.parquet')
    write_json(manifest, doc)
    with pytest.raises(ValueError, match='outside'): import_valuations(store.root, manifest, file_sha(manifest))
    manifest, _ = bind_raw(store, frames); doc = json.loads(manifest.read_text()); doc['source_files'][-1]['raw_rows'] += 1
    write_json(manifest, doc)
    with pytest.raises(ValueError, match='profile'): import_valuations(store.root, manifest, file_sha(manifest))


def test_unused_partition_cannot_contribute_request_rows(tmp_path):
    store, days, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames); import_valuations(store.root, manifest, sha)
    state = store.published(); expected, used = load_valuations(store, state, days[0], days[-1], FIELDS, 'provider_final')
    extra = expected[expected.instrument.eq('600001.SH')].copy(); extra['instrument'] = '600099.SH'
    coverage = store.load('valuation_coverage'); coverage['rows'] = 1
    store.publish(store.write_batch({
        'valuations_1d': {'202402': store.write_partition('valuations_1d', '202402', extra)},
        'valuation_coverage': {'202402': store.write_partition('valuation_coverage', '202402', coverage)}}))
    actual, actual_used = load_valuations(store, store.published(), days[0], days[-1], FIELDS, 'provider_final')
    pd.testing.assert_frame_equal(actual, expected)
    assert actual_used == used and '600099.SH' not in set(actual.instrument)


@pytest.mark.parametrize(('expression', 'expected'), [
    ('ps_ttm + ts_mean(close_adj, 240)', 0), ('ts_mean(ps_ttm + close_adj, 5)', 4),
    ('ts_mean(ps_ttm, 5) + ts_mean(close_adj, 240)', 4), ('ts_delay(ts_mean(ps_ttm, 5), 3)', 7),
    ('ts_corr(ps_ttm, ts_delay(close_adj, 200), 10)', 9), ('ts_std(pcf_ncf_ttm, 20)', 19),
    ('where(close_adj > 0, ts_delay(ps_ttm, 2), pcf_ncf_ttm)', 2), ('ts_mean(close_adj, 240)', 0),
])
def test_valuation_leaf_warmup(expression, expected):
    assert valuation_lookback([expression], parse) == expected


@pytest.mark.parametrize(('start', 'end'), [('2024-01-01', '2024-01-04'), ('2024-01-02', '2024-01-08')])
def test_request_outside_frozen_calendar_never_returns_shorter_history(tmp_path, start, end):
    store, _, frames = fixture(tmp_path); manifest, sha = bind_raw(store, frames); import_valuations(store.root, manifest, sha)
    with pytest.raises(InputBlocked, match='valuation_calendar_range'):
        query_valuations(store.root, store.snapshot(), start, end, ['600001.SH'], mode='provider_final')


def test_known_closed_dates_are_valid_query_boundaries(tmp_path):
    store, days, frames = fixture(tmp_path); calendar = store.load('calendar')
    calendar.loc[len(calendar)] = [pd.Timestamp('2024-01-05').date(), False]
    calendar.loc[len(calendar)] = [pd.Timestamp('2024-01-06').date(), False]
    store.publish(store.write_batch({'calendar': {'all': store.write_partition('calendar', 'all', calendar)}}))
    manifest, sha = bind_raw(store, frames); import_valuations(store.root, manifest, sha)
    result = query_valuations(store.root, store.snapshot(), days[0], '2024-01-06', ['600001.SH'], mode='provider_final')
    assert len(result['data']) == 3
