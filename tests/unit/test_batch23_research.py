import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from observe.runs import file_sha, write_json
from observe.data.store import fingerprint
from scripts import review_strategy_batch23 as batch


def test_centered_regression_handles_large_price_offset():
    low = np.array([1000001., 1000004., 1000002., 1000007., 1000009.])
    high = low * 1.03 + [1., 2., 1., 3., 2.]
    slope, r2 = batch.reference_regression(low, high)
    actual = stats.linregress(low, high)
    assert slope == pytest.approx(actual.slope, abs=1e-12)
    assert r2 == pytest.approx(actual.rvalue ** 2, abs=1e-12)


@pytest.mark.parametrize('low,high', [([], []), ([1], [2]), ([1, 2], [1]),
    ([1, np.nan], [2, 3]), ([1, 2], [2, np.inf]), ([1, 1], [2, 3]), ([1, 2], [3, 3])])
def test_invalid_or_degenerate_regression_blocks(low, high):
    with pytest.raises(ValueError): batch.reference_regression(low, high)


def test_original_tail_availability_changes_with_future_row():
    x = np.arange(70., dtype=float)
    data = pd.DataFrame({'low': x + 100, 'high': x + 102 + np.sin(x)})
    before = batch.reference_indicators(data.iloc[:60], 4, 8)
    after = batch.reference_indicators(data.iloc[:61], 4, 8)
    assert np.flatnonzero(np.isfinite(before[0]))[0] == 10
    assert np.isnan(before[1][-10:]).all() and np.isnan(before[2][-10:]).all()
    np.testing.assert_array_equal(before[0], after[0][:60])
    assert np.isnan(before[2][50]) and np.isfinite(after[2][50])


def sample():
    dates = pd.date_range('2005-01-05', periods=801).strftime('%Y-%m-%d').tolist()
    dates[-1] = '2026-09-29'
    return pd.DataFrame({'date': dates, 'index': '000300.SH', 'open': 100., 'high': 102., 'low': 99., 'close': 101.})


@pytest.mark.parametrize('defect', ['schema', 'index', 'duplicate', 'first', 'last', 'short', 'nan', 'inf', 'zero', 'high', 'low'])
def test_index_contract_rejects_wrong_scope_and_prices(defect):
    frame = sample()
    if defect == 'schema': frame = frame.drop(columns='high')
    elif defect == 'index': frame.loc[1, 'index'] = '000905.SH'
    elif defect == 'duplicate': frame.loc[1, 'date'] = frame.loc[0, 'date']
    elif defect == 'first': frame.loc[0, 'date'] = '2005-01-04'
    elif defect == 'last': frame.loc[800, 'date'] = '2026-09-28'
    elif defect == 'short': frame = frame.iloc[:800]
    elif defect == 'nan': frame.loc[0, 'high'] = np.nan
    elif defect == 'inf': frame.loc[0, 'low'] = np.inf
    elif defect == 'zero': frame.loc[0, 'open'] = 0
    elif defect == 'high': frame.loc[0, 'high'] = 100
    elif defect == 'low': frame.loc[0, 'low'] = 101
    with pytest.raises(ValueError): batch.index_sample(frame)


def test_accepted_upstream_baseline_cannot_be_rewritten(tmp_path, monkeypatch):
    partition = tmp_path / 'index.parquet'; sample().to_parquet(partition)
    state = {'tables': {'index_1d': {'all': {'file': partition.name, 'sha': fingerprint(pd.read_parquet(partition)), 'rows': 801}}}}
    baseline = tmp_path / 'baseline.json'; receipt = tmp_path / 'receipt.json'
    write_json(baseline, {'published': state})
    write_json(receipt, {'status': 'ok', 'reviews': {'snapshot': batch.SNAPSHOT},
        'checks': {'evidence_sha256': {'baseline.json': file_sha(baseline)}}})
    monkeypatch.setattr(batch, 'UPSTREAM', receipt); monkeypatch.setattr(batch, 'UPSTREAM_BASELINE', baseline)
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(root=tmp_path, state=lambda snapshot: state))
    assert batch.index_binding(tmp_path)['partition_sha256'] == file_sha(partition)
    sample().assign(close=100.5).to_parquet(partition)
    with pytest.raises(ValueError, match='Frozen index partition differs'): batch.index_binding(tmp_path)
    write_json(baseline, {'published': state, 'extra': 'rewritten'})
    with pytest.raises(ValueError, match='Accepted upstream baseline differs'): batch.index_binding(tmp_path)


@pytest.fixture
def probe_evidence(tmp_path):
    import baostock as bs
    fn = bs.query_hs300_stocks; code = inspect.getsource(fn)
    write_json(tmp_path / 'existing-api.json', {'version': bs.__version__, 'name': 'query_hs300_stocks',
        'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'signature': str(inspect.signature(fn)), 'query': batch.QUERY})
    raw = tmp_path / 'raw.parquet'; pd.DataFrame({'code': ['sh.600000']}).to_parquet(raw)
    row = {'query': batch.QUERY, 'status': 'success', 'published': False,
        'api_sha256': file_sha(tmp_path / 'existing-api.json'),
        'files': [{'file': str(raw), 'sha256': file_sha(raw), 'rows': 1, 'columns': ['code']}]}
    write_json(tmp_path / 'probe/result.json', row)
    return tmp_path, raw, row


@pytest.mark.parametrize('defect', ['raw', 'query', 'rows', 'schema', 'status', 'api_sha', 'published', 'count'])
def test_probe_metadata_and_bytes_are_verified(probe_evidence, defect):
    directory, raw, row = probe_evidence
    assert batch.validate_probe(directory)['status'] == 'success'
    if defect == 'raw': pd.DataFrame({'code': ['sh.600001']}).to_parquet(raw)
    elif defect == 'query': row['query'] = {**batch.QUERY, 'date': '2026-09-29'}
    elif defect == 'rows': row['files'][0]['rows'] = 2
    elif defect == 'schema': row['files'][0]['columns'] = ['other']
    elif defect == 'status': row['status'] = 'unknown'
    elif defect == 'api_sha': row['api_sha256'] = 'wrong'
    elif defect == 'published': row['published'] = True
    elif defect == 'count': row['files'] = []
    write_json(directory / 'probe/result.json', row)
    with pytest.raises(ValueError): batch.validate_probe(directory)


def test_failed_probe_cannot_accept_raw_input(probe_evidence):
    directory, _, row = probe_evidence
    row.update(status='failed', error='ConnectionError')
    write_json(directory / 'probe/result.json', row)
    with pytest.raises(ValueError, match='Failed probe accepted input'): batch.validate_probe(directory)


def test_coverage_rejects_missing_calendar_session(tmp_path, monkeypatch):
    frame = sample(); calendar = tmp_path / 'calendar.parquet'; pool = tmp_path / 'pool.parquet'
    pd.DataFrame({'date': frame.date[:-1], 'is_open': True}).to_parquet(calendar)
    pd.DataFrame({'date': ['2024-03-29'], 'index': ['000300.SH']}).to_parquet(pool)
    state = {'tables': {name: {'all': {'file': path.name, 'sha': fingerprint(pd.read_parquet(path))}}
        for name, path in [('calendar', calendar), ('index_constituents', pool)]}}
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(root=tmp_path, state=lambda snapshot: state,
        load_state=lambda state, name, **kw: pd.read_parquet(tmp_path / state['tables'][name]['all']['file'])))
    with pytest.raises(ValueError, match='Index calendar incomplete'): batch.coverage(tmp_path, frame)
    pd.DataFrame({'date': frame.date, 'is_open': True}).to_parquet(calendar)
    with pytest.raises(ValueError, match='Coverage partition changed'): batch.coverage(tmp_path, frame)


def test_absent_local_constituents_are_explicit_not_remote_pool(tmp_path, monkeypatch):
    frame = sample(); path = tmp_path / 'calendar.parquet'
    pd.DataFrame({'date': frame.date, 'is_open': True}).to_parquet(path)
    state = {'tables': {'calendar': {'all': {'file': path.name, 'sha': fingerprint(pd.read_parquet(path))}}}}
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(root=tmp_path, state=lambda snapshot: state,
        load_state=lambda state, name, **kw: pd.read_parquet(path) if name == 'calendar' else pd.DataFrame()))
    assert batch.coverage(tmp_path, frame) == {'rows': 0, 'first': None, 'last': None,
        'strict_usable': False, 'not_original_daily_pool_proof': True}


def test_initializer_callback_name_is_available_before_noop_schedule(monkeypatch):
    class Scheduled(Exception): pass
    def selected(directory, number, names, ns):
        if number == 0:
            ns['market_open'] = lambda context: setattr(ns['g'], 'init', False)
        elif names == ('initialize', 'cal_date_fzxy'):
            assert callable(ns['handle'])
            ns['run_daily'](ns['handle'], time='before_open')
            raise Scheduled()
        return 'selected'
    monkeypatch.setattr(batch, 'selected', selected)
    with pytest.raises(Scheduled): batch.diagnostics(None, None)


def test_review_supplement_cannot_change_trading_rules(tmp_path):
    after = batch.REVIEWS[batch.SOURCES[1]]
    import json
    before = json.loads(json.dumps(after)); before['gaps'][0] = batch.OLD_POOL_GAP
    initial = tmp_path / 'source-reviews/review.json'
    write_json(initial, {'sources': [{}, {'source_sha256': 'source', 'review': before}]})
    row = {'review_sha256': file_sha(initial), 'source_sha256': 'source', 'before': before, 'after': after}
    write_json(tmp_path / 'source-review-supplement.json', row)
    assert batch.clarified_reviews(tmp_path) == row
    row['before']['rules']['selection'] = 'wrong'
    write_json(tmp_path / 'source-review-supplement.json', row)
    with pytest.raises(ValueError, match='Review supplement differs'): batch.clarified_reviews(tmp_path)
