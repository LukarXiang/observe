import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch22 as batch


def test_original_factors_have_distinct_first_valid_rows():
    close = np.arange(100, 120, dtype=float)
    open_values = close - 1
    actual = batch.reference_factors(open_values, close)
    assert {k: len(v) for k, v in actual.items()} == {'M': 5, 'M0': 6, 'M1': 5}
    assert actual['M'][0] == close[15] / close[0] - 1
    assert actual['M0'][0] == pytest.approx(sum(close[:15] / open_values[:15] - 1), abs=1e-15)
    assert actual['M1'][0] == pytest.approx(sum(open_values[1:16] / close[:15] - 1), abs=1e-15)


@pytest.mark.parametrize('defect', ['short', 'shape', 'length', 'nan', 'inf', 'zero', 'negative'])
def test_invalid_factor_inputs_block(defect):
    opens, closes = np.ones(20), np.ones(20)
    if defect == 'short': opens, closes = opens[:15], closes[:15]
    elif defect == 'shape': opens = opens.reshape(2, 10)
    elif defect == 'length': closes = closes[:-1]
    elif defect == 'nan': opens[0] = np.nan
    elif defect == 'inf': closes[0] = np.inf
    elif defect == 'zero': closes[0] = 0
    elif defect == 'negative': opens[0] = -1
    with pytest.raises(ValueError, match='Invalid factor inputs'): batch.reference_factors(opens, closes)


@pytest.mark.parametrize('descending,expected', [(False, [2.5, 1., 2.5, 4.]), (True, [2.5, 4., 2.5, 1.])])
def test_ranks_average_ties_in_both_directions(descending, expected):
    assert batch.average_ranks([2, 1, 2, 3], descending) == expected


def sample():
    return pd.DataFrame({'代码': '801030', '日期': pd.date_range('2005-12-01', periods=20), '开盘': 10., '收盘': 11.})


@pytest.mark.parametrize('defect', ['wrong_code', 'duplicate', 'nan', 'zero', 'schema'])
def test_invalid_industry_sample_blocks(defect):
    frame = sample()
    if defect == 'wrong_code': frame.loc[1, '代码'] = '801020'
    elif defect == 'duplicate': frame.loc[1, '日期'] = frame.loc[0, '日期']
    elif defect == 'nan': frame.loc[0, '开盘'] = np.nan
    elif defect == 'zero': frame.loc[0, '收盘'] = 0
    elif defect == 'schema': frame = frame.drop(columns='代码')
    with pytest.raises(ValueError): batch.industry_sample(frame)


def test_provider_rows_after_original_period_do_not_enter_research():
    frame = sample()
    frame.loc[20] = ['801030', pd.Timestamp('2026-09-29'), 1000., 1000.]
    actual = batch.industry_sample(frame)
    assert len(actual) == 20 and actual.date.max() == '2005-12-20'


def test_original_provider_string_prices_are_strictly_parsed():
    frame = sample().astype({'开盘': str, '收盘': str})
    actual = batch.industry_sample(frame)
    assert actual.open.tolist() == [10.] * 20 and actual.close.tolist() == [11.] * 20
    assert actual[['open', 'close']].dtypes.tolist() == [np.dtype('float64')] * 2
    frame.loc[0, '开盘'] = 'not-a-price'
    with pytest.raises(ValueError): batch.industry_sample(frame)


def test_calendar_gap_blocks_crossing_windows_and_recovers_after_full_width():
    sessions = [str(k).zfill(2) for k in range(10)]
    dates = [d for d in sessions if d != '04']
    actual = batch.calendar_windows(dates, sessions, 3)
    assert actual == {'00': False, '01': False, '02': True, '03': True, '05': False, '06': False, '07': True, '08': True, '09': True}
    with pytest.raises(ValueError): batch.calendar_windows(dates[::-1], sessions, 3)


def test_changed_upstream_terminal_rejected_by_accepted_receipt(tmp_path, monkeypatch):
    data = tmp_path / 'input.parquet'; sample().to_parquet(data)
    terminal = tmp_path / 'result.json'; receipt = tmp_path / 'receipt.json'
    write_json(terminal, {'status': 'success', 'endpoint': 'sw801030',
        'files': [{'file': str(data), 'sha256': file_sha(data)}], 'wire_responses': []})
    write_json(receipt, {'status': 'ok', 'reviews': {'snapshot': batch.SNAPSHOT},
        'checks': {'evidence_sha256': {'probes/sw801030/result.json': file_sha(terminal)}}})
    monkeypatch.setattr(batch, 'UPSTREAM', receipt); monkeypatch.setattr(batch, 'TERMINAL', terminal)
    assert batch.upstream()['result']['files'][0]['sha256'] == file_sha(data)
    sample().assign(收盘=99.).to_parquet(data)
    with pytest.raises(ValueError, match='Upstream input differs'): batch.upstream()
    write_json(terminal, {'status': 'success', 'endpoint': 'sw801030',
        'files': [{'file': str(data), 'sha256': file_sha(data)}], 'wire_responses': []})
    with pytest.raises(ValueError, match='Accepted upstream terminal differs'): batch.upstream()


@pytest.fixture
def probe_evidence(tmp_path):
    import akshare as ak
    apis = []
    for name in sorted({n for n, _ in batch.QUERIES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'signature': str(inspect.signature(fn))})
    write_json(tmp_path / 'existing-apis.json', {'version': ak.__version__, 'apis': apis,
        'queries': {k: [n, p] for k, (n, p) in batch.QUERIES.items()}})
    outside = tmp_path / 'raw' / 'sample.parquet'; outside.parent.mkdir(); pd.DataFrame({'x': [1.]}).to_parquet(outside)
    results = []
    for endpoint in batch.QUERIES:
        row = {'endpoint': endpoint, 'published': False, 'api_evidence_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'error': 'ConnectionError: archived fixture', 'files': [], 'wire_responses': []}
        if not results:
            row['status'] = 'success'; row.pop('error')
            row['files'] = [{'file': str(outside), 'sha256': file_sha(outside), 'rows': 1, 'columns': ['x'], 'parameters': batch.QUERIES[endpoint][1]}]
        results.append(row)
        write_json(tmp_path / 'probes' / endpoint / 'result.json', row)
    write_json(tmp_path / 'probe-results.json', {'results': results, 'published': False})
    return tmp_path, results, outside


def test_raw_probe_modified_after_analysis_is_rejected(probe_evidence):
    directory, _, outside = probe_evidence
    assert len(batch.validate_probes(directory)) == 4
    pd.DataFrame({'x': [999.]}).to_parquet(outside)
    with pytest.raises(ValueError, match='Probe artifact changed'): batch.validate_probes(directory)


@pytest.mark.parametrize('defect', ['parameters', 'rows', 'columns', 'file_count', 'status', 'api_sha'])
def test_self_consistent_probe_terminal_metadata_cannot_bypass_contract(probe_evidence, defect):
    directory, results, _ = probe_evidence; row = results[0]
    if defect == 'parameters': row['files'][0]['parameters'] = {'symbol': 'wrong'}
    elif defect == 'rows': row['files'][0]['rows'] = 0
    elif defect == 'columns': row['files'][0]['columns'] = ['not_x']
    elif defect == 'file_count': row['files'] = []
    elif defect == 'status': row['status'] = 'invented_success'
    elif defect == 'api_sha': row['api_evidence_sha256'] = 'wrong'
    write_json(directory / 'probes' / row['endpoint'] / 'result.json', row)
    write_json(directory / 'probe-results.json', {'results': results, 'published': False})
    with pytest.raises(ValueError): batch.validate_probes(directory)


@pytest.mark.parametrize('key,value', [('status', 'different'), ('differences', 1), ('original_unchanged', False),
    ('original_sha256', 'wrong'), ('recomputed_sha256', 'wrong'), ('network', 'network allowed'), ('not_a_strategy_reproduction', False)])
def test_wrong_offline_receipt_cannot_be_accepted(tmp_path, key, value):
    write_json(tmp_path / 'component-research.json', {'factor': 1})
    write_json(tmp_path / 'component-offline.json', {'factor': 1})
    sha = file_sha(tmp_path / 'component-research.json')
    row = {'status': 'match', 'differences': 0, 'original_unchanged': True, 'original_sha256': sha, 'recomputed_sha256': sha,
        'network': 'socket connect/connect_ex and requests forbidden', 'not_a_strategy_reproduction': True}
    write_json(tmp_path / 'offline-verification.json', row)
    assert batch.validate_offline(tmp_path) == row
    row[key] = value; write_json(tmp_path / 'offline-verification.json', row)
    with pytest.raises(ValueError, match='Offline receipt status/binding differs'): batch.validate_offline(tmp_path)


def test_review_supplement_only_corrects_known_metrics_error(tmp_path):
    after = batch.REVIEWS[batch.SOURCES[2]]; before = json.loads(json.dumps(after))
    before['rules']['metrics'] = 'LGBMClassifier.predict实际硬标签，被当概率求AUC/多阈值'
    initial = tmp_path / 'source-reviews/review.json'
    write_json(initial, {'sources': [{}, {}, {'review': before, 'source_sha256': 'source-sha'}]})
    supplement = {'review_sha256': file_sha(initial), 'source_sha256': 'source-sha', 'before': before, 'after': after}
    write_json(tmp_path / 'source-review-supplement.json', supplement)
    assert batch.clarified_reviews(tmp_path) == supplement
    before['rules']['label'] = 'wrong label'
    write_json(initial, {'sources': [{}, {}, {'review': before, 'source_sha256': 'source-sha'}]})
    supplement.update(review_sha256=file_sha(initial), before=before)
    write_json(tmp_path / 'source-review-supplement.json', supplement)
    with pytest.raises(ValueError, match='Supplement changed unrelated rules'): batch.clarified_reviews(tmp_path)
