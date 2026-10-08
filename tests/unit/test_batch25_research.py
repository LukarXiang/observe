import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.data.store import fingerprint
from observe.runs import file_sha, write_json
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch25 as batch


@pytest.fixture
def sources(tmp_path):
    records = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Original ignored strategy inputs must be restored for source-function verification')
        records.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': records})
    return tmp_path


@pytest.mark.parametrize('kind,n,delay', [('rolling', 3, 2), ('continuous', 2, 1), ('ma', 8, 3)])
@pytest.mark.parametrize('how', ['up', 'down'])
def test_original_macro_signal_matches_independent_scalar_with_nan(sources, kind, n, delay, how):
    ns = {'pd': pd}; batch.selected(sources, 1, batch.SIGNALS, ns); series = batch.synthetic_series()
    fn = ns[batch.SIGNALS[('rolling', 'continuous', 'ma').index(kind)]]
    actual = fn(series, n, delay=delay, how=how) if kind != 'ma' else fn(series, long_n=n, short_n=2, delay=delay, how=how)
    expected = batch.signal_reference(series, kind, n, delay, how)
    assert actual.index.tolist() == [series.index[k] for k, _ in expected]
    assert actual.position.tolist() == [v for _, v in expected]


def test_original_warmup_policies_differ_and_delay_is_observed_rows(sources):
    ns = {'pd': pd}; batch.selected(sources, 1, batch.SIGNALS, ns)
    series = pd.Series([1., 2., 3., 4., 5.], index=['2024-02', '2024-03', '2024-05', '2024-06', '2024-07'])
    rolling = ns[batch.SIGNALS[0]](series, 3, delay=1)
    ma = ns[batch.SIGNALS[2]](series, long_n=3, short_n=2, delay=1)
    assert rolling.index.tolist() == ['2024-03', '2024-05', '2024-06', '2024-07']
    assert rolling.position.tolist() == [0., 0., 0., 1.]
    assert ma.index.tolist() == ['2024-06', '2024-07']


@pytest.mark.parametrize('values,n,delay,how,kind', [([np.inf], 2, 1, 'up', 'rolling'), ([1.], 0, 1, 'up', 'rolling'),
    ([1.], 2, -1, 'up', 'rolling'), ([1.], 2, 1, 'unknown', 'rolling'), ([1.], 2, 1, 'up', 'unknown')])
def test_invalid_reference_contract_blocks(values, n, delay, how, kind):
    with pytest.raises(ValueError): batch.signal_reference(values, kind, n, delay, how)


def test_fund_original_double_quarter_lag_and_age_filters(sources):
    rows = batch.fund_quarter_diagnostic(sources)
    assert rows[1]['stock_value_quarter'] == '2024-03-31'
    assert rows[1]['stock_rate_quarter'] == '2023-12-31'
    criteria = rows[1]['queries'][0]['filters']
    assert ('lt', 'start_date', '2023-04-01') in criteria
    assert ('or', ('gt', 'end_date', '2024-03-31'), ('eq', 'end_date', None)) in criteria
    assert ('ge', 'stock_rate', 70) in rows[1]['queries'][1]['filters']


def test_python2_function_slice_runs_without_top_level_conversion(sources):
    ns = {}; batch.selected(sources, 0, ('good_cpi',), ns)
    assert [ns['good_cpi'](v) for v in [-1., 0., 4.99, 5., np.nan]] == [0., 1., 1., 0., 0.]
    with pytest.raises(SyntaxError): batch.ast.parse(read_source(Path('repo/量化策略源代码') / batch.SOURCES[0])[0])


def test_diagnostic_results_are_json_roundtrip_stable(sources, monkeypatch):
    frame = pd.DataFrame({'date': ['2009-01-05', '2009-01-06'], 'close': [10., 11.]})
    path = sources / 'index-input.parquet'; frame.to_parquet(path, index=False)
    write_json(sources / 'input-analysis.json', {'file': str(path), 'sha256': file_sha(path)})
    monkeypatch.setattr(batch, 'index_sample', lambda root: frame)
    result = batch.diagnostics(None, sources)
    assert result == json.loads(json.dumps(result))
    cases = {r['case']: r for r in result['cases']}
    assert len(cases) == 9 and cases['fund_retry_not_incremented']['stub_sell_buy_calls'] == 20
    assert cases['fund_finishes_with_codes_only']['finish_trade']
    assert cases['empty_fund_dict_no_division']['targets'] == {}
    assert cases['short_inventory_overwritten_and_duplicated']['inventory_columns'] == 2


def test_index_partition_fingerprint_and_byte_changes_rejected(tmp_path, monkeypatch):
    frame = pd.DataFrame({'close': [100.]}); path = tmp_path / 'input.parquet'; frame.to_parquet(path, index=False)
    tables = {n: {'all': {'file': path.name, 'sha': fingerprint(frame), 'rows': 1}} for n in ('index_1d', 'calendar')}
    baseline = tmp_path / 'baseline.json'; write_json(baseline, {'published': {'tables': tables}})
    monkeypatch.setattr(batch, 'BASELINE', baseline); monkeypatch.setattr(batch, 'upstream', lambda: {'accepted': True})
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(root=tmp_path, state=lambda snapshot: {'tables': tables}))
    original = batch.index_binding(tmp_path)
    assert original['partitions'][0]['sha256'] == file_sha(path)
    frame.assign(close=200.).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Index partition differs'): batch.index_binding(tmp_path)


@pytest.fixture
def probes(tmp_path, monkeypatch):
    evidence = {'version': 'frozen', 'queries': batch.QUERIES, 'apis': []}
    monkeypatch.setattr(batch, 'api_evidence', lambda: evidence); write_json(tmp_path / 'existing-apis.json', evidence)
    rows = []
    for endpoint, (name, params) in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'function': name, 'parameters': params, 'status': 'failed', 'files': [], 'wire_responses': [],
            'published': False, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'), 'error': 'Archived failure'}
        rows.append(row); write_json(tmp_path / 'probes' / endpoint / 'result.json', row)
    write_json(tmp_path / 'probe-results.json', {'results': rows})
    return tmp_path, rows


@pytest.mark.parametrize('defect', ['function', 'parameters', 'api_sha', 'status', 'raw', 'terminal'])
def test_probe_scope_and_terminals_bound(probes, defect):
    path, rows = probes; assert len(batch.validate_probes(path)) == 4
    if defect == 'function': rows[0]['function'] = 'wrong'
    elif defect == 'parameters': rows[0]['parameters'] = {'date': 'wrong'}
    elif defect == 'api_sha': rows[0]['api_sha256'] = 'wrong'
    elif defect == 'status': rows[0]['status'] = 'unknown'
    elif defect == 'raw': rows[0]['files'] = [{'file': 'unproved'}]
    elif defect == 'terminal': rows[0]['error'] = 'different'
    if defect != 'terminal': write_json(path / 'probes' / rows[0]['endpoint'] / 'result.json', rows[0])
    write_json(path / 'probe-results.json', {'results': rows})
    with pytest.raises(ValueError): batch.validate_probes(path)
