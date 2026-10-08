from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch27 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        p = Path('repo/量化策略源代码') / name
        if not p.is_file(): pytest.skip('Restore ignored original sources to verify exact functions')
        rows.append({'source_copy': str(p), 'source_sha256': file_sha(p)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows}); return tmp_path


def test_exact_value_function_preserves_missing_and_sample_standardization(sources):
    holder, fn, _ = batch.value_function(sources)
    holder['frame'] = pd.DataFrame({'code': ['a', 'b', 'c'], 'pb_ratio': [1., 2., 3.],
        'pe_ratio': [3., 2., 1.], 'ps_ratio': [1., np.nan, 3.]})
    result = fn(['a', 'b', 'c'], type('Context', (), {'previous_date': '2024-01-02'})()).value
    assert result['a'] == pytest.approx(-1 / np.sqrt(2) / 3)
    assert np.isnan(result['b']) and result['c'] == pytest.approx(1 / np.sqrt(2) / 3)


def test_constant_cross_section_stays_undefined():
    result = batch.reference_z([[1., 2.], [1., 2.]])
    assert np.isnan(result).all()
    with pytest.raises(ValueError): batch.reference_z([[np.inf], [1.]])


def test_source_diagnostics_show_defects_and_real_refresh_period(sources):
    cases = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert cases['actual_30_callback_refresh']['selection_callbacks'] == [0, 30, 60, 90]
    assert cases['empty_factor_list']['error'] == 'ZeroDivisionError'
    assert cases['sum_paused_dataframe']['error'] == 'TypeError'
    assert cases['legacy_dataframe_sort']['error'] == 'AttributeError'
    assert cases['legacy_series_integer_index']['error'] == 'KeyError'


def test_turtle_uses_81high_and_negative_nine_row_return_arithmetic(sources):
    code, _ = batch.expression(sources, 1, 'choice_stock', 'zdf')
    assert eval(code, {}, {'df1': {'close': np.array([100., 101., 100., 99., 97., 96., 95., 94., 90.])}}) == -.1
    code, _ = batch.expression(sources, 1, 'choice_stock', 'p_max')
    assert eval(code, {}, {'df55': pd.DataFrame({'high': [103., 100., 101.]})}) == 103.


def test_original_ma10_ma60_means_use_actual_windows(sources):
    a, _ = batch.expression(sources, 2, 'check_holding', 'MAs')
    b, _ = batch.expression(sources, 2, 'check_holding', 'MAl')
    scope = {'close_data_MAs': pd.DataFrame({'close': np.arange(51, 61)}),
        'close_data_MAl': pd.DataFrame({'close': np.arange(1, 61)})}
    assert eval(a, {}, scope) == 55.5 and eval(b, {}, scope) == 30.5


def test_changed_source_rejected_before_expression_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][1]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Source changed'): batch.expression(sources, 1, 'choice_stock', 'zdf')


def test_ambiguous_expression_refused(sources):
    with pytest.raises(ValueError, match='Ambiguous/missing'): batch.expression(sources, 1, 'choice_stock', 'not_present')


def test_frozen_input_bytes_refused_before_parse(tmp_path):
    p = tmp_path / 'price-input.parquet'; pd.DataFrame({'date': ['2024-01-02'], 'instrument': ['x']}).to_parquet(p)
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'pool': list(batch.INSTRUMENTS),
        'not_a_backtest': True, 'inputs': {'price': {'file': str(p), 'sha256': '0' * 64}}})
    with pytest.raises(ValueError, match='Component input changed'): batch.inputs(tmp_path)


@pytest.mark.parametrize('change', ['none', 'price', 'single_null'])
def test_prepare_retains_matching_paused_null_and_rejects_changed_raw_price(tmp_path, monkeypatch, change):
    accepted = tmp_path / 'accepted'; accepted.mkdir(); destination = tmp_path / 'output'; destination.mkdir()
    price = pd.DataFrame({'date': ['2024-01-02', '2024-01-03'], 'instrument': ['600519.SH'] * 2,
        'close': [10., np.nan], 'close_adj': [10., np.nan], 'back_factor': [1., 1.],
        'is_trading': [True, False], 'adjustment_status': ['usable', 'usable']})
    price.to_parquet(accepted / 'price-input.parquet', index=False)
    price[['date', 'instrument']].to_parquet(accepted / 'value-input.parquet', index=False)
    bars = price[['date', 'instrument', 'close', 'is_trading']].assign(high=[11., np.nan], low=[9., np.nan])
    if change == 'price': bars.loc[0, 'close'] = 10.1
    if change == 'single_null': bars.loc[1, 'close'] = 10.
    monkeypatch.setattr(batch, 'ACCEPTED', accepted); monkeypatch.setattr(batch, 'binding', lambda *a: {})
    monkeypatch.setattr(batch, 'archive', lambda *a: {'archived': True})
    monkeypatch.setattr(batch, 'Store', lambda *a: SimpleNamespace(state=lambda *a: {'tables': {}}, load_state=lambda *a, **k: bars))
    if change != 'none':
        with pytest.raises(AssertionError): batch.prepare(tmp_path, destination)
        assert not (destination / 'price-input.parquet').exists()
    else:
        assert batch.prepare(tmp_path, destination)['archived']
        assert pd.read_parquet(destination / 'price-input.parquet').close.isna().sum() == 1
        with pytest.raises(ValueError, match='Prepared input archive exists'): batch.prepare(tmp_path, destination)


@pytest.mark.parametrize('change', ['none', 'file', 'rows', 'fingerprint', 'extra_part'])
def test_snapshot_actual_partition_reference_must_match_accepted_binding(tmp_path, monkeypatch, change):
    accepted = tmp_path / 'accepted'; accepted.mkdir(); tables = {}; rows = []
    for table in ('bars_1d', 'adj_factors', 'adj_coverage'):
        p = tmp_path / f'{table}.parquet'; p.write_bytes(b'accepted-partition')
        tables[table] = {'all': {'file': p.name, 'rows': 1, 'sha': 'known-fingerprint'}}
        rows.append({'table': table, 'part': 'all', 'file': str(p), 'rows': 1, 'fingerprint': 'known-fingerprint', 'sha256': file_sha(p)})
    write_json(accepted / 'stock-binding.json', rows)
    for name in ('price-input.parquet', 'value-input.parquet'): (accepted / name).write_bytes(b'accepted-input')
    upstream = tmp_path / 'upstream.json'; receipt = tmp_path / 'receipt.json'
    write_json(upstream, {'status': 'ok', 'recovery': {'snapshot': batch.SNAPSHOT, 'batch_id': 'accepted-batch'}})
    write_json(receipt, {'status': 'ok', 'checks': {'evidence_sha256': {p.name: file_sha(p) for p in accepted.iterdir()}}})
    if change == 'file':
        wrong = tmp_path / 'wrong.parquet'; wrong.write_bytes(b'wrong-high-and-low'); tables['bars_1d']['all']['file'] = wrong.name
    if change == 'rows': tables['bars_1d']['all']['rows'] = 2
    if change == 'fingerprint': tables['bars_1d']['all']['sha'] = 'changed'
    if change == 'extra_part': tables['bars_1d']['extra'] = tables['bars_1d']['all'].copy()
    monkeypatch.setattr(batch, 'ACCEPTED', accepted); monkeypatch.setattr(batch, 'UPSTREAM', upstream); monkeypatch.setattr(batch, 'PRICE_RECEIPT', receipt)
    monkeypatch.setattr(batch, 'Store', lambda *a: SimpleNamespace(root=tmp_path, state=lambda *a: {'batch_id': 'accepted-batch', 'tables': tables}))
    if change == 'none': assert batch.binding(tmp_path)['snapshot'] == batch.SNAPSHOT
    else:
        with pytest.raises(ValueError, match='Accepted stock'): batch.binding(tmp_path)


@pytest.mark.parametrize('change', ['none', 'parameters', 'api', 'status', 'raw'])
def test_supplement_failures_remain_bound_without_accepted_raw(tmp_path, monkeypatch, change):
    api = {'function': 'financial_indicator', 'parameters': {'symbol': '600519', 'start_year': '2020'}}
    monkeypatch.setattr(batch, 'supplement_api', lambda: api)
    write_json(tmp_path / 'supplement-api.json', api)
    row = {'function': api['function'], 'parameters': api['parameters'], 'status': 'failed', 'files': [], 'wire': [],
        'published': False, 'api_sha256': file_sha(tmp_path / 'supplement-api.json'), 'error': 'DNS failed'}
    if change == 'parameters': row['parameters'] = {'symbol': 'different'}
    if change == 'api': row['api_sha256'] = '0' * 64
    if change == 'status': row['status'] = 'unknown'
    if change == 'raw': row['files'] = [{'file': 'unaccepted'}]
    write_json(tmp_path / 'supplement-result.json', row)
    if change == 'none': assert batch.validate_supplement(tmp_path)['status'] == 'failed'
    else:
        with pytest.raises(ValueError, match='Supplement|Failed'): batch.validate_supplement(tmp_path)
