import ast
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch71 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [batch.previous.parsed_source(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_original_definitions_and_missing_helper_scope(monkeypatch):
    trees = originals(monkeypatch)
    assert [len(batch.definitions(t)) for t in trees] == [30, 2]
    assert [next(n.lineno for n in t.body if isinstance(n, ast.FunctionDef)) for t in trees] == [17, 13]
    defined = {n.name for t in trees for n in t.body if isinstance(n, ast.FunctionDef)}
    assert not defined.intersection(batch.HELPERS)


def test_original_callback_order_query_and_failure_states(monkeypatch):
    originals(monkeypatch)
    result = batch.diagnostics(Path('unused'))
    cases = {r['case']: r for r in result['cases']}
    assert len(cases) == 21
    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert result['synthetic_fixture'] and result['not_a_backtest']
    assert not result['platform_equivalent'] and result['real_historical_trade_windows'] == 0
    defaults = cases['wizard_defaults_five_callbacks_ten_slots']
    assert defaults['industry_count'] == 28
    assert defaults['score_input'] == {'market_cap': ['asc', 50]}
    assert defaults['outer_sort'] == 'desc' and not defaults['external_helpers_proved']
    assert defaults['scheduled'] == [['sell_every_day', 'open'], ['risk_management', 'every_bar'],
        ['check_stocks', 'open'], ['trade', 'open'], ['selled_security_list_count', 'after_close']]
    assert cases['wizard_pool_sorted_deduplicated']['pool'] == ['A', 'B']
    assert cases['wizard_user_security_string_extends_characters']['pool'] == ['A', 'B']
    assert cases['wizard_false_hold_filter_excludes_existing']['remaining'] == ['C']
    assert cases['wizard_selection_callback_numbers_1_6_11']['counter_at_selection'] == [0, 5, 5]
    assert cases['wizard_None_pending_sell_retained']['pending'] == ['D']
    assert cases['wizard_dynamic_keys_lose_immediate_sold_history']['sold_counter'] == {}
    assert cases['wizard_external_size_stubs_only_increment_intent']['orders'] == [['increment', 'D', 100, 'synthetic-style']]
    assert cases['wizard_overfull_negative_slice_calls_allocator_but_no_order']['allocator_candidates'] == ['20', '21', '22', '23']
    rotation = cases['rotation_same_date_PEgt1_two_indices_increment_retained_cash_div10']
    assert rotation['orders'] == [['target', 'OLD', 0], ['increment_value', 'A', 100.], ['increment_value', 'C', 100.]]
    assert rotation['index_calls'] == ['000001.XSHG', '399106.XSHE']
    assert rotation['query']['filters'] == [['in', 'code', ['A', 'C']], ['gt', 'pe_ratio', 1]]
    assert rotation['query']['sort'] == [['asc', 'market_cap']]
    assert rotation['query']['kwargs'] == {'date': '2024-01-02'}
    assert rotation['scheduled'] == [['daily', {'time': 'every_bar'}]]
    assert cases['rotation_every10_callbacks_not_proved_days']['selected_callbacks'] == [1, 11, 21]
    assert cases['rotation_immediate_sell_mutates_iteration_RuntimeError']['error'] == 'RuntimeError'
    assert cases['rotation_exception_preserves_counter']['counter'] == 1
    assert cases['rotation_empty_candidates_no_order_but_counter_advances']['counter'] == 2
    assert cases['rotation_empty_same_day_paused_IndexError']['error'] == 'IndexError'


@pytest.mark.parametrize('code', [
    '@unsafe\ndef f():\n    pass',
    'def f(value=open("secret")):\n    pass',
    'def f(*, value=open("secret")):\n    pass',
])
def test_selected_rejects_eager_execution(monkeypatch, code):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(code))
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected(Path('unused'), 0, ['f'], {})


def test_PE_boundary_retains_original_comparison_and_supplier_scope(monkeypatch):
    tree = originals(monkeypatch)[1]
    pe = [-1., 0., .5, 1., np.nextafter(1., 2.), 10., np.nan, np.inf, -np.inf]
    frame = pd.DataFrame({'date': [dt.date(2020, 1, 2)]*len(pe),
        'instrument': [str(i) for i in range(len(pe))], 'pe_ttm': pe, 'is_trading': [True]*len(pe)})
    values, summary = batch.numeric_values(frame, tree)
    assert list(values.original_PE_gt1) == [False, False, False, False, True, True, False, True, False]
    assert summary['predicate_passes'] == 3 and summary['daily_coverage'][0]['nonfinite_PE'] == 3
    assert not summary['original_signal_reproduced'] and summary['not_a_backtest']


def test_cutoff_and_duplicate_daily_input(monkeypatch):
    tree = originals(monkeypatch)[1]
    frame = pd.DataFrame({'date': ['2026-09-29', '2026-09-30'],
        'instrument': ['A', 'A'], 'pe_ttm': [2., 999999.], 'is_trading': [True, True]})
    values, summary = batch.numeric_values(frame, tree)
    assert len(values) == 1 and summary['rows_after_cutoff_excluded'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        batch.numeric_values(pd.concat([frame, frame.iloc[[0]]]), tree)


@pytest.mark.parametrize('resolution', ['s', 'us', 'ns'])
def test_daily_dates_keep_exact_dtype_through_parquet(tmp_path, resolution):
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-09-29']).astype(f'datetime64[{resolution}]'),
        'instrument': ['A'], 'pe_ttm': [2.], 'is_trading': [True]})
    values, _ = batch.numeric_values(frame, ast.parse('valuation.pe_ratio > 1'))
    path = tmp_path / 'values.parquet'
    values.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(values, pd.read_parquet(path), check_exact=True)
    assert str(values.date.dtype) == 'datetime64[ms]'


def test_date_normalization_cannot_silently_lose_input_values():
    frame = pd.DataFrame({'date': [pd.Timestamp('2026-09-29 00:00:00.000000001')],
        'instrument': ['A'], 'pe_ttm': [2.], 'is_trading': [True]})
    with pytest.raises(ValueError, match='Daily date precision loss'):
        batch.numeric_values(frame, ast.parse('valuation.pe_ratio > 1'))


@pytest.mark.parametrize('expression', ['valuation.pe_ratio > 0', 'valuation.pe_ratio >= 1', 'other.pe_ratio > 1'])
def test_PE_expression_changes_are_rejected(expression):
    with pytest.raises(ValueError, match='Original PE predicate'):
        batch.pe_predicate(ast.parse(expression), pd.Series([2.]))


def frozen_api(tmp_path, monkeypatch):
    ak = pytest.importorskip('akshare')
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    return ak


def wire(body):
    return json.dumps({'Result': [{'DisplayData': {'resultData': {'tplData': {'result':
        {'chartInfo': [{'body': body, 'unit': '亿'}]}}}}}]}).encode()


def mock_response(content, status=200):
    response = batch.requests.Response()
    response.status_code = status
    response._content = content
    response.encoding = 'utf-8'
    response.url = batch.URL
    return response


def test_worker_saves_wire_single_GET_and_exact_date_gaps(tmp_path, monkeypatch):
    frozen_api(tmp_path, monkeypatch)
    calls = []
    raw = wire([['2021-01-29', '100'], ['2021-01-30', '101'], ['2026-09-30', '102']])
    def request(session, method, url, **kwargs):
        assert not session.trust_env and kwargs['timeout'] == (8, 10)
        assert kwargs['allow_redirects'] is False
        calls.append((method, url, kwargs['params']))
        return mock_response(raw)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', request)
    row = batch.worker(tmp_path, '000001-all')
    assert row['status'] == 'success' and row['rows'] == 3 and len(calls) == len(row['attempts']) == 1
    assert not row['strict_usable'] and not row['published']
    assert Path(row['wire_responses'][0]['file']).read_bytes() == raw
    batch.save(tmp_path / 'input-binding.json', {'reused_cap_samples': {}})
    batch.save(tmp_path / 'probe-results.json', {'results': [row]})
    pd.DataFrame({'date': pd.to_datetime(['2021-01-29', '2021-02-01']),
        'is_open': [True, True]}).to_parquet(tmp_path / 'calendar-input.parquet', index=False)
    cap = batch.total_cap_coverage(tmp_path)['samples'][0]
    assert cap['matched_dates'] == 1 and cap['missing_dates'] == ['2021-02-01']
    assert cap['nontrading_dates'] == 1 and cap['future_rows_excluded'] == 1
    assert cap['unit'] == '亿' and cap['parser_match']
    with pytest.raises(FileExistsError):
        batch.worker(tmp_path, '000001-all')


def test_worker_failed_parse_keeps_original_response(tmp_path, monkeypatch):
    frozen_api(tmp_path, monkeypatch)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', lambda *a, **k: mock_response(b'not json'))
    row = batch.worker(tmp_path, '000001-5y')
    assert row['status'] == 'failed' and len(row['wire_responses']) == 1
    assert Path(row['wire_responses'][0]['file']).read_bytes() == b'not json'
    assert row == batch.read(tmp_path / 'probes/000001-5y/result.json')


def test_worker_repeated_request_rejected_with_first_response_retained(tmp_path, monkeypatch):
    ak = frozen_api(tmp_path, monkeypatch)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', lambda *a, **k: mock_response(b'first response'))
    def twice(**kwargs):
        params = {'code': kwargs['symbol'], 'query': kwargs['indicator'], 'tag': kwargs['indicator'],
            'chart_select': kwargs['period'], 'market': 'ab'}
        batch.requests.get(batch.URL, params=params)
        batch.requests.get(batch.URL, params=params)
    monkeypatch.setattr(ak, 'stock_zh_valuation_baidu', twice)
    row = batch.worker(tmp_path, '000001-all')
    assert row['status'] == 'failed' and 'Unexpected or repeated' in row['error']
    assert len(row['attempts']) == len(row['wire_responses']) == 1
    assert Path(row['wire_responses'][0]['file']).read_bytes() == b'first response'


def test_api_evidence_JSON_roundtrip():
    pytest.importorskip('akshare')
    evidence = batch.api_evidence()
    assert evidence == json.loads(json.dumps(evidence, allow_nan=False))
