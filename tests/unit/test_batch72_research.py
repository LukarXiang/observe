import ast
import datetime as dt
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch72 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths):
        pytest.skip('Read-only strategy source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_original_full_definition_scope_and_line_numbers(monkeypatch):
    trees = originals(monkeypatch)
    assert [len(batch.definitions(t)) for t in trees] == [7, 7]
    assert [next(n.lineno for n in t.body if isinstance(n, ast.FunctionDef)) for t in trees] == [14, 8]
    assert not any(isinstance(n, ast.Import) and any(a.name == 'pandas' for a in n.names) for n in trees[1].body)
    assert any(isinstance(n, ast.Subscript) and ast.unparse(n) == 'df.loc[0:top]' for n in ast.walk(trees[1]))


def test_original_quarters_thresholds_cascade_ordering_and_failures(monkeypatch):
    originals(monkeypatch)
    result = batch.diagnostics(Path('unused'))
    cases = {r['case']: r for r in result['cases']}
    assert len(cases) == len(result['cases'])
    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert result['synthetic_fixture'] and result['not_a_backtest'] and not result['platform_equivalent']
    assert result['real_historical_trade_windows'] == 0
    assert cases['dividend70_monthly_five_and_no_explicit_slippage']['scheduled'] == [['main', [1], {'time': '9:30'}]]
    calls = cases['dividend70_HS300_PEgt0_current_fundamentals']['calls']
    assert calls[0] == '000300.XSHG'
    assert calls[1]['query']['filters'] == [['gt', 'pe_ratio', 0], ['in', 'income.code', ['A', 'B']]]
    assert calls[1]['kwargs'] == {}
    qualifying = cases['dividend70_qualifying_native_append_AttributeError']
    assert qualifying['error'] == 'AttributeError'
    assert qualifying['financial_locals']['one_year'] == 600
    assert qualifying['financial_locals']['two_year'] == 400
    assert qualifying['financial_locals']['three_year'] == 200
    assert qualifying['financial_locals']['market_cap'] == 10000
    query = cases['dividend70_query_report_date_is_not_availability']['query']
    assert query['count'] == 12 and query['sort'] == [['desc', 'report_date']]
    assert query['filters'] == [['eq', 'dividend.code', 'A'], ['le', 'report_date', '2024-12-31']]
    assert cases['dividend70_report_month_floor_division_q0_excluded']['result_empty']
    assert cases['dividend70_missing_report_date_fill_zero_AttributeError']['error'] == 'AttributeError'
    for count, length in ((10, 0), (11, 1), (19, 9), (20, 10), (100, 10)):
        case = cases[f'dividend70_cascade_count_{count}']
        assert len(case['bten']) == length and len(case['bfive']) == min(length, 5)
    assert cases['dividend70_cascade_count_100']['bten'] == [f'S{i:03}' for i in range(99, 89, -1)]
    assert cases['dividend70_cash_equality_blocked_at_price_mean_equality']['orders'] == []
    assert cases['dividend70_cash_above_half_intends_full_fifth']['orders'] == [['target_value', 'A', 200.]]
    assert cases['dividend70_immediate_sell_mutates_keys_RuntimeError']['error'] == 'RuntimeError'
    assert cases['dividend12_unheld_plain_dict_KeyError']['error'] == 'KeyError'
    for name in ('price_95percent_equality_no_buy', 'loss10_equality_no_add', 'today100_blocks_add',
            'today_positive_blocks_ST_sell', 'profit16_equality_no_sell', 'sell_MA_unused'):
        assert cases['dividend12_'+name]['orders'] == []
    for name in ('below_95percent_fixed_10000', 'loss_above10_adds_10000'):
        assert cases['dividend12_'+name]['orders'] == [['increment_value', 'A', 10000]]
    for name in ('ST_target_zero', 'profit_above16_target_zero'):
        assert cases['dividend12_'+name]['orders'] == [['target', 'A', 0]]
    assert cases['dividend12_None_buy_repeated_pool_and_positions']['orders'] == [['increment_value', 'A', 10000]]*2
    assert cases['dividend12_zero_cost_ZeroDivisionError']['error'] == 'ZeroDivisionError'
    selection = cases['dividend12_unused_Shanghai_pool_04_equality_first10']
    assert len(selection['selected']) == 10 and selection['unused_index'] == ['000001.XSHG']
    assert cases['dividend12_under10_choosePool_KeyError']['error'] == 'KeyError'
    query = cases['dividend12_PE10_25_totalcap_desc100_all_market']['query']
    assert query['query']['filters'] == [['lt', 'pe_ratio', 25], ['gt', 'pe_ratio', 10]]
    assert query['query']['sort'] == [['desc', 'market_cap']] and query['query']['count'] == 100
    assert query['kwargs'] == {'date': '2023-12-29'}
    assert cases['dividend12_native_DataFrame_sort_AttributeError']['error'] == 'AttributeError'


@pytest.mark.parametrize('code', [
    '@unsafe\ndef f():\n    pass',
    'def f(value=open("secret")):\n    pass',
    'def f(*, value=open("secret")):\n    pass',
    'def f(value: unsafe):\n    pass',
    'def f() -> unsafe:\n    pass',
])
def test_selected_rejects_eager_execution(monkeypatch, code):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(code))
    with pytest.raises(ValueError, match='Unsafe'):
        batch.selected(Path('unused'), 0, ['f'], {})


def test_original_three_distinct_cascade_rankings(monkeypatch):
    originals(monkeypatch)
    g = SimpleNamespace(stock_num=5)
    namespace = batch.selected(Path('unused'), 0, ['setSmallStocks'], {'g': g, 'log': SimpleNamespace(info=lambda *a: None)})
    data = [{'code': str(i), 'gxl': (i*37) % 101, 'gxl2': (i*53) % 101, 'gxl3': (i*71) % 101} for i in range(100)]
    expected = sorted(data, key=lambda r: r['gxl'], reverse=True)[:80]
    expected = sorted(expected, key=lambda r: r['gxl2'], reverse=True)[:40]
    expected = sorted(expected, key=lambda r: r['gxl3'], reverse=True)[:20]
    expected = sorted(expected, key=lambda r: r['gxl'], reverse=True)[:10]
    namespace['setSmallStocks'](pd.DataFrame(data))
    assert g.bten == [r['code'] for r in expected] and g.bfive == g.bten[:5]


def test_original_dividend_selection_nan_passes_negative_comparison(monkeypatch):
    originals(monkeypatch)
    g = SimpleNamespace()
    values = pd.DataFrame({'DividendRatio': [np.nan, .039, .04, np.inf, -np.inf]+[.05]*5,
        'SecuCode': [str(i) for i in range(10)]})
    namespace = batch.selected(Path('unused'), 1, ['choosePool'], {'g': g,
        'get_index_stocks': lambda *a: ['UNUSED'], 'DividendRatio2': lambda *a: values,
        'normalize_code': lambda code: code, 'log': SimpleNamespace(info=lambda *a: None)})
    namespace['choosePool'](SimpleNamespace(previous_date=dt.date(2020, 1, 2)))
    assert g.security == ['0', '2', '3', '5', '6', '7', '8', '9']


@pytest.mark.parametrize('name', ['corp-action-input.parquet', 'corp-input-binding.json', 'calendar-input.parquet',
    'supplier-PE-components.parquet', 'historical-mean-components.parquet', 'component-research.json'])
def test_prepare_rejects_partial_outputs_before_any_input_or_write(tmp_path, monkeypatch, name):
    (tmp_path / name).write_bytes(b'old partial output')
    before = file_sha(tmp_path / name)
    def denied(*a, **k): raise AssertionError('Inputs must not be touched')
    monkeypatch.setattr(batch, 'binding', denied)
    with pytest.raises(ValueError, match='Component output exists'):
        batch.prepare('unused', tmp_path)
    assert file_sha(tmp_path / name) == before and len(list(tmp_path.iterdir())) == 1


def test_binding_rejects_changed_local_price_copy(tmp_path, monkeypatch):
    checks = {'evidence_sha256': {'price-input.parquet': 'stable', 'daily-predicate-values.parquet': 'stable'}}
    receipt = {'status': 'ok', 'snapshot': batch.SNAPSHOT, 'checks': checks}
    proof = {'status': 'ok', 'receipt_sha256': 'stable', 'checked_state_sha256': 'stable'}
    documents = {batch.RECEIPT: receipt, batch.ACCEPTED / 'final-binding-verification.json': proof,
        Path('docs/handoff/2026-10-07-batch47-verification.json'): {'checks': checks}}
    monkeypatch.setattr(batch, 'read', lambda name: documents[Path(name)])
    monkeypatch.setattr(batch.previous, 'binding', lambda *a: {'upstream': 'stable'})
    monkeypatch.setattr(batch, 'file_sha', lambda name: 'damaged' if Path(name) == tmp_path / 'price-input.parquet' else 'stable')
    documents[tmp_path / 'input-binding.json'] = batch.binding('unused')
    with pytest.raises(ValueError, match='Frozen price copy changed'):
        batch.binding('unused', tmp_path)


def PE_trees():
    return [ast.parse('valuation.pe_ratio > 0'), ast.parse('valuation.pe_ratio < 25\nvaluation.pe_ratio > 10')]


def PE_input(values):
    return pd.DataFrame({'date': ['2026-09-29']*len(values), 'instrument': [str(i) for i in range(len(values))],
        'pe_ttm': values, 'is_trading': [True]*len(values)})


def test_PE_strict_bounds_nonfinite_cutoff_and_duplicate():
    pe = [-1., 0., np.nextafter(0., 1.), 10., np.nextafter(10., 11.), 25., np.nan, np.inf, -np.inf]
    frame = PE_input(pe)
    values, summary = batch.numeric_PE(frame, PE_trees())
    assert list(values.source70_PE_gt0) == [False, False, True, True, True, True, False, True, False]
    assert list(values.source12_PE_10_25) == [False, False, False, False, True, False, False, False, False]
    assert summary['not_a_backtest'] and not summary['original_signal_reproduced']
    frame.loc[0, 'date'] = '2026-09-30'
    assert batch.numeric_PE(frame, PE_trees())[1]['excluded_after_cutoff'] == 1
    with pytest.raises(ValueError, match='Invalid daily identities'):
        batch.numeric_PE(pd.concat([frame, frame.iloc[[1]]]), PE_trees())


@pytest.mark.parametrize('resolution', ['s', 'us', 'ns'])
def test_PE_dates_lossless_parquet_roundtrip(tmp_path, resolution):
    frame = PE_input([11.])
    frame['date'] = pd.to_datetime(frame.date).astype(f'datetime64[{resolution}]')
    values, _ = batch.numeric_PE(frame, PE_trees())
    values.to_parquet(tmp_path / 'values.parquet', index=False)
    pd.testing.assert_frame_equal(values, pd.read_parquet(tmp_path / 'values.parquet'), check_exact=True)
    assert str(values.date.dtype) == 'datetime64[ms]'


def test_PE_date_precision_loss_rejected():
    frame = PE_input([11.])
    frame['date'] = [pd.Timestamp('2026-09-29 00:00:00.000000001')]
    with pytest.raises(ValueError, match='Daily date precision loss'):
        batch.numeric_PE(frame, PE_trees())


@pytest.mark.parametrize('expression', ['valuation.pe_ratio >= 0', 'valuation.pe_ratio > 1', 'other.pe_ratio > 0'])
def test_PE_changed_source_rule_rejected(expression):
    with pytest.raises(ValueError, match='PE'):
        batch.numeric_PE(PE_input([11.]), [ast.parse(expression), PE_trees()[1]])


def history_fixture():
    days = pd.bdate_range('2020-01-02', periods=38)
    calendar = pd.DataFrame({'date': days, 'is_open': True})
    price = pd.DataFrame({'date': days, 'instrument': 'A', 'is_trading': True,
        'close_adj': np.arange(1., 39.)*2, 'back_factor': 2.})
    tree = ast.parse("thirty_mean = stock_hitory['close'][-30:].mean()")
    return price, calendar, tree


def test_35_row_mean_uses_last30_and_dynamic_basis():
    price, calendar, tree = history_fixture()
    values, summary = batch.numeric_means(price, calendar, tree)
    assert len(values) == 4 and values.iloc[0].hist35_last30_mean == 20.5
    assert values.iloc[0].history_end == str(price.iloc[34].date.date())
    price.loc[34, 'back_factor'] = 4.
    assert batch.numeric_means(price, calendar, tree)[0].iloc[0].hist35_last30_mean == 10.25
    assert summary['not_a_backtest']


def test_history_known_pause_skipped_unknown_gap_and_invalid_price_blocked():
    price, calendar, tree = history_fixture()
    price.loc[10, 'is_trading'] = False
    values, summary = batch.numeric_means(price, calendar, tree)
    assert len(values) == 3 and summary['known_pause_rows_excluded'] == 1
    assert summary['invalid_or_unknown_windows_blocked'] == 0
    values, summary = batch.numeric_means(price.drop(index=10), calendar, tree)
    assert values.empty and summary['invalid_or_unknown_windows_blocked'] == 3
    price, calendar, tree = history_fixture()
    price.loc[20, 'close_adj'] = np.nan
    assert batch.numeric_means(price, calendar, tree)[0].empty
    with pytest.raises(ValueError, match='Duplicate'):
        batch.numeric_means(pd.concat([price, price.iloc[[0]]]), calendar, tree)


def frozen_api(tmp_path, monkeypatch):
    ak = pytest.importorskip('akshare')
    monkeypatch.setattr(batch, 'api_evidence', lambda: {'frozen': True})
    batch.save(tmp_path / 'existing-apis.json', {'frozen': True})
    return ak


def wire_cninfo():
    return json.dumps({'records': [{'F006D': '2020-06-01', 'F044V': '年度分红', 'F011N': 0,
        'F010N': 0, 'F012N': 2., 'F018D': '2020-06-05', 'F020D': '2020-06-08',
        'F023D': '2020-06-08', 'F025D': None, 'F007V': '每10股派2元', 'F001V': '2019年报'}]}, ensure_ascii=False).encode()


def wire_sina():
    frame = pd.DataFrame([['2020-06-01', 0, 0, 2, '实施', '2020-06-08', '2020-06-05', '-', '详细']],
        columns=pd.MultiIndex.from_tuples([('分红', '分红', str(i)) for i in range(9)]))
    html = '<html><body>'+pd.DataFrame({'dummy': [1]}).to_html()*12+frame.to_html(index=False)+'</body></html>'
    return html.encode('gbk')


def response(raw, status=200, encoding='utf-8'):
    result = batch.requests.Response()
    result.status_code = status; result._content = raw; result.encoding = encoding
    return result


@pytest.mark.parametrize('case', ['sina-000001', 'cninfo-000001'])
def test_worker_single_request_response_encoding_archive(tmp_path, monkeypatch, case):
    frozen_api(tmp_path, monkeypatch); calls = []
    raw = wire_sina() if case.startswith('sina') else wire_cninfo()
    def request(session, method, url, **kwargs):
        calls.append((method.upper(), url, kwargs.get('params') or {}))
        assert not session.trust_env and kwargs['timeout'] == (8, 10) and not kwargs['allow_redirects']
        return response(raw, encoding='gbk' if case.startswith('sina') else 'utf-8')
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', request)
    row = batch.worker(tmp_path, case)
    assert row['status'] == 'success' and row['rows'] == 1
    assert calls == [batch.expected_request(case)] and len(row['attempts']) == len(row['wire_responses']) == 1
    assert Path(row['wire_responses'][0]['file']).read_bytes() == raw
    assert not row['strict_usable'] and not row['published']
    assert 'Accept-Enckey' not in json.dumps(row)
    with pytest.raises(FileExistsError): batch.worker(tmp_path, case)


@pytest.mark.parametrize('status,raw', [(301, b'redirect'), (200, b'not json')])
def test_failed_request_or_parse_keeps_wire(tmp_path, monkeypatch, status, raw):
    frozen_api(tmp_path, monkeypatch)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', lambda *a, **k: response(raw, status))
    row = batch.worker(tmp_path, 'cninfo-000001')
    assert row['status'] == 'failed' and row['error'] and len(row['wire_responses']) == 1
    assert Path(row['wire_responses'][0]['file']).read_bytes() == raw
    assert row == batch.read(tmp_path / 'probes/cninfo-000001/result.json')


def test_repeated_request_rejected_retains_first_response(tmp_path, monkeypatch):
    ak = frozen_api(tmp_path, monkeypatch)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', lambda *a, **k: response(b'first'))
    def twice(**kwargs):
        method, url, params = batch.expected_request('cninfo-000001')
        batch.requests.request(method, url, params=params)
        batch.requests.request(method, url, params=params)
    monkeypatch.setattr(ak, 'stock_dividend_cninfo', twice)
    row = batch.worker(tmp_path, 'cninfo-000001')
    assert row['status'] == 'failed' and 'Unexpected or repeated' in row['error']
    assert len(row['attempts']) == len(row['wire_responses']) == 1


def test_saved_native_parsers_replay_with_sockets_disabled(tmp_path, monkeypatch):
    frozen_api(tmp_path, monkeypatch)
    def request(session, method, url, **kwargs):
        return response(wire_sina(), encoding='gbk') if method.upper() == 'GET' else response(wire_cninfo())
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', request)
    rows = [batch.worker(tmp_path, case) for case in batch.QUERIES]
    assert all(r['status'] == 'success' for r in rows)
    batch.save(tmp_path / 'probe-results.json', {'results': rows})
    pd.DataFrame({'instrument': ['000001.SZ'], 'ex_date': [dt.date(2020, 6, 8)],
        'cash_per_share': [.2]}).to_parquet(tmp_path / 'corp-action-input.parquet', index=False)
    def denied(*a, **k): raise AssertionError('Network access')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(batch.requests.sessions.Session, 'request', denied)
    result = batch.parse_probes(tmp_path)
    assert len(result['profiles']) == 6 and all(p['native_parser_match'] for p in result['profiles'])
    assert result['new_admitted_rows'] == result['new_trading_backtests'] == 0
    assert all('JY_daily_DividendRatio' in p['missing_original_fields'] for p in result['profiles'])


def test_installed_api_evidence_roundtrip_and_scope():
    pytest.importorskip('akshare')
    evidence = batch.api_evidence()
    assert evidence == json.loads(json.dumps(evidence, allow_nan=False))
    assert len(evidence['queries']) == 6 and len(evidence['functions']) == 2
    assert len(evidence['cninfo_installed_js_sha256']) == 64


def test_parser_date_normalization_handles_all_missing_without_losing_values(tmp_path):
    frame = pd.DataFrame({'missing': pd.Series([pd.NaT], dtype='datetime64[s]'),
        'date': pd.to_datetime(['2020-06-01']), 'value': [2.]})
    result = batch.parquet_dates(frame)
    result.to_parquet(tmp_path / 'dates.parquet', index=False)
    pd.testing.assert_frame_equal(result, pd.read_parquet(tmp_path / 'dates.parquet'), check_exact=True)
    assert str(frame.missing.dtype) == 'datetime64[s]'
    with pytest.raises(ValueError, match='Parser date precision loss'):
        batch.parquet_dates(pd.DataFrame({'date': [pd.Timestamp('2020-06-01 00:00:00.000000001')]}))


def test_parser_mixed_dates_preserve_missingness_and_finite_values(tmp_path):
    day = dt.date(2020, 6, 1)
    original = pd.DataFrame({'date': [day, pd.NaT], 'value': [2., np.nan]})
    result = batch.parquet_dates(original)
    assert result.date.tolist() == [day, None] and original.date.iloc[1] is pd.NaT
    assert result.value.iloc[0] == 2. and np.isnan(result.value.iloc[1])
    result.to_parquet(tmp_path / 'mixed.parquet', index=False)
    pd.testing.assert_frame_equal(result, pd.read_parquet(tmp_path / 'mixed.parquet'), check_exact=True)
