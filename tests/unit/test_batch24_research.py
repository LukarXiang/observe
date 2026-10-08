import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.data import standardize
from observe.data.store import fingerprint
from observe.runs import file_sha, write_json
from scripts import review_strategy_batch24 as batch


def test_gate_uses_last_four_prices_and_excludes_latest_price_from_amplitudes():
    prices = [130, 120, 110, 100, 90, 100, 110, 110, 110, 110, 110, 109, 108, 107, 106, 105, 100, 102, 104, 110]
    assert batch.reference_gate(prices) is False
    prices[-1] = 99
    assert batch.reference_gate(prices) is True


@pytest.mark.parametrize('defect', ['short', 'nan', 'inf', 'zero', 'negative'])
def test_bad_gate_prices_block(defect):
    prices = [100.] * 20
    if defect == 'short': prices.pop()
    elif defect == 'nan': prices[0] = np.nan
    elif defect == 'inf': prices[0] = np.inf
    elif defect == 'zero': prices[0] = 0
    elif defect == 'negative': prices[0] = -1
    with pytest.raises(ValueError, match='Invalid gate inputs'): batch.reference_gate(prices)


def test_cross_section_uses_sample_std_and_preserves_zero_dispersion_nan():
    actual = batch.reference_z([[1., 5.], [2., 5.], [3., 5.]])
    np.testing.assert_array_equal(actual[:, 0], [-1., 0., 1.])
    assert np.isnan(actual[:, 1]).all()
    with pytest.raises(ValueError): batch.reference_z([[1., np.inf], [2., 3.]])


def test_wave_matches_below_mean_denominator_for_positive_returns():
    returns = np.tile([[.01, .002, .005], [.02, .003, .001], [.03, .001, .002]], (21, 1))
    prices = np.concatenate([np.ones((1, 3)) * 100, np.cumprod(1 + returns, axis=0) * 100])
    ret = pd.DataFrame(prices).pct_change().iloc[1:]
    below_mean = (((ret - ret.mean()) * (ret < ret.mean())) ** 2).sum().div(62).pow(.5)
    assert below_mean.gt(0).all() and ret.gt(0).all().all()
    factors = pd.DataFrame({'his': ret.std(), 'down': below_mean})
    expected = ((factors - factors.mean()) / factors.std()).mean(axis=1)
    np.testing.assert_allclose(batch.reference_wave(prices), expected, atol=1e-12)


@pytest.mark.parametrize('defect', ['short', 'one_stock', 'zero', 'nan', 'inf'])
def test_bad_wave_inputs_block(defect):
    prices = np.ones((64, 3)) * 100
    if defect == 'short': prices = prices[:-1]
    elif defect == 'one_stock': prices = prices[:, :1]
    elif defect == 'zero': prices[0, 0] = 0
    elif defect == 'nan': prices[0, 0] = np.nan
    elif defect == 'inf': prices[0, 0] = np.inf
    with pytest.raises(ValueError, match='Invalid wave inputs'): batch.reference_wave(prices)


def valuation():
    return pd.DataFrame({'date': ['2024-03-29'] * 2, 'code': [standardize.to_baostock(batch.POOL[0]), 'sh.600000'],
        'peTTM': ['-3.5', '9'], 'pbMRQ': ['1.2', '1'], 'psTTM': ['2.4', '2'], 'pcfNcfTTM': ['99.0', '3']})


def test_raw_ps_recovered_independently_of_pcf_and_negative_pe_retained():
    result = batch.valuation_sample(valuation(), '2024-03-29')
    assert len(result) == 1 and result.instrument.iloc[0] == batch.POOL[0]
    assert result.pe_ttm.iloc[0] == -3.5 and result.ps_ttm.iloc[0] == 2.4
    assert result.pcf_ncf_ttm.iloc[0] == 99.
    frame = valuation(); frame.loc[0, 'psTTM'] = ''
    assert pd.isna(batch.valuation_sample(frame, '2024-03-29').ps_ttm.iloc[0])


@pytest.mark.parametrize('defect', ['schema', 'date', 'duplicate', 'bad_number', 'inf'])
def test_raw_value_contract_rejects_unverified_inputs(defect):
    frame = valuation()
    if defect == 'schema': frame = frame.drop(columns='psTTM')
    elif defect == 'date': frame.loc[0, 'date'] = '2024-03-28'
    elif defect == 'duplicate': frame.loc[1, 'code'] = frame.loc[0, 'code']
    elif defect == 'bad_number': frame.loc[0, 'psTTM'] = 'unknown'
    elif defect == 'inf': frame.loc[0, 'psTTM'] = 'inf'
    with pytest.raises(ValueError): batch.valuation_sample(frame, '2024-03-29')


def test_index_sample_cannot_use_other_index_or_fill_missing_session():
    frame = pd.DataFrame({'date': ['2005-01-05', '2005-01-06'], 'code': ['sh.000001'] * 2, 'close': ['1251.937', '1239.430']})
    dates = frame.date.tolist()
    assert batch.index_sample(frame, dates).close.tolist() == [1251.937, 1239.43]
    with pytest.raises(ValueError): batch.index_sample(frame.iloc[:1], dates)
    frame.loc[0, 'code'] = 'sh.000300'
    with pytest.raises(ValueError): batch.index_sample(frame, dates)


def test_known_pause_null_is_retained_and_blocks_full_return_window():
    data = pd.DataFrame({'date': ['2024-03-29'] * len(batch.POOL), 'instrument': batch.POOL,
        'close_adj': 100., 'is_trading': True, 'back_factor': 1., 'adjustment_status': 'usable'})
    data.loc[0, 'is_trading'] = False; data.loc[0, 'close_adj'] = np.nan
    assert pd.isna(batch.validate_prices(data).close_adj.iloc[0])
    prices = np.ones((64, 10)) * 100; flags = np.ones((64, 10), dtype=bool)
    assert batch.wave_window_usable(prices, flags)
    prices[0, 0] = np.nan; flags[0, 0] = False
    assert not batch.wave_window_usable(prices, flags)
    prices[0, 0] = 100
    assert not batch.wave_window_usable(prices, flags)
    data.loc[0, 'is_trading'] = True
    with pytest.raises(ValueError, match='Invalid traded price'): batch.validate_prices(data)


def test_stock_partition_fingerprint_and_bytes_are_both_bound(tmp_path, monkeypatch):
    frame = pd.DataFrame({'date': ['2024-03-29'], 'instrument': [batch.POOL[0]], 'close': [100.]})
    tables = {}
    for table in ('bars_1d', 'adj_factors', 'adj_coverage'):
        path = tmp_path / f'{table}.parquet'; frame.to_parquet(path, index=False)
        tables[table] = {'all': {'file': path.name, 'sha': fingerprint(frame), 'rows': 1}}
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(root=tmp_path, state=lambda snapshot: {'tables': tables}))
    bound = batch.stock_binding(tmp_path); batch.verify_stock_binding(tmp_path, bound)
    path = tmp_path / 'bars_1d.parquet'; frame.assign(close=200.).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Accepted stock partition differs'): batch.stock_binding(tmp_path)
    with pytest.raises(ValueError, match='Stock partition bytes/binding differs'): batch.verify_stock_binding(tmp_path, bound)


def test_lof_diagnostic_context_has_time_before_logging(monkeypatch):
    class Reached(Exception): pass
    def selected(directory, number, names, ns):
        if number == 0: ns['indexwarn'] = lambda ctx: (_ for _ in ()).throw(NameError('floor'))
        elif number == 1:
            name = names[0]
            error = {'get_holding_list': AttributeError('sort'), 'set_feasible_stocks': TypeError('paused'), 'rebalance': ZeroDivisionError()}[name]
            ns[name] = lambda *args: (_ for _ in ()).throw(error)
        elif number == 2:
            def callback(context):
                assert context.current_dt.time().hour == 9
                raise Reached()
            ns['market_open'] = callback
        return 'selected'
    monkeypatch.setattr(batch, 'selected', selected)
    with pytest.raises(Reached): batch.diagnostics(None, None)


def test_changed_upstream_raw_and_terminal_rejected(tmp_path, monkeypatch):
    terminal = tmp_path / 'terminal.json'; receipt = tmp_path / 'receipt.json'; raw = tmp_path / 'raw.parquet'
    upstream = tmp_path / 'upstream.json'; baseline = tmp_path / 'baseline.json'; state = {'tables': {}}
    pd.DataFrame({'close': [1.]}).to_parquet(raw)
    row = {'status': 'success', 'published': False, 'parameters': {'code': 'sh.000001', 'start': '2005-01-05', 'end': '2026-09-29', 'adjustflag': '3'},
        'raw_file': str(raw), 'raw_sha256': file_sha(raw)}
    write_json(terminal, row)
    write_json(receipt, {'status': 'ok_with_deferred_strategies', 'probes': {'results': [{'endpoint': 'bs_000001', 'result': row,
        'evidence_files': [{'file': str(terminal), 'sha256': file_sha(terminal)}]}]}})
    write_json(baseline, {'published': state})
    write_json(upstream, {'status': 'ok', 'reviews': {'snapshot': batch.SNAPSHOT}, 'checks': {'evidence_sha256': {'baseline.json': file_sha(baseline)}}})
    monkeypatch.setattr(batch, 'INDEX_TERMINAL', terminal); monkeypatch.setattr(batch, 'INDEX_RECEIPT', receipt)
    monkeypatch.setattr(batch, 'UPSTREAM', upstream); monkeypatch.setattr(batch, 'BASELINE', baseline)
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(state=lambda snapshot: state))
    assert batch.binding(tmp_path)['index_raw_sha256'] == file_sha(raw)
    pd.DataFrame({'close': [2.]}).to_parquet(raw)
    with pytest.raises(ValueError, match='Accepted index raw differs'): batch.binding(tmp_path)
    row['raw_sha256'] = file_sha(raw); write_json(terminal, row)
    with pytest.raises(ValueError, match='Accepted index probe differs'): batch.binding(tmp_path)


@pytest.fixture
def probe_evidence(tmp_path):
    import akshare as ak
    import baostock as bs
    apis = []
    for fn in (bs.query_history_k_data_plus, ak.fund_etf_hist_sina):
        code = inspect.getsource(fn); apis.append({'name': fn.__name__, 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'signature': str(inspect.signature(fn))})
    write_json(tmp_path / 'existing-apis.json', {'akshare_version': ak.__version__, 'baostock_version': bs.__version__, 'apis': apis, 'queries': batch.QUERIES})
    rows = []
    for endpoint in batch.QUERIES:
        row = {'endpoint': endpoint, 'parameters': batch.QUERIES[endpoint], 'status': 'failed', 'files': [], 'wire_responses': [],
            'published': False, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'), 'error': 'archived failure'}
        rows.append(row); write_json(tmp_path / 'probes' / endpoint / 'result.json', row)
    write_json(tmp_path / 'probe-results.json', {'results': rows, 'published': False})
    return tmp_path, rows


@pytest.mark.parametrize('defect', ['query', 'status', 'api_sha', 'raw', 'terminal'])
def test_probe_contract_blocks_self_consistent_or_stale_metadata(probe_evidence, defect):
    directory, rows = probe_evidence
    assert len(batch.validate_probes(directory)) == 2
    if defect == 'query': rows[0]['parameters'] = {'symbol': 'wrong'}
    elif defect == 'status': rows[0]['status'] = 'unknown'
    elif defect == 'api_sha': rows[0]['api_sha256'] = 'wrong'
    elif defect == 'raw': rows[0]['files'] = [{'file': 'unverified'}]
    elif defect == 'terminal': rows[0]['error'] = 'different'
    if defect != 'terminal': write_json(directory / 'probes' / rows[0]['endpoint'] / 'result.json', rows[0])
    write_json(directory / 'probe-results.json', {'results': rows, 'published': False})
    with pytest.raises(ValueError): batch.validate_probes(directory)


@pytest.mark.parametrize('defect', ['not_backtest', 'pool', 'rows', 'first', 'path', 'constituents'])
def test_input_profile_tamper_cannot_be_accepted(tmp_path, monkeypatch, defect):
    dates = ['2024-03-29']; inputs = {}
    for name in ('index', 'price', 'value'):
        frame = pd.DataFrame({'date': dates}) if name == 'index' else pd.DataFrame({'date': dates * len(batch.POOL), 'instrument': batch.POOL})
        path = tmp_path / f'{name}-input.parquet'; frame.to_parquet(path)
        inputs[name] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'first': dates[0], 'last': dates[0]}
    info = {'snapshot': batch.SNAPSHOT, 'inputs': inputs, 'calendar': dates, 'research_pool': batch.POOL,
        'local_constituent_tables': [], 'not_a_backtest': True}
    monkeypatch.setattr(batch, 'calendar_dates', lambda root: dates)
    monkeypatch.setattr(batch, 'Store', lambda root: SimpleNamespace(state=lambda snapshot: {'tables': {}}))
    write_json(tmp_path / 'input-analysis.json', info); batch.validate_inputs(tmp_path, tmp_path)
    if defect == 'not_backtest': info['not_a_backtest'] = False
    elif defect == 'pool': info['research_pool'] = ['fake']
    elif defect == 'rows': info['inputs']['index']['rows'] = 2
    elif defect == 'first': info['inputs']['index']['first'] = '2021-01-04'
    elif defect == 'path': info['inputs']['index']['file'] = 'other'
    elif defect == 'constituents': info['local_constituent_tables'] = ['2024']
    write_json(tmp_path / 'input-analysis.json', info)
    with pytest.raises(ValueError): batch.validate_inputs(tmp_path, tmp_path)
