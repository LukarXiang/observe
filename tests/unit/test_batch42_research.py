from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch42 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def cases(directory):
    with redirect_stdout(io.StringIO()): result = batch.diagnostics(directory)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result['model_loaded'] is False and result['not_a_backtest'] is True
    return {r['case']: r for r in result['cases']}


def test_original_pool_dates_and_price_slope_order(sources):
    result = cases(sources)
    assert result['trend0_date_and_absolute_slope_order']['requests'][0]['kwargs'] == {'date': '2021-01-04'}
    assert result['trend1_date_and_absolute_slope_order']['requests'][0]['kwargs'] == {}
    for n in (0, 1):
        assert result[f'trend{n}_date_and_absolute_slope_order']['targets'] == ['B', 'A']
        assert result[f'trend{n}_rejected_intents_sell_adjust_new']['orders'] == [['old', 0], ['B', 500.], ['new', 500.]]


def test_bull_original_inverted_gate_and_current_bar_request(sources):
    result = cases(sources)
    assert result['bull_sells_before_return_no_buy']['orders'] == [['old', 0]]
    assert result['bull_original_request_includes_now']['requests'] == [
        {'args': ['000001.XSHG', 10, '1d', 'close'], 'kwargs': {'include_now': True}}]
    assert result['bull_hysteresis_keeps_true_at_mean']['is_bull'] is True
    assert result['bull_down_switch']['is_bull'] is False


def test_dqn_shapes_cash_target_and_execution_flags(sources):
    result = cases(sources)
    assert result['dqn_close_input_six']['dimensions'] == 6
    assert result['dqn_default_other_missing_reduces_input']['dimensions'] == 5
    assert result['dqn_synthetic_action0']['orders'] == []
    assert result['dqn_synthetic_action1']['orders'] == [['000065.XSHE', 200.]]
    assert result['dqn_synthetic_action1']['logs'] == ['买入', '持有']
    assert result['dqn_synthetic_action2']['orders'] == [['000065.XSHE', 0]]
    assert result['dqn_empty_sell_fault']['error'].startswith('KeyError:')
    assert result['dqn_unscheduled_uninitialised_helper']['error'].startswith('AttributeError:')
    assert result['dqn_partial_buy_true']['result'] is True
    assert result['dqn_partial_sell_false']['result'] is False
    assert result['dqn_full_held_sell_true']['result'] is True


def test_original120_regression_and_inclusive_filters(sources):
    kernel, _ = batch.trend_kernel(sources)
    close = 100.+np.arange(120); high = np.repeat(close[-1]*1.1, 30); volume = np.ones(180)
    actual, flags = batch.original_trend(close, high, volume, kernel)
    expected, ref_flags = batch.reference_trend(close, high, volume)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert flags == ref_flags == [True, True, True, True]
    high[-1] += .0001
    assert batch.original_trend(close, high, volume, kernel)[1][1] is False


def test_duplicate_dqn_return_kernel(sources):
    code, _ = batch.dqn_kernel(sources); ns = {'df': pd.DataFrame({'close': [1., 2., 4., 8., 16., 32., 64.]})}
    exec(code, ns)
    assert ns['df']['涨跌幅'].tolist() == [1.]*6


def test_incomplete_trend_window_does_not_become_signal(sources):
    kernel, _ = batch.trend_kernel(sources)
    with pytest.raises(ValueError, match='Invalid complete trend operands'):
        batch.original_trend(np.ones(119), np.ones(30), np.ones(180), kernel)


def test_components_count_unique_windows_and_never_infer(sources, monkeypatch):
    price = pd.DataFrame({'instrument': ['stock']*182, 'date': pd.bdate_range('2005-01-05', periods=182).strftime('%Y-%m-%d'),
        'close_adj': 100.+np.arange(182), 'high_adj': 100.+np.arange(182), 'back_factor': np.ones(182), 'volume': np.ones(182)})
    dqn = price.iloc[:9].copy(); index = price.iloc[:12].copy().rename(columns={'close_adj': 'close'})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'input_frames', lambda *a: (price, dqn, 0, 0))
    monkeypatch.setattr(batch, 'inputs', lambda *a: (index, []))
    result = batch.compute(sources)
    assert result['not_a_backtest'] is True and result['platform_equivalent'] is False
    assert result['trend']['unique_windows'] == 3
    assert result['dqn']['unique_windows'] == 3 and result['dqn']['inference'] is False
    assert result['bull_operator']['windows'] == 3 and result['bull_operator']['state_mismatches'] == []
    assert result['trend']['condition_boundaries'] == []
    monkeypatch.setattr(batch, 'input_frames', lambda *a: (price.iloc[:179], dqn, 0, 0))
    with pytest.raises(ValueError, match='Incomplete component history'): batch.compute(sources)


def test_source_corruption_blocks_selected_code(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0'*64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.trend_kernel(sources)
    with pytest.raises(ValueError, match='Selected source changed'): batch.diagnostics(sources)


def test_probe_sample_does_not_publish_or_become_strict(tmp_path, monkeypatch):
    import akshare as ak
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'stock_zh_index_daily_em', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(batch.raw, 'save', save)
    row = batch.worker('diagnostic-root', tmp_path, 'index_daily')
    assert row['status'] == 'sample' and row['strict_usable'] is False and row['published'] is False
    assert calls == [['batch42_dependency_probe', 'index_daily', tmp_path.name]]


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_substitution_rejected(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0'*64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted', 'sha256': '0'*64}]
        if change == 'status': row['status'] = 'accepted'
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises((ValueError, FileNotFoundError)): batch.validate_probes(tmp_path)
