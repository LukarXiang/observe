from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from observe.strategy_catalog import catalog_strategies
from scripts import review_strategy_batch41 as batch


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
    return {row['case']: row for row in result['cases']}


@pytest.mark.parametrize('legacy', [False, True])
def test_history_financial_gap_without_field_qualifier(tmp_path, legacy):
    folder = tmp_path / 'source'; folder.mkdir()
    code = 'def trade(context):\n    get_history_fundamentals(stocks, fields=fields)\n    order_value("600000.XSHG", 1000)\n'
    if legacy: code += '    print context.current_dt\n'
    (folder / 'test.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', folder)
    row = batch.read(Path(result['output']) / 'catalog.json')[0]
    assert row['financial_fields'] == []
    assert '历史财务版本/公告时间或每日市值/股本待核实' in row['gaps']
    assert row['status'] == '待数据'


def test_stop_cache_and_date_fault(sources):
    result = cases(sources); cache = result['stop_cache_old_pre_scale_not_reanchored']
    assert cache['cache_length'] == 100 and cache['first'] == 100. and cache['last'] == 50.
    assert cache['requests'][0][1] == {'skip_paused': True, 'df': True, 'fq': 'pre'}
    assert result['stop_update_date_integer_fault']['error'] == 'KeyError: -1'
    records = result['stop_record_passes_series_to_platform']['records']
    assert len(records) == 4 and all(next(iter(r.values()))['type'] == 'Series' for r in records)


def test_stop_boundary_restore_and_switch(sources):
    result = cases(sources)
    assert result['stop_yesterday_boundary_96.0']['worth'] == 0
    assert result['stop_yesterday_boundary_96.00000001']['worth'] == 1
    assert result['stop_recovery_mean_equality']['worth'] == 1
    assert result['stop_same_selected_no_same_call_rebuy']['orders'] == [['old', 0]]
    assert result['stop_empty_below_mean_no_restore']['orders'] == []
    assert result['stop_empty_mean_restore']['orders'] == [['old', 100.]]
    assert result['stop_switch_ignores_selected_mean_gate']['orders'] == [['old', 0], ['new', 100.]]
    assert result['stop_only_first_holding_sold']['orders'] == [['old1', 0], ['new', 100.]]


def test_weighted_bounds_and_holdings(sources):
    result = cases(sources)
    assert result['weighted_rank_strict_lower_inclusive_upper']['ranks'] == ['five', 'middle']
    assert result['weighted_all_negative_safe_fund']['ranks'] == ['511880.XSHG']
    assert result['weighted_held_target_not_rebalanced']['orders'] == []
    assert result['weighted_rejected_sell_blocks_new_buy']['orders'] == [['old', 0]]
    assert 'NameError' in result['weighted_math_requires_platform_injection']['error']
    assert result['weighted_constant_score_nonfinite_excluded']['finite'] is False
    assert result['weighted_constant_score_nonfinite_excluded']['eligible'] is False


def test_opening_strict_boundaries_and_actual_requests(sources):
    result = cases(sources)
    for name in ('open_equal_one_empty_no_buy', 'open_equal_below_one_held_keep', 'open_above_other_held_keep'):
        assert result[name]['orders'] == []
    for name in ('open_equal_above_one_buy', 'open_beats_other_below_one_buy'):
        assert result[name]['orders'] == [{'args': ['510180.XSHG'], 'kwargs': {'value': 100., 'side': 'long'}}]
    assert result['open_below_other_below_one_sell']['orders'] == [{'args': ['510180.XSHG', 0], 'kwargs': {}}]
    requests = result['open_equal_above_one_buy']['requests']
    assert [r[0][0] for r in requests] == ['399300.XSHE', '399303.XSHE']
    assert result['open_original_date_integer_fault']['error'] == 'KeyError: 0'


def test_mixed_weight_formula_matches_independent_centered_regression(sources):
    sample = np.exp(np.linspace(0., .05, 25) + .03*np.sin(np.arange(25)))
    scores = batch.original_scores(sources, sample)
    assert scores == pytest.approx(batch.reference_scores(sample), rel=1e-10, abs=1e-10)
    assert scores[0] != pytest.approx(scores[1], rel=1e-5, abs=1e-5)


def test_source_corruption_blocks_math_and_diagnostics(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][0]['source_sha256'] = '0'*64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'): batch.score_kernel(sources)
    with pytest.raises(ValueError, match='Selected source changed'): batch.diagnostics(sources)


def test_component_complete_windows_and_no_trading_claim(sources, monkeypatch):
    frame = pd.DataFrame({'date': pd.bdate_range('2005-01-05', periods=30).strftime('%Y-%m-%d'), 'close': 100.+np.arange(30)*.1})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'inputs', lambda *a: (frame, []))
    result = batch.compute(sources)
    assert result['windows_per_formula'] == 6 and result['weighted_score_boundaries'] == []
    assert result['not_a_backtest'] is True and result['platform_equivalent'] is False
    assert max(result['max_score_differences']) < 1e-9
    monkeypatch.setattr(batch, 'inputs', lambda *a: (frame.iloc[:24], []))
    with pytest.raises(ValueError, match='Incomplete score history'): batch.compute(sources)


def test_probe_sample_is_unproven_batch41_archive(tmp_path, monkeypatch):
    import akshare as ak
    api = {'apis': []}; calls = []; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    monkeypatch.setattr(ak, 'fund_etf_hist_em', lambda **kw: pd.DataFrame({'close': [1.]}))
    def save(root, upstream, endpoint, key, frame):
        calls.append([upstream, endpoint, key]); path = tmp_path / 'raw.parquet'; frame.to_parquet(path, index=False); return path
    monkeypatch.setattr(batch.raw, 'save', save)
    row = batch.worker('diagnostic-root', tmp_path, 'fund_daily')
    assert row['status'] == 'sample' and row['strict_usable'] is False and row['published'] is False
    assert calls == [['batch41_dependency_probe', 'fund_daily', tmp_path.name]]


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data', 'status'])
def test_probe_substitution_rejected(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api); monkeypatch.setattr(batch, 'api_evidence', lambda: api)
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
