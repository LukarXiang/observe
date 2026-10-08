from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch34 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows})
    return tmp_path


def test_original_shared_functions_do_not_merge_distinct_pools_options(sources):
    _, hashes = batch.kernels(sources)
    assert len(hashes) == 8
    a, acalls = batch.configuration(sources, 0); b, bcalls = batch.configuration(sources, 1)
    assert a.stock_pool == ['159915.XSHE', '510300.XSHG', '510500.XSHG']
    assert b.stock_pool == ['510180.XSHG', '159915.XSHE', '513100.XSHG', '510500.XSHG']
    assert ['option', 'avoid_future_data', True] in acalls and ['option', 'avoid_future_data', True] not in bcalls
    assert a.momentum_day == b.momentum_day == 29 and a.N == 18 and a.M == 600
    assert len(a.slope_series) == 599 and a.slope_series[-1] == 598


def test_accepted_directory_matches_authoritative_receipt_stage():
    if not batch.RECEIPT.is_file(): pytest.skip('Restore ignored accepted batch receipt')
    receipt = batch.read(batch.RECEIPT)
    assert receipt['status'] == 'ok'
    assert batch.ACCEPTED == Path(receipt['checks']['commands'][0]['log']).parent


@pytest.mark.parametrize('changed', ['publication', 'partial_binding'])
def test_resume_refuses_changed_baseline_or_overwriting_partial_start(tmp_path, monkeypatch, changed):
    state = {'batch_id': 'old'}
    write_json(tmp_path/'baseline.json', {'published':state})
    write_json(tmp_path/'progress-start.json', {'sources':695})
    current = {'batch_id':'changed'} if changed == 'publication' else state
    monkeypatch.setattr(batch,'Store',lambda root: batch.SimpleNamespace(published=lambda:current))
    if changed == 'partial_binding': (tmp_path/'input-binding.json').write_bytes(b'existing')
    with pytest.raises(ValueError, match='Startup baseline differs|Cannot resume completed start'):
        batch.resume('diagnostic-root',tmp_path)
    if changed == 'partial_binding': assert (tmp_path/'input-binding.json').read_bytes() == b'existing'


def test_source_or_duplicate_drift_blocks_execution(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][3]['source_sha256'] = '0'*64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='duplicate differs'): batch.kernels(sources)


@pytest.mark.parametrize('growth', [.005, -.005])
def test_original_log_momentum_annualizes250_days_with_r2(sources, growth):
    code, _ = batch.kernels(sources); values = 100*np.exp(np.arange(29)*growth)
    result = batch.momentum_case(code, values)
    assert result[0] == pytest.approx(growth) and result[2] == pytest.approx(1.)
    assert result[-1] == pytest.approx(np.expm1(growth*250))


@pytest.mark.parametrize('values', [[], [1.]*28, [1.]*30, [1.]*28+[np.nan], [1.]*28+[0.]])
def test_invalid_momentum_windows_rejected(sources, values):
    code, _ = batch.kernels(sources)
    with pytest.raises(ValueError, match='Invalid momentum'): batch.momentum_case(code, values)


def test_original_seed_omits_penultimate_complete_ols_window(sources):
    x = np.arange(618.)+100.; frame = pd.DataFrame({'low': x, 'high': x+np.sin(x), 'close': x})
    ns, holder, _, _ = batch.component_namespace(sources, 0, frame)
    ns['g'].slope_series = ns['initial_slope_series']()[:-1]
    assert len(ns['g'].slope_series) == 599
    assert ns['g'].slope_series[-1] == pytest.approx(ns['get_ols'](frame.low.iloc[598:616], frame.high.iloc[598:616])[1])
    with batch.redirect_stdout(batch.io.StringIO()): ns['get_timing_signal']('ignored')
    assert len(ns['g'].slope_series) == 600
    assert ns['g'].slope_series[-1] == pytest.approx(ns['get_ols'](frame.low.iloc[600:618], frame.high.iloc[600:618])[1])
    assert holder['end'] == 617


def test_original_bias_component_matches_independent60_10_30_formula(sources):
    x = 100+np.arange(200.)*.1+np.sin(np.arange(200.)/7.)
    frame = pd.DataFrame({'close': x}); ns, _, scores, _ = batch.component_namespace(sources, 2, frame)
    ns['get_timing_signal']('ignored')
    assert scores[-1] == pytest.approx(batch.reference_bias(x), abs=1e-10)


def test_original_normalized_rank_returns_mixed_target(sources):
    frame = pd.DataFrame({'close': np.arange(20.)+100.}); ns, _, _, _ = batch.component_namespace(sources, 2, frame)
    ns['g'].stock_pool = ['000300.XSHG']; rank = ns['get_rank'](['ignored'])
    assert rank[0] == '000300.XSHG' and rank[1] == pytest.approx(.01)


def test_diagnostics_preserve_wrong_targets_keep_rebalance_thresholds_and_partial_fills(sources):
    diagnostic = batch.diagnostics(sources); assert json.loads(json.dumps(diagnostic)) == diagnostic
    rows = {r['case']: r for r in diagnostic['cases']}
    assert rows['keep_actually_rebalances']['targets'] == [['candidate']]
    assert rows['bias_score_enters_order_loop']['orders'] == [['fund',10000.], [.001,10000.]]
    assert rows['rank_ignores_argument']['result'] == ['actual']
    assert rows['rsrs_threshold_0.7']['signal'] == rows['rsrs_threshold_-0.7']['signal'] == 'KEEP'
    assert rows['rsrs_threshold_0.700001']['signal'] == 'BUY'
    assert rows['rsrs_threshold_-0.700001']['signal'] == 'SELL'
    assert rows['bias_threshold_4.0']['signal'] == rows['bias_threshold_-4.0']['signal'] == 'KEEP'
    assert rows['bias_threshold_4.000001']['signal'] == 'BUY'
    assert rows['bias_threshold_-4.000001']['signal'] == 'SELL'
    assert rows['stop_0_0.2']['orders'] == [['fund',0]] and rows['stop_0_0.20001']['orders'] == []
    assert rows['stop_2_0.9']['orders'] == [] and rows['stop_2_0.89999']['orders'] == [['fund',0]]
    assert rows['partial_fill_open_true_close_false'] == {'case':'partial_fill_open_true_close_false', 'open':True, 'close':False}
    assert rows['constant_momentum_not_repaired']['finite_score'] is False


@pytest.mark.parametrize('values', [[], [1.]*199, [1.]*200, [1.]*199+[np.nan], [1.]*199+[0.]])
def test_invalid_or_degenerate_independent_bias_blocks(values):
    with pytest.raises(ValueError): batch.reference_bias(values)


def test_frozen_input_byte_drift_rejected_before_read(tmp_path):
    write_json(tmp_path / 'input-analysis.json', {'snapshot':batch.SNAPSHOT,'not_a_backtest':True,'platform_equivalent':False,
        'component_instrument':'000300.SH','profiles': {n:{'file':str(tmp_path/n),'sha256':'0'*64}
            for n in ('index-input.parquet','calendar-input.parquet')}})
    (tmp_path/'index-input.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input binding changed'): batch.inputs(tmp_path)


@pytest.mark.parametrize('change', ['query','sha','publication','failed_data','status'])
def test_probe_substitution_rejected(tmp_path, monkeypatch, change):
    api = {'apis':[]}; write_json(tmp_path/'existing-apis.json',api); monkeypatch.setattr(batch,'api_evidence',lambda:api)
    for endpoint,query in batch.QUERIES.items():
        row = {'endpoint':endpoint,'query':query,'api_sha256':file_sha(tmp_path/'existing-apis.json'),
            'status':'failed','published':False,'files':[],'wire':[],'error':'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0'*64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file':'unaccepted','sha256':'0'*64}]
        if change == 'status': row['status'] = 'accepted'
        write_json(tmp_path/f'probe-{endpoint}.json',row)
    with pytest.raises((ValueError,FileNotFoundError)): batch.validate_probes(tmp_path)
