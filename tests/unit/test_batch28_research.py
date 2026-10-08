import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

from observe.runs import file_sha, write_json
from scripts import review_strategy_batch28 as batch


@pytest.fixture
def sources(tmp_path):
    rows = []
    for name in batch.SOURCES:
        path = Path('repo/量化策略源代码') / name
        if not path.is_file(): pytest.skip('Restore ignored original strategy sources')
        rows.append({'source_copy': str(path), 'source_sha256': file_sha(path)})
    write_json(tmp_path / 'source-reviews/review.json', {'sources': rows}); return tmp_path


def test_original_diagnostics_preserve_faults_and_northbound_order_intents(sources):
    rows = {r['case']: r for r in batch.diagnostics(sources)['cases']}
    assert rows['global_windows_override']['windows'] == [18, 700]
    assert isinstance(rows['third_group_nested']['grouping'][0], list)
    state = rows['first_literal_and_nan_exit']
    assert state['flag'] == [1, 1, 1, 0, 0, 1, 0]
    assert state['buy'] == [5] and state['sell'] == [3]
    assert rows['date_rank_index']['error'] == 'KeyError'
    assert rows['mesa_append']['error'] == 'AttributeError'
    assert rows['negative_buy_quantity']['cash'] == 1050.
    assert rows['multiasset_buy_total']['source_total'] == 900.
    assert rows['multiasset_sell_total']['source_total'] == 900.
    for name in ('northbound_first_baseline', 'northbound_below_filter_disappears', 'northbound_reappears',
                 'northbound_low_rsi_block', 'northbound_binary_delta'):
        assert rows[name]['orders'] == []
    assert len(rows['northbound_stale_hold_cap']['orders']) == 2
    assert rows['northbound_rsi40_inclusive']['orders'] == [['buy', 'a', 20000.]]
    assert rows['northbound_rsi80_inclusive']['orders'] == [['buy', 'a', 20000.]]


@pytest.mark.parametrize('values', [np.arange(100.), -np.arange(100.), np.ones(100),
    np.array([100 + np.sin(k) * 2 + k / 10 for k in range(100)])])
def test_wilder_reference_agrees_with_installed_engine(values):
    assert batch.wilder(values) == pytest.approx(talib.RSI(values)[-1], abs=1e-11)


@pytest.mark.parametrize('values', [[1.], [1.] * 14, [1.] * 15 + [np.nan], [[1., 2.]]])
def test_invalid_rsi_operand_not_filled(values):
    with pytest.raises(ValueError, match='Invalid RSI'): batch.wilder(values)


def test_source_sha_checked_before_compiling(sources):
    doc = batch.read(sources / 'source-reviews/review.json'); doc['sources'][1]['source_sha256'] = '0' * 64
    write_json(sources / 'source-reviews/review.json', doc)
    with pytest.raises(ValueError, match='Selected source changed'):
        batch.methods(sources, 1, 'RSRS', ('_mark_flag',), {'pd': pd})


def test_extract_methods_excludes_top_level_auth_and_class_default(tmp_path):
    p = tmp_path / 'source.txt'
    p.write_text('raise RuntimeError("top level")\nclass Original:\n    context = missing_context()\n    @staticmethod\n    def core(x):\n        return x + 1\n', encoding='utf-8')
    write_json(tmp_path / 'source-reviews/review.json', {'sources': [{'source_copy': str(p), 'source_sha256': file_sha(p)}]})
    cls, _ = batch.methods(tmp_path, 0, 'Original', ('core',), {})
    assert cls.core(2) == 3 and not hasattr(cls, 'context')
    with pytest.raises(ValueError, match='Missing/ambiguous'): batch.methods(tmp_path, 0, 'Original', ('absent',), {})


def test_expression_keeps_actual_rsi14_and_source_sigma(sources):
    code, _ = batch.expr(sources, 3, 'bx_strategy', 'rsi')
    a = np.array([100 + np.sin(k) for k in range(100)])
    assert eval(code, {'numpy': np, 'talib': talib}, {'a': a}) == talib.RSI(a)[-1]
    code, _ = batch.expr(sources, 2, 'Filter', 'spectrum')
    fn = eval(code, {'np': np, 'pi': np.pi, 'alpha1': .05, 'alpha2': -.1, 'sigma': .075})
    assert fn(0.) == pytest.approx(.15 / 1.05 ** 2)


def test_date_rank_lambda_can_only_verify_explicit_last_operand(sources):
    node = next(n for n in ast.walk(batch.tree(sources, 1)) if isinstance(n, ast.FunctionDef) and n.name == '_cal_ret_quantile')
    fn = eval(compile(ast.Expression(next(n for n in ast.walk(node) if isinstance(n, ast.Lambda))), '<rank>', 'eval'))
    assert fn(pd.Series([1., 3., 3.], index=[0, 1, -1])) == 5 / 6
    with pytest.raises(KeyError): fn(pd.Series([1., 3., 3.], index=pd.date_range('2020-01-01', periods=3)))


def test_input_bytes_rejected_before_parsing(tmp_path):
    path = tmp_path / 'price-input.parquet'; path.write_bytes(b'not parquet')
    write_json(tmp_path / 'input-analysis.json', {'snapshot': batch.SNAPSHOT, 'not_a_backtest': True,
        'pool': list(batch.prior.INSTRUMENTS), 'inputs': {'price': {'file': str(path), 'sha256': '0' * 64}, 'index': {}}})
    with pytest.raises(ValueError, match='Component input changed'): batch.inputs(tmp_path)


def test_vendor_preclose_retains_original_first_return_and_nonadjacent_value(sources):
    frame = pd.DataFrame({'date': ['2020-01-01', '2020-01-02'], 'index': ['000300.SH'] * 2,
        'close': [10., 12.], 'preclose': [8., 11.], 'high': [11., 13.], 'low': [9., 10.], 'volume': [100., 200.]})
    result, _ = batch.index_sample(sources, frame)
    assert result.instrument.tolist() == ['000300.SH'] * 2
    assert result.ret.tolist() == [10 / 8 - 1, 12 / 11 - 1]
    assert result.pre_close.tolist() == [8., 11.] and 'preclose' in frame and 'index' in frame
    assert result.ret.iloc[1] != 12 / 10 - 1
    with pytest.raises(ValueError, match='schema missing'): batch.index_sample(sources, frame.drop(columns='preclose'))


@pytest.mark.parametrize('change', ['query', 'sha', 'publication', 'failed_data'])
def test_probe_binding_refuses_misleading_results(tmp_path, monkeypatch, change):
    api = {'apis': []}; write_json(tmp_path / 'existing-apis.json', api)
    monkeypatch.setattr(batch, 'api_evidence', lambda: api)
    for endpoint, query in batch.QUERIES.items():
        row = {'endpoint': endpoint, 'query': query, 'api_sha256': file_sha(tmp_path / 'existing-apis.json'),
            'status': 'failed', 'published': False, 'files': [], 'wire': [], 'error': 'synthetic failure'}
        if change == 'query': row['query'] = {}
        if change == 'sha': row['api_sha256'] = '0' * 64
        if change == 'publication': row['published'] = True
        if change == 'failed_data': row['files'] = [{'file': 'unaccepted'}]
        write_json(tmp_path / f'probe-{endpoint}.json', row)
    with pytest.raises(ValueError): batch.validate_probes(tmp_path)
