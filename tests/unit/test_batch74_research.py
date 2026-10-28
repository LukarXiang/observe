"""Keep original cash frequency, identities and increment/target distinction."""
import ast
from pathlib import Path

import pytest

from scripts import review_strategy_batch74 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_original_callback_deposit_pool_and_reference(cases):
    row = cases['each_callback_deposits400']
    assert row['calls'] == [[400, {'pindex': 0}], [400, {'pindex': 0}]]
    assert row['pool'] == list(batch.FUNDS)
    assert row['reference'] == '000002.XSHG'
    assert row['weights']['518880.XSHG'] == 4
    assert row['bond'] == '163210.XSHE'


def test_date_liquidity_and_trade_boundaries(cases):
    assert cases['listing_strict300d']['retained'] == {'older': 10}
    assert cases['last_trade_29d_weekday4']['eligible'] is False
    assert cases['last_trade_30d_weekday4']['eligible'] is True
    assert cases['last_trade_31d_weekday5']['eligible'] is False
    assert cases['absolute_stoploss_2.0']['value'] is False
    assert cases['absolute_stoploss_3.0']['value'] is True
    assert cases['absolute_stoploss_4.0']['value'] is False
    assert cases['absolute_stoploss_-3.0']['value'] is True


def test_original_errors_are_preserved(cases):
    assert cases['implicit_datetime_missing']['exception'] == 'NameError'
    assert cases['pandas3_fillna_method_incompatible']['exception'] == 'TypeError'
    assert cases['pandas_integer_labels_fail']['exception'] == 'KeyError'
    assert cases['undefined_ratio_returns50']['value'] == 50


def test_es_fixed_denominator_and_unused_var(cases):
    assert cases['short_fixed_denominator']['ES'] == pytest.approx(.01)
    assert cases['short_fixed_denominator']['VaR'] == 1
    assert cases['constant_positive_ES']['ES'] == pytest.approx(-.01)
    assert cases['zero_ES']['ES'] == 0


def test_activated_indices_and_signal_strict_boundaries(cases):
    assert cases['active_indices_and_1500_schedule']['codes'] == [
        '000016.XSHG', '000016.XSHG', '399006.XSHE', '399006.XSHE', '000012.XSHG', '000012.XSHG']
    assert cases['active_indices_and_1500_schedule']['schedule'] == [[1, '15:00']]
    for name, signal in [('equal_positive', 3), ('winner1', 1), ('winner2', 2), ('both_negative', 3)]:
        assert cases['signal_'+name]['value'] == signal


def test_incremental_buy_is_not_target_weight(cases):
    assert cases['orders_signal1_incremental']['orders'] == [
        ['target', '399006.XSHE', 0], ['target', '000012.XSHG', pytest.approx(2000)],
        ['increment', '000016.XSHG', 8000], ['increment', '000012.XSHG', 3000]]
    assert cases['orders_signal3_incremental']['orders'] == [
        ['target', '000016.XSHG', 0], ['target', '399006.XSHE', 0], ['increment', '000012.XSHG', 3000]]


def test_probe_scope_reuses_old_etf_and_exact_markets():
    assert len(batch.QUERIES) == 20
    assert 'fund-510310' not in batch.QUERIES
    assert batch.expected_request('index-sh000002')[1]['secid'] == '1.000002'
    assert batch.expected_request('fund-161907')[1]['secid'] == '0.161907'
    assert batch.expected_request('fund-501021')[1]['secid'] == '1.501021'
    assert all(q['parameters']['end_date'] == '20260929' for q in batch.QUERIES.values())


@pytest.mark.parametrize('definition', ['@danger()\ndef f(): pass', 'def f(x=boom()): pass', 'def f(x: boom()): pass'])
def test_original_selector_rejects_immediate_execution(definition):
    assert not batch.previous.safe_function(ast.parse(definition).body[0])


def test_frozen_source_hashes_and_full_definition_counts():
    batch.validate_sources(batch.DIRECTORY)
    rows = batch.read(batch.DIRECTORY / 'source-reviews/review.json')['sources']
    assert [r['complete_lines_read'] for r in rows] == [648, 125]
    assert [len([n for n in batch.source_tree(batch.DIRECTORY, i).body if isinstance(n, ast.FunctionDef)]) for i in (0, 1)] == [17, 7]
    assert Path(rows[0]['source_copy']).is_file()


def test_rank_caps_are_applied_before_adjustment_and_states_persist():
    cases = {r['case']: r for r in batch.expanded_diagnostics(batch.DIRECTORY)['cases']}
    assert cases['ranking_adjustment_after_caps']['ratios']['A'] > .3
    assert 0 < cases['ranking_adjustment_after_caps']['ratios']['D'] < .1
    assert cases['state_smoothing80_20']['smoothed_A'] == pytest.approx(.8*300+.2*600)
    assert cases['growth_lock_strict_unlock']['at30'] == cases['growth_lock_strict_unlock']['at5'] == 1
    assert cases['growth_lock_strict_unlock']['below5'] == 0
    assert cases['growth_lock_strict_unlock']['locked_ratio_A'] == 0
    assert cases['growth_lock_strict_unlock']['unlocked_ratio_A'] > 0
    assert cases['marginal_old_holding_preserved']['ratios']['E'] == 0
    assert cases['rounded_delta_1499']['orders'] == []
    assert cases['rounded_delta_1500']['orders'] == [['A', 1500]]
    assert cases['numpy_single_array_float_incompatible']['exception'] == 'TypeError'


def test_fallback_preserves_exact_source_symbols_and_native_decoder():
    from scripts import probe_strategy_batch74_fallback as fallback
    assert set(fallback.QUERIES) == set(batch.QUERIES)
    assert fallback.QUERIES['index-sh000002']['parameters'] == {'symbol': 'sh000002'}
    assert fallback.QUERIES['fund-161907']['parameters'] == {'symbol': 'sz161907'}
    assert fallback.expected_request('index-sh000002')[1] == {'d': '2020_2_4'}
    assert fallback.expected_request('fund-161907')[1] is None


def test_index_component_excludes_future_rows_and_keeps_n11(tmp_path, monkeypatch):
    import pandas as pd
    original = batch.selected
    monkeypatch.setattr(batch, 'selected', lambda *a, **kw: original(batch.DIRECTORY, 1, ['yfdebld_signal'],
        {'log': batch.SimpleNamespace(info=lambda *a: None)}))
    folder = tmp_path / 'sina-fallback'; folder.mkdir(); results = []
    for case, values in [('index-sh000016', list(range(100, 114))), ('index-sz399006', [100.]*14)]:
        path = folder / (case+'.parquet')
        pd.DataFrame({'date': pd.date_range('2026-09-19', periods=14), 'close': values}).to_parquet(path)
        results.append({'case': case, 'status': 'success', 'file': str(path), 'sha256': batch.file_sha(path)})
    batch.save(folder / 'results.json', {'results': results})
    frame, summary = batch.numeric_indices(tmp_path)
    assert len(frame) == 1
    assert frame.date.iloc[0].isoformat() == '2026-09-29'
    assert frame.signal.tolist() == frame.independent_signal.tolist() == [1]
    assert [r['after_cutoff'] for r in summary['inputs']] == [3, 3]


def test_repeat_index_study_cannot_overwrite_payload(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(batch, 'binding', lambda *a: None)
    monkeypatch.setattr(batch, 'numeric_indices', lambda *a: (pd.DataFrame({'signal': [1]}), {'status': 'ok'}))
    payload = tmp_path / 'index-reference-components.parquet'; payload.write_bytes(b'immutable original payload')
    summary = tmp_path / 'index-component-research.json'; summary.write_text('{}')
    with pytest.raises(ValueError):
        batch.index_study('unused', tmp_path)
    assert payload.read_bytes() == b'immutable original payload'


@pytest.mark.parametrize('tamper', [None, 'data_field', 'ast_proof'])
def test_final_inventory_only_admits_proven_format_sha_change(tmp_path, tamper):
    import shutil
    source = batch.DIRECTORY
    frozen = batch.read(source / 'dependency-inventory.json')
    batch.save(tmp_path / 'dependency-inventory.json', frozen)
    shutil.copyfile(source / 'book-tested-before-whitespace.py', tmp_path / 'book-tested-before-whitespace.py')
    before = batch.implementation(); scope = batch.read(source / 'ledger-scope-revalidation-final.json')
    scope['implementation_sha256'] = before
    if tamper == 'ast_proof': scope['book_ast_sha256'] = '0'*64
    batch.save(tmp_path / 'ledger-scope-revalidation-final-amended.json', scope)
    actual = dict(frozen); actual['ledger_sha256'] = before['src/observe/ledger/book.py']
    if tamper == 'data_field': actual['original_trades_admitted'] = 1
    if tamper is None:
        assert batch.validated_final_inventory(tmp_path, actual, before) == actual
    else:
        with pytest.raises(ValueError): batch.validated_final_inventory(tmp_path, actual, before)
    assert batch.read(tmp_path / 'dependency-inventory.json') == frozen
