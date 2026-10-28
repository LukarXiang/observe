"""Check tick dependency detection and original stock/bond callback semantics."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from observe.strategy_catalog import catalog_strategies
from scripts import review_strategy_batch81 as batch


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_current_tick_registers_intraday_dependency_without_schedule(tmp_path, prefix, legacy):
    source = tmp_path/'source'; source.mkdir()
    code = f'def trade(context):\n    {prefix}get_current_tick("000001.XSHG")\n    order_value("600000.XSHG", 1000)\n'
    if legacy: code += '    print context.current_dt\n'
    (source/'tick.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path/'data', source)
    row = json.loads((Path(result['output'])/'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_current_tick' in row['apis']
    assert '盘中事件/撮合与对应分钟或Tick股票池' in row['gaps']
    assert row['status'] == '暂不可复现'
    assert bool(row['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['get_current_tick_extra', 'my_get_current_tick'])
def test_similar_tick_function_names_do_not_create_gap(tmp_path, name):
    source = tmp_path/'source'; source.mkdir()
    (source/'other.py').write_text(f'def trade(context):\n    {name}(q)\n    order_value("600000.XSHG", 1000)\n', encoding='utf-8')
    result = catalog_strategies(tmp_path/'data', source)
    row = json.loads((Path(result['output'])/'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_current_tick' not in row['apis']
    assert '盘中事件/撮合与对应分钟或Tick股票池' not in row['gaps']


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


@pytest.mark.parametrize('number,names,times', [(0, ['market_open', 'market_1455'], ['9:31', '14:55']),
    (1, ['before_market_open', 'market_open', 'after_market_close'], ['before_open', '9:30', 'after_close'])])
def test_actual_stock_only_costs_and_schedules(cases, number, names, times):
    row = cases[f'{number}_defaults']
    assert [r[0] for r in row['schedules']] == names
    assert [r[1]['time'] for r in row['schedules']] == times
    assert row['costs'] == [[{'close_tax': .001, 'open_commission': .0003,
        'close_commission': .0003, 'min_commission': 5}, {'type': 'stock'}]]
    assert row['fund_cost_explicit'] is False and row['slippage_explicit'] is False


def test_initial_button_is_not_a_real_debt_holding(cases):
    row = cases['57_initial_no_signal']
    assert row['orders'] == [] and row['final_values'] == {}
    assert row['states'] == [{'day': 1, 'button': 0, 'cash': 100.}]


@pytest.mark.parametrize('label', ['series_label_error', 'missing_close'])
def test_original_history_compatibility_errors_are_retained(cases, label):
    row = cases['57_'+label]
    assert row['error'] in ('KeyError', 'IndexError') and row['orders'] == [] and row['states'] == []


@pytest.mark.parametrize('label,entered', [('y_equal', True), ('y_above', False), ('t_equal', True), ('t_above', False)])
def test_entry_uses_inclusive_two_decline_thresholds(cases, label, entered):
    row = cases['57_'+label]
    assert bool(row['orders']) is entered
    assert row['states'][-1]['button'] == int(entered)


def test_repeated_trigger_does_not_reset_held_counter(cases):
    row = cases['57_repeat_held']; assert row['orders'] == []
    assert row['states'] == [{'day': 8, 'button': 1, 'cash': 100.}]


def test_entry_increments_same_callback_and_19_callbacks_can_exit(cases):
    row = cases['57_entry_19_callbacks']
    assert [r['day'] for r in row['states']] == [*range(2, 20), 1]
    assert row['states'][-1]['button'] == 0
    assert row['orders'][-2:] == [['512100.XSHG', 0], ['511010.XSHG', 100.]]
    same = cases['57_entry_and_exit_same_callback']
    assert same['orders'] == [['511010.XSHG', 0], ['512100.XSHG', 1100.],
        ['512100.XSHG', 0], ['511010.XSHG', 1100.]]
    assert same['states'] == [{'day': 1, 'button': 0, 'cash': 0.}]


def test_exit_does_not_require_another_trigger(cases):
    assert cases['57_exit_without_trigger']['orders'] == [['512100.XSHG', 0], ['511010.XSHG', 1100.]]


@pytest.mark.parametrize('mode,amount', [('immediate', 1100.), ('delayed', 100.), ('partial', 600.), ('failed', 100.)])
def test_sale_cash_changes_entry_amount_but_failed_buy_still_updates_counter(cases, mode, amount):
    row = cases['57_switch_'+mode]
    assert row['orders'] == [['511010.XSHG', 0], ['512100.XSHG', amount]]
    assert row['states'][-1]['day'] == 2 and row['states'][-1]['button'] == 1
    if mode in ('delayed', 'failed'): assert '512100.XSHG' not in row['final_values']


@pytest.mark.parametrize('initial,label,expected', [
    (0, 'below95', 2), (0, 'equal95', 1), (0, 'equal975', 0), (0, 'above99', 0),
    (1, 'below95', 2), (1, 'equal95', 1), (1, 'equal99', 1), (1, 'above99', 0),
    (2, 'equal975', 2), (2, 'above975', 1), (2, 'equal99', 1), (2, 'above99', 0)])
def test_drawdown_state_strict_boundaries(cases, initial, label, expected):
    row = cases[f'76_state{initial}_{label}']; assert row['status'] == expected
    assert row['stock_cash'] == pytest.approx(row['total']*(.3, .4, .5)[expected])


def test_bad_sign_overrides_larger_stock_allocation(cases):
    row = cases['76_declining_means']
    assert row['status'] == 2 and row['bad_sign'] == 1
    assert row['stock_cash'] == 0 and row['debt_cash'] == row['total']
    assert row['orders'] == [['511010.XSHG', row['total']], *[[s, 0.] for s in batch.STOCKS]]


@pytest.mark.parametrize('label,expected', [('peak9_equal', 0), ('peak9_below', 1), ('peak19_equal', 0), ('peak19_below', 1)])
def test_peak_price_thresholds_are_strict(cases, label, expected):
    assert cases['76_'+label]['bad_sign'] == expected


def test_original_slices_exclude_boundary_rows_and_latest_price(cases):
    row = cases['76_exact_slices']
    assert row['windows'] == [{'kind': 'mean', 'indices': list(range(40, 59))},
        {'kind': 'mean', 'indices': list(range(20, 39))}, {'kind': 'mean', 'indices': list(range(19))},
        {'kind': 'max', 'indices': list(range(50, 59))}, {'kind': 'max', 'indices': list(range(40, 59))}]


def test_debt_first_submission_can_fail_before_stock_sales(cases):
    row = cases['76_stock_to_debt_immediate']
    assert row['orders'][0] == ['511010.XSHG', 10000.]
    assert row['events'][0]['accepted'] is False
    assert row['final_cash'] == 10000. and row['final_values'] == {}
    assert cases['76_stock_to_debt_partial']['final_cash'] == 5000.
    assert cases['76_stock_to_debt_delayed']['final_cash'] == 0.


def test_only_after_close_updates_highwater(cases):
    assert cases['76_highwater_after_close']['states'] == [
        {'close_equity': 10000., 'max_before': 10000., 'max_after': 10000.},
        {'close_equity': 11000., 'max_before': 10000., 'max_after': 11000.},
        {'close_equity': 10900., 'max_before': 11000., 'max_after': 11000.}]


@pytest.mark.parametrize('values', [[1.]*59, [1.]*59+[math.nan], [1.]*59+[0.]])
def test_missing_or_invalid_risk_operand_is_blocked(values):
    with pytest.raises(ValueError): batch.risk_arithmetic(values)


def test_price_risk_reference_executes_original_function(monkeypatch):
    import pandas as pd
    dates = pd.bdate_range('2020-01-01', periods=61).strftime('%Y-%m-%d')
    frame = pd.DataFrame({'date': dates, 'close': np.linspace(120., 100., 61)})
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'index_frames', lambda *a: (frame, frame))
    monkeypatch.setattr(batch, 'fund_frames', lambda *a: (pd.DataFrame({'history_end': dates}), []))
    tables, summary = batch.components(batch.DIRECTORY)
    assert summary['original_cash_management_calls'] == 2 and summary['original_risk_differences'] == 0
    assert tables[0].bad_sign.tolist() == [1, 1]
    assert tables[0].next_decision_date.iloc[0] == dates[-1]
    assert pd.isna(tables[0].next_decision_date.iloc[1])
    assert not tables[1].complete_entry_signal_available.any()
    assert summary['complete57_entry_signals'] == summary['real76_equity_trajectories'] == summary['new_trading_backtests'] == 0


def test_repeat_study_refuses_overwriting(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('key,value', [('original_risk_differences', 1), ('yesterday_condition_differences', 1),
    ('original_cash_management_calls', 0), ('complete57_entry_signals', 1), ('real76_equity_trajectories', 1)])
def test_finish_rejects_disagreement_or_fabricated_trade_scope(monkeypatch, tmp_path, key, value):
    sha = 'frozen'; code = {'entry.py': sha}
    checked = {'status': 'passed', 'implementation_sha256': code, 'evidence_sha256': {n: sha for n in batch.CORE},
        'commands': [{'command': cmd, 'returncode': 0, 'log': 'log', 'sha256': sha} for cmd in batch.commands()]}
    summary = {'original_risk_differences': 0, 'yesterday_condition_differences': 0, 'original_cash_management_calls': 3,
        'risk_components': {'rows': 3}, 'complete57_entry_signals': 0, 'real76_equity_trajectories': 0, 'new_trading_backtests': 0, key: value}
    documents = {'checked-state.json': checked, 'offline-verification.json': {'implementation_sha256': code}, 'component-research.json': summary}
    monkeypatch.setattr(batch, 'binding', lambda *a: None); monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'validate_catalog', lambda *a: None); monkeypatch.setattr(batch, 'implementation', lambda: code)
    monkeypatch.setattr(batch, 'file_sha', lambda *a: sha); monkeypatch.setattr(batch, 'read', lambda p: documents[p.name])
    monkeypatch.setattr(batch, 'protect', lambda *a: pytest.fail('Final publication must not be reached'))
    with pytest.raises(ValueError, match='Independent original risk'): batch.finish('unused', tmp_path)
