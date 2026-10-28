"""Check original reversal windows and fixed-target callback boundaries."""
import ast
import math

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch80 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


@pytest.mark.parametrize('number,time', [(0, '9:30'), (1, '9:35')])
def test_actual_defaults(cases, number, time):
    row = cases[f'{number}_defaults']; costs = row['costs'][0][0]
    assert row['schedules'] == [['trade', time]] and row['slippage'] == [.002]
    assert costs['open_commission'] == costs['close_commission'] == .0002
    assert costs['min_commission'] == 5 and costs['close_tax'] == 0
    assert row['pool'] == (batch.WEIGHTS if number else list(batch.POOL))


@pytest.mark.parametrize('name', ['math', 'mean', 'std'])
def test_missing_injections_are_original_errors(cases, name):
    assert cases['38_missing_'+name]['exception'] == 'NameError'
    assert cases['38_missing_'+name]['missing'] == name


def test_beta_keeps_229_windows_and_two_unused_final_rows(cases):
    row = cases['38_beta_windows']; windows = row['windows']
    assert len(windows) == 229 and windows == [[i, i+19] for i in range(229)]
    assert windows[-1] == [228, 247]
    assert row['value'] == pytest.approx(np.mean(np.arange(229))-2*np.std(np.arange(229)))


@pytest.mark.parametrize('label,buys', [('c_equal_lower', False), ('c_below_lower', False), ('c_above_lower', True),
    ('c_equal_upper', False), ('c_above_upper', False), ('c_below_upper', True)])
def test_spread_is_strict_on_both_sides(cases, label, buys):
    assert cases['38_'+label]['orders'] == ([[batch.POOL[0], 1000.]] if buys else [])


def test_existing_winner_is_not_rebalanced_and_negative_slope_clears(cases):
    assert cases['38_same_holding']['orders'] == []
    assert cases['38_slope_below']['orders'] == [[batch.POOL[0], 0]]


def test_strict_slope_equality_and_negative_rank_winner():
    ns = batch.namespace(batch.DIRECTORY, 0); orders = []
    ns['get_rank'] = lambda pool: pd.DataFrame({'score': [-1., -2., -3., -4.]}, index=batch.POOL)
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'low': [1., 2.], 'high': [1., 2.]})
    ns['np'] = type('Slope', (), {'polyfit': staticmethod(lambda *a: [1., 0.])})
    ns['getBeta'] = lambda *a: 1.
    ns['order_target_value'] = lambda s, v: orders.append([s, v])
    from collections import defaultdict
    from types import SimpleNamespace
    ctx = SimpleNamespace(portfolio=SimpleNamespace(positions=defaultdict(lambda: SimpleNamespace(total_amount=0)), available_cash=1000.))
    ns['trade'](ctx); assert orders == []
    ns['getBeta'] = lambda *a: .9
    ns['trade'](ctx); assert orders == [[batch.POOL[0], 1000.]]


def test_sale_fill_state_changes_original_switch_intent(cases):
    assert cases['38_switch_immediate']['orders'] == [[batch.POOL[1], 0], [batch.POOL[0], 1100.]]
    assert cases['38_switch_delayed']['orders'] == cases['38_switch_partial']['orders'] == [[batch.POOL[1], 0]]


@pytest.mark.parametrize('label,orders', [('deviation_equal', []), ('deviation_above', [['510880.XSHG', 800.]]),
    ('shares_equal', []), ('shares_above', [['510880.XSHG', 800.]]), ('cash_equal', []),
    ('cash_above', [['510880.XSHG', 800.]])])
def test_fixed_weight_strict_boundaries(cases, label, orders):
    assert cases['90_'+label]['orders'] == orders


def test_tied_balances_keep_source_dictionary_order(cases):
    row = cases['90_empty_immediate']
    assert row['orders'] == [[s, 10000*w] for s, w in batch.WEIGHTS.items()]
    assert row['events'][1]['total_before'] != row['initial_total']
    assert row['orders'][1][1] == 800.


def test_immediate_cash_and_delayed_or_failed_sale_are_not_equivalent(cases):
    row = cases['90_sell_cash_immediate']
    assert row['orders'] == [['511010.XSHG', 10000*batch.WEIGHTS['511010.XSHG']],
        *[[s, 10000*batch.WEIGHTS[s]] for s in ('510880.XSHG', '513100.XSHG', '518880.XSHG')]]
    assert cases['90_sell_cash_delayed']['orders'] == [['511010.XSHG', 7000.]]
    assert cases['90_sell_cash_partial']['orders'] == row['orders']
    assert cases['90_sell_cash_partial']['final_synthetic_values'] != row['final_synthetic_values']


@pytest.mark.parametrize('label,exception', [('zero_price', 'ZeroDivisionError'), ('missing_price', 'KeyError')])
def test_invalid_intraday_inputs_are_not_substituted(cases, label, exception):
    row = cases['90_'+label]; assert row['error'] == exception and row['orders'] == []


def test_selector_rejects_eager_defaults():
    assert not batch.safe_function(ast.parse('def f(x=boom()): pass').body[0])


def test_repeat_study_refuses_overwriting(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


def test_independent_score_preserves_reversal_coefficient():
    prices = np.exp(np.arange(200)*.001)
    score = batch.previous.previous.independent(prices[-25:], False)[0]-batch.previous.previous.independent(prices, False)[0]/6
    assert score == pytest.approx((math.exp(.001*250)-1)*5/6, abs=1e-12)


def test_valid_20_window_cache_survives_constant_18_window(monkeypatch):
    index = pd.date_range('2020-01-01', periods=254)
    lows = np.arange(254, dtype=float)+10.; lows[234:252] = lows[234]
    native = {s: pd.DataFrame({'low': lows, 'high': 2*lows+np.sin(np.arange(254))*.01,
        'close': 1.4*lows+np.arange(254)*.001*(n+1)}, index=index)
        for n, s in enumerate(sorted(set(batch.POOL) | set(batch.WEIGHTS)))}
    monkeypatch.setattr(batch, 'frames', lambda directory: (native, []))
    tables, _ = batch.components(batch.DIRECTORY)
    betas = tables[1]
    assert set(betas.loc[betas.history_end == str(index[-1].date()), 'instrument']) == set(batch.POOL)
    assert not (betas.history_end == str(index[251].date())).any()


@pytest.mark.parametrize('key,value', [('rank_differences', 1), ('spread_differences', 1),
    ('beta_decision_differences', 1), ('original_full_rank_calls', 0), ('original_injected_beta_calls', 0)])
def test_finish_rejects_missing_original_calls_and_independent_disagreement(monkeypatch, tmp_path, key, value):
    sha = 'frozen'; code = {'entry.py': sha}
    checked = {'status': 'passed', 'implementation_sha256': code,
        'evidence_sha256': {n: sha for n in batch.CORE},
        'commands': [{'command': cmd, 'returncode': 0, 'log': 'log', 'sha256': sha} for cmd in batch.commands()]}
    summary = {'rank_differences': 0, 'spread_differences': 0, 'beta_decision_differences': 0,
        'original_full_rank_calls': 3, 'ranks': {'rows': 3}, 'original_injected_beta_calls': 1, key: value}
    documents = {'checked-state.json': checked, 'offline-verification.json': {'implementation_sha256': code},
        'component-research.json': summary}
    monkeypatch.setattr(batch, 'binding', lambda *a: None)
    monkeypatch.setattr(batch, 'validate_sources', lambda *a: None)
    monkeypatch.setattr(batch, 'validate_catalog', lambda *a: None)
    monkeypatch.setattr(batch, 'implementation', lambda: code)
    monkeypatch.setattr(batch, 'file_sha', lambda *a: sha)
    monkeypatch.setattr(batch, 'read', lambda p: documents[p.name])
    monkeypatch.setattr(batch, 'protect', lambda *a: pytest.fail('Protection/final publication must not be reached'))
    with pytest.raises(ValueError, match='Independent ranking/beta'): batch.finish('unused', tmp_path)
