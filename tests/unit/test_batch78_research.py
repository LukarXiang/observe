"""Guard the original nonstandard weighting, ranking, orders and frozen inputs."""
import ast
import math
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import review_strategy_batch78 as batch


@pytest.fixture(scope='module')
def cases():
    return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


@pytest.mark.parametrize('number', [0, 1])
def test_original_defaults_pool_cost_and_daily_open(cases, number):
    row = cases[f'{number}_defaults']
    assert row['pool'] == list(batch.POOL) and row['period'] == 25
    assert row['schedules'] == [['trade', '9:30']]
    assert row['costs'] == [[{'open_tax': 0, 'close_tax': 0, 'open_commission': .0002,
        'close_commission': .0002, 'close_today_commission': 0, 'min_commission': 5}, {'type': 'fund'}]]


@pytest.mark.parametrize('number', [0, 1])
def test_negative_momentum_still_selects_a_winner(cases, number):
    assert all(v < 0 for v in cases[f'{number}_negative']['scores'])
    assert cases[f'{number}_negative']['rank'][0] == batch.POOL[0]
    assert cases[f'{number}_positive']['rank'][0] == batch.POOL[-1]
    assert cases[f'{number}_tied']['rank'] == list(batch.POOL)


def test_original_record_series_are_not_replaced_with_scalars(cases):
    assert len(cases['0_record_types']['records']) == 12
    assert all(r == {'type': 'Series', 'length': 1} for r in cases['0_record_types']['records'])
    assert cases['1_record_types']['records'] == []


@pytest.mark.parametrize('number', [0, 1])
def test_implicit_math_missing_and_position_shape_errors(cases, number):
    assert cases[f'{number}_implicit_math_error']['exception'] == 'NameError'
    assert cases[f'{number}_missing_position_key']['exception'] == 'KeyError'


@pytest.mark.parametrize('number', [0, 1])
def test_original_trade_depends_on_instant_cash_and_positions(cases, number):
    assert cases[f'{number}_winner_held']['orders'] == []
    assert cases[f'{number}_delayed_sale']['orders'] == [[batch.POOL[1], 0]]
    assert cases[f'{number}_immediate_sale']['orders'] == [[batch.POOL[1], 0], [batch.POOL[0], 600]]


@pytest.mark.parametrize('number', [0, 1])
def test_original_numeric_errors_and_short_history_not_repaired(cases, number):
    assert cases[f'{number}_overflow']['exception'] == 'OverflowError'
    assert not all(cases[f'{number}_constant']['finite'])
    assert cases[f'{number}_short']['score'] == pytest.approx(math.exp(.001*250)-1)


@pytest.mark.parametrize('weighted', [False, True])
def test_independent_exact_log_line_has_unit_r2(weighted):
    result = batch.independent(np.exp(.005*np.arange(25)+2), weighted)
    assert result[1] == pytest.approx(.005, abs=1e-14)
    assert result[2] == pytest.approx(2, abs=1e-14)
    assert result[3] == pytest.approx(1, abs=1e-14)


def test_weighted_fit_uses_squared_weights_and_r2_keeps_original_mean():
    x = np.arange(25, dtype=float); y = .003*x+.0006*x*x; prices = np.exp(y)
    score, slope, intercept, r2, annualized = batch.independent(prices, True)
    w = np.linspace(1, 2, 25)
    fit = np.linalg.lstsq(np.column_stack((x, np.ones(25)))*w[:, None], y*w, rcond=None)[0]
    assert slope == pytest.approx(fit[0], abs=1e-13)
    assert intercept == pytest.approx(fit[1], abs=1e-13)
    assert r2 == pytest.approx(1-np.sum(w*(y-(slope*x+intercept))**2)/np.sum(w*(y-y.mean())**2))
    single_fit = np.linalg.lstsq(np.column_stack((x, np.ones(25)))*np.sqrt(w)[:, None], y*np.sqrt(w), rcond=None)[0]
    assert abs(slope-single_fit[0]) > .0001
    assert score == annualized*r2


def test_no_constant_price_variance_is_invented():
    assert batch.independent([1.]*25, False) is None
    assert batch.independent([1.]*25, True) is None


@pytest.mark.parametrize('number', [0, 1])
def test_extracted_arithmetic_retains_original_statements(number):
    node = batch.reference_node(batch.DIRECTORY, number)
    assert batch.safe_function(node)
    assert isinstance(node.body[-1], ast.Return)
    assert [n.id for n in node.body[-1].value.elts] == ['score', 'slope', 'intercept', 'r_squared', 'annualized_returns']
    ns = batch.namespace(batch.DIRECTORY, number); prices = np.exp(np.arange(25)*.001+.0001*np.arange(25)**2)
    ns['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': prices})
    actual = batch.reference_function(batch.DIRECTORY, number, ns)(batch.POOL[0])
    assert actual == pytest.approx(batch.independent(prices, bool(number)), rel=1e-10, abs=1e-12)


def test_fixed_request_rejects_repeats_redirect_targets_and_posts():
    assert batch.expected_request('GET', batch.ENDPOINT, {}, 0)['parameters'] is None
    for method, url, number in [('POST', batch.ENDPOINT, 0), ('GET', batch.ENDPOINT, 1), ('GET', 'https://example.com', 0)]:
        with pytest.raises(ValueError): batch.expected_request(method, url, {}, number)


def test_selector_rejects_eager_source_code():
    assert not batch.safe_function(ast.parse('def f(x=boom()): pass').body[0])


def test_repeat_study_refused_before_writes(tmp_path):
    path = tmp_path/'diagnostics.json'; path.write_bytes(b'original')
    with pytest.raises(ValueError): batch.study('unused', tmp_path)
    assert path.read_bytes() == b'original' and list(tmp_path.iterdir()) == [path]


def test_missing_fourth_fund_keeps_all_pool_ranking_blocked(tmp_path):
    folder = tmp_path/'source-reviews'; folder.mkdir()
    for name in batch.SOURCES:
        shutil.copyfile(batch.DIRECTORY/'source-reviews'/(batch.strategy_id(name)+'.source'), folder/(batch.strategy_id(name)+'.source'))
    batch.save(tmp_path/'reused-inputs.json', {'inputs': {f'fund-{s[:6]}': {'status': 'failed'} for s in (batch.POOL[0], batch.POOL[2], batch.POOL[3])}})
    batch.save(tmp_path/'probe-results.json', {'results': [{'status': 'failed'}]})
    frame, ranks, summary, _ = batch.components(tmp_path)
    assert frame.empty and ranks.empty and summary['common_rank_dates'] == 0
    assert summary['new_trading_backtests'] == summary['original_input_rows_admitted'] == 0
    assert len(summary['blocked']) == 4


@pytest.mark.parametrize('bumps,leader_differences', [((1, 2, 3, 4), 2), ((4, 1, 3, 2), 0)])
def test_near_tie_independent_full_rank_differences_are_counted(tmp_path, monkeypatch, bumps, leader_differences):
    folder = tmp_path/'source-reviews'; folder.mkdir(); operands = {}
    for name in batch.SOURCES:
        shutil.copyfile(batch.DIRECTORY/'source-reviews'/(batch.strategy_id(name)+'.source'), folder/(batch.strategy_id(name)+'.source'))
    for s in batch.POOL:
        path = tmp_path/(s[:6]+'.parquet')
        pd.DataFrame({'date': pd.date_range('2020-01-01', periods=25, freq='B'), 'close': np.exp(np.arange(25)*.001)}).to_parquet(path, index=False)
        operands['fund-'+s[:6]] = {'status': 'success', 'file': str(path), 'sha256': batch.file_sha(path)}
    last = operands.pop(batch.CASE)
    batch.save(tmp_path/'reused-inputs.json', {'inputs': operands})
    batch.save(tmp_path/'probe-results.json', {'results': [last]})
    original = batch.independent; calls = []
    def perturb(prices, weighted):
        result = list(original(prices, weighted)); number = len(calls); calls.append(True)
        if number >= 8: result[0] += bumps[number % 4]*1e-12
        return tuple(result)
    monkeypatch.setattr(batch, 'independent', perturb)
    _, ranks, summary, _ = batch.components(tmp_path)
    assert summary['independent_leader_differences'] == leader_differences
    assert summary['independent_rank_differences'] == 2
    assert (ranks['rank'] != ranks.independent_rank).all()


@pytest.mark.parametrize('leader_differences', [0, 1])
def test_finish_rejects_nonzero_independent_ranking_before_acceptance(tmp_path, monkeypatch, leader_differences):
    for name in ('binding', 'validate_sources', 'validate_catalog'):
        monkeypatch.setattr(batch, name, lambda *a: None)
    monkeypatch.setattr(batch, 'implementation', lambda: {})
    monkeypatch.setattr(batch, 'commands', lambda: [])
    monkeypatch.setattr(batch, 'CORE', set())
    monkeypatch.setattr(batch, 'Store', lambda _: type('Stub', (), {'published': lambda self: {}})())
    monkeypatch.setattr(batch, 'protect', lambda *a: pytest.fail('Must reject rank mismatch before protection/acceptance'))
    batch.save(tmp_path/'checked-state.json', {'status': 'passed', 'commands': [], 'implementation_sha256': {}, 'evidence_sha256': {}})
    batch.save(tmp_path/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_sha256': {}})
    batch.save(tmp_path/'baseline.json', {'published': {}})
    batch.save(tmp_path/'component-research.json', {'independent_leader_differences': leader_differences, 'independent_rank_differences': 1})
    with pytest.raises(ValueError, match='Independent ranking'):
        batch.finish('unused', tmp_path)
