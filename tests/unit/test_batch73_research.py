import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch73 as batch


def originals(monkeypatch):
    paths = [Path('repo/量化策略源代码') / name for name in batch.SOURCES]
    if not all(p.is_file() for p in paths): pytest.skip('Read-only source checkout is not installed')
    assert [file_sha(p) for p in paths] == list(batch.SOURCE_SHA)
    trees = [ast.parse(read_source(p)[0]) for p in paths]
    monkeypatch.setattr(batch, 'validate_sources', lambda d: None)
    monkeypatch.setattr(batch, 'source_tree', lambda d, n: trees[n])
    return trees


def test_complete_source_scope_and_lines(monkeypatch):
    trees = originals(monkeypatch)
    assert [len(read_source(Path('repo/量化策略源代码') / n)[0].splitlines()) for n in batch.SOURCES] == [171, 237]
    assert [next(n.lineno for n in t.body if isinstance(n, ast.FunctionDef)) for t in trees] == [17, 29]
    assert sum(isinstance(n, ast.FunctionDef) for n in trees[0].body) == 5
    assert sum(isinstance(n, ast.FunctionDef) for n in trees[1].body) == 7
    assert [n.name for n in trees[1].body if isinstance(n, ast.ClassDef)] == ['QuantTiming']


def test_original_diagnostics_and_order_semantics(monkeypatch):
    originals(monkeypatch); result = batch.diagnostics(Path('unused')); cases = {r['case']: r for r in result['cases']}
    assert len(cases) == len(result['cases']) and result == json.loads(json.dumps(result, allow_nan=False))
    assert result['not_a_backtest'] and result['synthetic_fixture'] and not result['platform_equivalent']
    assert result['real_historical_trade_windows'] == result['new_trading_backtests'] == 0
    case = cases['ebit_negative_EV_negative_EBIT_and_infinite_rank']
    assert case['rank'] == ['C', 'B', 'A', 'E', 'D', 'F'] and case['positive_infinity_first']
    assert case['negative_EV_ratio'] == 2 and case['missing_codes'] == ['D', 'F'] and case['ST_new_listing_retained']
    assert case['query'][0]['factors'] == ['market_cap', 'financial_liability', 'financial_assets', 'EBIT']
    case = cases['ebit_index_concat_duplicate_targets_reset_entry_dates']
    assert case['security'] == ['A', 'B', 'B', 'C'] and case['buy_list'] == ['B', 'B', 'A', 'C']
    assert case['entry_dates'] == {'A': None, 'B': None, 'C': None}
    case = cases['ebit_held_paused_target_and_unheld_paused_skipped']
    assert case['orders'] == [['target', 'B', 800.], ['target_value', 'X', 0], ['target', 'A', 800.], ['target', 'C', 800.]]
    assert case['entry_dates']['Z'] is None and case['entry_dates']['A'] is not None
    assert cases['ebit_duplicate_unfilled_orders_and_entry_registration']['orders'] == [['target', 'A', 1600.]]*2
    assert cases['ebit_empty_list_sells_only_unpaused']['orders'] == [['target_value', 'X', 0]]
    assert cases['ebit_zero_lot_still_registers_entry']['orders'] == [['target', 'A', 0.]]
    assert cases['ebit_zero_lot_still_registers_entry']['entry_date'] is not None
    assert cases['ebit_missing_factor_KeyError']['error'] == 'KeyError'
    assert cases['ebit_zero_open_ZeroDivisionError']['error'] == 'ZeroDivisionError'
    case = cases['PE_initialize_no_schedule_no_security_weeks']
    assert case['scheduled'] == [] and not any(case[n] for n in ('has_security', 'has_weeks', 'has_stdH'))
    assert case['costs'][1][1]['type'] == 'fund' and case['costs'][1][0]['close_tax'] == .001
    assert cases['PE_trade_before_change_AttributeError']['error'] == 'AttributeError'
    schedule = cases['PE_after_change_five_callbacks']['scheduled']
    assert len(schedule) == 5 and schedule[1][:3] == ['weekly', 'market_open', [-2]]
    assert schedule[3][:3] == ['weekly', 'code_run_weekly', [1]]
    assert schedule[4][:3] == ['monthly', 'wages_monthily', [6]]
    case = cases['PE_query_code322002_desc2500_log10_sample_std']
    assert case['query']['filters'] == [['le', 'date', '2023-12-29'], ['eq', 'exchange_code', '322002']]
    assert case['query']['sort'] == [['desc', 'date']] and case['query']['count'] == 2500
    assert case['V'] == [1., 2.] and case['now'] == 1.
    assert case['high'] == pytest.approx(1.5+.05*np.sqrt(.5))
    assert case['low'] == pytest.approx(1.5-.25*np.sqrt(.5))
    assert cases['PE_now_label0_KeyError']['error'] == 'KeyError'
    assert cases['PE_nonpositive_missing_log10_unprotected']['negative_infinity']
    assert cases['PE_nonpositive_missing_log10_unprotected']['nan_values'] == 2
    assert cases['PE_single_row_sample_std_nan']['high_is_nan'] and cases['PE_single_row_sample_std_nan']['low_is_nan']
    for name in ('high_equality', 'low_equality', 'NaN_normal50'):
        assert cases['PE_weeks_'+name]['weeks'] == 50.
        assert cases['PE_target_'+name]['orders'] == [['target_value', '510300.XSHG', 31400.]]
    assert cases['PE_weeks_above_high']['weeks'] == pytest.approx(-5.)
    assert cases['PE_target_above_high']['orders'][0][2] == pytest.approx(24000.)
    assert cases['PE_weeks_below_low']['weeks'] == pytest.approx(25.)
    assert cases['PE_weeks_deep_low_clamp1']['weeks'] == 1.
    assert cases['PE_target_deep_low_clamp1']['orders'] == [['target_value', '510300.XSHG', 100000.]]
    assert cases['PE_target_extreme_high_negative_target']['orders'][0][2] < 0
    assert cases['PE_zero_high_ZeroDivisionError']['error'] == cases['PE_zero_low_ZeroDivisionError']['error'] == 'ZeroDivisionError'
    assert cases['PE_monthly_5000_external_flow']['flows'] == [[5000, {'pindex': 0}]]


@pytest.mark.parametrize('code', [
    '@unsafe\ndef f():\n    pass', 'def f(a=open("secret")):\n    pass',
    'def f(*, a=open("secret")):\n    pass', 'def f(a: unsafe):\n    pass', 'def f() -> unsafe:\n    pass',
    'class f(unsafe):\n    pass', 'class f(metaclass=unsafe):\n    pass',
    '@unsafe\nclass f:\n    def method(self):\n        pass', 'class f:\n    value=unsafe()',
    'class f:\n    @unsafe\n    def method(self):\n        pass',
    'class f:\n    def method(self, a=unsafe()):\n        pass',
])
def test_rejects_eager_selected_definitions(monkeypatch, code):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse(code))
    with pytest.raises(ValueError, match='Unsafe'): batch.selected(Path('unused'), 0, ['f'], {})


def test_missing_and_duplicate_selected_definitions(monkeypatch):
    monkeypatch.setattr(batch, 'source_tree', lambda *a: ast.parse('def f():\n    pass\ndef f():\n    pass'))
    with pytest.raises(ValueError, match='Missing/duplicate'): batch.selected(Path('unused'), 0, ['f'], {})


@pytest.mark.parametrize('case', batch.QUERIES)
def test_request_scope(case):
    url, parameters = batch.expected_request(case)
    assert not any(word in str(parameters).lower() for word in ('token', 'cookie', 'password'))
    if case.startswith('sse-'):
        assert url == 'https://query.sse.com.cn/commonQuery.do'
        assert parameters['SEARCH_DATE'].replace('-', '') == case[4:]
    else:
        assert parameters['txtQueryDate'] == '2026-09-29'


def test_original_threshold_2500_count_and_sample_std(monkeypatch):
    originals(monkeypatch)
    ns = batch.selected(Path('unused'), 1, ['QuantTiming'], {'np': np})
    query_counts = []
    class Saved(ns['QuantTiming']):
        def Q(self, count): query_counts.append(count); return pd.DataFrame({'V': [1., 3., 6.]})
    high, low = Saved('unused', 'unused').std()
    assert query_counts == [2500]
    assert high == pytest.approx(3+.05*np.std([1., 3., 6.], ddof=1))
    assert low == pytest.approx(3-.25*np.std([1., 3., 6.], ddof=1))


def test_no_replacement_ledger_or_trade_claims():
    assert 'scripts/review_strategy_batch73.py' in batch.FILES
    assert all(n not in batch.CORE for n in ('equity.parquet', 'trades.parquet'))
    assert 'market-input-audit.json' in batch.CORE
    assert batch.ACCEPTED.name.endswith('-review-correction')


def test_study_rejects_existing_output_before_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, 'binding', lambda *a: None)
    monkeypatch.setattr(batch, 'diagnostics', lambda *a: pytest.fail('diagnostics must not run'))
    (tmp_path / 'diagnostics.json').touch()
    with pytest.raises(ValueError, match='Study outputs exist'): batch.study('unused', tmp_path)


def test_initialize_retains_fixed_amount_slippage(monkeypatch):
    originals(monkeypatch); values = []; g = SimpleNamespace()
    namespace = batch.selected(Path('unused'), 1, ['initialize'], {'g': g,
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None,
        'log': SimpleNamespace(info=lambda *a: None), 'FixedSlippage': lambda v: v,
        'set_slippage': values.append, 'OrderCost': lambda **kw: kw, 'set_order_cost': lambda *a, **kw: None})
    namespace['initialize'](SimpleNamespace())
    assert values == [.02] and (g.stdpeH, g.stdpeL, g.now_pe) == (1.5, 1.2, 1)
