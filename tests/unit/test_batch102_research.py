"""Original annotated definitions, query boundaries, duplicate keys and strict rules."""
import ast
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_strategy_batch102 as study


def test_builtin_annotations_are_retained_and_namespaces_isolated():
    a, b = study.literal(0), study.literal(0)
    assert a['g'] is not b['g'] and a['order_func'].__annotations__ == {'total_cash': float, 'available_cash': float, 'fund_price': float}


@pytest.mark.parametrize('source', ['def x(a: side_effect()): pass', 'def x() -> unknown.Type: pass',
    '@side_effect\ndef x(): pass', 'def x(a=side_effect()): pass'])
def test_annotations_decorators_and_defaults_cannot_execute_unknown_expressions(source):
    assert not study.safe_definition(ast.parse(source).body[0])


def test_actual_diagnostics_JSON_roundtrip_preserves_original_errors():
    doc = study.diagnostics(); assert json.loads(json.dumps(doc, allow_nan=False)) == doc
    rows = {r['case']: r for r in doc['cases']}
    assert rows['macro_DataFrame_current_scalar_error']['error'] == 'TypeError'
    assert rows['ETF_datetime_index_minus1_KeyError']['error'] == 'KeyError'
    assert rows['percentile_single_zero_denominator']['error'] == 'ZeroDivisionError'


def test_original13_month_query_excludes_current_month():
    result = study.macro_query('get_PMI'); query = result['queries'][0]
    assert query == {'fields': ['pmi'], 'filters': [['ge', 'stat_month', '2023-03'], ['lt', 'stat_month', '2024-04']], 'order': ['asc', 'stat_month']}
    assert len(result['value']) == 13


@pytest.mark.parametrize('name, fields', [('calc_provide_and_need', ['stat_month', 'produce_idx', 'new_orders_idx']),
    ('calc_save', ['stat_month', 'raw_material_idx', 'finished_produce_idx'])])
def test_original_subindices_query_uses_previous_month(name, fields):
    query = study.macro_query(name)['queries'][0]
    assert query['fields'] == fields and query['filters'][-1] == ['le', 'stat_month', '2024-03']
    assert query['order'] == ['desc', 'stat_month']


def test_fixed_annual_denominator_and_tie_percentile_are_not_repaired():
    ns = study.literal(0)
    assert ns['year_move_average']({'a': 10., 'b': 20., 'c': 30.}) == 6.
    assert ns['percentile'](np.array([1., 2., 2.]), 2.) == 50.
    with pytest.raises(ZeroDivisionError): ns['percentile'](np.array([1.]), 1.)


@pytest.mark.parametrize('mode, price, expected', [(1, 98., 0), (1, 97.9, 1), (2, 95., 0), (2, 94.9, 1)])
def test_original_Decimal_thresholds_with_explicit_shape_stubs(mode, price, expected):
    result = study.macro_state(mode, price)
    assert result['times'] == expected and result['price_shape_stub']


def test_original_pool_keeps_duplicate512100_and_order_intents():
    ns, _ = study.initialize(1)
    assert tuple(ns['g'].etf) == study.POOL and len(study.POOL) == 18 and len(set(study.POOL)) == 17
    frame = pd.DataFrame({'inc': [.1, .1, .06], 'malong': [100.]*3, 'mashort': [101.]*3}, index=['A', 'A', 'B'])
    result = study.etf_orders(frame, immediate=True)
    assert result['calls'] == [['A', 500.], ['A', 500.]] and result['holding_keys'] == ['A']


def test_metrics_are20_rows19_intervals_with_explicit_protocol_only():
    values = np.linspace(100., 110., 20)
    frame = study.reference_metrics({'X': values})
    assert np.isclose(frame.loc['X', 'inc'], .1)
    assert np.isclose(frame.loc['X', 'malong'], values.mean()) and np.isclose(frame.loc['X', 'mashort'], values[-5:].mean())


def test_deferred_sell_feedback_can_block_buys():
    frame = pd.DataFrame({'inc': [.1, .08], 'malong': [100.]*2, 'mashort': [101.]*2}, index=['A', 'B'])
    delayed = study.etf_orders(frame, ('old1', 'old2'))
    immediate = study.etf_orders(frame, ('old1', 'old2'), True)
    assert delayed['calls'] == [['old1', 0.], ['old2', 0.]]
    assert immediate['calls'] == [['old1', 0.], ['old2', 0.], ['A', 500.], ['B', 500.]]
