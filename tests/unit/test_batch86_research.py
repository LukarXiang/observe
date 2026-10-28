"""Check literal thresholds, Python compatibility and original order-state boundaries."""
import ast

import pytest

from scripts import review_strategy_batch86 as batch


@pytest.fixture(scope='module')
def cases(): return {r['case']: r for r in batch.diagnostics(batch.DIRECTORY)['cases']}


def test_parent_frozen_evidence_cannot_change(monkeypatch):
    original = batch.file_sha; target = batch.previous.DIRECTORY/'diagnostics.json'
    monkeypatch.setattr(batch, 'file_sha', lambda p: '0'*64 if p == target else original(p))
    with pytest.raises(ValueError, match='Accepted85 evidence changed'): batch.binding('data')


def test_print_adaptation_retains_missing_names_and_source_lines():
    code = batch.adapted_source('\n'*8+'# code\ndef f():\n print mean([1])\n', 0)
    tree = ast.parse(code)
    assert tree.body[0].lineno == 10
    assert 'print(mean([1]))' in code and 'import' not in code


@pytest.mark.parametrize('label,expected', [('exact30', True), ('only29', False), ('missing_anchor', False),
    ('red_close', False), ('doji', True), ('volume_below', False), ('volume_above', True),
    ('amplitude_equal', True), ('amplitude_above', False), ('strict_anchor_equal', False), ('strict_anchor_above', True)])
def test_literal_buy_boundaries_and_partial_history(cases, label, expected):
    row = cases['54_buy_'+label]
    assert row['original'] is expected and row['independent'] is expected
    assert row['history_rows'] < 270


@pytest.mark.parametrize('label,error', [('missing_mean', 'NameError'), ('empty_history', 'IndexError')])
def test_original_missing_and_empty_history(cases, label, error): assert cases['54_'+label]['error'] == error


def test_source54_missing_timedelta_and_calendar_day(cases):
    assert 'timedelta' in cases['54_missing_timedelta']['error']
    assert cases['54_turnover_previous_sunday']['trace'][0][1] == '2020-01-05'


def test_source54_actual_sell_short_circuit(cases):
    assert cases['54_sell_branch_0.1']['trace'] == [.1, 'turnover', .2]
    assert cases['54_sell_branch_0.1']['result'] is False
    assert cases['54_sell_branch_0.2']['trace'] == [.1, .2]
    assert cases['54_sell_branch_0.2']['result'] is True


def test_source54_cash_and_exact_slot_gate(cases):
    assert cases['54_cash_4999.0_positions_0']['orders'] == []
    assert cases['54_cash_5000.0_positions_0']['orders'] == [['s', 0]]
    assert cases['54_cash_10000.0_positions_10']['orders'] == []
    assert cases['54_cash_10000.0_positions_11']['orders'] == [['s', -1000]]


def test_source54_add_comment_actually_reduces_target(cases):
    row = cases['54_add_comment_targets_lower_quantity']
    assert row['orders'] == [['s', 100]] and row['held_quantity'] == 2000


def test_no_message_sent(cases): assert cases['54_message_bytes_type']['messages_sent'] == 0


def test_original95_technical_and_flow_arguments(cases):
    assert cases['95_technical_arguments']['trace'] == [['MACD_judge_jincha', 'a', 12, 26, 26],
        ['BBands_lowerband_judge_dayu', 'a', 6, 5, 2, 2], ['MA_judge_duotou', 'a', 5, 10]]
    trace = cases['95_flow_arguments_and_order']['trace']
    assert trace == [[['a', 'net_pct_main', 15], {'which_day': 'previous'}],
        [['a', 'net_pct_main', 10], {'which_day': 'previous'}], ['sell', ['a']], ['buy', ['a']]]


def test_zero_weight_nonempty_sort_is_not_ignored(cases):
    row = cases['95_nonempty_zero_weight_sort']
    assert len(row['calls']) == 3 and all(v[1] == 0 for v in row['weights'].values())
    assert 'sort' in row['error']


def test_retry_deduplicates_positions_but_does_not_clean_list(cases):
    assert cases['95_retry_no_cleanup'] == {'case': '95_retry_no_cleanup', 'orders': [['held', 0]], 'retry': ['held', 'held', 'absent']}
    assert cases['95_holded_false']['result'] == ['new', 'new']


def test_independent_search_uses_closest_anchor():
    h = batch.fixture_history(40, volume=80.); h['volume'][10] = 201.
    assert batch.buy_reference(h) is False
    h['volume'][10] = 200.
    assert bool(batch.buy_reference(h)) is True


def test_no_long_flow_history_or_new_requests_claimed():
    summary = batch.supplements()
    assert summary['new_requests'] == summary['new_rows'] == 0 and summary['published'] is False
    assert len(summary['results']) == 2


def test_authoritative_frozen_shape_and_duplicate_key_gate():
    prices = batch.pd.read_parquet(batch.OPERAND)
    assert len(prices) == 46781 and int(prices.is_trading.sum()) == 46063
    batch.validate_price_shape(prices)
    with pytest.raises(ValueError, match='shape'): batch.validate_price_shape(prices.iloc[:-1])
    prices.iloc[-1] = prices.iloc[0]
    with pytest.raises(ValueError, match='shape'): batch.validate_price_shape(prices)
