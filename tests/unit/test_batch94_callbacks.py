"""Concrete original callback behavior under explicit synthetic feedback models."""
import pytest
import sys

from scripts import diagnose_strategy_batch94 as batch


@pytest.mark.parametrize('case,result', [
    ('open_rejected', False), ('open_zero_fill', False), ('open_partial_canceled', True), ('open_full', True),
    ('close_rejected', False), ('close_zero_fill', False), ('close_partial_canceled', False),
    ('close_full_held', True), ('close_full_canceled', False),
])
def test_original_fill_status_result(case, result):
    row = batch.run_case(1, case)
    assert row['result'] is result
    assert row['error'] is None
    assert len(row['orders']) == 1


def test_plain_empty_positions_prevent_first_buy():
    row = batch.run_case(0, 'empty_plain_dict')
    assert row['error']['type'] == 'KeyError'
    assert row['orders'] == []


@pytest.mark.parametrize('case', ['rejected_buy_numeric_score', 'zero_buy_numeric_score'])
def test_rank_score_reaches_order_api_after_failed_first_buy(case):
    row = batch.run_case(0, case)
    assert row['error'] is None
    assert row['orders'] == [
        {'security': {'kind': 'security', 'value': batch.A}, 'value': 1000.},
        {'security': {'kind': 'number', 'value': .125}, 'value': 1000.}]


def test_array_rank_score_is_unhashable_after_rejection():
    row = batch.run_case(1, 'rejected_buy_array_score')
    assert row['error']['type'] == 'TypeError'
    assert len(row['orders']) == 1


@pytest.mark.parametrize('case,amount', [('full_buy_masks_score', 100), ('partial_buy_masks_score', 30)])
def test_successful_first_buy_break_masks_second_rank_item(case, amount):
    row = batch.run_case(1, case)
    assert row['error'] is None and len(row['orders']) == 1
    assert row['positions'] == [{'security': {'kind': 'security', 'value': batch.A}, 'amount': amount}]


def test_immediate_sell_changes_live_iteration():
    row = batch.run_case(1, 'sell_mutates_iteration')
    assert row['error']['type'] == 'RuntimeError'
    assert len(row['orders']) == 1
    assert row['positions'][0]['security']['value'] == batch.A
    assert row['orders'][0]['security']['value'] == batch.B


@pytest.mark.parametrize('case,amount', [('partial_sell_blocks_buy', 70), ('deferred_sell_blocks_buy', 100)])
def test_residual_position_count_blocks_purchase(case, amount):
    row = batch.run_case(1, case)
    assert row['error'] is None and len(row['orders']) == 1
    assert row['orders'][0]['value'] == 0
    assert row['positions'][0]['amount'] == amount


def test_KEEP_calls_rank_timing_then_rotates():
    row = batch.run_case(0, 'KEEP_rotates')
    assert row['steps'] == ['rank', 'timing']
    assert row['orders'] == [{'security': {'kind': 'security', 'value': batch.B}, 'value': 0.}]


def test_rank_error_blocks_even_SELL_liquidation():
    row = batch.run_case(1, 'rank_error_blocks_SELL')
    assert row['steps'] == ['rank'] and row['error']['type'] == 'KeyError'
    assert row['orders'] == []
    assert row['positions'][0]['security']['value'] == batch.B


@pytest.mark.parametrize('case', ['not_0930_0929', 'not_0930_0931', 'not_0930_1500'])
def test_original_exact_time_guard(case):
    row = batch.run_case(1, case)
    assert row['steps'] == row['orders'] == []


def test_two_open_callback_orders_have_different_final_holdings():
    first = batch.run_case(1, 'trade_then_stop'); second = batch.run_case(1, 'stop_then_trade')
    assert first['error'] is second['error'] is None
    assert first['positions'] == [] and len(first['orders']) == 1
    assert second['positions'][0]['security']['value'] == batch.A and len(second['orders']) == 2
    assert second['orders'][1]['value'] == 2000.


def test_initialize_fees_and_schedule_parameters_preserved():
    row = batch.run_case(1, 'initialize_with_seed_spy'); result = row['result']
    assert result['seed_count'] == 599 and result['seed_last'] == 598
    assert result['N'] == 18 and result['M'] == 600 and result['threshold'] == .7
    assert result['stock_num'] == 1 and result['stock_pool'] == list(batch.stage.POOL)
    events = result['events']
    assert ['benchmark', '399006.XSHE'] in events
    assert ['slippage', {'FixedSlippage': .001}] in events
    cost = next(e for e in events if e[0] == 'cost')
    assert cost[1] == {'open_tax': 0, 'close_tax': 0, 'open_commission': .0003,
        'close_commission': .0003, 'close_today_commission': 0, 'min_commission': 5}
    assert cost[2] == {'type': 'fund'}
    assert [(e[1], e[2]['time']) for e in events if e[0] == 'schedule'] == [
        ('my_trade', '9:30'), ('check_lose', 'open'), ('print_trade_info', '15:30')]


def test_stop_threshold_zero_cost_and_snapshot_semantics():
    assert len(batch.run_case(1, 'stop_exact_minus90')['orders']) == 1
    assert batch.run_case(1, 'stop_above_minus90')['orders'] == []
    assert batch.run_case(1, 'stop_zero_cost_error')['error']['type'] == 'ZeroDivisionError'
    row = batch.run_case(1, 'stop_snapshot_survives_deletion')
    assert row['error'] is None and len(row['orders']) == 2 and row['positions'] == []


def test_inserted_zero_keys_count_as_positions_without_fills():
    row = batch.run_case(0, 'inserting_zero_positions_pollute_count')
    assert row['error'] is None and len(row['positions']) == 2
    assert [p['amount'] for p in row['positions']] == [0, 0]


def test_all_source_callbacks_and_duplicate_scope():
    result = batch.diagnostics()
    assert result['case_count'] == 102 and result['unique_scenarios'] == 34
    assert {r['source_number'] for r in result['cases']} == {0, 1, 2}
    assert not result['duplicate_is_independent_economic_sample']
    assert not result['platform_feedback_equivalence_proven']
    assert result['new_trading_backtests'] == 0


@pytest.fixture
def offline_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['diagnose_strategy_batch94', 'offline'])
    bound = {'parent': 'checked by separate ancestry gate'}
    result = {'case_count': 102}
    monkeypatch.setattr(batch, 'parent_binding', lambda: bound)
    monkeypatch.setattr(batch, 'diagnostics', lambda: result)
    directory = batch.DIRECTORY
    directory.mkdir(parents=True)
    batch.save(directory/'parent-binding.json', bound)
    batch.save(directory/'diagnostics.json', result)
    hashes = {}
    for name in batch.FILES:
        live = tmp_path/name; frozen = directory/'implementation'/name
        live.parent.mkdir(parents=True, exist_ok=True)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        live.write_text('original immutable bytes\n')
        frozen.write_bytes(live.read_bytes())
        hashes[name] = batch.file_sha(live)
    batch.save(directory/'implementation-binding.json', hashes)
    return directory, hashes


def test_offline_complete_implementation_binding_succeeds(offline_tree):
    directory, _ = offline_tree
    batch.main()
    assert batch.read(directory/'offline-verification.json')['status'] == 'match'


@pytest.mark.parametrize('kind', ['empty', 'subset', 'extra'])
def test_offline_incomplete_or_extra_manifest_refuses_receipt(offline_tree, kind):
    directory, hashes = offline_tree
    manifest = {} if kind == 'empty' else dict(hashes)
    if kind == 'subset': manifest.pop(batch.FILES[-1])
    if kind == 'extra': manifest['unreviewed.py'] = next(iter(hashes.values()))
    (directory/'implementation-binding.json').write_text(batch.json.dumps(manifest))
    with pytest.raises(ValueError): batch.main()
    assert not (directory/'offline-verification.json').exists()


@pytest.mark.parametrize('copy', ['live', 'archive'])
def test_offline_tampered_implementation_refuses_receipt(offline_tree, copy):
    directory, _ = offline_tree
    target = batch.Path(batch.FILES[-1]) if copy == 'live' else directory/'implementation'/batch.FILES[-1]
    target.write_text('tampered implementation\n')
    with pytest.raises(ValueError): batch.main()
    assert not (directory/'offline-verification.json').exists()


@pytest.mark.parametrize('copy', ['manifest', 'live', 'archive', 'archive_parent'])
def test_offline_symlink_implementation_refuses_receipt(offline_tree, copy):
    directory, _ = offline_tree
    if copy == 'archive_parent':
        target = directory/'implementation'/'tests'
    elif copy == 'manifest': target = directory/'implementation-binding.json'
    elif copy == 'live': target = batch.Path(batch.FILES[-1])
    else: target = directory/'implementation'/batch.FILES[-1]
    replacement = target.with_name(target.name+'.original')
    target.rename(replacement)
    target.symlink_to(replacement.name, target_is_directory=copy == 'archive_parent')
    with pytest.raises(ValueError): batch.main()
    assert not (directory/'offline-verification.json').exists()
