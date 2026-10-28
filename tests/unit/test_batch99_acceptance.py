"""Reject unrelated source changes, silently repaired rules and corrupted real bands."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from scripts import accept_strategy_batch99 as acceptance
from scripts import diagnose_strategy_batch99_timing as timing
from scripts import research_strategy_batch99 as stage
from scripts import recover_strategy_batch99_namespace as recovery


def catalog_fixture():
    old = [{'path': 'source'+str(i), 'review_status': '人工审查完成' if i < 282 else 'pending',
            'bytes_sha256': str(i), 'rules': {}} for i in range(695)]
    selected = {'source282': {'review_status': '人工审查完成', 'rules': {'window': 15}},
                'source283': {'review_status': '人工审查完成', 'rules': {'window': 3}}}
    new = deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(deepcopy(selected[row['path']]))
    return old, new, selected


def loader(n):
    return stage.definitions(Path('repo/量化策略源代码')/stage.SOURCE_PATHS[n], n)


def test_only_two_source_reviews_are_accepted():
    old, new, selected = catalog_fixture()
    assert acceptance.candidate_checks(old, new, selected)['remaining'] == 411


@pytest.mark.parametrize('mode', ['unrelated', 'missing', 'sha', 'identity', 'rule', 'duplicate'])
def test_catalog_identity_and_unrelated_rule_changes_rejected(mode):
    old, new, selected = catalog_fixture()
    if mode == 'unrelated': new[0]['rules'] = {'unexpected': 1}
    elif mode == 'missing': new[282] = deepcopy(old[282])
    elif mode == 'sha': new[282]['bytes_sha256'] = 'wrong'
    elif mode == 'identity': new[282]['path'] = 'wrong'
    elif mode == 'rule': new[282]['rules']['window'] = 14
    elif mode == 'duplicate': new[-1] = deepcopy(new[0])
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)


@pytest.mark.parametrize('case,value', [('K0_is_min_diff_not_largest_strike_below_F', 2.5),
    ('EMA_equal_buys', False), ('edge_strike_only_two', ['low', 'mid', 'high']),
    ('call_put_order', "['low', 'mid', 'high']")])
def test_original_boundaries_and_persisted_list_shapes_rejected(case, value):
    doc = stage.diagnostics([loader(0), loader(1)])
    assert acceptance.diagnostic_checks(doc)['preserved_original_exceptions'] == 13
    next(r for r in doc['cases'] if r['case'] == case)['value'] = value
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_missing_case_and_silent_exception_repair_rejected():
    doc = stage.diagnostics([loader(0), loader(1)])
    missing = deepcopy(doc); missing['cases'].pop()
    with pytest.raises(ValueError): acceptance.diagnostic_checks(missing)
    next(r for r in doc['cases'] if r['case'] == 'missing_put_field')['error'] = None
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_actual_timing_cutoff_rejects_current_date_and_held_rebalance():
    doc = timing.study(loader)
    assert acceptance.timing_checks(doc)['cases'] == 8
    broken = deepcopy(doc); broken['cases'][0]['calls'][0][1]['end_date'] = '2018-07-23'
    with pytest.raises(ValueError): acceptance.timing_checks(broken)
    broken = deepcopy(doc); broken['cases'][1]['calls'].append(['order_target_value', ['510050.XSHG', 123456.]])
    with pytest.raises(ValueError): acceptance.timing_checks(broken)


@pytest.mark.parametrize('mode', ['constant_stub', 'scalar', 'date', 'rounding_disclosure'])
def test_real_saved_history_gate_rejects_contamination_and_scope(mode):
    frame = pd.read_parquet(recovery.DIRECTORY/'study/ETF-range-components.parquet')
    operand = pd.read_parquet(recovery.DIRECTORY/'ETF-input.parquet')
    summary = stage.read(recovery.DIRECTORY/'study/summary.json')
    assert acceptance.range_checks(frame, operand, summary)['rows'] == 5249
    if mode == 'constant_stub': frame['original_up'], frame['original_down'] = 3., 2.
    elif mode == 'scalar': frame.loc[0, 'scalar_up'] += .001
    elif mode == 'date': frame.loc[0, 'last_previous_date'] = frame.loc[0, 'date']
    elif mode == 'rounding_disclosure': summary['rounding_difference_days']['scalar'] = 0
    with pytest.raises(ValueError): acceptance.range_checks(frame, operand, summary)


def test_catalog_program_only_inserts_review_entries():
    code = acceptance.catalog_program(b'x=1\nREVIEWS = {\n}\ny=2\n', {'a': {'window': 3}})
    assert code.startswith(b'x=1\nREVIEWS = {\n') and code.endswith(b'}\ny=2\n')
    with pytest.raises(ValueError): acceptance.catalog_program(b'x=1', {})
