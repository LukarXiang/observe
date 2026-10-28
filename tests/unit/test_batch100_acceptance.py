"""Reject repaired original rules, unrelated reviews and corrupted saved feature history."""
from copy import deepcopy

import pandas as pd
import pytest

from scripts import accept_strategy_batch100 as acceptance
from scripts import research_strategy_batch100 as study
from scripts import recover_strategy_batch100 as recovery
from scripts.run_strategy_batch4 import read


def catalog_fixture():
    old = [{'path': 'source'+str(i), 'review_status': '人工审查完成' if i < 284 else 'pending',
            'bytes_sha256': str(i), 'rules': {}} for i in range(695)]
    selected = {'source284': {'review_status': '人工审查完成', 'rules': {'bars': 4800}},
                'source285': {'review_status': '人工审查完成', 'rules': {'window': 400}}}
    new = deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(deepcopy(selected[row['path']]))
    return old, new, selected


def test_only_two_source_rules_can_be_promoted():
    old, new, selected = catalog_fixture()
    assert acceptance.candidate_checks(old, new, selected)['remaining'] == 409


@pytest.mark.parametrize('mode', ['unrelated', 'missing', 'sha', 'identity', 'rules', 'duplicate'])
def test_catalog_identity_and_scope_guard(mode):
    old, new, selected = catalog_fixture()
    if mode == 'unrelated': new[0]['rules'] = {'changed': True}
    elif mode == 'missing': new[284] = deepcopy(old[284])
    elif mode == 'sha': new[284]['bytes_sha256'] = 'wrong'
    elif mode == 'identity': new[284]['path'] = 'other'
    elif mode == 'rules': new[284]['rules']['bars'] = 20
    else: new[-1] = deepcopy(new[0])
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)


@pytest.mark.parametrize('mode', ['ETF_mapping', 'complete_sell', 'equal_threshold', 'rank_scalar'])
def test_literal_original_rule_repair_is_not_silently_accepted(mode):
    doc = study.diagnostics()
    assert acceptance.diagnostic_checks(doc)['cases'] == 32
    rows = {r['case']: r for r in doc['cases']}
    if mode == 'ETF_mapping': rows['sector_identity_mapping']['value']['ETFs']['000932.XSHG'] = '159928.XSHE'
    elif mode == 'complete_sell': rows['risk_off_two_positions']['value'].insert(1, ['target', 'old2', 0])
    elif mode == 'equal_threshold': rows['xgb_upper_equal']['value'].append(['buy', '510050.XSHG', 1234.])
    else: rows['rank_returns_series']['value'] = 2.5
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_missing_original_exception_or_real_model_scope_is_rejected():
    doc = study.diagnostics(); broken = deepcopy(doc)
    broken['real_pickle_executed'] = True
    with pytest.raises(ValueError): acceptance.diagnostic_checks(broken)
    next(r for r in doc['cases'] if r['case'] == 'rolling_rank_current_pandas_fails')['error'] = None
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_minute_history_cannot_be_replaced_with_daily_rows():
    doc = study.diagnostics()
    next(r for r in doc['cases'] if r['case'] == 'stable_tie_first_sector_and_minute_counts')['value']['calls'][0][1] = 10
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_actual_json_matrix_list_shape_is_required():
    doc = study.diagnostics()
    next(r for r in doc['cases'] if r['case'] == 'xgb_buy')['value'][0][-1] = '[[3.0,1.0]]'
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def real_history():
    return (pd.read_parquet(recovery.DIRECTORY/'study/assembled/feature-prefix.parquet'),
            pd.read_parquet(study.DIRECTORY/'ETF-input.parquet'), read(recovery.DIRECTORY/'study/assembled/summary.json'))


def test_complete_actual_saved_feature_scope():
    frame, operand, summary = real_history()
    result = acceptance.feature_checks(frame, operand, summary, scalar_dates=[frame.decision_date.iloc[0], frame.decision_date.iloc[-1]])
    assert result['windows'] == 4852 and result['features'] == 1059
    assert result['producer_independent_derived_comparisons'] == 4852*1034
    assert result['base25_comparisons_reuse_original_function_and_library'] is True


@pytest.mark.parametrize('mode', ['raw_price', 'strict_flag', 'current_date', 'full_pipeline', 'counts', 'scope', 'derived_tail'])
def test_actual_longest_saved_input_scope_and_corruption_rejected(mode):
    frame, operand, summary = real_history()
    if mode == 'raw_price': frame.loc[0, 'close'] += .01
    elif mode == 'strict_flag': frame.loc[0, 'long_MA5_flag'] = 1-frame.loc[0, 'long_MA5_flag']
    elif mode == 'current_date': frame.loc[0, 'last_input_date'] = frame.loc[0, 'decision_date']
    elif mode == 'full_pipeline': frame['alpha005'] = 0.
    elif mode == 'counts': summary['independent_scalar_cells'] = 0
    elif mode == 'scope': summary['strict_usable'] = True
    else: frame.loc[0, 'close_mean_rl_20D'] += .01
    with pytest.raises(ValueError):
        acceptance.feature_checks(frame, operand, summary, scalar_dates=[frame.decision_date.iloc[0]])


def test_catalog_program_changes_only_review_insertion():
    code = acceptance.catalog_program(b'before=1\nREVIEWS = {\n}\nafter=2\n', {'path': {'window': 400}})
    assert code.startswith(b'before=1\nREVIEWS = {\n') and code.endswith(b'}\nafter=2\n')
    with pytest.raises(ValueError): acceptance.catalog_program(b'no_marker', {})
