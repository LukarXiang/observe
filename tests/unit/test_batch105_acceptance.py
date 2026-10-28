"""Reject unrelated catalog edits, identities, fictional signals and partial review counts."""
import copy

import pytest

from scripts import accept_strategy_batch105 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path':str(i),'review_status':'人工审查完成' if i < 294 else '未审查','bytes_sha256':str(i)} for i in range(693)]
    old += [{'path':name,'review_status':'未审查','bytes_sha256':acceptance.reference.study.SOURCE_SHA[i],
        'status':'非交易研究脚本','duplicate_of':None} for i,name in enumerate(selected)]
    new = copy.deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(selected[row['path']])
    return old,new,selected


def test_only_two_nontrading_sources_advance_to296():
    old,new,selected = fixture(); result = acceptance.candidate_checks(old,new,selected)
    assert result['reviewed'] == 296 and result['remaining'] == 399 and result['unrelated_sources_unchanged'] == 693
    assert all(row['status'] == '非交易研究脚本' and not row['original_strategy_complete'] and not row['implementations'] for row in new[-2:])


@pytest.mark.parametrize('mode',['unrelated','identity','rules','duplicate','count','duplicate_mapping','false_trades','implementation'])
def test_catalog_corruption_or_false_trade_claim_rejected(mode):
    old,new,selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = ['invented NHNL algorithm']
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    elif mode == 'count': old[0]['review_status'] = new[0]['review_status'] = '未审查'
    elif mode == 'duplicate_mapping': new[-1]['duplicate_of'] = 'wrong'
    elif mode == 'false_trades': new[-1]['original_strategy_complete'] = True
    else: new[-1]['implementations'] = ['unauthorized_trade']
    with pytest.raises(ValueError): acceptance.candidate_checks(old,new,selected)
