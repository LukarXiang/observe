"""Source publication may change only two reviews and must not claim ETF trades."""
import copy

import pytest

from scripts import accept_strategy_batch106 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path':str(i),'review_status':'人工审查完成' if i<296 else '未审查','bytes_sha256':str(i)} for i in range(693)]
    old += [{'path':n,'review_status':'未审查','bytes_sha256':str(i)} for i,n in enumerate(selected)]
    new = copy.deepcopy(old)
    for r in new:
        if r['path'] in selected: r.update(selected[r['path']])
    return old,new,selected


def test_formal_source_review_does_not_admit_trading():
    old,new,selected = fixture()
    assert acceptance.candidate_checks(old,new,selected) == {'sources':695,'reviewed':298,'remaining':397,'unrelated_sources_unchanged':693}
    assert all(r['implementations'] == [] and r['original_strategy_complete'] is False for r in selected.values())


@pytest.mark.parametrize('mode',['unrelated','identity','rules','duplicate','count','trades'])
def test_catalog_scope_or_fabricated_trading_is_rejected(mode):
    old,new,selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = {'signal':'fabricated'}
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    elif mode == 'count': old[0]['review_status'] = new[0]['review_status'] = '未审查'
    else: new[-1]['implementations'] = ['fictional_ETF_backtest']
    with pytest.raises(ValueError): acceptance.candidate_checks(old,new,selected)
