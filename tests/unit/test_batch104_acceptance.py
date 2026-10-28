"""Limit acceptance to two original sources; forbid unrelated changes and false trades."""
import copy

import pytest

from scripts import accept_strategy_batch104 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path': str(i), 'review_status': '人工审查完成' if i < 292 else '未审查', 'bytes_sha256': str(i)} for i in range(693)]
    old += [{'path': name, 'review_status': '未审查', 'bytes_sha256': acceptance.price.study.SOURCE_SHA[i],
        'status': '待数据', 'duplicate_of': None} for i, name in enumerate(selected)]
    new = copy.deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(selected[row['path']])
    return old, new, selected


def test_two_source_rules_advance_to294_without_claiming_trading():
    old, new, selected = fixture()
    result = acceptance.candidate_checks(old, new, selected)
    assert result['reviewed'] == 294 and result['remaining'] == 401 and result['new_independent_sources'] == 2
    assert all(row['status'] == '待数据' and row['original_strategy_complete'] is False and
        row['implementations'] == [] for row in new[-2:])


@pytest.mark.parametrize('mode', ['unrelated', 'identity', 'rules', 'duplicate', 'count', 'duplicate_mapping', 'false_trades', 'implementation'])
def test_catalog_corruption_or_false_completion_rejected(mode):
    old, new, selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = ['invented deterministic replacement']
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    elif mode == 'count': old[0]['review_status'] = new[0]['review_status'] = '未审查'
    elif mode == 'duplicate_mapping': new[-1]['duplicate_of'] = 'wrong'
    elif mode == 'false_trades': new[-1]['original_strategy_complete'] = True
    else: new[-1]['implementations'] = ['unapproved_trade']
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)
