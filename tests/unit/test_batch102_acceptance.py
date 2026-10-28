"""Whole-catalog gate preserves canonical duplicates and rejects false trading claims."""
import copy

import pytest

from scripts import accept_strategy_batch102 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path': str(i), 'review_status': '人工审查完成' if i < 288 else '未审查', 'bytes_sha256': str(i)} for i in range(693)]
    old += [{'path': name, 'review_status': '未审查', 'bytes_sha256': str(i),
        'status': '重复版本' if i == 0 else '待数据', 'duplicate_of': 'S-d6db552a3293' if i == 0 else None} for i, name in enumerate(selected)]
    new = copy.deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(selected[row['path']])
    return old, new, selected


def test_only_two_source_reviews_advance_and_macro_stays_duplicate():
    old, new, selected = fixture()
    result = acceptance.candidate_checks(old, new, selected)
    assert result['reviewed'] == 290 and result['new_independent_sources'] == 1
    assert new[-2]['status'] == '重复版本' and new[-2]['duplicate_of'] == 'S-d6db552a3293'


@pytest.mark.parametrize('mode', ['unrelated', 'identity', 'rules', 'duplicate', 'count', 'duplicate_mapping', 'false_trades'])
def test_catalog_corruption_is_rejected(mode):
    old, new, selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = ['invented replacement']
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    elif mode == 'count': old[0]['review_status'] = new[0]['review_status'] = '未审查'
    elif mode == 'duplicate_mapping': new[-2]['duplicate_of'] = 'wrong'
    else: new[-1]['original_strategy_complete'] = True
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)
