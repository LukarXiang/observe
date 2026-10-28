"""Formal source catalog gates forbid unrelated changes or fabricated trading claims."""
import copy

import pytest

from scripts import accept_strategy_batch101 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path': str(i), 'review_status': '人工审查完成' if i < 286 else '未审查', 'bytes_sha256': str(i)} for i in range(693)]
    old += [{'path': name, 'review_status': '未审查', 'bytes_sha256': str(i)} for i, name in enumerate(selected)]
    new = copy.deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(selected[row['path']])
    return old, new, selected


def test_only_two_sources_can_advance_without_trading_claims():
    old, new, selected = fixture()
    assert acceptance.candidate_checks(old, new, selected)['reviewed'] == 288
    assert all(not row['original_strategy_complete'] and row['implementations'] == [] for row in selected.values())


@pytest.mark.parametrize('mode', ['unrelated', 'identity', 'rules', 'duplicate', 'count'])
def test_catalog_scope_and_rule_corruption_is_rejected(mode):
    old, new, selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = ['fabricated replacement']
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    else: old[0]['review_status'] = new[0]['review_status'] = '未审查'
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)
