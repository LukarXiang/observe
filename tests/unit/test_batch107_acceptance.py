"""Source acceptance must preserve the other693 records and never invent ETF trades."""
import copy

import pytest

from scripts import accept_strategy_batch107 as acceptance


def fixture():
    selected = acceptance.reviews()
    old = [{'path': str(i), 'review_status': '人工审查完成' if i < 298 else '未审查',
        'bytes_sha256': str(i)} for i in range(693)]
    old += [{'path': n, 'review_status': '未审查', 'bytes_sha256': str(i)} for i, n in enumerate(selected)]
    new = copy.deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(selected[row['path']])
    return old, new, selected


def test_only_source_research_is_accepted():
    old, new, selected = fixture()
    assert acceptance.candidate_checks(old, new, selected) == {
        'sources': 695, 'reviewed': 300, 'remaining': 395, 'unrelated_sources_unchanged': 693}
    assert all(r['implementations'] == [] and r['original_strategy_complete'] is False for r in selected.values())


@pytest.mark.parametrize('mode', ['unrelated', 'identity', 'rules', 'duplicate', 'count', 'trades'])
def test_changed_scope_identity_or_fabricated_trading_is_rejected(mode):
    old, new, selected = fixture()
    if mode == 'unrelated': new[300]['review_status'] = '人工审查完成'
    elif mode == 'identity': new[-1]['bytes_sha256'] = 'changed'
    elif mode == 'rules': new[-1]['rules'] = {'signal': 'fabricated'}
    elif mode == 'duplicate': new[300]['path'] = new[301]['path']
    elif mode == 'count': old[0]['review_status'] = new[0]['review_status'] = '未审查'
    else: new[-1]['implementations'] = ['fictional_ETF_backtest']
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)
