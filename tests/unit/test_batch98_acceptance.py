"""Reject incomplete diagnostics and unrelated catalog identity/rule changes."""
from copy import deepcopy
import json

import pytest

from scripts import accept_strategy_batch98 as acceptance
from scripts import research_strategy_batch98 as stage
from scripts import recover_strategy_batch98_diagnostics as recovery


def catalog_fixture():
    old = [{'path': 'source'+str(i), 'review_status': '人工审查完成' if i < 280 else 'pending',
            'bytes_sha256': str(i), 'rules': {}} for i in range(695)]
    selected = {'source280': {'review_status': '人工审查完成', 'rules': {'period': 250}},
                'source281': {'review_status': '人工审查完成', 'rules': {'period': 18}}}
    new = deepcopy(old)
    for row in new:
        if row['path'] in selected: row.update(deepcopy(selected[row['path']]))
    return old, new, selected


def test_catalog_changes_exactly_two_manual_rules():
    old, new, selected = catalog_fixture()
    assert acceptance.candidate_checks(old, new, selected)['remaining'] == 413


@pytest.mark.parametrize('mode', ['unrelated', 'missing', 'sha', 'identity', 'rule', 'duplicate'])
def test_catalog_rejects_scope_and_identity_mutations(mode):
    old, new, selected = catalog_fixture()
    if mode == 'unrelated': new[0]['rules'] = {'unrequested': True}
    elif mode == 'missing': new[280] = old[280]
    elif mode == 'sha': new[280]['bytes_sha256'] = 'changed'
    elif mode == 'identity': new[280]['path'] = 'unexpected'
    elif mode == 'rule': new[280]['rules']['period'] = 249
    elif mode == 'duplicate': new[-1] = deepcopy(new[0])
    with pytest.raises(ValueError): acceptance.candidate_checks(old, new, selected)


def diagnostics_fixture(monkeypatch):
    def original(number, **kwargs):
        return stage.definitions(stage.Path('repo/量化策略源代码')/stage.SOURCE_PATHS[number], number, kwargs)
    monkeypatch.setattr(stage, 'source_namespace', original)
    return stage.diagnostics()


def test_actual_original_diagnostics_gate(monkeypatch):
    doc = diagnostics_fixture(monkeypatch)
    assert acceptance.diagnostic_checks(doc)['cases'] == 27
    bad = deepcopy(doc); bad['cases'] = bad['cases'][:-1]
    with pytest.raises(ValueError): acceptance.diagnostic_checks(bad)


def test_actual_persisted_ranking_format_rejected_then_recovered(monkeypatch):
    original = diagnostics_fixture(monkeypatch)
    with pytest.raises(ValueError, match='Ranking/exit/stat'):
        acceptance.diagnostic_checks(json.loads(json.dumps(original, default=str)))
    monkeypatch.setattr(stage, 'canonical', recovery.canonical)
    fixed = diagnostics_fixture(monkeypatch)
    assert acceptance.diagnostic_checks(json.loads(json.dumps(fixed, allow_nan=False)))['cases'] == 27


@pytest.mark.parametrize('case,value', [('quantile17_always_nan', 0.),
    ('first_RSRS_zero_no_stock_history', 1.), ('interval_inclusive', [0, 0, 0, 0]),
    ('same_target_set_blocks_RSRS_exit', [None, [['sell', 'A', 0]]])])
def test_diagnostic_gate_rejects_silent_rule_repairs(monkeypatch, case, value):
    doc = diagnostics_fixture(monkeypatch)
    next(r for r in doc['cases'] if r['case'] == case)['value'] = value
    with pytest.raises(ValueError): acceptance.diagnostic_checks(doc)


def test_catalog_program_insertion_cannot_replace_body():
    assert acceptance.catalog_program(b'x=1\nREVIEWS = {\n}\ny=2\n', {'a': {'x': 1}}).endswith(b'}\ny=2\n')
    with pytest.raises(ValueError): acceptance.catalog_program(b'x=1', {})
