"""Reject mismatched source catalogs, diagnostic scopes, and executable program edits."""
from copy import deepcopy

import pytest

from scripts import accept_strategy_batch97 as gate


def catalogs():
    old=[{'path':f'inert{i}','review_status':'人工审查完成' if i<278 else '静态扫描，尚未人工审查','bytes_sha256':str(i)} for i in range(695)]
    for i,name in enumerate(gate.SOURCE_REVIEWS,278): old[i]['path']=name
    now=deepcopy(old)
    for row in now:
        if row['path'] in gate.SOURCE_REVIEWS: row.update(deepcopy(gate.SOURCE_REVIEWS[row['path']]))
    return old,now


def test_only_two_review_entries_are_accepted():
    old,now=catalogs(); result=gate.candidate_checks(old,now)
    assert result['reviewed']==280 and result['remaining']==415


def test_unrelated_source_edit_is_rejected():
    old,now=catalogs(); now[0]['bytes_sha256']='drift'
    with pytest.raises(ValueError,match='Unrelated'): gate.candidate_checks(old,now)


def test_review_cannot_change_source_sha():
    old,now=catalogs(); now[278]['bytes_sha256']='different source'
    with pytest.raises(ValueError,match='identity'): gate.candidate_checks(old,now)


def test_review_cannot_change_economic_rule():
    old,now=catalogs()
    now[278]['rules']['fees']['commission']=.5
    with pytest.raises(ValueError,match='Rules'): gate.candidate_checks(old,now)


def test_duplicate_path_does_not_prove_695_sources():
    old,now=catalogs(); now[-1]['path']=now[-2]['path']
    with pytest.raises(ValueError,match='inventory'): gate.candidate_checks(old,now)


def test_unchanged_new_source_is_not_accepted():
    old,now=catalogs(); now[278]=deepcopy(old[278])
    with pytest.raises(ValueError,match='missing'): gate.candidate_checks(old,now)


def test_truncated_diagnostics_are_not_accepted():
    with pytest.raises(ValueError,match='scope'): gate.diagnostic_checks({'case_count':0,'cases':[]})


def test_duplicate_diagnostic_labels_are_not_accepted():
    with pytest.raises(ValueError,match='scope'):
        gate.diagnostic_checks({'case_count':31,'cases':[{'case':'duplicate'}]*31})


def test_catalog_program_extends_dictionary_without_touching_other_bytes():
    before=b'prefix\nREVIEWS = {\n    "legacy": {},\n}\nsuffix\n'
    proposed=gate.catalog_program(before)
    assert proposed.startswith(b'prefix\nREVIEWS = {\n')
    assert proposed.endswith(b'    "legacy": {},\n}\nsuffix\n')
    assert b'120rows' in proposed


def test_catalog_insertion_marker_must_be_unique():
    with pytest.raises(ValueError,match='marker'): gate.catalog_program(b'REVIEWS = {\nREVIEWS = {\n')
