"""Formal catalog acceptance rejects changed scope, hollow checks and unsafe publication."""
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from observe.strategy_catalog import API_NAMES, CATALOG_VERSION
from scripts import accept_strategy_batch95 as batch
from scripts.review_strategy_batch18 import save


def catalog_pair():
    assert batch.file_sha(batch.BASE_CATALOG)==batch.BASE_CATALOG_SHA
    old=batch.read(batch.BASE_CATALOG);new=deepcopy(old)
    for row in new:
        if row['path']==batch.SUPPLEMENTAL_SOURCE:
            row['implementation_evidence'].append(batch.supplemental_proof())
        if row['path'] in batch.SOURCES:
            row.update(deepcopy(batch.REVIEWS[row['path']]))
            if row['duplicate_of']:row.update(status='重复版本',next_step='共用实现但保留文件映射；核对 canonical_source 状态')
    return old,new


def test_catalog_changes_are_exact():
    old,new=catalog_pair();batch.validate_changes(old,new)
    wrong=deepcopy(new);wrong[-1]['title']+='unrelated'
    with pytest.raises(ValueError):batch.validate_changes(old,wrong)
    wrong=deepcopy(new);next(r for r in wrong if r['path']==batch.SOURCES[0])['rules']['RSRS']['N']=19
    with pytest.raises(ValueError):batch.validate_changes(old,wrong)


def test_source_reviews_preserve_nontrading_rules():
    batch.validate_sources()
    a,b,c=(batch.REVIEWS[n]['rules'] for n in batch.SOURCES)
    assert a==b and a['pool']!=c['pool']
    assert a['fund_fees']['min_commission']==5 and c['fund_fees']['min_commission']==0
    assert c['schedule']==['my_trade 9:30','check_lose open','hold_check 13:00']
    assert '600保留' in c['seed'] and '成599' in a['seed']
    assert '13:50' in c['buy_back'] and '不执行' in c['buy_back']
    assert all(batch.REVIEWS[n]['implementations']==[] for n in batch.SOURCES)


@pytest.mark.parametrize('field',['run_id','review_status','rules'])
def test_existing_supplement_only_updates_verified_evidence(field):
    old,new=catalog_pair(); batch.validate_changes(old,new)
    row=next(r for r in new if r['path']==batch.SUPPLEMENTAL_SOURCE)
    if field=='run_id':row['implementation_evidence'][-1]['run_id']='unverified'
    else:row[field]='unrelated change'
    with pytest.raises(ValueError):batch.validate_changes(old,new)


def test_alternate_root_refused_before_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'stage_evidence', lambda: pytest.fail('must refuse before reads'))
    with pytest.raises(ValueError, match='canonical'): batch.prepare(tmp_path, batch.DIRECTORY)


@pytest.fixture
def copied_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); names = ('script.py', 'test.py')
    monkeypatch.setattr(batch, 'FILES', names)
    directory = Path('acceptance'); hashes = {}
    for name in names:
        Path(name).write_text('original\n'); target = directory/'implementation-checked'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(name).read_bytes())
        hashes[name] = batch.file_sha(name)
    return directory, hashes


@pytest.mark.parametrize('kind', ['empty', 'subset', 'extra'])
def test_implementation_inventory_refuses_empty_subset_extra(copied_code, kind):
    directory, hashes = copied_code
    assert batch.implementation(directory, hashes) == hashes
    wrong = {} if kind == 'empty' else dict(hashes)
    if kind == 'subset': wrong.pop(batch.FILES[-1])
    if kind == 'extra': wrong['unreviewed.py'] = 'a'*64
    with pytest.raises(ValueError): batch.implementation(directory, wrong)


@pytest.mark.parametrize('copy', ['live', 'copy'])
def test_implementation_inventory_checks_current_and_copy(copied_code, copy):
    directory, hashes = copied_code
    path = Path(batch.FILES[0]) if copy == 'live' else directory/'implementation-checked'/batch.FILES[0]
    path.write_text('tampered\n')
    with pytest.raises(ValueError): batch.implementation(directory, hashes)


def junit(path, cases, skipped=False):
    tree = ET.Element('testsuite', tests=str(len(cases)), failures='0', errors='0', skipped=str(int(skipped)))
    for i, (cls, name) in enumerate(cases):
        t = ET.SubElement(tree, 'testcase', classname=cls, name=name)
        if skipped and i == 0: ET.SubElement(t, 'skipped')
    ET.ElementTree(tree).write(path)


def test_junit_exact_identity_refuses_empty_subset_duplicate_and_skip(tmp_path, monkeypatch):
    expected = [('scope', 'a'), ('scope', 'b')]; monkeypatch.setattr(batch, 'expected_tests', lambda: expected)
    path = tmp_path/'junit.xml'; junit(path, expected)
    assert batch.coverage(path)['passed'] == 2
    for cases, skipped in (([], False), (expected[:1], False), ([expected[0]]*2, False), (expected, True)):
        junit(path, cases, skipped)
        with pytest.raises(ValueError): batch.coverage(path)


def test_command_gate_refuses_empty_and_substituted(tmp_path, monkeypatch):
    log = tmp_path/'command.log'; log.write_text('actually ran\n')
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(batch, 'check_commands', lambda: [['ruff'], ['pytest'], ['git']])
    commands = [{'command': c, 'returncode': 0, 'log': str(log), 'sha256': batch.file_sha(log)} for c in batch.check_commands()]
    batch.validate_commands(commands)
    for wrong in ([], commands[:1], [dict(c, command=['true']) for c in commands]):
        with pytest.raises(ValueError): batch.validate_commands(wrong)


@pytest.mark.parametrize('field', ['rules', 'status', 'implementations', 'original_strategy_complete', 'bytes_sha256'])
def test_new_source_values_cannot_claim_false_trading(field):
    old, new = catalog_pair(); batch.validate_changes(old, new)
    row = next(r for r in new if r['path'] == batch.SOURCES[0])
    if field == 'rules': row['rules']['RSRS']['threshold'] = 999
    else: row[field] = {'status': '已验证', 'implementations': ['unauthorized'],
        'original_strategy_complete': True, 'bytes_sha256': '0'*64}[field]
    with pytest.raises(ValueError): batch.validate_changes(old, new)


@pytest.mark.parametrize('field', ['catalog_id', 'summary', 'parquet', 'output', 'ancestry'])
def test_catalog_artifacts_refuse_self_consistent_false_proof(tmp_path, monkeypatch, field):
    old, rows = catalog_pair()
    addition=deepcopy(next(r for r in rows if r['path']==batch.SUPPLEMENTAL_SOURCE)['implementation_evidence'][-1])
    monkeypatch.setattr(batch,'supplemental_proof',lambda:addition)
    monkeypatch.chdir(tmp_path); directory = Path('acceptance'); directory.mkdir()
    identifier = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    summary = {'catalog_version': CATALOG_VERSION, 'catalog_id': identifier,
        'source_root': str(Path('repo/量化策略源代码').resolve()), 'files': 695,
        'distinct_contents': len({r['content_sha256'] for r in rows}),
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'suffix_counts': dict(Counter(Path(r['path']).suffix.lower() for r in rows)),
        'encoding_counts': dict(Counter(r['encoding'] for r in rows)),
        'api_file_counts': {a: sum(a in r['apis'] for r in rows) for a in API_NAMES},
        'reviewed_sources': [r['strategy_id'] for r in rows if r['review_status'] == '人工审查完成'],
        'caveat': '未审查文件的依赖/分类为静态候选；近似变体不会自动合并，不代表全量策略已实现'}
    base = Path('data/catalog/strategies/old'); base.mkdir(parents=True); save(base/'catalog.json', old)
    old_sha=batch.file_sha(base/'catalog.json')
    save(directory/'catalog-before.json', {'catalog_id':base.name,'directory':str(base),'sha256':old_sha})
    monkeypatch.setattr(batch,'BASE_CATALOG',base/'catalog.json')
    monkeypatch.setattr(batch,'BASE_CATALOG_SHA',old_sha)
    monkeypatch.setattr(batch, 'implementation', lambda d: {'actual.py': 'sha'})
    monkeypatch.setattr(batch, 'validate_sources', lambda: None)
    monkeypatch.setattr(batch, 'validate_catalog_code', lambda d: None)
    def generate(identifier, summary, malformed=False, elsewhere=False):
        descriptors=[]; hashes={}
        for role in ('catalog-candidate', 'catalog-offline'):
            folder=directory/role/'catalog/strategies'/identifier
            if elsewhere: folder=directory/role/'elsewhere'/identifier
            folder.mkdir(parents=True, exist_ok=True)
            batch.write_json(folder/'catalog.json', rows); batch.write_json(folder/'summary.json', summary)
            table=pd.DataFrame([{k: json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                for k,v in r.items()} for r in rows])
            table.to_parquet(folder/'catalog.parquet',index=False)
            if malformed: (folder/'catalog.parquet').write_bytes(b'not parquet')
            hashes={n:batch.file_sha(folder/n) for n in ('catalog.json','summary.json','catalog.parquet')}
            descriptors.append({**summary,'catalog_id':identifier,'output':str(folder)})
        batch.write_json(directory/'candidate.json',descriptors[0])
        batch.write_json(directory/'catalog-offline.json',{'candidate':descriptors[0],'duplicate':descriptors[1],
            'status':'match','file_sha256':hashes,'implementation_sha256':{'actual.py':'sha'}})
    generate(identifier,summary); batch.validate_candidate(directory)
    wrong=deepcopy(summary)
    if field=='summary': wrong['files']=0
    if field=='ancestry':
        old[-1]['title']+='coordinated corruption'; rows[-1]['title']=old[-1]['title']
        batch.write_json(base/'catalog.json',old)
        identifier=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
        wrong['catalog_id']=identifier
        batch.write_json(directory/'catalog-before.json',{'catalog_id':base.name,'directory':str(base),
            'sha256':batch.file_sha(base/'catalog.json')})
    generate('not-derived' if field=='catalog_id' else identifier,wrong,field=='parquet',field=='output')
    with pytest.raises((ValueError, AssertionError)): batch.validate_candidate(directory)


def test_new_run_inventory_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('acceptance'); directory.mkdir()
    save(directory/'baseline.json', {'controls': {'runs/old/config.json': 'old'}})
    path = Path('data/runs/old/config.json'); path.parent.mkdir(parents=True); path.write_text('{}')
    batch.validate_runs(directory)
    extra = Path('data/runs/new/config.json'); extra.parent.mkdir(); extra.write_text('{}')
    with pytest.raises(ValueError): batch.validate_runs(directory)


def test_progress_checks_exact_catalog_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); folder = Path('data/catalog/strategies/test'); folder.mkdir(parents=True)
    save(folder/'catalog.json', []); digest = batch.file_sha(folder/'catalog.json')
    published = {'catalog_id': 'test', 'output': str(folder)}
    save(folder.parent/'latest.json', {'catalog_id': 'test', 'directory': str(folder), 'sha256': digest})
    batch.validate_progress({'catalog_id': 'test', 'catalog_sha256': digest}, published, {'catalog.json': digest})
    with pytest.raises(ValueError): batch.validate_progress({'catalog_id': 'other', 'catalog_sha256': digest}, published, {'catalog.json': digest})


def test_environment_removes_selection_flags(monkeypatch):
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k nonexistent'); monkeypatch.setenv('PYTEST_PLUGINS', 'unknown')
    env = batch.environment()
    assert 'PYTEST_ADDOPTS' not in env and 'PYTEST_PLUGINS' not in env
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_unrelated_baseline_code_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); Path('unrelated.py').write_text('good')
    directory = Path('acceptance'); directory.mkdir()
    save(directory/'isolation-baseline.json', {'code': [{'path': 'unrelated.py', 'sha256': batch.file_sha('unrelated.py')}]})
    batch.validate_baseline(directory); Path('unrelated.py').write_text('changed')
    with pytest.raises(ValueError): batch.validate_baseline(directory)


@pytest.fixture
def publication_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory = Path('acceptance'); directory.mkdir()
    base = Path('data/catalog/strategies'); base.mkdir(parents=True)
    old = {'catalog_id': 'old'}; save(base/'latest.json', old); save(directory/'catalog-before.json', old)
    source = directory/'candidate'; source.mkdir(); candidate = {'catalog_id': 'new', 'output': str(source)}
    hashes = {}
    for name in ('catalog.json', 'summary.json', 'catalog.parquet'):
        (source/name).write_bytes(('validated '+name).encode()); hashes[name] = batch.file_sha(source/name)
    save(directory/'catalog-offline.json', {'file_sha256': hashes})
    monkeypatch.setattr(batch, 'validate_candidate', lambda d: candidate)
    return directory, candidate, hashes


def test_candidate_publication_uses_validated_bytes(publication_tree):
    directory, candidate, hashes = publication_tree
    published = batch.publish_candidate_locked('data', directory, candidate, hashes)
    assert {n: batch.file_sha(Path(published['output'])/n) for n in hashes} == hashes


def test_candidate_publication_refuses_changed_navigation(publication_tree):
    directory, candidate, hashes = publication_tree
    path = Path('data/catalog/strategies/latest.json'); path.write_text('{"catalog_id":"competitor"}')
    with pytest.raises(ValueError): batch.publish_candidate_locked('data', directory, candidate, hashes)
    assert not Path('data/catalog/strategies/new').exists()


def test_symlink_copy_refused(copied_code):
    directory, hashes = copied_code
    path = directory/'implementation-checked'/batch.FILES[0]
    content = path.with_suffix('.original'); path.rename(content); path.symlink_to(content.name)
    with pytest.raises(ValueError): batch.implementation(directory, hashes)


def test_final_bindings_refuse_evidence_changed_during_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory=Path('acceptance'); directory.mkdir()
    path=directory/'candidate.json'; path.write_text('checked bytes')
    checked={'evidence_sha256':{str(path):batch.file_sha(path)}}
    save(directory/'checked-state.json',checked); save(directory/'review-completion.json',{'status':'completed'})
    checked_sha=batch.file_sha(directory/'checked-state.json'); review_sha=batch.file_sha(directory/'review-completion.json')
    monkeypatch.setattr(batch,'validate_reviews',lambda d: (batch.read(d/'review-completion.json'), batch.file_sha(d/'review-completion.json')))
    batch.validate_final_bindings(directory,checked,checked_sha,review_sha)
    path.write_text('bytes changed during child replay')
    with pytest.raises(ValueError): batch.validate_final_bindings(directory,checked,checked_sha,review_sha)


@pytest.mark.parametrize('mutation', ['findings', 'status', 'implementation', 'role', 'late_bytes'])
def test_review_summary_cannot_hide_actual_review(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(tmp_path); directory=Path('acceptance'); directory.mkdir()
    code={'owned.py': 'frozen'}
    monkeypatch.setattr(batch, 'implementation', lambda d: code)
    reviewers=[]
    for role in ('security','architecture','assumptions','composition','cascade','abuse'):
        path=directory/(role+'.json')
        actual={'status':'returned','role':role,'scope':'batch95_formal_source_acceptance',
            'findings':[],'implementation_sha256':code}
        save(path,actual)
        reviewers.append({'scope':role,'status':'returned','findings':[],
            'file':str(path),'sha256':batch.file_sha(path)})
    completion={'status':'completed','implementation_sha256':code,'reviewers':reviewers}
    save(directory/'review-completion.json',completion)
    batch.validate_reviews(directory)
    target=Path(reviewers[0]['file']); actual=batch.read(target)
    if mutation=='late_bytes':
        target.write_text('{}')
    else:
        key={'implementation':'implementation_sha256'}.get(mutation,mutation)
        actual[key]={'findings':['unresolved publication finding'],'status':'pending',
            'implementation':{'old.py':'old'},'role':'different'}[mutation]
        batch.write_json(target,actual)
        reviewers[0]['sha256']=batch.file_sha(target)
        batch.write_json(directory/'review-completion.json',completion)
    with pytest.raises(ValueError): batch.validate_reviews(directory)


@pytest.mark.parametrize('failure', ['archive','receipt','competitor','foreign_progress'])
def test_failed_publication_restores_only_owned_navigation(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path); directory=Path('acceptance'); directory.mkdir()
    base=Path('data/catalog/strategies'); (base/'progress').mkdir(parents=True)
    catalog=base/'latest.json'; progress=base/'progress/latest.json'
    save(catalog,{'catalog_id':'old'}); save(progress,{'checkpoint':'old'})
    previous={p:p.read_bytes() for p in (catalog,progress)}
    source=directory/'candidate'; source.mkdir(); save(source/'catalog.json',[])
    candidate={'catalog_id':'new','output':str(source)}
    with pytest.raises(RuntimeError,match=failure):
        with batch.publication_guard('data',directory,candidate) as publication:
            batch.write_json(catalog,{'catalog_id':'new','directory':str(base/'new'),
                'sha256':batch.file_sha(source/'catalog.json')})
            folder=base/'progress/20261010-235900-abcd'; folder.mkdir()
            save(folder/'summary.json',{'catalog_id':'new'})
            ours={'checkpoint':folder.name,'directory':str(folder),'summary_sha256':batch.file_sha(folder/'summary.json')}
            batch.write_json(progress,ours); publication['progress_navigation']=ours
            if failure=='competitor':batch.write_json(catalog,{'catalog_id':'competitor'})
            if failure=='foreign_progress':batch.write_json(progress,{**ours,'checkpoint':'another-writer'})
            raise RuntimeError(failure)
    if failure=='foreign_progress':assert batch.read(progress)['checkpoint']=='another-writer'
    else:assert progress.read_bytes()==previous[progress]
    if failure=='competitor':assert batch.read(catalog)=={'catalog_id':'competitor'}
    else:assert catalog.read_bytes()==previous[catalog]
    assert source.exists() and folder.exists() and (directory/'publication-failure.json').exists()


def test_locked_inputs_refuse_archive_time_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); directory=Path('acceptance'); directory.mkdir()
    save(directory/'input-binding.json',{'inventory':{'frozen':True}})
    save(directory/'baseline.json',{'published':{},'controls':{},'partitions':{}})
    current={'frozen':True}
    monkeypatch.setattr(batch.preparation,'input_inventory',lambda r:current)
    monkeypatch.setattr(batch,'validate_runs',lambda d:None)
    monkeypatch.setattr(batch,'Store',lambda r:type('StoreFixture',(),{'published':lambda self:{}})())
    batch.validate_locked_inputs('data',directory)
    current['frozen']=False
    with pytest.raises(ValueError,match='inventory'):batch.validate_locked_inputs('data',directory)

