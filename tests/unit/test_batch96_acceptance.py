"""Batch96 formal gate tests: refuse false scope, drift and unsafe publication."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS
from scripts import accept_strategy_batch96 as formal
from scripts import review_strategy_batch96 as stage


def catalog_pair():
    old=json.loads(formal.BASE_CATALOG.read_bytes());new=deepcopy(old)
    for row in new:
        if row['path'] in formal.SOURCES:row.update(deepcopy(REVIEWS[row['path']]))
    return old,new


def trees():
    old="REVIEWS = {'old': {'status': 'ok'}}\ndef helper():\n    return 1\n"
    pairs="'old': {'status':'ok'}, "+', '.join(repr(n)+": {'status':'pending'}" for n in formal.SOURCES)
    new='REVIEWS = {'+pairs+'}\ndef helper():\n    return 1\n'
    return old,new


def test_catalog_additions_only():
    formal.validate_catalog_trees(*trees())
    formal.validate_catalog_code()
    formal.validate_changes(*catalog_pair())


@pytest.mark.parametrize('mode',['old_value','old_delete','third_key','other_code'])
def test_catalog_ast_refuses_other_changes(mode):
    old,new=trees()
    if mode=='old_value':new=new.replace("'status':'ok'","'status':'broken'")
    elif mode=='old_delete':new=new.replace("'old': {'status':'ok'}, ",'')
    elif mode=='third_key':new=new.replace('REVIEWS = {',"REVIEWS = {'third': {}, ")
    else:new=new.replace('return 1','return 2')
    with pytest.raises(ValueError):formal.validate_catalog_trees(old,new)


@pytest.mark.parametrize('mode',['sha','trade','implementation','rules','unrelated'])
def test_false_catalog_rows_refused(mode):
    old,new=catalog_pair();row=next(r for r in new if r['path']==formal.SOURCES[0])
    if mode=='sha':row['bytes_sha256']='0'*64
    elif mode=='trade':row['original_strategy_complete']=True
    elif mode=='implementation':row['implementations']=['fake']
    elif mode=='rules':row['rules']['active_training']=True
    else:next(r for r in new if r['path'] not in formal.SOURCES)['status']='fake'
    with pytest.raises(ValueError):formal.validate_changes(old,new)


@pytest.mark.parametrize('mode',['status','differences','diagnostics','windows','network','trades'])
def test_replay_scope_refused(mode):
    doc={'status':'match','differences':0,'diagnostics':39,'windows':5082,
        'new_trading_backtests':0,'network_forbidden':True,'not_a_trading_backtest':True}
    formal.replay_scope(doc)
    key={'network':'network_forbidden','trades':'new_trading_backtests'}.get(mode,mode)
    doc[key]='ok' if mode=='status' else False if mode=='network' else 1
    with pytest.raises(ValueError):formal.replay_scope(doc)


def test_exact_junit_identity_not_empty_or_skipped(tmp_path,monkeypatch):
    monkeypatch.setattr(formal,'expected_tests',lambda:[('tests.s','one')])
    path=tmp_path/'tests.xml'
    root=ET.Element('testsuite');ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):formal.coverage(path)
    case=ET.SubElement(root,'testcase',classname='tests.s',name='one');ET.ElementTree(root).write(path)
    assert formal.coverage(path)['passed']==1
    ET.SubElement(case,'skipped');ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):formal.coverage(path)


def test_environment_clears_selection_and_import_flags(monkeypatch):
    for name in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONPATH'):monkeypatch.setenv(name,'untrusted')
    env=formal.environment()
    assert not any(n in env for n in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONPATH'))
    assert env['PYTHONDONTWRITEBYTECODE']=='1'


def test_source_contract_retains_original_rules():
    formal.validate_sources()
    assert REVIEWS[formal.SOURCES[0]]['rules']['period']==3
    assert REVIEWS[formal.SOURCES[1]]['rules']['trade_instrument']=='159901.XSHE'
    assert not REVIEWS[formal.SOURCES[1]]['rules']['reference_switch']['reset_persistent_lists']


def test_old_stage_binding_remains_strict():
    # The new path does not loosen stage.validate; catalog drift must still refuse it.
    with pytest.raises(ValueError,match='Frozen implementation differs'):stage.validate(stage.DIRECTORY)
    formal.stage_evidence()


def test_alternate_root_and_symlink_refused(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);Path('data').mkdir();Path('other').mkdir()
    with pytest.raises(ValueError):formal.require_root('other')
    Path('file').write_bytes(b'original');Path('link').symlink_to('file')
    with pytest.raises(ValueError):formal.bound('link',file_sha('file'))


def test_implementation_inventory_refuses_subset_or_drift(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(formal,'FILES',('one','two'))
    for name in formal.FILES:
        Path(name).write_text(name);p=Path('proof/implementation-checked')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    code=formal.implementation();formal.implementation(Path('proof'),code)
    with pytest.raises(ValueError):formal.implementation(Path('proof'),{'one':code['one']})
    Path('two').write_text('changed')
    with pytest.raises(ValueError):formal.implementation(Path('proof'),code)


def review_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(formal,'implementation',lambda *a:{'code':'a'*64})
    Path('proof').mkdir();rows=[]
    for role in ('security','architecture','assumptions','composition','cascade','abuse'):
        path=Path('proof')/(role+'.json')
        write_json(path,{'status':'returned','role':role,'scope':'batch96_formal_source_acceptance',
            'findings':[],'implementation_sha256':{'code':'a'*64}})
        rows.append({'scope':role,'file':str(path),'sha256':file_sha(path),'status':'returned','findings':[]})
    write_json('proof/review-completion.json',{'status':'completed','implementation_sha256':{'code':'a'*64},'reviewers':rows})
    return Path('proof'),rows


def test_review_body_cannot_hide_findings(tmp_path,monkeypatch):
    directory,rows=review_fixture(tmp_path,monkeypatch);formal.validate_reviews(directory)
    path=Path(rows[0]['file']);doc=json.loads(path.read_bytes());doc['findings']=[{'severity':'high'}];write_json(path,doc)
    completion=json.loads((directory/'review-completion.json').read_bytes());completion['reviewers'][0]['sha256']=file_sha(path)
    write_json(directory/'review-completion.json',completion)
    with pytest.raises(ValueError,match='Actual final review'):formal.validate_reviews(directory)


def test_checked_evidence_drift_refused(tmp_path,monkeypatch):
    directory,_=review_fixture(tmp_path,monkeypatch);monkeypatch.setattr(formal,'FILES',('code',))
    monkeypatch.setattr(formal,'checked_names',lambda d:{'output'})
    Path('output').write_text('fixed')
    checked={'implementation_sha256':{'code':'a'*64},'evidence_sha256':{'output':file_sha('output')}}
    write_json(directory/'checked-state.json',checked);sha=file_sha(directory/'checked-state.json');rsha=file_sha(directory/'review-completion.json')
    formal.validate_final_bindings(directory,checked,sha,rsha)
    Path('output').write_text('replacement')
    with pytest.raises(ValueError):formal.validate_final_bindings(directory,checked,sha,rsha)


def publication_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);directory=Path('proof');directory.mkdir()
    before={'catalog_id':'old','directory':'data/catalog/strategies/old','sha256':'b'*64}
    write_json('data/catalog/strategies/latest.json',before);write_json(directory/'catalog-before.json',before)
    source=directory/'candidate';source.mkdir();hashes={}
    for name in ('catalog.json','summary.json','catalog.parquet'):
        (source/name).write_bytes(name.encode());hashes[name]=file_sha(source/name)
    candidate={'catalog_id':'a'*20,'output':str(source)}
    write_json(directory/'catalog-offline.json',{'file_sha256':hashes})
    monkeypatch.setattr(formal,'validate_candidate',lambda d:candidate)
    return directory,candidate,hashes


def test_publication_copies_validated_bytes(tmp_path,monkeypatch):
    directory,candidate,hashes=publication_fixture(tmp_path,monkeypatch)
    result=formal.publish_candidate_locked('data',directory,candidate,hashes)
    assert all(file_sha(Path(result['output'])/n)==sha for n,sha in hashes.items())
    assert json.loads(Path('data/catalog/strategies/latest.json').read_bytes())['catalog_id']==candidate['catalog_id']


def test_publication_refuses_foreign_navigation(tmp_path,monkeypatch):
    directory,candidate,hashes=publication_fixture(tmp_path,monkeypatch)
    write_json('data/catalog/strategies/latest.json',{'catalog_id':'foreign'})
    with pytest.raises(ValueError):formal.publish_candidate_locked('data',directory,candidate,hashes)
    assert not (Path('data/catalog/strategies')/candidate['catalog_id']).exists()


def test_failure_restores_only_owned_navigation(tmp_path,monkeypatch):
    directory,candidate,hashes=publication_fixture(tmp_path,monkeypatch)
    progress=Path('data/catalog/strategies/progress/latest.json');write_json(progress,{'checkpoint':'old'})
    before=Path('data/catalog/strategies/latest.json').read_bytes()
    with pytest.raises(RuntimeError),formal.shared.publication_guard('data',directory,candidate):
        formal.publish_candidate_locked('data',directory,candidate,hashes)
        write_json(progress,{'checkpoint':'foreign'})
        raise RuntimeError('receipt failed')
    assert Path('data/catalog/strategies/latest.json').read_bytes()==before
    assert json.loads(progress.read_bytes())=={'checkpoint':'foreign'}


def test_runtime_replay_refuses_tampered_output(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(formal,'implementation',lambda *a:{'code':'a'*64})
    monkeypatch.setattr(stage,'OUTPUTS',('diagnostics.json',));monkeypatch.setattr(stage,'DIRECTORY',Path('stage'))
    directory=Path('proof');write_json(directory/'replay/diagnostics.json',{'cases':[1]});write_json('stage/diagnostics.json',{'cases':[1]})
    doc={'status':'match','differences':0,'diagnostics':39,'windows':5082,'new_trading_backtests':0,
        'network_forbidden':True,'not_a_trading_backtest':True,'implementation_sha256':{'code':'a'*64},
        'file_sha256':{'diagnostics.json':file_sha(directory/'replay/diagnostics.json')}}
    write_json(directory/'replay.json',doc);(directory/'replay.log').write_text('worker ran')
    write_json(directory/'replay-command.json',{'command':[sys.executable,'-B','-m','scripts.accept_strategy_batch96','worker','--output','replay'],
        'returncode':0,'log':str(directory/'replay.log'),'sha256':file_sha(directory/'replay.log')})
    formal.validate_replay(directory)
    write_json(directory/'replay/diagnostics.json',{'cases':[999]})
    with pytest.raises(ValueError):formal.validate_replay(directory)


def test_empty_checked_inventory_refused(tmp_path,monkeypatch):
    directory,_=review_fixture(tmp_path,monkeypatch);monkeypatch.setattr(formal,'FILES',('code',))
    checked={'implementation_sha256':{'code':'a'*64},'evidence_sha256':{}}
    write_json(directory/'checked-state.json',checked)
    with pytest.raises(ValueError):formal.validate_final_bindings(directory,checked,file_sha(directory/'checked-state.json'),file_sha(directory/'review-completion.json'))


def test_stage_old_asset_drift_cannot_become_new_baseline(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(stage,'DIRECTORY',Path('stage'))
    monkeypatch.setattr(stage,'inventory',lambda:{'inventory':1});monkeypatch.setattr(formal,'stage_evidence',lambda:{})
    monkeypatch.setattr(formal.shared,'validate_runs',lambda d:None)
    published={'batch_id':'fixture'};monkeypatch.setattr(formal,'Store',lambda root:SimpleNamespace(published=lambda:published))
    asset=Path('data/runs/old/config.json');asset.parent.mkdir(parents=True);asset.write_text('old')
    write_json('stage/baseline.json',{'published':published,'controls':{'runs/old/config.json':file_sha(asset)},'partitions':{}})
    asset.write_text('changed before formal preparation')
    directory=Path('proof')
    write_json(directory/'baseline.json',{'published':published,'controls':{'runs/old/config.json':file_sha(asset)},'partitions':{}})
    write_json(directory/'input-inventory.json',{'inventory':1});write_json(directory/'stage-evidence.json',{})
    write_json(directory/'implementation-evidence.before.json',[]);write_json('data/catalog/strategies/implementation-evidence.json',[])
    with pytest.raises(ValueError):formal.validate_inputs(directory)


def test_isolated_manifest_cannot_omit_runtime_files(tmp_path,monkeypatch):
    import tempfile
    monkeypatch.chdir(tmp_path);destination=Path(tempfile.mkdtemp(prefix='batch96-formal-isolated-',dir='/tmp'))
    (destination/'code').write_text('code');before={'code':file_sha(destination/'code')}
    monkeypatch.setattr(formal,'implementation',lambda *a:before);monkeypatch.setattr(formal,'coverage',lambda p:{})
    directory=Path('proof');directory.mkdir()
    (directory/'isolated-tests.log').write_text('tests');(directory/'isolated-tests.xml').write_text('<testsuite/>')
    write_json(directory/'isolated-files.json',{'destination':str(destination),'files':before,'implementation_sha256':before})
    write_json(directory/'isolated-imports.json',{'command':formal.import_command(),'returncode':0,
        'origins':{n:str(destination/p) for n,p in formal.IMPORT_FILES.items()}})
    write_json(directory/'isolated-verification.json',{'status':'passed','returncode':0,'command':formal.test_command(),
        'implementation_sha256':before,'coverage':{},'log_sha256':file_sha(directory/'isolated-tests.log'),
        'junit_sha256':file_sha(directory/'isolated-tests.xml'),'manifest_sha256':file_sha(directory/'isolated-files.json'),
        'imports_sha256':file_sha(directory/'isolated-imports.json')})
    with pytest.raises(ValueError):formal.validate_isolation(directory)
