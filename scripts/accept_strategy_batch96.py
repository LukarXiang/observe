"""Formal Batch96 source research; exact historical replay, never trading admission.

The old stage intentionally rejects current catalog changes. This new route verifies
all its frozen bytes, allows exactly two new REVIEWS keys, and calls only its unchanged
arithmetic/diagnostic functions under fresh-process forbidden-network guards.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source
from scripts import accept_strategy_batch95 as shared
from scripts import review_strategy_batch96 as stage
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch96/20261011-formal-source-acceptance-v2')
RECEIPT = Path('docs/handoff/2026-10-11-batch96-verification.json')
STAGE_RECEIPT = Path('docs/handoff/2026-10-11-batch96-callback-stage.json')
STAGE_SHA = '239d024111215b5087f8ddc83700f72d177cac220d6434614c6351fa3c1d0803'
BASE_CATALOG = Path('data/catalog/strategies/09bee9b103454931c9c9/catalog.json')
BASE_SHA = '0045b4eb9b7f483a10b93258e315930bedcd5f3d892b426eb35d3a3720a18f28'
SOURCES = ('2020年度精选策略/51 HS300--随机森林拐点识别.txt',
    '2021年度精选策略/69.（12.3更新v1.2）朱虽自用1号（神虽顶底+趋势）.txt')
FILES = tuple(dict.fromkeys(('src/observe/strategy_catalog.py', 'scripts/accept_strategy_batch96.py',
    'tests/unit/test_batch96_acceptance.py', *stage.FILES, 'scripts/accept_strategy_batch95.py',
    'scripts/verify_strategy_batch95.py', 'scripts/review_strategy_batch95.py',
    'scripts/review_strategy_batch94.py', 'scripts/review_strategy_batch59.py')))
TEST_FILES = ('tests/unit/test_batch96_stage.py', 'tests/unit/test_research_metadata.py',
    'tests/unit/test_batch96_acceptance.py')
IMPORT_FILES = {'observe':'src/observe/__init__.py', 'observe.strategy_catalog':'src/observe/strategy_catalog.py',
    'scripts.accept_strategy_batch96':'scripts/accept_strategy_batch96.py',
    'scripts.review_strategy_batch96':'scripts/review_strategy_batch96.py',
    'scripts.accept_strategy_batch95':'scripts/accept_strategy_batch95.py'}
NEW_CASES = ('test_catalog_additions_only',
    *(f'test_catalog_ast_refuses_other_changes[{v}]' for v in ('old_value','old_delete','third_key','other_code')),
    *(f'test_false_catalog_rows_refused[{v}]' for v in ('sha','trade','implementation','rules','unrelated')),
    *(f'test_replay_scope_refused[{v}]' for v in ('status','differences','diagnostics','windows','network','trades')),
    'test_exact_junit_identity_not_empty_or_skipped', 'test_environment_clears_selection_and_import_flags',
    'test_source_contract_retains_original_rules', 'test_old_stage_binding_remains_strict',
    'test_alternate_root_and_symlink_refused', 'test_implementation_inventory_refuses_subset_or_drift',
    'test_review_body_cannot_hide_findings', 'test_checked_evidence_drift_refused',
    'test_publication_copies_validated_bytes', 'test_publication_refuses_foreign_navigation',
    'test_failure_restores_only_owned_navigation', 'test_runtime_replay_refuses_tampered_output',
    'test_empty_checked_inventory_refused', 'test_stage_old_asset_drift_cannot_become_new_baseline',
    'test_isolated_manifest_cannot_omit_runtime_files')
regular = shared.regular
read_bound = shared.read_bound


def bound(path, sha):
    data = regular(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha, 'Bound bytes changed: '+str(path))
    return data


def require_root(root):
    shared.require_root(root)


def implementation(directory=None, manifest=None):
    hashes = {n:file_sha(regular(n)) for n in FILES}
    if manifest is not None:
        require(directory is not None and set(manifest) == set(FILES) and manifest == hashes,
            'Implementation inventory differs')
        for n,sha in manifest.items(): bound(directory/'implementation-checked'/n,sha)
    return hashes


def stage_evidence():
    doc = json.loads(bound(STAGE_RECEIPT, STAGE_SHA)); evidence = {str(STAGE_RECEIPT):STAGE_SHA}
    require(doc['status']=='ok' and not doc['formal_source_research_acceptance'] and
        not doc['formal_trading_acceptance'] and doc['new_trading_backtests']==0 and
        doc['manually_reviewed']==276 and (doc['diagnostics'],doc['daily_arithmetic_windows'])==(39,5082),
        'Historical stage scope differs')
    for n,sha in doc['evidence_sha256'].items(): bound(n,sha); evidence[n]=sha
    baseline_sha=doc['protection']['baseline_sha256']
    bound(stage.DIRECTORY/'baseline.json',baseline_sha);evidence[str(stage.DIRECTORY/'baseline.json')]=baseline_sha
    bound(doc['report'],doc['report_sha256']); evidence[doc['report']]=doc['report_sha256']
    hashes = read(stage.DIRECTORY/'implementation-binding.json')
    require(set(hashes)==set(stage.FILES),'Historical implementation inventory differs')
    for n,sha in hashes.items():
        frozen = stage.DIRECTORY/'implementation'/n; bound(frozen,sha); evidence[str(frozen)]=sha
        if n!='src/observe/strategy_catalog.py': bound(n,sha); evidence[n]=sha
    require(stage.parent_binding()==read(stage.DIRECTORY/'parent-binding.json'),'Accepted ancestry differs')
    parent = read(stage.PARENT); evidence[str(stage.PARENT)] = stage.PARENT_SHA
    for key in ('final_binding','receipt'):
        n=parent[key+'_file'];sha=parent[key+'_sha256'];bound(n,sha);evidence[n]=sha
    for n,sha in read(parent['final_binding_file'])['evidence_sha256'].items(): bound(n,sha);evidence[n]=sha
    for number,name in enumerate(SOURCES):
        for path in (Path('repo/量化策略源代码')/name, stage.DIRECTORY/'sources'/f'{number}.source',
            stage.PREREAD/f'{number}.source'):
            bound(path,stage.SHA[number]);evidence[str(path)]=stage.SHA[number]
    for row in read(stage.DIRECTORY/'input-inventory.json')['candidates']:
        for key in ('registry','prior_checked_state'):
            bound(row[key],row[key+'_sha256']);evidence[row[key]]=row[key+'_sha256']
        native=row['row'];bound(native['file'],native['sha256']);evidence[native['file']]=native['sha256']
        for wire in native['wire_responses']:bound(wire['file'],wire['sha256']);evidence[wire['file']]=wire['sha256']
    return evidence


def validate_catalog_trees(before, current):
    trees=[ast.parse(t) for t in (before,current)]; assignments=[]
    for tree in trees:
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='REVIEWS' for t in n.targets)]
        require(len(nodes)==1 and isinstance(nodes[0].value,ast.Dict),'Catalog REVIEWS assignment differs')
        assignments.append(nodes[0].value)
    old,new=assignments
    keys=lambda node:[ast.dump(k,include_attributes=False) for k in node.keys]
    additions={ast.dump(ast.Constant(n),include_attributes=False) for n in SOURCES}
    require(len(set(keys(old)))==len(keys(old)) and len(set(keys(new)))==len(keys(new)) and
        not additions&set(keys(old)) and set(keys(new))==set(keys(old))|additions,
        'Catalog keys changed beyond exact additions')
    pairs=[(k,v) for k,v in zip(new.keys,new.values,strict=True) if ast.dump(k,include_attributes=False) not in additions]
    new.keys=[p[0] for p in pairs];new.values=[p[1] for p in pairs]
    require(ast.dump(trees[0],include_attributes=False)==ast.dump(trees[1],include_attributes=False),
        'Old catalog code or review changed')


def validate_catalog_code():
    hashes=read(stage.DIRECTORY/'implementation-binding.json')
    before=bound(stage.DIRECTORY/'implementation/src/observe/strategy_catalog.py',hashes[FILES[0]])
    validate_catalog_trees(before.decode(),regular(FILES[0]).read_text())


def validate_sources():
    rows=read(stage.DIRECTORY/'source-rule-review.json')['sources']
    require(len(rows)==2,'Source rule count differs')
    for i,name in enumerate(SOURCES):
        text,encoding=read_source(stage.DIRECTORY/'sources'/f'{i}.source')
        require(rows[i]['path']==name and rows[i]['source_sha256']==stage.SHA[i] and
            len(text.splitlines())==rows[i]['lines_read']==(808 if i==0 else 467) and encoding==rows[i]['encoding'],
            'Complete source contract differs')
        if i==0:text='\n'*11+''.join(text.splitlines(keepends=True)[11:])
        functions=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
        definitions=[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,
            'direct_named_calls':sorted({c.func.id for c in ast.walk(n) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name)})}
            for n in functions]
        require(definitions==[{k:r[k] for k in definitions[0]} for r in rows[i]['functions']], 'Function inventory differs')
        review=REVIEWS[name]
        require(review['review_status']=='人工审查完成' and review['status']=='待数据' and
            not review['original_strategy_complete'] and review['implementations']==[], 'Source became trading')
    require(REVIEWS[SOURCES[0]]['rules']['active_training'] is False and
        REVIEWS[SOURCES[0]]['rules']['active_smallcap_filter'] is False and
        REVIEWS[SOURCES[0]]['rules']['indices']==['000016.XSHG','399333.XSHE'] and
        REVIEWS[SOURCES[1]]['rules']['reference_switch']['reset_persistent_lists'] is False and
        REVIEWS[SOURCES[1]]['rules']['callback_values']==200, 'Original active rules differ')


def validate_changes(old,new):
    require(isinstance(old,list) and isinstance(new,list) and len(old)==len(new)==695,'Catalog inventory differs')
    a={r['path']:r for r in old};b={r['path']:r for r in new}
    require(len(a)==len(b)==695 and set(a)==set(b) and {n for n in b if a[n]!=b[n]}==set(SOURCES),
        'Unrelated catalog changes')
    require(sum(r['review_status']=='人工审查完成' for r in b.values())==278,'Review count differs')
    for i,name in enumerate(SOURCES):
        expected=deepcopy(a[name]);expected.update(deepcopy(REVIEWS[name]))
        require(not a[name]['duplicate_of'] and expected==b[name] and b[name]['bytes_sha256']==stage.SHA[i],
            'New source rows differ')


def validate_baseline(directory):
    for row in read(directory/'isolation-baseline.json')['code']:
        if row['path']==FILES[0]:
            require(row['sha256']==read(stage.DIRECTORY/'implementation-binding.json')[FILES[0]],'Catalog baseline differs')
        else:bound(row['path'],row['sha256'])
    validate_catalog_code();validate_sources()


def validate_ancestry(directory):
    old=read(stage.DIRECTORY/'baseline.json');current=read(directory/'baseline.json')
    require(old['published']==current['published'] and
        all(current['controls'].get(n)==sha for n,sha in old['controls'].items()) and
        all(current['partitions'].get(n)==item for n,item in old['partitions'].items()),
        'Historical stage assets changed before formal baseline')


def validate_inputs(directory):
    require(stage.inventory()==read(directory/'input-inventory.json'),'Actual input inventory differs')
    require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence changed')
    validate_ancestry(directory)
    shared.validate_runs(directory)
    baseline=read(directory/'baseline.json')
    require(Store('data').published()==baseline['published'],'Publication differs')
    for n,sha in baseline['controls'].items():bound(Path('data')/n,sha)
    for n,item in baseline['partitions'].items():
        stat=regular(Path('data')/n).stat()
        require(stat.st_size==item['size'] and stat.st_mtime_ns==item['mtime_ns'],'Old partition metadata changed')
    bound('data/catalog/strategies/implementation-evidence.json',file_sha(directory/'implementation-evidence.before.json'))


def prepare(root,directory):
    require_root(root);require(directory==DIRECTORY and not directory.exists(),'Acceptance directory exists or differs')
    history=stage_evidence();inputs=stage.inventory();validate_catalog_code();validate_sources()
    index=read('data/metadata/index.json');code=read(index['implementation_files'])
    for r in code:
        if r['path']!=FILES[0]:bound(r['path'],r['sha256'])
    with operation_lock(root,DATA_WRITER):
        require(inputs==stage.inventory(),'Inputs changed while waiting')
        require(read('data/catalog/strategies/latest.json')=={'catalog_id':BASE_CATALOG.parent.name,
            'directory':str(BASE_CATALOG.parent),'sha256':BASE_SHA},'Active catalog is not accepted parent')
        bound(BASE_CATALOG,BASE_SHA)
        checkpoint(root,directory,'Batch96 formal research starts; no new trades')
        save(directory/'stage-evidence.json',history);save(directory/'input-inventory.json',inputs)
        save(directory/'isolation-baseline.json',{'code':code,'metadata_index':index})
        shutil.copyfile('data/catalog/strategies/latest.json',directory/'catalog-before.json')
        validate_ancestry(directory)
    return {'status':'prepared','new_trading_backtests':0}


def catalog(root,directory):
    require_root(root);require(directory==DIRECTORY and not (directory/'candidate.json').exists(),'Candidate exists or differs')
    validate_inputs(directory);validate_baseline(directory);old=json.loads(bound(BASE_CATALOG,BASE_SHA));descriptors=[]
    for role in ('catalog-candidate','catalog-offline'):
        folder=directory/role;folder.mkdir();target=folder/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True);shutil.copyfile(directory/'implementation-evidence.before.json',target)
        with network_blocked():item=catalog_strategies(folder,'repo/量化策略源代码')
        validate_changes(old,read(Path(item['output'])/'catalog.json'));descriptors.append(item)
    a,b=descriptors;shared.validate_catalog_artifacts(directory,a,b)
    hashes={n:file_sha(Path(a['output'])/n) for n in ('catalog.json','summary.json','catalog.parquet')}
    require(hashes=={n:file_sha(Path(b['output'])/n) for n in hashes},'Independent catalog bytes differ')
    save(directory/'candidate.json',a);save(directory/'catalog-offline.json',{'candidate':a,'duplicate':b,'file_sha256':hashes})
    return {'status':'candidate_only','catalog_id':a['catalog_id']}


def validate_candidate(directory):
    doc=read(directory/'catalog-offline.json');a=read(directory/'candidate.json');b=doc['duplicate']
    require(read(directory/'catalog-before.json')=={'catalog_id':BASE_CATALOG.parent.name,
        'directory':str(BASE_CATALOG.parent),'sha256':BASE_SHA},'Accepted catalog ancestry differs')
    require(a==doc['candidate'] and set(doc['file_sha256'])=={'catalog.json','summary.json','catalog.parquet'},'Candidate proof differs')
    validate_catalog_code();validate_sources();validate_changes(json.loads(bound(BASE_CATALOG,BASE_SHA)),read(Path(a['output'])/'catalog.json'))
    shared.validate_catalog_artifacts(directory,a,b)
    for n,sha in doc['file_sha256'].items():bound(Path(a['output'])/n,sha);bound(Path(b['output'])/n,sha)
    return a


def replay_scope(result):
    require(result['status']=='match' and result['differences']==result['new_trading_backtests']==0 and
        result['diagnostics']==39 and result['windows']==5082 and result['network_forbidden'] is True and
        result['not_a_trading_backtest'] is True,'Replay proof scope differs')


def worker(directory,output):
    require(directory==DIRECTORY and output in ('replay','final-replay') and sys.dont_write_bytecode and
        os.environ.get('OBSERVE_BATCH96_ACCEPT_WORKER')=='1','Fresh worker required')
    validate_baseline(directory);require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence changed')
    before=implementation();target=directory/output;require(not target.exists(),'Replay output exists');target.mkdir()
    with network_blocked():diagnostics=stage.diagnostics(stage.DIRECTORY);frame,summary=stage.components(stage.DIRECTORY)
    require(diagnostics==read(stage.DIRECTORY/'diagnostics.json') and summary==read(stage.DIRECTORY/'component-research.json'),
        'Original callback/summary replay differs')
    pd.testing.assert_frame_equal(frame,pd.read_parquet(stage.DIRECTORY/'daily-arithmetic.parquet'),check_exact=True)
    save(target/'diagnostics.json',diagnostics);save(target/'component-research.json',summary)
    with (target/'daily-arithmetic.parquet').open('xb') as stream:frame.to_parquet(stream,index=False)
    require(before==implementation() and stage_evidence()==read(directory/'stage-evidence.json'),'Replay inputs changed')
    result={'status':'match','differences':0,'diagnostics':len(diagnostics['cases']),'windows':len(frame),
        'network_forbidden':True,'not_a_trading_backtest':True,'new_trading_backtests':0,
        'implementation_sha256':before,'file_sha256':{n:file_sha(target/n) for n in stage.OUTPUTS}}
    replay_scope(result);save(directory/(output+'.json'),result);return result


def environment():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for n in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONPATH'):env.pop(n,None)
    return env


def replay(directory,output='replay'):
    command=[sys.executable,'-B','-m','scripts.accept_strategy_batch96','worker','--output',output]
    env=environment();env['OBSERVE_BATCH96_ACCEPT_WORKER']='1';log=directory/(output+'.log')
    with log.open('x') as stream:process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    save(directory/(output+'-command.json'),{'command':command,'returncode':process.returncode,'log':str(log),'sha256':file_sha(log)})
    require(process.returncode==0,'Fresh replay failed; retain log');return validate_replay(directory,output)


def validate_replay(directory,output='replay'):
    doc=read(directory/(output+'.json'));cmd=read(directory/(output+'-command.json'));replay_scope(doc)
    require(cmd['command']==[sys.executable,'-B','-m','scripts.accept_strategy_batch96','worker','--output',output] and
        cmd['returncode']==0 and cmd['log']==str(directory/(output+'.log')),'Replay command differs')
    bound(cmd['log'],cmd['sha256']);require(doc['implementation_sha256']==implementation() and set(doc['file_sha256'])==set(stage.OUTPUTS),'Replay code/output identity differs')
    for n,sha in doc['file_sha256'].items():
        bound(directory/output/n,sha)
        if n.endswith('.parquet'):pd.testing.assert_frame_equal(pd.read_parquet(directory/output/n),pd.read_parquet(stage.DIRECTORY/n),check_exact=True)
        else:require(read(directory/output/n)==read(stage.DIRECTORY/n),'Replay JSON differs')
    return doc


def test_command():return [sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*TEST_FILES]


def expected_tests():
    cases=list(ET.parse(stage.DIRECTORY/'tests.xml').getroot().iter('testcase'))
    return [(c.get('classname'),c.get('name')) for c in cases]+[('tests.unit.test_batch96_acceptance',n) for n in NEW_CASES]


def coverage(path):
    cases=list(ET.parse(path).getroot().iter('testcase'));ids=[(c.get('classname'),c.get('name')) for c in cases]
    require(len(ids)==len(set(ids)) and set(ids)==set(expected_tests()) and
        all(not any(c.find(tag) is not None for tag in ('failure','error','skipped')) for c in cases),'Exact JUnit coverage differs')
    return {'passed':len(ids),'skipped':0,'test_identities':[list(i) for i in ids]}


def import_command():
    script='import importlib,json; print(json.dumps({n:importlib.import_module(n).__file__ for n in '+repr(tuple(IMPORT_FILES))+'}))'
    return [sys.executable,'-B','-c',script]


def isolation_names(directory):
    before=implementation()
    names={r['path']:before.get(r['path'],r['sha256']) for r in read(directory/'isolation-baseline.json')['code']}
    names.update(stage_evidence());names.update(before);names[str(BASE_CATALOG)]=BASE_SHA
    require(all(p in names for p in IMPORT_FILES.values()),'Runtime imports absent from isolation inventory')
    return names


def isolate(directory):
    require(directory==DIRECTORY and not (directory/'isolated-files.json').exists(),'Isolation exists')
    validate_baseline(directory);before=implementation();destination=Path(tempfile.mkdtemp(prefix='batch96-formal-isolated-',dir='/tmp'))
    names=isolation_names(directory)
    for n,sha in names.items():
        require(not Path(n).is_absolute() and '..' not in Path(n).parts,'Unsafe copied path')
        target=destination/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bound(n,sha))
    save(directory/'isolated-files.json',{'destination':str(destination),'files':names,'implementation_sha256':before})
    env=environment();env['PYTHONPATH']=str(destination/'src')+os.pathsep+str(destination)
    process=subprocess.run(import_command(),cwd=destination,env=env,capture_output=True,text=True,timeout=30)
    require(process.returncode==0,'Isolated imports failed: '+process.stderr);origins=json.loads(process.stdout)
    require(origins=={n:str(destination/p) for n,p in IMPORT_FILES.items()},'Imports escaped copied source')
    save(directory/'isolated-imports.json',{'command':import_command(),'returncode':0,'origins':origins})
    log=directory/'isolated-tests.log';junit=directory/'isolated-tests.xml'
    with log.open('x') as stream:process=subprocess.run([*test_command(),'--junitxml='+str(junit.resolve())],cwd=destination,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    require(process.returncode==0,'Isolated tests failed; retain log');cov=coverage(junit)
    require(before==implementation(),'Tested implementation changed')
    save(directory/'isolated-verification.json',{'status':'passed','command':test_command(),'returncode':0,'coverage':cov,
        'implementation_sha256':before,'log_sha256':file_sha(log),'junit_sha256':file_sha(junit),
        'manifest_sha256':file_sha(directory/'isolated-files.json'),'imports_sha256':file_sha(directory/'isolated-imports.json')})
    return cov


def validate_isolation(directory):
    doc=read(directory/'isolated-verification.json');manifest=read(directory/'isolated-files.json');before=implementation()
    require(all(p in manifest['files'] for p in IMPORT_FILES.values()),'Runtime imports missing from copied inventory')
    require(manifest['files']==isolation_names(directory),'Exact isolated file inventory differs')
    require(doc['status']=='passed' and doc['returncode']==0 and doc['command']==test_command() and
        doc['implementation_sha256']==manifest['implementation_sha256']==before and
        doc['coverage']==coverage(directory/'isolated-tests.xml'),'Isolated check scope differs')
    for name,key in (('isolated-tests.log','log_sha256'),('isolated-tests.xml','junit_sha256'),
        ('isolated-files.json','manifest_sha256'),('isolated-imports.json','imports_sha256')):bound(directory/name,doc[key])
    destination=Path(manifest['destination'])
    require(destination.parent==Path('/tmp') and destination.name.startswith('batch96-formal-isolated-') and
        destination.is_dir() and not destination.is_symlink(),'Isolated destination differs')
    for n,sha in manifest['files'].items():
        path=destination/n
        require(not Path(n).is_absolute() and '..' not in Path(n).parts and path.is_file() and
            not any(p.is_symlink() for p in (path,*path.parents)) and file_sha(path)==sha,'Isolated copied bytes changed')
    require(all(manifest['files'].get(n)==sha for n,sha in before.items()),'Copied implementation inventory differs')
    imports=read(directory/'isolated-imports.json')
    require(imports=={'command':import_command(),'returncode':0,
        'origins':{n:str(Path(manifest['destination'])/p) for n,p in IMPORT_FILES.items()}},'Actual isolated imports differ')


def check_commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests',*FILES,
        'scripts/prepare_handoff.py','scripts/build_strategy_catalog.py','scripts/verify_financial_import.py'],test_command(),['git','diff','--check']]


def checks(directory):
    require(directory==DIRECTORY and not (directory/'checks').exists(),'Checks already archived')
    validate_baseline(directory);validate_inputs(directory);validate_candidate(directory);validate_replay(directory);validate_isolation(directory)
    before=implementation();folder=directory/'checks';folder.mkdir();commands=[]
    for i,command in enumerate(check_commands()):
        log=folder/f'{i}.log';actual=[*command,'--junitxml='+str(folder/'tests.xml')] if i==1 else command
        with log.open('x') as stream:process=subprocess.run(actual,env=environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        commands.append({'command':command,'returncode':process.returncode,'log':str(log),'sha256':file_sha(log)})
        require(process.returncode==0,'Required checks failed; retain log')
    cov=coverage(folder/'tests.xml');require(before==implementation(),'Tested implementation changed')
    for n,sha in before.items():
        target=directory/'implementation-checked'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bound(n,sha))
    save(directory/'checked-state.json',{'status':'passed','commands':commands,'coverage':cov,'implementation_sha256':before,
        'evidence_sha256':{n:file_sha(n) for n in sorted(checked_names(directory))}})
    return cov


def validate_reviews(directory):
    doc,sha=read_bound(directory/'review-completion.json');code=implementation()
    roles={'security','architecture','assumptions','composition','cascade','abuse'}
    require(doc['status']=='completed' and doc['implementation_sha256']==code and len(doc['reviewers'])==6 and
        {r['scope'] for r in doc['reviewers']}==roles,'Final review incomplete')
    for row in doc['reviewers']:
        actual=json.loads(bound(row['file'],row['sha256']))
        require(row['status']=='returned' and row['findings']==[] and actual['role']==row['scope'] and
            actual['status']=='returned' and actual['scope']=='batch96_formal_source_acceptance' and
            actual['findings']==[] and actual['implementation_sha256']==code,'Actual final review differs')
    return doc,sha


def checked_names(directory):
    names={'baseline.json','implementation-evidence.before.json','progress-start.json','stage-evidence.json',
        'input-inventory.json','isolation-baseline.json','catalog-before.json','candidate.json','catalog-offline.json',
        'replay.json','replay-command.json','replay.log','isolated-files.json','isolated-imports.json',
        'isolated-tests.log','isolated-tests.xml','isolated-verification.json',
        'checks/0.log','checks/1.log','checks/2.log','checks/tests.xml'}
    names.update('implementation-checked/'+n for n in FILES)
    candidate=read(directory/'candidate.json')
    for role in ('catalog-candidate','catalog-offline'):
        prefix=role+'/catalog/strategies/'
        names.update(prefix+n for n in ('implementation-evidence.json','latest.json'))
        names.update(prefix+candidate['catalog_id']+'/'+n for n in ('catalog.json','summary.json','catalog.parquet'))
    return {str(directory/n) for n in names}


def validate_final_bindings(directory,checked,checked_sha,review_sha):
    bound(directory/'checked-state.json',checked_sha);bound(directory/'review-completion.json',review_sha)
    require(set(checked['implementation_sha256'])==set(FILES),'Checked code inventory differs')
    require(bool(checked['evidence_sha256']) and set(checked['evidence_sha256'])==checked_names(directory),
        'Exact checked evidence inventory differs')
    implementation(directory,checked['implementation_sha256'])
    for n,sha in checked['evidence_sha256'].items():bound(n,sha)
    require(validate_reviews(directory)[1]==review_sha,'Review changed during replay')


def publish_candidate_locked(root,directory,candidate,hashes):
    require(validate_candidate(directory)==candidate and hashes==read(directory/'catalog-offline.json')['file_sha256'],'Candidate changed')
    latest=Path(root)/'catalog/strategies/latest.json';before=read(directory/'catalog-before.json')
    require(read(latest)==before,'Active catalog changed before publication')
    target=latest.parent/candidate['catalog_id'];source=Path(candidate['output'])
    if target.exists():
        for n,sha in hashes.items():bound(target/n,sha)
    else:
        temporary=Path(tempfile.mkdtemp(prefix='.batch96-catalog-',dir=target.parent))
        for n,sha in hashes.items():(temporary/n).write_bytes(bound(source/n,sha))
        os.rename(temporary,target)
    for n,sha in hashes.items():bound(target/n,sha);bound(source/n,sha)
    require(read(latest)==before,'Foreign navigation changed before switch')
    write_json(latest,{'catalog_id':candidate['catalog_id'],'directory':str(target),'sha256':hashes['catalog.json']})
    return {**candidate,'output':str(target)}


def finish(root,directory):
    require_root(root);require(directory==DIRECTORY and not RECEIPT.exists() and not (directory/'final-binding.json').exists(),'Acceptance exists or differs')
    validate_baseline(directory);validate_inputs(directory);validate_isolation(directory);checked,checked_sha=read_bound(directory/'checked-state.json')
    require(checked['status']=='passed' and checked['coverage']==coverage(directory/'checks/tests.xml') and
        [r['command'] for r in checked['commands']]==check_commands() and all(r['returncode']==0 for r in checked['commands']),
        'Checked command scope differs')
    for row in checked['commands']:bound(row['log'],row['sha256'])
    _,review_sha=validate_reviews(directory);validate_final_bindings(directory,checked,checked_sha,review_sha)
    candidate=validate_candidate(directory);validate_replay(directory);fresh=replay(directory,'final-replay')
    with operation_lock(root,DATA_WRITER):
        validate_inputs(directory);validate_final_bindings(directory,checked,checked_sha,review_sha)
        validate_replay(directory);validate_replay(directory,'final-replay');protected=protect(root,directory)
        hashes=read(directory/'catalog-offline.json')['file_sha256']
        with shared.publication_guard(root,directory,candidate) as ownership:
            published=publish_candidate_locked(root,directory,candidate,hashes)
            progress=archive(root,'Batch96 two sources formally researched;39diagnostics/5082daily reference;no trades')
            ownership['progress_navigation']={'checkpoint':progress['checkpoint'],'directory':progress['output'],
                'summary_sha256':file_sha(Path(progress['output'])/'summary.json')}
            require(read(Path(root)/'catalog/strategies/progress/latest.json')==ownership['progress_navigation'] and
                (progress['manually_reviewed'],progress['locally_backtested_sources'],progress['local_strategy_runs'])==(278,12,77),
                'Final totals/navigation differ')
            shared.validate_progress(progress,published,hashes);validate_final_bindings(directory,checked,checked_sha,review_sha)
            validate_inputs(directory);validate_replay(directory,'final-replay')
            receipt={'status':'ok','progress':progress,'accepted_directory':str(directory),'stage_receipt_sha256':STAGE_SHA,
                'checked_state':str(directory/'checked-state.json'),'checked_state_sha256':checked_sha,
                'review_completion':str(directory/'review-completion.json'),'review_completion_sha256':review_sha,
                'implementation_sha256':checked['implementation_sha256'],'catalog':published,'fresh_replay':fresh,'protection':protected,
                'newly_reviewed':2,'formal_source_research_acceptance':True,'formal_trading_acceptance':False,'new_trading_backtests':0,
                'source_sha256':list(stage.SHA),'original_strategy_complete':False,'not_a_trading_backtest':True,
                'restore_required':['Original CSV generation/version/availability and ind/tradestat provenance',
                    'Original 14:50 index inputs; 199 daily+14:30 1m indices with original switch/persistent state',
                    '159901 fund identity/events/adjustment/state/fees/execution; concrete economic repairs require confirmation']}
            receipt_sha=hashlib.sha256(json.dumps(receipt,ensure_ascii=False,indent=1,default=str).encode()).hexdigest()
            save(directory/'final-binding.json',{'receipt_sha256':receipt_sha,'progress':progress,
                'evidence_sha256':{str(p):file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
            save(RECEIPT,receipt);require(read(RECEIPT)==receipt and file_sha(RECEIPT)==receipt_sha,'Receipt read-back differs')
    return {'status':'ok','checkpoint':progress['checkpoint'],'reviewed':278,'new_trading_backtests':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','catalog','worker','replay','isolate','checks','finish'))
    parser.add_argument('--root',default='data');parser.add_argument('--output',choices=('replay','final-replay'),default='replay');args=parser.parse_args()
    require_root(args.root)
    if args.action=='prepare':result=prepare(args.root,DIRECTORY)
    elif args.action=='catalog':result=catalog(args.root,DIRECTORY)
    elif args.action=='worker':result=worker(DIRECTORY,args.output)
    elif args.action=='replay':result=replay(DIRECTORY,args.output)
    elif args.action=='isolate':result=isolate(DIRECTORY)
    elif args.action=='checks':result=checks(DIRECTORY)
    else:result=finish(args.root,DIRECTORY)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
