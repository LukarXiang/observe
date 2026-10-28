"""Batch95 formal source research acceptance; no fund/60m trading admission."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha, write_json
from observe.strategy_catalog import API_NAMES, CATALOG_VERSION, REVIEWS, catalog_strategies, read_source
from scripts import review_strategy_batch95 as stage
from scripts import verify_strategy_batch95 as preparation
from scripts import recover_strategy_batch94_reference as recovery
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch95/20261011-magic3-v2-formal-acceptance')
RECEIPT = Path('docs/handoff/2026-10-11-batch95-verification.json')
PREPARATION_SHA = '692a0bcc2a3957e9c5a15f05ea080177fc35e847c400a092c037ee2481e841ab'
SOURCES = stage.SOURCES
BASE_CATALOG = Path('data/catalog/strategies/021347e0816f40d6a6b9/catalog.json')
BASE_CATALOG_SHA = '211bd8c261b55082523bdc123e39f06b81053a919b921cad5f457d21e89ed6ce'
SUPPLEMENTAL_SOURCE = '2020年度精选策略/02 我好像破解了聚宽擂台排第一的策略？！.txt'
SUPPLEMENTAL_FILES = {
    'data/catalog/strategies/implementation-evidence.json':'6fdd59cb86cba6e3c83f4fb3b9e060b72936b90bafa137b8f0afdf76671ab29b',
    'docs/handoff/2026-10-10-yili-cmb-earlier-window-final-review.json':'302a21b28c492fff0765ed9f99aef0d24f6bcfccc37eb3705f11f279a0f6929a',
    'data/staging/strategies-history-supplement/20261010-yili-cmb-pre2013/pair_yili_cmb_anchored_v1-verification.json':'16dc4df620da1302bc5696482121a8ea37b5836fda195ef5890e78c9b3fbb34a'}
FILES = ('src/observe/strategy_catalog.py', 'scripts/accept_strategy_batch95.py',
    'tests/unit/test_batch95_acceptance.py', *(n for n in preparation.FILES if n != 'src/observe/strategy_catalog.py'))
TEST_FILES = ('tests/unit/test_batch95_stage.py', 'tests/unit/test_batch95_preparation.py',
    'tests/unit/test_batch95_acceptance.py')
REPLAY_NAMES = stage.OUTPUTS
CORE = {'baseline.json', 'catalog-before.json', 'strategy_catalog.before.py', 'input-binding.json',
    'isolation-baseline.json', 'candidate.json', 'catalog-offline.json', 'replay.json', 'replay-command.json',
    'isolated-verification.json', 'isolated-files.json', 'isolated-imports.json', 'stage-ancestry.json', 'stage-evidence.json'}
IMPORT_FILES = {'observe':'src/observe/__init__.py',
    'observe.strategy_catalog':'src/observe/strategy_catalog.py',
    'scripts.accept_strategy_batch95':'scripts/accept_strategy_batch95.py',
    'scripts.verify_strategy_batch95':'scripts/verify_strategy_batch95.py',
    'scripts.review_strategy_batch95':'scripts/review_strategy_batch95.py'}
NEW_CASES = ('test_alternate_root_refused_before_reads',
    *(f'test_implementation_inventory_refuses_empty_subset_extra[{v}]' for v in ('empty','subset','extra')),
    *(f'test_implementation_inventory_checks_current_and_copy[{v}]' for v in ('live','copy')),
    'test_junit_exact_identity_refuses_empty_subset_duplicate_and_skip',
    'test_command_gate_refuses_empty_and_substituted', 'test_catalog_changes_are_exact',
    *(f'test_new_source_values_cannot_claim_false_trading[{v}]' for v in
        ('rules','status','implementations','original_strategy_complete','bytes_sha256')),
    *(f'test_catalog_artifacts_refuse_self_consistent_false_proof[{v}]' for v in
        ('catalog_id','summary','parquet','output','ancestry')),
    'test_new_run_inventory_refused','test_progress_checks_exact_catalog_identity',
    'test_environment_removes_selection_flags','test_unrelated_baseline_code_refused',
    'test_candidate_publication_uses_validated_bytes','test_candidate_publication_refuses_changed_navigation',
    'test_symlink_copy_refused','test_final_bindings_refuse_evidence_changed_during_replay',
    'test_source_reviews_preserve_nontrading_rules',
    *(f'test_review_summary_cannot_hide_actual_review[{v}]' for v in
        ('findings','status','implementation','role','late_bytes')),
    *(f'test_failed_publication_restores_only_owned_navigation[{v}]' for v in ('archive','receipt','competitor','foreign_progress')),
    'test_locked_inputs_refuse_archive_time_drift',
    *(f'test_existing_supplement_only_updates_verified_evidence[{v}]' for v in ('run_id','review_status','rules')))


def require_root(root):
    preparation.require_root(root)


def stage_evidence():
    doc=json.loads(preparation.bound(preparation.RECEIPT, PREPARATION_SHA))
    require(doc['status']=='ok' and doc['scope']=='acceptance_preparation_only' and
        not doc['formal_source_research_acceptance'] and not doc['formal_trading_acceptance'] and
        doc['new_trading_backtests']==0 and doc['progress']['manually_reviewed']==273,
        'Preparation scope differs')
    require(set(doc['implementation_sha256'])==set(preparation.FILES), 'Historical preparation inventory differs')
    for name,sha in doc['implementation_sha256'].items():
        preparation.bound(preparation.DIRECTORY/'implementation'/name, sha)
        if name != FILES[0]: preparation.bound(name,sha)
    for name,sha in doc['evidence_sha256'].items(): preparation.bound(name,sha)
    completion=read(preparation.DIRECTORY/'preparation-review-completion.json')
    require(completion['status']=='completed' and completion['review_angles']==6 and
        len(completion['reviews'])==6 and completion['local_passed']==completion['isolated_passed']==53,
        'Preparation review/check scope differs')
    for review in completion['reviews']:
        parsed=json.loads(preparation.bound(review['file'],review['sha256']))
        require(not parsed['findings'], 'Historical preparation findings')
    tests=json.loads(preparation.bound(completion['test_receipt'],completion['test_receipt_sha256']))
    require(tests['local_passed']==tests['isolated_passed']==53 and len(tests['test_identities'])==53,
        'Historical exact tests differ')
    return {'receipt_sha256':PREPARATION_SHA,'historical_stage_evidence':preparation.ancestry(),
        'review_completion_sha256':file_sha(preparation.DIRECTORY/'preparation-review-completion.json'),
        'test_receipt':completion['test_receipt'],'test_receipt_sha256':completion['test_receipt_sha256']}


def ancestry(directory):
    old=read(preparation.DIRECTORY/'baseline.json');now=read(directory/'baseline.json')
    require(old['published']==now['published'] and all(now['controls'].get(n)==sha for n,sha in old['controls'].items()) and
        all(now['partitions'].get(n)==v for n,v in old['partitions'].items()), 'Stage old assets changed')
    validate_runs(directory)
    return {'status':'ok','old_controls':len(old['controls']),'old_partitions':len(old['partitions']),
        'baseline_sha256':file_sha(directory/'baseline.json'),'stage_baseline_sha256':file_sha(preparation.DIRECTORY/'baseline.json')}


def validate_sources():
    rows=read(stage.DIRECTORY/'source-reviews.json')['sources']
    require(len(rows)==3,'Source count differs')
    for number,name in enumerate(SOURCES):
        original=Path('repo/量化策略源代码')/name;frozen=stage.DIRECTORY/'sources'/f'{number}.source'
        require(file_sha(regular(original))==file_sha(regular(frozen))==stage.SOURCE_SHA[number], 'Original bytes differ')
        a,b=(read_source(p)[0] for p in (original,frozen))
        require(a==b and len(a.splitlines())==rows[number]['lines']==(249 if number==2 else 171),'Full source differs')
        functions=[n.name for n in ast.parse(a).body if isinstance(n,ast.FunctionDef)]
        require(functions==rows[number]['functions'] and len(functions)==(15 if number==2 else 13),'Definitions differ')
        review=REVIEWS[name]
        require(review['review_status']=='人工审查完成' and review['status']=='待数据' and
            review['implementations']==[] and not review['original_strategy_complete'] and
            review['rules']['pool']==list(stage.POOL[number]) and
            review['rules']['fund_fees']['min_commission']==(0 if number==2 else 5),'Source review became trading/changed pool')


def validate_catalog_code(directory):
    before=preparation.DIRECTORY/'implementation'/FILES[0]
    expected=read(preparation.RECEIPT)['implementation_sha256'][FILES[0]]
    require(file_sha(regular(before))==expected,'Frozen prior catalog implementation differs')
    trees=[ast.parse(read_source(p)[0]) for p in (before,Path(FILES[0]))]
    for tree in trees:
        assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REVIEWS' for t in n.targets)]
        require(len(assignments)==1,'Catalog review assignment differs');assignments[0].value=ast.Constant(value=None)
    require(ast.dump(trees[0],include_attributes=False)==ast.dump(trees[1],include_attributes=False),'Catalog changed beyond reviews')


def supplemental_proof():
    docs={n:json.loads(preparation.bound(n,sha)) for n,sha in SUPPLEMENTAL_FILES.items()}
    evidence,accepted,validation=docs.values()
    require(accepted['status']=='ok' and accepted['formal_trading_acceptance'] and
        accepted['reproduction']['result']=='match' and accepted['reproduction']['differences']==0,
        'Existing supplemental trading acceptance differs')
    proof=[r for r in evidence if r['run_id']==accepted['run_id']]
    require(len(proof)==1 and proof[0]['source_sha256']==accepted['source_sha256'] and
        proof[0]['reproduction_run_id']==accepted['reproduction_run_id'] and proof[0]['reproduction_result']=='match' and
        proof[0]['differences']==0 and proof[0]['validation_file']==list(SUPPLEMENTAL_FILES)[2] and
        proof[0]['validation_sha256']==list(SUPPLEMENTAL_FILES.values())[2] and isinstance(validation,dict),
        'Supplemental proof provenance differs')
    return proof[0]


def validate_changes(old,new):
    require(isinstance(old,list) and isinstance(new,list) and len(old)==len(new)==695,'Catalog row inventory differs')
    old={r['path']:r for r in old};new={r['path']:r for r in new}
    require(len(old)==len(new)==695 and set(old)==set(new) and
        {n for n in new if new[n]!=old[n]}=={*SOURCES,SUPPLEMENTAL_SOURCE},'Unrelated catalog changes')
    require(sum(r['review_status']=='人工审查完成' for r in new.values())==276,'Review count differs')
    for number,name in enumerate(SOURCES):
        expected=deepcopy(old[name]);expected.update(deepcopy(REVIEWS[name]))
        if old[name]['duplicate_of']:expected.update(status='重复版本',next_step='共用实现但保留文件映射；核对 canonical_source 状态')
        require(new[name]==expected and new[name]['bytes_sha256']==stage.SOURCE_SHA[number],'Candidate source values differ')
    expected=deepcopy(old[SUPPLEMENTAL_SOURCE]);addition=supplemental_proof()
    require(addition['strategy_id']==expected['strategy_id'] and addition['source_sha256']==expected['bytes_sha256'],
        'Supplemental source identity differs')
    expected['implementation_evidence']=[*expected['implementation_evidence'],addition]
    require(new[SUPPLEMENTAL_SOURCE]==expected,'Existing source changed beyond accepted supplemental evidence')


def prepare(root,directory):
    require_root(root);require(directory==DIRECTORY and not directory.exists(),'Acceptance directory exists or differs')
    evidence=stage_evidence();inputs=preparation.input_inventory(root)
    require(inputs==read(preparation.DIRECTORY/'input-inventory.json'),'Actual inventory differs')
    validate_catalog_code(directory);validate_sources()
    index=read('data/metadata/index.json');code=read(index['implementation_files'])
    require(all(file_sha(regular(r['path']))==r['sha256'] for r in code),'Metadata code baseline differs')
    with operation_lock(root,DATA_WRITER):
        require(inputs==preparation.input_inventory(root),'Inputs changed while waiting for writer lock')
        checkpoint(root,directory,'Batch95 formal source acceptance starts; no new trades')
        save(directory/'stage-evidence.json',evidence)
        save(directory/'input-binding.json',{'inventory':inputs,'source_sha256':list(stage.SOURCE_SHA),'preparation_receipt_sha256':PREPARATION_SHA})
        save(directory/'isolation-baseline.json',{'code':code,'metadata_index':index,'metadata_index_sha256':file_sha('data/metadata/index.json')})
        shutil.copyfile(Path(root)/'catalog/strategies/latest.json',directory/'catalog-before.json')
        shutil.copyfile(FILES[0],directory/'strategy_catalog.before.py')
        save(directory/'stage-ancestry.json',ancestry(directory))
    return {'status':'prepared','new_trading_backtests':0}


def validate_candidate(directory):
    candidate=read(directory/'candidate.json');proof=read(directory/'catalog-offline.json')
    require(candidate==proof['candidate'] and proof['status']=='match' and candidate['catalog_id']==proof['duplicate']['catalog_id'] and
        set(proof['file_sha256'])=={'catalog.json','summary.json','catalog.parquet'} and proof['implementation_sha256']==implementation(directory),
        'Candidate proof differs')
    for name,sha in proof['file_sha256'].items():
        require(file_sha(regular(Path(candidate['output'])/name))==sha==file_sha(regular(Path(proof['duplicate']['output'])/name)),'Candidate bytes changed')
    validate_changes(validate_catalog_ancestry(directory),read(Path(candidate['output'])/'catalog.json'))
    validate_catalog_artifacts(directory,candidate,proof['duplicate']);validate_sources();validate_catalog_code(directory)
    return candidate


def catalog(root,directory):
    require_root(root);require(directory==DIRECTORY and not (directory/'candidate.json').exists(),'Candidate exists or differs')
    require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence differs')
    validate_baseline(directory);ancestry(directory);validate_sources();validate_catalog_code(directory)
    require(read(Path(root)/'catalog/strategies/latest.json')==read(directory/'catalog-before.json'),'Active catalog changed')
    old=validate_catalog_ancestry(directory);descriptors=[]
    for role in ('catalog-candidate','catalog-offline'):
        folder=directory/role;folder.mkdir(exist_ok=False)
        target=folder/'catalog/strategies/implementation-evidence.json';target.parent.mkdir(parents=True)
        shutil.copyfile(Path(root)/'catalog/strategies/implementation-evidence.json',target)
        with recovery.network_blocked():candidate=catalog_strategies(folder,'repo/量化策略源代码')
        validate_changes(old,read(Path(candidate['output'])/'catalog.json'));descriptors.append(candidate)
    candidate,duplicate=descriptors;require(candidate['catalog_id']==duplicate['catalog_id'],'Repeated ID differs')
    hashes={n:file_sha(Path(candidate['output'])/n) for n in ('catalog.json','summary.json','catalog.parquet')}
    require(hashes=={n:file_sha(Path(duplicate['output'])/n) for n in hashes},'Repeated catalog bytes differ')
    validate_catalog_artifacts(directory,candidate,duplicate)
    save(directory/'catalog-offline.json',{'status':'match','candidate':candidate,'duplicate':duplicate,'file_sha256':hashes,'implementation_sha256':implementation(directory)})
    save(directory/'candidate.json',candidate)
    return {'status':'candidate_only','catalog_id':candidate['catalog_id'],'active_catalog_changed':False}


def worker(directory,output):
    require(directory==DIRECTORY and output in ('replay','final-replay') and sys.dont_write_bytecode and
        os.environ.get('OBSERVE_BATCH95_ACCEPT_WORKER')=='1','Fresh worker required')
    require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence differs');validate_baseline(directory)
    before=implementation(directory);target=directory/output;require(not target.exists(),'Replay output exists');target.mkdir()
    with recovery.network_blocked():outputs=stage.compute(stage.DIRECTORY)
    require(set(outputs)==set(REPLAY_NAMES),'Replay inventory differs')
    for name,value in outputs.items():
        if isinstance(value,pd.DataFrame):
            pd.testing.assert_frame_equal(value,pd.read_parquet(stage.DIRECTORY/name),check_exact=True)
            with (target/name).open('xb') as out:value.to_parquet(out,index=False)
        else:require(value==read(stage.DIRECTORY/name),'Replay JSON differs');save(target/name,value)
    require(before==implementation(directory),'Replay code changed')
    result={'status':'match','differences':0,'scope':[len(outputs['rsrs-magic3.parquet']),len(outputs['rsrs-v2-reference.parquet']),len(outputs['native-momentum-reference.parquet']),outputs['diagnostics.json']['count']],
        'network_forbidden':True,'not_a_trading_backtest':True,'new_trading_backtests':0,'implementation_sha256':before,
        'file_sha256':{n:file_sha(target/n) for n in REPLAY_NAMES}}
    require(result['scope']==[4662,4662,22044,108],'Actual numerical scope differs');save(directory/(output+'.json'),result);return result


def replay(directory,output='replay'):
    require(directory==DIRECTORY and not (directory/(output+'-command.json')).exists(),'Replay command exists')
    command=[sys.executable,'-B','-m','scripts.accept_strategy_batch95','worker','--output',output]
    env=environment();env['OBSERVE_BATCH95_ACCEPT_WORKER']='1';env.pop('PYTHONPATH',None);log=directory/(output+'.log')
    with log.open('x') as stream:process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    save(directory/(output+'-command.json'),{'command':command,'returncode':process.returncode,'log':str(log),'sha256':file_sha(log)})
    require(process.returncode==0,'Fresh replay failed; preserve log');return validate_replay(directory,output)


def validate_replay(directory,output='replay'):
    result=read(directory/(output+'.json'));command=read(directory/(output+'-command.json'))
    require(command['command']==[sys.executable,'-B','-m','scripts.accept_strategy_batch95','worker','--output',output] and command['returncode']==0 and
        command['log']==str(directory/(output+'.log')) and file_sha(regular(command['log']))==command['sha256'],'Replay command differs')
    require(result['status']=='match' and result['scope']==[4662,4662,22044,108] and result['differences']==result['new_trading_backtests']==0 and
        result['network_forbidden'] and result['not_a_trading_backtest'] and result['implementation_sha256']==implementation(directory) and
        set(result['file_sha256'])==set(REPLAY_NAMES),'Replay proof scope differs')
    for n,sha in result['file_sha256'].items():
        preparation.bound(directory/output/n,sha)
        if n.endswith('.parquet'):pd.testing.assert_frame_equal(pd.read_parquet(directory/output/n),pd.read_parquet(stage.DIRECTORY/n),check_exact=True)
        else:require(read(directory/output/n)==read(stage.DIRECTORY/n),'Replay JSON values differ')
    return result


def test_command():return [sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*TEST_FILES]


def expected_tests():
    proof=stage_evidence();doc=json.loads(preparation.bound(proof['test_receipt'],proof['test_receipt_sha256']))
    return [*(tuple(i) for i in doc['test_identities']),*(('tests.unit.test_batch95_acceptance',n) for n in NEW_CASES)]


def check_commands():
    return [[str(Path(sys.executable).with_name('ruff')),'check','src','tests',*FILES,'scripts/prepare_handoff.py','scripts/build_strategy_catalog.py','scripts/verify_financial_import.py'],test_command(),['git','diff','--check']]


def import_command():
    script='import importlib,json; print(json.dumps({n:importlib.import_module(n).__file__ for n in '+repr(tuple(IMPORT_FILES))+'},sort_keys=True))'
    return [sys.executable,'-B','-c',script]


def isolate(directory):
    require(directory==DIRECTORY and not (directory/'isolated-files.json').exists(),'Isolation exists')
    validate_baseline(directory);candidate=implementation(directory);destination=Path(tempfile.mkdtemp(prefix='batch95-formal-isolated-',dir='/tmp'));files={}
    proof=read(preparation.RECEIPT)
    names={r['path']:candidate.get(r['path'],r['sha256']) for r in read(directory/'isolation-baseline.json')['code']}
    names.update(candidate);names.update(stage_evidence()['historical_stage_evidence'])
    names.update({str(preparation.RECEIPT):PREPARATION_SHA,str(BASE_CATALOG):BASE_CATALOG_SHA})
    names.update({str(preparation.DIRECTORY/'implementation'/n):sha for n,sha in proof['implementation_sha256'].items()})
    names.update(proof['evidence_sha256'])
    names.update(SUPPLEMENTAL_FILES)
    completion=read(preparation.DIRECTORY/'preparation-review-completion.json')
    names.update({r['file']:r['sha256'] for r in completion['reviews']});names[completion['test_receipt']]=completion['test_receipt_sha256']
    for n,sha in names.items():
        source=regular(n);require(not source.is_absolute() and '..' not in source.parts,'Unsafe copy path')
        target=destination/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(preparation.bound(source,sha));files[n]=sha
    save(directory/'isolated-files.json',{'destination':str(destination),'files':files,'candidate_implementation_sha256':candidate})
    env=environment();env['PYTHONPATH']=str(destination/'src')+os.pathsep+str(destination)
    imports=subprocess.run(import_command(),cwd=destination,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    origins=json.loads(imports.stdout) if imports.returncode==0 else {}
    require(imports.returncode==0 and origins=={n:str(destination/p) for n,p in IMPORT_FILES.items()},
        'Isolated runtime imported outside copied tree: '+imports.stderr)
    save(directory/'isolated-imports.json',{'command':import_command(),'returncode':imports.returncode,
        'origins':origins,'stdout':imports.stdout,'stderr':imports.stderr})
    log=directory/'isolated-tests.log';junit=directory/'isolated-tests.xml'
    with log.open('x') as stream:process=subprocess.run([*test_command(),'--junitxml='+str(junit.resolve())],cwd=destination,env=env,stdout=stream,stderr=subprocess.STDOUT)
    cov=coverage(junit);require(process.returncode==0 and candidate==implementation(directory),'Isolated checks failed or changed')
    save(directory/'isolated-verification.json',{'status':'passed','command':test_command(),'returncode':process.returncode,'coverage':cov,
        'log_sha256':file_sha(log),'junit_sha256':file_sha(junit),'manifest_sha256':file_sha(directory/'isolated-files.json'),'implementation_sha256':candidate})
    return cov


def validate_isolation(directory):
    proof=read(directory/'isolated-verification.json');copied=read(directory/'isolated-files.json')
    require(proof['status']=='passed' and proof['returncode']==0 and proof['command']==test_command() and
        proof['implementation_sha256']==copied['candidate_implementation_sha256']==implementation(directory) and
        proof['log_sha256']==file_sha(regular(directory/'isolated-tests.log')) and proof['junit_sha256']==file_sha(regular(directory/'isolated-tests.xml')) and
        proof['manifest_sha256']==file_sha(regular(directory/'isolated-files.json')) and coverage(directory/'isolated-tests.xml')==proof['coverage'],'Isolation proof differs')
    require(all(file_sha(Path(copied['destination'])/n)==sha for n,sha in copied['files'].items()) and
        all(copied['files'].get(n)==sha for n,sha in implementation(directory).items()),'Copied identity changed')
    imports=read(directory/'isolated-imports.json')
    require(imports['command']==import_command() and imports['returncode']==0 and
        imports['origins']==json.loads(imports['stdout'])=={n:str(Path(copied['destination'])/p) for n,p in IMPORT_FILES.items()},
        'Actual isolated import proof differs')


def finish(root,directory):
    require_root(root);require(directory==DIRECTORY and not RECEIPT.exists(),'Acceptance exists or differs')
    require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence changed')
    validate_baseline(directory);validate_isolation(directory);checked,checked_sha=read_bound(directory/'checked-state.json')
    require(checked['status']=='passed' and CORE<={Path(n).relative_to(directory).as_posix() for n in checked['evidence_sha256']},'Checked scope differs')
    implementation(directory,checked['implementation_sha256']);validate_commands(checked['commands']);coverage(directory/'checks/tests.xml')
    validate_replay(directory);candidate=validate_candidate(directory)
    _,review_sha=validate_reviews(directory)
    hashes=read(directory/'catalog-offline.json')['file_sha256'];fresh=replay(directory,'final-replay')
    require({k:v for k,v in fresh.items() if k!='file_sha256'}=={k:v for k,v in read(directory/'replay.json').items() if k!='file_sha256'},'Final actual replay differs')
    with operation_lock(root,DATA_WRITER):
        validate_final_bindings(directory,checked,checked_sha,review_sha)
        require(stage_evidence()==read(directory/'stage-evidence.json'),'Historical evidence changed')
        validate_replay(directory);validate_replay(directory,'final-replay');implementation(directory,checked['implementation_sha256'])
        require(preparation.input_inventory(root)==read(directory/'input-binding.json')['inventory'],'Locked input inventory differs')
        validate_runs(directory);require(ancestry(directory)==read(directory/'stage-ancestry.json'),'Ancestry differs')
        require(Store(root).published()==read(directory/'baseline.json')['published'],'Published input changed')
        protected=protect(root,directory);validate_final_bindings(directory,checked,checked_sha,review_sha)
        validate_locked_inputs(root,directory)
        with publication_guard(root,directory,candidate) as publication:
            published=publish_candidate_locked(root,directory,candidate,hashes)
            progress=archive(root,'Batch95 three sources formally researched;9324RSRS/22044native windows/108diagnostics;no trades')
            publication['progress_navigation']={'checkpoint':progress['checkpoint'],'directory':progress['output'],
                'summary_sha256':file_sha(regular(Path(progress['output'])/'summary.json'))}
            require(read(Path(root)/'catalog/strategies/progress/latest.json')==publication['progress_navigation'],
                'Progress navigation changed after archive')
            require((progress['manually_reviewed'],progress['locally_backtested_sources'],progress['local_strategy_runs'])==(276,12,77),'Final totals differ')
            validate_progress(progress,published,hashes);validate_final_bindings(directory,checked,checked_sha,review_sha)
            validate_replay(directory,'final-replay');implementation(directory,checked['implementation_sha256'])
            receipt={'status':'ok','progress':progress,'accepted_directory':str(directory),'preparation_receipt_sha256':PREPARATION_SHA,
                'checked_state':str(directory/'checked-state.json'),'checked_state_sha256':checked_sha,
                'review_completion':str(directory/'review-completion.json'),'review_completion_sha256':review_sha,
                'implementation_sha256':checked['implementation_sha256'],'catalog':published,'fresh_replay':fresh,'protection':protected,
                'newly_reviewed':3,'formal_source_research_acceptance':True,'formal_trading_acceptance':False,'new_trading_backtests':0,
                'source_sha256':list(stage.SOURCE_SHA),'original_strategy_complete':False,'not_a_trading_backtest':True,
                'restore_required':['159928 and all four fund identity/events/adjustment/state/reference/execution fees','v2 60m visibility and platform namespace/callback feedback; concrete repairs require approval']}
            receipt_sha=hashlib.sha256(json.dumps(receipt,ensure_ascii=False,indent=1,default=str).encode()).hexdigest()
            validate_locked_inputs(root,directory)
            save(directory/'final-binding.json',{'receipt_sha256':receipt_sha,'progress':progress,
                'evidence_sha256':{str(p):file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
            save(RECEIPT,receipt)
    return {'status':'ok','checkpoint':progress['checkpoint'],'reviewed':276,'new_trading_backtests':0}



def regular(path):
    path = Path(path)
    require(path.is_file() and path.resolve().is_relative_to(Path.cwd().resolve()) and
        not any(p.is_symlink() for p in (path, *path.parents)), 'Unrooted or symlink evidence: '+str(path))
    return path


def read_bound(path):
    """Parse and identify the same bytes, then later refuse any replacement."""
    data = regular(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def implementation(directory, manifest=None):
    hashes = {n: file_sha(regular(n)) for n in FILES}
    if manifest is not None:
        require(isinstance(manifest, dict) and set(manifest) == set(FILES) and manifest == hashes,
            'Implementation inventory differs')
        for name, sha in manifest.items():
            require(file_sha(regular(directory/'implementation-checked'/name)) == sha, 'Checked implementation copy differs')
    return hashes


def validate_baseline(directory):
    rows = read(directory/'isolation-baseline.json')['code']
    require(bool(rows), 'Empty isolation code baseline')
    for row in rows:
        require(row['path'] in FILES[:3] or file_sha(regular(row['path'])) == row['sha256'],
            'Unrelated baseline code changed')


def validate_runs(directory):
    baseline = read(directory/'baseline.json')
    old = {n for n in baseline['controls'] if Path(n).parts[0] == 'runs' and Path(n).name == 'config.json'}
    now = {p.relative_to('data').as_posix() for p in Path('data/runs').rglob('config.json')}
    require(bool(old) and old == now, 'Run inventory changed')


def coverage(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    identities = [(t.get('classname'), t.get('name')) for t in cases]
    require(len(identities) == len(set(identities)) and set(identities) == set(expected_tests()) and
        all(not any(t.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for t in cases),
        'Exact JUnit coverage differs')
    return {'passed': len(cases), 'skipped': 0, 'files': dict(Counter(c for c, _ in identities))}


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for n in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'): env.pop(n, None)
    return env


def validate_commands(commands):
    require([r['command'] for r in commands] == check_commands(), 'Exact required commands differ')
    require(all(r['returncode'] == 0 and file_sha(regular(r['log'])) == r['sha256'] for r in commands), 'Check log changed')


def checks(directory):
    require(directory == DIRECTORY and not (directory/'checks').exists(), 'Checks already archived')
    validate_baseline(directory); validate_replay(directory); before = implementation(directory)
    folder = directory/'checks'; folder.mkdir(); commands = []
    for i, command in enumerate(check_commands()):
        log = folder/f'{i}.log'
        actual = [*command, '--junitxml='+str(folder/'tests.xml')] if i == 1 else command
        with log.open('x') as stream: result = subprocess.run(actual, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
        commands.append({'command': command, 'returncode': result.returncode, 'log': str(log), 'sha256': file_sha(log)})
        require(result.returncode == 0, 'Required checks failed; preserve log')
    cov = coverage(folder/'tests.xml'); validate_commands(commands)
    require(before == implementation(directory), 'Tested code changed')
    for name in FILES:
        target = directory/'implementation-checked'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': commands,
        'implementation_sha256': before, 'coverage': cov,
        'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return cov


def validate_catalog_ancestry(directory):
    before = read(regular(directory/'catalog-before.json'))
    require(before == {'catalog_id': BASE_CATALOG.parent.name, 'directory': str(BASE_CATALOG.parent),
        'sha256': BASE_CATALOG_SHA} and file_sha(regular(BASE_CATALOG)) == BASE_CATALOG_SHA,
        'Frozen parent catalog provenance differs')
    return read(BASE_CATALOG)


def validate_catalog_artifacts(directory, candidate, duplicate):
    rows = read(Path(candidate['output'])/'catalog.json')
    identifier = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    require(isinstance(candidate['catalog_id'], str) and re.fullmatch(r'[0-9a-f]{20}', candidate['catalog_id'])
        and candidate['catalog_id'] == identifier, 'Catalog ID is not content-derived')
    summary = {'catalog_version': CATALOG_VERSION, 'catalog_id': identifier,
        'source_root': str(Path('repo/量化策略源代码').resolve()), 'files': len(rows),
        'distinct_contents': len({r['content_sha256'] for r in rows}),
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'suffix_counts': dict(Counter(Path(r['path']).suffix.lower() for r in rows)),
        'encoding_counts': dict(Counter(r['encoding'] for r in rows)),
        'api_file_counts': {a: sum(a in r['apis'] for r in rows) for a in API_NAMES},
        'reviewed_sources': [r['strategy_id'] for r in rows if r['review_status'] == '人工审查完成'],
        'caveat': '未审查文件的依赖/分类为静态候选；近似变体不会自动合并，不代表全量策略已实现'}
    expected_table = pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        for k, v in row.items()} for row in rows])
    for descriptor, role in ((candidate, 'catalog-candidate'), (duplicate, 'catalog-offline')):
        expected_output = directory/role/'catalog/strategies'/identifier
        require(descriptor == {**summary, 'output': str(expected_output)}, 'Candidate summary or output path differs')
        require(read(regular(expected_output/'summary.json')) == summary, 'Catalog summary values differ')
        pd.testing.assert_frame_equal(expected_table, pd.read_parquet(regular(expected_output/'catalog.parquet')),
            check_exact=True)


def validate_final_bindings(directory, checked, checked_sha, review_sha):
    require(file_sha(regular(directory/'checked-state.json')) == checked_sha and
        file_sha(regular(directory/'review-completion.json')) == review_sha and
        all(file_sha(regular(n)) == sha for n, sha in checked['evidence_sha256'].items()),
        'Checked evidence changed during final replay')
    _, current_review_sha = validate_reviews(directory)
    require(current_review_sha == review_sha, 'Reviewed completion changed')


def validate_reviews(directory):
    review, sha = read_bound(directory/'review-completion.json')
    code = implementation(directory)
    require(review['status']=='completed' and review['implementation_sha256']==code and
        {r['scope'] for r in review['reviewers']}=={'security','architecture','assumptions','composition','cascade','abuse'} and
        len(review['reviewers'])==6 and all(r['status']=='returned' and not r['findings'] for r in review['reviewers']),
        'Final review incomplete')
    for row in review['reviewers']:
        actual = json.loads(preparation.bound(row['file'],row['sha256']))
        require(actual['status']=='returned' and actual['role']==row['scope'] and
            actual['scope']=='batch95_formal_source_acceptance' and actual['findings']==[] and
            actual['implementation_sha256']==code, 'Actual final review differs')
    return review, sha


def validate_locked_inputs(root, directory):
    require(preparation.input_inventory(root)==read(directory/'input-binding.json')['inventory'],
        'Locked input inventory differs')
    validate_runs(directory)
    baseline=read(directory/'baseline.json')
    require(Store(root).published()==baseline['published'], 'Published input changed')
    require(all(file_sha(regular(Path(root)/name))==sha for name,sha in baseline['controls'].items()),
        'Old snapshot or experiment changed during acceptance')
    for name,item in baseline['partitions'].items():
        stat=regular(Path(root)/name).stat()
        require(stat.st_size==item['size'] and stat.st_mtime_ns==item['mtime_ns'],
            'Old partition metadata changed during acceptance')


def restore_pointer(path, data):
    fd, temporary=tempfile.mkstemp(prefix='.batch95-restore-',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


@contextmanager
def publication_guard(root, directory, candidate):
    """Restore only our mutable navigation on failure; keep all new evidence."""
    catalog=Path(root)/'catalog/strategies/latest.json'
    progress=catalog.parent/'progress/latest.json'
    before={path:regular(path).read_bytes() for path in (catalog,progress)}
    owned_catalog={'catalog_id':candidate['catalog_id'],
        'directory':str(catalog.parent/candidate['catalog_id']),
        'sha256':file_sha(regular(Path(candidate['output'])/'catalog.json'))}
    save(directory/'publication-before.json',{str(p):hashlib.sha256(b).hexdigest() for p,b in before.items()})
    ownership={}
    try:
        yield ownership
    except BaseException as exc:
        restored=[]; conflicts=[]
        for path,data in before.items():
            if regular(path).read_bytes()==data:continue
            doc=read(path)
            if path==catalog:
                ours=doc==owned_catalog
            else:
                ours=doc==ownership.get('progress_navigation')
            if ours:restore_pointer(path,data);restored.append(str(path))
            else:conflicts.append(str(path))
        save(directory/'publication-failure.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}',
            'restored_navigation':restored,'foreign_navigation_preserved':conflicts,
            'new_evidence_preserved':True,'formal_source_research_acceptance':False})
        raise


def publish_candidate_locked(root, directory, candidate, expected):
    """Caller holds DATA_WRITER; publish only the exact previously checked bytes."""
    require(validate_candidate(directory) == candidate, 'Candidate identity changed')
    latest = Path(root)/'catalog/strategies/latest.json'; before = read(directory/'catalog-before.json')
    require(read(latest) == before, 'Active catalog changed before publication')
    require(expected == read(directory/'catalog-offline.json')['file_sha256'], 'Frozen catalog bytes changed')
    source = Path(candidate['output']); dest = Path(root)/'catalog/strategies'/candidate['catalog_id']
    if dest.exists():
        require(all(file_sha(regular(dest/n)) == sha for n, sha in expected.items()), 'Existing catalog differs; no overwrite')
    else:
        temporary = Path(tempfile.mkdtemp(prefix='.batch95-catalog-', dir=dest.parent))
        for name, sha in expected.items():
            require(file_sha(regular(source/name)) == sha, 'Validated source catalog changed')
            shutil.copyfile(source/name, temporary/name)
            require(file_sha(temporary/name) == sha, 'Validated catalog copy changed')
        os.rename(temporary, dest)
    require(read(latest) == before and all(file_sha(regular(dest/n)) == sha == file_sha(regular(source/n))
        for n, sha in expected.items()), 'Catalog or navigation changed before switch')
    write_json(latest, {'catalog_id': candidate['catalog_id'], 'directory': str(dest), 'sha256': expected['catalog.json']})
    return {**candidate, 'output': str(dest)}


def validate_progress(progress, published, hashes):
    require(progress['catalog_id'] == published['catalog_id'] and progress['catalog_sha256'] == hashes['catalog.json'] and
        read('data/catalog/strategies/latest.json') == {'catalog_id': published['catalog_id'],
            'directory': published['output'], 'sha256': hashes['catalog.json']} and
        file_sha(Path(published['output'])/'catalog.json') == hashes['catalog.json'], 'Final catalog identity differs')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','catalog','worker','replay','isolate','checks','finish'))
    parser.add_argument('--root',default='data');parser.add_argument('--output',choices=('replay','final-replay'),default='replay')
    args=parser.parse_args();require_root(args.root)
    if args.action=='prepare':result=prepare(args.root,DIRECTORY)
    elif args.action=='catalog':result=catalog(args.root,DIRECTORY)
    elif args.action=='worker':result=worker(DIRECTORY,args.output)
    elif args.action=='replay':result=replay(DIRECTORY,args.output)
    elif args.action=='isolate':result=isolate(DIRECTORY)
    elif args.action=='checks':result=checks(DIRECTORY)
    else:result=finish(args.root,DIRECTORY)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
