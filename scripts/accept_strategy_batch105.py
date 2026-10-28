"""Accept two nontrading source reviews after longest operand checks; no NHNL or trade claims."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import reference_strategy_batch105 as reference
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-formal-source-acceptance')
CHECK = Path('data/staging/strategies-batch105/20261011-formal-source-validation')
STAGE = Path('data/staging/strategies-batch105/20261011-component-stage-recovery/checked.json')
RECEIPT = Path('docs/handoff/2026-10-11-batch105-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零五批市场宽度与行业新高新低-正式研究验收.md')
SPECS = ('docs/handoff/batch105-y2024b-052-source-review.yaml','docs/handoff/batch105-y2024a-089-source-review.yaml')
TESTS = ('tests/unit/test_batch105_research.py','tests/unit/test_batch105_reference.py','tests/unit/test_batch105_acceptance.py',
    'tests/unit/test_batch105_precision_recovery.py')


def reviews():
    result = {}
    for number,name in enumerate(SPECS):
        spec = yaml.safe_load(Path(name).read_text())
        require(spec['source'] == reference.study.SOURCES[number] and spec['source_sha256'] == reference.study.SOURCE_SHA[number]
            and not spec['formal_acceptance'] and not spec['original_strategy_complete'] and not spec['economic_variant_approved'],
            'Source/spec identity or scope differs')
        result[spec['source']] = {'review_status':'人工审查完成','status':'非交易研究脚本','scope':spec['review_scope'],
            'rules':spec['rules'],'fidelity':spec['allowed_evidence'],'gaps':spec['data_gaps']+spec['defects'],
            'next_step':spec['next'],'implementations':[],'original_strategy_complete':False}
    return result


def candidate_checks(old,new,selected):
    require(len(old) == len(new) == 695 and set(selected) == set(reference.study.SOURCES),'Catalog scope differs')
    previous = {r['path']:r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and set(previous) == {r['path'] for r in new},'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected),'Unrelated or missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before,allowed = previous[row['path']],selected[row['path']]
        require(before['review_status'] != '人工审查完成' and before.get('duplicate_of') is None and row.get('duplicate_of') is None,
            'Prior review or duplicate mapping differs')
        require({k:v for k,v in row.items() if k not in allowed} == {k:v for k,v in before.items() if k not in allowed},
            'Static source identity changed')
        require(all(row[k] == v for k,v in allowed.items()) and row['status'] == '非交易研究脚本'
            and not row['original_strategy_complete'] and row['implementations'] == [],'Rules or trading claim differs')
    require(sum(r['review_status'] == '人工审查完成' for r in old) == 294
        and sum(r['review_status'] == '人工审查完成' for r in new) == 296,'Manual review count differs')
    return {'sources':695,'reviewed':296,'remaining':399,'changed':sorted(selected),'unrelated_sources_unchanged':693}


def components():
    reference.validate(); proof = read(STAGE)
    require(proof['status'] == 'ok_component_stage' and proof['tests'] == {'local':22,'isolated':22}
        and not proof['formal_source_research_acceptance'] and not proof['formal_trading_acceptance'],'Component gate incomplete')
    for name,sha in proof['evidence_sha256'].items(): require(file_sha(name) == sha,'Stage changed: '+name)
    recovery = read(STAGE.parent/'recovery.json')
    require(recovery['status'] == 'ok' and recovery['checked_sha256'] == file_sha(STAGE),'Date recovery incomplete')
    for name,sha in recovery['evidence_sha256'].items(): require(file_sha(name) == sha,'Date recovery evidence changed')
    require(all(Path(name).is_relative_to((STAGE.parent/'isolated').resolve())
        for name in proof['actual_imports']['isolated'].values()),'Copied imports escaped')
    offline = read(reference.DIRECTORY/'offline-verification.json')
    require(offline['status'] == 'ok' and offline['fresh_network_blocked_processes'] == 2 and not offline['differences']
        and len(offline['outputs_sha256']) == 5,'Fresh workers incomplete')
    for record in offline['commands']:
        require(record['returncode'] == 0 and file_sha(record['log']) == record['log_sha256'],'Worker terminal evidence changed')
    for name,sha in offline['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(name.replace('/compute/','/replay/')),'Actual replay bytes differ')
    saved = proof['saved_readback']
    require((saved['native_rows'],saved['fixed40_rows'],saved['age_count_cells'],saved['age_security_calendar_cells'],
        saved['native_boundary_differences'],saved['fixed40_boundary_differences'],saved['engine_differences'],
        saved['age_count_differences'],saved['new_trading_backtests']) == (6811232,95860,10560,32503680,373,6,0,0,0),
        'Longest component/readback scope differs')
    boundary = read('data/staging/strategies-batch105/20261011-real-boundaries-and-protocol/checked.json')
    require(boundary['native_vs_fixed40_original_comparison_differences'] == 7 and not boundary['economic_rule_changes'],
        'Window protocol limitation not preserved')
    for name,sha in boundary['evidence_sha256'].items(): require(file_sha(name) == sha,'Boundary evidence changed')
    return {'tests':proof['tests'],'saved_readback':saved,'fresh_network_blocked_processes':2,
        'offline_outputs':5,'offline_differences':0,'original_diagnostics':15,'window_protocol_differences':7,
        'new_trading_backtests':0,'original_nhnl_core_reproduced':False,'complete_original_pool_signals':0}


def check():
    components(); require(not CHECK.exists(),'Immutable formal checks exist'); CHECK.mkdir(parents=True)
    isolated = CHECK/'isolated'; isolated.mkdir(); project = Path.cwd().resolve()
    files = set(read(reference.DIRECTORY/'prepared.json')['implementation_sha256'])|set(SPECS)|set(TESTS)|{'scripts/accept_strategy_batch105.py'}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).resolve().is_relative_to(project):
            rel = Path(name).resolve().relative_to(project).as_posix()
            if rel.startswith(('scripts/','src/observe/')): files.add(rel)
    files |= {str(reference.study.PREREAD/f'{n}.source') for n in (0,1)}
    files |= {'repo/量化策略源代码/'+name for name in reference.study.SOURCES}
    manifest = {}
    for name in sorted(files):
        target = isolated/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,target)
        manifest[name] = file_sha(name); require(file_sha(target) == manifest[name],'Formal copy differs')
    save(CHECK/'copy-manifest.json',manifest)
    worker = """import json,sys
from pathlib import Path
from scripts.recover_strategy_batch94_reference import network_blocked
import pytest
with network_blocked(): status=pytest.main(sys.argv[2:])
names=('scripts.accept_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr')
Path(sys.argv[1]).write_text(json.dumps({n:str(Path(sys.modules[n].__file__).resolve()) for n in names}))
raise SystemExit(status)
"""
    tests = {}; imports = {}
    for tag,cwd in (('local',project),('isolated',isolated.resolve())):
        xml = (CHECK/(tag+'-tests.xml')).resolve(); output = (CHECK/(tag+'-imports.json')).resolve()
        command = [sys.executable,'-c',worker,str(output),'-q',*TESTS,'--junitxml='+str(xml)]
        log = CHECK/(tag+'.log'); env = {**os.environ,'PYTHONPATH':str(cwd/'src')+os.pathsep+str(cwd)}
        with log.open('xb') as stream: result = subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        save(CHECK/(tag+'-command.json'),{'returncode':result.returncode,'command':command,'log_sha256':file_sha(log)})
        require(result.returncode == 0,'Formal tests failed: '+tag)
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        require(all(int(s.attrib.get(key,0)) == 0 for s in suites for key in ('errors','failures','skipped')),'Formal test scope incomplete')
        tests[tag] = sum(int(s.attrib['tests']) for s in suites); imports[tag] = read(output)
        require(all(Path(name).is_relative_to(cwd) for name in imports[tag].values()),'Formal imports escaped')
    require(tests == {'local':31,'isolated':31},'Formal test count differs')
    result = {'status':'ok_formal_ready','tests':tests,'actual_imports':imports,'copied_files':len(manifest),
        'evidence_sha256':{name:file_sha(name) for name in [str(STAGE),str(CHECK/'copy-manifest.json'),*SPECS,*TESTS,
            'scripts/accept_strategy_batch105.py']},'new_trading_backtests':0}
    save(CHECK/'checked.json',result); return {'status':result['status'],'tests':tests,'copied_files':len(manifest)}


def prepare():
    result = components(); selected = reviews(); guard = read(CHECK/'checked.json')
    require(guard['status'] == 'ok_formal_ready' and guard['tests'] == {'local':31,'isolated':31},'Formal tests incomplete')
    for name,sha in guard['evidence_sha256'].items(): require(file_sha(name) == sha,'Formal evidence changed')
    manifest = read(CHECK/'copy-manifest.json')
    require(len(manifest) == guard['copied_files'],'Formal copied file count differs')
    for name,sha in manifest.items():
        require(file_sha(name) == sha == file_sha(CHECK/'isolated'/name),'Formal implementation/copy changed: '+name)
    require(not DIRECTORY.exists() and not RECEIPT.exists() and not (set(selected) & set(REVIEWS)), 'Immutable directory or prior rule exists')
    active = read('data/catalog/strategies/latest.json'); before = Path('src/observe/strategy_catalog.py').read_bytes()
    old = read(Path(active['directory'])/'catalog.json')
    checkpoint('data',DIRECTORY,'Batch105 formal preparation;294 sources and12/87 trading counts unchanged')
    (DIRECTORY/'strategy_catalog.py.before').write_bytes(before); save(DIRECTORY/'source-reviews.json',selected)
    REVIEWS.update(selected); digests = []; candidate = None
    for tag in ('candidate','candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(DIRECTORY/'implementation-evidence.before.json',target)
        proposed = catalog_strategies(root,'repo/量化策略源代码'); folder = Path(proposed['output'])
        candidate_checks(old,read(folder/'catalog.json'),selected)
        digests.append({name:file_sha(folder/name) for name in ('catalog.json','catalog.parquet','summary.json')})
        if tag == 'candidate': candidate = proposed
    require(digests[0] == digests[1],'Candidate replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before,selected))
    bound = [*SPECS,str(STAGE),str(CHECK/'checked.json'),'scripts/accept_strategy_batch105.py',str(DIRECTORY/'source-reviews.json')]
    save(DIRECTORY/'prepared.json',{'components':result,'candidate':candidate,'catalog_bytes':digests[0],
        'previous_catalog':active,'evidence_sha256':{name:file_sha(name) for name in bound},
        'previous_code_sha256':file_sha(DIRECTORY/'strategy_catalog.py.before'),
        'proposed_code_sha256':file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status':'candidate_verified','catalog_id':candidate['catalog_id'],'not_published':True}


def publish():
    require(not RECEIPT.exists() and REPORT.is_file(),'Immutable receipt exists or report missing')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name,sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha,'Prepared evidence changed')
    before = (DIRECTORY/'strategy_catalog.py.before').read_bytes(); proposed = (DIRECTORY/'strategy_catalog.proposed.py').read_bytes()
    require(proposed == catalog_program(before,selected) and file_sha(DIRECTORY/'strategy_catalog.proposed.py') == doc['proposed_code_sha256'],
        'Proposed code differs')
    candidate = Path(doc['candidate']['output']); previous = read(Path(doc['previous_catalog']['directory'])/'catalog.json')
    candidate_checks(previous,read(candidate/'catalog.json'),selected)
    for name,sha in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == sha,'Candidate bytes differ')
    with operation_lock('data',DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); pointer = Path('data/catalog/strategies/progress/latest.json')
        source = Path('src/observe/strategy_catalog.py')
        require(read(latest) == doc['previous_catalog'] and source.read_bytes() == before,'Actual publication antecedent changed')
        require(components() == doc['components'],'Bound components changed')
        guard = read(CHECK/'checked.json')
        for name,sha in guard['evidence_sha256'].items(): require(file_sha(name) == sha,'Formal gate changed')
        for name,sha in read(CHECK/'copy-manifest.json').items():
            require(file_sha(name) == sha == file_sha(CHECK/'isolated'/name),'Formal implementation/copy changed before publish')
        protection = read(DIRECTORY/'protection.json'); baseline = read(DIRECTORY/'baseline.json')
        require(protection['status'] == 'ok' and protection['baseline_sha256'] == file_sha(DIRECTORY/'baseline.json')
            and protection['old_controls_unchanged'] == len(baseline['controls'])
            and protection['old_partition_metadata_unchanged'] == len(baseline['partitions']),'Fresh formal protection incomplete')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name,sha in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name,dest/name); require(file_sha(dest/name) == sha,'Installed catalog differs')
        old_latest = latest.read_bytes(); old_pointer = pointer.read_bytes()
        try:
            source.write_bytes(proposed)
            _atomic_json(latest,{'catalog_id':dest.name,'directory':str(dest),'summary':str(dest/'summary.json'),'sources':'repo/量化策略源代码'})
            progress = archive('data','Batch105 two nontrading sources accepted;6811232 native/95860 fixed40 cells,32503680 listing-age cells;31 local+31 isolated;373/6 boundaries and7 protocol differences preserved;trade inputs blocked')
            require((progress['manually_reviewed'],progress['locally_backtested_sources'],progress['locally_reproduced_sources'],progress['local_strategy_runs'])
                == (296,12,12,87),'Published research/trading counts differ')
            bound = [str(REPORT),*SPECS,str(STAGE),str(CHECK/'checked.json'),str(DIRECTORY/'prepared.json'),str(DIRECTORY/'protection.json'),
                str(dest/'catalog.json'),'src/observe/strategy_catalog.py','scripts/accept_strategy_batch105.py',
                'docs/handoff/2026-10-11-batch105-compatibility-stage.json']
            save(RECEIPT,{'status':'ok','formal_source_research_acceptance':True,'formal_trading_acceptance':False,
                'original_strategy_complete':False,'library_goal_complete':False,'progress':progress,'components':doc['components'],
                'formal_tests':read(CHECK/'checked.json')['tests'],'protection':protection,'source_sha256':list(reference.study.SOURCE_SHA),
                'new_independent_sources':2,'new_trading_backtests':0,'new_strategy_parent_runs':0,
                'evidence_sha256':{name:file_sha(name) for name in bound},'review_method':'Full sources and copied tests;no independent agent review'})
        except Exception:
            source.write_bytes(before); latest.write_bytes(old_latest); pointer.write_bytes(old_pointer); raise
    return {'status':'ok','checkpoint':progress['checkpoint'],'formal_reviewed':296,'remaining':399,'new_trading_backtests':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('check','prepare','protect','publish'))
    action = parser.parse_args().action
    result = check() if action == 'check' else prepare() if action == 'prepare' else protect('data',DIRECTORY) if action == 'protect' else publish()
    print(json.dumps(result,ensure_ascii=False))
