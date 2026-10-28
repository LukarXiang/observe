"""Check formal source publication readiness; keep component and trading claims separate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts import accept_strategy_batch104 as acceptance
from scripts.recover_strategy_batch104_stage_receipt import numeric_readback
from scripts.verify_strategy_batch104_stage import readback as original_readback
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-formal-validation')
TESTS = ('tests/unit/test_batch104_research.py', 'tests/unit/test_batch104_reference.py',
         'tests/unit/test_batch104_receipt_recovery.py','tests/unit/test_batch104_acceptance.py')


def readback(root):
    values = original_readback(root)
    # The frozen readback produces numpy integer counters. Normalize the control
    # receipt only; the source functions and saved strategy operands stay unchanged.
    return numeric_readback({key:value if key == 'mean_max_abs_errors' else int(value)
        for key,value in values.items()})


def verify():
    prepared = reference.validate(); offline = read(reference.DIRECTORY/'offline-verification.json')
    require(offline['status'] == 'ok' and offline['fresh_network_blocked_processes'] == 2 and not offline['differences'], 'Fresh replay evidence differs')
    require(not DIRECTORY.exists(), 'Immutable stage checks exist'); DIRECTORY.mkdir(parents=True)
    isolated = DIRECTORY/'isolated'; isolated.mkdir()
    files = set(prepared['implementation_sha256'])|{__file__,*TESTS,*acceptance.SPECS}
    project = Path.cwd().resolve()
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).resolve().is_relative_to(project):
            rel = Path(name).resolve().relative_to(project).as_posix()
            if rel.startswith(('scripts/','src/observe/')): files.add(rel)
    files |= {str(reference.study.PREREAD/f'{n}.source') for n in (0,1)}
    files |= {str(Path('repo/量化策略源代码')/name) for name in reference.study.SOURCES}
    files |= {str(reference.DIRECTORY/'prices.parquet')}
    files |= {str(reference.DIRECTORY/'compute'/name) for name in reference.OUTPUTS}
    copy_hashes = {}
    for name in sorted(files):
        source = Path(name); name = source.resolve().relative_to(project).as_posix()
        dest = isolated/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
        copy_hashes[name] = file_sha(source); require(file_sha(dest) == copy_hashes[name], 'Copied bytes differ')
    save(DIRECTORY/'copy-manifest.json',copy_hashes)
    script = """import json,sys
from pathlib import Path
from scripts.recover_strategy_batch94_reference import network_blocked
import pytest
with network_blocked():
    status=pytest.main(sys.argv[2:])
modules={n:str(Path(sys.modules[n].__file__).resolve()) for n in ('scripts.research_strategy_batch104','scripts.reference_strategy_batch104','scripts.accept_strategy_batch104','scripts.recover_strategy_batch104_stage_receipt','observe.strategy_catalog')}
Path(sys.argv[1]).write_text(json.dumps(modules))
raise SystemExit(status)
"""
    counts = {}; imported = {}
    for tag,cwd in [('local',project),('isolated',isolated.resolve())]:
        module_output = (DIRECTORY/(tag+'-imports.json')).resolve(); xml = (DIRECTORY/(tag+'-tests.xml')).resolve()
        command = [sys.executable,'-c',script,str(module_output),'-q',*TESTS,'--junitxml='+str(xml)]
        env = {**os.environ,'PYTHONPATH':str(cwd/'src')+os.pathsep+str(cwd)}
        with (DIRECTORY/(tag+'.log')).open('xb') as log:
            result = subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        save(DIRECTORY/(tag+'-command.json'),{'command':command,'returncode':result.returncode,'log_sha256':file_sha(DIRECTORY/(tag+'.log'))})
        require(result.returncode == 0,'Stage tests failed: '+tag)
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        require(all(int(s.attrib.get(key,0)) == 0 for s in suites for key in ('errors','failures','skipped')), 'Incomplete test scope')
        counts[tag] = sum(int(s.attrib['tests']) for s in suites)
        if tag == 'isolated':
            imported = read(module_output)
            require(all(Path(name).is_relative_to(isolated.resolve()) for name in imported.values()), 'Actual imports escaped copied tree')
    actual = readback(reference.DIRECTORY); copied = readback(isolated/reference.DIRECTORY)
    require(actual == copied and counts == {'local':39,'isolated':39}, 'Actual readback/test scope differs')
    save(DIRECTORY/'saved-readback.json',actual); save(DIRECTORY/'isolated-saved-readback.json',copied)
    for name,sha in offline['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(name.replace('/compute/','/replay/')), 'Replay output changed')
    paths = [str(DIRECTORY/(tag+suffix)) for tag in ('local','isolated')
        for suffix in ('-command.json','-tests.xml','.log','-imports.json')]
    paths += [str(DIRECTORY/name) for name in ('saved-readback.json','isolated-saved-readback.json','copy-manifest.json')]
    paths += [__file__,*TESTS,*acceptance.SPECS]
    result = {'status':'ok_formal_ready','tests':counts,'saved_readback':actual,'copied_files':len(copy_hashes),
        'actual_imports':imported,'evidence_sha256':{str(Path(name).resolve().relative_to(project)):file_sha(name) for name in paths},
        'copy_manifest_sha256':file_sha(DIRECTORY/'copy-manifest.json'),
        'formal_source_research_acceptance':False,'formal_trading_acceptance':False,'new_trading_backtests':0}
    save(DIRECTORY/'checked.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('verify',))
    parser.parse_args(); print(json.dumps(verify(),ensure_ascii=False,allow_nan=False))
