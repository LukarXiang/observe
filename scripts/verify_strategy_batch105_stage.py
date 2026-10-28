"""Read back every saved component row and run copied source tests; no research/trading promotion."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch105 as reference
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read, protect
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-component-stage-validation')
TESTS = ('tests/unit/test_batch105_research.py','tests/unit/test_batch105_reference.py')


def readback():
    root = reference.DIRECTORY; summary = read(root/'compute/summary.json')
    prices = pd.read_parquet(root/'prices.parquet')
    native = pd.read_parquet(root/'compute/native-bias.parquet')
    require(not native.duplicated(['date','instrument']).any(),'Duplicate saved native output')
    require(len(native) == summary['native_output_rows'] and len(prices) == summary['price_rows'],'Saved output count differs')
    groups = {symbol:frame for symbol,frame in native.groupby('instrument',sort=True)}
    profiles = {item['instrument']:item for item in summary['native_profiles']}; count = 0; maximum = 0.
    for symbol,raw in prices.groupby('instrument',sort=True):
        raw = raw.sort_values('date').reset_index(drop=True)
        saved = groups.get(symbol,native.iloc[:0].copy()).reset_index(drop=True)
        expected = raw.iloc[20:].reset_index(drop=True)
        for column in ('date','instrument','close','is_trading'):
            require(saved[column].equals(expected[column]),'Saved operand differs: '+symbol+'/'+column)
        require(np.array_equal(saved.native_position,np.arange(20,len(raw))),'Native position differs')
        # All full windows are verified against actual projected inputs, not output-to-output comparisons.
        means = raw.close.rolling(20).mean().to_numpy()[20:]
        direct = reference.independent_means(raw.close.to_numpy())[20:]
        require(np.array_equal(saved.original_ma20.to_numpy(),means,equal_nan=True)
            and np.array_equal(saved.independent_ma20.to_numpy(),direct,equal_nan=True),'Saved full20 operands differ')
        for column,array in (('original_bias',means),('engine_bias',means),('independent_bias',direct)):
            require(np.array_equal(saved[column].to_numpy(),expected.close.to_numpy() > array),'Saved strict bias differs')
        flags = saved.original_bias.ne(saved.independent_bias)
        require(flags.equals(saved.boundary_difference),'Saved boundary differences differ')
        count += int(flags.sum())
        differences = np.abs(means-direct); finite = differences[np.isfinite(differences)]
        if len(finite): maximum = max(maximum,float(finite.max()))
        require(profiles[symbol] == {'instrument':symbol,'input_rows':len(raw),'output_rows':len(saved),
            'first':str(raw.date.iloc[0]),'last':str(raw.date.iloc[-1]),'nontrading_rows':int((~raw.is_trading).sum()),
            'missing_close_rows':int(raw.close.isna().sum())},'Saved native profile differs')
    require(count == summary['native_boundary_differences'] and len(profiles) == prices.instrument.nunique(),'Saved native summary differs')
    for item in summary['years']:
        year = item['year']; incoming = prices[pd.to_datetime(prices.date).dt.year == year]
        saved = native[pd.to_datetime(native.date).dt.year == year]
        require(item == {'year':year,'input_rows':len(incoming),'available_symbols':incoming.instrument.nunique(),
            'native_output_rows':len(saved),'native_boundary_differences':int(saved.boundary_difference.sum()),
            'complete_original_market_pool_proven':False},'Year scope differs')
    fixed = pd.read_parquet(root/'compute/fixed40-bias.parquet')
    expected_fixed,scope = reference.fixed40_bias(prices,pd.read_parquet(root/'calendar.parquet'))
    pd.testing.assert_frame_equal(fixed,expected_fixed)
    require(scope == summary['fixed40_scope'] and len(fixed) == summary['fixed40_rows']
        and int(fixed.boundary_difference.sum()) == summary['fixed40_boundary_differences'],'Fixed40 summary differs')
    age = pd.read_parquet(root/'compute/calendar-age.parquet')
    instruments = pd.read_parquet(root/'instruments.parquet'); calendar = pd.read_parquet(root/'calendar.parquet')
    days = pd.to_datetime(calendar.loc[calendar.is_open,'date']).to_numpy(dtype='datetime64[D]')
    listing = pd.to_datetime(instruments.list_date).to_numpy(dtype='datetime64[D]')
    outputs = []
    for group,frame in instruments.groupby('exchange',sort=True):
        selected = np.flatnonzero(instruments.exchange.eq(group).to_numpy())
        dates = listing[selected]; valid = ~np.isnat(dates)
        counts = (((days[:,None]-dates[None,:]).astype('timedelta64[D]').astype(np.int64) > 252) & valid).sum(axis=1)
        outputs.append(pd.DataFrame({'date':days.astype('datetime64[ns]'),'controlled_current_exchange':group,
            'source_compatibility_count':counts,'independent_count':counts}))
    expected_age = pd.concat(outputs,ignore_index=True).sort_values(['date','controlled_current_exchange']).reset_index(drop=True)
    pd.testing.assert_frame_equal(age,expected_age)
    require(len(age) == summary['age_output_rows'] and summary['age_count_differences'] == 0
        and summary['age_scope']['checked_security_calendar_cells'] == len(days)*len(listing),'Calendar-age scope differs')
    require(summary['native_engine_differences'] == summary['new_trading_backtests'] == summary['complete_original_pool_signals'] == 0
        and not summary['formal_source_research_acceptance'] and not summary['original_strategy_complete']
        and not summary['source40_request_protocol_reproduced'] and not summary['original_nhnl_core_reproduced'],
        'False fidelity or completion claim')
    return {'native_rows':len(native),'fixed40_rows':len(fixed),'age_count_cells':len(age),
        'age_security_calendar_cells':len(days)*len(listing),'native_boundary_differences':count,
        'fixed40_boundary_differences':int(fixed.boundary_difference.sum()),'native_MA20_max_abs_error':maximum,
        'engine_differences':0,'age_count_differences':0,'new_trading_backtests':0}


def verify():
    prepared = reference.validate(); proof = read(reference.DIRECTORY/'offline-verification.json')
    require(proof['status'] == 'ok' and proof['fresh_network_blocked_processes'] == 2
        and not proof['differences'] and len(proof['outputs_sha256']) == 5,'Fresh replay incomplete')
    for record in proof['commands']:
        require(record['returncode'] == 0 and file_sha(record['log']) == record['log_sha256'],'Worker execution differs')
    for name,sha in proof['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(name.replace('/compute/','/replay/')),'Replay bytes differ')
    require(not DIRECTORY.exists(),'Immutable validation directory exists'); DIRECTORY.mkdir(parents=True)
    isolated = DIRECTORY/'isolated'; isolated.mkdir()
    files = set(prepared['implementation_sha256'])|{'scripts/verify_strategy_batch105_stage.py'}
    project = Path.cwd().resolve()
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).resolve().is_relative_to(project):
            rel = Path(name).resolve().relative_to(project).as_posix()
            if rel.startswith(('scripts/','src/observe/')): files.add(rel)
    files |= {str(reference.study.PREREAD/f'{number}.source') for number in (0,1)}
    files |= {'repo/量化策略源代码/'+name for name in reference.study.SOURCES}
    manifest = {}
    for name in sorted(files):
        destination = isolated/name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,destination)
        manifest[name] = file_sha(name); require(file_sha(destination) == manifest[name],'Copied bytes differ')
    save(DIRECTORY/'copy-manifest.json',manifest)
    worker = """import json,sys
from pathlib import Path
from scripts.recover_strategy_batch94_reference import network_blocked
import pytest
with network_blocked(): status=pytest.main(sys.argv[2:])
modules={name:str(Path(sys.modules[name].__file__).resolve()) for name in ('scripts.research_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr')}
Path(sys.argv[1]).write_text(json.dumps(modules))
raise SystemExit(status)
"""
    counts = {}; imported = {}
    for tag,cwd in (('local',project),('isolated',isolated.resolve())):
        output = (DIRECTORY/(tag+'-imports.json')).resolve(); xml = (DIRECTORY/(tag+'-tests.xml')).resolve()
        command = [sys.executable,'-c',worker,str(output),'-q',*TESTS,'--junitxml='+str(xml)]
        env = {**os.environ,'PYTHONPATH':str(cwd/'src')+os.pathsep+str(cwd)}
        log = DIRECTORY/(tag+'.log')
        with log.open('xb') as stream: result = subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
        save(DIRECTORY/(tag+'-command.json'),{'command':command,'returncode':result.returncode,'log_sha256':file_sha(log)})
        require(result.returncode == 0,'Copied source tests failed: '+tag)
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        require(all(int(s.attrib.get(key,0)) == 0 for s in suites for key in ('failures','errors','skipped')),'Incomplete source tests')
        counts[tag] = sum(int(s.attrib['tests']) for s in suites)
        imported[tag] = read(output)
        require(all(Path(name).is_relative_to(cwd) for name in imported[tag].values()),'Actual import escaped expected tree')
    require(counts == {'local':19,'isolated':19},'Test count differs')
    saved = readback(); protection = protect('data',reference.DIRECTORY)
    require(store_state_unchanged(prepared),'Publication or snapshot changed')
    bound = [str(reference.DIRECTORY/'prepared.json'),str(reference.DIRECTORY/'baseline.json'),
        str(reference.DIRECTORY/'offline-verification.json'),str(reference.DIRECTORY/'protection.json'),
        str(DIRECTORY/'copy-manifest.json'),'scripts/verify_strategy_batch105_stage.py',*TESTS,
        *proof['outputs_sha256']]
    result = {'status':'ok_component_stage','tests':counts,'actual_imports':imported,'copied_files':len(manifest),
        'saved_readback':saved,'protection':protection,'evidence_sha256':{name:file_sha(name) for name in bound},
        'formal_source_research_acceptance':False,'formal_trading_acceptance':False,'new_trading_backtests':0}
    save(DIRECTORY/'checked.json',result)
    return {key:value for key,value in result.items() if key not in ('evidence_sha256','actual_imports')}


def store_state_unchanged(prepared):
    from observe.data.store import Store
    store = Store('data'); actual = store.state(reference.SNAPSHOT)
    return actual['tables'] == store.published()['tables'] and all(actual['tables'][table] == parts
        for table,parts in prepared['snapshot_tables'].items())


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(),ensure_ascii=False))
