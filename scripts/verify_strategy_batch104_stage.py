"""Verify saved raw operands and copied original-function tests; no formal acceptance."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-stage-validation')
TESTS = ('tests/unit/test_batch104_research.py', 'tests/unit/test_batch104_reference.py')


def readback(root):
    root = Path(root); ma = pd.read_parquet(root/'compute'/reference.OUTPUTS[1])
    relative = pd.read_parquet(root/'compute'/reference.OUTPUTS[2]); summary = read(root/'compute/summary.json')
    prices = pd.read_parquet(root/'prices.parquet')
    groups = {s: f.sort_values('date').reset_index(drop=True) for s,f in prices.groupby('instrument')}
    positions = {s: {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i,d in enumerate(f.date)} for s,f in groups.items()}
    require(set(groups) == set(summary['profiles']), 'Profile stock scope differs')
    for symbol,frame in groups.items():
        profile = summary['profiles'][symbol]
        expected = {'rows':len(frame),'first':pd.Timestamp(frame.date.iloc[0]).strftime('%Y-%m-%d'),
            'last':pd.Timestamp(frame.date.iloc[-1]).strftime('%Y-%m-%d'),
            'short31_windows':max(0,len(frame)-30),'relative15_windows':max(0,len(frame)-14),
            'relative30_windows':max(0,len(frame)-29),'nontrading_rows':int((~frame.is_trading).sum()),
            'native_missing_price_rows':int(frame[['close','high','low']].isna().any(axis=1).sum())}
        require(profile == expected,'Saved stock profile differs: '+symbol)
    errors = {'close': 0., 'volume': 0.}; disagreements = 0; all_disagreements = 0
    for row in ma.itertuples(index=False):
        frame = groups[row.instrument]; end = positions[row.instrument][row.window_last]
        window = frame.iloc[end-30:end+1]
        require(len(window) == 31 and pd.Timestamp(window.date.iloc[0]).strftime('%Y-%m-%d') == row.window_first, '31 row window differs')
        for field, sizes, expected, actual in [('close', (5,10,20,30), row.independent_close_means, row.close_means),
                ('volume', (5,10), row.independent_volume_means, row.volume_means)]:
            values = window[field].to_numpy(); direct = []
            for shift in (0,1):
                for size in sizes:
                    part = values[-size:] if shift == 0 else values[-size-1:-1]
                    valid = [float(x) for x in part if not pd.isna(x)]
                    direct.append(math.fsum(valid)/len(valid) if valid else math.nan)
            require(np.array_equal(np.asarray(expected), direct, equal_nan=True), 'Saved independent mean differs')
            for a,b in zip(actual, direct, strict=True):
                if pd.isna(a) and pd.isna(b): continue
                require(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10), 'Original arithmetic exceeds declared tolerance')
                errors[field] = max(errors[field], abs(a-b))
        def flags(price, volume):
            return [price[4] > price[5] and price[0] < price[1], price[2] > price[6] and price[3] > price[7],
                    volume[2] > volume[3] and volume[0]*1.2 > volume[1]]
        original = flags(row.close_means,row.volume_means); independent = flags(row.independent_close_means,row.independent_volume_means)
        count = sum(a != b for a,b in zip(original, independent, strict=True))
        require(list(row.original_conditions) == original and list(row.independent_conditions) == independent and
            row.condition_disagreements == count and row.original_all_conditions == all(original) and
            row.independent_all_conditions == all(independent), 'Saved conditions or boundary count differ')
        disagreements += count; all_disagreements += all(original) != all(independent)
        require(row.nontrading_rows == int((~window.is_trading).sum()) and not row.platform_adjustment_or_pause_semantics_proven,
                'Trading/pause claim differs')
    for row in relative.itertuples(index=False):
        frame = groups[row.instrument]; end = positions[row.instrument][row.window_last]
        window = frame.iloc[end-row.length+1:end+1]
        highs = [float(x) for x in window.high if not pd.isna(x)]
        lows = [float(x) for x in window.low if not pd.isna(x)]
        maximum = max(highs) if highs else math.nan; minimum = min(lows) if lows else math.nan
        close = window.close.iloc[-1]
        ratio = (close-minimum)/(maximum-minimum) if maximum != minimum else math.nan
        def equal(a,b): return a == b or pd.isna(a) and pd.isna(b)
        require(len(window) == row.length and pd.Timestamp(window.date.iloc[0]).strftime('%Y-%m-%d') == row.window_first and
            equal(row.maximum_high,maximum) and equal(row.minimum_low,minimum) and equal(row.close,close) and
            equal(row.independent_ratio,ratio), 'Saved relative operands differ')
        upper = .3 if row.length == 15 else .5
        require(row.original_upper_bound == upper and row.independent_selected == (0. <= ratio <= upper) and
            row.disagreement == (row.original_selected != row.independent_selected),
                'Relative condition differs')
        require(row.zero_range == (maximum == minimum) and row.nontrading_rows == int((~window.is_trading).sum()) and
            not row.platform_adjustment_or_pause_semantics_proven, 'Relative range/pause claim differs')
    require(len(ma) == summary['short31_windows'] and len(relative) == summary['relative_position_windows'] and
        disagreements == summary['short_condition_disagreements'] and all_disagreements == summary['short_all_condition_disagreements'] and
        int(relative.disagreement.sum()) == summary['relative_position_disagreements'], 'Summary scope differs')
    require(summary['short_year_windows'] == ma.window_last.str[:4].value_counts().sort_index().to_dict() and
        summary['relative_year_windows'] == {str(n): f.window_last.str[:4].value_counts().sort_index().to_dict() for n,f in relative.groupby('length')},
        'Saved year profiles differ')
    require(summary['new_trading_backtests'] == summary['complete_original_pool_signals'] == 0 and
        not summary['original_strategy_complete'] and not summary['historical_original_pool_proven'], 'False completion claim')
    require(summary['native_missing_prices_preserved'] and not relative.disagreement.any(),
            'Native missing rows or original relative comparison not verified')
    return {'short31_windows': len(ma), 'relative_windows': len(relative), 'mean_max_abs_errors': errors,
        'condition_disagreements_preserved': disagreements, 'all_condition_disagreements_preserved': all_disagreements,
        'relative_disagreements': int(relative.disagreement.sum()), 'new_trading_backtests': 0}


def verify():
    prepared = reference.validate(); offline = read(reference.DIRECTORY/'offline-verification.json')
    require(offline['status'] == 'ok' and offline['fresh_network_blocked_processes'] == 2 and not offline['differences'], 'Fresh replay evidence differs')
    require(not DIRECTORY.exists(), 'Immutable stage checks exist'); DIRECTORY.mkdir(parents=True)
    isolated = DIRECTORY/'isolated'; isolated.mkdir()
    files = set(prepared['implementation_sha256'])|{__file__}
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
modules={n:str(Path(sys.modules[n].__file__).resolve()) for n in ('scripts.research_strategy_batch104','scripts.reference_strategy_batch104','observe.strategy_catalog')}
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
    require(actual == copied and counts == {'local':24,'isolated':24}, 'Actual readback/test scope differs')
    save(DIRECTORY/'saved-readback.json',actual); save(DIRECTORY/'isolated-saved-readback.json',copied)
    for name,sha in offline['outputs_sha256'].items():
        require(file_sha(name) == sha == file_sha(name.replace('/compute/','/replay/')), 'Replay output changed')
    result = {'status':'ok_stage_only','tests':counts,'saved_readback':actual,'copied_files':len(copy_hashes),
        'actual_imports':imported,'copy_manifest_sha256':file_sha(DIRECTORY/'copy-manifest.json'),
        'formal_source_research_acceptance':False,'formal_trading_acceptance':False,'new_trading_backtests':0}
    save(DIRECTORY/'checked.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('verify',))
    parser.parse_args(); print(json.dumps(verify(),ensure_ascii=False))
