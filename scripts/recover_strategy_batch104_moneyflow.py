"""Preserve native string failure; separately evaluate explicit numeric change-pct operands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from observe.runs import file_sha
from scripts import probe_strategy_batch104 as failed
from scripts import research_strategy_batch104 as study
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-moneyflow-explicit-numeric-reference')
OUTPUTS = ('native-type-diagnostic.json','numeric-change-pct-reference.parquet','summary.json')


def prepare():
    require(not DIRECTORY.exists(),'Immutable recovery exists')
    command = read(failed.DIRECTORY/'compute-command.json'); log = failed.DIRECTORY/'compute.log'
    require(command['returncode'] == 1 and file_sha(log) == command['log_sha256'] and
        'Can only string multiply by an integer.' in log.read_text(),'Native failure antecedent differs')
    doc = read(failed.DIRECTORY/'inventory.json')
    for name,sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha,'Original inventory changed')
    DIRECTORY.mkdir(parents=True)
    paths = [str(failed.DIRECTORY/'inventory.json'),str(failed.DIRECTORY/'compute-command.json'),str(log),
        'scripts/recover_strategy_batch104_moneyflow.py','scripts/research_strategy_batch104.py']
    result = {'status':'explicit_numeric_reference_only','observed_native_failure':'Arrow string multiplication TypeError',
        'original_platform_dtype_proven':False,'old_files_sha256':{str(p):file_sha(p) for p in failed.DIRECTORY.rglob('*') if p.is_file()},
        'evidence_sha256':{name:file_sha(name) for name in paths},'new_trading_backtests':0,
        'conversion_is_not_approved_for_trading':True}
    save(DIRECTORY/'prepared.json',result); return result


def worker(mode):
    prepared = read(DIRECTORY/'prepared.json')
    for name,sha in {**prepared['old_files_sha256'],**prepared['evidence_sha256']}.items(): require(file_sha(name) == sha,'Recovery antecedent changed')
    out = DIRECTORY/mode; out.mkdir(exist_ok=False)
    with network_blocked():
        registry = read(failed.REGISTRY); require(file_sha(registry['raw_file']) == registry['raw_sha256'],'Raw money-flow changed')
        raw = pd.read_parquet(registry['raw_file'])
        dates = pd.to_datetime(raw['日期']).dt.strftime('%Y-%m-%d')
        native = pd.DataFrame({'date':dates,'sec_code':'600000.XSHG','change_pct':raw['涨跌幅']})
        native = native.loc[native.date.le('2026-09-29')].reset_index(drop=True)
        require(len(native) == 119 and native.change_pct.notna().all(),'Retained raw scope differs')
        holder = {'frame':native.iloc[:2]}; ns = study.literal(0,{'get_money_flow':lambda **k: holder['frame']})
        error = None
        try: ns['stock_filter3'](['600000.XSHG'],native.date.iloc[1])
        except TypeError as exc: error = {'type':'TypeError','message':str(exc)}
        require(error and error['message'] == 'Can only string multiply by an integer.','Native type behavior differs')
        frame = native.copy(); frame['change_pct'] = pd.to_numeric(native.change_pct,errors='raise').astype('float64')
        rows = []
        for end in range(1,len(frame)):
            window = frame.iloc[end-1:end+1]; holder['frame'] = window
            original = ns['stock_filter3'](['600000.XSHG'],frame.date.iloc[end])
            a,b = map(float,window.change_pct); direct = a*b < 0
            rows.append({'first':window.date.iloc[0],'last':window.date.iloc[1],'first_change_pct':a,'last_change_pct':b,
                'explicit_numeric_original_selected':bool(original),'independent_selected':direct,
                'disagreement':bool(original)!=direct,'platform_equivalence_proven':False})
        result = pd.DataFrame(rows); result.to_parquet(out/OUTPUTS[1],index=False)
        save(out/OUTPUTS[0],{'native_dtype':str(native.change_pct.dtype),'native_original_error':error,
            'explicit_numeric_dtype':'float64','platform_input_dtype_proven':False,'approved_for_trading':False})
        summary = {'retained_rows':len(frame),'future_rows_excluded':len(raw)-len(frame),'windows':len(result),
            'first':frame.date.iloc[0],'last':frame.date.iloc[-1],'selected':int(result.explicit_numeric_original_selected.sum()),
            'disagreements':int(result.disagreement.sum()),'native_original_type_error_preserved':True,
            'complete_original_pool_signals':0,'new_trading_backtests':0,'platform_equivalence_proven':False,'network_blocked':True}
        save(out/OUTPUTS[2],summary)
    return summary


def run():
    require(not (DIRECTORY/'offline-verification.json').exists(),'Immutable replay exists'); commands = []
    for mode in ('compute','replay'):
        command = [sys.executable,'-m','scripts.recover_strategy_batch104_moneyflow',mode]
        with (DIRECTORY/(mode+'.log')).open('xb') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        commands.append({'command':command,'returncode':result.returncode,'log_sha256':file_sha(DIRECTORY/(mode+'.log'))})
        save(DIRECTORY/(mode+'-command.json'),commands[-1]); require(result.returncode == 0,'Recovery worker failed')
    require(all(file_sha(DIRECTORY/'compute'/name) == file_sha(DIRECTORY/'replay'/name) for name in OUTPUTS),'Numeric replay differs')
    result = {'status':'ok_component_only','commands':commands,'fresh_network_blocked_processes':2,'differences':0,
        'outputs_sha256':{str(DIRECTORY/'compute'/name):file_sha(DIRECTORY/'compute'/name) for name in OUTPUTS},'new_trading_backtests':0}
    save(DIRECTORY/'offline-verification.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','run','compute','replay'))
    action = parser.parse_args().action
    result = globals()[action]() if action in ('prepare','run') else worker(action)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
