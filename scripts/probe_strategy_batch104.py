"""Inventory actual original-input gaps and reuse archived money-flow rows; no downloads."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys

import akshare
import pandas as pd
import pyarrow.parquet as pq

from observe.data.store import Store
from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts import research_strategy_batch104 as study
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-input-inventory')
REGISTRY = Path('data/staging/strategies-batch16/20261005-momentum-dependencies/probes/money_flow/result.json')
OUTPUTS = ('change-pct-reference.parquet', 'summary.json')


def prepare():
    require(not DIRECTORY.exists(), 'Immutable inventory exists')
    store = Store('data'); state = store.state(reference.SNAPSHOT)
    snapshot = Path('data/snapshots')/(reference.SNAPSHOT+'.json')
    require(file_sha(snapshot) == reference.SNAPSHOT_SHA and state['tables'] == store.published()['tables'], 'Snapshot differs')
    registry = read(REGISTRY); raw = Path(registry['raw_file'])
    require(registry['status'] == 'success' and registry['api'] == 'stock_individual_fund_flow' and
        file_sha(raw) == registry['raw_sha256'], 'Archived fund-flow registration differs')
    for response in registry['wire_responses']: require(file_sha(response['file']) == response['sha256'], 'Archived response differs')
    frame = pd.read_parquet(raw); require(len(frame) == registry['rows'] == 120, 'Archived fund-flow row count differs')
    DIRECTORY.mkdir(parents=True)
    (DIRECTORY/'stock_individual_fund_flow.py').write_text(inspect.getsource(akshare.stock_individual_fund_flow))
    columns = {}; counts = {}
    for table in ('calendar','instruments','bars_1d','bars_5m','adj_factors','adj_coverage','corp_actions'):
        parts = state['tables'].get(table,{})
        counts[table] = sum(row['rows'] for row in parts.values())
        columns[table] = pq.read_schema(Path('data')/next(iter(parts.values()))['file']).names if parts else []
    evidence = [str(snapshot),str(REGISTRY),str(raw),*(r['file'] for r in registry['wire_responses']),
        str(DIRECTORY/'stock_individual_fund_flow.py'),'scripts/probe_strategy_batch104.py']
    doc = {'status':'inventory_only','snapshot':reference.SNAPSHOT,'snapshot_sha256':reference.SNAPSHOT_SHA,
        'manifest_table_rows_not_fresh_full_table_audit':counts,'actual_partition_columns':columns,
        'moneyflow_registration':str(REGISTRY),'moneyflow_rows':120,'moneyflow_stock':'600000.SH',
        'moneyflow_original_platform_equivalence_proven':False,
        'moneyflow_original_full_pool_coverage_proven':False,
        'missing_original_fields_in_snapshot':{'historical_display_name':True,'get_money_flow_change_pct':True,
            'native_high_limit':True,'native_low_limit':True},
        'fields_present_but_equivalence_not_proven':['bars_1d.preclose vs JoinQuant pre_close','bars_1d.is_st vs historical display_name',
            'bars_5m.bar_end vs handle_data frequency and 09:30/11:25/14:30 last_price'],
        'source80_MACD_seed_history_and_pause_protocol_proven':False,
        'source80_preopen_day_open_availability_proven':False,
        'source78_default_stock_fees_and_slippage_version_proven':False,
        'source78_actual_ST_field_is_not_used_by_original_gain_filters':True,
        'akshare_version':akshare.__version__,'existing_interface_scope':'Single-stock recent Eastmoney fflow;no historical-range parameters',
        'absence_scope':'Named tables in the exact published snapshot;one archived money-flow registration;not whole-machine/provider absence',
        'new_requests':0,'new_responses':0,'new_publications':0,'new_trading_backtests':0,
        'evidence_sha256':{name:file_sha(name) for name in evidence}}
    save(DIRECTORY/'inventory.json',doc); return doc


def worker(mode):
    doc = read(DIRECTORY/'inventory.json')
    for name,sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha,'Inventory input changed: '+name)
    out = DIRECTORY/mode; out.mkdir(exist_ok=False)
    with network_blocked():
        registry = read(REGISTRY); raw = pd.read_parquet(registry['raw_file'])
        dates = pd.to_datetime(raw['日期']).dt.strftime('%Y-%m-%d')
        require(dates.is_monotonic_increasing and not dates.duplicated().any(),'Money-flow dates differ')
        frame = pd.DataFrame({'date':dates,'sec_code':'600000.XSHG','change_pct':raw['涨跌幅']})
        frame = frame.loc[frame.date.le('2026-09-29')].reset_index(drop=True)
        require(len(frame) == 119 and frame.change_pct.notna().all(),'Retained fund-flow scope differs')
        holder = {}; ns = study.literal(0,{'get_money_flow':lambda **k: holder['frame']})
        rows = []
        for end in range(1,len(frame)):
            window = frame.iloc[end-1:end+1]; holder['frame'] = window
            original = ns['stock_filter3'](['600000.XSHG'],frame.date.iloc[end])
            direct = float(window.change_pct.iloc[0])*float(window.change_pct.iloc[1]) < 0
            rows.append({'first':window.date.iloc[0],'last':window.date.iloc[1],
                'first_change_pct':float(window.change_pct.iloc[0]),'last_change_pct':float(window.change_pct.iloc[1]),
                'original_selected':bool(original),'independent_selected':direct,'disagreement':bool(original)!=direct,
                'JoinQuant_equivalence_proven':False,'original_full_pool':False})
        result = pd.DataFrame(rows); result.to_parquet(out/OUTPUTS[0],index=False)
        summary = {'retained_rows':len(frame),'future_rows_excluded':len(raw)-len(frame),'windows':len(result),
            'first':frame.date.iloc[0],'last':frame.date.iloc[-1],'selected':int(result.original_selected.sum()),
            'disagreements':int(result.disagreement.sum()),'complete_original_pool_signals':0,
            'new_trading_backtests':0,'platform_equivalence_proven':False,'network_blocked':True}
        save(out/OUTPUTS[1],summary)
    return summary


def run():
    require(not (DIRECTORY/'offline-verification.json').exists(),'Immutable replay exists'); commands = []
    for mode in ('compute','replay'):
        command = [sys.executable,'-m','scripts.probe_strategy_batch104',mode]
        with (DIRECTORY/(mode+'.log')).open('xb') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        commands.append({'command':command,'returncode':result.returncode,'log_sha256':file_sha(DIRECTORY/(mode+'.log'))})
        save(DIRECTORY/(mode+'-command.json'),commands[-1]); require(result.returncode == 0,'Money-flow reference worker failed')
    require(all(file_sha(DIRECTORY/'compute'/name) == file_sha(DIRECTORY/'replay'/name) for name in OUTPUTS),'Replay differs')
    result = {'status':'ok_component_only','commands':commands,'fresh_network_blocked_processes':2,
        'outputs_sha256':{str(DIRECTORY/'compute'/name):file_sha(DIRECTORY/'compute'/name) for name in OUTPUTS},
        'differences':0,'new_trading_backtests':0}
    save(DIRECTORY/'offline-verification.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','run','compute','replay'))
    action = parser.parse_args().action
    result = globals()[action]() if action in ('prepare','run') else worker(action)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
