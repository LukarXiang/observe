"""Snapshot-bound longest native-row breadth and calendar-age arithmetic, never market-wide trades."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.factors.expr import compute as expression
from observe.runs import file_sha
from scripts import research_strategy_batch105 as study
from scripts.reference_strategy_batch105_age import function as age_function, verify_environment
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-longest-operands-v1')
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
OUTPUTS = ('native-bias.parquet','fixed40-bias.parquet','calendar-age.parquet','diagnostics.json','summary.json')


def independent_means(values):
    """Full20 observations, with stable independent summation; keep strict boundaries visible."""
    values = np.asarray(values,dtype=float)
    result = np.full(len(values),np.nan)
    if len(values) >= 20:
        windows = np.lib.stride_tricks.sliding_window_view(values,20)
        valid = np.isfinite(windows).all(axis=1)
        for position in np.flatnonzero(valid):
            result[position+19] = math.fsum(windows[position])/20
    return result


def native_bias(prices):
    require(not prices.duplicated(['date','instrument']).any(),'Duplicate native date/code')
    require(not np.isinf(prices.close.to_numpy()).any(),'Infinite native close')
    program = study.fragment({'df_bias'}); outputs = []; profiles = []
    for symbol,raw in prices.groupby('instrument',sort=True):
        raw = raw.sort_values('date').reset_index(drop=True)
        close = pd.DataFrame({symbol:raw.close.to_numpy()},index=pd.to_datetime(raw.date))
        namespace = {'df_close':close}; exec(program,namespace)
        original = namespace['df_bias'].iloc[:,0].to_numpy()
        engine = expression('close > ts_mean(close,20)',{'close':close},fields={'close'}).iloc[20:,0].fillna(0).astype(bool).to_numpy()
        means = close.iloc[:,0].rolling(20).mean().to_numpy()[20:]
        independent = independent_means(raw.close.to_numpy())[20:]
        direct_flags = raw.close.to_numpy()[20:] > independent
        require(np.array_equal(original,engine),'Factor expression differs from original AST')
        require(np.allclose(means,independent,rtol=1e-12,atol=1e-9,equal_nan=True),'MA20 independent arithmetic exceeds tolerance')
        item = raw.iloc[20:].copy()
        item['native_position'] = np.arange(20,len(raw),dtype=np.int32)
        item['original_ma20'] = means; item['independent_ma20'] = independent
        item['original_bias'] = original; item['engine_bias'] = engine
        item['independent_bias'] = direct_flags; item['boundary_difference'] = original != direct_flags
        outputs.append(item)
        profiles.append({'instrument':symbol,'input_rows':len(raw),'output_rows':len(item),
            'first':str(raw.date.iloc[0]),'last':str(raw.date.iloc[-1]),
            'nontrading_rows':int((~raw.is_trading).sum()),'missing_close_rows':int(raw.close.isna().sum())})
    return pd.concat(outputs,ignore_index=True),profiles


def fixed40_bias(prices,calendar):
    dates = pd.to_datetime(calendar.loc[calendar.is_open,'date'])
    dates = dates[dates <= pd.Timestamp('2022-09-02')].tail(40)
    require(len(dates) == 40 and dates.iloc[-1] == pd.Timestamp('2022-09-02'),'Original fixed40 date boundary missing')
    raw = prices.loc[pd.to_datetime(prices.date).isin(dates)]
    close = raw.pivot(index='date',columns='instrument',values='close')
    close.index = pd.to_datetime(close.index)
    require(close.index.equals(pd.DatetimeIndex(dates.to_numpy())),'Native fixed40 calendar rows differ')
    original = study.original_bias(close)
    engine = expression('close > ts_mean(close,20)',{'close':close},fields={'close'}).iloc[20:].fillna(0).astype(bool)
    require(original.equals(engine),'Fixed40 original/engine arithmetic differs')
    means = close.rolling(20).mean().iloc[20:]
    stable = pd.DataFrame({symbol:independent_means(close[symbol].to_numpy())[20:] for symbol in close},index=original.index)
    direct = close.iloc[20:] > stable
    require(np.allclose(means,stable,rtol=1e-12,atol=1e-9,equal_nan=True),'Fixed40 independent means exceed tolerance')
    index = pd.MultiIndex.from_product([original.index,original.columns],names=['date','instrument'])
    result = pd.DataFrame({'close':close.iloc[20:].to_numpy().ravel(),'original_ma20':means.to_numpy().ravel(),
        'independent_ma20':stable.to_numpy().ravel(),'original_bias':original.to_numpy().ravel(),
        'engine_bias':engine.to_numpy().ravel(),'independent_bias':direct.to_numpy().ravel()},index=index).reset_index()
    result['boundary_difference'] = result.original_bias.ne(result.independent_bias)
    return result,{'calendar_first':str(dates.iloc[0].date()),'calendar_last':str(dates.iloc[-1].date()),
        'calendar_rows':40,'output_days':len(original),'available_symbols':len(close.columns),
        'missing_close_cells':int(close.isna().sum().sum()),'complete_original_pool_proven':False,
        'industry_membership_proven':False,'source40_request_protocol_reproduced':False}


def calendar_age(instruments,calendar):
    verify_environment()
    require(not instruments.instrument.duplicated().any(),'Duplicate instrument record')
    require(instruments.exchange.notna().all(),'Missing controlled exchange label')
    basic = instruments[['instrument','list_date']].rename(columns={'instrument':'code'}).copy()
    basic['list_date'] = pd.to_datetime(basic.list_date)
    # Actual snapshot exchange labels are controlled function inputs, never SW industry membership.
    mapping = dict(zip(instruments.instrument,instruments.exchange,strict=True))
    periods = pd.to_datetime(calendar.loc[calendar.is_open,'date']).to_numpy(dtype='datetime64[ns]')
    require(pd.DatetimeIndex(periods).is_unique and pd.DatetimeIndex(periods).is_monotonic_increasing,'Calendar order differs')
    listing = basic.list_date.to_numpy(dtype='datetime64[D]')
    valid = ~np.isnat(listing); labels = np.asarray(instruments.exchange)
    outputs = []
    for start in range(0,len(periods),128):
        part = periods[start:start+128]
        original = age_function(basic,part)(mapping,'controlled_first','controlled_last',252)
        elapsed = part.astype('datetime64[D]')[:,None]-listing[None,:]
        eligible = valid[None,:] & (elapsed.astype('timedelta64[D]').astype(np.int64) > 252)
        direct = pd.DataFrame({group:eligible[:,labels == group].sum(axis=1) for group in sorted(set(labels))},index=part)
        require(original.index.equals(direct.index) and original.columns.equals(direct.columns)
            and np.array_equal(original.to_numpy(),direct.to_numpy()),'Source compatibility/independent age counts differ')
        for group in direct:
            outputs.append(pd.DataFrame({'date':part,'controlled_current_exchange':group,
                'source_compatibility_count':original[group].to_numpy(),'independent_count':direct[group].to_numpy()}))
    result = pd.concat(outputs,ignore_index=True).sort_values(['date','controlled_current_exchange']).reset_index(drop=True)
    return result,{'calendar_days':len(periods),'input_instruments':len(basic),
        'missing_listing_dates':int((~valid).sum()),'checked_security_calendar_cells':int(len(periods)*len(basic)),
        'groups':sorted(set(labels)),'source_industry_mapping_proven':False,'historical_listing_versions_proven':False,
        'threshold':252,'comparison':'strictly greater than natural days','chunk_days':128}


def prepare():
    require(not DIRECTORY.exists(),'Immutable operand directory exists')
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA
        and state['tables'] == store.published()['tables'],'Actual snapshot/publication changed')
    index = read('data/metadata/index.json'); records = read(index['indexes']['assets'])
    assets = {a['path']:a for a in records['assets']}
    inputs = {'data/snapshots/'+SNAPSHOT+'.json':SNAPSHOT_SHA}
    for table in ('bars_1d','calendar','instruments'):
        for entry in state['tables'][table].values():
            name = 'data/'+entry['file']; asset = assets[name]
            require(asset['store_fingerprint'] == entry['sha'] and asset['rows'] == entry['rows']
                and file_sha(name) == asset['file_sha256'],'Actual partition SHA differs: '+name)
            inputs[name] = asset['file_sha256']
    for number in (0,1):
        study.tree(number)
        for name in (str(study.PREREAD/f'{number}.source'),'repo/量化策略源代码/'+study.SOURCES[number]):
            inputs[name] = study.SOURCE_SHA[number]
    checkpoint('data',DIRECTORY,'Batch105 fresh protection/start:longest snapshot arithmetic;formal294/trading12/parents87 unchanged')
    columns = ['date','instrument','close','is_trading']
    frames = [pd.read_parquet('data/'+entry['file'],columns=columns) for year,entry in sorted(state['tables']['bars_1d'].items())]
    require(all(len(frame) == entry['rows'] for frame,(_,entry) in zip(frames,sorted(state['tables']['bars_1d'].items()),strict=True)),
        'Actual projected partition rows differ')
    prices = pd.concat(frames,ignore_index=True).sort_values(['instrument','date']).reset_index(drop=True)
    prices.to_parquet(DIRECTORY/'prices.parquet',index=False)
    for table in ('calendar','instruments'):
        entry = state['tables'][table]['all']; shutil.copyfile('data/'+entry['file'],DIRECTORY/(table+'.parquet'))
    files = {p.as_posix() for p in Path('src/observe').rglob('*.py')}
    files |= {'scripts/reference_strategy_batch105.py','scripts/reference_strategy_batch105_age.py',
        'scripts/research_strategy_batch105.py','scripts/probe_strategy_batch105.py',
        'tests/unit/test_batch105_research.py','tests/unit/test_batch105_reference.py','pyproject.toml','uv.lock'}
    project = Path.cwd().resolve()
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and Path(name).resolve().is_relative_to(project):
            rel = Path(name).resolve().relative_to(project).as_posix()
            if rel.startswith(('scripts/','src/observe/')): files.add(rel)
    for name in sorted(files):
        dest = DIRECTORY/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,dest)
    save(DIRECTORY/'prepared.json',{'status':'component_only','snapshot':SNAPSHOT,'snapshot_sha256':SNAPSHOT_SHA,
        'snapshot_tables':{table:state['tables'][table] for table in ('bars_1d','calendar','instruments')},
        'metadata_SHA_used_only_to_verify_actual_partitions':True,'actual_inputs_sha256':inputs,
        'operands_sha256':{name:file_sha(DIRECTORY/name) for name in ('prices.parquet','calendar.parquet','instruments.parquet')},
        'implementation_sha256':{name:file_sha(name) for name in sorted(files)},
        'price_rows':len(prices),'native_symbols':prices.instrument.nunique(),'source_raw_price_platform_equivalence_proven':False,
        'new_trading_backtests':0,'formal_source_research_acceptance':False})
    return {'status':'prepared','price_rows':len(prices),'native_symbols':prices.instrument.nunique(),'frozen_files':len(files)}


def validate():
    doc = read(DIRECTORY/'prepared.json')
    for name,sha in doc['actual_inputs_sha256'].items(): require(file_sha(name) == sha,'Input changed: '+name)
    for name,sha in doc['operands_sha256'].items(): require(file_sha(DIRECTORY/name) == sha,'Projected bytes changed')
    for name,sha in doc['implementation_sha256'].items():
        require(file_sha(name) == sha == file_sha(DIRECTORY/'implementation'/name),'Implementation changed: '+name)
    return doc


def worker(mode):
    doc = validate(); require(mode in ('compute','replay'),'Unknown worker mode')
    dest = DIRECTORY/mode; dest.mkdir(exist_ok=False)
    with network_blocked():
        prices = pd.read_parquet(DIRECTORY/'prices.parquet'); calendar = pd.read_parquet(DIRECTORY/'calendar.parquet')
        native,profiles = native_bias(prices); fixed,scope = fixed40_bias(prices,calendar)
        age,age_scope = calendar_age(pd.read_parquet(DIRECTORY/'instruments.parquet'),calendar)
        diagnostics = study.diagnostics()
        native.to_parquet(dest/OUTPUTS[0],index=False); fixed.to_parquet(dest/OUTPUTS[1],index=False)
        age.to_parquet(dest/OUTPUTS[2],index=False); save(dest/OUTPUTS[3],diagnostics)
        years = []
        for year,group in prices.groupby(pd.to_datetime(prices.date).dt.year):
            subset = native[pd.to_datetime(native.date).dt.year == year]
            years.append({'year':int(year),'input_rows':len(group),'available_symbols':group.instrument.nunique(),
                'native_output_rows':len(subset),'native_boundary_differences':int(subset.boundary_difference.sum()),
                'complete_original_market_pool_proven':False})
        summary = {'status':'ok_component_only','price_rows':len(prices),'native_symbols':len(profiles),
            'native_output_rows':len(native),'native_engine_differences':int(native.original_bias.ne(native.engine_bias).sum()),
            'native_boundary_differences':int(native.boundary_difference.sum()),'native_profiles':profiles,
            'first':str(prices.date.min()),'last':str(prices.date.max()),'years':years,'fixed40_scope':scope,
            'fixed40_rows':len(fixed),'fixed40_boundary_differences':int(fixed.boundary_difference.sum()),
            'age_scope':age_scope,'age_output_rows':len(age),'age_count_differences':int(age.source_compatibility_count.ne(age.independent_count).sum()),
            'original_diagnostic_cases':diagnostics['case_count'],'new_trading_backtests':0,
            'formal_source_research_acceptance':False,'original_strategy_complete':False,'network_blocked':True,
            'source40_request_protocol_reproduced':False,'original_nhnl_core_reproduced':False,
            'complete_original_pool_signals':0,'native_row_windows_are_not_common_calendar_windows':True,
            'native_nontrading_and_missing_rows_preserved':True,'source_raw_price_platform_equivalence_proven':False,
            'input_manifest_sha256':file_sha(DIRECTORY/'prepared.json'),'verified_input_rows':doc['price_rows']}
        save(dest/OUTPUTS[4],summary)
    validate()
    return {k:v for k,v in summary.items() if k not in ('native_profiles','years')}


def run():
    validate(); require(not (DIRECTORY/'offline-verification.json').exists(),'Immutable replay receipt exists')
    commands = []
    for mode in ('compute','replay'):
        command = [sys.executable,'-m','scripts.reference_strategy_batch105',mode]
        log = DIRECTORY/(mode+'.log')
        with log.open('xb') as stream: result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
        record = {'command':command,'returncode':result.returncode,'log':str(log),'log_sha256':file_sha(log)}
        commands.append(record); save(DIRECTORY/(mode+'-command.json'),record)
        require(result.returncode == 0,'Fresh worker failed: '+mode)
    differences = [name for name in OUTPUTS if file_sha(DIRECTORY/'compute'/name) != file_sha(DIRECTORY/'replay'/name)]
    save(DIRECTORY/'offline-verification.json',{'status':'ok' if not differences else 'failed',
        'fresh_network_blocked_processes':2,'commands':commands,'differences':differences,
        'outputs_sha256':{str(DIRECTORY/'compute'/name):file_sha(DIRECTORY/'compute'/name) for name in OUTPUTS},
        'new_trading_backtests':0,'formal_source_research_acceptance':False})
    require(not differences,'Actual saved outputs differ')
    return {'status':'ok_component_only','outputs':len(OUTPUTS),'differences':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','compute','replay','run'))
    action = parser.parse_args().action
    print(json.dumps(prepare() if action == 'prepare' else run() if action == 'run' else worker(action),ensure_ascii=False))
