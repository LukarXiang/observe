"""Approved pair66 short-island signals; a preflight, without orders or NAV."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import file_sha
from observe.strategy_rotation import generate
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.supplement_pair66_history import CONFIG, CONFIG_SHA, SOURCE_SHA, SNAPSHOT
from scripts.verify_strategy_batch3 import require

DIRECTORY=Path('data/staging/strategies-history-supplement/20261011-pair66-short-islands-preflight')
PREPARED=Path('data/staging/strategies-history-supplement/20261009-pair66-pre2020/prepared-windows.json')
PREPARED_SHA='1e9351a94cc5964691919e88019dc4be6c13419c81ab2793de37df04cf45b093'
EXPECTED=(('2011-03-16','2011-04-08',16),('2011-08-02','2011-09-15',32),
    ('2015-05-28','2015-07-08',29),('2015-12-09','2016-02-19',47),
    ('2017-03-03','2017-09-15',136),('2019-07-08','2019-09-30',60))


def main():
    require(not DIRECTORY.exists(),'Preflight is immutable; choose a new directory for new inputs')
    require(file_sha(CONFIG)==CONFIG_SHA and file_sha(PREPARED)==PREPARED_SHA,'Approved inputs changed')
    policy=json.loads(Path('configs/research_metadata.json').read_bytes())
    require(next(d for d in policy['decisions'] if d['id']=='D-rotation66')['status']=='approved','Variant approval missing')
    config=yaml.safe_load(CONFIG.read_text());require(file_sha(config['source_path'])==SOURCE_SHA,'Source changed')
    manifest=json.loads(PREPARED.read_bytes());windows=[r for r in manifest['all_pre2020_candidates'] if 0<r['post_warmup_sessions']<200]
    require(tuple((r['start'],r['end'],r['post_warmup_sessions']) for r in windows)==EXPECTED,'Short window inventory differs')
    pair=config['execution_instruments'];store=Store('data');state=store.state(SNAPSHOT)
    with network_blocked():
        filters=[('instrument','in',pair)]
        view=with_adjusted(store.load_state(state,'bars_1d',filters=filters),store.load_state(state,'adj_factors',filters=filters),
            store.load_state(state,'adj_coverage',filters=filters))
        calendar=sessions(store.load_state(state,'calendar'));view['date']=pd.to_datetime(view.date).dt.date
        adjusted=view.pivot(index='date',columns='instrument',values='close_adj').reindex(calendar)
        bases=view.pivot(index='date',columns='instrument',values='back_factor').reindex(calendar)
        trading=view.pivot(index='date',columns='instrument',values='is_trading').reindex(calendar)
        usable=view.pivot(index='date',columns='instrument',values='adjustment_status').reindex(calendar)
        results=[];all_factors=[]
        for window in windows:
            days=[d for d in calendar if window['start']<=str(d)<=window['end']]
            require(len(days)==window['post_warmup_sessions'],'Calendar count differs')
            factors,_,targets,_=generate({'close_adj':adjusted},bases,days,pair)
            held=0;differences=0;maximum=0.;actions=[]
            for day,row in zip(days,factors,strict=True):
                k=calendar.index(day);history=adjusted[pair].iloc[k-59:k+1]
                require(len(history)==60 and trading[pair].iloc[k-59:k+1].eq(True).all().all() and
                    usable[pair].iloc[k-59:k+1].eq('usable').all().all(),'Strict paired input window missing')
                first=[float(v)/float(bases.at[day,pair[0]]) for v in history[pair[0]]]
                second=[float(v)/float(bases.at[day,pair[1]]) for v in history[pair[1]]]
                spread=[a-b for a,b in zip(first,second,strict=True)];avg=math.fsum(spread)/60
                std=math.sqrt(math.fsum((v-avg)**2 for v in spread)/59)
                zraw=(spread[-1]-avg)/std;z=float(round(np.float64(zraw),4));previous=held;action=0
                if held!=1 and z<=-2:held=action=1
                elif held!=2 and z>=2:held=action=2
                maximum=max(maximum,abs(row['z_raw']-zraw))
                require(math.isclose(row['mean'],avg,rel_tol=1e-10,abs_tol=1e-9) and math.isclose(row['std'],std,rel_tol=1e-10,abs_tol=1e-9),'Independent arithmetic differs')
                differences+=int((row['z_rounded'],row['previous_state'],row['state'],row['action'])!=(z,previous,held,action))
                actual=[r for r in targets if r['decision_date']==day]
                require(len(actual)==2 and all(r['buy']==(action==n) and r['sell']==(action!=0 and action!=n)
                    for n,r in enumerate(actual,1)),'Independent target intent differs')
                if action:
                    next_day=days[days.index(day)+1] if day!=days[-1] else None
                    actions.append({'decision':str(day),'action':action,'next_execution_in_window':str(next_day) if next_day else None})
                all_factors.append({**row,'window_start_date':window['start']})
            require(differences==0,'Independent rounded signal/state differs')
            results.append({**window,'decision_days':len(days),'signal_actions':actions,
                'executable_signal_actions':sum(r['next_execution_in_window'] is not None for r in actions),
                'independent_differences':differences,'maximum_raw_z_difference':maximum,
                'execution_eligibility_checked':False,'zero_signal_window':not actions})
    DIRECTORY.mkdir(parents=True)
    frame=pd.DataFrame(all_factors)
    with (DIRECTORY/'factors.parquet').open('xb') as stream:frame.to_parquet(stream,index=False)
    save(DIRECTORY/'preflight.json',{'status':'ok','scope':'approved_variant_signal_preflight_only','windows':results,
        'decision_days':len(frame),'parameters_changed':False,'snapshot':SNAPSHOT,'snapshot_sha256':file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')),
        'source_sha256':SOURCE_SHA,'prepared_sha256':PREPARED_SHA,'config_sha256':CONFIG_SHA,
        'code_sha256':{n:file_sha(n) for n in ('scripts/preflight_pair66_short_windows.py','src/observe/strategy_rotation.py','scripts/supplement_pair66_history.py')},
        'factors_sha256':file_sha(DIRECTORY/'factors.parquet'),'network_forbidden':True,
        'new_trade_backtests':0,'orders_executed':0,'funds_earned':None,'separate_empty_initial_signal_state':True,
        'limits':['No execution/Book/NAV/fee acceptance in this preflight; check actual next-open inputs before backtests',
            'Six independent short islands; no stitched NAV or claim of longer continuous history',
            'Zero-signal or only-last-day signal windows cannot count as trading backtest accomplishments',
            'Dynamic-anchor daily approximation already approved, original intraday/platform equivalence not proven',
            'Snapshot file bytes bound; standard partition identity inherited, not newly full-byte-hashed here']})
    print(json.dumps({'status':'ok','decision_days':len(frame),'signals':[len(r['signal_actions']) for r in results],
        'executable_signals':[r['executable_signal_actions'] for r in results],'new_trade_backtests':0}))


if __name__=='__main__':main()
