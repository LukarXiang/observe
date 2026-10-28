"""Accept two fully reviewed sources and arithmetic evidence; never trading acceptance."""
import argparse
import json
from pathlib import Path
import pprint
import shutil

import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch97 as stage
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch97/20261011-formal-source-acceptance')
PREPARED_SHA = '5c789f0997afc0d091e62c0af25442d09b363c9a831e22cd6c076a91bd557b9e'
SUMMARY_SHA = 'd5a741911b885e3e38cda743c0f7189e7ff664d71edf993f283c6d4ff1900818'
REPORT = Path('docs/tasks/2026-10-11-第九十七批七日趋势与拥挤率-正式研究验收.md')
RECEIPT = Path('docs/handoff/2026-10-11-batch97-final-review.json')
SPEC_FILES = ('docs/handoff/batch97-y2023-049-source-review.yaml', 'docs/handoff/batch97-y2023-081-source-review.yaml')
SOURCE_REVIEWS = {
    '2023年度精选策略/49.【7日趋势】交易策略.txt': {
        'review_status':'人工审查完成', 'scope':'120交易观察的七日趋势筛选评分、开盘补位与分钟高点退出',
        'status':'待数据','implementations':[], 'original_strategy_complete':False,
        'fidelity':'原规则全文与受控回调诊断；十股原价日线参考，原全池信号和交易未复现',
        'rules':{'benchmark':'000905.XSHG','schedule':['09:00 prior-day selection','09:30 stop then open filter then incremental buys'],
            'valuation':'2<turnover_ratio<=10 and 30<market_cap<=500; turnover descending',
            'universe':'get_all_securities(stock) without asof; exclude688/<365trade-days/ST/*/退/KeyError',
            'history':'120rows through previous_date; skip_paused=True,fq=pre',
            'price_volume':'last6 MA13>21 and21>55 each>=5; eight volume ratios strictly>1.7 count>=3; lastvolume>=1.5previous,lastclose/open>=1.05,lastclose/previousclose>=1',
            'score':'MA13/21/55 median-relative RMS sum(-2..-21) minus bullish-array relative gap sum(last5); ascending(score,code),score>.005',
            'open':'daily count1 at current_dt;1<open/pre_close<1.03;first3',
            'allocation':'held0 cash/3 all,held1 cash/2 first2,held2 cash first1,held>2 none;order_value including held targets',
            'stop':'minute high init_time..previous_date;loss<-7%;peak>20pct and drawdown/current>3pct;15pct<peak<=20pct and drawdown/current>5pct;inclusive days>=3 gain<5pct;days1 gain<0',
            'fees':{'commission':.0003,'sell_tax':.001,'minimum':5,'participation':.25,'fixed_slippage':0}},
        'future_information_risks':['无asof名称/全池、当日daily开盘及原前复权成交量/持仓价的历史语义未证'],
        'gaps':['缺可准入历史每日总市值/股本事件和原全池名称、上市身份；现有供应商样本admitted_rows=0',
            '原120条前复权价量和当前开盘可用性未证；固定十股原价参考不替代完整选股',
            '缺原持仓一分钟高点、当前持仓价/持有天数、增量报单和即时成交反馈；五分钟表不等价',
            '原实际规则与五日形态/13日线止损注释不同；不排持仓目标，默认参数保持原样'],
        'next_step':'补齐原交易输入及平台语义后唯一Book最长三成本/逐年/禁网复现；重要修正另确认，日线参考不计交易'},
    '2023年度精选策略/81.拥挤率指标-择时大盘顶底-Clone1-第二版.txt': {
        'review_status':'人工审查完成','scope':'结束日单名单、分块金额阈值含同额证券的拥挤率图表',
        'status':'非交易研究脚本','implementations':[],'original_strategy_complete':False,
        'fidelity':'原统计定义在现有日金额观察集复算；未证明原全市场统计，不添加交易规则',
        'rules':{'population':'get_all_securities(date=date_now) once for entire backward period',
            'default_days':1000,'chunk_days':100,'threshold':'sorted_money[floor(N*.05)] inclusive>= including all ties',
            'crowd':'selected money/total*100 rounded1;zero total invalid',
            'overlay':'000001.XSHG Shanghai close/100 rounded2,not HS300 despite variable name',
            'chart':'pyecharts/snapshot_selenium;clock-valued default and top-level chart not executed',
            'trading_rules':None},
        'future_information_risks':['结束日名单向前使用存在幸存者偏差；未证明逐日历史全池/金额口径'],
        'gaps':['原结束日名单及平台金额未证；现有1392日/5432证券是观察集参考',
            '0起始floor5%含下一排名及所有同额证券，并非恰好前5%；总额0和daycount/piece_count异常未修',
            '原图表依赖未安装或运行；无交易规则、收益、成本，不能计策略回测'],
        'next_step':'补原结束日证券范围/版本与money口径后核原统计；保留偏差，交易研究需明确新规则另立项'}
}


def diagnostic_checks(doc):
    rows = {r['case']:r for r in doc['cases']}
    require(doc['case_count']==len(rows)==31 and len(doc['cases'])==31, 'Diagnostic scope shrank')
    require(rows['crowd_n20_tieFalse_zeroFalse']['value'][0]['crowd_rate']==18.6 and
            rows['crowd_n20_tieTrue_zeroFalse']['value'][0]['crowd_rate']==100 and
            rows['crowd_n2_tieTrue_zeroTrue']['value'][0]['crowd_rate'] is None,'Crowd hand cases differ')
    require(rows['volume_eighth_comparison_only']['value'] is True and
            rows['current_open_100.0']['value']==rows['current_open_103.0']['value']==[] and
            rows['current_open_100.001']['value']==rows['current_open_102.999']['value']==['A'], 'Volume/open boundaries differ')
    for held, values in ((0,[200.,200.,200.]),(1,[300.,300.]),(2,[600.]),(3,[])):
        values_actual=rows[f'trade_{held}_holds_includes_held_target']['value']
        require([r['value'] for r in values_actual]==values,'Original allocation differs')
        if values: require(values_actual[0]['stock']=='H0','Held-target incremental intent lost')
    for name in ('stop_loss_exact','stop_day3_5pct_exact','stop_day1_equal_cost'):
        require(rows[name]['value']==[],'Strict exit boundary differs')
    for name in ('stop_loss_below','stop_gain20_exact','stop_gain20_above','stop_day3_below5pct','stop_day1_below_cost'):
        require(rows[name]['value']==[['A',0]],'Original exit branch differs')
    require(rows['inclusive_hold_days']['value']==2,'Inclusive holding days differ')
    errors={'crowd_zero_days':'IndexError','crowd_zero_piece':'ZeroDivisionError'}
    require({name:r['error']['type'] for name,r in rows.items() if r['error']}==errors,'Unexpected/missing original errors')
    return {'cases':31,'hand_boundaries':'match','recorded_intents_are_not_fills':True}


def components():
    require(file_sha(stage.DIRECTORY/'prepared.json')==PREPARED_SHA and
            file_sha(stage.DIRECTORY/'study/summary.json')==SUMMARY_SHA,'Frozen stage identity differs')
    prepared=read(stage.DIRECTORY/'prepared.json'); receipt=read(stage.DIRECTORY/'study-receipt.json')
    require(receipt['prepared_sha256']==PREPARED_SHA and set(receipt['outputs_sha256'])==set(stage.OUTPUTS), 'Study inventory differs')
    for name,digest in receipt['outputs_sha256'].items():
        require(file_sha(stage.DIRECTORY/'study'/name)==file_sha(stage.DIRECTORY/'replay'/name)==digest,'Actual worker output differs')
    offline=read(stage.DIRECTORY/'offline-verification.json'); command=read(stage.DIRECTORY/'fresh-worker/command.json')
    require(offline['status']=='match' and offline['differences']==0 and offline['files']==4 and
            offline['prepared_sha256']==PREPARED_SHA and command['returncode']==0 and
            command['command'][-3:]==['-m','scripts.research_strategy_batch97','worker'] and
            file_sha(stage.DIRECTORY/'fresh-worker/stdout.log')==command['log_sha256'],'Fresh worker evidence differs')
    stdout=json.loads((stage.DIRECTORY/'fresh-worker/stdout.log').read_text().splitlines()[-1])
    require(stdout['status']=='ok' and stdout['outputs_sha256']==receipt['outputs_sha256'],'Worker stdout is not actual study output')
    for name,digest in prepared['operands_sha256'].items(): require(file_sha(stage.DIRECTORY/name)==digest,'Operand changed')
    for name,digest in prepared['implementation_sha256'].items():
        require(file_sha(stage.DIRECTORY/'implementation'/name)==digest,'Frozen program changed')
        if name!='src/observe/strategy_catalog.py': require(file_sha(name)==digest,'Stage helper changed')
    state=stage.Store('data').state(stage.SNAPSHOT)
    require(prepared['tables']=={t:state['tables'][t] for t in ('bars_1d','calendar','instruments')} and
            prepared['partitions']==stage.verify_entries('data',prepared['tables']),'Input byte inventory differs')
    trend=pd.read_parquet(stage.DIRECTORY/'study/trend-components.parquet')
    crowd=pd.read_parquet(stage.DIRECTORY/'study/crowd-components.parquet')
    require(len(trend)==44873 and set(trend.instrument)==set(stage.SYMBOLS) and
            not trend.duplicated(['date','instrument']).any() and trend.window_rows.eq(120).all() and
            trend.stable_predicates_match.all(),'Trend scope or independent predicates differ')
    require(len(crowd)==1392 and not crowd.date.duplicated().any() and
            crowd.crowd_rate.eq(crowd.rounded).all() and int(crowd.rows.sum())==6848484,'Crowd scope or independent arithmetic differs')
    diag=diagnostic_checks(read(stage.DIRECTORY/'study/diagnostics.json'))
    return {'reference_windows':len(trend),'price_volume_prefilter_true':int(trend.price_volume_prefilter.sum()),
        'prefilter_and_score_true':int((trend.price_volume_prefilter & trend.score_threshold).sum()),
        'crowd_days':len(crowd),'amount_observations':int(crowd.rows.sum()),'diagnostics':diag,
        'offline':'match/0','new_trading_backtests':0}


def candidate_checks(old,now):
    require(len(old)==len(now)==695 and len({r['path'] for r in now})==695,'Source inventory differs')
    previous={r['path']:r for r in old}
    require({r['path'] for r in now}==set(previous),'Source identities differ')
    changes={r['path'] for r in now if r!=previous[r['path']]}
    require(changes==set(SOURCE_REVIEWS),'Unrelated or missing catalog changes')
    for row in now:
        if row['path'] in SOURCE_REVIEWS:
            before=previous[row['path']]
            require(before['review_status']!='人工审查完成' and
                {k:v for k,v in row.items() if k not in SOURCE_REVIEWS[row['path']]}==
                {k:v for k,v in before.items() if k not in SOURCE_REVIEWS[row['path']]},'Source static identity changed')
            require(all(row[k]==v for k,v in SOURCE_REVIEWS[row['path']].items()),'Rules differ')
    require(sum(r['review_status']=='人工审查完成' for r in now)==280,'Reviewed count differs')
    return {'sources':695,'changed':sorted(changes),'reviewed':280,'remaining':415}


def catalog_program(before):
    marker=b'REVIEWS = {\n'; require(before.count(marker)==1,'Catalog insertion marker differs')
    entries=pprint.pformat(SOURCE_REVIEWS,sort_dicts=False,width=100)[1:-1]
    return before.replace(marker,marker+(entries+',\n').encode(),1)


def prepare():
    require(not DIRECTORY.exists(),'Immutable formal directory exists')
    results=components(); stage.binding()
    isolated=read('data/staging/strategies-batch97/20261011-guard-validation/isolated.json')
    require(isolated['status']=='ok' and isolated['tests_passed']==19 and isolated['skips']==0,'Isolated tests incomplete')
    for name,digest in isolated['copied_sha256'].items():
        require(file_sha(Path(isolated['cwd'])/name)==digest,'Isolated copy changed')
    source=read(stage.PREREAD/'source-preread.json')
    for number,row in enumerate(source['sources']):
        require(file_sha(Path('repo/量化策略源代码')/row['path'])==file_sha(stage.DIRECTORY/'sources'/f'{number}.source')==stage.SOURCE_SHA[number], 'Original source changed')
    DIRECTORY.mkdir(parents=True,exist_ok=False)
    save(DIRECTORY/'component-checks.json',results)
    save(DIRECTORY/'source-reviews.json',SOURCE_REVIEWS)
    old=read(stage.PREREAD/'catalog-before.json')
    expected_catalog=read(stage.PREREAD/'source-preread.json')['catalog_before']
    require(read('data/catalog/strategies/latest.json')==expected_catalog,'Active catalog changed')
    REVIEWS.update(SOURCE_REVIEWS)
    manifests=[]
    for tag in ('candidate','candidate-replay'):
        root=DIRECTORY/tag
        evidence=root/'catalog/strategies/implementation-evidence.json'; evidence.parent.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(stage.PREREAD/'implementation-evidence.before.json',evidence)
        result=catalog_strategies(root,'repo/量化策略源代码')
        records=read(Path(result['output'])/'catalog.json'); candidate_checks(old,records)
        manifests.append({n:file_sha(Path(result['output'])/n) for n in ('catalog.json','summary.json','catalog.parquet')})
        if tag=='candidate': candidate=result
    require(manifests[0]==manifests[1],'Fresh catalog bytes differ')
    old_program=(stage.PREREAD/'strategy_catalog.before.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes()==old_program,'Catalog source changed before candidate check')
    with (DIRECTORY/'strategy_catalog.proposed.py').open('xb') as out: out.write(catalog_program(old_program))
    names=[*SPEC_FILES,'scripts/accept_strategy_batch97.py','tests/unit/test_batch97_acceptance.py',
        'data/staging/strategies-batch97/20261011-dependencies/inventory.json',
        'data/staging/strategies-batch97/20261011-dependencies/reference-files.json',
        'data/staging/strategies-batch97/20261011-guard-validation/isolated.json',
        'data/staging/strategies-batch97/20261011-components-v1/offline-verification.json',
        str(DIRECTORY/'source-reviews.json'),str(DIRECTORY/'component-checks.json')]
    save(DIRECTORY/'prepared.json',{'status':'candidate_verified_not_published','candidate':candidate,
        'catalog_bytes':manifests[0],'previous_catalog':expected_catalog,'components':results,
        'evidence_sha256':{n:file_sha(n) for n in names},
        'proposed_catalog_code_sha256':file_sha(DIRECTORY/'strategy_catalog.proposed.py'),
        'source_research_only':True,'original_strategy_complete':False})
    return {'status':'candidate_verified','catalog_id':candidate['catalog_id'],'reviewed_after_publication':280,
        'not_yet_published':True}


def publish():
    require(not RECEIPT.exists(),'Formal receipt already exists')
    doc=read(DIRECTORY/'prepared.json'); results=components()
    for name,digest in doc['evidence_sha256'].items(): require(file_sha(name)==digest,'Prepared evidence changed')
    require(results==doc['components'],'Verified component checks changed')
    require(Path('src/observe/strategy_catalog.py').read_bytes()==catalog_program((stage.PREREAD/'strategy_catalog.before.py').read_bytes()) and
        file_sha('src/observe/strategy_catalog.py')==doc['proposed_catalog_code_sha256'],'Only the two verified source review entries may extend catalog code')
    require(all(REVIEWS.get(n)==v for n,v in SOURCE_REVIEWS.items()),'Imported current reviews differ')
    require(REPORT.is_file(),'Human-readable final report missing')
    candidate=Path(doc['candidate']['output']); old=read(stage.PREREAD/'catalog-before.json')
    candidate_checks(old,read(candidate/'catalog.json'))
    for n,digest in doc['catalog_bytes'].items(): require(file_sha(candidate/n)==digest,'Candidate bytes changed')
    with operation_lock('data',DATA_WRITER):
        latest=Path('data/catalog/strategies/latest.json'); progress_latest=Path('data/catalog/strategies/progress/latest.json')
        require(read(latest)==doc['previous_catalog'],'Active navigation changed')
        previous_latest=latest.read_bytes(); previous_progress=progress_latest.read_bytes()
        protection=protect('data',stage.PREREAD)
        dest=Path('data/catalog/strategies')/doc['candidate']['catalog_id']; require(not dest.exists(),'Catalog destination already exists')
        dest.mkdir()
        for n,digest in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/n,dest/n); require(file_sha(dest/n)==digest,'Published frozen bytes differ')
        try:
            _atomic_json(latest,{'catalog_id':dest.name,'directory':str(dest),'sha256':doc['catalog_bytes']['catalog.json']})
            progress=archive('data','Batch97 original7day/crowd81 source research accepted;44873raw windows/1392money days/offline match0;no new trades')
            require((progress['manually_reviewed'],progress['locally_reproduced_sources'],progress['local_strategy_runs'])==(280,12,87),'Progress counts differ')
            require(read(latest)['catalog_id']==progress['catalog_id']==dest.name and
                    read(progress_latest)['checkpoint']==progress['checkpoint'],'Final navigation identity differs')
            paths={**doc['evidence_sha256'],str(DIRECTORY/'prepared.json'):file_sha(DIRECTORY/'prepared.json'),
                str(REPORT):file_sha(REPORT),'src/observe/strategy_catalog.py':file_sha('src/observe/strategy_catalog.py'),
                str(stage.PREREAD/'protection.json'):file_sha(stage.PREREAD/'protection.json'),
                str(dest/'catalog.json'):file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'):file_sha(Path(progress['output'])/'summary.json')}
            for name,digest in paths.items(): require(file_sha(name)==digest,'Final readback changed')
            save(RECEIPT,{'status':'ok','formal_source_research_acceptance':True,'formal_trading_acceptance':False,
                'original_strategy_complete':False,'library_goal_complete':False,'progress':progress,'components':results,
                'source_sha256':list(stage.SOURCE_SHA),'evidence_sha256':paths,'readback_bound_files':len(paths),
                'protection':protection,'new_trading_backtests':0,'new_strategy_parent_runs':0,
                'review_method':'Full-source manual review, independent scalar arithmetic, fresh-worker replay, unit and isolated tests; no independent agent review claimed'})
        except Exception:
            latest.write_bytes(previous_latest); progress_latest.write_bytes(previous_progress); raise
    return {'status':'ok','checkpoint':progress['checkpoint'],'reviewed':280,'remaining':415,'trades_added':0,'receipt':str(RECEIPT)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','publish'))
    args=parser.parse_args(); print(json.dumps({'prepare':prepare,'publish':publish}[args.action](),ensure_ascii=False))
