"""Publish two fully reviewed source records; no missing ETF trades are admitted."""
import argparse
import ast
import json
from pathlib import Path
import shutil

import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.accept_strategy_batch100 import catalog_program
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch106 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch106/20261011-formal-source-acceptance-v1')
RECEIPT = Path('docs/handoff/2026-10-11-batch106-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百零六批月度槽位与价值平均-正式研究验收.md')
SPECS = ('docs/handoff/batch106-y2025-085-source-review.yaml', 'docs/handoff/batch106-y2024b-069-source-review.yaml')


def reviews():
    result = {}
    for k,name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == study.SOURCES[k] and doc['sha256'] == study.SOURCE_SHA[k] and
            doc['formal_source_research_acceptance'] is False and doc['original_strategy_complete'] is False,
            'Source review identity or stage scope changed')
        fields = ('schedule','signals','slots','orders','fees') if k == 0 else ('assets','entry','extrema','target','profit','cost')
        result[doc['source']] = {'review_status':'人工审查完成', 'status':'待数据',
            'scope':'月度MACD五槽位ETF轮换及年度再建仓' if k == 0 else '深证红利指数极值择时与159905价值平均、止盈重入',
            'rules':{f:doc[f] for f in fields}, 'gaps':doc['gaps'], 'next_step':doc['next'], 'implementations':[],
            'original_strategy_complete':False,
            'fidelity':'全文/原函数状态诊断与原生长历史操作数参考；基金事件/复权/盘中执行未齐，原交易0',
            'future_information_risks':['固定标的事后选择；当前供应商历史不证明平台复权/指标种子或当时可交易身份']}
    return result


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and len(selected) == 2, 'Catalog scope differs')
    previous = {r['path']:r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and set(previous) == {r['path'] for r in new}, 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated or missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
            {k:v for k,v in before.items() if k not in allowed} == {k:v for k,v in row.items() if k not in allowed}, 'Static identity changed')
        require(all(row[k] == v for k,v in allowed.items()), 'Literal source rules changed')
        require(row['original_strategy_complete'] is False and row['implementations'] == [] and row['status'] == '待数据', 'Fabricated trading acceptance')
    require(sum(r['review_status'] == '人工审查完成' for r in new) == 298, 'Reviewed count differs')
    return {'sources':695, 'reviewed':298, 'remaining':397, 'unrelated_sources_unchanged':693}


def components():
    path = study.DIRECTORY/'checked.json'; proof = read(path)
    require(proof['status'] == 'ok_component_stage' and proof['new_trading_backtests'] == 0 and
        proof['formal_source_research_acceptance'] is False and proof['saved_exact_readback'] is True and
        proof['offline_differences'] == 0 and proof['fresh_network_blocked_processes'] == 2, 'Component scope differs')
    for name,sha in proof['evidence_sha256'].items(): require(file_sha(name) == sha, 'Verified component bytes changed: '+name)
    require(proof['tests']['local_tests'] == proof['tests']['isolated_tests'] == 24, 'Actual test coverage differs')
    for label in ('local','isolated'):
        command = read(study.DIRECTORY/(label+'-tests.json'))
        origins = json.loads(command['stdout'].splitlines()[0])['actual_imports']; base = Path(command['cwd']).resolve()
        require(command['returncode'] == 0 and len(origins) == 3 and all(Path(n).is_relative_to(base) for n in origins), 'Actual import origins differ')
    require(proof['summary']['one_year_windows'] == 4776 and proof['summary']['two_horizon_windows'] == 4094 and
        proof['summary']['annual_scalar_reference_rows'] == 225 and proof['summary']['diagnostic_cases'] == 29,
        'Full native reference scope differs')
    for k in (0,1):
        p = study.DIRECTORY/'sources'/f'{k}.source'
        require(file_sha(p) == file_sha(Path('repo/量化策略源代码')/study.SOURCES[k]) == study.SOURCE_SHA[k], 'Complete original source changed')
    return {k:proof[k] for k in ('summary','tests','protection','output_files','fresh_network_blocked_processes','offline_differences','new_trading_backtests')}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    component = components(); study.runtime_binding(study.DIRECTORY); parent = read(study.PARENT)
    old_pointer = read('data/catalog/strategies/latest.json')
    require(old_pointer['catalog_id'] == parent['progress']['catalog_id'], 'Formal parent catalog changed')
    old = read(Path(old_pointer['directory'])/'catalog.json'); selected = reviews()
    before = Path('src/observe/strategy_catalog.py').read_bytes(); proposed = catalog_program(before,selected)
    DIRECTORY.mkdir(parents=True); (DIRECTORY/'strategy_catalog.before.py').write_bytes(before)
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(proposed); ast.parse(proposed)
    # Isolate the candidate catalog in a new data root; active Store and source records stay untouched.
    originals = dict(REVIEWS)
    try:
        REVIEWS.update(selected); digests = []
        for label in ('candidate','replay'):
            root = DIRECTORY/label; target = root/'catalog/strategies/implementation-evidence.json'; target.parent.mkdir(parents=True)
            shutil.copyfile(study.DIRECTORY/'implementation-evidence.before.json',target)
            generated = catalog_strategies(root,'repo/量化策略源代码'); folder = Path(generated['output'])
            candidate_checks(old,read(folder/'catalog.json'),selected)
            digests.append({n:file_sha(folder/n) for n in ('catalog.json','summary.json','catalog.parquet')})
            if label == 'candidate': candidate = generated
        require(digests[0] == digests[1], 'Catalog byte replay differs')
    finally:
        REVIEWS.clear(); REVIEWS.update(originals)
    paths = [*SPECS, str(study.DIRECTORY/'checked.json'), str(study.PARENT),
        'data/staging/strategies-batch106/20261011-fund-event-probe-v1/checked.json',
        'data/staging/strategies-batch106/20261011-calendar-coverage-v1/checked.json',
        'data/staging/strategies-batch106/20261011-development-fixture-failure/failure.json',
        'scripts/accept_strategy_batch106.py','tests/unit/test_batch106_acceptance.py']
    save(DIRECTORY/'prepared.json', {'component':component, 'previous_catalog':old_pointer, 'candidate':candidate,
        'catalog_bytes':digests[0], 'old_code_sha256':file_sha(DIRECTORY/'strategy_catalog.before.py'),
        'proposed_code_sha256':file_sha(DIRECTORY/'strategy_catalog.proposed.py'),
        'evidence_sha256':{p:file_sha(p) for p in paths}})
    return {'status':'candidate_verified_not_published','catalog_id':candidate['catalog_id']}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name,sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Prepared evidence changed')
    require(components() == doc['component'] and REPORT.exists(), 'Formal result/report missing or changed')
    require(file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
        all(REVIEWS.get(n) == v for n,v in selected.items()), 'Actual runtime review rules differ')
    tests = read(DIRECTORY/'tests.json')
    require(tests['returncode'] == 0 and '31 passed' in tests['stdout'] and tests['network_blocked'] is True, 'Formal tests missing')
    folder = Path(doc['candidate']['output']); old = read(Path(doc['previous_catalog']['directory'])/'catalog.json')
    candidate_checks(old,read(folder/'catalog.json'),selected)
    for n,sha in doc['catalog_bytes'].items(): require(file_sha(folder/n) == sha, 'Frozen candidate changed')
    with operation_lock('data',DATA_WRITER):
        require(read('data/catalog/strategies/latest.json') == doc['previous_catalog'], 'Active catalog pointer changed')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for n,sha in doc['catalog_bytes'].items():
            shutil.copyfile(folder/n,dest/n); require(file_sha(dest/n) == sha, 'Published bytes differ')
        _atomic_json(Path('data/catalog/strategies/latest.json'), {'catalog_id':dest.name,'directory':str(dest),'sha256':file_sha(dest/'catalog.json')})
        progress = archive('data','Batch106 two sources accepted;29 original diagnostics,4776 one-year/4094 two-horizon/225 annual scalar references;24 local+24 copied and31 formal tests;ETF trading blocked')
        require((progress['manually_reviewed'],progress['locally_backtested_sources'],progress['locally_reproduced_sources'],progress['local_strategy_runs']) == (298,12,12,87), 'Actual progress counts differ')
        paths = [*doc['evidence_sha256'], str(REPORT), str(DIRECTORY/'prepared.json'), str(DIRECTORY/'tests.json'),
            'src/observe/strategy_catalog.py',str(dest/'catalog.json'),str(Path(progress['output'])/'summary.json')]
        save(RECEIPT, {'status':'ok','formal_source_research_acceptance':True,'formal_trading_acceptance':False,
            'original_strategy_complete':False,'library_goal_complete':False,'progress':progress,'components':doc['component'],
            'source_sha256':list(study.SOURCE_SHA),'new_trading_backtests':0,'new_strategy_parent_runs':0,
            'evidence_sha256':{p:file_sha(p) for p in paths},
            'review_method':'Full original sources;literal diagnostics;native operand full exact readback and independent extrema;no independent agent review'})
    return {'status':'ok','checkpoint':progress['checkpoint'],'reviewed':298,'remaining':397}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','publish'))
    args = parser.parse_args(); print(json.dumps(globals()[args.action](),ensure_ascii=False,allow_nan=False))
