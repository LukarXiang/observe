"""Final immutable receipt/catalog readback, comparing progress fields without auxiliary output paths."""
import json
from pathlib import Path

from observe.runs import file_sha
from scripts.recover_strategy_batch105_formal import acceptance
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-formal-readback')


def check():
    require(not DIRECTORY.exists(),'Immutable final readback exists')
    receipt = read(acceptance.RECEIPT)
    require(receipt['status'] == 'ok' and receipt['formal_source_research_acceptance']
        and not receipt['formal_trading_acceptance'] and not receipt['original_strategy_complete']
        and not receipt['library_goal_complete'] and receipt['new_trading_backtests'] == 0,'Formal scope differs')
    for name,sha in receipt['evidence_sha256'].items(): require(file_sha(name) == sha,'Formal evidence changed: '+name)
    progress = receipt['progress']; archived = read('data/catalog/strategies/progress/'+progress['checkpoint']+'/summary.json')
    require({key:value for key,value in progress.items() if key != 'output'} == archived,'Formal checkpoint fields differ')
    require((archived['manually_reviewed'],archived['locally_backtested_sources'],archived['locally_reproduced_sources'],
        archived['local_strategy_runs'],archived['sources']) == (296,12,12,87,695),'Final counts differ')
    pointer = read('data/catalog/strategies/progress/latest.json'); latest = read('data/catalog/strategies/latest.json')
    require(pointer['checkpoint'] == progress['checkpoint'] and pointer['summary_sha256'] == file_sha(Path(pointer['directory'])/'summary.json')
        and latest['catalog_id'] == progress['catalog_id'],'Actual pointers differ')
    prepared = read(acceptance.DIRECTORY/'prepared.json')
    old = read(Path(prepared['previous_catalog']['directory'])/'catalog.json'); new = read(Path(latest['directory'])/'catalog.json')
    scope = acceptance.candidate_checks(old,new,acceptance.reviews())
    inputs = read(acceptance.reference.DIRECTORY/'prepared.json')
    for name,sha in inputs['actual_inputs_sha256'].items(): require(file_sha(name) == sha,'Actual source/snapshot/partition changed')
    recovery = read(acceptance.STAGE.parent/'recovery.json')
    require(recovery['status'] == 'ok' and recovery['checked_sha256'] == file_sha(acceptance.STAGE),'Recovered readback changed')
    for name,sha in recovery['evidence_sha256'].items(): require(file_sha(name) == sha,'Date recovery changed')
    formal = read(acceptance.CHECK/'checked.json')
    require(formal['tests'] == receipt['formal_tests'] == {'local':31,'isolated':31},'Formal test scope differs')
    manifest = read(acceptance.CHECK/'copy-manifest.json')
    for name,sha in manifest.items(): require(file_sha(acceptance.CHECK/'isolated'/name) == sha,'Formal archived copy changed')
    for tag in ('local','isolated'):
        command = read(acceptance.CHECK/(tag+'-command.json'))
        require(command['returncode'] == 0 and command['log_sha256'] == file_sha(acceptance.CHECK/(tag+'.log')),'Formal test terminal changed')
    bound = [str(acceptance.RECEIPT),str(acceptance.CHECK/'checked.json'),str(acceptance.STAGE),
        str(acceptance.STAGE.parent/'recovery.json'),str(acceptance.DIRECTORY/'prepared.json'),
        str(Path(latest['directory'])/'catalog.json'),'scripts/readback_strategy_batch105_formal.py']
    result = {'status':'ok_formal_readback','checkpoint':progress['checkpoint'],'scope':scope,
        'formal_binding_SHA_checked':len(receipt['evidence_sha256']),'actual_input_SHA_checked':len(inputs['actual_inputs_sha256']),
        'tests':formal['tests'],'archived_copies_verified':len(manifest),'auxiliary_output_field_excluded_explicitly':True,
        'new_trading_backtests':0,'library_goal_complete':False,'evidence_sha256':{name:file_sha(name) for name in bound}}
    DIRECTORY.mkdir(parents=True); save(DIRECTORY/'checked.json',result)
    return {key:value for key,value in result.items() if key != 'evidence_sha256'}


if __name__ == '__main__': print(json.dumps(check(),ensure_ascii=False))
