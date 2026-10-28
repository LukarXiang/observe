"""Read back the actual formal catalog publication without rerunning frozen components."""
import argparse
import json
from pathlib import Path

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS
from scripts import accept_strategy_batch104 as acceptance
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch104/20261011-formal-readback')


def check():
    require(not DIRECTORY.exists(), 'Immutable final readback exists')
    receipt = read(acceptance.RECEIPT); prepared = read(acceptance.DIRECTORY/'prepared.json')
    require(receipt['status'] == 'ok' and receipt['formal_source_research_acceptance'] and
        not receipt['formal_trading_acceptance'] and not receipt['library_goal_complete'] and
        not receipt['original_strategy_complete'] and receipt['new_trading_backtests'] ==
        receipt['new_strategy_parent_runs'] == 0, 'Formal scope differs')
    for name,sha in receipt['evidence_sha256'].items():
        require(file_sha(name) == sha, 'Final bound evidence changed: '+name)
    latest = read('data/catalog/strategies/latest.json')
    progress_latest = read('data/catalog/strategies/progress/latest.json')
    progress = receipt['progress']; catalog = read(Path(latest['directory'])/'catalog.json')
    previous = read(Path(prepared['previous_catalog']['directory'])/'catalog.json')
    selected = acceptance.reviews()
    delta = acceptance.candidate_checks(previous,catalog,selected)
    require(all(REVIEWS[name] == rules for name,rules in selected.items()), 'Live catalog rules differ')
    require(latest['catalog_id'] == progress['catalog_id'] == prepared['candidate']['catalog_id'] and
        file_sha(Path(latest['directory'])/'catalog.json') == latest['sha256'] == progress['catalog_sha256'] ==
        prepared['catalog_bytes']['catalog.json'], 'Actual catalog identity differs')
    require(progress_latest['checkpoint'] == progress['checkpoint'] and
        read(Path(progress_latest['directory'])/'summary.json') == progress and
        file_sha(Path(progress_latest['directory'])/'summary.json') == progress_latest['summary_sha256'],
        'Actual progress navigation differs')
    require((progress['sources'],progress['manually_reviewed'],progress['locally_backtested_sources'],
        progress['locally_reproduced_sources'],progress['local_strategy_runs']) == (695,294,12,12,87), 'Counts differ')
    require(file_sha('src/observe/strategy_catalog.py') == prepared['proposed_code_sha256'], 'Published code differs')
    baseline = read(acceptance.DIRECTORY/'baseline.json')
    require(Store('data').published() == baseline['published'], 'Market/financial publication changed')
    command_root = Path('data/staging/strategies-batch104/20261011-formal-command-records')
    commands = {}
    for tag in ('verify','prepare','protect','publish'):
        command = read(command_root/(tag+'-command.json'))
        require(command['returncode'] == 0 and file_sha(command_root/(tag+'.log')) == command['log_sha256'],
            'Formal command did not reach success: '+tag)
        commands[tag] = {'returncode':command['returncode'],'receipt_sha256':file_sha(command_root/(tag+'-command.json'))}
    require(receipt['components']['tests'] == {'local':39,'isolated':39}, 'Formal test scope differs')
    result = {'status':'ok','formal_source_research_acceptance':True,'formal_trading_acceptance':False,
        'progress':progress,'catalog_delta':delta,'other_source_records_unchanged':693,
        'bound_files_rechecked':len(receipt['evidence_sha256']),'formal_commands':commands,
        'publication_unchanged':True,'receipt_sha256':file_sha(acceptance.RECEIPT),
        'readback_code_sha256':file_sha(__file__),'new_trading_backtests':0,'library_goal_complete':False}
    DIRECTORY.mkdir(parents=True); save(DIRECTORY/'checked.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('check',))
    parser.parse_args(); print(json.dumps(check(),ensure_ascii=False,allow_nan=False))
