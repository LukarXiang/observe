"""Recover numeric receipt types after a terminal CLI serialization failure; retain all artifacts."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

OLD = Path('data/staging/strategies-batch104/20261011-stage-validation')
DIRECTORY = Path('data/staging/strategies-batch104/20261011-stage-receipt-recovery')
COUNTS = ('short31_windows','relative_windows','condition_disagreements_preserved',
          'all_condition_disagreements_preserved','relative_disagreements','new_trading_backtests')


def numeric_readback(doc):
    require(set(doc) == {*COUNTS,'mean_max_abs_errors'},'Receipt readback fields differ')
    result = {}
    for key in COUNTS:
        value = doc[key]
        require(type(value) is int or type(value) is str and value.isdecimal(),'Counter is not an integer: '+key)
        result[key] = int(value); require(result[key] >= 0,'Negative counter: '+key)
    result['mean_max_abs_errors'] = {key:float(value) for key,value in doc['mean_max_abs_errors'].items()}
    require(set(result['mean_max_abs_errors']) == {'close','volume'} and
        all(math.isfinite(value) and value >= 0 for value in result['mean_max_abs_errors'].values()),'Invalid mean error')
    return result


def recover():
    reference.validate(); require(not DIRECTORY.exists(),'Immutable receipt recovery exists')
    previous = read(OLD/'checked.json')
    require(previous['status'] == 'ok_stage_only' and previous['tests'] == {'local':24,'isolated':24} and
        not previous['formal_source_research_acceptance'] and not previous['formal_trading_acceptance'],'Prior saved gate differs')
    manifest = OLD/'copy-manifest.json'; require(file_sha(manifest) == previous['copy_manifest_sha256'],'Old copy manifest changed')
    hashes = read(manifest); require(len(hashes) == previous['copied_files'] == 176,'Copied scope differs')
    for name,sha in hashes.items():
        copied = (OLD/'isolated'/name).resolve()
        require(copied.is_relative_to((OLD/'isolated').resolve()) and file_sha(copied) == sha,'Copied file changed: '+name)
    evidence = {str(OLD/'checked.json'):file_sha(OLD/'checked.json'),str(manifest):file_sha(manifest)}
    for tag in ('local','isolated'):
        command = read(OLD/(tag+'-command.json'))
        require(command['returncode'] == 0 and file_sha(OLD/(tag+'.log')) == command['log_sha256'],'Actual tests failed')
        suites = list(ET.parse(OLD/(tag+'-tests.xml')).getroot().iter('testsuite'))
        require(sum(int(s.attrib['tests']) for s in suites) == 24 and
            all(int(s.attrib.get(k,0)) == 0 for s in suites for k in ('failures','errors','skipped')),'Test scope incomplete')
        for suffix in ('-command.json','-tests.xml','.log','-imports.json'):
            path = OLD/(tag+suffix); evidence[str(path)] = file_sha(path)
    imports = read(OLD/'isolated-imports.json')
    require(imports == previous['actual_imports'] and len(imports) == 3 and
        all(Path(path).is_relative_to((OLD/'isolated').resolve()) for path in imports.values()),'Actual imports escaped copies')
    first = numeric_readback(read(OLD/'saved-readback.json')); second = numeric_readback(read(OLD/'isolated-saved-readback.json'))
    require(first == second == numeric_readback(previous['saved_readback']),'Saved readbacks differ')
    summary = read(reference.DIRECTORY/'compute/summary.json')
    require((first['short31_windows'],first['relative_windows'],first['condition_disagreements_preserved'],
        first['all_condition_disagreements_preserved'],first['relative_disagreements'],first['new_trading_backtests']) ==
        (summary['short31_windows'],summary['relative_position_windows'],summary['short_condition_disagreements'],
         summary['short_all_condition_disagreements'],summary['relative_position_disagreements'],0),'Counters differ from source outputs')
    for name in ['saved-readback.json','isolated-saved-readback.json']:
        path = OLD/name; evidence[str(path)] = file_sha(path)
    for name in ['scripts/recover_strategy_batch104_stage_receipt.py','tests/unit/test_batch104_receipt_recovery.py']:
        evidence[name] = file_sha(name)
    DIRECTORY.mkdir(parents=True)
    save(DIRECTORY/'failure.json',{'status':'observed_CLI_transport_failure_after_saved_checks',
        'error':'TypeError: Object of type int64 is not JSON serializable','original_command_returncode':1,
        'saved_counter_type_before_recovery':'condition_disagreements_preserved stored as string35',
        'old_artifacts_sha256':{str(p):file_sha(p) for p in OLD.rglob('*') if p.is_file()},'source_strategy_modified':False})
    result = {**previous,'saved_readback':first,'status':'ok_stage_only','receipt_numeric_types_recovered':True,
        'recovery_changes_control_receipt_only':True,'old_checks_overall_CLI_returncode':1,
        'evidence_sha256':evidence,'new_trading_backtests':0}
    # Check the actual CLI transport before writing the new receipt.
    encoded = json.dumps(result,ensure_ascii=False,allow_nan=False)
    require(json.loads(encoded) == result,'Recovered receipt JSON differs')
    save(DIRECTORY/'checked.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('recover',))
    parser.parse_args(); print(json.dumps(recover(),ensure_ascii=False,allow_nan=False))
