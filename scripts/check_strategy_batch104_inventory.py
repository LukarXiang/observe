"""Check every relevant partition schema, without inferring platform field equivalence."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from observe.data.store import Store
from observe.runs import file_sha
from scripts import reference_strategy_batch104 as reference
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require


def check(output):
    state = Store('data').state(reference.SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(reference.SNAPSHOT+'.json')) == reference.SNAPSHOT_SHA,'Snapshot changed')
    rows = []; fields = set()
    for table in ('instruments','bars_1d','bars_5m','adj_factors','adj_coverage','corp_actions'):
        for key,entry in state['tables'][table].items():
            path = Path('data')/entry['file']; columns = pq.read_schema(path).names
            fields.update(columns)
            rows.append({'table':table,'partition':key,'file':str(path),'columns':columns,
                         'logical_sha256':entry['sha'],'rows_in_manifest':entry['rows']})
    require(not {'display_name','change_pct','high_limit','low_limit'}&fields,'An original-named field is present;recheck gap claim')
    result = {'status':'ok_schema_inventory_only','snapshot':reference.SNAPSHOT,'snapshot_sha256':reference.SNAPSHOT_SHA,
        'partitions_checked':len(rows),'partition_schemas':rows,
        'absence_scope':'All partitions of six named tables in exact snapshot;not other raw/repo/provider sources',
        'original_named_fields_absent':['display_name','change_pct','high_limit','low_limit'],
        'aliases_or_derived_values_are_not_proven_equivalent':True,'no_fresh_full_partition_byte_hash':True,
        'new_trading_backtests':0}
    output = Path(output); output.mkdir(parents=True,exist_ok=False); save(output/'checked.json',result)
    return {'status':result['status'],'partitions_checked':len(rows),'new_trading_backtests':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    print(json.dumps(check(parser.parse_args().output),ensure_ascii=False))
