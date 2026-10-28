"""Retain real strict-boundary rows and compare native versus fixed40 operand protocols."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import reference_strategy_batch105 as reference
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch105/20261011-real-boundaries-and-protocol')


def diagnose():
    require(not DIRECTORY.exists(),'Immutable boundary evidence exists')
    root = reference.DIRECTORY/'compute'; summary = read(root/'summary.json')
    native_path = root/'native-bias.parquet'; fixed_path = root/'fixed40-bias.parquet'
    boundary = pd.read_parquet(native_path,filters=[('boundary_difference','==',True)])
    fixed = pd.read_parquet(fixed_path); fixed_boundary = fixed[fixed.boundary_difference].copy()
    require(len(boundary) == summary['native_boundary_differences'] == 373
        and len(fixed_boundary) == summary['fixed40_boundary_differences'] == 6,'Actual boundary scope differs')
    native = pd.read_parquet(native_path,filters=[('date','>=',fixed.date.min().date()),('date','<=',fixed.date.max().date())],
        columns=['date','instrument','close','original_ma20','original_bias','native_position'])
    native['date'] = pd.to_datetime(native.date)
    common = fixed.merge(native,on=['date','instrument'],how='left',validate='one_to_one',suffixes=('_fixed40','_native'))
    present = common.native_position.notna(); overlap = common.loc[present]
    require(np.array_equal(overlap.close_fixed40.to_numpy(),overlap.close_native.to_numpy(),equal_nan=True),
        'Common current close differs')
    flags = overlap.original_bias_fixed40.ne(overlap.original_bias_native)
    changed = overlap[flags].copy()
    DIRECTORY.mkdir(parents=True)
    boundary.to_parquet(DIRECTORY/'native-boundaries.parquet',index=False)
    fixed_boundary.to_parquet(DIRECTORY/'fixed40-boundaries.parquet',index=False)
    changed.to_parquet(DIRECTORY/'native-vs-fixed40-differences.parquet',index=False)
    result = {'status':'ok_diagnostic_only','native_boundary_rows':len(boundary),'fixed40_boundary_rows':len(fixed_boundary),
        'fixed40_cells':len(fixed),'common_output_cells':len(overlap),'missing_native_output_cells':int((~present).sum()),
        'native_vs_fixed40_original_comparison_differences':int(flags.sum()),
        'native_window_protocol_is_not_platform_40_calendar_query':True,
        'economic_rule_changes':False,'new_trading_backtests':0,'formal_source_research_acceptance':False,
        'no_signal_variant_approved_by_this_diagnostic':True,
        'boundary_year_counts':{str(year):len(frame) for year,frame in boundary.groupby(pd.to_datetime(boundary.date).dt.year)},
        'source_sha256':list(reference.study.SOURCE_SHA),
        'evidence_sha256':{str(path):file_sha(path) for path in (native_path,fixed_path,root/'summary.json',
            DIRECTORY/'native-boundaries.parquet',DIRECTORY/'fixed40-boundaries.parquet',
            DIRECTORY/'native-vs-fixed40-differences.parquet',Path('scripts/diagnose_strategy_batch105.py'))}}
    save(DIRECTORY/'checked.json',result)
    return result


if __name__ == '__main__': print(json.dumps(diagnose(),ensure_ascii=False))
