"""Recover only the recorded lossless Parquet date-unit promotion; keep the failed verifier intact."""
import json
from pathlib import Path

import pandas as pd

from observe.runs import file_sha
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

ORIGINAL = Path('scripts/verify_strategy_batch105_stage.py')
FAILURE = Path('data/staging/strategies-batch105/20261011-stage-date-precision-failure/checked.json')
DIRECTORY = Path('data/staging/strategies-batch105/20261011-component-stage-recovery')


def assert_saved_frame(saved,expected):
    left,right = saved.copy(),expected.copy()
    if 'date' in left and 'date' in right and pd.api.types.is_datetime64_any_dtype(left.date.dtype) and pd.api.types.is_datetime64_any_dtype(right.date.dtype):
        for frame in (left,right):
            original = frame.date.copy(); converted = original.dt.as_unit('ns')
            require(converted.dt.as_unit(original.dt.unit).equals(original),'Date-unit conversion loses information')
            frame['date'] = converted
    pd.testing.assert_frame_equal(left,right,check_exact=True)


def recover():
    failure = read(FAILURE)
    require(file_sha(ORIGINAL) == failure['evidence_sha256'][str(ORIGINAL)],'Original failed verifier changed')
    source = ORIGINAL.read_text()
    replacements = {
        "Path('data/staging/strategies-batch105/20261011-component-stage-validation')":'Path('+repr(str(DIRECTORY))+')',
        'pd.testing.assert_frame_equal(fixed,expected_fixed)':'assert_saved_frame(fixed,expected_fixed)',
        'pd.testing.assert_frame_equal(age,expected_age)':'assert_saved_frame(age,expected_age)',
        "TESTS = ('tests/unit/test_batch105_research.py','tests/unit/test_batch105_reference.py')":
            "TESTS = ('tests/unit/test_batch105_research.py','tests/unit/test_batch105_reference.py','tests/unit/test_batch105_precision_recovery.py')",
        "counts == {'local':19,'isolated':19}":"counts == {'local':22,'isolated':22}",
        "files = set(prepared['implementation_sha256'])|{'scripts/verify_strategy_batch105_stage.py'}":
            "files = set(prepared['implementation_sha256'])|{'scripts/verify_strategy_batch105_stage.py','scripts/recover_strategy_batch105_stage.py','tests/unit/test_batch105_precision_recovery.py'}",
        "('scripts.research_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr')":
            "('scripts.research_strategy_batch105','scripts.reference_strategy_batch105','scripts.reference_strategy_batch105_age','observe.factors.expr','scripts.recover_strategy_batch105_stage')",
    }
    for before,after in replacements.items():
        require(source.count(before) == 1,'Recovery patch scope differs: '+before)
        source = source.replace(before,after,1)
    namespace = {'__name__':'scripts.batch105_recovered_verifier','__file__':str(ORIGINAL.resolve()),
        'assert_saved_frame':assert_saved_frame}
    exec(compile(source,'<batch105-verifier-explicit-date-unit-recovery>','exec'),namespace)
    result = namespace['verify']()
    checked = read(DIRECTORY/'checked.json')
    recovery = {'status':'ok','original_verifier_unchanged':True,'original_failure':str(FAILURE),
        'date_unit_conversion':'lossless nanosecond representation solely for exact readback equality',
        'other_dtypes_and_values_checked_exactly':True,'historical_outputs_modified':False,
        'tests':checked['tests'],'checked_sha256':file_sha(DIRECTORY/'checked.json'),
        'evidence_sha256':{str(FAILURE):file_sha(FAILURE),str(ORIGINAL):file_sha(ORIGINAL),
            'scripts/recover_strategy_batch105_stage.py':file_sha('scripts/recover_strategy_batch105_stage.py')}}
    from scripts.review_strategy_batch18 import save
    save(DIRECTORY/'recovery.json',recovery)
    return result


if __name__ == '__main__': print(json.dumps(recover(),ensure_ascii=False))
