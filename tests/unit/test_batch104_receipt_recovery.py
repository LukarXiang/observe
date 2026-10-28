"""Consumer-facing integer and strict-JSON contract for historical receipt recovery."""
import json

import pytest

from scripts.recover_strategy_batch104_stage_receipt import numeric_readback


def fixture():
    return {'short31_windows':46481,'relative_windows':93132,'condition_disagreements_preserved':'35',
        'all_condition_disagreements_preserved':0,'relative_disagreements':0,'new_trading_backtests':0,
        'mean_max_abs_errors':{'close':9.094947017729282e-13,'volume':0.}}


def test_legacy_count_string_becomes_JSON_number_without_changing_count():
    result = numeric_readback(fixture())
    decoded = json.loads(json.dumps(result,allow_nan=False))
    assert decoded['condition_disagreements_preserved'] == 35
    assert type(decoded['condition_disagreements_preserved']) is int
    assert decoded['new_trading_backtests'] == 0


@pytest.mark.parametrize('bad',[False,-1,'35.5',35.5,'NaN'])
def test_noninteger_or_negative_counts_cannot_be_silently_coerced(bad):
    doc = fixture(); doc['condition_disagreements_preserved'] = bad
    with pytest.raises(ValueError): numeric_readback(doc)
