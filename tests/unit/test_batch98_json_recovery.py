"""Actual JSON round trip retains native Arrow ranking elements."""
import json

import pandas as pd

from scripts import recover_strategy_batch98_diagnostics as recovery
from scripts import research_strategy_batch98 as original


def test_arrow_rank_survives_nested_actual_json_round_trip():
    value = {'nested': [{'rank': pd.Series(['C18', 'C17', 'C16'], dtype='str').values}]}
    with recovery.patch.object(original, 'canonical', recovery.canonical):
        result = json.loads(json.dumps(recovery.canonical(value), allow_nan=False))
    assert result == {'nested': [{'rank': ['C18', 'C17', 'C16']}]}


def test_nonranking_json_rules_preserve_null_nan_and_order():
    value = {'empty': None, 'labels': ['A', 'A', 'B'], 'nan': float('nan')}
    with recovery.patch.object(original, 'canonical', recovery.canonical):
        assert recovery.canonical(value) == {'empty': None, 'labels': ['A', 'A', 'B'], 'nan': {'nonfinite': 'nan'}}
