from copy import deepcopy
from datetime import date

import pandas as pd
import pytest

from scripts.run_strategy_batch19 import annual_scope, publication_references, reference_ema


def test_reference_ema_seed_recursion():
    result = reference_ema([2., 4., 6., 10.], 3)
    assert result[2:] == [4., 7.]


@pytest.mark.parametrize('fault', ['other_table', 'other_year', 'missing_year', 'no_change'])
def test_publication_cannot_replace_unrelated_table_or_year(fault):
    old = {'tables': {'bars_1d': {'2005': {'file': 'old'}, '2021': {'file': 'full'}}, 'calendar': {'all': {'file': 'calendar'}}}}
    new = deepcopy(old); new['tables']['bars_1d']['2005'] = {'file': 'expanded'}
    publication_references(old, new, ['2005'])
    if fault == 'other_table': new['tables']['calendar']['all'] = {'file': 'changed'}
    elif fault == 'other_year': new['tables']['bars_1d']['2021'] = {'file': 'changed'}
    elif fault == 'missing_year': del new['tables']['bars_1d']['2021']
    else: new = deepcopy(old)
    with pytest.raises(ValueError): publication_references(old, new, ['2005'])


@pytest.mark.parametrize('listing,allowed', [(date(2013, 9, 18), True), (date(2005, 12, 30), False), (None, False)])
def test_scope_metadata_omits_only_proven_not_yet_listed_names(listing, allowed):
    merged = pd.DataFrame({'instrument': ['000651.SZ'], 'date': [date(2005, 12, 30)]})
    master = pd.DataFrame({'instrument': ['000333.SZ'], 'list_date': [listing]})
    scope = {'instruments': ['000651.SZ', '000333.SZ']}
    if allowed: assert annual_scope(scope, merged, master) == (['000651.SZ'], ['000333.SZ'])
    else:
        with pytest.raises(ValueError, match='Listed or unknown'): annual_scope(scope, merged, master)
