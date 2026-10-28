"""NaN quantile has a finite-score exception at exactly-perfect OLS fit."""
import math

import pytest

from scripts import diagnose_strategy_batch98 as followup
from scripts import research_strategy_batch98 as stage


def test_actual_definition_perfect_fit_exception():
    def original():
        return stage.definitions(stage.Path('repo/量化策略源代码')/stage.SOURCE_PATHS[1], 1)
    rows = followup.cases(original)['cases']
    assert rows[0]['score'] == pytest.approx(2.25/math.sqrt(6.83/4))
    assert rows[1]['score'] == pytest.approx(-1.3416407864998738)
    assert [r['exclude_below_minus_point4'] for r in rows] == [False, True]
