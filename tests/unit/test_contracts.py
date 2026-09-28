from datetime import date

import pandas as pd

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.ledger.rules import Rule, RuleSet


def test_adjustment_coverage_controls_each_date_and_unknown_gap():
    bars = pd.DataFrame({'date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                        'instrument': ['A'] * 3, 'close': [10.0] * 3})
    factors = pd.DataFrame({'instrument': ['A'], 'ex_date': [date(2024, 1, 1)], 'back_factor': [2.0]})
    coverage = pd.DataFrame([{'instrument': 'A', 'status': 'complete', 'verified_from': date(2024, 1, 1),
                              'verified_through': date(2024, 1, 2), 'has_start_basis': True, 'has_gap': False}])
    out = with_adjusted(bars, factors, coverage)
    assert out.loc[out.date == date(2024, 1, 2), 'close_adj'].iloc[0] == 20.0
    assert pd.isna(out.loc[out.date == date(2024, 1, 3), 'close_adj']).all()
    gap = coverage.assign(has_gap=True)
    assert pd.isna(with_adjusted(bars, factors, gap).close_adj).all()


def test_rule_fingerprint_excludes_runtime_usage_state():
    rules = RuleSet([Rule(start=date(2020, 1, 1), verified=False)])
    before = rules.config_fingerprint(); rules.fee_rule(date(2021, 1, 1))
    assert rules.config_fingerprint() == before
    assert before != RuleSet([Rule(start=date(2020, 1, 1), commission_rate=0.001)]).config_fingerprint()


def test_audit_commit_records_zero_problem_result(tmp_path):
    store = Store(tmp_path)
    aid = store.commit_audit('batch-1', pd.DataFrame(columns=['level', 'rule']), 'rule-hash', {'days': 0}, scope='incremental')
    meta = (tmp_path / 'audits' / f'{aid}.json').read_text(encoding='utf-8')
    assert '"status": "passed"' in meta and (tmp_path / 'audits' / f'{aid}.issues.csv').exists()
