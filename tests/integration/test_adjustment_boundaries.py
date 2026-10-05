"""复权视图边界与连续增量覆盖：无事件基准、显式估计模式、布尔解析、因子合法性、覆盖冲突；增量更新推进或不推进覆盖。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from observe.data.prices import normalize_coverage, with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.data.update import update_daily
from tests.unit.test_data import DAYS, FakeBS

X = '600001.SH'
DD = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
BARS = pd.DataFrame({'date': DD, 'instrument': X, 'open': [10.0, 10.0, 5.0], 'close': [10.0, 10.0, 5.0], 'preclose': [10.0, 10.0, 5.0]})


def cov(**kw):
    return pd.DataFrame([{'instrument': X, 'status': 'complete', 'verified_from': date(1990, 1, 1), 'verified_through': DD[-1], 'has_start_basis': True, 'has_gap': False, **kw}])


def consistent(out):
    """状态、数值与原因一致"""
    for r in out.itertuples():
        if r.adjustment_status == 'usable': assert np.isfinite(r.close_adj) and r.adjusted_unavailable_reason is None
        elif r.adjustment_status == 'unavailable': assert np.isnan(r.close_adj) and r.adjusted_unavailable_reason
        else: assert r.adjustment_status == 'estimated' and np.isfinite(r.close_adj) and r.adjusted_unavailable_reason


# 7. 无事件完整覆盖：因子 = 1，复权价 = 原始价，usable 与数值一致 ---------------------------------------------------
def test_confirmed_no_events_uses_basis_one():
    out = with_adjusted(BARS, pd.DataFrame(), cov(status = 'no_events', confirmed_no_events = True))
    assert (out.back_factor == 1.0).all() and (out.close_adj == out.close).all() and (out.adjustment_status == 'usable').all()
    consistent(out)


def test_complete_coverage_uses_basis_one_before_first_event():
    adj = pd.DataFrame({'instrument': [X], 'ex_date': [DD[2]], 'back_factor': [2.0]})
    out = with_adjusted(BARS, adj, cov())
    assert out.back_factor.tolist() == [1.0, 1.0, 2.0] and out.close_adj.tolist() == [10.0, 10.0, 10.0] and out.ret.tolist()[1:] == [0.0, 0.0]
    consistent(out)


def test_estimated_mode_is_explicit_and_marked():
    adj = pd.DataFrame({'instrument': [X], 'ex_date': [DD[0]], 'back_factor': [2.0]})
    partial = cov(status = 'partial', has_gap = True, has_start_basis = False)
    strict = with_adjusted(BARS, adj, partial)
    assert strict.close_adj.isna().all() and (strict.adjustment_status == 'unavailable').all()
    est = with_adjusted(BARS, adj, partial.assign(has_gap = False), allow_estimated = True)
    assert est.close_adj.tolist() == [20.0, 20.0, 10.0] and (est.adjustment_status == 'estimated').all() and (est.adjusted_unavailable_reason == 'outside_verified_coverage').all()
    consistent(strict); consistent(est)


def test_string_booleans_are_parsed_explicitly():
    assert normalize_coverage(cov(has_gap = 'False')).has_gap.tolist() == [False]
    assert normalize_coverage(cov(has_gap = 'True')).has_gap.tolist() == [True]
    assert with_adjusted(BARS, pd.DataFrame(), cov(has_gap = 'False', confirmed_no_events = 'true', status = 'no_events')).close_adj.notna().all()
    with pytest.raises(ValueError, match = 'boolean'): normalize_coverage(cov(has_gap = 'maybe'))


@pytest.mark.parametrize('factor', [0.0, -1.0, float('inf'), float('nan')])
def test_invalid_factor_makes_instrument_unavailable(factor):
    adj = pd.DataFrame({'instrument': [X, X], 'ex_date': [DD[0], DD[2]], 'back_factor': [1.0, factor]})
    out = with_adjusted(BARS, adj, cov())
    # 事件记录的因子缺失也不能跳过：否则会跨过除权日沿用旧因子
    assert out.close_adj.isna().all() and (out.adjusted_unavailable_reason == 'invalid_factor').all()
    consistent(out)


def test_no_event_claim_conflicting_with_events_is_reported():
    adj = pd.DataFrame({'instrument': [X], 'ex_date': [DD[1]], 'back_factor': [2.0]})
    out = with_adjusted(BARS, adj, cov(status = 'no_events', confirmed_no_events = True))
    assert out.close_adj.isna().all() and (out.adjusted_unavailable_reason == 'coverage_conflict').all()
    consistent(out)


# 8. 连续无事件增量推进覆盖；缺日或中途失败的增量不填平缺口 ------------------------------------------------------
def coverage_of(root, inst = '600519.SH'): return normalize_coverage(Store(root).load('adj_coverage')).set_index('instrument').loc[inst]


def test_continuous_increment_extends_verified_through(tmp_path):
    bs = FakeBS(DAYS)
    assert update_daily(tmp_path, DAYS[0], DAYS[1], source = BaoStock(tmp_path, bs), factor_codes = ['sh.600519'])['status'] == 'published'
    assert coverage_of(tmp_path).verified_through == DAYS[1]
    r = update_daily(tmp_path, DAYS[2], DAYS[2], source = BaoStock(tmp_path, bs))            # 当日确认无复权事件
    c = coverage_of(tmp_path)
    assert r['status'] == 'published' and c.verified_through == DAYS[2] and c.status == 'complete' and not c.has_gap
    r = update_daily(tmp_path, DAYS[3], DAYS[4], source = BaoStock(tmp_path, bs))            # 2025-06-26 茅台除息，事件由当日查询给出
    c = coverage_of(tmp_path)
    assert r['status'] == 'published' and r['adj_events'] == 1 and c.verified_through == DAYS[4] and c.status == 'complete'
    s = Store(tmp_path)
    view = with_adjusted(s.load('bars_1d').query('instrument == "600519.SH"'), s.load('adj_factors'), s.load('adj_coverage'))
    assert (view.adjustment_status == 'usable').all() and view.close_adj.notna().all()


def test_increment_after_missing_session_does_not_bridge_gap(tmp_path):
    bs = FakeBS(DAYS)
    update_daily(tmp_path, DAYS[0], DAYS[1], source = BaoStock(tmp_path, bs), factor_codes = ['sh.600519'])
    r = update_daily(tmp_path, DAYS[3], DAYS[4], source = BaoStock(tmp_path, bs))            # 跳过了 DAYS[2]
    assert r['status'] == 'published' and coverage_of(tmp_path).verified_through == DAYS[1]


def test_failed_increment_does_not_extend_coverage(tmp_path):
    class Broken(FakeBS):
        def query_daily_adjust_factor(self, day):
            if day == str(DAYS[3]): raise RuntimeError('adjust factor query failed')
            return super().query_daily_adjust_factor(day)
    update_daily(tmp_path, DAYS[0], DAYS[1], source = BaoStock(tmp_path, FakeBS(DAYS)), factor_codes = ['sh.600519'])
    r = update_daily(tmp_path, DAYS[2], DAYS[4], source = BaoStock(tmp_path, Broken(DAYS)))
    assert r['status'] == 'rejected' and coverage_of(tmp_path).verified_through == DAYS[1]
