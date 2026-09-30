"""按证券的盈亏归因（diagnose.attribute_run / concentration）：截止日来自净值日历、空仓市值为零、与账户净值勾稽。手工构造的回放产物。"""
import json
from types import SimpleNamespace

import pytest

from observe.diagnose import attribute_run, concentration

DAYS = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']


def run(tmp_path, name, equity, cash_events = (), positions = (), issues = None):
    d = tmp_path / name; d.mkdir()
    (d / 'equity.json').write_text(json.dumps(equity), encoding = 'utf-8'); (d / 'cash_events.json').write_text(json.dumps(list(cash_events)), encoding = 'utf-8')
    (d / 'positions_daily.json').write_text(json.dumps(list(positions)), encoding = 'utf-8'); (d / 'trading.json').write_text(json.dumps({}), encoding = 'utf-8')
    (d / 'status.json').write_text(json.dumps({'issues': issues or []}), encoding = 'utf-8'); return d


def eq(day, cash, mv = 0.0, receivable = 0.0): return {'date': day, 'cash': cash, 'market_value': mv, 'receivable': receivable, 'equity': round(cash + mv + receivable, 2)}
def ev(day, kind, amount, inst = 'X'): return {'date': day, 'kind': kind, 'amount': amount, **({'instrument': inst} if inst else {})}
def pos(day, qty, price, inst = 'X'): return {'date': day, 'instrument': inst, 'qty': qty, 'last_price': price}


def test_round_trip_ending_flat_earns_100_not_1100(tmp_path):
    d = run(tmp_path, 'a', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000), eq(DAYS[2], 1000100)],
            [ev(DAYS[1], 'buy', -1000), ev(DAYS[2], 'sell', 1100)], [pos(DAYS[1], 100, 10)])           # 最后一天已空仓，positions_daily 里最后一条记录是第二天的持仓
    r = attribute_run(d); assert r['end'] == DAYS[2] and r['holdings_at_end'] == 'flat' and r['pnl']['X'] == pytest.approx(100)
    assert r['reconciliation']['reconciled'] and r['reconciliation']['residual_unexplained'] == pytest.approx(0) and r['reconciliation']['account_equity_change'] == pytest.approx(100)


def test_partial_holding_on_the_last_day_is_marked_at_the_last_day(tmp_path):
    d = run(tmp_path, 'b', [eq(DAYS[0], 1e6), eq(DAYS[1], 998000, 2000), eq(DAYS[2], 999100, 1200)], [ev(DAYS[1], 'buy', -2000), ev(DAYS[2], 'sell', 1100)], [pos(DAYS[1], 200, 10), pos(DAYS[2], 100, 12)])
    r = attribute_run(d); assert r['holdings_at_end'] == 'listed' and r['pnl']['X'] == pytest.approx(300) and r['reconciliation']['reconciled']


def test_all_cash_run_returns_empty_attribution_without_errors(tmp_path):
    d = run(tmp_path, 'c', [eq(day, 1e6) for day in DAYS]); r = attribute_run(d)
    assert len(r['pnl']) == 0 and r['holdings_at_end'] == 'flat' and r['reconciliation']['reconciled'] and r['reconciliation']['account_equity_change'] == 0


def test_dividends_are_listed_as_unallocated_not_spread_or_relabelled(tmp_path):
    receivable = run(tmp_path, 'd1', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000), eq(DAYS[2], 1000100, 0, 50)], [ev(DAYS[1], 'buy', -1000), ev(DAYS[2], 'sell', 1100)], [pos(DAYS[1], 100, 10)])
    r = attribute_run(receivable); rec = r['reconciliation']       # 分红已除息、未到账：应收计入净值，但不属于任何证券的成交现金流
    assert r['pnl']['X'] == pytest.approx(100) and rec['identified_not_allocated']['receivable_at_end'] == 50 and rec['residual_unexplained'] == pytest.approx(0) and rec['account_equity_change'] == pytest.approx(150)
    paid = run(tmp_path, 'd2', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000), eq(DAYS[2], 1000150)], [ev(DAYS[1], 'buy', -1000), ev(DAYS[2], 'sell', 1100), ev(DAYS[2], 'dividend_paid', 50, None)], [pos(DAYS[1], 100, 10)])
    rec = attribute_run(paid)['reconciliation']                     # 已到账：现金流水有支持，但旧产物不带证券，整笔计入未分配
    assert rec['identified_not_allocated']['dividends_paid_unallocated'] == 50 and rec['identified_not_allocated']['receivable_at_end'] == 0 and rec['residual_unexplained'] == pytest.approx(0)
    broken = run(tmp_path, 'd3', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000), eq(DAYS[2], 1000130)], [ev(DAYS[1], 'buy', -1000), ev(DAYS[2], 'sell', 1100)], [pos(DAYS[1], 100, 10)])
    rec = attribute_run(broken)['reconciliation']                   # 净值里多出 30 元、现金流水与应收都没有支持：留作残差，不命名为分红
    assert rec['residual_unexplained'] == pytest.approx(30) and rec['reconciled'] is False


def test_cutoff_before_a_block_and_unknown_holdings(tmp_path):
    d = run(tmp_path, 'e', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000), eq(DAYS[2], 1000100)], [ev(DAYS[1], 'buy', -1000), ev(DAYS[2], 'sell', 1100)], [pos(DAYS[1], 100, 10)])
    r = attribute_run(d, until = DAYS[2]); assert r['end'] == DAYS[1] and r['pnl']['X'] == pytest.approx(0) and r['reconciliation']['reconciled']        # 阻断日之前：成交与持仓都只算到 01-03
    unknown = run(tmp_path, 'f', [eq(DAYS[0], 1e6), eq(DAYS[1], 999000, 1000)], [ev(DAYS[1], 'buy', -1000)], [])
    assert 'insufficient_data' in attribute_run(unknown) and '无法区分空仓与状态缺失' in attribute_run(unknown)['insufficient_data']       # 净值有市值而持仓明细缺失：不猜
    assert 'insufficient_data' in attribute_run(d, until = DAYS[0])


def test_concentration_aligns_both_arms_to_the_common_cutoff(tmp_path):
    def arm(name, n, issues = None):
        eqs = [eq(DAYS[0], 1e6)] + [eq(DAYS[k], 999000, 1000 + 10 * k) for k in range(1, n)]
        return run(tmp_path, name, eqs, [ev(DAYS[1], 'buy', -1000)], [pos(DAYS[k], 100, 10 + 0.1 * k) for k in range(1, n)], issues)
    base, ext = arm('base', 3), arm('ext', 5)                          # 双方最后有效日期不同：基础侧到 01-04，加分钟侧到 01-08
    s = {'cfg': SimpleNamespace(backtest_models = ['m'], initial_cash = 1e6), 'sub': {'backtests': [{'model': 'm', 'arm': 'base', 'status': 'success', 'output': str(base)}, {'model': 'm', 'arm': 'extended', 'status': 'success', 'output': str(ext)}]}}
    c = concentration(s, [])['m']; assert c['evaluated_through'] == DAYS[2] and c['planned_arm_ends'] == {'base': DAYS[2], 'extended': DAYS[4]} and c['arms']['extended']['holdings_at_end'] == 'listed'
    blocked = arm('blk', 5, [{'date': DAYS[3], 'kind': 'delisted_holding', 'instrument': 'X'}])                    # 一侧在 01-05 被阻断：共同截止日是阻断日之前
    s['sub']['backtests'][0].update(status = 'blocked', output = str(blocked))
    c = concentration(s, [])['m']; assert c['evaluated_through'] == DAYS[2] and c['valid_portfolio_comparison'] is False and c['attribution_status'] in ('complete', 'partial', 'unreconciled')
