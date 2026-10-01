"""公开回放入口的黄金样本：临时 Parquet 快照 → run_offline（真实执行适配器、公司行动接入与账本）。
期望值全部是手算常量。手算基准（主板 2024 年费率：佣金万 2.5 最低 5 元、过户费 0.01‰、印花税仅卖出）：
  10 元买入，目标金额 100000：10000 股加费用超出现金，缩量到 9900 股；成交额 99000，佣金 24.75，过户费 0.99，费用 25.74；
  剩余现金 974.26；按 10 元估值净值 99974.26。"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from observe import replay
from observe.ledger import Rule, RuleSet
from observe.loop import run_loop
from observe.replay import reproduce, run_offline
from observe.runs import RunDirError
from tests.integration.helpers import A, BIG, D, action, bar, flat, instruments, snapshot, tree_hash

CASH_AFTER_BUY, EQUITY = 974.26, 99974.26


def run(root, sid, **kw): return run_offline(root, snapshot = sid, initial_cash = 100000, execution = {'liquidity_window': 2}, **kw)


def table(result, name): return json.loads((Path(result['output']) / f'{name}.json').read_text(encoding = 'utf-8'))


def by_date(rows, key = 'date'): return {r[key]: r for r in rows}


# 1. 原始价执行，复权基准不影响订单与资金账 ----------------------------------------------------------------
@pytest.mark.parametrize('factor', [2.0, 3.0])
def test_raw_prices_drive_execution_even_with_adjust_factor(tmp_path, factor):
    adj = pd.DataFrame({'instrument': [A], 'ex_date': [D[0]], 'back_factor': [factor]})
    cov = pd.DataFrame([{'instrument': A, 'status': 'complete', 'verified_from': D[0], 'verified_through': D[-1], 'has_start_basis': True, 'has_gap': False}])
    r = run(tmp_path, snapshot(tmp_path, flat(A), adj = adj, coverage = cov))
    fills = table(r, 'fills')
    assert r['status'] == 'success' and fills[0]['fill_price'] == 10.0 and fills[0]['qty_filled'] == 9900 and fills[0]['fee'] == 25.74
    assert not [o for o in table(r, 'orders') if o.get('reject_reason') == 'limit_up']          # 复权开盘价 20 对原始前收盘 10 会误判涨停
    eq = by_date(table(r, 'equity'))
    assert eq[str(D[3])]['cash'] == CASH_AFTER_BUY and eq[str(D[-1])]['equity'] == EQUITY


def test_changing_adjust_basis_does_not_change_orders_or_ledger(tmp_path):
    out = []
    for k, factor in enumerate((2.0, 7.5)):
        root = tmp_path / str(k); adj = pd.DataFrame({'instrument': [A], 'ex_date': [D[0]], 'back_factor': [factor]})
        r = run(root, snapshot(root, flat(A), adj = adj)); out.append([table(r, n) for n in ('orders', 'fills', 'cash_events', 'equity')])
    assert out[0] == out[1]


# 2. 停牌：持仓保留、陈旧估值，不触发缺行情 ------------------------------------------------------------------
def test_suspended_holding_is_kept_and_marked_stale(tmp_path):
    rows = flat(A, days = D[:4]) + [bar(D[4], A, 10.0, trading = False)] + flat(A, days = D[5:])
    r = run(tmp_path, snapshot(tmp_path, rows))
    assert r['status'] == 'success'
    pos = {(p['date'], p['instrument']): p for p in table(r, 'positions_daily')}
    assert pos[(str(D[4]), A)]['qty'] == 9900 and pos[(str(D[4]), A)]['stale'] is True and pos[(str(D[4]), A)]['last_price'] == 10.0
    eq = by_date(table(r, 'equity'))
    assert eq[str(D[4])]['stale_price'] is True and eq[str(D[4])]['equity'] == EQUITY and eq[str(D[5])]['stale_price'] is False
    assert not [f for f in table(r, 'fills') if f['side'] == 'sell']                               # 停牌不是退出研究候选的理由


# 3. 10 送 10：股数翻倍、净值不出现虚假亏损；公开入口与底层循环一致 -----------------------------------------------
def bonus_rows(): return flat(A, days = D[:4]) + [bar(d, A, 5.0) for d in D[4:]]


def test_bonus_share_doubles_quantity_without_fake_loss(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, bonus_rows(), actions = [action(A, D[4], bonus = 1.0)]))
    assert r['status'] == 'success'
    pos = {(p['date'], p['instrument']): p for p in table(r, 'positions_daily')}
    assert pos[(str(D[4]), A)]['qty'] == 19800 and pos[(str(D[4]), A)]['pending'] == 9900 and pos[(str(D[4]), A)]['sellable'] == 9900
    assert pos[(str(D[5]), A)]['pending'] == 0 and pos[(str(D[5]), A)]['sellable'] == 19800        # 红股上市日缺失，按下一交易日推断
    assert [e['equity'] for e in table(r, 'equity')][1:] == [EQUITY] * 5
    status = json.loads((Path(r['output']) / 'status.json').read_text(encoding = 'utf-8'))
    assert [a['field'] for a in status['assumptions']] == ['bonus_list_date']


def test_bonus_share_public_entry_matches_direct_loop(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, bonus_rows(), actions = [action(A, D[4], bonus = 1.0)]))
    rules = RuleSet.from_yaml('configs/rule_profiles/main_board.yaml')
    px = {d: (10.0 if d <= D[3] else 5.0) for d in D}
    market = {d: {A: {'open': px[d], 'close': px[d], 'preclose': px[d], 'suspended': False, 'board': 'main', 'is_st': False, 'avg_amount_20d': BIG}} for d in D[2:]}
    book, _ = run_loop(D[2:], market, {d: {A: 0.0} for d in D[2:]}, 100000, rules, eligible_by_date = {d: {A} for d in D[2:]},
                       actions = {D[4]: [{'instrument': A, 'ex_date': D[4], 'bonus_ratio': 1.0}]}, n = 1, max_weight = 1.0, rebalance_every = 1, calendar = D)
    assert [e['equity'] for e in table(r, 'equity')] == [x['equity'] for x in book.equity_rows]
    assert [e['cash'] for e in table(r, 'equity')] == [x['cash'] for x in book.equity_rows]


def test_price_reset_without_recorded_action_blocks_instead_of_fake_loss(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, bonus_rows()))
    assert r['status'] == 'blocked' and 'unrecorded corporate action' in r['blocked'][0]['detail']
    status = json.loads((Path(r['output']) / 'status.json').read_text(encoding = 'utf-8'))
    assert status['status'] == 'blocked' and status['stages']['loop']['state'] == 'blocked'


# 4. 现金分红：除息日记应收，到账日转可用现金，不重复计收益 ------------------------------------------------------
def test_cash_dividend_receivable_then_paid_once(tmp_path):
    rows = flat(A, days = D[:4]) + [bar(d, A, 9.5) for d in D[4:]]
    r = run(tmp_path, snapshot(tmp_path, rows, actions = [action(A, D[4], cash = 0.5, pay = D[6])]))
    assert r['status'] == 'success'
    eq = by_date(table(r, 'equity'))
    assert eq[str(D[4])]['receivable'] == 4950.0 and eq[str(D[4])]['cash'] == CASH_AFTER_BUY and eq[str(D[4])]['equity'] == EQUITY
    assert eq[str(D[5])]['receivable'] == 4950.0 and eq[str(D[6])]['receivable'] == 0.0 and eq[str(D[6])]['cash'] == 5924.26
    assert [e['equity'] for e in table(r, 'equity')][2:] == [EQUITY] * 4
    paid = [e for e in table(r, 'cash_events') if e['kind'] == 'dividend_paid']
    assert paid == [{'date': str(D[6]), 'kind': 'dividend_paid', 'amount': 4950.0}]
    assert {(x['date'], x['pay_date'], x['amount']) for x in table(r, 'receivables')} == {(str(d), str(D[6]), 4950.0) for d in D[4:6]}


# 5. 整个应有交易日缺失：阻断，不缩短回放 -------------------------------------------------------------------
def test_missing_whole_session_blocks(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, [x for x in flat(A) if x['date'] != D[4]]))
    assert r['status'] == 'blocked' and r['blocked'][0]['kind'] == 'missing_session' and r['blocked'][0]['dates'] == [str(D[4])]
    assert not (Path(r['output']) / 'equity.json').exists()


# 6. 成交额不足：参与度限制在正常入口生效，参考成交额只用决策时点之前的数据 ------------------------------------------
def test_participation_limit_uses_history_before_decision(tmp_path):
    rows = [bar(d, A, 10.0, amount = 1e6) for d in D[:3]] + [bar(d, A, 10.0) for d in D[3:]]
    r = run(tmp_path, snapshot(tmp_path, rows))
    first = table(r, 'fills')[0]
    # 1/4、1/5 平均成交额 100 万 × 5% = 5 万元 → 5000 股；执行日当天 10 亿成交额不能使用
    assert first['date'] == str(D[3]) and first['qty_requested'] == 10000 and first['qty_filled'] == 5000 and first['remaining_qty'] == 5000 and first['status'] == 'partial'


def test_no_liquidity_history_rejects_orders_instead_of_infinite_default(tmp_path):
    rows = flat(A)
    r = run_offline(tmp_path, snapshot = snapshot(tmp_path, rows), initial_cash = 100000, execution = {'liquidity_window': 2}, start = D[0])
    assert r['status'] == 'success_limited' and r['limitations'][0]['kind'] == 'liquidity_warmup_short'
    first = table(r, 'orders')[0]
    assert first['exec_date'] == str(D[1]) and first['reject_reason'] == 'no_liquidity_reference'


def test_candidates_are_enabled_board_stocks_only(tmp_path):
    rows = flat('000001.SH') + flat('300001.SZ') + flat(A)
    inst = pd.concat([instruments('000001.SH', kind = 'index').assign(board = 'index'), instruments('300001.SZ').assign(board = 'gem'), instruments(A)])
    r = run(tmp_path, snapshot(tmp_path, rows, inst = inst))
    assert {s['instrument'] for s in table(r, 'scores')} == {A} and table(r, 'fills')[0]['instrument'] == A


# 11. 原实验保护 ---------------------------------------------------------------------------------------
def test_default_runs_never_share_a_directory(tmp_path):
    sid = snapshot(tmp_path, flat(A)); a, b = run(tmp_path, sid), run(tmp_path, sid)
    assert a['output'] != b['output'] and (Path(a['output']) / 'manifest.json').exists()


def test_explicit_existing_output_is_refused_and_untouched(tmp_path):
    sid = snapshot(tmp_path, flat(A)); first = run(tmp_path, sid); before = tree_hash(first['output'])
    with pytest.raises(RunDirError): run(tmp_path, sid, output = first['output'])
    assert tree_hash(first['output']) == before


@pytest.mark.parametrize('where', ['same', 'inside', 'dotted'])
def test_reproduce_output_overlapping_source_is_refused(tmp_path, where):
    sid = snapshot(tmp_path, flat(A)); src = run(tmp_path, sid)['output']; before = tree_hash(src)
    p = Path(src); target = {'same': p, 'inside': p / 'again', 'dotted': p / '..' / p.name}[where]
    with pytest.raises(RunDirError): reproduce(tmp_path, src, output = str(target))
    assert tree_hash(src) == before


def test_default_reproduce_uses_new_directories_and_keeps_source(tmp_path):
    sid = snapshot(tmp_path, flat(A)); src = run(tmp_path, sid)['output']; before = tree_hash(src)
    a, b = reproduce(tmp_path, src), reproduce(tmp_path, src)
    assert a['output'] != b['output'] and src not in (a['output'], b['output']) and tree_hash(src) == before


def test_failure_keeps_running_status_and_completed_stages(tmp_path, monkeypatch):
    sid = snapshot(tmp_path, flat(A)); src = run(tmp_path, sid)['output']; before = tree_hash(src)
    def boom(*a, **k): raise RuntimeError('evaluation exploded')
    monkeypatch.setattr(replay, 'evaluate', boom)
    with pytest.raises(RuntimeError): run(tmp_path, sid, cache = False)
    with pytest.raises(RuntimeError): reproduce(tmp_path, src)
    failed = [json.loads(p.read_text(encoding = 'utf-8')) for p in (tmp_path / 'runs').glob('*/status.json')]
    failed = [s for s in failed if s['status'] == 'failed']
    assert len(failed) == 2 and all(s['stages']['loop']['state'] == 'done' and 'evaluation exploded' in s['error'] for s in failed)
    assert all(not (tmp_path / 'runs' / s['run_id'] / 'manifest.json').exists() for s in failed)
    assert tree_hash(src) == before


def test_unknown_snapshot_fails_before_creating_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError): run_offline(tmp_path, snapshot = 'nope')
    assert not (tmp_path / 'runs').exists()


def test_rules_constant_sanity():
    fees = RuleSet([Rule(start = date(2023, 8, 28), stamp_tax = 0.0005, transfer_fee = 0.00001, commission_rate = 0.00025, min_commission = 5.0)]).fees(99000, 'buy', D[3])
    assert fees['fee'] == 25.74


# 组合层交易统计与成本情景 -----------------------------------------------------------------------------
def test_trading_stats_match_hand_calculation(tmp_path):
    r = run(tmp_path, snapshot(tmp_path, flat(A))); t = table(r, 'trading')
    # 只有 1/5 一笔买入：成交额 99000 / 前一日净值 100000 = 0.99，六个交易日平均 0.165
    assert t['turnover_two_sided_daily_mean'] == pytest.approx(0.99 / 6) and t['turnover_half_daily_mean'] == pytest.approx(0.99 / 12)
    assert t['fees'] == {'commission': 24.75, 'stamp_tax': 0.0, 'transfer_fee': 0.99, 'fee': 25.74} and t['fee_ratio_to_initial'] == pytest.approx(25.74 / 100000)
    assert t['orders'] == 1 and t['fill_rate'] == 1.0 and t['reject_reasons'] == {}
    assert t['max_single_weight_mean'] == pytest.approx(5 * 99000 / EQUITY / 6) and t['cash_share_mean'] == pytest.approx((1 + 5 * CASH_AFTER_BUY / EQUITY) / 6)


def test_fee_multiplier_reruns_the_loop_with_scaled_costs(tmp_path):
    sid = snapshot(tmp_path, flat(A))
    r = run_offline(tmp_path, snapshot = sid, initial_cash = 100000, execution = {'liquidity_window': 2, 'fee_multiplier': 2.0})
    f = table(r, 'fills')[0]
    assert f['commission'] == 49.5 and f['transfer_fee'] == 1.98 and f['fee'] == 51.48 and table(r, 'equity')[-1]['cash'] == round(100000 - 99000 - 51.48, 2)
    assert reproduce(tmp_path, r['output'])['reproduction']['result'] == 'match'
