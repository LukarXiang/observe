"""数据层：标准化、分区与批次发布、快照、清理保护、端到端更新、审计、跨进程锁（模块 10、决策 19）。离线，用探测数据截取的片段。"""
import subprocess, sys, threading
from datetime import date

import pandas as pd
import pytest

from observe.data import standardize as std
from observe.data.audit import audit_daily
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.data.update import update_daily

FX = 'tests/fixtures/raw/'
RAW = pd.read_csv(FX + 'bs_daily_2015-07-08.csv', dtype = str, keep_default_na = False)


# 假 BaoStock：调用方式与真实客户端一致（翻页式结果集），数据来自探测留存片段 ---------------------------------
class RS:
    def __init__(self, df): self.error_code, self.error_msg, self.fields, self.rows, self.k = '0', '', list(df.columns), df.astype(str).values.tolist(), -1
    def next(self): self.k += 1; return self.k < len(self.rows)
    def get_row_data(self): return self.rows[self.k]


class FakeBS:
    def __init__(self, days, corrupt = None): self.days, self.corrupt, self.calls = days, corrupt, []
    def login(self): return RS(pd.DataFrame())
    def logout(self): pass
    def query_trade_dates(self, s, e):
        cal = pd.bdate_range(s, e); return RS(pd.DataFrame({'calendar_date': cal.strftime('%Y-%m-%d'), 'is_trading_day': ['1' if d.date() in self.days else '0' for d in cal]}))
    def query_stock_basic(self): return RS(pd.read_csv(FX + 'bs_stock_basic.csv', dtype = str, keep_default_na = False))
    def query_daily_history_k_AStock(self, day):
        self.calls.append(day); d = RAW.copy(); d['date'] = day
        if self.corrupt == day: d.loc[0, 'low'] = '99'                                     # 最低价高于收盘价
        return RS(d)
    def query_daily_adjust_factor(self, day):
        a = pd.read_csv(FX + 'bs_adjfactor_600519.csv', dtype = str); return RS(a[a.dividOperateDate == day] if len(a[a.dividOperateDate == day]) else a.iloc[:0])
    def query_adjust_factor(self, code, start_date, end_date): return RS(pd.read_csv(FX + 'bs_adjfactor_600519.csv', dtype = str))


class FakeTdx:
    def xdxr(self, code): return pd.read_csv(FX + 'tdx_xdxr_600519.csv') if code == '600519' else None


DAYS = [date(2025, 6, 23), date(2025, 6, 24), date(2025, 6, 25), date(2025, 6, 26), date(2025, 6, 27)]


# 标准化 ------------------------------------------------------------------------------------------------
def test_standardize_daily_units_suspension_and_boards():
    d = std.daily(RAW).set_index('instrument')
    sus = d[~d.is_trading]; assert len(sus) == (RAW.tradestatus == '0').sum() == 5 and sus[['open', 'high', 'low', 'close']].isna().all().all() and sus.preclose.notna().all()
    assert (sus.volume == 0).all() and d.loc['600000.SH', 'turnover'] == pytest.approx(0.05615244)
    assert d.is_st.sum() == 2 and d.loc['600000.SH', 'volume'] == 837950416
    assert [std.board(x) for x in ('001979.SZ', '003816.SZ', '002415.SZ', '300750.SZ', '688981.SH', '920000.BJ')] == ['main', 'main', 'main', 'gem', 'star', 'bse']
    assert std.instrument('sh.600000') == '600000.SH' and std.instrument('000001') == '000001.SZ'


def test_standardize_instruments_and_corp_actions():
    inst = std.instruments(pd.read_csv(FX + 'bs_stock_basic.csv', dtype = str, keep_default_na = False)).set_index('instrument')
    assert inst.loc['000418.SZ', 'delist_date'] == date(2019, 6, 21) and not inst.loc['000418.SZ', 'listed'] and inst.loc['000300.SH', 'kind'] == 'index'
    ca = std.corp_actions(pd.read_csv(FX + 'tdx_xdxr_600519.csv'), '600519.SH').set_index('ex_date')
    r = ca.loc[date(2002, 7, 25)]; assert (r.cash_per_share, r.bonus_ratio) == (0.6, 0.1)            # 10 送 1 派 6 元
    assert ca.pay_date.isna().all()                                                                      # 到账日由 BaoStock 补，缺失时账本从严推断


# 分区、发布、快照、清理 ---------------------------------------------------------------------------------------
def frame(v): return pd.DataFrame({'date': [date(2025, 1, 2)], 'instrument': ['600000.SH'], 'close': [v]})


def test_partition_fingerprint_dedup_and_same_content_same_file(tmp_path):
    s = Store(tmp_path); a = s.write_partition('bars_1d', '2025', pd.concat([frame(1.0), frame(2.0)]))
    assert a['rows'] == 1 and s.write_partition('bars_1d', '2025', frame(2.0))['file'] == a['file']   # 主键去重保留最后一条
    assert s.write_partition('bars_1d', '2025', frame(3.0))['file'] != a['file']


def test_publish_merges_and_stale_base_is_refused(tmp_path):
    s = Store(tmp_path); b1 = s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', frame(1.0))}}); s.publish(b1)
    b2 = s.write_batch({'adj_factors': {'all': s.write_partition('adj_factors', 'all', pd.DataFrame({'instrument': ['A'], 'ex_date': [date(2025, 1, 2)], 'back_factor': [1.0]}))}})
    b3 = s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', frame(9.0))}})
    s.publish(b2); assert set(s.published()['tables']) == {'bars_1d', 'adj_factors'}
    with pytest.raises(RuntimeError, match = '重新生成'): s.publish(b3)                               # b3 基于 b1，已过时


def test_snapshot_is_frozen_against_later_publish(tmp_path):
    s = Store(tmp_path); s.publish(s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', frame(1.0))}}))
    sid = s.snapshot(); s.publish(s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', frame(2.0))}}))
    assert s.load('bars_1d', snapshot = sid).close.tolist() == [1.0] and s.load('bars_1d').close.tolist() == [2.0]


def test_snapshot_never_sees_half_updated_tables_during_concurrent_publish(tmp_path):
    """每个批次同时改日线与复权因子（值相同）；并发快照读到的两张表必须来自同一批次"""
    s = Store(tmp_path); errors = []
    def writer():
        for v in range(1, 30):
            p = {'bars_1d': {'2025': s.write_partition('bars_1d', '2025', frame(float(v)))},
                 'adj_factors': {'all': s.write_partition('adj_factors', 'all', pd.DataFrame({'instrument': ['A'], 'ex_date': [date(2025, 1, 2)], 'back_factor': [float(v)]}))}}
            with operation_lock(tmp_path, DATA_WRITER, timeout = 5): s.publish(s.write_batch(p))
    def reader():
        for _ in range(60):
            try:
                if not s.published()['batch_id']: continue
                sid = s.snapshot(); b, a = s.load('bars_1d', snapshot = sid).close[0], s.load('adj_factors', snapshot = sid).back_factor[0]
                if b != a: errors.append((sid, b, a))
            except Exception as e: errors.append(repr(e))   # noqa: BLE001
    t = [threading.Thread(target = writer), threading.Thread(target = reader)]; [x.start() for x in t]; [x.join() for x in t]
    assert errors == []


def test_gc_protects_published_snapshots_pins_and_pending_batches(tmp_path):
    s = Store(tmp_path); w = lambda v: s.write_partition('bars_1d', '2025', frame(v))
    s.publish(s.write_batch({'bars_1d': {'2025': w(1.0)}})); s.snapshot()                           # 1.0 被快照保护
    s.publish(s.write_batch({'bars_1d': {'2025': w(2.0)}})); s.pin('job-1')                         # 2.0 已发布且被任务固定
    s.publish(s.write_batch({'bars_1d': {'2025': w(3.0)}}))                                          # 3.0 当前发布
    s.write_batch({'bars_1d': {'2025': w(4.0)}})                                                     # 4.0 待发布
    orphan = w(5.0)['file']                                                                          # 5.0 没有任何引用
    assert s.gc() == [orphan] and (tmp_path / orphan).exists()                                       # 默认只列出
    s.unpin('job-1'); assert len(s.gc()) == 2                                                        # 解除固定后 2.0 已无引用（当前发布的是 3.0）
    assert s.gc(apply = True) and not (tmp_path / orphan).exists() and len(s.load('bars_1d')) == 1


# 端到端更新 ------------------------------------------------------------------------------------------------
def test_update_publishes_and_rerun_downloads_nothing_new(tmp_path):
    bs = FakeBS(DAYS); r = update_daily(tmp_path, DAYS[0], DAYS[-1], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    assert r['status'] == 'published' and r['days'] == 5 and r['rows'] == 5 * len(RAW) and r['adj_events'] == 1   # 2025-06-26 茅台除息
    s = Store(tmp_path); ca = s.load('corp_actions'); assert (ca.instrument == '600519.SH').all() and len(ca) > 20
    assert s.load('bars_1d').groupby('date').size().tolist() == [len(RAW)] * 5
    r2 = update_daily(tmp_path, DAYS[0], DAYS[-1], source = BaoStock(tmp_path, bs)); assert r2['days'] == 0 and len(bs.calls) == 5
    assert (tmp_path / 'raw' / 'requests.jsonl').exists() and (tmp_path / 'raw' / 'baostock' / 'daily_market' / '2025-06-23.parquet').exists()


def test_update_with_blocking_audit_issue_is_not_published(tmp_path):
    ok = update_daily(tmp_path, DAYS[0], DAYS[1], source = BaoStock(tmp_path, FakeBS(DAYS)))
    bad = update_daily(tmp_path, DAYS[2], DAYS[2], source = BaoStock(tmp_path, FakeBS(DAYS, corrupt = str(DAYS[2]))))
    assert bad['status'] == 'rejected' and 'block/ohlc_order' in bad['issues'] and Store(tmp_path).published()['batch_id'] == ok['batch_id']


def test_update_reuses_successful_staged_days_after_source_failure(tmp_path):
    class Failing(FakeBS):
        def __init__(self, days, fail): super().__init__(days); self.fail = fail
        def query_daily_history_k_AStock(self, day):
            if day == str(self.fail): raise RuntimeError('mid-run failure')
            return super().query_daily_history_k_AStock(day)
    with pytest.raises(RuntimeError): update_daily(tmp_path, DAYS[0], DAYS[2], source = BaoStock(tmp_path, Failing(DAYS, DAYS[2])))
    retry = Failing(DAYS, date(2099, 1, 1)); result = update_daily(tmp_path, DAYS[0], DAYS[2], source = BaoStock(tmp_path, retry))
    assert result['status'] == 'published' and retry.calls == [str(DAYS[2])]


def test_partial_published_day_is_requested_again(tmp_path):
    bs = FakeBS(DAYS)
    first = update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    assert first['status'] == 'published'
    s = Store(tmp_path); old = s.load('bars_1d', parts = ['2025']).query('date == @DAYS[0]').iloc[[0]]
    s.publish(s.write_batch({'bars_1d': {'2025': s.write_partition('bars_1d', '2025', old)}}))
    bs.calls.clear(); result = update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    assert bs.calls == [str(DAYS[0])] and result['verified_days'] == 1


def test_empty_requested_day_is_reported_as_missing(tmp_path):
    class Empty(FakeBS):
        def query_daily_history_k_AStock(self, day):
            self.calls.append(day); return RS(RAW.iloc[:0])
    result = update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, Empty(DAYS)), tdx = FakeTdx())
    assert result['status'] == 'rejected' and result['missing_days'] == [str(DAYS[0])] and 'block/empty_day' in result['issues']


def test_force_reloads_a_valid_checkpoint(tmp_path):
    bs = FakeBS(DAYS); update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    bs.calls.clear(); result = update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx(), force = True)
    assert result['status'] == 'published' and bs.calls == [str(DAYS[0])]


def test_corrupt_checkpoint_is_not_published_without_refetch(tmp_path):
    bs = FakeBS(DAYS); update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    (tmp_path / 'staging' / 'daily' / f'{DAYS[0]}.parquet').write_bytes(b'not parquet')
    bs.calls.clear(); result = update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    assert result['status'] == 'published' and bs.calls == [str(DAYS[0])]


def test_corp_action_refresh_failure_keeps_old_history_and_reports_limit(tmp_path):
    class BrokenTdx(FakeTdx):
        def xdxr(self, code): raise RuntimeError('tdx unavailable')
    bs = FakeBS(DAYS); update_daily(tmp_path, DAYS[0], DAYS[-1], source = BaoStock(tmp_path, bs), tdx = FakeTdx())
    old = Store(tmp_path).load('corp_actions')
    result = update_daily(tmp_path, DAYS[3], DAYS[3], source = BaoStock(tmp_path, bs), tdx = BrokenTdx(), force = True)
    assert result['status'] == 'rejected' and result['component_failures'] and len(Store(tmp_path).load('corp_actions')) == len(old)


def test_audit_limits_st_and_fresh_listing_exemption():
    b = pd.DataFrame({'date': [DAYS[1]] * 3, 'instrument': ['600001.SH', '600002.SH', '600003.SH'], 'open': [10.0] * 3, 'high': [11.0] * 3, 'low': [10.0] * 3,
                      'close': [10.8, 10.8, 10.8], 'preclose': [10.0] * 3, 'volume': [100] * 3, 'amount': [1050.0] * 3, 'is_trading': True, 'is_st': [False, True, False], 'board': 'main'})
    inst = pd.DataFrame({'instrument': ['600001.SH', '600002.SH', '600003.SH'], 'list_date': [date(2000, 1, 4), date(2000, 1, 4), DAYS[0]]})
    iss = audit_daily(b, DAYS[:2], inst); over = iss[iss.rule == 'beyond_limit'].instrument.tolist()
    assert over == ['600002.SH']                                  # ST 涨 8% 超 5%；普通股 8% 不超；上市第 2 天的新股豁免
    assert audit_daily(b.iloc[:0], DAYS[:1]).rule.tolist() == ['empty_day']


# 跨进程锁 ---------------------------------------------------------------------------------------------------
def hold(tmp_path, name):
    # Keep the competing process alive for the full queue subprocess startup;
    # the assertion is about lock rejection, not a timing race.
    code = f"import time; from observe.data.locks import operation_lock\nwith operation_lock(r'{tmp_path}', '{name}'):\n    print('held', flush = True); time.sleep(30)"
    p = subprocess.Popen([sys.executable, '-c', code], stdout = subprocess.PIPE, text = True); assert p.stdout.readline().strip() == 'held'; return p


def test_update_fails_fast_when_another_process_holds_writer_lock(tmp_path):
    p = hold(tmp_path, DATA_WRITER)
    try:
        with pytest.raises(RuntimeError, match = '锁被占用'): update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, FakeBS(DAYS)))
    finally: p.wait()
    assert update_daily(tmp_path, DAYS[0], DAYS[0], source = BaoStock(tmp_path, FakeBS(DAYS)))['status'] == 'published'


def test_baostock_lock_is_exclusive_across_processes(tmp_path):
    p = hold(tmp_path, 'baostock')
    try:
        with pytest.raises(RuntimeError, match = '锁被占用'):
            with BaoStock(tmp_path, FakeBS(DAYS)).session(): pass
    finally: p.wait()


# 后复权价 --------------------------------------------------------------------------------------------------
def test_adjusted_prices_use_factor_in_effect_and_unknown_stays_missing():
    from observe.data.prices import with_adjusted
    adj = std.adj_factors(pd.read_csv(FX + 'bs_adjfactor_600519.csv', dtype = str))
    d = [date(2002, 7, 24), date(2002, 7, 25), date(2002, 7, 26)]
    bars = pd.DataFrame({'date': d * 2, 'instrument': ['600519.SH'] * 3 + ['600000.SH'] * 3, 'open': [39.0, 35.0, 35.5, 10, 10, 10], 'high': 40.0, 'low': 30.0,
                         'close': [39.0, 35.0, 35.5, 10, 10, 10], 'preclose': 39.0})
    coverage = pd.DataFrame([{'instrument': '600519.SH', 'status': 'complete', 'verified_from': d[0], 'verified_through': d[-1], 'has_start_basis': True, 'has_gap': False}])
    m = with_adjusted(bars, adj, coverage).set_index(['instrument', 'date'])
    assert m.loc[('600519.SH', d[0]), 'back_factor'] == 1.0 and m.loc[('600519.SH', d[1]), 'back_factor'] == pytest.approx(1.11828)
    assert m.loc[('600519.SH', d[1]), 'ret'] == pytest.approx(35 * 1.11828 / 39 - 1)                        # 除权日收益按复权价连续
    assert m.loc[('600000.SH', d[0]), 'back_factor'] != m.loc[('600000.SH', d[0]), 'back_factor']          # 没有复权记录 → 缺失


def test_audit_uses_rule_profile_st_limit_changed_on_2026_07_06():
    from observe.ledger.rules import RuleSet
    r = RuleSet.from_yaml('configs/rule_profiles/main_board.yaml')
    mk = lambda d: pd.DataFrame({'date': [d], 'instrument': ['600730.SH'], 'open': [10.0], 'high': [10.8], 'low': [10.0], 'close': [10.8], 'preclose': [10.0],
                                 'volume': [100], 'amount': [1050.0], 'is_trading': True, 'is_st': True, 'board': 'main'})
    assert audit_daily(mk(date(2026, 7, 3)), [date(2026, 7, 3)], rules = r).rule.tolist() == ['beyond_limit']     # 改制前 ST 5%
    assert audit_daily(mk(date(2026, 7, 6)), [date(2026, 7, 6)], rules = r).empty                                 # 2026-07-06 起 10%
