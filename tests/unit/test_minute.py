"""外部分钟线导入（模块 10、决策 17）：格式变体、逐文件校验、日线对账、5 分钟合成、分钟股票池、端到端发布。离线，全部合成数据。"""
import io, json, zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.data import minute as mn
from observe.data.store import Store


def gen(seed, base = 10.0):
    """一天 240 根 1 分钟线：成交量按手取整、成交额 = 成交量 × 收盘价，满足 OHLC 关系"""
    rng = np.random.default_rng(seed); close = np.round(base + np.cumsum(rng.normal(0, 0.01, 240)), 2); open_ = np.r_[close[0], close[:-1]]
    high = np.round(np.maximum(open_, close) + 0.01, 2); low = np.round(np.minimum(open_, close) - 0.01, 2)
    vol = 100 * rng.integers(1, 20, 240); return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'amount': np.round(vol * close, 2), 'volume': vol})


def csv(df, order = None, with_time = False):
    d = df[order or mn.COLS].copy()
    if with_time: d.insert(0, 'time', mn.TIME_STR)
    return d.to_csv(index = False)


def make_zip(path, files):
    path.parent.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(path, 'w') as z:
        for name, text in files.items(): z.writestr(name, text)


def test_file_key_variants():
    assert mn._key('sh/600000.csv') == ('sh', '600000')
    assert mn._key('20260617/sz/sz000001.csv') == ('sz', '000001')
    assert mn._key('20251201/bj/920000.bj.csv') == ('bj', '920000')
    assert mn._key('bj/bj920001.csv') == ('bj', '920001')
    assert mn._key('20210727.zip') is None


def test_five_minute_bars_align_to_session_edges():
    df = gen(1); f = mn.to_5m(df)
    assert f.shape == (48, 6)
    assert (f[0, 0], f[0, 3]) == (round(df.open[0], 2), round(df.close[4], 2))
    assert f[0, 1] == round(df.high[:5].max(), 2) and f[0, 2] == round(df.low[:5].min(), 2)
    assert f[:, 4].sum() == df.volume.sum() and abs(f[:, 5].sum() - df.amount.sum()) < 1e-6
    assert mn.BAR_OFFSETS[0] == pd.Timedelta('9h35min') and mn.BAR_OFFSETS[23] == pd.Timedelta('11h30min')
    assert mn.BAR_OFFSETS[24] == pd.Timedelta('13h05min') and mn.BAR_OFFSETS[47] == pd.Timedelta('15h')
    assert f[23, 3] == round(df.close[119], 2) and f[24, 0] == round(df.open[120], 2)   # 上午最后一根与下午第一根各自成组，不跨午休


def test_parse_by_header_name_not_position():
    df = gen(2)
    std_ = mn.parse_file(csv(df).encode()); swapped = mn.parse_file(csv(df, ['open', 'high', 'low', 'close', 'volume', 'amount'], with_time = True).encode())
    assert mn.check(std_) is None and mn.check(swapped) is None
    assert (swapped.volume.to_numpy() == df.volume.to_numpy()).all() and (swapped.amount.to_numpy() == df.amount.to_numpy()).all()
    assert mn.parse_file(('open,high,low,close,amount,volume\n').encode()) is None and mn.check(None) == 'empty_file'
    with pytest.raises(ValueError, match = '缺少必要列'): mn.parse_file(b'a,b\n1,2\n')


@pytest.mark.parametrize('mutate,why', [
    (lambda d: d.iloc[:239], 'bad_row_count'),
    (lambda d: d.assign(low = d.low + 5), 'ohlc_inconsistent'),
    (lambda d: d.assign(close = d.close.where(d.index != 3)), 'has_nan'),
    (lambda d: d.assign(volume = -d.volume), 'bad_value'),
])
def test_check_rejects_broken_files(mutate, why):
    assert mn.check(mutate(gen(3))) == why


def test_check_time_column_must_match_positions():
    df = gen(4); good = mn.parse_file(csv(df, with_time = True).encode()); assert mn.check(good) is None
    bad = good.copy(); bad.loc[0, 'time'] = '09:30'
    assert mn.check(bad) == 'time_mismatch'
    start = good.copy(); start['time'] = mn.START_STR; assert mn.check(start) is None                          # 20251201-03 的起点标签变体
    one = good.copy(); one.loc[119, 'time'] = '13:01'; assert mn.check(one) == 'time_mismatch'                 # 其它标签不是任何已知格式


def test_start_labelled_files_are_grouped_by_time_not_position():
    """20251201-03：时间列是分钟起点。09:35 那根只含 4 行（起点 09:31..09:34），13:05 那根含起点 13:00..13:04，15:00 那根含起点 14:56..15:00"""
    df = gen(5); df['time'] = mn.START_STR; g = mn.bar_groups(df)
    assert np.bincount(g).tolist() == [4] + [5] * 22 + [5, 5] + [5] * 22 + [6]
    f = mn.to_5m(df)
    assert f.shape == (48, 6) and f[:, 4].sum() == df.volume.sum() and abs(f[:, 5].sum() - df.amount.sum()) < 1e-6
    assert f[0, 4] == df.volume[:4].sum() and f[23, 4] == df.volume[115:119].sum() + df.volume[114] and f[24, 4] == df.volume[119:124].sum()
    assert f[24, 0] == round(df.open[119], 2) and f[47, 3] == round(df.close[239], 2) and f[47, 4] == df.volume[234:240].sum()
    assert (mn.to_5m(df.drop(columns = 'time')) != f).any()                                                     # 同样的数据按位置分组结果不同


def wanted_for(files):
    """按分钟数据自身算出与日线完全一致的对账口径"""
    return {i: (float(d.volume.sum()), float(d.amount.sum()), True) for i, d in files.items()}


def test_process_day_filters_reconciles_and_flags(tmp_path):
    day = date(2024, 5, 8); a, b, c, s = (gen(i, 10 + i) for i in range(4))
    bad = gen(9).assign(low = lambda d: d.low + 5)
    z = tmp_path / 'x.zip'
    make_zip(z, {'20240508/sh/600001.csv': csv(a), '20240508/sh/sh600001.csv': csv(a),        # 同一证券两种写法，内容相同
                 '20240508/sz/000002.sz.csv': csv(b), '20240508/sh/600004.csv': csv(bad), '20240508/sz/000005.csv': csv(s),
                 '20240508/sh/600009.csv': csv(gen(7))})                                        # 600009 不在名单里
    want = {'600001.SH': (float(a.volume.sum()), float(a.amount.sum()), True),
            '000002.SZ': (float(b.volume.sum()) * 1.05, float(b.amount.sum()), True),               # 日线成交量高 5%：对账不过
            '600004.SH': (1e6, 1e7, True), '600003.SH': (1e6, 1e7, True),                            # 一个坏文件、一个缺文件
            '000005.SZ': (0.0, 0.0, False)}                                                          # 日线停牌但分钟线有成交
    out, issues, st = mn.process_day((day, str(z), want, None))
    rules = {(i['rule'], i['instrument']) for i in issues}
    assert set(out.instrument) == {'600001.SH'} and len(out) == 48 and (st['expected'], st['kept'], st['validated']) == (4, 1, ['600001.SH']) and sorted(st['failed']) == ['000002.SZ', '600003.SH', '600004.SH'] and st['not_trading'] == ['000005.SZ']
    assert ('excluded_daily_mismatch', '000002.SZ') in rules and ('excluded_ohlc_inconsistent', '600004.SH') in rules
    assert ('missing_file', '600003.SH') in rules and ('bars_on_suspended_day', '000005.SZ') in rules
    assert out.bar_end.iloc[0] == pd.Timestamp('2024-05-08 09:35') and out.bar_end.iloc[-1] == pd.Timestamp('2024-05-08 15:00')
    assert (out.source == mn.SOURCE).all() and out.volume.sum() == a.volume.sum()


@pytest.mark.parametrize('amount_too', [False, True])
def test_process_day_rescales_units_only_when_whole_day_is_off_by_100(tmp_path, amount_too):
    day = date(2025, 12, 1); files = {f'{600000 + i}.SH': gen(i, 5 + i) for i in range(25)}
    z = tmp_path / 'lots.zip'
    shrink = (lambda d: d.assign(volume = d.volume // 100, amount = d.amount / 100)) if amount_too else (lambda d: d.assign(volume = d.volume // 100))
    make_zip(z, {f'20251201/sh/{k[:6]}.sh.csv': csv(shrink(d), ['open', 'high', 'low', 'close', 'volume', 'amount']) for k, d in files.items()})
    out, issues, st = mn.process_day((day, str(z), wanted_for(files), None))
    assert st['kept'] == 25 and [i['rule'] for i in issues] == ['unit_rescaled'] and ('成交额' in issues[0]['detail']) == amount_too
    assert out.groupby('instrument').volume.sum().eq([d.volume.sum() for d in files.values()]).all()
    assert np.allclose(out.groupby('instrument').amount.sum().to_numpy(), [d.amount.sum() for d in files.values()], rtol = 1e-6)


def test_seven_zip_needed_but_missing_gives_clear_error(tmp_path, monkeypatch):
    p = tmp_path / '20260902.zip'; p.write_bytes(b'7z\xbc\xaf\x27\x1c' + b'\0' * 32)      # 扩展名是 zip、内容是 7z
    monkeypatch.setattr(mn, 'SEVENZIP', ()); monkeypatch.delenv('OBSERVE_7Z', raising = False)
    with pytest.raises(RuntimeError, match = '7-Zip'): mn.read_members(str(p), {('sh', '600000'): '600000.SH'})


def test_plan_days_warmup_and_year_overlap():
    s = list(pd.bdate_range('2021-12-01', '2022-01-14').date)
    p = mn.plan_days(s, {2022}, warm = 3); assert min(p) == date(2021, 12, 29) and set().union(*p.values()) == {2022}
    p = mn.plan_days(s, {2021, 2022}, warm = 3); assert p[date(2021, 12, 28)] == {2021} and p[date(2021, 12, 30)] == {2021, 2022} and p[date(2022, 1, 5)] == {2022}
    assert list(mn.plan_days(s, {2022}, start = date(2022, 1, 5), end = date(2022, 1, 6), warm = 3)) == [date(2022, 1, 5), date(2022, 1, 6)]


def daily_frame(sessions, insts, amounts, board = 'main'):
    return pd.DataFrame([{'date': d, 'instrument': i, 'amount': amounts[i], 'volume': 1e6, 'is_trading': True, 'board': board if i != '300001.SZ' else 'gem'} for d in sessions for i in insts])


def test_universe_takes_top_main_board_with_enough_days_and_full_prior_year():
    s = list(pd.bdate_range('2021-01-04', '2021-12-31').date) + [date(2022, 1, 4)]
    amt = {'600001.SH': 5e8, '600002.SH': 9e8, '600003.SH': 1e8, '300001.SZ': 1e10, '600004.SH': 8e8}
    b = daily_frame(s, list(amt), amt); b = b[~((b.instrument == '600004.SH') & (b.date > date(2021, 4, 30)))]           # 600004 只有约 85 个交易日
    u = mn.build_universe(b, top = 2)
    assert list(u.year.unique()) == [2022] and u.instrument.tolist() == ['600002.SH', '600001.SH'] and u['rank'].tolist() == [1, 2]
    assert u.generated_asof.iloc[0] == date(2021, 12, 31) and set(u.traded_days) == {len(s) - 1}
    assert len(mn.build_universe(b[b.date >= date(2021, 3, 1)], top = 2)) == 0        # 上一年不完整：不生成


@pytest.fixture
def world(tmp_path):
    """已发布的日线 + 分钟压缩包：2021 全年日线，2022-01-04..07；压缩包只放预热 20 个交易日 + 2022 年初 4 个交易日，共 24 天"""
    root = tmp_path / 'data'; store = Store(root)
    s21 = list(pd.bdate_range('2021-01-04', '2021-12-31').date); s22 = list(pd.bdate_range('2022-01-04', '2022-01-07').date); sessions = s21 + s22
    insts = ['600001.SH', '000002.SZ', '600003.SH']; amt = {'600001.SH': 9e8, '000002.SZ': 8e8, '600003.SH': 1e8}
    fwd = {(d, i): gen(hash((str(d), i)) % 10 ** 6, 10 + n) for d in sessions[-24:] for n, i in enumerate(insts)}
    rows = []
    for d in sessions:
        for i in insts:
            m = fwd.get((d, i)); rows.append({'date': d, 'instrument': i, 'amount': float(m.amount.sum()) if m is not None else amt[i], 'volume': float(m.volume.sum()) if m is not None else 1e6, 'is_trading': True, 'board': 'main'})
    bars = pd.DataFrame(rows); cal = pd.DataFrame({'date': sessions, 'is_open': True})
    parts = {'bars_1d': {str(y): store.write_partition('bars_1d', str(y), g) for y, g in bars.groupby(bars.date.map(lambda x: x.year))}, 'calendar': {'all': store.write_partition('calendar', 'all', cal)}}
    store.publish(store.write_batch(parts, 'base'))
    src = tmp_path / 'src'
    for d in sessions[-24:]:
        make_zip(src / f'{d:%Y}' / f'{d:%m}' / f'{d:%Y%m%d}.zip', {f'sh/{i[:6]}.csv' if i.endswith('SH') else f'sz/{i[:6]}.csv': csv(fwd[(d, i)]) for i in insts})
    return root, str(src), insts, len(sessions[-24:])


def test_import_minute_publishes_bars_5m_and_keeps_existing_tables(world):
    root, src, insts, ndays = world; store = Store(root); before = store.published()
    r = mn.import_minute(root, src, top = 2, workers = 1)
    assert r['status'] == 'published' and r['issues'] == {}
    pub = store.published(); assert pub['batch_id'] != before['batch_id']
    assert pub['tables']['bars_1d'] == before['tables']['bars_1d'] and pub['tables']['calendar'] == before['tables']['calendar']    # 原有表不动
    b5 = store.load('bars_5m'); u = store.load('minute_universe')
    assert set(u.instrument) == {'600001.SH', '000002.SZ'} and (u.year == 2022).all()                               # 只取名单内的
    assert set(b5.instrument) == set(u.instrument) and len(b5) == 2 * 48 * ndays == r['rows']
    assert b5.groupby([b5.bar_end.dt.date, 'instrument']).size().eq(48).all() and not b5.duplicated(['bar_end', 'instrument']).any()
    assert set(store.published()['tables']['bars_5m']) == {'202112', '202201'}
    meta = json.loads((root / 'minute_audits' / f'{r["batch_id"]}.json').read_text(encoding = 'utf-8')); assert meta['stock_days_kept'] == 2 * ndays
    assert mn.import_minute(root, src, top = 2, workers = 1)['status'] == 'no_change'                            # 重跑同一批不产生新批次
    assert store.published()['batch_id'] == pub['batch_id']


def test_import_minute_records_missing_days_and_fails_without_any_data(world, tmp_path):
    root, src, insts, ndays = world; store = Store(root)
    import os; victim = next(p for p in sorted((tmp_path / 'src').rglob('*.zip')) if p.stem == '20220105'); os.remove(victim)
    r = mn.import_minute(root, src, top = 2, workers = 1)
    assert r['issues'] == {'warn/missing_day': 2} and len(store.load('bars_5m')) == 2 * 48 * (ndays - 1)      # 整日缺失按证券日记录（两只证券）
    assert r['coverage'] == {'planned': 2 * ndays, 'new_validated': 2 * (ndays - 1), 'reused_old': 0, 'missing': 2, 'invalidated': 0, 'published': 2 * (ndays - 1), 'failed': 2} and r['status'] == 'published_partial'
    empty = tmp_path / 'empty'; empty.mkdir(); pub = store.published()['batch_id']
    with pytest.raises(RuntimeError, match = '没有任何证券'): mn.import_minute(root, str(empty), top = 2, workers = 1)
    assert store.published()['batch_id'] == pub                                                                   # 失败不改已发布状态


# 重导入保护旧数据 ---------------------------------------------------------------
def zip_of(src, day): return Path(src) / f'{day:%Y}' / f'{day:%m}' / f'{day:%Y%m%d}.zip'


def rewrite(src, day, mutate):
    """读出当日压缩包，逐个成员交给 mutate(成员名, DataFrame) → DataFrame 或 None（删掉该成员），再写回"""
    p = zip_of(src, day)
    with zipfile.ZipFile(p) as z: files = {n: pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist()}
    out = {n: r.to_csv(index = False) for n, df in files.items() if (r := mutate(n, df)) is not None}
    p.unlink(); make_zip(p, out)


def bump(df): return df.assign(**{c: df[c] + 0.01 for c in ('open', 'high', 'low', 'close')})              # 价格整体加 1 分：对账不受影响，内容与旧记录不同


def day_rows(store, day, snapshot = None):
    b = store.load('bars_5m', snapshot = snapshot); return b[pd.to_datetime(b.bar_end).dt.date == day].sort_values(['instrument', 'bar_end']).reset_index(drop = True)


D5, D6, D7 = date(2022, 1, 5), date(2022, 1, 6), date(2022, 1, 7)


def test_reimport_with_only_the_last_day_refreshed_keeps_earlier_days(world):
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1)
    old = {d: day_rows(store, d) for d in (D5, D6, D7)}
    for d in (D5, D6): zip_of(src, d).unlink()                                    # 前两天压缩包丢了
    rewrite(src, D7, lambda n, df: bump(df))
    r = mn.import_minute(root, src, start = D5, end = D7, top = 2, workers = 1)
    assert r['status'] == 'published_partial'                                     # 有刷新失败：不宣称已完整更新
    assert r['coverage'] == {'planned': 6, 'new_validated': 2, 'reused_old': 4, 'missing': 0, 'invalidated': 0, 'published': 6, 'failed': 4}
    assert r['issues']['warn/refresh_failed_kept_old'] == 4 and r['issues']['warn/missing_day'] == 4
    for d in (D5, D6): pd.testing.assert_frame_equal(day_rows(store, d), old[d])   # 前两天原样保留
    new7 = day_rows(store, D7); assert (new7.close.round(2) == (old[D7].close + 0.01).round(2)).all()          # 第三天确实被替换
    b = store.load('bars_5m'); assert not b.duplicated(['bar_end', 'instrument']).any() and len(b) == 2 * 48 * ndays
    csv_ = pd.read_csv(root / 'minute_audits' / f'{r["batch_id"]}.issues.csv'); assert (csv_.rule == 'refresh_failed_kept_old').sum() == 4


def test_one_instrument_fails_on_the_same_day_only_the_other_is_replaced(world):
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1); old = day_rows(store, D5)
    rewrite(src, D5, lambda n, df: bump(df) if n.startswith('sh/') else df.assign(low = df.low + 5))       # 600001 正常、000002 的 OHLC 自相矛盾
    r = mn.import_minute(root, src, start = D5, end = D5, top = 2, workers = 1); now = day_rows(store, D5)
    a_old, a_new = old[old.instrument == '600001.SH'], now[now.instrument == '600001.SH']; b_old, b_new = old[old.instrument == '000002.SZ'], now[now.instrument == '000002.SZ']
    assert (a_new.close.round(2) == (a_old.close + 0.01).round(2)).all()
    pd.testing.assert_frame_equal(b_new.reset_index(drop = True), b_old.reset_index(drop = True))            # 失败的一只沿用旧记录
    iss = pd.read_csv(root / 'minute_audits' / f'{r["batch_id"]}.issues.csv')
    kept = iss[iss.rule == 'refresh_failed_kept_old']; assert kept.instrument.tolist() == ['000002.SZ'] and (iss.rule == 'excluded_ohlc_inconsistent').sum() == 1
    assert r['coverage']['new_validated'] == 1 and r['coverage']['reused_old'] == 1 and r['status'] == 'published_partial'


def test_evidence_that_old_data_is_wrong_is_quarantined_unlike_a_failed_refresh(world):
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1)
    daily = store.load('bars_1d', parts = ['2022']); hit = (daily.instrument == '000002.SZ') & (pd.to_datetime(daily.date).dt.date == D6)
    daily.loc[hit, ['is_trading', 'volume', 'amount']] = [False, 0.0, 0.0]                                       # 日线更正：D6 这天该证券停牌
    store.publish(store.write_batch({'bars_1d': {'2022': store.write_partition('bars_1d', '2022', daily)}}, 'daily correction'))
    rewrite(src, D6, lambda n, df: None if n.startswith('sz/') else df)          # D6 的 000002 没有文件，日线也说它停牌：旧记录被推翻
    rewrite(src, D7, lambda n, df: None if n.startswith('sz/') else df)          # D7 的 000002 文件缺失但日线仍有交易：只是刷新失败
    r = mn.import_minute(root, src, start = D6, end = D7, top = 2, workers = 1)
    assert day_rows(store, D6).instrument.unique().tolist() == ['600001.SH']                                      # 被推翻的旧记录移出
    assert sorted(day_rows(store, D7).instrument.unique()) == ['000002.SZ', '600001.SH']                            # 刷新失败的沿用旧记录
    assert r['coverage']['invalidated'] == 1 and r['coverage']['reused_old'] == 1 and r['issues']['warn/invalidated_old'] == 1 and r['issues']['warn/refresh_failed_kept_old'] == 1
    q = pd.read_parquet(root / 'minute_audits' / f'{r["batch_id"]}.quarantine.parquet'); assert len(q) == 48 and q.instrument.unique().tolist() == ['000002.SZ']
    src_ = store.load('minute_source'); assert not ((pd.to_datetime(src_.date).dt.date == D6) & (src_.instrument == '000002.SZ')).any()


def test_old_snapshots_are_untouched_by_later_imports(world):
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1); sid = store.snapshot('before')
    frozen = store.load('bars_5m', snapshot = sid); files = {v['file']: v['sha'] for v in store.state(sid)['tables']['bars_5m'].values()}
    rewrite(src, D7, lambda n, df: bump(df)); zip_of(src, D5).unlink()
    mn.import_minute(root, src, start = D5, end = D7, top = 2, workers = 1)
    pd.testing.assert_frame_equal(store.load('bars_5m', snapshot = sid), frozen)
    assert all((root / f).exists() for f in files) and not store.load('bars_5m').equals(frozen)                       # 已发布的变了，快照没变


def test_provenance_is_recorded_per_validated_stock_day(world):
    import hashlib
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1)
    p = store.load('minute_source'); assert len(p) == 2 * ndays and not p.duplicated(['date', 'instrument']).any()
    row = p[(pd.to_datetime(p.date).dt.date == D7) & (p.instrument == '000002.SZ')].iloc[0]
    assert row.archive_sha256 == hashlib.sha256(zip_of(src, D7).read_bytes()).hexdigest() and row.archive.endswith('2022/01/20220107.zip')
    assert (row.member, row.header, row.grouping, row.volume_scale, row.amount_scale, row.processing_version, row.bars) == ('sz/000002.csv', 'open,high,low,close,amount,volume', 'position', 1.0, 1.0, mn.PROCESSING_VERSION, 48)
    old_sha = dict(zip(p.date.astype(str) + p.instrument, p.archive_sha256))
    rewrite(src, D7, lambda n, df: bump(df)); zip_of(src, D6).unlink(); mn.import_minute(root, src, start = D6, end = D7, top = 2, workers = 1)
    q = store.load('minute_source'); new_sha = dict(zip(q.date.astype(str) + q.instrument, q.archive_sha256))
    assert new_sha[f'{D6}000002.SZ'] == old_sha[f'{D6}000002.SZ'] and new_sha[f'{D7}000002.SZ'] != old_sha[f'{D7}000002.SZ']    # 沿用的保留旧来源，重新取得的换成新来源


def test_identical_reimport_with_failures_is_no_change_but_leaves_a_record(world):
    root, src, insts, ndays = world; store = Store(root); mn.import_minute(root, src, top = 2, workers = 1); batch = store.published()['batch_id']
    zip_of(src, D7).unlink(); r = mn.import_minute(root, src, top = 2, workers = 1)
    assert r['status'] == 'no_change' and store.published()['batch_id'] == batch and r['coverage']['reused_old'] == 2 and r['coverage']['published'] == r['coverage']['planned']
    rec = json.loads((root / 'minute_audits' / f'{r["refresh_record"]}.json').read_text(encoding = 'utf-8')); assert rec['coverage']['failed'] == 2 and rec['published_batch'] == batch
