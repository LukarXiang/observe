"""数据更新：下载 → 留底 → 标准化 → 与已发布数据合并 → 写批次 → 审计 → 通过才发布（模块 10、决策 19）。
全程持 data-writer 锁；BaoStock 会话持 baostock 锁。"""
from datetime import date, datetime
import json

import pandas as pd

from . import raw, standardize as std
from .audit import audit_daily
from .locks import DATA_WRITER, operation_lock
from .sources.baostock import BaoStock
from .store import Store


def _d(x): return x if isinstance(x, date) else date.fromisoformat(str(x))


def _progress_path(root):
    p = Store(root).root / 'coverage' / 'daily_downloads.jsonl'; p.parent.mkdir(parents = True, exist_ok = True); return p


def _progress(root, record):
    with _progress_path(root).open('a', encoding = 'utf-8') as h:
        h.write(json.dumps({'at': datetime.now().isoformat(timespec = 'seconds'), **record}, ensure_ascii = False, default = str) + '\n')


def _last_progress(root):
    p = _progress_path(root); out = {}
    if not p.exists(): return out
    for line in p.read_text(encoding = 'utf-8').splitlines():
        try:
            r = json.loads(line); out[r['date']] = r
        except (KeyError, ValueError):
            continue
    return out


def _atomic_parquet(path, frame):
    path = path if hasattr(path, 'parent') else Store(path).root
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(f'.{path.name}.tmp')
    frame.to_parquet(tmp, index = False)
    pd.read_parquet(tmp)  # validate that the staged artifact is readable before commit
    tmp.replace(path)


def _merge(store, table, part, new, drop = None):
    """new 覆盖已发布分区中主键相同的行；drop(old) 先删掉要整体替换的行"""
    old = store.load(table, parts = [part])
    if len(old) and drop is not None: old = old[~drop(old)]
    return store.write_partition(table, part, pd.concat([old, new], ignore_index = True) if len(old) else new)


RULES = 'configs/rule_profiles/main_board.yaml'


def default_rules():
    from pathlib import Path
    from ..ledger.rules import RuleSet
    return RuleSet.from_yaml(RULES) if Path(RULES).exists() else None


def update_daily(root, start, end, source = None, tdx = None, factor_codes = (), force = False, rules = None):
    """下载 [start, end] 的交易日历、证券资料、全市场日线与当日复权因子变动；当期有除权的证券从通达信刷新公司行动。
    factor_codes：需要取全部复权因子历史的证券（首次初始化用）。返回摘要；审计有阻断问题时批次标为 rejected、不发布"""
    start, end = _d(start), _d(end); store = Store(root); parts, days_done, downloaded_days, missing_days, component_failures = {}, [], [], [], []
    with operation_lock(root, DATA_WRITER):
        src = source or BaoStock(root)
        pub = store.published()
        progress = _last_progress(root); staging = store.root / 'staging' / 'daily'; staging.mkdir(parents = True, exist_ok = True)
        with src.session():
            cal = std.calendar(src.calendar(start, end)); raw.save(root, 'baostock', 'calendar', f'{start}_{end}', cal)
            days = [d for d in cal[cal.is_open].date if start <= d <= end]
            # Published rows alone do not prove a complete day. Only a validated
            # checkpoint with both required components may be reused.
            todo = days
            inst_raw = src.stock_basic(); raw.save(root, 'baostock', 'stock_basic', str(end), inst_raw)
            bars, events = [], []
            for d in todo:
                key = str(d)
                staged = staging / f'{key}.parquet'
                one = None
                if not force and progress.get(key, {}).get('status') == 'success' and progress.get(key, {}).get('adj_status') in ('success', 'no_events') and staged.exists():
                    try: one = pd.read_parquet(staged)
                    except Exception: one = None
                    published = store.load('bars_1d', parts = [str(d.year)])
                    published_count = int((published.date == d).sum()) if len(published) else 0
                    if one is not None and {'date', 'instrument', 'open', 'high', 'low', 'close'}.issubset(one.columns) and len(one) and (not published_count or len(one) == published_count):
                        bars.append(one); days_done.append(d)
                        af = staging / f'{key}.adj.parquet'
                        try: a = pd.read_parquet(af)
                        except Exception: a = None
                        if a is not None: events.append(a); continue
                try:
                    published = store.load('bars_1d', parts = [str(d.year)])
                    published_count = int((published.date == d).sum()) if len(published) else 0
                    checkpoint_ok = not published_count or (one is not None and len(one) == published_count)
                    if not force and checkpoint_ok and progress.get(key, {}).get('status') == 'success' and staged.exists():
                        try: one = pd.read_parquet(staged)
                        except Exception: one = None
                        if one is None:
                            r = src.daily_market(d); downloaded_days.append(d); raw.save(root, 'baostock', 'daily_market', key, r)
                            one = std.daily(r) if len(r) else pd.DataFrame()
                            if len(one): _atomic_parquet(staged, one)
                    else:
                        r = src.daily_market(d); downloaded_days.append(d); raw.save(root, 'baostock', 'daily_market', key, r)
                        one = std.daily(r) if len(r) else pd.DataFrame()
                        if len(one): _atomic_parquet(staged, one)
                    status = 'success' if len(one) else 'unknown_empty'
                    if not len(one): missing_days.append(d)
                    adj_status = 'unknown_empty'
                    a = pd.DataFrame()
                    try:
                        a = src.adjust_factor_day(d); _atomic_parquet(staging / f'{key}.adj.parquet', a); adj_status = 'success' if len(a) else 'no_events'
                    except Exception as exc:
                        component_failures.append({'date': key, 'component': 'adjust_factor', 'error': str(exc)[:300]}); adj_status = 'failed'
                    _progress(root, {'date': key, 'requested_start': key, 'requested_end': key, 'actual_start': key if len(one) else None,
                                     'actual_end': key if len(one) else None, 'records': len(one), 'status': status, 'adj_status': adj_status,
                                     'evidence': 'daily_market_response'})
                    if len(one): bars.append(one); days_done.append(d)
                    if len(a): events.append(a)
                    continue
                except Exception as exc:
                    _progress(root, {'date': key, 'requested_start': key, 'requested_end': key, 'records': 0, 'status': 'failed', 'adj_status': 'failed', 'error': str(exc)[:300]})
                    raise
            full = [src.adjust_factor(c) for c in factor_codes]
        inst = std.instruments(inst_raw)
        parts['calendar'] = {'all': _merge(store, 'calendar', 'all', cal)}; parts['instruments'] = {'all': store.write_partition('instruments', 'all', inst)}
        new = pd.concat([b for b in bars if len(b)], ignore_index = True) if any(len(b) for b in bars) else pd.DataFrame(columns = ['date', 'instrument', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'is_trading', 'is_st', 'board'])
        if len(new):
            parts['bars_1d'] = {str(y): _merge(store, 'bars_1d', str(y), g) for y, g in new.groupby(new.date.map(lambda x: x.year))}
        adj = pd.concat([std.adj_factors(x) for x in events + full if len(x)], ignore_index = True) if any(len(x) for x in events + full) else pd.DataFrame()
        changed = sorted(set(adj.instrument)) if len(adj) else []
        if len(adj): parts['adj_factors'] = {'all': _merge(store, 'adj_factors', 'all', adj)}
        coverage_rows = []
        for code in factor_codes:
            coverage_rows.append({'instrument': std.instrument(code), 'status': 'complete', 'requested_start': '1990-01-01', 'requested_end': '2099-12-31', 'verified_through': str(end), 'has_start_basis': True, 'has_gap': False, 'source': 'baostock.full_adjust_factor', 'evidence': 'full_adjust_factor_query'})
        if changed:
            known = {r['instrument'] for r in coverage_rows}
            coverage_rows.extend({'instrument': i, 'status': 'partial', 'requested_start': str(start), 'requested_end': str(end), 'verified_through': str(end), 'has_start_basis': False, 'has_gap': True, 'source': 'baostock.daily_adjust_factor', 'evidence': 'daily_adjust_factor_query'} for i in changed if i not in known)
        if coverage_rows:
            parts['adj_coverage'] = {'all': _merge(store, 'adj_coverage', 'all', pd.DataFrame(coverage_rows))}
        if tdx is not None and changed:
            acts, refreshed = [], []
            for i in changed:
                try: x = tdx.xdxr(i.split('.')[0])
                except Exception as exc:
                    component_failures.append({'instrument': i, 'component': 'corp_actions', 'error': str(exc)[:300]}); continue
                if x is not None and len(x): acts.append(std.corp_actions(x, i)); refreshed.append(i)
            if acts:
                acts = pd.concat(acts, ignore_index = True)
                parts['corp_actions'] = {'all': _merge(store, 'corp_actions', 'all', acts, drop = lambda o: o.instrument.isin(refreshed))}
        issues = audit_daily(new, days, inst, rules or default_rules()) if days else pd.DataFrame(columns = ['level'])
        bid = store.write_batch(parts, note = f'daily {start}..{end}')
        if missing_days: issues = pd.concat([issues, pd.DataFrame([{'level': 'block', 'rule': 'missing_requested_day', 'date': d, 'instrument': None, 'detail': 'no validated daily artifact'} for d in missing_days])], ignore_index = True)
        if component_failures: issues = pd.concat([issues, pd.DataFrame([{'level': 'block', 'rule': 'component_failed', 'date': x.get('date'), 'instrument': x.get('instrument'), 'detail': x['error']} for x in component_failures])], ignore_index = True)
        blocked = issues[issues.level == 'block'] if len(issues) else issues
        if len(blocked): store.reject(bid, f'{len(blocked)} 条阻断级审计问题'); status = 'rejected'
        else: store.publish(bid); status = 'published'
        if len(issues): issues.to_csv(store.root / 'batches' / f'{bid}.issues.csv', index = False)
    return {'batch_id': bid, 'status': status, 'base': pub['batch_id'], 'days': len(downloaded_days), 'verified_days': len(days_done), 'missing_days': [str(x) for x in missing_days], 'component_failures': component_failures, 'rows': len(new), 'adj_events': len(adj),
            'issues': {f'{l}/{r}': int(n) for (l, r), n in issues.groupby(['level', 'rule']).size().items()} if len(issues) else {}}
