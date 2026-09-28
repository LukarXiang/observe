"""数据更新：下载 → 留底 → 标准化 → 与已发布数据合并 → 写批次 → 审计 → 通过才发布（模块 10、决策 19）。
全程持 data-writer 锁；BaoStock 会话持 baostock 锁。"""
from datetime import date

import pandas as pd

from . import raw, standardize as std
from .audit import audit_daily
from .locks import DATA_WRITER, operation_lock
from .sources.baostock import BaoStock
from .store import Store


def _d(x): return x if isinstance(x, date) else date.fromisoformat(str(x))


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
    start, end = _d(start), _d(end); store = Store(root); parts, days_done = {}, []
    with operation_lock(root, DATA_WRITER):
        src = source or BaoStock(root)
        pub = store.published()
        with src.session():
            cal = std.calendar(src.calendar(start, end)); raw.save(root, 'baostock', 'calendar', f'{start}_{end}', cal)
            days = [d for d in cal[cal.is_open].date if start <= d <= end]
            have = set() if force else set(store.load('bars_1d', parts = sorted({str(d.year) for d in days}), columns = ['date']).get('date', []))
            todo = [d for d in days if d not in have]
            inst_raw = src.stock_basic(); raw.save(root, 'baostock', 'stock_basic', str(end), inst_raw)
            bars, events = [], []
            for d in todo:
                r = src.daily_market(d); raw.save(root, 'baostock', 'daily_market', str(d), r); bars.append(std.daily(r) if len(r) else pd.DataFrame())
                a = src.adjust_factor_day(d); events.append(a); days_done.append(d)
            full = [src.adjust_factor(c) for c in factor_codes]
        inst = std.instruments(inst_raw)
        parts['calendar'] = {'all': _merge(store, 'calendar', 'all', cal)}; parts['instruments'] = {'all': store.write_partition('instruments', 'all', inst)}
        new = pd.concat([b for b in bars if len(b)], ignore_index = True) if any(len(b) for b in bars) else pd.DataFrame(columns = ['date'])
        if len(new):
            parts['bars_1d'] = {str(y): _merge(store, 'bars_1d', str(y), g) for y, g in new.groupby(new.date.map(lambda x: x.year))}
        adj = pd.concat([std.adj_factors(x) for x in events + full if len(x)], ignore_index = True) if any(len(x) for x in events + full) else pd.DataFrame()
        changed = sorted(set(adj.instrument)) if len(adj) else []
        if len(adj): parts['adj_factors'] = {'all': _merge(store, 'adj_factors', 'all', adj)}
        if tdx is not None and changed:
            acts, refreshed = [], []
            for i in changed:
                try: x = tdx.xdxr(i.split('.')[0])
                except Exception: continue
                if x is not None and len(x): acts.append(std.corp_actions(x, i)); refreshed.append(i)
            if acts:
                acts = pd.concat(acts, ignore_index = True)
                parts['corp_actions'] = {'all': _merge(store, 'corp_actions', 'all', acts, drop = lambda o: o.instrument.isin(refreshed))}
        issues = audit_daily(new, days_done, inst, rules or default_rules()) if days_done else pd.DataFrame(columns = ['level'])
        bid = store.write_batch(parts, note = f'daily {start}..{end}')
        blocked = issues[issues.level == 'block'] if len(issues) else issues
        if len(blocked): store.reject(bid, f'{len(blocked)} 条阻断级审计问题'); status = 'rejected'
        else: store.publish(bid); status = 'published'
        if len(issues): issues.to_csv(store.root / 'batches' / f'{bid}.issues.csv', index = False)
    return {'batch_id': bid, 'status': status, 'base': pub['batch_id'], 'days': len(days_done), 'rows': len(new), 'adj_events': len(adj),
            'issues': {f'{l}/{r}': int(n) for (l, r), n in issues.groupby(['level', 'rule']).size().items()} if len(issues) else {}}
