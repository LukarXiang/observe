"""只读检查：已发布的 bars_5m 与「计划处理的证券日」和历史审计问题清单是否对得上，并与指定快照逐分区比较。
不写数据表，不修复；报告写到 data/minute_audits/publish_check-<时间>.json。用法：python scripts/check_minute_publish.py --root data [--snapshot <id>]"""
import argparse, json
from datetime import datetime
from pathlib import Path

import pandas as pd

from observe.data.minute import build_universe, plan_days
from observe.data.store import Store


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default = 'data'); ap.add_argument('--snapshot'); ap.add_argument('--top', type = int, default = 800); a = ap.parse_args()
    root = Path(a.root); store = Store(root); pub = store.published()
    bars = store.load('bars_1d', columns = ['date', 'instrument', 'volume', 'amount', 'is_trading', 'board']); bars['date'] = pd.to_datetime(bars.date).dt.date
    cal = store.load('calendar'); sessions = [d for d in pd.to_datetime(cal[cal.is_open].date).dt.date]
    uni = build_universe(bars, top = a.top); pools = {int(y): set(g.instrument) for y, g in uni.groupby('year')}; plan = plan_days(sessions, set(pools))
    exp = set()
    for d, g in bars[bars.date.isin(plan)].groupby('date'):
        want = set().union(*(pools[y] for y in plan[d])); exp |= {(d, i) for i in g[g.is_trading.astype(bool) & g.instrument.isin(want)].instrument}
    b = store.load('bars_5m', columns = ['bar_end', 'instrument']); b['d'] = pd.to_datetime(b.bar_end).dt.date
    have = set(zip(b.d, b.instrument)); per_day = b.groupby(['d', 'instrument']).size()
    explained, missing_days = set(), set()
    for f in sorted((root / 'minute_audits').glob('*.issues.csv')):
        x = pd.read_csv(f)
        if not len(x): continue
        missing_days |= set(pd.to_datetime(x[x.instrument.isna() & (x.rule == 'missing_day')].date).dt.date)          # 旧版整日缺失只记一条、没有证券
        x = x[x.instrument.notna() & x.rule.str.startswith(('excluded_', 'missing_', 'unreadable_'))]; explained |= set(zip(pd.to_datetime(x.date).dt.date, x.instrument))
    short = exp - have; explained |= {k for k in short if k[0] in missing_days}; unexplained = short - explained
    report = {'checked_at': datetime.now().isoformat(timespec = 'seconds'), 'published_batch': pub['batch_id'], 'processing': 'read-only',
              'planned_stock_days': len(exp), 'present_stock_days': len(have & exp), 'extra_stock_days_not_planned': len(have - exp), 'short_stock_days': len(short),
              'short_explained_by_audit_issues': len(short & explained), 'short_unexplained': len(unexplained), 'unexplained_by_date': {str(d): int(n) for d, n in pd.Series([k[0] for k in unexplained]).value_counts().sort_index().items()} if unexplained else {}, 'unexplained_sample': [[str(d), i] for d, i in sorted(unexplained)[:10]],
              'incomplete_stock_days_not_48_bars': int((per_day != 48).sum()), 'has_minute_source_table': 'minute_source' in pub['tables']}
    if a.snapshot:
        snap = store.state(a.snapshot)['tables'].get('bars_5m', {}); cur = pub['tables'].get('bars_5m', {})
        report['snapshot'] = {'id': a.snapshot, 'partitions': len(snap), 'same_as_published': sorted(k for k in snap if cur.get(k, {}).get('sha') == snap[k]['sha']) == sorted(snap) and set(snap) == set(cur),
                              'differing_partitions': sorted(k for k in set(snap) | set(cur) if snap.get(k, {}).get('sha') != cur.get(k, {}).get('sha'))}
    out = root / 'minute_audits' / f'publish_check-{datetime.now():%Y%m%d-%H%M%S}.json'; out.write_text(json.dumps(report, ensure_ascii = False, indent = 1), encoding = 'utf-8')
    print(json.dumps(report, ensure_ascii = False, indent = 1)); print(out)


if __name__ == '__main__': main()
