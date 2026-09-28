"""阶段 1 验收核对（真实数据）：在市证券数 vs BaoStock 证券清单、日线 vs 新浪抽样逐项比对、审计问题汇总、后复权连续性。
用法：D:\\envs\\quant\\.venv\\Scripts\\python.exe scripts/verify_stage1.py [--root data] [--sample 50]"""
import argparse, json, os, random, time

import pandas as pd

from observe.data import standardize as std
from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store

os.environ['NO_PROXY'] = '*'; os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)   # 国内站点不走本机代理（决策 12）


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default = 'data'); ap.add_argument('--sample', type = int, default = 50); a = ap.parse_args()
    s = Store(a.root); bars = s.load('bars_1d'); out = {'batch_id': s.published()['batch_id'], 'days': int(bars.date.nunique()), 'rows': len(bars)}
    days = sorted(bars.date.unique()); check = [days[0], days[-1]]
    with BaoStock(a.root).session() as bs:                                   # 1. 在市证券数 vs 当日证券清单
        for d in check:
            lst = bs._rows('query_all_stock', {'day': d}, lambda: bs.bs.query_all_stock(str(d)))
            stocks = {std.instrument(c) for c in lst.code if std.board(std.instrument(c)) != 'other' and not c.startswith(('sh.000', 'sz.399'))}
            have = set(bars[bars.date == d].instrument)
            out[f'listing_{d}'] = {'bars': len(have), 'list_stocks': len(stocks), 'only_in_bars': sorted(have - stocks)[:10], 'only_in_list': sorted(stocks - have)[:10]}
    import akshare as ak                                                     # 2. 日线 vs 新浪抽样
    random.seed(7); trade = bars[bars.is_trading]; insts = random.sample(sorted(trade.instrument.unique()), a.sample); rows = []
    for i in insts:
        try: sina = ak.stock_zh_a_daily(symbol = i.split('.')[1].lower() + i.split('.')[0], start_date = str(days[0]).replace('-', ''), end_date = str(days[-1]).replace('-', ''))
        except Exception as e: rows.append({'instrument': i, 'error': str(e)[:80]}); continue   # noqa: BLE001
        sina['date'] = pd.to_datetime(sina.date).dt.date; mine = trade[trade.instrument == i]
        m = mine.merge(sina, on = 'date', suffixes = ('', '_sina')).sample(n = min(4, len(mine)), random_state = 1)
        for r in m.itertuples():
            rows.append({'instrument': i, 'date': r.date, **{c: abs(getattr(r, c) - getattr(r, f'{c}_sina')) < 0.0051 for c in ('open', 'high', 'low', 'close')},
                         'volume': abs(r.volume - r.volume_sina) <= 100})
        time.sleep(0.5)
    cmp = pd.DataFrame(rows); ok = cmp.dropna(subset = ['open']) if 'open' in cmp else cmp
    out['sina_pairs'] = len(ok); out['sina_equal_pct'] = {c: round(float(ok[c].mean() * 100), 2) for c in ('open', 'high', 'low', 'close', 'volume')} if len(ok) else {}
    out['sina_mismatch'] = ok[~ok[['open', 'high', 'low', 'close']].all(axis = 1)].head(10).to_dict('records') if len(ok) else []
    out['sina_errors'] = int(cmp['error'].notna().sum()) if 'error' in cmp else 0
    p = s.root / 'batches' / f"{out['batch_id']}.issues.csv"                  # 3. 审计问题
    out['issues'] = pd.read_csv(p).groupby(['level', 'rule']).size().to_dict() if p.exists() else {}
    adj = s.load('adj_factors'); full = adj.groupby('instrument').ex_date.min(); full = full[full < pd.Timestamp('2010-01-01').date()].index   # 4. 后复权连续性
    v = with_adjusted(bars[bars.instrument.isin(full)], adj); out['adj_instruments'] = len(full)
    out['adj_max_abs_daily_ret'] = float(v.ret.abs().max()) if len(v) else None; out['adj_missing_factor_rows'] = int(v.back_factor.isna().sum())
    print(json.dumps(out, ensure_ascii = False, indent = 1, default = str))


if __name__ == '__main__': main()
