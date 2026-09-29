"""公开入口测试用的临时快照：真实 Store 写 Parquet 分区、发布、冻结，入口按快照编号读取。"""
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from observe.data.standardize import board
from observe.data.store import Store

D = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
A, B = '600001.SH', '600002.SH'
BIG = 1e9            # 足够大的真实历史成交额：参与度 5% 不构成约束


def bar(day, inst, px, pre = None, amount = BIG, trading = True, close = None):
    close = px if close is None else close
    if not trading: return {'date': day, 'instrument': inst, 'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan, 'preclose': pre if pre is not None else px,
                            'volume': 0.0, 'amount': 0.0, 'is_trading': False, 'is_st': False, 'board': board(inst)}
    return {'date': day, 'instrument': inst, 'open': px, 'high': max(px, close), 'low': min(px, close), 'close': close, 'preclose': px if pre is None else pre,
            'volume': amount / px, 'amount': amount, 'is_trading': True, 'is_st': False, 'board': board(inst)}


def flat(inst, px = 10.0, days = D, amount = BIG): return [bar(d, inst, px, amount = amount) for d in days]


def instruments(*insts, kind = 'stock'):
    return pd.DataFrame({'instrument': list(insts), 'name': list(insts), 'kind': kind, 'list_date': date(2010, 1, 4), 'delist_date': pd.NaT, 'listed': True,
                         'exchange': [i[-2:] for i in insts], 'board': [board(i) for i in insts]})


def action(inst, ex, cash = 0.0, bonus = 0.0, pay = None, listed = None):
    return {'instrument': inst, 'ex_date': ex, 'cash_per_share': cash, 'bonus_ratio': bonus, 'rights_ratio': 0.0, 'rights_price': 0.0,
            'record_date': None, 'pay_date': pay, 'bonus_list_date': listed, 'source': 'test'}


def snapshot(root, rows, inst = None, actions = (), sessions = D, adj = None, coverage = None):
    s = Store(root); b = pd.DataFrame(rows)
    cal = pd.DataFrame({'date': sessions, 'is_open': True})
    parts = {'calendar': {'all': s.write_partition('calendar', 'all', cal)},
             'bars_1d': {str(y): s.write_partition('bars_1d', str(y), g) for y, g in b.groupby(b.date.map(lambda x: x.year))},
             'instruments': {'all': s.write_partition('instruments', 'all', inst if inst is not None else instruments(*sorted(set(b.instrument))))}}
    if actions: parts['corp_actions'] = {'all': s.write_partition('corp_actions', 'all', pd.DataFrame(list(actions)))}
    if adj is not None: parts['adj_factors'] = {'all': s.write_partition('adj_factors', 'all', adj)}
    if coverage is not None: parts['adj_coverage'] = {'all': s.write_partition('adj_coverage', 'all', coverage)}
    s.publish(s.write_batch(parts)); return s.snapshot()


def tree_hash(path):
    """目录下全部文件的内容哈希：用于证明源实验目录没有被改动"""
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).rglob('*')) if p.is_file()}
