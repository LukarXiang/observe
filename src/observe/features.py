"""研究特征（模块 12）：因子集登记、研究视图宽表、因子长表。

研究视图只用后复权价（data.prices.with_adjusted），停牌日在日历上占位、值为缺失；时间序列算子对全部在市证券计算，
横截面算子只在当天研究候选上算，输出只保留研究候选行。"""
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from .factors import parse
from .factors.expr import compute

FIELDS = ('open_adj', 'high_adj', 'low_adj', 'close_adj', 'ret', 'volume', 'amount', 'turnover')


def load_factor_set(path):
    """读取并校验因子集：名字唯一、公式可解析、方向为 ±1、有经济含义说明。返回 {'text', 'sha256', 'min_obs_ratio', 'factors': [...]}"""
    data = Path(path).read_bytes(); text = data.decode('utf-8'); y = yaml.safe_load(text) or {}   # 哈希按文件字节算，与冻结副本的校验一致
    specs, seen = [], set()
    for f in y.get('factors', []):
        unknown = set(f) - {'name', 'expr', 'direction', 'desc'}
        if unknown: raise ValueError(f"因子 {f.get('name')} 有未知字段 {sorted(unknown)}")
        if f['name'] in seen: raise ValueError(f"因子名重复：{f['name']}")
        if f.get('direction') not in (1, -1): raise ValueError(f"因子 {f['name']} 的 direction 必须是 1 或 -1")
        if not str(f.get('desc', '')).strip(): raise ValueError(f"因子 {f['name']} 缺少经济含义说明")
        p = parse(f['expr']); seen.add(f['name'])
        specs.append({**f, 'lookback': p.lookback, 'fields': sorted(p.fields)})
    if not specs: raise ValueError('因子集为空')
    return {'text': text, 'sha256': hashlib.sha256(data).hexdigest(), 'version': y.get('version'), 'min_obs_ratio': float(y.get('min_obs_ratio', 1.0)), 'factors': specs}


def panel(view, days, instruments):
    """研究视图长表 → {字段: 宽表(交易日 × 证券)}；停牌日的量、额、换手置为缺失，不当作 0 进入窗口"""
    v = view[view.instrument.isin(set(instruments))].copy(); v['date'] = pd.to_datetime(v.date).dt.date
    idx, cols = pd.Index(days), sorted(set(instruments))
    trading = v.pivot(index = 'date', columns = 'instrument', values = 'is_trading').reindex(index = idx, columns = cols).fillna(False).astype(bool)
    out = {}
    for f in FIELDS:
        w = v.pivot(index = 'date', columns = 'instrument', values = f).reindex(index = idx, columns = cols).astype(float) if f in v else pd.DataFrame(float('nan'), index = idx, columns = cols)
        out[f] = w.where(trading)
    return out


def factor_frame(fset, wide, eligible):
    """eligible：宽表（交易日 × 证券）布尔；返回研究候选行的长表 date / instrument / 各因子（原始值，未做横截面预处理）"""
    cols = {}
    for f in fset['factors']:
        v = compute(f['expr'], wide, eligible = eligible, min_obs_ratio = fset['min_obs_ratio']).where(eligible)
        cols[f['name']] = v.stack(future_stack = True)
    out = pd.DataFrame(cols); keep = eligible.stack(future_stack = True).reindex(out.index).fillna(False).astype(bool)
    out = out[keep.to_numpy()].reset_index(); out.columns = ['date', 'instrument', *cols]
    return out
