"""质量审计：阻断级问题存在时批次不发布（模块 10）。"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

GEM_20 = date(2020, 8, 24)   # 创业板涨跌幅由 10% 改为 20%


def _fallback_limit(board, is_st, day):
    if board == 'main': return 0.05 if is_st else 0.10
    if board == 'gem': return 0.20 if day >= GEM_20 else (0.05 if is_st else 0.10)
    return {'star': 0.20, 'bse': 0.30}.get(board, np.nan)


def issue(level, rule, day = None, inst = None, detail = ''): return {'level': level, 'rule': rule, 'date': day, 'instrument': inst, 'detail': detail}


def _limit_fn(rules):
    """涨跌幅优先取执行规则集（与账本同一份），规则集没有覆盖的板块退回内置值"""
    def f(board, is_st, day):
        if rules is not None:
            try: return rules.on(day, board, bool(is_st)).limit_pct
            except ValueError: pass
        return _fallback_limit(board, is_st, day)
    return f


def audit_daily(bars, trading_days, instruments = None, rules = None):
    """bars: 标准化后的 bars_1d（可只含本批次的日子）；trading_days: 本批次应有的交易日；rules: 执行规则集（涨跌幅来源）"""
    if not len(bars):
        return pd.DataFrame([issue('block', 'empty_day', d, detail = '全市场 0 行') for d in sorted(trading_days)], columns = ['level', 'rule', 'date', 'instrument', 'detail'])
    out = []; t = bars[bars.is_trading]; s = bars[~bars.is_trading]
    count = bars.groupby('date').size().reindex(sorted(trading_days), fill_value = 0)
    out += [issue('block', 'empty_day', d, detail = '全市场 0 行') for d, n in count.items() if n == 0]
    px = t[['open', 'high', 'low', 'close']]
    out += [issue('block', 'bad_price', r.date, r.instrument) for r in t[(px.isna() | (px <= 0)).any(axis = 1)].itertuples()]
    bad = t[(t.low > t[['open', 'close']].min(axis = 1) + 1e-9) | (t.high < t[['open', 'close']].max(axis = 1) - 1e-9)]
    out += [issue('block', 'ohlc_order', r.date, r.instrument, f'o{r.open} h{r.high} l{r.low} c{r.close}') for r in bad.itertuples()]
    out += [issue('block', 'suspended_has_price', r.date, r.instrument) for r in s[s[['open', 'high', 'low', 'close']].notna().any(axis = 1)].itertuples()]
    jump = count[count > 0].pct_change().abs()
    out += [issue('warn', 'count_jump', d, detail = f'{v:.1%}') for d, v in jump.items() if v > 0.05]
    fresh = np.zeros(len(t), bool)                                            # 上市前 5 个交易日（含首日）另有涨跌幅规则，不查
    if instruments is not None and 'list_date' in instruments.columns:
        cal = np.array(sorted(trading_days), dtype = object); ld = t.instrument.map(instruments.set_index('instrument').list_date)
        l = ld.fillna(date.min).to_numpy(); pos = np.searchsorted(cal, t.date.to_numpy())
        lpos = np.where(l < cal[0], -10 ** 9, np.searchsorted(cal, l))              # 窗口之前就上市的不算新股
        fresh = ld.notna().to_numpy() & (pos - lpos < 5)
    lim_of = _limit_fn(rules); cache = {}
    chg = (t.close / t.preclose - 1).abs(); lim = [cache.setdefault((b, st, d), lim_of(b, st, d)) for b, st, d in zip(t.board, t.is_st, t.date)]
    over = t[(chg > np.asarray(lim) + 0.005).to_numpy() & ~fresh]
    out += [issue('warn', 'beyond_limit', r.date, r.instrument, f'{r.close / r.preclose - 1:+.2%}') for r in over.itertuples()]
    v = t[t.volume > 0]; vwap = v.amount / v.volume
    out += [issue('warn', 'vwap_outside', r.date, r.instrument, f'{w:.3f}') for r, w in zip(v.itertuples(), vwap) if not (r.low * 0.99 <= w <= r.high * 1.01)]
    return pd.DataFrame(out, columns = ['level', 'rule', 'date', 'instrument', 'detail'])


def audit_status(root, batch_id, rules):
    """某批次最近一次审计的状态，页面与运行前检查共用。
    rules：当前执行规则集；为 None（加载失败）时不能确认指纹，状态是 rules_unavailable，不会被当成通过。
    只审计了增量区间的通过记为 passed_incremental，不能显示成全快照通过。"""
    path = Path(root) / 'audits'; found = []
    for m in path.glob('*.json'):
        try:
            info = json.loads(m.read_text(encoding = 'utf-8'))
            if info.get('batch_id') == batch_id and info.get('audit_id'): found.append(info)
        except (OSError, ValueError): continue
    info = max(found, key = lambda x: x.get('audited_at', '')) if found else None
    csv = path / f"{info['audit_id']}.issues.csv" if info else None
    complete = bool(info and csv.exists() and info.get('status') in ('passed', 'problem'))
    rows = pd.read_csv(csv) if complete else pd.DataFrame()
    if not info:
        status = 'not_audited'
        try:
            old = json.loads((path / 'latest.json').read_text(encoding = 'utf-8'))
            if old.get('batch_id') != batch_id: status = 'expired'
            elif old.get('audit_id') and not ((path / f"{old['audit_id']}.json").exists() and (path / f"{old['audit_id']}.issues.csv").exists()): status = 'problem'
        except (OSError, ValueError): pass
    elif not complete: status = 'problem'
    elif rules is None: status = 'rules_unavailable'
    elif info.get('rule_fingerprint') != rules.config_fingerprint(): status = 'expired'
    elif len(rows): status = 'problem'
    else: status = 'passed' if info.get('scope') == 'snapshot' else 'passed_incremental'
    return {'source': f"audit {info['audit_id']}" if info else f'batch {batch_id}', 'batch_id': batch_id, 'status': status, 'rows': rows,
            'audit_id': info.get('audit_id') if info else None, 'scope': info.get('scope') if info else None,
            'input_range': info.get('input_range') if info else None, 'audited_at': info.get('audited_at') if info else None}
