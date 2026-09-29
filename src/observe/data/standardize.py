"""标准化：统一代码（600000.SH）、单位（股、元、小数）、日期；停牌行的占位价格置空（模块 10）。"""
import numpy as np
import pandas as pd

MAIN = ('600', '601', '603', '605', '000', '001', '002', '003')   # 深市中小板 2021 年并入主板，涨跌幅规则一直与主板相同


def instrument(code):
    """sh.600000 / sz.000001 / bj.920000 / 600000 → 600000.SH"""
    code = str(code).strip(); ex, _, num = code.rpartition('.')
    if ex: return f'{num}.{ex.upper()}'
    return f"{code.zfill(6)}.{'SH' if code[0] in '569' and not code.startswith('92') else 'BJ' if code[0] in '48' or code.startswith('92') else 'SZ'}"


def to_baostock(inst):
    num, ex = inst.split('.'); return f'{ex.lower()}.{num}'


def board(inst):
    num, ex = inst.split('.')
    if ex == 'BJ': return 'bse'
    if num.startswith(MAIN): return 'main'
    if num.startswith(('300', '301', '302')): return 'gem'
    if num.startswith(('688', '689')): return 'star'
    return 'other'


def _num(s): return pd.to_numeric(s.replace('', np.nan), errors = 'coerce')


TRUE, FALSE = frozenset(('true', '1', 'yes', 't', 'y')), frozenset(('false', '0', 'no', 'f', 'n', ''))


def flag(x, missing = False):
    """明确解析布尔值：字符串 'False' 不能当成 True；缺失返回 missing；无法识别的值报错"""
    if x is None: return missing
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        if x != x: return missing
        if x in (0, 1): return bool(x)
        raise ValueError(f'cannot parse boolean {x!r}')
    if x is pd.NA or x is pd.NaT: return missing
    s = str(x).strip().lower()
    if s in TRUE: return True
    if s in FALSE: return False if s else missing
    if s in ('nan', 'none', '<na>'): return missing
    raise ValueError(f'cannot parse boolean {x!r}')


def daily(raw):
    """BaoStock 全市场日线（query_daily_history_k_AStock）→ bars_1d"""
    d = pd.DataFrame({'date': pd.to_datetime(raw['date']).dt.date, 'instrument': raw['code'].map(instrument)})
    for c in ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount'): d[c] = _num(raw[c].astype(str))
    d['turnover'] = _num(raw['turn'].astype(str)) / 100                                  # 百分数 → 小数
    d['is_trading'] = raw['tradestatus'].astype(str) == '1'; d['is_st'] = raw['isST'].astype(str) == '1'
    for src, dst in (('peTTM', 'pe_ttm'), ('pbMRQ', 'pb_mrq')): d[dst] = _num(raw[src].astype(str)) if src in raw else np.nan
    d.loc[~d.is_trading, ['open', 'high', 'low', 'close']] = np.nan                     # 停牌行的开高低收是占位，不是成交价
    d.loc[~d.is_trading, ['volume', 'amount']] = d.loc[~d.is_trading, ['volume', 'amount']].fillna(0)
    d['board'] = d.instrument.map(board)
    return d.sort_values(['date', 'instrument']).reset_index(drop = True)


def market_input(bars, adjusted=None, coverage=None):
    """研究视图（因子、标签）使用的市场字段；原始行情不会冒充复权行情。账本行情只由 observe.execution 生成。"""
    out = bars.copy()
    if adjusted is not None:
        from .prices import with_adjusted
        out = with_adjusted(out, adjusted, coverage=coverage)
    rename = {'open_adj': 'adj_open', 'high_adj': 'adj_high', 'low_adj': 'adj_low', 'close_adj': 'adj_close'}
    for src, dst in rename.items():
        if src in out and dst not in out: out[dst] = out[src]
    if 'suspended' not in out and 'is_trading' in out: out['suspended'] = ~out['is_trading'].astype(bool)
    return out


def calendar(raw):
    return pd.DataFrame({'date': pd.to_datetime(raw['calendar_date']).dt.date, 'is_open': raw['is_trading_day'].astype(str) == '1'}).sort_values('date').reset_index(drop = True)


def instruments(raw):
    kind = raw['type'].astype(str).map({'1': 'stock', '2': 'index'})
    d = pd.DataFrame({'instrument': raw['code'].map(instrument), 'name': raw['code_name'], 'kind': kind,
                      'list_date': pd.to_datetime(raw['ipoDate'].replace('', None)).dt.date, 'delist_date': pd.to_datetime(raw['outDate'].replace('', None)).dt.date,
                      'listed': raw['status'].astype(str) == '1'})
    d = d[d.kind.notna()].copy(); d['exchange'] = d.instrument.str[-2:]; d['board'] = np.where(d.kind == 'stock', d.instrument.map(board), 'index')
    return d.sort_values('instrument').reset_index(drop = True)


def adj_factors(raw):
    """BaoStock 复权因子 → 自 ex_date 起生效的后复权因子"""
    return pd.DataFrame({'instrument': raw['code'].map(instrument), 'ex_date': pd.to_datetime(raw['dividOperateDate']).dt.date,
                         'back_factor': _num(raw['backAdjustFactor'].astype(str))}).drop_duplicates(['instrument', 'ex_date'], keep = 'last').sort_values(['instrument', 'ex_date']).reset_index(drop = True)


def corp_actions(raw, inst):
    """通达信除权除息（category 1）→ corp_actions；每 10 股的数值换成每股。到账日、红股上市日另由 BaoStock 补，缺失时账本按规则推断"""
    x = raw[raw['category'].astype(int) == 1]
    ex = pd.to_datetime(dict(year = x['year'].astype(int), month = x['month'].astype(int), day = x['day'].astype(int))).dt.date
    f = lambda c: pd.to_numeric(x[c], errors = 'coerce').fillna(0).to_numpy()
    d = pd.DataFrame({'instrument': inst, 'ex_date': ex.to_numpy(), 'cash_per_share': f('fenhong') / 10, 'bonus_ratio': f('songzhuangu') / 10,
                      'rights_ratio': f('peigu') / 10, 'rights_price': f('peigujia'), 'record_date': None, 'pay_date': None, 'bonus_list_date': None, 'source': 'tdx'})
    return d.sort_values('ex_date').reset_index(drop = True)


def shares(raw, inst):
    """通达信股本变化记录 → shares（万股 → 股）"""
    x = raw[raw['houzongguben'].notna()]
    dt = pd.to_datetime(dict(year = x['year'].astype(int), month = x['month'].astype(int), day = x['day'].astype(int))).dt.date
    return pd.DataFrame({'instrument': inst, 'date': dt.to_numpy(), 'total_shares': pd.to_numeric(x['houzongguben']).to_numpy() * 1e4,
                         'float_shares': pd.to_numeric(x['panhouliutong']).to_numpy() * 1e4}).drop_duplicates(['instrument', 'date'], keep = 'last')
