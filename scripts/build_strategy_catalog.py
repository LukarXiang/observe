"""只读扫描 repo/量化策略源代码（聚宽社区精选，未纳入 Git），生成策略目录 strategies/catalog.csv。

每个源文件一行：稳定编号、来源、标题、作者、原文链接、资产类别、调度频率、依赖数据、策略原型、重复关系。
分类全部来自源码正则，只是改写前的分诊，不代表策略能否复现；改写时人工核对后在 strategies/specs/ 里写定。
已有目录中的人工字段（status、spec、notes）按编号保留，不被重新扫描覆盖。
用法：python scripts/build_strategy_catalog.py [--source repo/量化策略源代码] [--out strategies/catalog.csv]"""
import argparse, hashlib, re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

COLLECTIONS = {'2020年度精选策略': 'y2020', '2021年度精选策略': 'y2021', '2022年度精选策略': 'y2022', '2023年度精选策略': 'y2023',
               '2024年度精选策略1': 'y2024a', '2024年度精选策略2': 'y2024b', '聚宽2025年精选': 'y2025'}
MANUAL = ('status', 'spec', 'notes')

# 依赖数据：命中即记录（与 observe 已发布表对照见 docs/08）
DATA = {
    'valuation': r'\bvaluation\.|market_cap|circulating_market_cap|turnover_ratio|pe_ratio|pb_ratio|ps_ratio|pcf_ratio',
    'financials': r'\b(income|balance|cash_flow|indicator)\.|get_history_fundamentals|finance\.run_query|\bSTK_\w+',
    'index_members': r'get_index_stocks|get_index_weights',
    'industry': r'get_industry|get_industries|get_industry_stocks|sw_l[123]|jq_l[12]|zjw',
    'concept': r'get_concept',
    'jq_factors': r'get_factor_values|jqfactor|alpha101|alpha191|get_all_factors',
    'money_flow': r'get_money_flow',
    'margin': r'get_mtss',
    'northbound': r'hk_hold|get_hk_hold|北向资金|沪股通|深股通',
    'billboard': r'get_billboard',
    'call_auction': r'get_call_auction',
    'ticks': r'get_ticks',
    'minute_bars': r"""(['"])(1m|5m|15m|30m|60m|minute)\1""",
    'macro': r'\bmacro\.|宏观',
    'dividend': r'STK_XR_XD|bonus_ratio|dividend|股息',
    'limit_price': r'high_limit|low_limit',
    'index_bars': r"""(['"])(000001|000300|000905|000852|000016|399001|399006|399101|399303|000985|932000)\.XSH[GE]\1""",
}
ASSETS = {
    # 不用 set_order_cost(type='fund' / 'futures')：聚宽模板的费率设置里普遍带它们，与是否交易该类资产无关
    'futures': r'CCFX|XSGE|XDCE|XZCE|XINE|GFEX|get_dominant_future|get_future_contracts',
    'option': r'get_option|opt\.run_query|OPT_\w+',
    'conv_bond': r'conbond|可转债|bond\.run_query',
    'etf': r"""(['"])(5[0-9]\d{4}|15\d{4}|16\d{4})\.XSH[GE]\1|get_all_securities\(\s*\[?\s*.(etf|fund|lof)""",
}
ARCHETYPES = [   # (名称, 标题正则, 代码正则)，顺序即优先级：先逐条匹配标题，都不中再逐条匹配代码；代码正则只用特征性写法，避免通用过滤函数误判
    ('futures', r'期货|商品|股指', r'CCFX|XSGE|XDCE|XZCE|XINE|get_dominant_future'),
    ('conv_bond', r'可转债|转债', r'conbond|bond\.run_query'),
    ('etf_rotation', r'ETF|etf|基金', r"""etf_pool|g\.etf|(['"])(5[01]\d{4}|15\d{4})\.XSH[GE]\1"""),
    ('limit_board', r'涨停|打板|首板|连板|一进二|弱转强|高开|低开|最高板|追.板|龙头|竞价', r'get_call_auction|g\.(yesterday_)?(hl|zt)_list|连板|首板'),
    ('pairs', r'配对|协整', r'coint\('),
    ('grid', r'网格', r'grid'),
    ('ml', r'机器学习|随机森林|SVM|LSTM|神经网络|强化学习|XGB|lightgbm|决策树', r'sklearn|lightgbm|xgboost|torch|tensorflow|keras'),
    ('small_cap', r'小市值|微盘|小盘|低价', r'(circulating_)?market_cap\.asc|order_by\(\s*valuation\.(circulating_)?market_cap\b(?!\.desc)'),
    ('value_dividend', r'股息|红利|价值|PEG|ROE|白马|估值|大市值|质量', r'dividend|roe|peg'),
    ('northbound', r'北上|北向|港资|外资|聪明钱', r'hk_hold|get_hk_hold'),
    ('sector_rotation', r'行业轮动|板块轮动|行业|板块', r'get_industry_stocks'),
    ('multi_factor', r'多因子|因子|alpha', r'get_factor_values|alpha191|alpha101'),
    ('index_timing', r'择时|RSRS|均线|MACD|KDJ|BIAS|DMI|趋势|海龟|突破', r'RSRS|rsrs'),
]
SCHEDULE = {'monthly': r'run_monthly', 'weekly': r'run_weekly', 'daily': r'run_daily', 'bar': r'def handle_data'}


def read(p):
    b = p.read_bytes()
    for enc in ('utf-8', 'gbk'):
        try: return b.decode(enc)
        except UnicodeDecodeError: pass
    return b.decode('utf-8', 'replace')


def header(text, key):
    m = re.search(rf'^#\s*{key}[：:]\s*(.+?)\s*$', text, re.M); return m.group(1) if m else ''


def code_only(text):
    """去掉注释与空白后用于判重，避免文章抬头不同的同一策略被当成两份"""
    return re.sub(r'\s+', '', re.sub(r'#.*', '', text))


def scan(source):
    rows = []
    for p in sorted(source.rglob('*')):
        if p.suffix not in ('.txt', '.py', '.ipynb') or p.parent.name not in COLLECTIONS: continue
        t = read(p); c = COLLECTIONS[p.parent.name]
        m = re.match(r'\s*(\d+)', p.stem); num = int(m.group(1)) if m else None
        hit = lambda rx: bool(re.search(rx, t, re.I | re.M))
        assets = [k for k, rx in ASSETS.items() if hit(rx)]
        title = header(t, '标题') or re.sub(r'^\s*\d+[.、\s]*', '', p.stem)
        arch = next((k for k, tr, _ in ARCHETYPES if re.search(tr, title, re.I)), None) or next((k for k, _, cr in ARCHETYPES if re.search(cr, t, re.I)), 'other')
        times = sorted(set(re.findall(r"""run_(?:daily|weekly|monthly)\([^)]*?time\s*=\s*['"]([^'"]+)['"]""", t)))
        rows.append({'collection': c, 'num': num, 'file': f'{p.parent.name}/{p.name}', 'title': title, 'author': header(t, '作者'),
                     'url': header(t, '克隆自聚宽文章'), 'lines': t.count('\n') + 1, 'backtestable': hit(r'def initialize\s*\('),
                     'assets': ','.join(['stock'] * (not {'futures', 'option', 'conv_bond'} & set(assets) or hit(r'XSHE|XSHG')) + assets),
                     'schedule': ','.join(k for k, rx in SCHEDULE.items() if hit(rx)), 'times': ','.join(times),
                     'data': ','.join(k for k, rx in DATA.items() if hit(rx)), 'archetype': arch,
                     'sha': hashlib.sha256(t.encode()).hexdigest()[:12], '_code': code_only(t)})
    df = pd.DataFrame(rows)
    df['id'] = [f'{c}-{n:03d}' if n is not None else f'{c}-x{k:02d}' for k, (c, n) in enumerate(zip(df.collection, df.num))]
    dup = df.id.duplicated(keep = False)                      # 同一合集内编号重复时追加序号
    df.loc[dup, 'id'] = df[dup].id + '-' + df[dup].groupby('id').cumcount().astype(str)
    df['duplicate_of'] = ''
    seen = []                                                 # (编号, 去注释代码)，按目录顺序第一次出现者为原件
    for k, r in df.iterrows():
        if not r.backtestable: continue
        for sid, code in seen:
            if abs(len(code) - len(r._code)) < 0.1 * max(len(code), 1) and SequenceMatcher(None, code, r._code, autojunk = False).quick_ratio() > 0.97 \
               and SequenceMatcher(None, code, r._code, autojunk = False).ratio() > 0.97:
                df.at[k, 'duplicate_of'] = sid; break
        else: seen.append((r.id, r._code))
    return df.drop(columns = ['_code', 'num'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', default = 'repo/量化策略源代码'); ap.add_argument('--out', default = 'strategies/catalog.csv'); a = ap.parse_args()
    df = scan(Path(a.source)); out = Path(a.out); out.parent.mkdir(parents = True, exist_ok = True)
    if out.exists():
        old = pd.read_csv(out, dtype = str, keep_default_na = False).set_index('id')
        for col in MANUAL: df[col] = df.id.map(old[col]) if col in old else ''
    for col in MANUAL:
        if col not in df: df[col] = ''
    df['status'] = df.status.mask(df.status.fillna('') == '', 'todo').fillna('todo')
    df[['notes', 'spec']] = df[['notes', 'spec']].fillna('')
    cols = ['id', 'collection', 'file', 'title', 'author', 'url', 'backtestable', 'duplicate_of', 'assets', 'archetype', 'schedule', 'times', 'data', 'lines', 'sha', *MANUAL]
    df[cols].to_csv(out, index = False, encoding = 'utf-8')
    b = df[df.backtestable & (df.duplicate_of == '')]
    print(f'{len(df)} 个文件，可回测 {df.backtestable.sum()}，去重后 {len(b)}；写入 {out}')
    print(b.archetype.value_counts().to_string())


if __name__ == '__main__': main()
