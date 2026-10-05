"""全量策略来源登记；静态依赖是审查线索，未经审查的文本命中不冒充有效策略。"""
import ast, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .data.store import _atomic_json
from .runs import file_sha

CATALOG_VERSION = 'strategy_catalog_v1'
API_NAMES = ('get_fundamentals', 'get_index_stocks', 'get_industry_stocks', 'get_industry', 'get_ticks', 'get_call_auction',
             'get_price', 'attribute_history', 'history', 'get_current_data', 'get_extras', 'get_factor_values', 'get_all_securities',
             'run_daily', 'run_weekly', 'run_monthly', 'order_target_value', 'order_target', 'order_value', 'order', 'get_future_contracts')
BP_SOURCE = '2022年度精选策略/74.基本面单因子测试——以BP因子为例.txt'
MA_SOURCE = '2020年度精选策略/66 高于MA10买入低于MA20卖出回测半年收益率20.4%.txt'
MICRO_SOURCE = '2024年度精选策略1/53.微盘400每日再平衡.txt'
SMALL_VALUE_SOURCE = '2022年度精选策略/100.实战型小盘价值策略--14年30倍.txt'
REVIEWS = {
    BP_SOURCE: {'review_status': '人工审查完成', 'scope': 'BP/EP 组件及主板近似变体', 'fidelity': '数据或执行受限的近似复现',
                'implementations': ['bp_component_v1', 'ep_component_v1'], 'status': '实现中',
                'rules': {'factors': ['1/pb_ratio', '1/pe_ratio'], 'universe': '中证800历史成分；前21交易日不停牌；剔除ST',
                          'rebalance': '每月首个交易日交易', 'weights': '分位名单等额目标；持仓内所有证券再平衡',
                          'parameters': {'shift': 21, 'precent': 0.10, 'index': '000906.XSHG', 'quantile': [0, 10]}},
                'gaps': ['缺中证800历史成分', '平台估值口径未证明与 BaoStock PE_TTM/PB_MRQ 一致', '平台排序/分位边界和历史成本未等价移植'],
                'differences': ['只复现倒数因子组件，单独登记主板股票池变体', '固定前10%正估值股票，保留负值因子但不纳入此探索组合',
                                '收盘决策后次日开盘执行；月末收盘下单以在下月首交易日执行', '使用现有原始价账本、分日期费率与成交限制']},
    MA_SOURCE: {'review_status': '人工审查完成', 'scope': '美的集团日频 MA10/MA20 规则', 'fidelity': '数据或执行受限的近似复现',
                'implementations': ['ma10_ma20_v1'], 'status': '实现中',
                'rules': {'instrument': '000333.SZ', 'buy': '前一日收盘 > 1.01 * 前10条日线收盘均值，花可用现金',
                          'sell': '前一日收盘 < 前20条日线收盘均值，清仓', 'else': '保持状态',
                          'parameters': {'short': 10, 'long': 20, 'buy_multiplier': 1.01}},
                'gaps': ['handle_data 的平台运行频率未在源码声明', 'attribute_history 默认跳过停牌，当前固定交易日窗口不会跳过'],
                'differences': ['明确按日频收盘决策、次日开盘执行', '用后复权价计算均线，原始价执行',
                                '目标权重在中性区保持上次状态，整手、费用与参与度可能留现金', '交易日窗口满额才生成新状态；缺数据保持已有状态']},
    MICRO_SOURCE: {'review_status': '人工审查完成', 'scope': '微盘400原始日内调仓规则及数据依赖', 'fidelity': '尚未实现，缺必要历史数据',
                   'implementations': [], 'status': '待数据',
                   'rules': {'universe': '当日在市股票；排除4/8/68前缀、ST和名称含ST/*/退',
                             'selection': '按总市值升序取400只', 'weights': '每只目标金额为总资产/400；差额从小到大下单',
                             'rebalance': '每天09:30', 'parameters': {'stock_num': 400, 'benchmark': '399303.XSHE'}},
                   'gaps': ['缺已核实的历史总股本变更与生效日，不能用年度控制变量市值或季度股本无限延续',
                            '缺历史证券名称/退市标签及平台当日在市证券集合等价证明', '09:30查询与调仓须明确数据可用时刻和执行近似'],
                   'next_step': '先核实股本事件、单位、生效日与可用时刻；有证据再发布shares和单独登记日频近似版本'},
    SMALL_VALUE_SOURCE: {'review_status': '人工审查完成', 'scope': '小盘价值100原始排名及再平衡规则', 'fidelity': '尚未实现，缺必要历史数据',
                         'implementations': [], 'status': '待数据',
                         'rules': {'universe': '上证综指与深证综指历史成分并集，PB/PE/PCF均为正且不缺失',
                                   'selection': '总市值排名 + (PB排名 + PE排名 + PCF排名)/3，升序取100只',
                                   'weights': '目标等权1/100；移出清仓，持仓偏离目标超过10%才再平衡',
                                   'rebalance': '每月第一次before_trading_start选股；handle_data交易',
                                   'parameters': {'choicenum': 100, 'weight_deviation': .1, 'benchmark': '399300.XSHE'}},
                         'gaps': ['缺历史总市值/股本事件', '缺已发布且口径核实的PCF', '缺000001/399106历史成分',
                                  '平台handle_data频率、注入函数和after_code_changed初始化生命周期待核实'],
                         'next_step': '核实股本、市现率及股票池证据后分别移植排名组件；不以当前中证800名单代替原股票池'},
}


def read_source(path):
    data = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gb18030', 'utf-16'):
        try: return data.decode(encoding).replace('\r\n', '\n').replace('\r', '\n'), encoding
        except UnicodeError: pass
    raise ValueError(f'源码编码无法无损识别：{path}')


def strategy_id(relative): return 'S-' + hashlib.sha256(relative.encode()).hexdigest()[:12]


def catalog_strategies(root, source):
    base = Path(source); records = []
    proof_path = Path(root) / 'catalog/strategies/implementation-evidence.json'
    implementation_evidence = json.loads(proof_path.read_text(encoding = 'utf-8')) if proof_path.exists() else []
    files = sorted(f for f in base.rglob('*') if f.suffix.lower() in ('.txt', '.py', '.ipynb'))
    if not files: raise ValueError('策略源目录没有 TXT/PY/IPYNB')
    hashes, variants = defaultdict(list), defaultdict(list)
    for file in files:
        relative = file.relative_to(base).as_posix(); sid = strategy_id(relative); decode_error = None
        try: text, encoding = read_source(file)
        except ValueError as exc: text, encoding, decode_error = '', None, str(exc)
        original_text = text; code = text
        if file.suffix.lower() == '.ipynb':
            try: code = '\n\n'.join(''.join(c.get('source', [])) for c in json.loads(text)['cells'] if c.get('cell_type') == 'code')
            except (KeyError, ValueError, TypeError) as exc: decode_error = f'Notebook 无法解析：{exc}'
        starts = [m.start() for m in re.finditer(r'(?m)^(?:import |from \w|def |class )', code)]
        if starts: code = code[min(starts):]
        syntax_error = None; calls = None
        try:
            tree = ast.parse(code); calls = sorted({n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))})
        except SyntaxError as exc: syntax_error = f'{exc.lineno}: {exc.msg}'
        apis = [api for api in API_NAMES if re.search(r'\b' + api + r'\s*\(', code)]
        imports = sorted(set(re.findall(r'(?m)^\s*(?:from|import)\s+([\w.]+)', code)))
        fin = sorted(set(re.findall(r'\b(?:valuation|indicator|income|balance|cash_flow)\.\w+', code)))
        static_freq = sorted(set(re.findall(r"(?:frequency\s*=\s*|unit\s*=\s*|,\s*)['\"](\d+[dm]|daily|minute)['\"]", code)))
        intraday = any(api in apis for api in ('get_ticks', 'get_call_auction')) or any(f.endswith('m') or f == 'minute' for f in static_freq)
        clock = sorted(set(re.findall(r"(?:time\s*=\s*)['\"]([^'\"]+)['\"]", code)))
        assets = []
        if re.search(r"['\"](?:51\d{4}|15\d{4}).(?:XSHG|XSHE)['\"]", code): assets.append('ETF/基金候选')
        if 'get_future_contracts' in apis or re.search(r'\b(?:futures|future|options)\b', code): assets.append('衍生品候选')
        if re.search(r"['\"](?:60\d{4}|00\d{4}|30\d{4}|68\d{4}).(?:XSHG|XSHE)['\"]", code) or fin: assets.append('股票候选')
        if not assets: assets = ['待人工识别']
        gaps = []
        if fin or 'get_fundamentals' in apis: gaps.append('历史财务版本/公告时间或每日市值/股本待核实')
        if 'get_index_stocks' in apis: gaps.append('历史指数成分')
        if any(a in apis for a in ('get_industry_stocks', 'get_industry')): gaps.append('历史行业')
        if intraday: gaps.append('盘中事件/撮合与对应分钟或Tick股票池')
        if 'ETF/基金候选' in assets: gaps.append('ETF行情/公司行动/执行规则')
        if '衍生品候选' in assets: gaps.append('专用资产行情与交易规则')
        status = '待数据' if gaps else '待审查'
        if intraday and any(a in apis for a in ('get_ticks', 'get_call_auction')): status = '暂不可复现'
        research_only = calls is not None and not any(a in calls for a in ('order', 'order_value', 'order_target', 'order_target_value', 'order_target_percent')) and not any(a.startswith('run_') for a in calls)
        if research_only: status = '非交易研究脚本'
        content_sha = hashlib.sha256(original_text.encode()).hexdigest() if not decode_error else file_sha(file)
        normalized = '\n'.join(re.sub(r'\s+', ' ', line.strip()) for line in code.splitlines() if line.strip() and not line.lstrip().startswith('#'))
        candidate_sha = hashlib.sha256(normalized.encode()).hexdigest()
        urls = re.findall(r'https?://[^\s\]\)]+', text)
        item = {'strategy_id': sid, 'path': relative, 'bytes_sha256': file_sha(file), 'content_sha256': content_sha, 'encoding': encoding,
                'article_urls': list(dict.fromkeys(urls)), 'title': file.stem, 'code_start_line': original_text[:min(starts)].count('\n') + 1 if starts else 1,
                'code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'syntax_error': syntax_error, 'decode_error': decode_error,
                'asset_scope': assets, 'frequencies': static_freq, 'scheduled_times': clock, 'apis': apis, 'api_detection': 'text_candidates_including_comments',
                'imports': imports, 'financial_fields': fin, 'called_functions': calls, 'status': status, 'fidelity': '未审查',
                'rules': None, 'future_information_risks': ['历史成分/行业须按时点取数'] if any(a in apis for a in ('get_index_stocks', 'get_industry')) else [],
                'gaps': gaps, 'next_step': '按依赖核实后人工阅读全部逻辑；未知 API 禁止空结果占位', 'review_status': '静态扫描，尚未人工审查',
                'duplicate_of': None, 'variant_candidates': [], 'implementations': []}
        if relative in REVIEWS: item.update(REVIEWS[relative])
        if implementation_evidence:
            proof = [e for e in implementation_evidence if e['strategy_id'] == sid and e['source_sha256'] == item['bytes_sha256']]
            if proof:
                item.update(status = '已验证', implementation_evidence = proof, original_strategy_complete = False,
                            implementations = sorted(set(item['implementations']) | {e['implementation'] for e in proof}),
                            next_step = '已有组件/近似版本通过冻结快照复现；原版缺口仍按 gaps 逐批推进')
                dated = [e for e in proof if e.get('historical_constituents')]
                if dated:
                    item.update(scope = '；'.join(dict.fromkeys(e['scope'] for e in proof)), historical_constituents = dated[-1]['historical_constituents'],
                                gaps = [g for g in item['gaps'] if g != '缺中证800历史成分'] + ['供应商成分按周更新，调整公告和临时调整的日精度未证明'])
        records.append(item); hashes[content_sha].append(sid); variants[candidate_sha].append(sid)
    by_id = {r['strategy_id']: r for r in records}
    for ids in hashes.values():
        for sid in ids[1:]: by_id[sid].update(duplicate_of = ids[0], status = '重复版本', next_step = '共用实现但保留文件映射；核对 canonical_source 状态')
    for ids in variants.values():
        if len(ids) > 1:
            for sid in ids: by_id[sid]['variant_candidates'] = [i for i in ids if i != sid and by_id[i]['content_sha256'] != by_id[sid]['content_sha256']]
    catalog_sha = hashlib.sha256(json.dumps(records, sort_keys = True, ensure_ascii = False).encode()).hexdigest()[:20]
    dest = Path(root) / 'catalog' / 'strategies' / catalog_sha; dest.mkdir(parents = True, exist_ok = True)
    summary = {'catalog_version': CATALOG_VERSION, 'catalog_id': catalog_sha, 'source_root': str(base.resolve()), 'files': len(records), 'distinct_contents': len(hashes),
               'status_counts': dict(Counter(r['status'] for r in records)), 'suffix_counts': dict(Counter(f.suffix.lower() for f in files)),
               'encoding_counts': dict(Counter(r['encoding'] for r in records)), 'api_file_counts': {a: sum(a in r['apis'] for r in records) for a in API_NAMES},
               'reviewed_sources': [r['strategy_id'] for r in records if r['review_status'] == '人工审查完成'],
               'caveat': '未审查文件的依赖/分类为静态候选；近似变体不会自动合并，不代表全量策略已实现'}
    _atomic_json(dest / 'catalog.json', records); _atomic_json(dest / 'summary.json', summary)
    pd.DataFrame([{k: json.dumps(v, ensure_ascii = False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in records]).to_parquet(dest / 'catalog.parquet', index = False)
    _atomic_json(Path(root) / 'catalog' / 'strategies' / 'latest.json', {'catalog_id': catalog_sha, 'directory': str(dest), 'sha256': file_sha(dest / 'catalog.json')})
    return {**summary, 'output': str(dest)}
