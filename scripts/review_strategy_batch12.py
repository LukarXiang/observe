"""Preserve complete reviews and bounded probes without publishing replacement data."""
import argparse
from contextlib import redirect_stdout
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import canonical, file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import read, protect
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-152153-eb05'
SOURCES = (
    '2020年度精选策略/91 Stoch（KDJ）——大盘择时.txt',
    '2021年度精选策略/65.【分享】多因子指数增强.txt',
    '2021年度精选策略/86.北上詹姆斯.txt',
    '2022年度精选策略/56.胜率为100%，盈亏比90000+的单品种倍投策略.txt',
    '2022年度精选策略/58.100%胜率！历史PE估值价值投资择时交易.txt',
    '2022年度精选策略/68.RSRS模型深入研究3-二八轮动策略及其探究分析.txt',
    '2022年度精选策略/76.向云帆致敬—— 宏观数据择时策略——极速版代码.txt',
    '2023年度精选策略/10.一个简单而持续稳定的懒人超额收益策略.txt',
    '2023年度精选策略/22.简单到发指的股指策略(2013到现在收益10949%).txt',
    '2020年度精选策略/90 MA均线金叉买入，死叉卖出.txt',
    '2024年度精选策略2/31.《趋势永存》持续16年跑赢大盘的真正靠谱策略.txt',
    '2024年度精选策略2/5.随机森林策略，低换手率，年化近50%.txt',
    '2022年度精选策略/24.龙头低波市值平均轮动策略.txt',
    '2024年度精选策略1/41.人工智能强化学习DQN交易智能体（回馈社区公开训练代码）.txt',
)
PROBES = ('bs_399300', 'bs_399006', 'bs_000001', 'market_pe_sh', 'market_pe_sz', 'macro_money', 'macro_pmi', 'northbound_flow')
API_NAMES = ('stock_market_pe_lg', 'macro_china_money_supply', 'macro_china_pmi', 'stock_hsgt_hist_em')
DEPENDENCIES = ('result_df.csv', 'test_predict_300_q.pkl', 'tgt_net.pt', 'money_supply_05-19.csv',
                'aggretate_signal_data_02_19.csv', 'cpi_ppi_0501_1902.csv', 'shibor.csv',
                'huilv_060101_190331.csv', 'guozhai_1m_10y_06_19.csv', 'qiyezhai_1m_06_19.csv')


def review(root, directory):
    require((directory / 'baseline.json').is_file(), 'Establish the batch checkpoint first')
    latest = read(Path(root) / 'catalog/strategies/latest.json')
    old = {r['path']: r for r in read(Path(latest['directory']) / 'catalog.json')}
    for name in SOURCES:
        require(file_sha(Path('repo/量化策略源代码') / name) == old[name]['bytes_sha256'], 'Original source differs from catalog')
        require(old[name]['review_status'] != '人工审查完成', 'Source already reviewed')
    target = directory / 'source-reviews'; target.mkdir(exist_ok = False)
    records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; text, encoding = read_source(source)
        copied = target / f'{strategy_id(name)}.source'
        with copied.open('xb') as stream: stream.write(source.read_bytes())
        records.append({'strategy_id': strategy_id(name), 'source_path': str(source), 'source_sha256': file_sha(source),
                        'source_copy': str(copied), 'source_copy_sha256': file_sha(copied), 'encoding': encoding,
                        'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    search = subprocess.run(['rg', '--files', '--hidden', 'repo', 'data', *[item for name in DEPENDENCIES for item in ('-g', name)]],
                            capture_output = True, text = True, check = False)
    require(search.returncode in (0, 1), search.stderr)
    store = Store(root); state = store.state(SNAPSHOT)
    pool = REVIEWS[SOURCES[12]]['rules']['pool']
    bars = store.load_state(state, 'bars_1d', columns = ['date', 'instrument', 'is_trading'], filters = [('instrument', 'in', pool)])
    minute = store.load_state(state, 'bars_5m', columns = ['bar_end', 'instrument'], filters = [('instrument', 'in', ['000002.SZ'])])
    import akshare as ak
    apis = [{'name': name, 'signature': str(inspect.signature(getattr(ak, name))), 'source': inspect.getsource(getattr(ak, name))} for name in API_NAMES]
    write_json(directory / 'existing-apis.json', apis)
    write_json(target / 'review.json', {'sources': records, 'snapshot': SNAPSHOT, 'published_batch': state['batch_id'],
               'dependency_search': {'argv': search.args, 'returncode': search.returncode, 'matches': search.stdout.splitlines(), 'pickle_executed': False},
               'fixed_pool_bars': bars.groupby('instrument').agg(rows = ('date', 'size'), trading = ('is_trading', 'sum'),
                   first = ('date', 'min'), last = ('date', 'max')).reset_index().to_dict('records'),
               'vanke_5m': {'rows': len(minute), 'first': str(minute.bar_end.min()) if len(minute) else None,
                            'last': str(minute.bar_end.max()) if len(minute) else None, 'not_equivalent_to_original_1m': True},
               'api_evidence_sha256': file_sha(directory / 'existing-apis.json'), 'strategy_results': [],
               'variant_confirmation': 'Low-volatility seven-stock execution correction pending; other existing questions remain pending'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    progress = archive(root, '第十二批14份完整源码审查与依赖查找已归档；固定七股低波策略待确认，指数/宏观/模型缺口保持显式')
    return {'status': 'ok', 'reviewed': len(records), 'catalog': catalog, 'progress': progress['output']}


def worker(root, directory, name):
    folder = directory / 'probes' / name; folder.mkdir(parents = True, exist_ok = False)
    socket.setdefaulttimeout(15)
    result = {'endpoint': name, 'status': 'failed', 'wire_responses': [], 'strict_usable': False, 'published': False}
    original = requests.sessions.Session.request

    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'Bounded probe request limit reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15)
        response = original(session, method, url, **kwargs)
        saved = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with saved.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(saved), 'sha256': file_sha(saved)})
        return response

    try:
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request):
            if name.startswith('bs_'):
                code = {'bs_399300': 'sz.399300', 'bs_399006': 'sz.399006', 'bs_000001': 'sh.000001'}[name]
                result['parameters'] = {'code': code, 'start': '2005-01-05', 'end': '2026-09-29', 'adjustflag': '3'}
                with BaoStock(root).session() as source: frame = source.index_daily(code, '2005-01-05', '2026-09-29')
            else:
                import akshare as ak
                if name.startswith('market_pe_'): frame = ak.stock_market_pe_lg(symbol = '上证' if name.endswith('sh') else '深证')
                elif name == 'macro_money': frame = ak.macro_china_money_supply()
                elif name == 'macro_pmi': frame = ak.macro_china_pmi()
                else: frame = ak.stock_hsgt_hist_em(symbol = '北向资金')
        path = raw.save(root, 'dependency_probe_batch12', name, directory.name, frame)
        result.update(status = 'success' if len(frame) else 'empty', rows = len(frame), columns = list(frame.columns),
                      raw_file = str(path), raw_sha256 = file_sha(path),
                      sample = canonical(frame.head(3).astype(object).where(frame.head(3).notna(), None).to_dict('records')))
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    finally: write_json(folder / 'result.json', result)
    return result


def probe(root, directory):
    output = directory / 'probe-results.json'
    if output.exists() or (directory / 'probes').exists(): raise FileExistsError(output)
    records = []
    for name in PROBES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch12', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', name]
        try:
            process = subprocess.run(command, capture_output = True, text = True, timeout = 75)
            record = {'endpoint': name, 'returncode': process.returncode, 'stdout': process.stdout[-2000:], 'stderr': process.stderr[-2000:]}
        except subprocess.TimeoutExpired: record = {'endpoint': name, 'status': 'timeout', 'timeout_seconds': 75}
        result = directory / 'probes' / name / 'result.json'
        record['evidence_files'] = [{'file': str(p), 'sha256': file_sha(p)} for p in sorted(result.parent.glob('*')) if p.is_file()]
        record['result'] = read(result) if result.exists() else {'status': 'timeout' if record.get('status') == 'timeout' else 'failed', 'error': 'Worker did not produce a terminal result; partial evidence retained'}
        records.append(record)
        print(json.dumps({'endpoint': name, 'status': record['result']['status'], 'rows': record['result'].get('rows')}), flush = True)
        archive(root, f'第十二批{name}接口探测：{record["result"]["status"]}，新增证据独立归档，未发布替代旧输入')
    write_json(output, {'results': records, 'published': False, 'snapshot': SNAPSHOT,
               'limitations': ['New current retrievals are not original CSV/weights/model versions',
                              'Historical index equivalence and macro publication vintages remain unproven',
                              'Northbound net buying is not proof of quota difference or per-stock share ratios']})
    return {'status': 'archived', 'output': str(output), 'probes': len(records)}


def analyze(root, directory):
    output = directory / 'input-analysis.json'
    if output.exists(): raise FileExistsError(output)
    results = read(directory / 'probe-results.json')['results']
    snapshot = Store(root).state(SNAPSHOT); frozen = Store(root).load_state(snapshot, 'index_1d')
    frozen['date'] = pd.to_datetime(frozen.date).dt.date
    profiles = []
    for record in results:
        result = record['result']; item = {'endpoint': record['endpoint'], 'status': result['status']}
        if result.get('raw_file'):
            path = Path(result['raw_file']); require(file_sha(path) == result['raw_sha256'], 'Raw probe changed')
            frame = pd.read_parquet(path)
            column = next((c for c in ('date', '日期', '月份') if c in frame), None)
            item.update(raw_file = str(path), raw_sha256 = file_sha(path), columns = list(frame), rows = len(frame))
            if column and len(frame):
                labels = frame[column].astype(str)
                dates = pd.to_datetime(labels.str.replace('年', '-', regex = False).str.replace('月份', '-01', regex = False), errors = 'coerce')
                known = dates.dropna(); item.update(date_column = column, unparsed_dates = int(dates.isna().sum()),
                    first = str(known.min().date()) if len(known) else None, last = str(known.max().date()) if len(known) else None,
                    duplicate_dates = int(known.duplicated().sum()), distinct_months = int(known.dt.to_period('M').nunique()),
                    maximum_rows_per_month = int(known.dt.to_period('M').value_counts().max()) if len(known) else 0)
            if record['endpoint'].startswith('bs_') and len(frame):
                prices = frame[['open', 'high', 'low', 'close', 'preclose']].apply(pd.to_numeric, errors = 'coerce')
                item['nonfinite_price_cells'] = int((~np.isfinite(prices.to_numpy(float))).sum())
                item['nonpositive_price_cells'] = int((prices <= 0).sum().sum())
            if record['endpoint'] == 'bs_399300' and len(frame):
                frame['date'] = pd.to_datetime(frame.date).dt.date
                joined = frozen[frozen['index'].eq('000300.SH')].merge(frame, on = 'date', validate = 'one_to_one', suffixes = ('_frozen', '_new'))
                fields = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount')
                differences = {c: int((joined[f'{c}_frozen'].astype(float).to_numpy() != joined[f'{c}_new'].astype(float).to_numpy()).sum()) for c in fields}
                item['snapshot_000300_comparison'] = {'common_days': len(joined), 'frozen_days': len(frozen[frozen['index'].eq('000300.SH')]),
                    'exact_differences_by_field': differences, 'scope': 'These vendor rows only; no original-platform equivalence or tradability proof'}
        profiles.append(item)
    source = Path('data/staging/strategies-batch7/20261005-rsi-slots/joinquant-api-response.json')
    copied = directory / 'cached-official-api.bin'
    with copied.open('xb') as stream: stream.write(source.read_bytes())
    text = BeautifulSoup(read(source)['data'], 'html.parser').get_text('\n', strip = True)
    start = text.index("get_bars(security, count"); end = text.index('get_current_tick', start)
    excerpt = directory / 'get-bars-official-excerpt.txt'
    with excerpt.open('x', encoding = 'utf-8') as stream: stream.write(text[start:end])
    result = {'profiles': profiles, 'snapshot': SNAPSHOT, 'published': False,
              'official_get_bars': {'original_file': str(source), 'copy': str(copied), 'sha256': file_sha(copied),
                  'excerpt': str(excerpt), 'excerpt_sha256': file_sha(excerpt),
                  'finding': 'get_bars excludes suspended bars, does not fill missing bars; include_now defaults False; backtest adjustment basis defaults to current date'},
              'limitations': ['Retrieval coverage alone does not establish disclosure time or historical revision versions',
                             'Monthly market PE cannot silently replace 2500 daily exchange observations',
                             'Index data do not make the original index orders tradable']}
    write_json(output, result)
    archive(root, '第十二批指数字段对比、市场PE频率与get_bars停牌规则已独立存档，未发布或替代冻结输入')
    return {'status': 'ok', 'output': str(output), 'profiles': profiles}


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch12-verification.json')
    if output.exists(): raise FileExistsError(output)
    reviews = read(directory / 'source-reviews/review.json')
    require(file_sha(directory / 'existing-apis.json') == reviews['api_evidence_sha256'], 'API evidence changed')
    for record in reviews['sources']:
        require(file_sha(Path(record['source_path'])) == record['source_sha256'] == file_sha(Path(record['source_copy'])), 'Reviewed source changed')
    probes = read(directory / 'probe-results.json')
    require(len(probes['results']) == len(PROBES) and {r['endpoint'] for r in probes['results']} == set(PROBES), 'Incomplete probes')
    for record in probes['results']:
        for item in record['evidence_files']: require(file_sha(Path(item['file'])) == item['sha256'], 'Probe evidence changed')
        payload = record['result']
        if payload.get('raw_file'): require(file_sha(Path(payload['raw_file'])) == payload['raw_sha256'], 'Probe table changed')
    checks = read(directory / 'checks.json')
    required = {str(Path(__file__).resolve()), str(Path('src/observe/strategy_catalog.py').resolve())}
    require(required.issubset({str(Path(name).resolve()) for name in checks['implementation_sha256']}), 'Required checked code missing')
    require(bool(checks['commands']) and all(r['returncode'] == 0 for r in checks['commands']), 'Batch checks failed')
    repository = Path.cwd().resolve()
    checked_files = []
    for name, sha in checks['implementation_sha256'].items():
        source = Path(name).resolve()
        try: relative = source.relative_to(repository)
        except ValueError as exc: raise ValueError(f'Checked implementation outside repository: {name}') from exc
        require(file_sha(source) == sha, 'Checked implementation changed')
        require(relative not in {r for _, r in checked_files}, 'Duplicate checked implementation')
        checked_files.append((source, relative))
    analysis = read(directory / 'input-analysis.json')
    documentation = analysis['official_get_bars']
    require(file_sha(Path(documentation['copy'])) == documentation['sha256'] == file_sha(Path(documentation['original_file'])), 'Official cached input changed')
    require(file_sha(Path(documentation['excerpt'])) == documentation['excerpt_sha256'], 'Official excerpt changed')
    require(Store(root).published()['batch_id'] == read(directory / 'baseline.json')['published']['batch_id'], 'Published inputs changed')
    protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok = False)
    for source, relative in checked_files:
        target = frozen / relative; target.parent.mkdir(parents = True, exist_ok = True)
        with target.open('xb') as stream: stream.write(source.read_bytes())
    result = {'status': 'ok_with_deferred_strategies', 'snapshot': SNAPSHOT, 'source_reviews': reviews,
              'probes': probes, 'input_analysis': analysis, 'checks': checks, 'protection': protection,
              'scope': 'Fourteen source reviews and bounded input probes; no new strategy backtest yet',
              'progress': archive(root, '第十二批14份审查与8接口探测验收；原来源和旧资产保留，七股低波修正待确认，全库继续')}
    write_json(output, result)
    return {'status': result['status'], 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['review', 'probe', 'worker', 'analyze', 'finish'])
    parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', type = Path, default = Path('data/staging/strategies-batch12/20261005-daily-candidates'))
    parser.add_argument('--endpoint', choices = PROBES); args = parser.parse_args()
    if args.action == 'worker':
        if not args.endpoint: parser.error('--endpoint is required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = {'review': review, 'probe': probe, 'analyze': analyze, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii = False), flush = True)
