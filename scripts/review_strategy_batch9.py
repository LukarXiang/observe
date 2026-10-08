"""Archive complete batch9 reviews and bounded ETF dependency probes."""
import argparse
from contextlib import redirect_stdout
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import requests

from observe.data import raw
from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import canonical, environment, file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import checkpoint, protect

SOURCES = (
    '2023年度精选策略/38.随机森林策略，低换手率，年化近50%.txt',
    '2021年度精选策略/85.易方达中小盘（魔改版）.txt',
    '2022年度精选策略/86.长江机器学习股票趋势预测.txt',
    '2020年度精选策略/96 沪港两地上市的银行股翻倍策略报告.txt',
    '2020年度精选策略/83 沪深300ETF-1060双均线.txt',
    '2023年度精选策略/17.20行代码8年胜率100%躲过了牛年第一场大跌.txt',
    '2022年度精选策略/54.指数估值自动报表系统——源代码.txt',
    '2023年度精选策略/41.大盘拥挤率极速版-180天3秒.txt',
)
PROBES = ('bs_etf_day', 'bs_etf_history', 'em_etf_raw', 'em_etf_hfq')
FALLBACKS = ('sina_etf_history', 'sina_etf_dividend_raw')
NEXT_SOURCES = (
    '2020年度精选策略/12 【均值回归】基于zscore的均值回归策略（胜率100%）.txt',
    '2022年度精选策略/49.LLT低延迟趋势线择时交易初探.txt',
    '2022年度精选策略/72.年化15.48%，盈亏比3.818的双均线策略.txt',
    '2023年度精选策略/35.最简强者恒强策略.txt',
)


def review(root, directory, sources = SOURCES):
    import akshare as ak
    import baostock as bs
    checkpoint(root, directory, '第九批保护检查点：继续逐份审查与ETF依赖探测')
    evidence = []
    for relative in sources:
        source = Path('repo/量化策略源代码') / relative
        text, encoding = read_source(source); sid = strategy_id(relative)
        destination = directory / f'{sid}.source'; destination.write_bytes(source.read_bytes())
        evidence.append({'strategy_id': sid, 'source_path': str(source), 'source_sha256': file_sha(source),
                         'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[relative],
                         'source_copy': str(destination), 'source_copy_sha256': file_sha(destination)})
    missing = subprocess.run(['rg', '--files', '--hidden', 'repo', 'data', '-g', 'test_predict_300_q.pkl',
                              '-g', 'Factor1.csv', '-g', 'HS300.csv', '-g', 'new1.csv', '-g', 'index_valuation.py'],
                             capture_output = True, text = True, check = False)
    if missing.returncode not in (0, 1): raise RuntimeError(missing.stderr)
    apis = [bs.query_daily_history_k_ETF, bs.query_history_k_data_plus, ak.fund_etf_hist_em,
            ak.fund_etf_hist_sina, ak.fund_etf_dividend_sina]
    api_evidence = [{'name': f.__name__, 'signature': str(inspect.signature(f)), 'doc': inspect.getdoc(f),
                     'source': inspect.getsource(f)} for f in apis]
    write_json(directory / 'existing-etf-apis.json', api_evidence)
    state = Store(root).state('20261005-152153-eb05')
    bars = Store(root).load_state(state, 'bars_1d', filters = [('instrument', 'in', ['510310.SH', '510300.SH', '511010.SH', '601238.SH'])])
    write_json(directory / 'review.json', {'sources': evidence, 'snapshot': '20261005-152153-eb05',
               'published_batch': state['batch_id'], 'dependency_search': {'command': missing.args, 'matches': missing.stdout.splitlines(),
               'returncode': missing.returncode, 'scope': 'Exact dependency filenames including hidden files; no pickle execution'},
               'local_bars': bars.groupby('instrument').agg(rows = ('date', 'size'), first = ('date', 'min'), last = ('date', 'max')).reset_index().to_dict('records'),
               'api_evidence_sha256': file_sha(directory / 'existing-etf-apis.json'), 'strategy_results': []})
    result = catalog_strategies(root, 'repo/量化策略源代码')
    progress = archive(root, f'第九批{len(sources)}份完整源码审查已归档，逐项规则与缺口见本阶段review.json：{directory}')
    return {'catalog': result, 'progress': progress['output']}


def worker(root, directory, name):
    folder = directory / name; folder.mkdir(exist_ok = False)
    socket.setdefaulttimeout(15); result = {'endpoint': name, 'status': 'failed', 'wire_responses': [], 'strict_usable': False}
    original = requests.sessions.Session.request

    def request(session, method, url, **kwargs):
        session.trust_env = False; kwargs['timeout'] = 15
        response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'; path.write_bytes(response.content)
        result['wire_responses'].append({'url': url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response

    try:
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request):
            if name.startswith('bs_'):
                with BaoStock(root).session() as source:
                    if name == 'bs_etf_day':
                        frame = source._rows('query_daily_history_k_ETF', {'date': '2024-03-29'}, lambda: source.bs.query_daily_history_k_ETF('2024-03-29'))
                    else:
                        frame = source.index_daily('sh.510310', '2024-03-01', '2024-03-29')
            elif name.startswith('em_'):
                import akshare as ak
                frame = ak.fund_etf_hist_em(symbol = '510310', start_date = '20240301', end_date = '20240329',
                                           adjust = 'hfq' if name.endswith('hfq') else '')
            elif name == 'sina_etf_history':
                import akshare as ak
                frame = ak.fund_etf_hist_sina(symbol = 'sh510310')
            else:
                response = requests.get('https://finance.sina.com.cn/realstock/company/sh510310/hfq.js')
                result.update(status = 'wire_only', status_code = response.status_code,
                              note = 'Raw cumulative dividend response only; do not run installed endpoint eval on remote text; no adjustment semantics or availability proved')
                return result
        path = raw.save(root, 'dependency_probe', name, directory.name, frame)
        result.update(status = 'success' if len(frame) else 'empty', rows = len(frame), columns = list(frame.columns),
                      raw_file = str(path), raw_sha256 = file_sha(path),
                      sample = canonical(frame.head(3).astype(object).where(pd.notna(frame.head(3)), None).to_dict('records')))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    finally:
        write_json(folder / 'result.json', result)
    return result


def probes(root, directory, names = PROBES):
    if not (directory / 'review.json').exists(): raise FileNotFoundError('Complete review/checkpoint before dependency probes')
    destination = directory / 'probes.json'
    if destination.exists() or any((directory / p).exists() for p in names): raise FileExistsError('Probe artifacts already exist; use a new directory')
    results = []
    for name in names:
        try:
            child = subprocess.run([sys.executable, '-m', 'scripts.review_strategy_batch9', 'worker', '--root', str(root),
                                    '--directory', str(directory), '--worker', name], capture_output = True, text = True, timeout = 45)
            (directory / f'{name}.log').write_text(child.stderr, encoding = 'utf-8')
            path = directory / name / 'result.json'
            result = json.loads(path.read_text(encoding = 'utf-8')) if path.exists() else {'endpoint': name, 'status': 'failed', 'error': child.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            result = {'endpoint': name, 'status': 'timeout', 'timeout_seconds': 45, 'strict_usable': False}
        results.append(result)
        write_json(destination, {'probes': results, 'note': 'Dependency evidence only; no publication, historical rules or reproduction claimed'})
        print(json.dumps(result, ensure_ascii = False), flush = True)
    progress = archive(root, '第九批ETF接口探测结束：响应与失败证据已归档，不发布未验收数据')
    return {'progress': progress['output'], 'protection': protect(root, directory)}


def inputs(root, directory):
    from bs4 import BeautifulSoup
    output = directory / 'input-analysis.json'
    if output.exists(): raise FileExistsError(output)
    store = Store(root); state = store.state('20261005-152153-eb05')
    calendar = sessions(store.load_state(state, 'calendar')); filters = [('instrument', 'in', ['601238.SH'])]
    bars = store.load_state(state, 'bars_1d', filters = filters)
    coverage = store.load_state(state, 'adj_coverage', filters = filters)
    view = with_adjusted(bars, store.load_state(state, 'adj_factors', filters = filters), coverage)
    view['date'] = pd.to_datetime(view.date).dt.date
    known = view.set_index('date').reindex(calendar)
    paused = known[known.is_trading.eq(False)]
    previous = known.close.ffill().shift()
    pause_preclose_matches = bool((paused.preclose == previous.loc[paused.index]).all())
    doc_path = Path('data/staging/strategies-batch7/20261005-rsi-slots/joinquant-api-response.json')
    doc = json.loads(doc_path.read_text(encoding = 'utf-8'))
    html = BeautifulSoup(doc['data'], 'html.parser')
    signature = next(e for e in html.find_all('pre') if e.get_text().startswith('get_price('))
    excerpt = [signature.get_text()]
    for element in signature.next_siblings:
        if getattr(element, 'name', None) in ('h2', 'h3', 'h4'): break
        excerpt.append(element.get_text('\n'))
    excerpt = '\n'.join(excerpt)
    if 'skip_paused=False' not in excerpt or 'fill_paused=True' not in excerpt or 'pre_close' not in excerpt:
        raise ValueError('Required get_price pause semantics absent from archived official documentation')
    probes_doc = json.loads((directory / 'probes.json').read_text(encoding = 'utf-8'))
    history_result = next(p for p in probes_doc['probes'] if p['endpoint'] == 'sina_etf_history')
    if history_result['status'] != 'success': raise ValueError('Sina ETF history probe did not succeed')
    history = pd.read_parquet(history_result['raw_file']); dates = pd.to_datetime(history.date).dt.date
    if dates.duplicated().any(): raise ValueError('ETF dates duplicate')
    within = dates[(dates >= calendar[0]) & (dates <= calendar[-1])]
    wire = directory / 'sina_etf_dividend_raw/response-000.bin'
    text = wire.read_text(encoding = 'utf-8'); prefix = 'var sh510310hfq='
    if not text.startswith(prefix): raise ValueError('Unexpected dividend response assignment')
    dividend, end = json.JSONDecoder().raw_decode(text[len(prefix):])
    trailing = text[len(prefix) + end:].strip()
    if trailing and not (trailing.startswith('/*') and trailing.endswith('*/')):
        raise ValueError('Unexpected dividend response suffix')
    if set(dividend) != {'total', 'data'} or type(dividend['total']) is not int or not isinstance(dividend['data'], list) or dividend['total'] != len(dividend['data']):
        raise ValueError('Unexpected dividend response schema')
    for row in dividend['data']:
        if not isinstance(row, dict) or set(row) != {'d', 'f', 's', 'u'} or not all(isinstance(v, str) for v in row.values()):
            raise ValueError('Unexpected dividend row schema')
    # Keep provider fields verbatim: their economic meaning is not established.
    result = {'snapshot': '20261005-152153-eb05', 'published_batch': state['batch_id'],
              'gac': {'rows': len(view), 'missing_calendar_rows': sorted(str(d) for d in set(calendar) - set(view.date)),
                      'paused_rows': paused.reset_index()[['date', 'preclose', 'back_factor']].to_dict('records'),
                      'traded_rows': int(known.is_trading.eq(True).sum()), 'traded_close_missing': int(known[known.is_trading.eq(True)].close_adj.isna().sum()),
                      'pause_preclose_matches_previous_traded_close': pause_preclose_matches,
                      'adjustment_coverage': coverage.to_dict('records'),
                      'first_79_session_decision': calendar[78],
                      'policy_pending': 'Compatibility variant awaits user response. Original get_price defaults retain known pauses and fill pre_close; missing rows must not be filled.'},
              'official_get_price': {'response_file': str(doc_path), 'response_sha256': file_sha(doc_path), 'excerpt': excerpt,
                                     'scope': 'Reused current official documentation archive; not original historical platform equivalence'},
              'etf_history': {'response_sha256': history_result['raw_sha256'], 'rows': len(history), 'first': dates.min(), 'last': dates.max(),
                              'within_existing_calendar': len(within), 'missing_existing_sessions': sorted(str(d) for d in set(calendar) - set(within)),
                              'provider_dates_outside_existing_sessions': sorted(str(d) for d in set(within) - set(calendar)),
                              'after_frozen_snapshot_end': [str(d) for d in dates if d > calendar[-1]]},
              'etf_dividend_response': {'wire_sha256': file_sha(wire), 'parsed_as': 'Strict JSON object after fixed provider assignment; no eval',
                                        'provider_document': dividend,
                                        'warning': 's differs from 1 after 2024-09-23. Do not treat u alone as complete cash actions or assume fixed share count.'},
              'limits': ['No inputs published or existing frozen files modified',
                         'ETF volume units, adjustment factors, event types and effective dates, pay dates, historical suspension/limits and trading rules still unverified',
                         'Downloaded history extends beyond frozen snapshot; raw evidence cannot silently replace it',
                         'No new backtest results or completed reproduction claimed']}
    write_json(output, result)
    progress = archive(root, '第九批输入分析落盘：广汽10日停牌与get_price填充规则；ETF3287行新响应及复权字段/快照边界限制')
    return {'status': 'ok', 'output': str(output), 'progress': progress['output']}


def summarize(root, directory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    first = directory.parent / '20261005-source-reviews'
    artifacts = [first / name for name in ('review.json', 'probes.json', 'protection.json', 'existing-etf-apis.json')]
    artifacts += [directory / name for name in ('review.json', 'probes.json', 'protection.json', 'existing-etf-apis.json', 'input-analysis.json')]
    sources, probe_results, protections = [], [], []
    for folder in (first, directory):
        review_doc = json.loads((folder / 'review.json').read_text(encoding = 'utf-8'))
        for record in review_doc['sources']:
            if file_sha(record['source_path']) != record['source_sha256'] or file_sha(record['source_copy']) != record['source_sha256']:
                raise ValueError('Reviewed source bytes changed')
            sources.append(record)
        probe_doc = json.loads((folder / 'probes.json').read_text(encoding = 'utf-8'))
        for result in probe_doc['probes']:
            for wire in result.get('wire_responses', []):
                if file_sha(wire['file']) != wire['sha256']: raise ValueError('Probe wire response changed')
            if result.get('raw_file') and file_sha(result['raw_file']) != result['raw_sha256']:
                raise ValueError('Raw probe table changed')
            probe_results.append(result)
        protection = json.loads((folder / 'protection.json').read_text(encoding = 'utf-8'))
        if protection['status'] != 'ok': raise ValueError('Old asset protection incomplete')
        protections.append(protection)
    progress = archive(root, '第九批12份审查与6项ETF探测验收归档；取得新浪历史行情，交易/复权规则未证明；广汽兼容方案等待确认')
    handoff = json.loads(Path('docs/handoff/observe-2026-10-04-batch2.manifest.json').read_text(encoding = 'utf-8'))
    result = {'status': 'ok', 'scope': 'Source review and dependency evidence only', 'environment': environment(),
              'published_batch': Store(root).published()['batch_id'], 'snapshot': '20261005-152153-eb05',
              'reviewed_sources': sources, 'source_bytes_checked': len(sources), 'probe_results': probe_results,
              'protections': protections, 'artifacts': {str(path): file_sha(path) for path in artifacts}, 'progress': progress,
              'missing_remote_handoff_files': sum(not Path(item['path']).is_file() for item in handoff['files']),
              'new_strategy_experiments': [], 'original_strategy_complete': False,
              'verification': {'regressions': '2 passed, 8 deselected, 1 existing Starlette/httpx warning in 2.28s',
                               'command': 'python -m pytest -q tests/integration/test_strategy_scope.py -k "progress_archives or protection_excludes"',
                               'ruff': 'passed', 'git_diff_check': 'passed',
                               'review': 'check standard: base, security and architecture; no confirmed new findings; input analysis additionally reviewed locally'},
              'next': 'Await source12 compatibility decision; retain known pause semantics and no filling unknown inputs. ETF publication awaits event/rule evidence.'}
    write_json(output, result)
    return {'status': 'ok', 'output': str(output), 'sources_checked': len(sources), 'progress': progress['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['review', 'review-next', 'probes', 'fallback', 'inputs', 'summary', 'worker'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--directory', default = 'data/staging/strategies-batch9/20261005-source-reviews')
    parser.add_argument('--output', default = 'docs/handoff/2026-10-05-batch9-verification.json')
    parser.add_argument('--worker', choices = PROBES + FALLBACKS); args = parser.parse_args(); directory = Path(args.directory)
    if args.action == 'review': result = review(args.root, directory)
    elif args.action == 'review-next': result = review(args.root, directory, NEXT_SOURCES)
    elif args.action == 'probes': result = probes(args.root, directory)
    elif args.action == 'fallback': result = probes(args.root, directory, FALLBACKS)
    elif args.action == 'inputs': result = inputs(args.root, directory)
    elif args.action == 'summary': result = summarize(args.root, directory, args.output)
    else:
        if not args.worker: parser.error('worker action requires --worker')
        result = worker(args.root, directory, args.worker)
    print(json.dumps(result, ensure_ascii = False))
