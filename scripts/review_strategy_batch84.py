"""Freeze industry ETF rotation/T0 sources and audit real candidate coverage."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch83 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch84/20261009-industry-ETF-T0')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略1/43.ETF轮动：年化收益82.68%，最大回撤13.54%.txt',
    '聚宽2025年精选/91ETF-T0动量策略，年化 18%，回撤3.31%.txt')
SOURCE_SHA = ('a8dbf037d749327d94a1fb613ad27fec615c7a278541b0d169933a0832c910dd',
    'eecd6937c4a473a728c53ffd47f9659003dc3706e2dc88af027ff471c0ecc61e')
POOL = ('161725.XSHE', '159992.XSHE', '560080.XSHG', '515700.XSHG', '515250.XSHG',
    '515790.XSHG', '515880.XSHG', '159819.XSHE', '512720.XSHG', '159740.XSHE',
    '159985.XSHE', '162411.XSHE', '518880.XSHG', '513100.XSHG')
REUSED = {
    '518880.XSHG': ('docs/handoff/2026-10-08-batch74-verification.json',
        'data/staging/strategies-batch74/20261008-FOF-two-eight/sina-fallback/fund-518880/result.json'),
    '513100.XSHG': ('docs/handoff/2026-10-09-batch78-verification.json',
        'data/staging/strategies-batch78/20261009-core-assets-weighted-momentum/probes/fund-513100/result.json')}
QUERIES = {'daily-'+i[:6]: {'function': 'fund_etf_hist_sina', 'parameters':
    {'symbol': ('sz' if i.endswith('XSHE') else 'sh')+i[:6]}, 'instrument': i}
    for i in (*POOL, '513030.XSHG') if i not in REUSED}
QUERIES['minute-513030'] = {'function': 'fund_etf_hist_min_em', 'parameters':
    {'symbol': '513030', 'start_date': '2021-08-01 09:30:00', 'end_date': '2026-09-29 15:00:00',
     'period': '1', 'adjust': ''}, 'instrument': '513030.XSHG'}
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/akshare', 'akshare/fund/fund_etf_em.py'), ('repo/qlib', 'qlib/contrib/strategy/signal_strategy.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch84.py', 'tests/unit/test_batch84_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'source-reviews/review.json', 'dependency-inventory.json', 'reused-inputs.json', 'existing-apis.json',
    'diagnostics.json', 'probe-results.json', 'candidate-audit.json', 'component-research.json',
    'rotation-values.parquet', 'gap-values.parquet', 'registered-fund-coverage.json', 'offline-verification.json', 'offline-catalog.json',
    'development-specialized-attempt01/result.json', 'development-specialized-diagnosis.json',
    'development-offline-attempt01/result.json', 'development-offline-order-red/result.json', 'development-offline-diagnosis.json'}


def implementation(): return {n: file_sha(n) for n in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch83-verification.json')
    review = Path('docs/handoff/2026-10-09-batch83-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted83 binding changed')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted83 evidence changed')
    for n in ('scripts/review_strategy_batch83.py', 'tests/unit/test_batch83_research.py'):
        require(file_sha(n) == accepted['implementation_sha256'][n], 'Accepted83 entrypoint changed')
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Input binding changed')
    return result


def reused_inputs():
    rows = []
    for instrument, (receipt, result) in REUSED.items():
        checked = read(receipt)['checks']['evidence_sha256']; base = Path(result).parents[2]
        # Accepted receipts bind the original raw and parsed candidates, not just their result summaries.
        relative = Path(result).relative_to(base).as_posix()
        require(checked[relative] == file_sha(result), 'Reused result changed')
        row = read(result)
        for name, sha in [(row['file'], row['sha256']), *((w['file'], w['sha256']) for w in row['wire_responses'])]:
            require(file_sha(name) == sha == checked[Path(name).relative_to(base).as_posix()], 'Reused input changed')
        rows.append({'instrument': instrument, 'receipt': receipt, 'receipt_sha256': file_sha(receipt),
            'result_file': result, 'result_sha256': file_sha(result), 'payload': row, 'strict_usable': False})
    return rows


def api_evidence():
    import akshare as ak
    functions = {}
    for name in ('fund_etf_hist_sina', 'fund_etf_hist_min_em'):
        fn = getattr(ak, name); module = inspect.getmodule(fn); source = inspect.getsource(fn)
        functions[name] = {'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'signature': str(inspect.signature(fn))}
        if name.endswith('sina'):
            functions[name].update(decoder=module.hk_js_decode,
                decoder_sha256=hashlib.sha256(module.hk_js_decode.encode()).hexdigest())
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'functions': functions, 'queries': QUERIES,
        'policy': 'One native call per missing symbol; no retries/proxies/redirects;8/10s,worker45s; minute explicit market1',
        'minute_contract': 'period1 uses ndays5; start/end slice response and do not request historical dates',
        'limits': ['Sina candidates lack original dynamic-adjustment/events/state and historical fund identity evidence',
            'Candidate filenames/suffixes are not proof of LOF/commodity/cross-border trading rules',
            '9:40 and14:59 executable quotes, T+0 and actual price tick/slippage remain separate gates']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    codes = [i[:6]+('.SZ' if i.endswith('XSHE') else '.SH') for i in (*POOL, '513030.XSHG')]
    frames = {t: store.load_state(state, t, filters=[('instrument', 'in', codes)])
        for t in ('instruments', 'adj_coverage')}
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(r['rows'] for r in entries.values()) for t, entries in state['tables'].items()},
        'fund_instruments': frames['instruments'].to_dict('records'),
        'fund_adj_coverage': frames['adj_coverage'].to_dict('records'),
        'minute_frequency': '5m only; registered rows do not prove ETF presence or 14:59 quotes',
        'pool': list(POOL), 'T0': '513030.XSHG', 'reused': reused_inputs(),
        'original_trades_admitted': 0, 'not_a_backtest': True}


def preflight(root, directory):
    bound = binding(root); old = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in old['implementation_sha256'].items()), 'Accepted implementation changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    for repo, name in REFERENCES: require((Path(repo)/name).is_file(), 'Reference missing')
    dep = inventory(root)
    checkpoint(root, directory, 'Batch84 parent83 verified; missing ETF and minute dependencies not substituted')
    save(directory/'input-binding.json', bound); save(directory/'dependency-inventory.json', dep)
    save(directory/'reused-inputs.json', dep['reused']); save(directory/'existing-apis.json', api_evidence())
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    return {'status': 'ok', 'reused_candidates': len(dep['reused']), 'not_a_backtest': True}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    rows = []; refs = []; (directory/'references').mkdir(exist_ok=False)
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source reviewed/changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()),
            'definition_ast': previous.definitions(ast.parse(text)), 'review': REVIEWS[name], 'newly_reviewed': True})
    for repo, name in REFERENCES:
        path = Path(repo)/name; copied = directory/'references'/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Native fund parsers and position-aware trading reference; no separate ledger'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(old) == len(current) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch84 two complete industry rotation/T0 sources frozen; economic and execution gates retained')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/name) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == sha and
            row['review'] == REVIEWS[name] and row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == previous.definitions(ast.parse(text)), 'Source/AST changed')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(reused_inputs() == read(directory/'reused-inputs.json') and api_evidence() == read(directory/'existing-apis.json'), 'Dependencies changed')


def selected(directory, number, names, namespace):
    row = read(directory/'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Selected source changed')
    return previous.selected_nodes(ast.parse(read_source(Path(row['source_copy']))[0]), names, namespace)


def diagnostics(directory):
    rows = []
    def record(case, **values): rows.append({'case': case, **previous.plain(values)})
    def init(number):
        g = SimpleNamespace(); schedules = []; fees = []; slips = []
        ns = {'g': g, 'np': np, 'pd': pd, 'log': SimpleNamespace(set_level=lambda *a: None),
            'set_benchmark': lambda *a: None, 'set_option': lambda *a: None,
            'OrderCost': lambda **k: k, 'FixedSlippage': lambda v: v,
            'set_order_cost': lambda v, **k: fees.append([v, k]), 'set_slippage': slips.append,
            'run_daily': lambda f, *a, **k: schedules.append([f.__name__, list(a), k])}
        names = ['initialize', 'trade', 'get_rank'] if number == 0 else ['initialize', 'before_market_open',
            'market_open', 'after_market_close', 'after_market_close2']
        selected(directory, number, names, ns)['initialize'](SimpleNamespace())
        record(str(number)+'_defaults', state=vars(g), schedules=schedules, fees=fees, slippage=slips)
        return ns
    ns = init(0); g = ns['g']; prices = {i: np.exp(np.arange(25)*(.02-j*.001)) for j, i in enumerate(POOL)}
    logs = []; ns.update(attribute_history=lambda i, *a, **k: pd.DataFrame({'close': prices[i]}), record=lambda **k: logs.append(k))
    try: ns['get_rank'](POOL)
    except NameError as exc: record('rotation_missing_math', error=type(exc).__name__)
    ns['math'] = math
    with redirect_stdout(StringIO()): rank = ns['get_rank'](POOL)
    record('rotation_positive', ranking=rank, logged_types=[type(v).__name__ for d in logs for v in d.values()])
    prices = {i: np.exp(-np.arange(25)*(.001+j*.001)) for j, i in enumerate(POOL)}
    with redirect_stdout(StringIO()): rank = ns['get_rank'](POOL)
    record('rotation_negative_still_selects', first=rank[0])
    prices = {i: np.ones(25) for i in POOL}
    with np.errstate(all='ignore'), redirect_stdout(StringIO()): rank = ns['get_rank'](POOL)
    record('rotation_constant', ranking=rank, logged_scores=[float(next(iter(d.values())).iloc[0]) for d in logs[-14:]])
    for mode in ('empty_dict', 'zero_missing', 'held_target', 'failed_sale', 'immediate_sale'):
        class Positions(dict):
            def __missing__(self, key): return SimpleNamespace(total_amount=0)
        positions = {} if mode == 'empty_dict' else Positions()
        if mode == 'held_target': positions[POOL[0]] = SimpleNamespace(total_amount=100)
        if mode in ('failed_sale', 'immediate_sale'): positions[POOL[1]] = SimpleNamespace(total_amount=100)
        portfolio = SimpleNamespace(positions=positions, available_cash=1000.)
        orders = []
        def order(i, value):
            orders.append([i, value])
            if mode == 'immediate_sale' and value == 0: del positions[i]; portfolio.available_cash += 500.
        ns.update(get_rank=lambda p: list(POOL), order_target_value=order)
        try:
            with redirect_stdout(StringIO()): ns['trade'](SimpleNamespace(portfolio=portfolio))
            record('rotation_'+mode, orders=orders)
        except KeyError as exc: record('rotation_'+mode, orders=orders, error=type(exc).__name__)
    ns = init(1); g = ns['g']; orders = []
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={}, available_cash=1000.))
    ns.update(get_current_data=lambda: {g.security1: SimpleNamespace(day_open=g.test_open)},
        attribute_history=lambda *a, **k: {'close': [g.test_close]},
        order_value=lambda *a: orders.append(list(a)), order_target=lambda *a: orders.append(list(a)))
    for gap in (math.nextafter(.003, -math.inf), .003, math.nextafter(.003, math.inf), .1,
                math.nextafter(.5, -math.inf), .5, math.nextafter(.5, math.inf), -.01, math.nan):
        orders.clear(); g.test_close = 0.; g.test_open = gap; ns['market_open'](context)
        record('T0_gap_'+repr(gap), gap=gap, orders=list(orders))
    orders.clear(); g.test_close = 1.; g.test_open = 1.003; ns['market_open'](context)
    record('T0_decimal_subtraction_boundary', actual_gap=g.open-g.yclose, orders=list(orders))
    for positions in ({}, {'unrelated': object()}, {g.security1: object()}):
        orders.clear(); context.portfolio.positions = positions; ns['after_market_close'](context)
        record('T0_exit_'+str(len(rows)), keys=list(positions), orders=list(orders))
    return {'cases': rows, 'case_count': len(rows), 'not_a_backtest': True,
        'scope': 'Frozen original selected functions with controlled inputs; injected math and zero-missing positions are explicit diagnostic models'}


def expected_request(case):
    q = QUERIES[case]
    if case.startswith('daily-'):
        return 'https://finance.sina.com.cn/realstock/company/'+q['parameters']['symbol']+'/hisdata_klc2/klc_kl.js', None
    return 'https://push2his.eastmoney.com/api/qt/stock/trends2/get', {
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13', 'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
        'ut': '7eea3edcaed734bea9cbfc24409ed989', 'ndays': '5', 'iscr': '0', 'secid': '1.513030'}


def invoke(case):
    import akshare as ak
    q = QUERIES[case]; fn = getattr(ak, q['function'])
    if case.startswith('minute-'):
        with patch.object(inspect.getmodule(fn), 'get_market_id', return_value=1): return fn(**q['parameters'])
    return fn(**q['parameters'])


def worker(directory, case):
    require(case in QUERIES, 'Unknown case'); folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'api_sha256': file_sha(directory/'existing-apis.json'), 'published': False, 'strict_usable': False}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        target, params = expected_request(case)
        require(not row['attempts'] and method.upper() == 'GET' and url == target and kwargs.get('params') == params, 'Unexpected/repeated request')
        row['attempts'].append({'method': 'GET', 'url': url, 'params': params}); session.trust_env = False
        kwargs.update(timeout=(8, 10), allow_redirects=False); response = original(session, method, url, **kwargs)
        path = folder/'response.bin'
        with path.open('xb') as out: out.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder/'parsed.parquet'
        with path.open('xb') as out: frame.to_parquet(out, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); validate_sources(directory)
    require(not (directory/'probes').exists() and not (directory/'probe-results.json').exists(), 'Probes already attempted')
    rows = []
    for case in QUERIES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch84', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory/f'worker-{case}.json', {'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory/f'worker-{case}.json', {'command': command, 'status': 'timeout', 'seconds': 45})
            folder = directory/'probes'/case; folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'result.json').exists(): save(folder/'result.json', {'case': case, 'query': QUERIES[case],
                'status': 'timeout', 'attempts': [], 'wire_responses': [], 'error': '45s deadline; completion unproved',
                'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()},
                'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False})
        rows.append(read(directory/'probes'/case/'result.json')); print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    save(directory/'probe-results.json', {'results': rows, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0})
    return archive(root, 'Batch84 bounded native ETF/day and exact1m attempts archived without replacing prior candidates')


def candidate_audit(directory):
    require(api_evidence() == read(directory/'existing-apis.json'), 'API changed')
    rows = read(directory/'probe-results.json')['results']; profiles = []
    require(len(rows) == len(QUERIES) and {r['case'] for r in rows} == set(QUERIES), 'Probe scope changed')
    for row in rows:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory/'existing-apis.json'), 'Candidate binding changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Response changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Parsed binding changed')
            calls = []
            def replay(session, method, url, **kwargs):
                target, params = expected_request(case)
                require(not calls and method.upper() == 'GET' and url == target and kwargs.get('params') == params, 'Unexpected replay')
                calls.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(frame) == row['rows'] and list(frame) == row['columns'] and len(calls) == 1, 'Parser scope differs')
            if len(frame):
                dates = pd.to_datetime(frame['时间'] if case.startswith('minute-') else frame.date)
                profile.update(rows=len(frame), first=str(dates.min()), last=str(dates.max()),
                    post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29 15:00:00')).sum()))
            profile['native_parser_match'] = True
        else: require(row.get('error'), 'Failure reason missing'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'new_trading_backtests': 0, 'not_a_backtest': True}


def score(values):
    y = np.log(np.asarray(values, dtype=float)); x = np.arange(len(y)); slope, intercept = np.polyfit(x, y, 1)
    annual = math.pow(math.exp(slope), 250)-1
    r2 = 1-sum((y-(slope*x+intercept))**2)/((len(y)-1)*np.var(y, ddof=1))
    return slope, annual, r2, annual*r2


def independent_score(values):
    y = [math.log(float(v)) for v in values]; n = len(y); xm = (n-1)/2; ym = math.fsum(y)/n
    slope = math.fsum((j-xm)*(v-ym) for j, v in enumerate(y))/math.fsum((j-xm)**2 for j in range(n))
    intercept = ym-slope*xm; sst = math.fsum((v-ym)**2 for v in y)
    r2 = 1-math.fsum((v-(slope*j+intercept))**2 for j, v in enumerate(y))/sst if sst else math.nan
    annual = math.pow(math.exp(slope), 250)-1
    return slope, annual, r2, annual*r2


def compute(directory):
    sources = {r['instrument']: r['payload'] for r in read(directory/'reused-inputs.json')}
    sources.update({r['query']['instrument']: r for r in read(directory/'probe-results.json')['results']
        if r['case'].startswith('daily-') and r['status'] == 'success'})
    frames = {}; profiles = []; values = []; gaps = []; differences = 0
    for instrument, row in sources.items():
        require(file_sha(row['file']) == row['sha256'], 'Component input changed')
        frame = pd.read_parquet(row['file']); frame['date'] = pd.to_datetime(frame.date)
        require(not frame.date.isna().any() and not frame.date.duplicated().any(), 'Bad candidate dates')
        frame = frame[frame.date.le(pd.Timestamp('2026-09-29'))].sort_values('date').reset_index(drop=True)
        for col in ('close', 'open'): frame[col] = pd.to_numeric(frame[col], errors='coerce')
        frames[instrument] = frame.set_index('date'); count = 0
        if instrument in POOL:
            for k in range(25, len(frame)):
                window = frame.close.iloc[k-25:k].to_numpy()
                if not np.isfinite(window).all() or (window <= 0).any(): continue
                a = score(window); b = independent_score(window)
                mismatch = not all(np.isclose(u, v, rtol=1e-8, atol=1e-9, equal_nan=True) for u, v in zip(a, b, strict=True))
                differences += int(mismatch); count += 1
                values.append({'date': frame.date.iloc[k], 'instrument': instrument, 'slope': a[0], 'annual': a[1], 'r2': a[2],
                    'score': a[3], 'independent_score': b[3], 'window_end': frame.date.iloc[k-1], 'not_a_backtest': True})
        else:
            for k in range(1, len(frame)):
                today, prior = frame.iloc[k], frame.iloc[k-1]; gap = today.open-prior.close
                valid = math.isfinite(gap) and today.open > 0 and prior.close > 0
                gaps.append({'date': today.date, 'prior_date': prior.date, 'open': today.open, 'prior_close': prior.close,
                    'absolute_gap': gap, 'threshold_hit': bool(valid and .003 <= gap <= .5), 'not_a_backtest': True})
        profiles.append({'instrument': instrument, 'rows_before_cutoff': len(frame), 'complete_25_row_components': count,
            'first': str(frame.date.min()), 'last': str(frame.date.max()), 'strict_usable': False})
    require(differences == 0, 'Independent regression differs')
    output = pd.DataFrame(values, columns=['date', 'instrument', 'slope', 'annual', 'r2', 'score', 'independent_score', 'window_end', 'not_a_backtest'])
    gapframe = pd.DataFrame(gaps, columns=['date', 'prior_date', 'open', 'prior_close', 'absolute_gap', 'threshold_hit', 'not_a_backtest'])
    full = output.groupby('date').instrument.nunique() if len(output) else pd.Series(dtype=int)
    complete_days = full[full.eq(len(POOL))]; rankings = []
    for day in complete_days.index:
        rows = output[output.date.eq(day)].set_index('instrument').reindex(POOL)
        original_rank = list(rows.sort_values('score', ascending=False).index)
        independent_rank = sorted(POOL, key=lambda i: rows.loc[i, 'independent_score'], reverse=True)
        ordered = rows.loc[original_rank, 'score'].to_numpy()
        finite_distinct = bool(np.isfinite(ordered).all() and all(not np.isclose(a, b, rtol=1e-8, atol=1e-9)
            for a, b in zip(ordered[:-1], ordered[1:], strict=True)))
        if finite_distinct: require(original_rank == independent_rank, 'Independent full-pool ranking differs')
        rankings.append({'date': str(day), 'original_rank': original_rank, 'independent_rank': independent_rank,
            'independent_order_verified': finite_distinct, 'tie_or_nonfinite_order_unproved': not finite_distinct})
    return output, gapframe, {'profiles': profiles, 'pool_size': len(POOL), 'covered_pool_size': len(set(frames)&set(POOL)),
        'missing_pool': sorted(set(POOL)-set(frames)), 'full_pool_25_row_days': len(complete_days), 'full_pool_rankings': rankings,
        'regression_rows': len(output), 'independent_formula_differences': differences, 'T0_gap_rows': len(gapframe),
        'T0_threshold_hits': int(gapframe.threshold_hit.sum()) if len(gapframe) else 0,
        'new_trading_backtests': 0, 'cost_backtests': [], 'not_a_backtest': True,
        'limits': ['Unadjusted candidate prior25 observed rows; original history skip-paused/dynamic basis not proved',
            'Missing funds block original14-fund ranking; individual score references do not select original winner',
            'Full-pool arithmetic still does not prove9:40 executable quotes, identity, events, fees or platform semantics',
            'T0 prior observed close is candidate arithmetic; no filled missing sessions/event basis or14:59 return',
            'Absolute thresholds include binary floating subtraction behavior; fixed pool is ex-post']}


def study(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    diag = diagnostics(directory); audit = candidate_audit(directory); values, gaps, summary = compute(directory)
    require(before == implementation(), 'Implementation changed during study')
    save(directory/'diagnostics.json', diag); save(directory/'candidate-audit.json', audit)
    save(directory/'registered-fund-coverage.json', registered_coverage(root))
    for name, frame in [('rotation-values.parquet', values), ('gap-values.parquet', gaps)]:
        with (directory/name).open('xb') as out: frame.to_parquet(out, index=False)
    save(directory/'component-research.json', summary)
    return archive(root, 'Batch84 original boundaries/native candidate coverage and longest prior25 arithmetic archived; trading gates retained')


def registered_coverage(root):
    store = Store(root); state = store.state(SNAPSHOT)
    codes = [i[:6]+('.SZ' if i.endswith('XSHE') else '.SH') for i in (*POOL, '513030.XSHG')]
    return {'snapshot': SNAPSHOT, 'instruments': codes, 'queried_rows': {
        table: len(store.load_state(state, table, columns=['instrument'], filters=[('instrument', 'in', codes)]))
        for table in ('bars_1d', 'bars_5m', 'minute_universe')}, 'not_a_backtest': True,
        'scope': 'Actual exact-code filter queries;5m rows cannot prove1m execution'}


def compare_temporal_frames(left, right, columns):
    left = left.copy(); right = right.copy()
    if len(left) or len(right):
        for column in columns:
            left[column] = left[column].dt.as_unit('ns'); right[column] = right[column].dt.as_unit('ns')
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def offline(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    def forbidden(*a, **k): raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        diag = diagnostics(directory); audit = candidate_audit(directory); values, gaps, summary = compute(directory)
        dep = inventory(root); coverage = registered_coverage(root); catalog = offline_catalog(root, directory)
    require(diag == read(directory/'diagnostics.json') and audit == read(directory/'candidate-audit.json') and
        summary == read(directory/'component-research.json') and dep == read(directory/'dependency-inventory.json') and
        coverage == read(directory/'registered-fund-coverage.json'), 'Offline evidence differs')
    compare_temporal_frames(values, pd.read_parquet(directory/'rotation-values.parquet'), ('date', 'window_end'))
    compare_temporal_frames(gaps, pd.read_parquet(directory/'gap-values.parquet'), ('date', 'prior_date'))
    require(before == implementation(), 'Implementation changed during offline')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'catalog': catalog, 'new_trading_backtests': 0, 'not_a_backtest': True})
    return archive(root, 'Batch84 forbidden-network diagnostic/native parser/coverage/arithmetic and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch84.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch84_research.py',
            'tests/unit/test_batch83_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'],
        ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Core missing')
    folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for k, command in enumerate(commands()):
        path = folder/f'check-{k}.log'
        with path.open('x') as out: process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': process.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(process.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True); require(not path.exists(), 'Checked copy exists'); shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and current == checked['implementation_sha256'] == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands(), 'Checks changed')
    for row in checked['commands']: require(row['returncode'] == 0 and file_sha(row['log']) == row['sha256'], 'Log changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch84 industry rotation/T0 source and candidate arithmetic accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch84-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory/'diagnostics.json'),
        'research': read(directory/'component-research.json'), 'candidates': read(directory/'candidate-audit.json'),
        'inventory': read(directory/'dependency-inventory.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': progress['catalog_sha256']})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'newly_reviewed': 2, 'new_trading_backtests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'start', 'probe', 'worker', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY); parser.add_argument('--case', choices=list(QUERIES))
    args = parser.parse_args(); result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
