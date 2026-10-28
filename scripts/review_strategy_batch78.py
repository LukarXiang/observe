"""Freeze core-asset rotation and compare original ordinary/weighted momentum."""
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
import warnings

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts import review_strategy_batch77 as previous
from scripts import review_strategy_batch76 as batch76
from scripts import review_strategy_batch75 as batch75
from scripts import probe_strategy_batch74_fallback as native74
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch78/20261009-core-assets-weighted-momentum')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2024年度精选策略1/3.【回顾3】ETF策略之核心资产轮动.txt',
    '聚宽2025年精选/38ETF策略之核心资产轮动（线性增加权重）.txt')
SOURCE_SHA = ('437a3f19f060e4c661e55e70a0713a48a348fd8efb4006587f9055c98b9cbb65',
    '4d00a6e469c5efa19487b328dd234d8d090d2f5333e58824fcb1a30efdd79bf7')
POOL = ('518880.XSHG', '513100.XSHG', '159915.XSHE', '510180.XSHG')
CASE = 'fund-513100'
QUERY = {'function': 'fund_etf_hist_sina', 'parameters': {'symbol': 'sh513100'}}
ENDPOINT = 'https://finance.sina.com.cn/realstock/company/sh513100/hisdata_klc2/klc_kl.js'
REFERENCES = (('repo/akshare', 'akshare/fund/fund_etf_sina.py'),
    ('repo/rqalpha', 'rqalpha/portfolio/account.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch78.py', 'tests/unit/test_batch78_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'reused-inputs.json', 'dependency-inventory.json', 'existing-apis.json', 'source-reviews/review.json',
    'probe-results.json', 'candidate-audit.json', 'diagnostics.json', 'source-extraction.json',
    'momentum-components.parquet', 'common-rank-components.parquet', 'component-research.json',
    'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch77-verification.json')
    review = Path('docs/handoff/2026-10-09-batch77-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted77 binding changed')
    for name in ('scripts/review_strategy_batch77.py', 'tests/unit/test_batch77_research.py'):
        require(file_sha(name) == accepted['implementation_sha256'][name], 'Accepted77 entrypoint changed')
    previous.binding(root, previous.DIRECTORY)
    result = {'snapshot': SNAPSHOT, 'receipt': str(receipt), 'receipt_sha256': file_sha(receipt),
        'review': str(review), 'review_sha256': file_sha(review), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory/'input-binding.json'), 'Parent binding changed')
    return result


def reused_inputs():
    result = {}
    for directory, name, case, checked_key in (
        (native74.DIRECTORY, 'results.json', 'fund-518880', 'sina-fallback/results.json'),
        (batch75.DIRECTORY, 'probe-results.json', '159915', 'probe-results.json'),
        (batch76.DIRECTORY, 'probe-results.json', 'fund-510180', 'probe-results.json')):
        checked_folder = directory.parent if directory == native74.DIRECTORY else directory
        checked = read(checked_folder/'checked-state.json')['evidence_sha256']
        require(file_sha(directory/name) == checked[checked_key], 'Prior candidate registry changed')
        selected_rows = [r for r in read(directory/name)['results'] if r['case'] == case]
        require(len(selected_rows) == 1, 'Prior candidate scope changed'); row = selected_rows[0]
        require(row['status'] == 'success' and file_sha(row['file']) == row['sha256'] and
            len(pd.read_parquet(row['file'])) == row['rows'], 'Prior candidate payload changed')
        result['fund-'+case.removeprefix('fund-')] = row
    require(set(result) == {'fund-518880', 'fund-159915', 'fund-510180'}, 'Reuse scope incomplete')
    return {'inputs': result, 'strict_usable': False, 'new_requests': 0, 'not_a_backtest': True}


def api_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_sina; source = inspect.getsource(fn); decoder = inspect.getmodule(fn).hk_js_decode
    polyfit = inspect.getsource(np.polyfit)
    return {'akshare': ak.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'query': QUERY,
        'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
        'decoder': decoder, 'decoder_sha256': hashlib.sha256(decoder.encode()).hexdigest(),
        'polyfit_source': polyfit, 'polyfit_sha256': hashlib.sha256(polyfit.encode()).hexdigest(),
        'policy': 'One fixed GET;8/10s,parent45s,no proxy/retry/redirect; native parser frozen',
        'limits': ['Native close rows do not prove adjusted histories/events/states/identity/fees or9:30 execution',
            'Original weighted fit uses squared w; original R2 uses single w and unweighted mean',
            'All post20260929 rows retained but excluded; fixed ex-post fund pool remains a limitation']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    canonical = [s[:6]+('.SH' if s.endswith('XSHG') else '.SZ') for s in POOL]
    counts = {t: len(store.load_state(state, t, filters=[('instrument', 'in', canonical)]))
        for t in ('instruments', 'bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'bars_5m', 'valuations_1d')}
    return {'snapshot': SNAPSHOT, 'funds': list(POOL), 'registered_fund_rows': counts,
        'original_trades_admitted': 0, 'not_a_backtest': True,
        'gates': ['Historical ETF identity/events/states/adjustment/fee versions incomplete',
            'Original9:30 execution/instant cash/positions missing-key semantics unproved',
            'Original03 record receives Series; platform acceptance and implicit math injection unproved',
            'Fixed ex-post pool and historical vendor vintage remain limitations']}


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == s for n, s in checked['implementation_sha256'].items()), 'Accepted implementation changed before launch')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    reused = reused_inputs(); checkpoint(root, directory, 'Batch78 ordinary/weighted core-assets launch baseline; three accepted candidate inputs reused')
    save(directory/'input-binding.json', bound); save(directory/'reused-inputs.json', reused)
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    save(directory/'dependency-inventory.json', inventory(root)); save(directory/'existing-apis.json', api_evidence())
    return {'status': 'ok', 'reused_inputs': 3, 'new_request_cases': 1}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; folder = directory/'source-reviews'; folder.mkdir(); rows = []; refs = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; copied = folder/(strategy_id(name)+'.source')
        require(old[name]['review_status'] != '人工审查完成' and file_sha(source) == old[name]['bytes_sha256'] == SOURCE_SHA[number], 'Source changed/reviewed')
        shutil.copyfile(source, copied); text, encoding = read_source(copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': file_sha(source),
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'definition_ast': definitions(ast.parse(text)),
            'review': REVIEWS[name], 'newly_reviewed': True, 'original_line_numbers_retained': True})
    (directory/'references').mkdir()
    for repository, name in REFERENCES:
        source = Path(repository)/name; copied = directory/'references'/source.name; shutil.copyfile(source, copied)
        refs.append({'file': str(source), 'copy': str(copied), 'sha256': file_sha(source),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(folder/'review.json', {'sources': rows, 'references': refs, 'catalog': catalog, 'snapshot': SNAPSHOT,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True})
    return archive(root, 'Batch78 two complete core-assets momentum sources frozen; original weighting and orders retained')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source binding changed')
    for number, row in enumerate(doc['sources']):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码')/SOURCES[number]) and
            file_sha(row['source_path']) == file_sha(row['source_copy']) == row['source_sha256'] == SOURCE_SHA[number] and
            row['review'] == REVIEWS[SOURCES[number]] and row['encoding'] == encoding and
            row['complete_lines_read'] == len(text.splitlines()) and row['definition_ast'] == definitions(ast.parse(text)), 'Source/AST changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')


def selected(directory, number, namespace):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed'); tree = ast.parse(read_source(source)[0])
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    require({n.name for n in nodes} == {'initialize', 'get_rank', 'trade'} | ({'MOM'} if number else set()) and
        all(safe_function(n) for n in nodes), 'Unsafe/missing definition')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch78-definitions>', 'exec'), namespace)
    return namespace


def namespace(directory, number, include_math=True):
    ns = {'g': SimpleNamespace(), 'np': np, 'pd': pd, 'log': SimpleNamespace(set_level=lambda *a: None),
        'set_benchmark': lambda *a: None, 'set_option': lambda *a: None, 'set_slippage': lambda *a: None,
        'FixedSlippage': lambda v: v, 'OrderCost': lambda **kw: kw, 'set_order_cost': lambda *a, **kw: None,
        'run_daily': lambda *a: None, 'record': lambda **kw: None}
    if include_math: ns['math'] = math
    selected(directory, number, ns); ns['initialize'](None)
    return ns


def reference_node(directory, number):
    source = directory/'source-reviews'/(strategy_id(SOURCES[number])+'.source')
    require(file_sha(source) == SOURCE_SHA[number], 'Frozen source changed')
    tree = ast.parse(read_source(source)[0])
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ('MOM' if number else 'get_rank'))
    if number: body = original.body[:-1]; require(isinstance(original.body[-1], ast.Return), 'MOM return changed')
    else:
        loop = next(n for n in original.body if isinstance(n, ast.For)); body = loop.body[:-1]
        require(ast.unparse(loop.body[-1]) == 'score_list.append(score)', 'Score loop changed')
    # Preserve original arithmetic statements; expose intermediates instead of list append/score return.
    node = ast.FunctionDef(name='reference_details', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='etf')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load())
        for n in ('score', 'slope', 'intercept', 'r_squared', 'annualized_returns')], ctx=ast.Load()))], decorator_list=[], type_params=[])
    ast.fix_missing_locations(node); require(safe_function(node), 'Unsafe extracted arithmetic'); return node


def reference_function(directory, number, ns):
    node = reference_node(directory, number)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch78-source-arithmetic>', 'exec'), ns)
    return ns['reference_details']


def independent(prices, weighted):
    y = [math.log(float(v)) for v in prices]; n = len(y); x = list(range(n))
    w = [1+i/(n-1) if weighted else 1. for i in x]; fit_w = [v*v for v in w]
    mass = math.fsum(fit_w); mx = math.fsum(a*b for a, b in zip(fit_w, x, strict=True))/mass
    my = math.fsum(a*b for a, b in zip(fit_w, y, strict=True))/mass
    slope = math.fsum(a*(b-mx)*(c-my) for a, b, c in zip(fit_w, x, y, strict=True))/math.fsum(a*(b-mx)**2 for a, b in zip(fit_w, x, strict=True))
    intercept = my-slope*mx; mean = math.fsum(y)/n
    denominator = math.fsum(a*(b-mean)**2 for a, b in zip(w, y, strict=True))
    if denominator == 0: return None
    r2 = 1-math.fsum(a*(c-(slope*b+intercept))**2 for a, b, c in zip(w, x, y, strict=True))/denominator
    annualized = math.pow(math.exp(slope), 250)-1
    return (annualized*r2, slope, intercept, r2, annualized)


def original_rank(ns, frames):
    scores = []
    def frame(*args, **kwargs):
        value = pd.DataFrame(*args, **kwargs); scores.extend(float(v) for v in value['score']); return value
    ns['pd'] = SimpleNamespace(DataFrame=frame)
    ns['attribute_history'] = lambda s, *a, **kw: frames[s].copy()
    with redirect_stdout(StringIO()): rank = ns['get_rank'](list(POOL))
    require(len(scores) == 4, 'Original rank score scope changed'); return rank, scores


def diagnostics(directory):
    validate_sources(directory); rows = []
    for number in range(2):
        ns = namespace(directory, number); g = ns['g']; costs = []; schedules = []
        ns['set_order_cost'] = lambda value, **kw: costs.append([value, kw])
        ns['run_daily'] = lambda fn, time: schedules.append([fn.__name__, time]); ns['initialize'](None)
        rows.append({'case': f'{number}_defaults', 'pool': g.etf_pool, 'period': g.m_days, 'costs': costs, 'schedules': schedules})
        records = []; ns['record'] = lambda **kw: records.extend({'type': type(v).__name__, 'length': len(v)} for v in kw.values())
        for name, sign in (('positive', 1), ('negative', -1), ('tied', 0)):
            frames = {s: pd.DataFrame({'close': np.exp(np.arange(25)*((k+1)*.001*sign if sign else .001))}) for k, s in enumerate(POOL)}
            rank, values = original_rank(ns, frames)
            rows.append({'case': f'{number}_{name}', 'rank': rank, 'scores': values})
        rows.append({'case': f'{number}_record_types', 'records': records})
        no_math = namespace(directory, number, include_math=False)
        no_math['attribute_history'] = lambda *a, **kw: pd.DataFrame({'close': np.exp(np.arange(25)*.001)})
        try:
            with redirect_stdout(StringIO()): no_math['get_rank'](list(POOL))
        except NameError: rows.append({'case': f'{number}_implicit_math_error', 'exception': 'NameError'})
        else: raise AssertionError('Expected missing implicit math')
        scorer = reference_function(directory, number, ns)
        for name, values in (('constant', np.ones(25)), ('zero', np.zeros(25)), ('negative_price', -np.ones(25)),
            ('nan', np.array([1.]*24+[np.nan])), ('short', np.exp(np.arange(2)*.001)), ('overflow', np.exp(np.arange(25)*10.))):
            ns['attribute_history'] = lambda *a, v=values, **kw: pd.DataFrame({'close': v})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); detail = scorer(POOL[0])
                rows.append({'case': f'{number}_{name}', 'finite': [bool(math.isfinite(v)) for v in detail],
                    'score': float(detail[0]) if math.isfinite(detail[0]) else None})
            except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
                rows.append({'case': f'{number}_{name}', 'exception': type(exc).__name__})
        ns['get_rank'] = lambda *a: list(POOL); orders = []
        ns['order_target_value'] = lambda s, v: orders.append([s, v])
        context = SimpleNamespace(portfolio=SimpleNamespace(positions={}, available_cash=100.))
        try:
            with redirect_stdout(StringIO()): ns['trade'](context)
        except KeyError: rows.append({'case': f'{number}_missing_position_key', 'exception': 'KeyError'})
        else: raise AssertionError('Expected ordinary dict missing position')
        for name, holdings, release in (('winner_held', {POOL[0]: 100}, False),
                ('delayed_sale', {POOL[1]: 500}, False), ('immediate_sale', {POOL[1]: 500}, True)):
            orders.clear(); positions = {s: SimpleNamespace(total_amount=v) for s, v in holdings.items()}
            context.portfolio.positions = positions; context.portfolio.available_cash = 100.
            def order(s, v):
                orders.append([s, v])
                if release and v == 0:
                    positions.pop(s); context.portfolio.available_cash += 500.
            ns['order_target_value'] = order
            if release:
                from collections import defaultdict
                positions = defaultdict(lambda: SimpleNamespace(total_amount=0), positions); context.portfolio.positions = positions
            with redirect_stdout(StringIO()): ns['trade'](context)
            rows.append({'case': f'{number}_{name}', 'orders': orders.copy(), 'cash': context.portfolio.available_cash})
    return {'cases': rows, 'not_a_backtest': True, 'original_formulas_unchanged': True,
        'platform_assumptions': 'Explicit math/local record/default-position/instant-fill stubs isolate original rules; no platform equivalence or real fills',
        'external_messages': 0}


def invoke():
    import akshare as ak
    return ak.fund_etf_hist_sina(**QUERY['parameters'])


def expected_request(method, url, kwargs, number):
    require(number == 0 and method.upper() == 'GET' and url == ENDPOINT and kwargs.get('params') is None, 'Unexpected/repeated request')
    return {'method': 'GET', 'url': url, 'parameters': None}


def worker(directory):
    folder = directory/'probes'/CASE; folder.mkdir(exist_ok=False)
    row = {'case': CASE, 'query': QUERY, 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory/'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        row['attempts'].append(expected_request(method, url, kwargs, len(row['attempts'])))
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder/'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory/'existing-apis.json'), 'Native API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(StringIO()): frame = invoke()
        path = folder/'parsed.parquet'
        with path.open('xb') as stream: frame.to_parquet(stream, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = type(exc).__name__
    finally: save(folder/'result.json', row)
    return row


def probe(root, directory):
    binding(root, directory); (directory/'probes').mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'scripts.review_strategy_batch78', 'worker', '--directory', str(directory)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        save(directory/'worker.json', {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        require(result.returncode == 0, 'Worker failed before archive')
    except subprocess.TimeoutExpired:
        save(directory/'worker.json', {'command': command, 'status': 'timeout', 'seconds': 45})
        folder = directory/'probes'/CASE; folder.mkdir(exist_ok=True)
        if not (folder/'result.json').exists():
            save(folder/'result.json', {'case': CASE, 'query': QUERY, 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s deadline; completion unproved',
                'api_sha256': file_sha(directory/'existing-apis.json'), 'strict_usable': False, 'published': False})
    save(directory/'probe-results.json', {'results': [read(directory/'probes'/CASE/'result.json')], 'original_input_rows_admitted': 0, 'new_trading_backtests': 0})
    return archive(root, 'Batch78 one bounded513100 native attempt archived; three earlier candidates preserved')


def candidate_audit(directory):
    validate_sources(directory); require(api_evidence() == read(directory/'existing-apis.json') and
        reused_inputs() == read(directory/'reused-inputs.json'), 'API/reused input changed')
    result = read(directory/'probe-results.json')['results']; require(len(result) == 1, 'New candidate scope changed'); row = result[0]
    require(row == read(directory/'probes'/CASE/'result.json') and row['case'] == CASE and row['query'] == QUERY and
        row['api_sha256'] == file_sha(directory/'existing-apis.json'), 'Candidate binding changed')
    for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Raw response changed')
    for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Partial response changed')
    if row['status'] in ('success', 'empty'):
        require(len(row['attempts']) == len(row['wire_responses']) == 1 and file_sha(row['file']) == row['sha256'], 'Capture changed')
        calls = []
        def replay(session, method, url, **kwargs):
            require(expected_request(method, url, kwargs, len(calls)) == row['attempts'][0], 'Parser replay changed'); calls.append(True)
            wire = row['wire_responses'][0]; response = requests.Response(); response.status_code = wire['status_code']
            response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
            response.raise_for_status(); return response
        with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke()
        pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
        require(len(calls) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Native parse scope differs')
    profiles = []
    for case, operand in {**read(directory/'reused-inputs.json')['inputs'], CASE: row}.items():
        profile = {'case': case, 'status': operand['status'], 'strict_usable': False, 'reused': case != CASE}
        if operand['status'] in ('success', 'empty'):
            require(file_sha(operand['file']) == operand['sha256'], 'Operand changed'); frame = pd.read_parquet(operand['file'])
            dates = pd.to_datetime(frame.date); require(dates.notna().all() and not dates.duplicated().any(), 'Invalid dates')
            profile.update(rows=len(frame), first=str(dates.min().date()) if len(frame) else None,
                last=str(dates.max().date()) if len(frame) else None, post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29')).sum()))
        else: require(operand.get('error'), 'Failure lacks reason'); profile['error'] = operand['error']
        profiles.append(profile)
    return {'profiles': profiles, 'new_native_parser_match': row['status'] in ('success', 'empty'),
        'new_requests': len(row['attempts']), 'http_statuses': [w['status_code'] for w in row['wire_responses']],
        'original_input_rows_admitted': 0, 'not_a_backtest': True}


def audit(root, directory):
    binding(root, directory); save(directory/'candidate-audit.json', candidate_audit(directory))
    return archive(root, 'Batch78 four fund candidates and513100 native parser coverage audited; trade gates retained')


def components(directory):
    operands = {**read(directory/'reused-inputs.json')['inputs'], CASE: read(directory/'probe-results.json')['results'][0]}
    spaces = [namespace(directory, k) for k in range(2)]
    scorers = [reference_function(directory, k, ns) for k, ns in enumerate(spaces)]
    records = []; valid = {}; frames = {}; blocked = []; extraction = {str(k): ast.dump(reference_node(directory, k), include_attributes=True) for k in range(2)}
    names = ('score', 'slope', 'intercept', 'r2', 'annualized')
    maxima = {name: 0. for name in names}
    for instrument in POOL:
        row = operands['fund-'+instrument[:6]]
        if row['status'] != 'success': blocked.append([instrument, None, 'candidate_unavailable']); continue
        require(file_sha(row['file']) == row['sha256'], 'History operand changed')
        frame = pd.read_parquet(row['file']).copy(); frame['date'] = pd.to_datetime(frame.date)
        require(frame.date.notna().all() and not frame.date.duplicated().any(), 'History dates invalid')
        frame = frame.loc[frame.date <= pd.Timestamp('2026-09-29')].sort_values('date').set_index('date')
        frame['close'] = pd.to_numeric(frame.close, errors='raise'); frames[instrument] = frame
        valid[instrument] = {}
        for end in range(24, len(frame)):
            window = frame.iloc[end-24:end+1][['close']].copy(); prices = window.close.to_numpy(float); date = window.index[-1]
            if not np.isfinite(prices).all() or not (prices > 0).all() or np.ptp(prices) == 0:
                blocked.append([instrument, str(date.date()), 'invalid_or_constant_close_window']); continue
            outputs = []
            for number, ns in enumerate(spaces):
                ns['attribute_history'] = lambda *a, f=window, **kw: f.copy(); detail = tuple(float(v) for v in scorers[number](instrument))
                other = independent(prices, bool(number)); require(other is not None and all(math.isfinite(v) for v in (*detail, *other)), 'Nonfinite momentum')
                for name, a, b in zip(names, detail, other, strict=True):
                    maxima[name] = max(maxima[name], abs(a-b)); require(math.isclose(a, b, rel_tol=3e-9, abs_tol=3e-10), 'Independent '+name+' differs')
                records.append({'instrument': instrument, 'history_end': str(date.date()), 'variant': 'weighted' if number else 'ordinary',
                    **dict(zip(names, detail, strict=True)), **{'independent_'+name: v for name, v in zip(names, other, strict=True)}})
                outputs.append(detail)
            valid[instrument][date] = (tuple(window.index), outputs, window)
    ranks = []; independent_leader_differences = 0; independent_rank_differences = 0
    common = sorted(set.intersection(*(set(valid.get(s, {})) for s in POOL)))
    for date in common:
        inputs = [valid[s][date] for s in POOL]
        if any(v[0] != inputs[0][0] for v in inputs[1:]): blocked.append(['ALL', str(date.date()), 'unequal25_native_dates']); continue
        history = {s: v[2] for s, v in zip(POOL, inputs, strict=True)}; leaders = []
        for number, ns in enumerate(spaces):
            rank, scores = original_rank(ns, history)
            expected = [v[1][number][0] for v in inputs]
            require(scores == expected, 'Full original ranking/extracted score differs')
            independent_scores = [independent(v[2].close.to_numpy(float), bool(number))[0] for v in inputs]
            order = pd.Series(independent_scores, index=POOL).sort_values(ascending=False).index.tolist()
            independent_leader_differences += rank[0] != order[0]
            independent_rank_differences += rank != order; leaders.append(rank[0])
            ranks.append({'history_end': str(date.date()), 'variant': 'weighted' if number else 'ordinary',
                'leader': rank[0], 'independent_leader': order[0], 'rank': '|'.join(rank), 'independent_rank': '|'.join(order)})
    frame = pd.DataFrame(records, columns=['instrument', 'history_end', 'variant', *names, *['independent_'+n for n in names]])
    ranked = pd.DataFrame(ranks, columns=['history_end', 'variant', 'leader', 'independent_leader', 'rank', 'independent_rank'])
    years = []; variant_differences = 0
    if len(ranked):
        pivot = ranked.pivot(index='history_end', columns='variant', values='leader'); changed = pivot.ordinary != pivot.weighted
        variant_differences = int(changed.sum())
        for year in sorted({d[:4] for d in pivot.index}):
            subset = pivot.loc[[d for d in pivot.index if d[:4] == year]]
            years.append({'year': year, 'common_dates': len(subset), 'variant_leader_differences': int((subset.ordinary != subset.weighted).sum()),
                'ordinary_leaders': subset.ordinary.value_counts().to_dict(), 'weighted_leaders': subset.weighted.value_counts().to_dict()})
    summary = {'not_a_backtest': True, 'new_trading_backtests': 0, 'single_fund_variant_windows': len(frame),
        'common_rank_dates': len(ranked)//2, 'original_rank_calls': len(ranked), 'blocked': blocked,
        'independent_leader_differences': independent_leader_differences, 'independent_rank_differences': independent_rank_differences,
        'variant_leader_differences': variant_differences,
        'max_independent_abs_differences': maxima, 'yearly': years,
        'first_history_end': ranked.history_end.min() if len(ranked) else None,
        'last_history_end': ranked.history_end.max() if len(ranked) else None,
        'limits': api_evidence()['limits'], 'original_input_rows_admitted': 0}
    return frame, ranked, summary, extraction


def study(root, directory):
    outputs = ('diagnostics.json', 'source-extraction.json', 'momentum-components.parquet', 'common-rank-components.parquet', 'component-research.json')
    require(all(not (directory/n).exists() for n in outputs), 'Study archive exists'); binding(root, directory)
    validate_sources(directory); require(api_evidence() == read(directory/'existing-apis.json'), 'API changed')
    before = implementation(); diag = diagnostics(directory); frame, ranks, summary, extraction = components(directory)
    require(before == implementation(), 'Implementation changed during study')
    for name, table in (('momentum-components.parquet', frame), ('common-rank-components.parquet', ranks)):
        with (directory/name).open('xb') as stream: table.to_parquet(stream, index=False)
    save(directory/'diagnostics.json', diag); save(directory/'source-extraction.json', extraction)
    save(directory/'component-research.json', {**summary, 'implementation_sha256': before})
    return archive(root, 'Batch78 original ordinary/weighted scores and full common-pool ranks archived; no trading returns')


def offline(root, directory):
    require(not (directory/'offline-verification.json').exists(), 'Offline archive exists'); binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch78 attempted network')
    with patch.object(socket, 'create_connection', denied), patch.object(socket.socket, 'connect', denied), patch.object(requests.sessions.Session, 'request', denied), redirect_stdout(StringIO()):
        diag = diagnostics(directory); frame, ranks, summary, extraction = components(directory)
        candidates = candidate_audit(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    for name, table in (('momentum-components.parquet', frame), ('common-rank-components.parquet', ranks)):
        pd.testing.assert_frame_equal(table, pd.read_parquet(directory/name), check_exact=True)
    require(before == implementation() and read(directory/'component-research.json') == {**summary, 'implementation_sha256': before} and
        diag == read(directory/'diagnostics.json') and extraction == read(directory/'source-extraction.json') and
        candidates == read(directory/'candidate-audit.json') and dep == read(directory/'dependency-inventory.json'), 'Offline evidence differs')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_sha256': before, 'not_a_backtest': True})
    return archive(root, 'Batch78 forbidden-network original diagnostics/score/rank/native parser and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch78.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch78_research.py', 'tests/unit/test_batch77_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/unit/test_research_metadata.py',
            'tests/unit/test_external_cash_flows.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def sanity(root, directory):
    binding(root, directory); path = directory/'specialized-tests.log'
    with path.open('x') as stream: result = subprocess.run([sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch78_research.py'], stdout=stream, stderr=subprocess.STDOUT)
    save(directory/'specialized-tests.json', {'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path), 'implementation_sha256': implementation()})
    require(result.returncode == 0, 'Specialized tests failed: '+str(path)); return {'status': 'passed'}


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(not (directory/'checked-state.json').exists(), 'Checked archive exists'); folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for number, command in enumerate(commands()):
        path = folder/f'check-{number}.log'
        with path.open('x') as stream: result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        copied = directory/'implementation-checked'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == current == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checks changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    summary = read(directory/'component-research.json')
    require(summary['independent_leader_differences'] == summary['independent_rank_differences'] == 0,
        'Independent ranking differs; reference acceptance blocked')
    protection = protect(root, directory); progress = archive(root, 'Batch78 core-assets ordinary/weighted momentum reference research accepted; original trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch78-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'diagnostics': read(directory/'diagnostics.json'), 'components': read(directory/'component-research.json'),
        'inventory': read(directory/'dependency-inventory.json'), 'candidates': read(directory/'candidate-audit.json'),
        'protection': protection, 'progress': progress, 'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory/'source-reviews/review.json')['catalog']['output'])/'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'sanity', 'worker', 'probe', 'audit', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); result = worker(args.directory) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
