"""One bounded Sina attempt per original index/fund; preserve primary failures."""
import argparse
from contextlib import redirect_stdout
import hashlib
import inspect
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import requests

from observe.runs import file_sha
from scripts import review_strategy_batch74 as batch
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = batch.DIRECTORY / 'sina-fallback'
QUERIES = {case: {'function': 'stock_zh_index_daily' if case.startswith('index-') else 'fund_etf_hist_sina',
    'parameters': {'symbol': q['parameters']['symbol'] if case.startswith('index-') else
        ('sh' if q['market'] == 1 else 'sz')+q['parameters']['symbol']}} for case, q in batch.QUERIES.items()}


def api_evidence():
    import akshare as ak
    code = {}
    for name in ('stock_zh_index_daily', 'fund_etf_hist_sina'):
        fn = getattr(ak, name); source = inspect.getsource(fn); module = inspect.getmodule(fn)
        code[name] = {'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'decoder': module.hk_js_decode, 'decoder_sha256': hashlib.sha256(module.hk_js_decode.encode()).hexdigest()}
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'queries': QUERIES, 'functions': code,
        'policy': 'One serial attempt each; native parser/embedded decoder frozen; no redirects/proxy/retry;8/10s,parent45s',
        'limits': ['Sina candidate closes/volume do not prove original adjusted price/amount/events/state/fees',
            'Endpoint may include dates after20260929; keep raw evidence and exclude future rows from components']}


def prepare(root, directory):
    batch.binding(root, batch.DIRECTORY); primary = batch.DIRECTORY / 'probe-results.json'
    require(len(read(primary)['results']) == 20, 'Primary attempts incomplete')
    directory.mkdir(exist_ok=False)
    save(directory / 'input-binding.json', {'primary': str(primary), 'sha256': file_sha(primary),
        'sources': list(batch.SOURCE_SHA), 'strict_usable': False})
    save(directory / 'existing-apis.json', api_evidence())
    return {'status': 'prepared', 'directory': str(directory), 'requests': len(QUERIES)}


def expected_request(case):
    symbol = QUERIES[case]['parameters']['symbol']
    path = 'hisdata' if case.startswith('index-') else 'hisdata_klc2'
    return f'https://finance.sina.com.cn/realstock/company/{symbol}/{path}/klc_kl.js', {'d': '2020_2_4'} if case.startswith('index-') else None


def invoke(case):
    import akshare as ak
    q = QUERIES[case]
    return getattr(ak, q['function'])(**q['parameters'])


def worker(directory, case):
    folder = directory / case; folder.mkdir(exist_ok=False)
    row = {'case': case, 'query': QUERIES[case], 'status': 'failed', 'attempts': [], 'wire_responses': [],
        'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')}
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        endpoint, params = expected_request(case)
        require(not row['attempts'] and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == params, 'Unexpected/repeated request')
        row['attempts'].append({'method': 'GET', 'url': url, 'parameters': params})
        session.trust_env = False; kwargs.update(timeout=(8, 10), allow_redirects=False)
        response = original(session, method, url, **kwargs); path = folder / 'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        row['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status_code': response.status_code, 'encoding': response.encoding})
        response.raise_for_status(); return response
    try:
        require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed Sina API changed')
        with patch.object(requests.sessions.Session, 'request', request), redirect_stdout(sys.stderr): frame = invoke(case)
        path = folder / 'parsed.parquet'; frame.to_parquet(path, index=False)
        row.update(status='success' if len(frame) else 'empty', file=str(path), sha256=file_sha(path), rows=len(frame), columns=list(frame))
    except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', row)
    return row


def probe(root, directory):
    batch.binding(root, batch.DIRECTORY); before = read(directory / 'input-binding.json')
    require(file_sha(before['primary']) == before['sha256'] and not (directory / 'results.json').exists(), 'Primary changed or fallback done')
    rows = []
    for case in QUERIES:
        require(not (directory / case).exists(), 'Fallback already attempted')
        command = [sys.executable, '-m', 'scripts.probe_strategy_batch74_fallback', 'worker', '--directory', str(directory), '--case', case]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=45)
            save(directory / f'worker-{case}.json', {'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Fallback worker failed before archive')
        except subprocess.TimeoutExpired:
            save(directory / f'worker-{case}.json', {'command': command, 'status': 'timeout', 'timeout_seconds': 45})
            folder = directory / case; folder.mkdir(exist_ok=True)
            if not (folder / 'result.json').exists():
                save(folder / 'result.json', {'case': case, 'query': QUERIES[case], 'status': 'timeout', 'attempts': [], 'wire_responses': [],
                    'partial_files': {str(p): file_sha(p) for p in folder.iterdir() if p.is_file()}, 'error': '45s deadline; completion unproved',
                    'strict_usable': False, 'published': False, 'api_sha256': file_sha(directory / 'existing-apis.json')})
        rows.append(read(directory / case / 'result.json'))
        print(case+': '+rows[-1]['status'], file=sys.stderr, flush=True)
    save(directory / 'results.json', {'results': rows, 'new_original_input_rows': 0, 'new_trading_backtests': 0})
    return batch.archive(root, 'Batch74 bounded Sina fallback attempts archived independently of immutable primary failures')


def verify(directory):
    require(api_evidence() == read(directory / 'existing-apis.json'), 'Installed Sina API changed')
    bound = read(directory / 'input-binding.json'); require(file_sha(bound['primary']) == bound['sha256'] and bound['sources'] == list(batch.SOURCE_SHA), 'Fallback input binding changed')
    results = read(directory / 'results.json')['results']
    require(len(results) == len(QUERIES) and {r['case'] for r in results} == set(QUERIES), 'Fallback scope changed')
    profiles = []
    for row in results:
        case = row['case']; require(row['query'] == QUERIES[case] and row['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Fallback query binding changed')
        for wire in row['wire_responses']: require(file_sha(wire['file']) == wire['sha256'], 'Fallback wire changed')
        for name, sha in row.get('partial_files', {}).items(): require(file_sha(name) == sha, 'Fallback partial changed')
        profile = {'case': case, 'status': row['status'], 'strict_usable': False}
        if row['status'] in ('success', 'empty'):
            require(file_sha(row['file']) == row['sha256'] and len(row['attempts']) == len(row['wire_responses']) == 1, 'Fallback parse binding changed')
            attempts = []
            def replay(session, method, url, **kwargs):
                endpoint, params = expected_request(case)
                require(not attempts and method.upper() == 'GET' and url == endpoint and kwargs.get('params') == params, 'Unexpected fallback replay')
                attempts.append(True); wire = row['wire_responses'][0]; response = requests.Response()
                response.status_code = wire['status_code']; response._content = Path(wire['file']).read_bytes(); response.encoding = wire['encoding']; response.url = url
                response.raise_for_status(); return response
            with patch.object(requests.sessions.Session, 'request', replay), redirect_stdout(StringIO()): frame = invoke(case)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(row['file']), check_exact=True)
            require(len(attempts) == 1 and len(frame) == row['rows'] and list(frame) == row['columns'], 'Fallback parse scope differs')
            if len(frame):
                dates = pd.to_datetime(frame.date)
                profile.update(rows=len(frame), first=str(dates.min().date()), last=str(dates.max().date()),
                    post_cutoff_rows=int((dates > pd.Timestamp('2026-09-29')).sum()), unique_dates=dates.nunique())
            profile['native_parser_match'] = True
        else: require(row.get('error'), 'Fallback failure lacks reason'); profile['error'] = row['error']
        profiles.append(profile)
    return {'profiles': profiles, 'original_input_rows_admitted': 0, 'not_a_backtest': True,
        'limits': api_evidence()['limits']}


def audit(root, directory):
    result = verify(directory); save(directory / 'audit.json', result)
    return batch.archive(root, 'Batch74 Sina native parsing and candidate coverage audited; original fund/index trading gates retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'probe', 'audit'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY); parser.add_argument('--root', default='data')
    parser.add_argument('--case', choices=QUERIES); args = parser.parse_args()
    result = worker(args.directory, args.case) if args.action == 'worker' else globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str))
