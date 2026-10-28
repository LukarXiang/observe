"""Bounded native history attempts for ETF61's missing pool, without publication."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import requests

from observe.runs import file_sha
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch107/20261011-original-pool-probe-v1')
FUNDS = ('159608', '159611', '159613', '159619', '159707', '159713', '159731',
    '512690', '516950', '512880', '516880', '515030', '515220', '515210', '512710',
    '159870', '159995', '512150', '588080', '159952', '512800', '515950', '159945')


def expected_url(symbol):
    require(symbol in FUNDS, 'Unselected pool instrument')
    code = ('sz' if symbol.startswith('1') else 'sh')+symbol
    return f'https://finance.sina.com.cn/realstock/company/{code}/hisdata_klc2/klc_kl.js'


def worker(directory, symbol):
    import akshare as ak
    folder = directory/symbol
    folder.mkdir(exist_ok=False)
    api = read(directory/'api.json')
    source = inspect.getsource(ak.fund_etf_hist_sina)
    require(hashlib.sha256(source.encode()).hexdigest() == api['function_sha256'], 'Frozen API changed')
    row = {'symbol': symbol, 'attempted': True, 'expected_url': expected_url(symbol),
        'attempts': [], 'wire_responses': [], 'strict_usable': False, 'published': False}
    original = requests.sessions.Session.request
    with requests.Session() as client:
        client.trust_env = False
        def bounded(session, method, url, **kwargs):
            require(not row['attempts'] and method.upper() == 'GET' and
                url == expected_url(symbol) and kwargs.get('params') is None, 'Unexpected native API request')
            row['attempts'].append({'method': method, 'url': url})
            response = original(client, method, url, timeout=(4, 8), allow_redirects=False, stream=True)
            try:
                require(response.status_code == 200, 'Native HTTP status '+str(response.status_code))
                blocks = []
                size = 0
                deadline = time.monotonic()+12
                for block in response.iter_content(65536):
                    size += len(block)
                    require(size <= 2*1024**2 and time.monotonic() < deadline, 'Native response exceeds bound')
                    blocks.append(block)
                payload = b''.join(blocks)
                response._content = payload
                response._content_consumed = True
                target = folder/'response.bin'
                target.write_bytes(payload)
                row['wire_responses'].append({'file': str(target), 'sha256': file_sha(target),
                    'status_code': response.status_code, 'encoding': response.encoding})
                return response
            finally:
                response.close()
        try:
            with patch.object(requests.sessions.Session, 'request', bounded):
                frame = ak.fund_etf_hist_sina(symbol=('sz' if symbol.startswith('1') else 'sh')+symbol)
            target = folder/'parsed.parquet'
            frame.to_parquet(target, index=False)
            row.update(status='candidate' if len(frame) else 'empty', file=str(target),
                sha256=file_sha(target), rows=len(frame), columns=list(frame.columns))
        except Exception as exc:
            message = str(exc)[:1500]
            row.update(status='failed', exception=type(exc).__name__, message=message,
                DNS_failure=any(s in message for s in ('NameResolutionError', 'name resolution', 'Failed to resolve')))
    save(folder/'result.json', row)
    return row


def probe(directory=DIRECTORY):
    import akshare as ak
    directory.mkdir(parents=True, exist_ok=False)
    source = inspect.getsource(ak.fund_etf_hist_sina)
    save(directory/'api.json', {'function': 'fund_etf_hist_sina', 'version': ak.__version__,
        'source': source, 'function_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'expected_provider_host': 'finance.sina.com.cn', 'parser': 'Installed SDK local decoder; provider response is data only',
        'timeouts': [4, 8], 'parent_timeout': 40, 'redirects': False, 'retries': 0, 'trust_env': False})
    rows = []
    stopped = False
    deadline = time.monotonic()+120
    for symbol in FUNDS:
        if stopped or time.monotonic() >= deadline:
            rows.append({'symbol': symbol, 'attempted': False, 'status': 'not_requested',
                'reason': 'Same-domain DNS failure or total deadline', 'strict_usable': False})
            continue
        command = [sys.executable, '-m', 'scripts.probe_strategy_batch107_pool', 'worker',
            '--directory', str(directory), '--symbol', symbol]
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=40)
            save(directory/(symbol+'-command.json'), {'command': command, 'returncode': child.returncode,
                'stdout': child.stdout, 'stderr': child.stderr})
            require(child.returncode == 0, 'Pool worker failed; keep evidence')
            row = read(directory/symbol/'result.json')
        except subprocess.TimeoutExpired:
            row = {'symbol': symbol, 'attempted': True, 'status': 'timeout',
                'reason': 'Worker killed by subprocess timeout; completion unproved', 'strict_usable': False}
            save(directory/(symbol+'-timeout.json'), row)
            stopped = True
        rows.append(row)
        stopped = stopped or row.get('DNS_failure', False)
    result = {'status': 'dependency_probe_only', 'api_sha256': file_sha(directory/'api.json'),
        'requests': rows, 'actual_requests': sum(len(r.get('attempts', [])) for r in rows),
        'responses': sum(len(r.get('wire_responses', [])) for r in rows),
        'new_candidate_rows': sum(r.get('rows', 0) for r in rows),
        'index399306_not_requested_as_fund': True, 'new_publications': 0, 'new_trading_backtests': 0,
        'script_sha256': file_sha('scripts/probe_strategy_batch107_pool.py')}
    save(directory/'checked.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probe', 'worker'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--symbol', choices=FUNDS)
    args = parser.parse_args()
    result = probe(args.directory) if args.action == 'probe' else worker(args.directory, args.symbol)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
