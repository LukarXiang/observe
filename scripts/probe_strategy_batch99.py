"""Two bounded historical option queries; no credentials, current chains, or publishing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import requests

from observe.runs import file_sha
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-bounded-option-probes')
REFERENCES = ('repo/akshare/akshare/option/option_finance_sina.py',
              'repo/akshare/akshare/option/option_risk_indicator_sse.py')
QUERIES = {
    'expired-contract-10001355-daily': {
        'endpoint': 'https://stock.finance.sina.com.cn/futures/api/jsonp_v2.php//StockOptionDaylineService.getSymbolInfo',
        'params': {'symbol': 'CON_OP_10001355'},
        'referer': 'https://stock.finance.sina.com.cn/option/quotes.html',
        'meaning': 'Expired contract named by original source; daily prices cannot supply historical minute fills'},
    'SSE-20180720-risk-identities': {
        'endpoint': 'http://query.sse.com.cn/commonQuery.do',
        'params': {'isPagination': 'false', 'trade_date': '20180720',
                   'sqlId': 'SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L', 'contractSymbol': ''},
        'referer': 'http://www.sse.com.cn/',
        'meaning': 'Single historical risk report; Greeks/IV and identities are not option-chain close prices'}}


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    references = {}
    for name in REFERENCES:
        target = DIRECTORY/'references'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes()); references[name] = file_sha(name)
    save(DIRECTORY/'prepared.json', {'references_sha256': references,
        'worker_sha256': file_sha(__file__), 'queries': QUERIES,
        'policy': 'one GET per case;8/10sec socket timeout,parent35sec,no retry/redirect/proxy',
        'source_credentials_used': False, 'published': False})
    return {'status': 'prepared', 'queries': len(QUERIES)}


def worker(case):
    before = read(DIRECTORY/'prepared.json')
    require(before['worker_sha256'] == file_sha(__file__) and before['queries'] == QUERIES,
            'Frozen query worker differs')
    require(all(file_sha(DIRECTORY/'references'/name) == sha
                for name, sha in before['references_sha256'].items()), 'Reference bytes differ')
    folder = DIRECTORY/case; folder.mkdir(exist_ok=False); query = QUERIES[case]
    result = {'case': case, 'query': query, 'status': 'failed', 'requests_attempted': 1,
              'responses': [], 'published': False, 'strict_usable': False, 'admitted_rows': 0}
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(query['endpoint'], params=query['params'],
                headers={'Referer': query['referer'], 'User-Agent': 'Mozilla/5.0'},
                timeout=(8, 10), allow_redirects=False)
        path = folder/'response.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['responses'].append({'file': str(path), 'sha256': file_sha(path),
            'status_code': response.status_code, 'bytes': len(response.content)})
        response.raise_for_status()
        if case.startswith('expired'):
            body = response.text
            payload = json.loads(body[body.find('(')+1:body.rfind(')')])
        else:
            payload = response.json()['result']
        require(isinstance(payload, list), 'Unexpected data container')
        save(folder/'parsed.json', payload)
        result.update(status='candidate' if payload else 'empty', candidate_rows=len(payload),
                      file=str(folder/'parsed.json'), sha256=file_sha(folder/'parsed.json'))
    except Exception as exc:
        result['error'] = (type(exc).__name__+': '+str(exc))[:1600]
    save(folder/'result.json', result); return result


def probe():
    require(not (DIRECTORY/'summary.json').exists(), 'Immutable probe summary exists')
    results = []
    for case in QUERIES:
        require(not (DIRECTORY/case).exists(), 'Already attempted; use existing evidence')
        command = [sys.executable, '-B', '-m', 'scripts.probe_strategy_batch99', 'worker', '--case', case]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=35)
            save(DIRECTORY/(case+'-command.json'), {'command': command, 'returncode': process.returncode,
                'stdout': process.stdout, 'stderr': process.stderr})
            require(process.returncode == 0, 'Probe worker failed before result')
            results.append(read(DIRECTORY/case/'result.json'))
        except subprocess.TimeoutExpired:
            save(DIRECTORY/(case+'-timeout.json'), {'command': command, 'timeout_seconds': 35})
            results.append({'case': case, 'status': 'timeout', 'request_completion_unproved': True,
                'partial_files': {str(p): file_sha(p) for p in (DIRECTORY/case).glob('*') if p.is_file()},
                'published': False, 'admitted_rows': 0})
    result = {'results': results, 'new_published_rows': 0, 'new_trading_backtests': 0,
              'source_credentials_used': False, 'prepared_sha256': file_sha(DIRECTORY/'prepared.json')}
    save(DIRECTORY/'summary.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'worker'))
    parser.add_argument('--case', choices=tuple(QUERIES)); args = parser.parse_args()
    result = worker(args.case) if args.action == 'worker' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
