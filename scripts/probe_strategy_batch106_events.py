"""One bounded event-page attempt for each newly selected fund, no SDK JavaScript."""
import argparse
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import time

import pandas as pd
import requests

from observe.runs import file_sha
from scripts.review_strategy_batch18 import save

DIRECTORY = Path('data/staging/strategies-batch106/20261011-fund-event-probe-v1')
REFERENCE = Path('repo/akshare/akshare/fund/fund_em.py')


def probe(directory=DIRECTORY):
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(REFERENCE, directory/'fund_em.reference.py')
    records = []; stop = False; deadline = time.monotonic()+30
    with requests.Session() as client:
        client.trust_env = False
        for symbol in ('159905','159949'):
            row = {'symbol':symbol, 'url':f'https://fundf10.eastmoney.com/fhsp_{symbol}.html',
                'attempted':False, 'strict_usable':False}
            if stop or time.monotonic() >= deadline:
                row.update(status='not_requested', reason='Same-domain name-resolution failure or overall time limit')
                records.append(row); save(directory/(symbol+'-request.json'),row); continue
            row['attempted'] = True; response = None
            try:
                response = client.get(row['url'], timeout=(4,8), allow_redirects=False, stream=True)
                row['http_status'] = response.status_code
                if response.status_code != 200:
                    row['status'] = 'http_status_blocked'
                else:
                    blocks = []; size = 0
                    for block in response.iter_content(65536):
                        size += len(block)
                        if size > 2*1024**2 or time.monotonic() >= deadline: raise ValueError('Bounded event page limit')
                        blocks.append(block)
                    raw = b''.join(blocks); path = directory/(symbol+'.response'); path.write_bytes(raw)
                    row.update(file=str(path),sha256=file_sha(path),bytes=len(raw))
                    encoding = requests.models.chardet.detect(raw).get('encoding') or 'utf-8'
                    tables = pd.read_html(StringIO(raw.decode(encoding)),displayed_only=False)
                    target = directory/(symbol+'-tables.json')
                    save(target, {'tables':[{'columns':[str(s) for s in f.columns], 'values':f.astype(str).values.tolist()} for f in tables],
                        'raw_sha256':row['sha256'], 'current_page_only':True, 'historical_versions_proven':False, 'strict_usable':False})
                    row.update(status='candidate',tables_file=str(target),tables_sha256=file_sha(target),encoding=encoding)
            except (requests.RequestException, ValueError, UnicodeError, LookupError) as exc:
                message = re.sub(r'(https?://)[^/\s@]+@',r'\1[redacted]@',str(exc))[:1500]
                stop = any(s in message for s in ('NameResolutionError','name resolution','Failed to resolve'))
                row.update(status='failed',exception=type(exc).__name__,message=message,DNS_failure=stop)
            finally:
                if response is not None: response.close()
            records.append(row); save(directory/(symbol+'-request.json'),row)
    result = {'status':'dependency_probe_only', 'reference':str(REFERENCE), 'reference_sha256':file_sha(REFERENCE),
        'script_sha256':file_sha('scripts/probe_strategy_batch106_events.py'), 'requests':records,
        'actual_requests':sum(r['attempted'] for r in records), 'responses':sum('http_status' in r for r in records),
        'candidate_pages':sum(r['status']=='candidate' for r in records), 'new_publications':0, 'new_trading_backtests':0,
        'limits':{'proxy_env':False,'timeout':[4,8],'retries':0,'redirects':False,'max_page_bytes':2*1024**2,'overall_seconds':30},
        'direct_page_only':True,'no_provider_JS_or_SDK_executed':True,'no_candidate_promoted_to_Store':True}
    save(directory/'checked.json',result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory',type=Path,default=DIRECTORY)
    args = parser.parse_args(); print(json.dumps(probe(args.directory),ensure_ascii=False,allow_nan=False))
