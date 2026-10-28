"""Bounded direct fund-event page recovery, without executing provider JavaScript."""
import argparse
from io import StringIO
from pathlib import Path
import shutil
import time

import pandas as pd
import requests

from observe.runs import file_sha
from scripts import review_strategy_batch89 as stage
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch89/20261009-all-weather-events')
REFERENCE = Path('repo/akshare/akshare/fund/fund_em.py')
SYMBOLS = ('513100', '510300', '518880', '511010')
MAX_BYTES = 2 * 1024 * 1024


def parse_tables(text):
    tables = pd.read_html(StringIO(text), displayed_only=False)
    # Preserve supplier labels/cells as textual candidates; do not infer events or usable times.
    return [{'columns': [str(c) for c in frame.columns], 'rows': frame.astype(str).values.tolist()}
            for frame in tables]


def probe(directory, session=None):
    receipt_path = Path('docs/handoff/2026-10-09-batch89-stage-verification.json')
    receipt = read(receipt_path)
    require(receipt['formal_acceptance'] is False and receipt['source_count'] == 695, 'Parent scope changed')
    for group in ('implementation_sha256', 'evidence_sha256'):
        require(all(file_sha(name) == sha for name, sha in receipt[group].items()), 'Parent source stage changed')
    require(REFERENCE.is_file(), 'Restore API project reference')
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(REFERENCE, directory/'fund_em.reference.py')
    save(directory/'input-binding.json', {'receipt': str(receipt_path), 'receipt_sha256': file_sha(receipt_path),
        'reference': str(REFERENCE), 'reference_sha256': file_sha(REFERENCE), 'snapshot': stage.SNAPSHOT,
        'source_SHA256': list(stage.SOURCE_SHA), 'symbols': list(SYMBOLS),
        'policy': {'timeout': [4, 8], 'retries': 0, 'redirects': False, 'proxy_env': False,
                   'max_bytes_per_page': MAX_BYTES, 'overall_budget_seconds': 45,
                   'stop_domain_after_DNS_failure': True},
        'SDK_not_called': True,
        'SDK_path_difference': 'fund_open_fund_info_em first fetches/executes pingzhongdata JS; only its event-page URL reused',
        'strict_usable': False, 'not_a_backtest': True})
    client = session or requests.Session()
    client.trust_env = False
    results = []; stop_domain = False; deadline = time.monotonic() + 45
    try:
        for symbol in SYMBOLS:
            url = f'https://fundf10.eastmoney.com/fhsp_{symbol}.html'
            row = {'symbol': symbol, 'url': url, 'attempted': False, 'strict_usable': False}
            if stop_domain or time.monotonic() >= deadline:
                reason = 'Same-domain DNS failure; no repeated request' if stop_domain else 'Overall time budget exhausted'
                results.append({**row, 'status': 'not_requested', 'reason': reason})
                continue
            row['attempted'] = True
            response = None
            try:
                response = client.get(url, timeout=(4, 8), allow_redirects=False, stream=True)
                row['http_status'] = response.status_code
                response.raise_for_status()
                require(response.status_code == 200, 'Non-200 response or redirect not followed')
                blocks = []; size = 0
                for block in response.iter_content(chunk_size=65536):
                    require(time.monotonic() < deadline, 'Overall time budget exhausted')
                    size += len(block)
                    require(size <= MAX_BYTES, 'Event page exceeds bounded response limit')
                    blocks.append(block)
                payload = b''.join(blocks)
                path = directory/(symbol+'.html'); path.write_bytes(payload)
                row.update(raw_file=str(path), raw_sha256=file_sha(path), raw_bytes=len(payload))
                encoding = requests.models.chardet.detect(payload).get('encoding') or 'utf-8'
                text = payload.decode(encoding, errors='strict')
                row['decode_encoding'] = encoding
                tables = parse_tables(text)
                parsed = directory/(symbol+'.tables.json')
                save(parsed, {'tables': tables, 'raw_sha256': row['raw_sha256'],
                    'current_supplier_page_only': True, 'event_chain_complete': False,
                    'historical_publication_times_proven': False, 'strict_usable': False})
                row.update(status='candidate', tables_file=str(parsed), tables_sha256=file_sha(parsed),
                           table_count=len(tables), table_rows=sum(len(t['rows']) for t in tables))
            except (requests.RequestException, ValueError, UnicodeError, LookupError) as exc:
                message = str(exc)
                stop_domain = isinstance(exc, requests.ConnectionError) and any(
                    term in message for term in ('NameResolutionError', 'name resolution', 'Failed to resolve'))
                row.update(status='failed', exception=type(exc).__name__, message=message,
                           DNS_failure=stop_domain)
            finally:
                if response is not None:
                    response.close()
            results.append(row)
            save(directory/(symbol+'.request.json'), row)
            print({'symbol': symbol, 'status': row['status'], 'DNS_failure': stop_domain}, flush=True)
    finally:
        client.close()
    summary = {'results': results, 'attempted_requests': sum(r['attempted'] for r in results),
        'responses': sum('http_status' in r for r in results),
        'candidate_pages': sum(r['status'] == 'candidate' for r in results),
        'strict_usable': False, 'not_a_backtest': True, 'new_trading_backtests': 0,
        'limits': ['Current supplier event tables are candidates; identity/units/effective/available times need primary proof',
                   'No price correction, adjustment-chain construction, standard publication or original ES calculation',
                   'Raw page and parse SHA retained when available; missing payloads never replaced']}
    save(directory/'probe-results.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(probe(args.directory))
