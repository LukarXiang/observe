"""Bounded existing AKShare query contracts, without executing source credentials."""
import argparse
import importlib.util
import json
from pathlib import Path

import requests

from observe.runs import file_sha
from scripts.review_strategy_batch18 import save
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch98/20261011-bounded-northbound-probes')
ENDPOINT = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
QUERIES = {
    'quota_shanghai': {'reportName': 'RPT_MUTUAL_DEAL_HISTORY', 'filter': '(MUTUAL_TYPE="001")'},
    'quota_shenzhen': {'reportName': 'RPT_MUTUAL_DEAL_HISTORY', 'filter': '(MUTUAL_TYPE="003")'},
    'holding2020': {'reportName': 'RPT_MUTUAL_STOCK_NORTHSTA',
        'filter': '(INTERVAL_TYPE="1")(MUTUAL_TYPE in ("001","003"))(TRADE_DATE="2020-09-30")'}}


def probe():
    require(not DIRECTORY.exists(), 'Immutable probe exists')
    DIRECTORY.mkdir(parents=True)
    source = Path('repo/akshare/akshare/stock_feature/stock_hsgt_em.py')
    save(DIRECTORY/'prepared.json', {'reference': str(source), 'source_sha256': file_sha(source),
        'queries': QUERIES, 'max_requests': len(QUERIES), 'timeout': [10, 15],
        'query_contract': 'Same public report/filter as AKShare; one first page each, no pagination',
        'holding_transport_change': 'HTTPS instead of archived http endpoint; original data semantics unproven',
        'tushare_installed': importlib.util.find_spec('tushare') is not None,
        'tushare_credentials_requested_or_used': False,
        'index_turnover_free_float': 'Original authenticated index_dailybasic absent from active Store; no ordinary turnover proxy'})
    result = []
    with requests.Session() as session:
        session.trust_env = False
        for name, query in QUERIES.items():
            params = {'sortColumns': 'TRADE_DATE', 'sortTypes': '-1', 'pageSize': '1000',
                      'pageNumber': '1', 'columns': 'ALL', 'source': 'WEB', 'client': 'WEB', **query}
            row = {'name': name, 'parameters': params, 'endpoint': ENDPOINT,
                   'wire_response': False, 'rows': 0, 'strict_usable': False, 'published': False}
            try:
                response = session.get(ENDPOINT, params=params, timeout=(10, 15))
                dest = DIRECTORY/(name+'.bin')
                with dest.open('xb') as stream: stream.write(response.content)
                row.update(wire_response=True, http_status=response.status_code,
                    response_file=str(dest), response_sha256=file_sha(dest))
                response.raise_for_status(); doc = response.json(); data = doc.get('result') or {}
                rows = data.get('data') or []
                row.update(status='candidate' if rows else 'empty', rows=len(rows),
                           pages=data.get('pages'), fields=sorted(set().union(*(r.keys() for r in rows))))
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
            save(DIRECTORY/(name+'.json'), row); result.append(row)
    summary = {'status': 'archived', 'requests': len(result), 'results': result,
        'wire_responses': sum(r['wire_response'] for r in result), 'candidate_rows': sum(r['rows'] for r in result),
        'published': False, 'original_quota_mapping_or_unit_proven': False, 'new_trading_backtests': 0,
        'limits': ['First-page candidates cannot prove full history or decision-time availability',
            'No quota_daily version schedule or JoinQuant link/share_ratio mapping verified',
            'Free-float index turnover, historical CSI300 pool and fund redemption inputs remain absent']}
    save(DIRECTORY/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['probe']); parser.parse_args()
    print(json.dumps(probe(), ensure_ascii=False))
