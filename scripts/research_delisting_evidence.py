"""Bounded first-party evidence collection for two unresolved holdings."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/staging/strategies-batch11/20261005-delisting-evidence')
    parser.add_argument('--supplement', action='store_true')
    parser.add_argument('--destination', action='store_true')
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.cninfo.com.cn/'})
    entries = []

    def fetch(name, url, payload=None):
        entry = {'name': name, 'url': url, 'payload': payload,
                 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        try:
            response = session.request('POST' if payload else 'GET', url, data=payload, timeout=(10, 20))
            extension = '.pdf' if response.content.startswith(b'%PDF') else '.bin'
            path = directory / (name + extension)
            with path.open('xb') as stream:
                stream.write(response.content)
            entry.update(status=response.status_code, path=str(path), bytes=len(response.content),
                         final_url=response.url, sha256=hashlib.sha256(response.content).hexdigest())
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            return None
        finally:
            entries.append(entry)
            print(name, entry.get('status'), entry.get('bytes'), entry.get('error', ''), flush=True)

    downloaded = set()
    metadata = []
    mapped = fetch('cninfo-stock-map', 'https://www.cninfo.com.cn/new/data/szse_stock.json')
    orgs = {x['code']: x['orgId'] for x in mapped.json()['stockList']} if mapped is not None else {}
    queries = [
        ('000662', '', '2021-02-01~2021-06-30'),
        ('000662', '确权', '2021-01-01~2026-09-29'),
        ('000662', '转让', '2021-01-01~2026-09-29'),
        ('600837', '', '2025-02-01~2025-03-31'),
    ]
    if args.supplement:
        queries = [('000662', '终止', '2021-01-01~2021-12-31'),
                   ('600837', '换股', '2025-01-01~2025-04-30')]
    if args.destination:
        queries = [('601211', '换股', '2025-03-01~2025-03-31')]
    for index, (code, keyword, dates) in enumerate(queries):
        payload = {'pageNum': 1, 'pageSize': 100, 'column': 'szse', 'tabName': 'fulltext',
                   'plate': '', 'stock': code + ',' + orgs.get(code, ''), 'searchkey': keyword,
                   'secid': '', 'category': '', 'trade': '', 'seDate': dates,
                   'sortName': 'time', 'sortType': 'asc', 'isHLtitle': 'true'}
        response = fetch(f'query-{code}-{index}', 'https://www.cninfo.com.cn/new/hisAnnouncement/query', payload)
        if response is None:
            continue
        body = response.json()
        print('TOTAL', code, body.get('totalAnnouncement'), flush=True)
        for item in body.get('announcements') or []:
            title = BeautifulSoup(item['announcementTitle'], 'html.parser').get_text()
            metadata.append(item | {'plain_title': title})
            if not any(word in title for word in ['终止上市并摘牌', '股票终止上市', '确权', '转让',
                                                   '实施', '现金选择权', '上市流通', '换股结果']):
                continue
            if item.get('adjunctSize', 0) > 3000 or item['announcementId'] in downloaded:
                continue
            downloaded.add(item['announcementId'])
            fetch(f"announcement-{code}-{item['announcementId']}",
                  'https://static.cninfo.com.cn/' + item['adjunctUrl'])
    if not args.supplement and not args.destination:
        fetch('company-home-legacy-tls', 'https://www.htsec.com')
        for code, dates in [('000662', '2021-01-01~2021-12-31'), ('600837', '2025-01-01~2025-04-30')]:
            fetch('query-code-only-' + code, 'https://www.cninfo.com.cn/new/hisAnnouncement/query',
                  {'pageNum': 1, 'pageSize': 30, 'column': 'szse', 'tabName': 'fulltext',
                   'stock': code, 'searchkey': '', 'seDate': dates})
            fetch('top-search-failure-' + code, 'https://www.cninfo.com.cn/new/information/topSearch/detailOfQuery',
                  {'keyWord': code, 'maxNum': 10})
    for name, body in [('manifest', {'scope': 'first-party research; no cash settlement assumed',
                                     'responses': entries}), ('announcements', metadata)]:
        with (directory / (name + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(body, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
