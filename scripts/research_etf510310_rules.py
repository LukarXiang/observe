"""Archive first-party ETF evidence without publishing or replacing data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main() -> None:
    directory = Path('data/staging/strategies-batch10/ETF证据/20261005-official')
    directory.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Mozilla/5.0'
    entries = []

    def fetch(name: str, url: str, payload: dict | None = None) -> requests.Response | None:
        entry = {'name': name, 'url': url, 'payload': payload,
                 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        try:
            response = session.request('POST' if payload else 'GET', url, json=payload, timeout=(10, 25))
            path = directory / (name + ('.pdf' if response.content.startswith(b'%PDF') else '.bin'))
            with path.open('xb') as stream:
                stream.write(response.content)
            entry.update(status=response.status_code, final_url=response.url,
                         headers=dict(response.headers), path=str(path), size=len(response.content),
                         sha256=hashlib.sha256(response.content).hexdigest())
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            return None
        finally:
            entries.append(entry)
            print(name, entry.get('status'), entry.get('size'), entry.get('error', ''), flush=True)

    product = fetch('efunds-product', 'https://www.efunds.com.cn/fund/510310.shtml')
    if product is not None:
        with (directory / 'efunds-product.txt').open('x', encoding='utf-8') as stream:
            stream.write(BeautifulSoup(product.content, 'html.parser').get_text('\n', strip=True))
    api = 'https://api.efunds.com.cn/xcowch/front/contents'
    common = {'siteID': '1', 'catalogAlias': 'xxplflwj,xxpldqgg,xxpllsgg',
              'fundCode': '510310', 'isIncludeTransFund': 'Y', 'pageSize': 100, 'pageIndex': 0}
    for name, extra in [('merger', {'title': '合并'}), ('dividend', {'title': '分红'}),
                        ('prospectus2021', {'title': '招募说明书', 'prop1': '2021-01-01,2021-12-31'})]:
        response = fetch('efunds-list-' + name, api, common | extra)
        if response is None:
            continue
        for item in response.json()['data']['data']:
            if name == 'dividend' and item['prop1'] < '2024-01-01':
                continue
            if name == 'prospectus2021' and item != response.json()['data']['data'][0]:
                continue
            fetch(f"efunds-{item['prop1']}-{item['id']}", item['path'])
    fetch('efunds-bonus-api', 'https://api.efunds.com.cn/xcowch/front/fundShareBonus/list',
          {'fundCode': '510310', 'pageIndex': 0, 'pageSize': 100, 'startDate': '', 'endDate': ''})
    fetch('sse-etf-basic', 'https://www.sse.com.cn/assortment/fund/list/etfinfo/basic/index.shtml?FUNDID=510310')
    fetch('sse-etf-overview', 'https://www.sse.com.cn/assortment/fund/etf/overview/')
    fetch('sse-etf-trading', 'https://www.sse.com.cn/assortment/fund/etf/trading/')
    fetch('sse-etf-knowledge', 'https://www.sse.com.cn/assortment/fund/etf/knowledge/')
    with (directory / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'first-party research evidence only; unpublished', 'responses': entries},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
