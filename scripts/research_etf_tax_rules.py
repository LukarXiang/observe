"""Archive official tax sources found through the government public search."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup


def main() -> None:
    directory = Path('data/staging/strategies-batch10/ETF证据/20261005-government-tax')
    directory.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    entries = []

    def fetch(name, url, payload=None, headers=None):
        entry = {'name': name, 'url': url, 'payload': payload,
                 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        try:
            response = session.request('POST' if payload else 'GET', url, json=payload,
                                       headers=headers, timeout=(10, 25))
            path = directory / (name + '.bin')
            with path.open('xb') as stream:
                stream.write(response.content)
            entry.update(status=response.status_code, path=str(path), size=len(response.content),
                         sha256=hashlib.sha256(response.content).hexdigest(), final_url=response.url)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            return None
        finally:
            entries.append(entry)
            print(name, entry.get('status'), entry.get('size'), entry.get('error', ''), flush=True)

    script = fetch('government-search-js', 'https://sousuo.www.gov.cn/sousuo/search.js')
    if script is not None:
        # The site's public frontend encrypts its public application identifier with RSA.
        match = re.search(r'-----END PUBLIC KEY-----"\),r.encrypt\(e\)\}\("([A-Za-z0-9+/=]+)","([a-z0-9]+)"', script.text)
        if match is None:
            raise ValueError('Public government search frontend parameters changed')
        result = subprocess.run(['node', '-e',
                                 "const c=require('node:crypto');let b='';process.stdin.on('data',x=>b+=x);process.stdin.on('end',()=>{const a=JSON.parse(b);const k='-----BEGIN PUBLIC KEY-----\\n'+a[0]+'\\n-----END PUBLIC KEY-----';process.stdout.write(c.publicEncrypt({key:k,padding:c.constants.RSA_PKCS1_PADDING},Buffer.from(a[1])).toString('base64'));});"],
                                input=json.dumps(match.groups()), text=True, capture_output=True, check=True, timeout=10)
        headers = {'athenaAppName': quote('国网搜索'), 'athenaAppKey': quote(result.stdout, safe=''),
                   'Referer': 'https://sousuo.www.gov.cn/'}
        api = 'https://sousuoht.www.gov.cn/athena/forward/2B22E8E39E850E17F95A016A74FCB6B673336FA8B6FEC0E2955907EF9AEE06BE'
        for index, word in enumerate(['中华人民共和国印花税法', '证券投资基金税收问题', '财税字 1998 55']):
            payload = {'code': '17da70961a7', 'dataTypeId': '107', 'searchWord': word,
                       'searchBy': 'title', 'orderBy': 'time', 'pageNo': 1, 'pageSize': 20,
                       'trackTotalHits': True, 'filters': []}
            response = fetch(f'search-{index}', api, payload, headers)
            if response is None:
                continue
            body = response.json()
            print('API', body.get('resultCode'), flush=True)
            items = body.get('result', {}).get('data', {}).get('middle', {}).get('list', [])
            for number, item in enumerate(items[:10]):
                url = item.get('url', '')
                title = BeautifulSoup(item.get('title', ''), 'html.parser').get_text()
                if not urlparse(url).hostname or not urlparse(url).hostname.endswith('.gov.cn'):
                    continue
                if not any(term in title for term in ['印花税法', '证券投资基金税收']):
                    continue
                article = fetch(f'article-{index}-{number}', url)
                if article is not None:
                    with (directory / f'article-{index}-{number}.txt').open('x', encoding='utf-8') as stream:
                        stream.write(BeautifulSoup(article.content, 'html.parser').get_text('\n', strip=True))
    with (directory / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'unpublished tax research evidence', 'responses': entries}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
