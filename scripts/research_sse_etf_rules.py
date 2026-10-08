"""Archive exchange rule pages and their structured DOCX attachments."""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from xml.etree import ElementTree
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='data/staging/strategies-batch10/ETF证据/20261005-sse')
    parser.add_argument('--source', nargs=2, action='append')
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    session.headers['Referer'] = 'https://www.sse.com.cn/'
    entries = []

    def fetch(name, url):
        entry = {'name': name, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        try:
            response = session.get(url, timeout=(10, 25))
            extension = '.docx' if response.content.startswith(b'PK') else '.bin'
            path = directory / (name + extension)
            with path.open('xb') as stream:
                stream.write(response.content)
            entry.update(status=response.status_code, final_url=response.url, path=str(path),
                         size=len(response.content), sha256=hashlib.sha256(response.content).hexdigest())
            response.raise_for_status()
            if extension == '.docx':
                with ZipFile(io.BytesIO(response.content)) as archive:
                    tree = ElementTree.fromstring(archive.read('word/document.xml'))
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text = '\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:p', ns))
            else:
                text = BeautifulSoup(response.content, 'html.parser').get_text('\n', strip=True)
            with path.with_suffix('.txt').open('x', encoding='utf-8') as stream:
                stream.write(text)
            return response
        except (requests.RequestException, ValueError) as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            return None
        finally:
            entries.append(entry)
            print(name, entry.get('status'), entry.get('size'), entry.get('error', ''), flush=True)

    sources = args.source or [
        ('etf-home', 'https://www.sse.com.cn/assortment/fund/etf/home/'),
        ('fund-rules-index', 'https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/'),
        ('general-rules-index', 'https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/'),
        ('etf-rules2020', 'https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20250606_10781071.shtml'),
        ('trade-rules2026', 'https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml'),
        ('exchange-fees2026', 'https://www.sse.com.cn/services/tradingservice/charge/ssecharge/'),
        ('reprinted50etf-article', 'https://www.sse.com.cn/assortment/fund/etf/home/c/c_20151111_4011009.shtml'),
    ]
    for name, url in sources:
        response = fetch(name, url)
        if response is None:
            continue
        links = sorted({urljoin(url, a['href']) for a in BeautifulSoup(response.content, 'html.parser').find_all('a', href=True)
                        if a['href'].endswith('.docx')})
        for index, link in enumerate(links):
            fetch(f'{name}-attachment-{index}', link)
    with (directory / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'unpublished research evidence', 'responses': entries}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
