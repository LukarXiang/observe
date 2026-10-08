"""Freeze official EMA documentation and SDK evidence for source review."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main() -> None:
    directory = Path('data/staging/strategies-batch11/20261005-ema-evidence')
    directory.mkdir(parents=True, exist_ok=False)
    source = Path('repo/量化策略源代码/2022年度精选策略/55.【策略研发】三进兵策略（变形版）.txt')
    cached = Path('data/staging/strategies-batch7/20261005-rsi-slots/joinquant-api-response.json')
    local = []
    for name, path in [('source55', source), ('cached-official-api', cached)]:
        data = path.read_bytes()
        with (directory / (name + '.bin')).open('xb') as stream:
            stream.write(data)
        local.append({'source': str(path), 'copy': str(directory / (name + '.bin')),
                      'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    session = requests.Session()
    session.trust_env = False
    entries = []
    sources = [
        ('technicalanalysis', 'https://www.joinquant.com/help/api/getContent?name=technicalanalysis'),
        ('faq', 'https://www.joinquant.com/help/api/getContent?name=faq'),
        ('dynamic-adjustment-shell', 'https://www.joinquant.com/view/community/detail/48502bfa85355991258093e990d74f35'),
        ('sdk-technical-analysis', 'https://raw.githubusercontent.com/JoinQuant/jqdatasdk/master/jqdatasdk/technical_analysis.py'),
        ('sdk-file-list', 'https://api.github.com/repos/JoinQuant/jqdatasdk/contents/jqdatasdk'),
    ]
    for name, url in sources:
        entry = {'name': name, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        try:
            response = session.get(url, timeout=(10, 20))
            path = directory / (name + '.bin')
            with path.open('xb') as stream:
                stream.write(response.content)
            entry.update(status=response.status_code, path=str(path), bytes=len(response.content),
                         final_url=response.url, sha256=hashlib.sha256(response.content).hexdigest())
            response.raise_for_status()
            if name in {'technicalanalysis', 'faq'}:
                body = response.json()
                if body.get('code') != '00000' or not isinstance(body.get('data'), str):
                    raise ValueError('Official document data absent')
                with (directory / (name + '.txt')).open('x', encoding='utf-8') as stream:
                    stream.write(BeautifulSoup(body['data'], 'html.parser').get_text('\n', strip=True))
        except (requests.RequestException, ValueError) as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        entries.append(entry)
        print(name, entry.get('status'), entry.get('bytes'), entry.get('error', ''), flush=True)
    with (directory / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'EMA source review; no published data or implementation',
                   'local_inputs': local, 'responses': entries}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
