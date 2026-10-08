"""Bounded, evidence-only probes for historical shares, PCF and security names."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import requests

from observe.data import raw
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.runs import canonical, environment, file_sha, write_json


PROBES = ('bs_profit', 'bs_pcf', 'bs_universe', 'em_shares', 'cninfo_shares', 'sz_names', 'sina_names')


def checkpoint(root, output):
    store = Store(root)
    controls = [store.published_path, *sorted((store.root / 'snapshots').glob('*.json'))]
    controls += [p for p in sorted((store.root / 'runs').rglob('*')) if p.is_file() and p.suffix not in ('.sqlite', '.db', '.log') and 'sqlite-' not in p.name]
    references = store.protected()
    missing = [f for f in sorted(references) if not (store.root / f).is_file()]
    if missing: raise ValueError(f'Missing frozen partitions: {missing}')
    handoff = json.loads(Path('docs/handoff/observe-2026-10-04-batch2.manifest.json').read_text(encoding = 'utf-8'))
    absent = [entry['path'] for entry in handoff['files'] if not Path(entry['path']).is_file()]
    return {'captured_at': datetime.now().isoformat(), 'environment': environment(), 'published': store.published(),
            'controls': {p.relative_to(store.root).as_posix(): file_sha(p) for p in controls},
            'partitions': {f: {'size': (store.root / f).stat().st_size, 'mtime_ns': (store.root / f).stat().st_mtime_ns, 'sha256': file_sha(store.root / f)} for f in sorted(references)},
            'handoff_missing_files': absent, 'source_files': len(list(Path('repo/量化策略源代码').rglob('*.txt'))),
            'output': str(output)}


def worker(root, output, name):
    socket.setdefaulttimeout(15)
    evidence = output / name; evidence.mkdir()
    result = {'endpoint': name, 'status': 'failed', 'wire_responses': [], 'strict_usable': False}
    original = requests.sessions.Session.request

    def request(session, method, url, **kwargs):
        session.trust_env = False
        kwargs['timeout'] = 20
        response = original(session, method, url, **kwargs)
        path = evidence / f'response-{len(result["wire_responses"]):03d}.bin'
        path.write_bytes(response.content)
        result['wire_responses'].append({'url': url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response

    try:
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request):
            if name.startswith('bs_'):
                with BaoStock(root).session() as source:
                    if name == 'bs_profit':
                        frame = source._rows('query_profit_data', {'code': 'sh.600519', 'year': 2024, 'quarter': 1},
                                             lambda: source.bs.query_profit_data('sh.600519', year = 2024, quarter = 1))
                    elif name == 'bs_pcf':
                        fields = 'date,code,close,peTTM,pbMRQ,pcfNcfTTM'
                        frame = source._rows('query_history_k_data_plus', {'code': 'sh.600519', 'start': '2024-03-29', 'end': '2024-06-28', 'fields': fields},
                                             lambda: source.bs.query_history_k_data_plus('sh.600519', fields, start_date = '2024-03-29', end_date = '2024-06-28', frequency = 'd', adjustflag = '3'))
                    else:
                        frame = source._rows('query_all_stock', {'day': '2024-03-29'}, lambda: source.bs.query_all_stock(day = '2024-03-29'))
            else:
                import akshare as ak
                if name == 'em_shares': frame = ak.stock_zh_a_gbjg_em(symbol = '600519.SH')
                elif name == 'cninfo_shares': frame = ak.stock_share_change_cninfo(symbol = '600519', start_date = '20010101', end_date = '20240628')
                elif name == 'sz_names': frame = ak.stock_info_sz_change_name(symbol = '简称变更')
                else: frame = ak.stock_info_change_name(symbol = '000001')
        path = raw.save(root, 'dependency_probe', name, output.name, frame)
        result.update(status = 'success' if len(frame) else 'empty', rows = len(frame), columns = list(frame.columns),
                      raw_file = str(path), raw_sha256 = file_sha(path), sample = canonical(frame.head(3).astype(object).where(pd.notna(frame.head(3)), None).to_dict('records')))
        if len(frame):
            date_columns = [c for c in frame if 'date' in c.lower() or '日期' in c]
            result['date_ranges'] = {c: {'min': str(frame[c].dropna().min()), 'max': str(frame[c].dropna().max()), 'missing': int(frame[c].isna().sum())} for c in date_columns}
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    finally:
        write_json(evidence / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--root', default = 'data'); parser.add_argument('--output', required = True)
    parser.add_argument('--worker', choices = PROBES)
    args = parser.parse_args(); output = Path(args.output)
    if args.worker:
        print(json.dumps(worker(args.root, output, args.worker), ensure_ascii = False)); return
    output.mkdir(parents = True, exist_ok = False)
    baseline = checkpoint(args.root, output); write_json(output / 'baseline.json', baseline)
    print(json.dumps({'checkpoint': str(output), 'controls': len(baseline['controls']), 'partitions': len(baseline['partitions']),
                      'missing_handoff_files': len(baseline['handoff_missing_files'])}), flush = True)
    probes = []
    for name in PROBES:
        try:
            child = subprocess.run([sys.executable, __file__, '--root', args.root, '--output', str(output), '--worker', name], capture_output = True, text = True, timeout = 60)
            (output / f'{name}.log').write_text(child.stderr, encoding = 'utf-8')
            path = output / name / 'result.json'
            result = json.loads(path.read_text(encoding = 'utf-8')) if path.exists() else {'endpoint': name, 'status': 'failed', 'returncode': child.returncode, 'error': child.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            result = {'endpoint': name, 'status': 'timeout', 'timeout_seconds': 60, 'strict_usable': False}
        probes.append(result)
        write_json(output / 'probes.json', {'probes': probes, 'baseline_sha256': file_sha(output / 'baseline.json')})
        print(json.dumps(result, ensure_ascii = False), flush = True)


if __name__ == '__main__': main()
