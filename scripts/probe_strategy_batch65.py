"""Batch65 bounded multi-security dependency evidence; never publishes inputs."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import inspect
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
from observe.runs import canonical, file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import checkpoint, protect, read


SECURITIES = {'000001': 'SZ', '600837': 'SH', '002594': 'SZ'}
CASES = tuple(f'{endpoint}-{code}' for code in SECURITIES for endpoint in ('cninfo', 'eastmoney', 'pcf'))
START, END = '2001-01-01', '2024-06-28'


def date_audit(frame, column):
    values = pd.to_datetime(frame[column], errors='coerce')
    present = values.dropna()
    return {'missing': int(values.isna().sum()), 'after_requested_end': int((values > pd.Timestamp(END)).sum()),
            'min': None if present.empty else str(present.min().date()),
            'max': None if present.empty else str(present.max().date())}


def shares_audit(frame, provider):
    change = '变动日期' if provider == 'cninfo' else '变更日期'
    total = pd.to_numeric(frame['总股本'], errors='coerce')
    result = {'rows': len(frame), 'dates': date_audit(frame, change),
              'duplicate_change_dates': int(frame[change].duplicated(keep=False).sum()),
              'missing_total': int(total.isna().sum()), 'nonpositive_total': int((total <= 0).sum()),
              'strict_usable': False, 'numeric_unit_not_inferred': True}
    if provider == 'cninfo':
        declared = pd.to_datetime(frame['公告日期'], errors='coerce')
        changed = pd.to_datetime(frame[change], errors='coerce')
        lag = (declared - changed).dt.days
        result.update(announcements=date_audit(frame, '公告日期'),
                      announcement_before_change=int((lag < 0).sum()),
                      announcement_after_change=int((lag > 0).sum()),
                      announcement_same_date=int((lag == 0).sum()))
    return result


def pcf_audit(frame):
    values = pd.to_numeric(frame['pcfNcfTTM'], errors='coerce')
    return {'rows': len(frame), 'dates': date_audit(frame, 'date'),
            'duplicate_code_dates': int(frame.duplicated(['code', 'date'], keep=False).sum()),
            'missing_pcf': int(values.isna().sum()), 'positive_pcf': int((values > 0).sum()),
            'negative_pcf': int((values < 0).sum()), 'zero_pcf': int((values == 0).sum()),
            'strict_usable': False, 'historical_versions_not_proven': True}


def ratio_audit(cninfo, eastmoney):
    # Compare only dates with one distinct positive value in each response.
    def unique(frame, column):
        work = pd.DataFrame({'date': pd.to_datetime(frame[column], errors='coerce'),
                             'total': pd.to_numeric(frame['总股本'], errors='coerce')})
        work = work[work.date.notna() & work.total.gt(0)]
        grouped = work.groupby('date').total
        return grouped.first()[grouped.nunique().eq(1)]
    c = unique(cninfo, '变动日期'); e = unique(eastmoney, '变更日期')
    both = c.index.intersection(e.index)
    ratios = e.loc[both] / c.loc[both]
    return {'unambiguous_common_dates': len(both),
            'observed_ratios': sorted(set(float(v) for v in ratios)),
            'examples': [{'date': str(day.date()), 'cninfo_total': float(c.loc[day]),
                          'eastmoney_total': float(e.loc[day]), 'ratio': float(ratios.loc[day])}
                         for day in both[:5]],
            'unit_conversion_authorized': False, 'strict_usable': False}


def worker(root, directory, case):
    provider, code = case.split('-'); evidence = directory / case
    evidence.mkdir(exist_ok=False); socket.setdefaulttimeout(15)
    result = {'case': case, 'status': 'failed', 'strict_usable': False, 'wire_responses': [],
              'requested_at': datetime.now().isoformat(), 'symbol': code}
    original = requests.sessions.Session.request

    def request(session, method, url, **kwargs):
        session.trust_env = False; kwargs['timeout'] = 20
        response = original(session, method, url, **kwargs)
        target = evidence / f'response-{len(result["wire_responses"]):03d}.bin'
        target.write_bytes(response.content)
        result['wire_responses'].append({'url': response.url, 'method': method,
                                         'status_code': response.status_code,
                                         'file': str(target), 'sha256': file_sha(target)})
        return response

    try:
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request):
            if provider == 'pcf':
                symbol = f'{SECURITIES[code].lower()}.{code}'
                fields = 'date,code,close,peTTM,pbMRQ,pcfNcfTTM'
                params = {'code': symbol, 'start': '2024-03-29', 'end': END, 'fields': fields,
                          'frequency': 'd', 'adjustflag': '3'}
                result['request'] = params
                with BaoStock(root).session() as source:
                    frame = source._rows('query_history_k_data_plus', params,
                                         lambda: source.bs.query_history_k_data_plus(
                                             symbol, fields, start_date=params['start'], end_date=END,
                                             frequency='d', adjustflag='3'))
            else:
                import akshare as ak
                fn = ak.stock_share_change_cninfo if provider == 'cninfo' else ak.stock_zh_a_gbjg_em
                parser_source = Path(inspect.getsourcefile(fn))
                parser_copy = evidence / 'parser.original.py'
                parser_copy.write_bytes(parser_source.read_bytes())
                result['parser_file'] = str(parser_copy)
                result['parser_sha256'] = file_sha(parser_copy)
                result['request'] = ({'symbol': code, 'start_date': START.replace('-', ''),
                                      'end_date': END.replace('-', '')} if provider == 'cninfo'
                                     else {'symbol': f'{code}.{SECURITIES[code]}'})
                frame = fn(**result['request'])
        path = raw.save(root, 'dependency_probe_batch65', provider, f'{directory.name}-{code}', frame)
        result.update(status='success' if len(frame) else 'empty', rows=len(frame), columns=list(frame.columns),
                      raw_file=str(path), raw_sha256=file_sha(path),
                      sample=canonical(frame.head(2).astype(object).where(pd.notna(frame.head(2)), None).to_dict('records')))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    write_json(evidence / 'result.json', result)
    return result


def probe(root, directory):
    destination = directory / 'probes.json'
    if destination.exists() or any((directory / c).exists() for c in CASES):
        raise FileExistsError('Use a new directory for each probe attempt')
    results = []
    for case in CASES:
        try:
            child = subprocess.run([sys.executable, '-m', 'scripts.probe_strategy_batch65', '--root', root,
                                    '--directory', str(directory), '--worker', case],
                                   capture_output=True, text=True, timeout=60)
            (directory / f'{case}.log').write_text(child.stderr, encoding='utf-8')
            path = directory / case / 'result.json'
            result = read(path) if path.exists() else {'case': case, 'status': 'failed',
                                                       'returncode': child.returncode, 'error': child.stderr[-1000:]}
        except subprocess.TimeoutExpired as exc:
            (directory / f'{case}.log').write_bytes(exc.stderr or b'')
            result = {'case': case, 'status': 'timeout', 'timeout_seconds': 60, 'strict_usable': False}
        results.append(result)
        print(json.dumps({'case': case, 'status': result['status'], 'rows': result.get('rows')}, ensure_ascii=False), flush=True)
    write_json(destination, {'probes': results, 'not_a_backtest': True, 'strict_usable': False})


def audit(directory):
    output = directory / 'coverage-audit.json'
    if output.exists(): raise FileExistsError(output)
    results = read(directory / 'probes.json')['probes']; frames = {}; audits = []
    for result in results:
        case = result['case']; provider, _ = case.split('-')
        detail = {'case': case, 'status': result['status'], 'strict_usable': False}
        if result.get('parser_file') and file_sha(result['parser_file']) != result['parser_sha256']:
            raise ValueError(f'{case}: frozen parser SHA differs')
        if result['status'] in ('success', 'empty'):
            if file_sha(result['raw_file']) != result['raw_sha256']: raise ValueError(f'{case}: raw SHA differs')
            frame = pd.read_parquet(result['raw_file']); frames[case] = frame
            if len(frame) != result['rows'] or list(frame.columns) != result['columns']:
                raise ValueError(f'{case}: parsed archive differs')
            if len(frame): detail.update(pcf_audit(frame) if provider == 'pcf' else shares_audit(frame, provider))
        for wire in result.get('wire_responses', []):
            if file_sha(wire['file']) != wire['sha256']: raise ValueError(f'{case}: wire SHA differs')
            if provider == 'cninfo':
                doc = read(wire['file']); records = doc.get('records', [])
                detail['wire_records'] = len(records)
                detail['wire_count'] = doc.get('count'); detail['wire_total'] = doc.get('total')
                detail['latest_record_flags'] = sorted(set(str(r.get('F027C')) for r in records))
                detail['parser_drops_latest_record_flag'] = '最新记录标识' not in result.get('columns', [])
                if result['status'] == 'success' and len(records) != result['rows']:
                    raise ValueError(f'{case}: wire/parsed row count differs')
        audits.append(detail)
    ratios = {code: ratio_audit(frames[f'cninfo-{code}'], frames[f'eastmoney-{code}']) for code in SECURITIES
              if all(f'{p}-{code}' in frames and not frames[f'{p}-{code}'].empty for p in ('cninfo', 'eastmoney'))}
    result = {'status': 'ok', 'audits': audits, 'ratios': ratios, 'strict_usable': False,
              'not_a_backtest': True, 'probes_sha256': file_sha(directory / 'probes.json'),
              'limits': ['Three deliberately selected securities are not whole-market coverage',
                         'No historical revisions or availability timestamps proven',
                         'No shares, PCF or instrument-status partition published']}
    write_json(output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', required=True)
    parser.add_argument('--step', choices=('start', 'probe', 'audit', 'protect', 'archive'))
    parser.add_argument('--worker', choices=CASES)
    args = parser.parse_args(); directory = Path(args.directory)
    if args.worker: result = worker(args.root, directory, args.worker)
    elif args.step == 'start':
        checkpoint(args.root, directory, 'Batch65 multi-security dependency evidence started')
        (directory / 'driver.py').write_bytes(Path(__file__).read_bytes()); result = {'status': 'started'}
    elif args.step == 'probe': result = probe(args.root, directory)
    elif args.step == 'audit': result = audit(directory)
    elif args.step == 'protect': result = protect(args.root, directory)
    elif args.step == 'archive':
        if read(directory / 'coverage-audit.json')['status'] != 'ok' or read(directory / 'protection.json')['status'] != 'ok':
            raise ValueError('Audit and protection required before archiving')
        destination = directory / 'progress-final.json'
        if destination.exists(): raise FileExistsError(destination)
        result = archive(args.root, 'Batch65 multi-security dependencies archived; historical timing and coverage remain gated')
        write_json(destination, result)
    else: parser.error('--step or --worker is required')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
