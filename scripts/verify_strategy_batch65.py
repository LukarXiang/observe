"""Offline read-back and old/new raw PCF comparison for batch65 evidence."""
import argparse
from datetime import date
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch

import pandas as pd
import requests

from observe.data.sources.baostock import BaoStock
from observe.runs import file_sha, write_json
from scripts.probe_strategy_batch65 import END, SECURITIES, audit
from scripts.run_strategy_batch4 import read


def archive_keys(frame):
    dates = sorted(frame.date.unique()); codes = sorted(frame.code.unique())
    for day in dates:
        try: parsed = date.fromisoformat(day)
        except (TypeError, ValueError) as exc: raise ValueError('Invalid PCF date') from exc
        if parsed.isoformat() != day or not date(2024, 3, 29) <= parsed <= date.fromisoformat(END):
            raise ValueError('PCF date outside canonical requested range')
    expected = {f'{exchange.lower()}.{code}' for code, exchange in SECURITIES.items()}
    if not set(codes) <= expected: raise ValueError('Unexpected PCF security')
    return dates, codes


def verify(root, directory):
    destination = directory / 'offline-verification-v2.json'
    if destination.exists(): raise FileExistsError(destination)
    frozen_driver = directory / 'driver.py'
    if file_sha(frozen_driver) != file_sha('scripts/probe_strategy_batch65.py'):
        raise ValueError('Driver differs from frozen batch65 copy')
    def forbidden(*args, **kwargs): raise AssertionError('Network forbidden during read-back')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), \
            patch.object(socket.socket, 'connect', forbidden), tempfile.TemporaryDirectory() as scratch:
        check = Path(scratch)
        (check / 'probes.json').write_bytes((directory / 'probes.json').read_bytes())
        repeated = audit(check)
        if repeated != read(directory / 'coverage-audit.json'): raise ValueError('Offline audit differs')
        probes = read(directory / 'probes.json')['probes']
        incoming = [pd.read_parquet(r['raw_file']) for r in probes
                    if r['case'].startswith('pcf-') and r['status'] == 'success']
        if not incoming: raise ValueError('No successful PCF inputs to compare')
        current = pd.concat(incoming, ignore_index=True)
        dates, codes = archive_keys(current)
        keys = ['date', 'code']; fields = keys + ['close', 'peTTM', 'pbMRQ', 'pcfNcfTTM']
        originals = []; sources = []; missing = []
        for day in dates:
            base = Path(root) / 'raw/baostock/daily_market'
            path = base / f'{day}.parquet'
            if not path.resolve().is_relative_to(base.resolve()): raise ValueError('PCF archive path escapes source directory')
            if not path.is_file(): missing.append(str(path)); continue
            frame = pd.read_parquet(path, columns=fields, filters=[('code', 'in', codes)])
            sources.append({'file': str(path), 'sha256': file_sha(path), 'selected_rows': len(frame)})
            originals.append(frame)
        old = pd.concat(originals, ignore_index=True) if originals else pd.DataFrame(columns=fields)
        if current.duplicated(keys).any() or old.duplicated(keys).any(): raise ValueError('Duplicate PCF keys')
        left = old[fields].sort_values(keys).reset_index(drop=True)
        right = current[fields].sort_values(keys).reset_index(drop=True)
        comparison = 'match'; error = None
        try: pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError as exc: comparison = 'different'; error = str(exc)[:2000]
        result = {'status': 'ok', 'scope': 'offline_evidence_readback_not_strategy_reproduction',
                  'network_disabled': True, 'audit_matches': True,
                  'date_and_security_scope_validated': True,
                  'verification_driver_sha256': file_sha(__file__),
                  'driver_sha256': file_sha(frozen_driver), 'audit_sha256': file_sha(directory / 'coverage-audit.json'),
                  'old_raw_comparison': {'result': comparison, 'missing_files': missing,
                                         'old_rows': len(old), 'new_rows': len(current), 'fields': fields,
                                         'codes': codes, 'raw_files': sources, 'error': error,
                                         'limits': 'A match now does not prove the historical as-of version'},
                  'strict_usable': False, 'not_a_backtest': True}
    write_json(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    result = verify(args.root, Path(args.directory))
    print({k: v for k, v in result.items() if k != 'old_raw_comparison'})
    print({k: v for k, v in result['old_raw_comparison'].items() if k != 'raw_files'})
