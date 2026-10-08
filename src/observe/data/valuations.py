"""Immutable recovery of vendor-final daily valuation fields from bound raw files."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import secrets

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .locks import DATA_WRITER, operation_lock
from .standardize import instrument
from .store import Store, _atomic_json, fingerprint
from ..execution import InputBlocked, sessions
from ..factors.expr import TS1, TS2, TS_LAG
from ..runs import file_sha

VERSION = 'baostock_daily_valuation_final_v1'
FIELDS = ('ps_ttm', 'pcf_ncf_ttm')
SOURCE_FIELDS = {'psTTM': 'ps_ttm', 'pcfNcfTTM': 'pcf_ncf_ttm'}
TABLES = ('valuations_1d', 'valuation_coverage')
LIMITATION = {'kind': 'valuation_provider_final', 'detail': 'BaoStock最终估值，历史披露/修订版本未知；PS与净现金流PCF分开，平台等价性未证明'}


class ValuationImportConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    manifest: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def _require(ok, message):
    if not ok: raise ValueError(message)


def _new_json(path, value):
    if Path(path).exists(): raise FileExistsError(path)
    _atomic_json(path, value)


def normalize(raw, source_sha256, manifest_sha256, day):
    """Preserve blanks, signed values and paused rows; reject malformed inputs."""
    _require({'date', 'code', 'tradestatus', *SOURCE_FIELDS} <= set(raw), 'Missing valuation raw fields')
    codes = raw.code.astype('string')
    _require(codes.str.fullmatch(r'(sh|sz|bj)\.\d{6}', na=False).all(), 'Invalid valuation security code')
    dates = pd.to_datetime(raw.date, format='%Y-%m-%d', errors='raise').dt.date
    _require(dates.eq(day).all() and len(raw) > 0, 'Raw date differs or file empty')
    status = raw.tradestatus.astype('string')
    _require(status.isin(['0', '1']).all(), 'Invalid valuation trading status')
    out = pd.DataFrame({'date': dates, 'instrument': codes.map(instrument), 'is_trading': status.eq('1')})
    _require(not out.duplicated(['date', 'instrument']).any(), 'Duplicate valuation keys')
    for source, field in SOURCE_FIELDS.items():
        text = raw[source].astype('string').str.strip()
        missing = text.isna() | text.eq('')
        values = pd.to_numeric(text.where(~missing), errors='coerce').astype(float)
        _require(not ((~missing & values.isna()) | np.isinf(values)).any(), f'Malformed valuation {source}')
        out[field] = values
    out['source_sha256'], out['source_row'] = source_sha256, np.arange(1, len(raw) + 1, dtype='int64')
    out['source_manifest_sha256'], out['processing_version'] = manifest_sha256, VERSION
    out['provider'], out['version_evidence'], out['strict_usable'] = 'baostock', 'final_vendor_revision_unknown', False
    return out


def _bound_manifest(root, path, sha):
    _require(re.fullmatch(r'[0-9a-f]{64}', sha) is not None, 'Invalid manifest SHA256')
    payload = Path(path).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == sha, 'Manifest SHA256 differs')
    doc = json.loads(payload); entries = doc.get('source_files', [])
    _require(bool(entries) and not doc.get('missing_dates'), 'Incomplete raw manifest')
    seen = set(); bound = []
    for entry in entries:
        p = Path(entry['file']).resolve()
        _require(p.is_relative_to((Path(root) / 'raw' / 'baostock' / 'daily_market').resolve()) and p.suffix == '.parquet', 'Raw file outside daily-market archive')
        day = pd.Timestamp(entry['date']).date()
        _require(str(day) == entry['date'] and day not in seen and p.stem == str(day), 'Invalid/duplicate raw date')
        _require(type(entry['raw_rows']) is int and entry['raw_rows'] > 0 and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']), 'Invalid raw binding')
        _require(type(entry['columns']) is list and len(set(entry['columns'])) == len(entry['columns']), 'Invalid raw schema')
        seen.add(day); bound.append((day, p, entry))
    return payload, sorted(bound, key=lambda row: row[0])


def _verified_read(path, entry):
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == entry['sha256'], f'Raw SHA256 differs: {path}')
    raw = pd.read_parquet(io.BytesIO(payload))
    _require(len(raw) == entry['raw_rows'] and list(raw) == entry['columns'], f'Raw profile differs: {path}')
    return raw


def _verify_parts(store, state, parts, hashes):
    for table, mapping in parts.items():
        for key, entry in mapping.items():
            _require(state['tables'].get(table, {}).get(key) == entry, 'Imported valuation references differ')
            path = store.root / entry['file']
            _require(file_sha(path) == hashes[entry['file']], 'Imported valuation bytes differ')
            frame = pd.read_parquet(path)
            _require(len(frame) == entry['rows'] and fingerprint(frame) == entry['sha'], 'Imported valuation fingerprint differs')


def import_valuations(root, manifest, manifest_sha256, log=None):
    """Validate all bound daily rows and audit new tables before one publication."""
    cfg = ValuationImportConfig(manifest=str(manifest), manifest_sha256=manifest_sha256)
    store = Store(root)
    with operation_lock(root, DATA_WRITER):
        state = store.published(); payload, entries = _bound_manifest(root, cfg.manifest, cfg.manifest_sha256)
        cal = sessions(store.load_state(state, 'calendar'))
        expected = [d for d in cal if entries[0][0] <= d <= entries[-1][0]]
        _require(expected == [d for d, _, _ in entries], 'Raw dates differ from frozen trading calendar')
        basis = {'version': VERSION, 'manifest_sha256': cfg.manifest_sha256}
        iid = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:20]
        folder = store.root / 'raw' / 'valuation_imports' / iid
        receipt = folder / 'receipt.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text(encoding='utf-8'))
            _require(prior['basis'] == basis, 'Valuation receipt basis differs')
            for _, path, entry in entries: _require(file_sha(path) == entry['sha256'], f'Raw SHA256 differs: {path}')
            _verify_parts(store, {'tables': prior['parts']}, prior['parts'], prior['partition_bytes'])
            _require(file_sha(folder / 'manifest.json') == cfg.manifest_sha256 and file_sha(prior['audit']) == prior['audit_sha256'], 'Valuation receipt evidence differs')
            batch = json.loads((store.root / 'batches' / f"{prior['result']['batch_id']}.json").read_text(encoding='utf-8'))
            _require(batch['tables'] == prior['parts'], 'Prepared valuation batch differs')
            if batch['status'] == 'pending': store.publish(batch['batch_id'])
            _verify_parts(store, store.published(), prior['parts'], prior['partition_bytes'])
            return {**prior['result'], 'status': 'unchanged'}
        attempt = folder / f'attempt-{secrets.token_hex(8)}'; attempt.mkdir(parents=True, exist_ok=False)
        if (folder / 'manifest.json').exists():
            _require((folder / 'manifest.json').read_bytes() == payload, 'Frozen raw manifest differs')
        else: (folder / 'manifest.json').write_bytes(payload)
        parts = {t: {} for t in TABLES}; profiles = []; pending = []; month = None
        def flush():
            if not pending: return
            _require(month not in state['tables'].get('valuations_1d', {}) and month not in state['tables'].get('valuation_coverage', {}), 'Valuation month already published; use an explicit new version')
            frame = pd.concat(pending, ignore_index=True)
            parts['valuations_1d'][month] = store.write_partition('valuations_1d', month, frame)
            coverage = pd.DataFrame([r for r in profiles if r['month'] == month]).drop(columns='month')
            parts['valuation_coverage'][month] = store.write_partition('valuation_coverage', month, coverage)
            pending.clear()
        try:
            for k, (day, path, entry) in enumerate(entries):
                current = f'{day:%Y%m}'
                if month is not None and month != current: flush()
                month = current; raw = _verified_read(path, entry)
                frame = normalize(raw, entry['sha256'], cfg.manifest_sha256, day); pending.append(frame)
                profiles.append({'date': day, 'month': month, 'rows': len(frame), 'trading_rows': int(frame.is_trading.sum()),
                    **{f'{f}_missing': int(frame[f].isna().sum()) for f in FIELDS},
                    **{f'{f}_negative': int(frame[f].lt(0).sum()) for f in FIELDS},
                    'source_sha256': entry['sha256'], 'source_manifest_sha256': cfg.manifest_sha256,
                    'processing_version': VERSION, 'strict_usable': False})
                if log and (k + 1) % 100 == 0: log(f'Valuation recovery validated {k + 1}/{len(entries)} daily files')
            flush()
            hashes = {v['file']: file_sha(store.root / v['file']) for ps in parts.values() for v in ps.values()}
            _verify_parts(store, {'tables': parts}, parts, hashes)
            # Recheck sources immediately before publication; a changed archive is never accepted.
            _require(Path(manifest).read_bytes() == payload, 'Raw manifest changed during import')
            for _, path, entry in entries: _require(file_sha(path) == entry['sha256'], f'Raw changed during import: {path}')
            audit = attempt / 'audit.json'
            summary = {'status': 'passed', 'basis': basis, 'daily_files': len(entries), 'rows': sum(r['rows'] for r in profiles),
                'first': str(entries[0][0]), 'last': str(entries[-1][0]), 'strict_usable': False,
                'missing': {f: sum(r[f'{f}_missing'] for r in profiles) for f in FIELDS},
                'negative': {f: sum(r[f'{f}_negative'] for r in profiles) for f in FIELDS},
                'trading_rows': sum(r['trading_rows'] for r in profiles), 'parts': parts, 'partition_bytes': hashes, 'limitations': [LIMITATION]}
            _new_json(audit, summary)
            bid = store.write_batch(parts, f'{VERSION}: {iid}; vendor-final, strict blocked')
            result = {'status': 'prepared', 'import_id': iid, 'batch_id': bid, 'rows': summary['rows'], 'daily_files': len(entries),
                'first': summary['first'], 'last': summary['last'], 'missing': summary['missing'], 'negative': summary['negative'],
                'audit': str(audit), 'receipt': str(receipt), 'strict_usable': False}
            _new_json(receipt, {'basis': basis, 'parts': parts, 'partition_bytes': hashes, 'audit': str(audit), 'audit_sha256': file_sha(audit), 'result': result})
            store.publish(bid)
            return {**result, 'status': 'published'}
        except Exception as exc:
            _new_json(attempt / 'failed.json', {'basis': basis, 'error': f'{type(exc).__name__}: {exc}', 'validated_daily_files': len(profiles)})
            raise


def required_fields(expressions, parse):
    return sorted({f for expression in expressions for f in parse(expression).fields} & set(FIELDS))


def valuation_lookback(expressions, parse):
    """Count windows only along paths ending in valuation leaves."""
    def visit(node):
        if isinstance(node, ast.Name): return 0 if node.id in FIELDS else None
        if isinstance(node, ast.Call):
            fn = node.func.id
            args = node.args[:1] if fn in TS1 | TS_LAG else node.args[:2] if fn in TS2 else node.args
            paths = [visit(arg) for arg in args]
            widths = [value for value in paths if value is not None]
            if not widths: return None
            extra = node.args[-1].value if fn in TS_LAG else node.args[-1].value - 1 if fn in TS1 | TS2 else 0
            return max(widths) + extra
        widths = [value for child in ast.iter_child_nodes(node) if (value := visit(child)) is not None]
        return max(widths) if widths else None
    return max([visit(parse(e).tree) or 0 for e in expressions] + [0])


def check_valuation_calendar(calendar, start, end):
    """Unknown calendar dates cannot be mistaken for known closed sessions."""
    dates = pd.to_datetime(calendar.date).dt.date
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if not len(dates) or s < dates.min() or e > dates.max():
        raise InputBlocked([{'kind': 'valuation_calendar_range', 'detail': '请求估值区间超出冻结日历，不能返回较短区间', 'start': str(s), 'end': str(e)}])


def load_valuations(store, state, start, end, fields, policy):
    """Read verified sidecar partitions only; default policy cannot use final values."""
    _require(set(fields) <= set(FIELDS) and bool(fields), 'Unknown/empty valuation fields')
    if policy != 'provider_final': raise InputBlocked([{'kind': 'valuation_policy', 'detail': '估值历史版本未知；须显式 valuation_policy: provider_final 才能作探索研究'}])
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    _require(s <= e, 'Invalid valuation date range')
    calendar = store.load_state(state, 'calendar')
    check_valuation_calendar(calendar, s, e)
    used = {t: {p: v for p, v in state['tables'].get(t, {}).items() if f'{s:%Y%m}' <= p <= f'{e:%Y%m}'} for t in TABLES}
    if not all(used.values()): raise InputBlocked([{'kind': 'valuation_missing', 'detail': '快照没有所需估值表/覆盖表'}])
    for table, mapping in used.items():
        for part, entry in mapping.items():
            frame = pd.read_parquet(store.root / entry['file'])
            _require(fingerprint(frame) == entry['sha'] and len(frame) == entry['rows'] and not frame.duplicated(['date', 'instrument'] if table == 'valuations_1d' else ['date']).any(), 'Valuation partition differs')
    filters = [('date', '>=', s), ('date', '<=', e)]
    values = store.load_state(state, 'valuations_1d', parts=used['valuations_1d'], filters=filters)
    coverage = store.load_state(state, 'valuation_coverage', parts=used['valuation_coverage'], filters=filters)
    for frame in (values, coverage):
        frame['date'] = pd.to_datetime(frame.date).dt.date
        _require(frame.processing_version.eq(VERSION).all() and frame.strict_usable.eq(False).all(), 'Unknown valuation version/evidence')
    missing = sorted(set(d for d in sessions(calendar) if s <= d <= e) - set(coverage.date))
    if missing: raise InputBlocked([{'kind': 'valuation_coverage', 'detail': '所需估值交易日缺失，不缩短请求范围', 'dates': list(map(str, missing[:50])), 'count': len(missing)}])
    _require(not values.duplicated(['date', 'instrument']).any() and not coverage.date.duplicated().any(), 'Duplicate valuation partitions')
    actual = values.groupby('date').size().to_dict()
    _require(actual == coverage.set_index('date').rows.to_dict(), 'Valuation coverage row counts differ')
    for field in fields: _require(not np.isinf(values[field]).any(), 'Infinite valuation')
    return values, used


def attach_valuations(bars, values, fields, start, end):
    """Missing source rows block; source blanks stay NaN, paused rows stay masked."""
    b = bars.copy(); b['date'] = pd.to_datetime(b.date).dt.date
    existing = set(fields) & set(b)
    _require(not existing, f'Valuation sidecar shadows bars fields: {sorted(existing)}')
    merged = b.merge(values[['date', 'instrument', 'is_trading', *fields]].rename(columns={'is_trading': '_valuation_trading'}),
                     on=['date', 'instrument'], how='left', validate='one_to_one', indicator=True)
    required = merged.date.between(pd.Timestamp(start).date(), pd.Timestamp(end).date()) & merged.is_trading
    missing = required & merged._merge.ne('both')
    if missing.any(): raise InputBlocked([{'kind': 'valuation_rows_missing', 'detail': '所需交易行情缺估值来源行', 'count': int(missing.sum()),
        'examples': merged.loc[missing, ['date', 'instrument']].head(10).astype(str).to_dict('records')}])
    mismatch = required & merged._valuation_trading.eq(False)
    if mismatch.any(): raise InputBlocked([{'kind': 'valuation_trading_mismatch', 'detail': '行情与估值来源交易状态不一致', 'count': int(mismatch.sum())}])
    for field in fields: merged[field] = merged[field].where(merged.is_trading & merged._valuation_trading.eq(True))
    detail = {**LIMITATION, 'source_rows': int(required.sum()), 'missing_values': {f: int(merged.loc[required, f].isna().sum()) for f in fields}}
    return merged.drop(columns=['_merge', '_valuation_trading']), detail


def query_valuations(root, snapshot, start, end, instruments, fields=FIELDS, mode='strict'):
    _require(mode in ('strict', 'provider_final'), 'Unknown valuation query mode')
    _require(bool(instruments) and all(re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', i) for i in instruments), 'Invalid/empty instruments')
    _require(set(fields) <= set(FIELDS) and bool(fields), 'Unknown/empty valuation fields')
    store = Store(root); state = store.state(snapshot)
    values, used = load_valuations(store, state, start, end, fields, 'provider_final')
    selected = values[values.instrument.isin(instruments)].sort_values(['date', 'instrument'])
    result = selected if mode == 'provider_final' else selected[selected.strict_usable]
    return {'data': result[['date', 'instrument', *fields]].to_dict('records'), 'coverage': {'mode': mode, 'source_rows': len(selected),
        'returned_rows': len(result), 'strict_excluded': len(selected) - len(result), 'missing': {f: int(selected[f].isna().sum()) for f in fields},
        'unknown_instruments': sorted(set(instruments) - set(selected.instrument)), 'strict_usable': False, 'limitations': [LIMITATION]}, 'used': used}
