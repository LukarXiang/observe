"""三套外部财务资料的完整归档、来源审计及原子发布；研究只读冻结 Parquet。"""
import hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .financial_schema import CORE, GROUPS, MISSING, VERSION, definitions, field_dictionary, parse_number, variable_labels
from .locks import DATA_WRITER, operation_lock
from .store import Store, _atomic_json
from ..execution import sessions
from ..runs import canonical, environment, file_sha

TABLES = ('financial_annual', 'financial_quarterly', 'company_controls_annual')


class FinancialImportConfig(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    annual: str = Field(min_length = 1)
    quarterly: str = Field(min_length = 1)
    controls: str = Field(min_length = 1)
    definitions_root: str | None = None
    chunksize: int = Field(4000, ge = 1)


def security_mapping(instruments):
    stocks = instruments[instruments.kind == 'stock'].copy()
    stocks['code'] = stocks.instrument.str.split('.').str[0]
    groups = stocks.groupby('code').instrument.agg(list)
    return {code: values[0] for code, values in groups.items() if len(set(values)) == 1}, {code for code, values in groups.items() if len(set(values)) != 1}


def next_session_available(announcement, calendar):
    """只有日期证据：严格取之后的下一交易日 09:30（Asia/Shanghai）。缺日历不猜。"""
    dates = pd.to_datetime(pd.Series(calendar)).sort_values().drop_duplicates().to_numpy(dtype = 'datetime64[ns]')
    parsed = pd.to_datetime(announcement, errors = 'coerce')
    if parsed.dt.tz is not None: parsed = parsed.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    base = parsed.dt.normalize().to_numpy(dtype = 'datetime64[ns]')
    where = np.searchsorted(dates, base, side = 'right')
    output = np.full(len(parsed), np.datetime64('NaT', 'ns'), dtype = 'datetime64[ns]')
    if len(dates):
        ok = parsed.notna().to_numpy() & (base >= dates[0]) & (where < len(dates))
        output[ok] = dates[where[ok]] + np.timedelta64(570, 'm')
    return pd.Series(output, index = announcement.index).dt.tz_localize('Asia/Shanghai')


def _normalize(raw, table, sha, offset, mapping, ambiguous, calendar, dictionary):
    out = raw.copy(); n = len(raw)
    code = raw['stkcode' if table == TABLES[2] else '证券代码'].astype('string').str.strip()
    valid = code.str.fullmatch(r'\d{6}', na = False)
    out['source_code'] = code
    out['instrument'] = code.map(mapping).astype('string')
    out['mapping_status'] = np.select([~valid, code.isin(ambiguous), out.instrument.isna()], ['invalid_code', 'ambiguous_code', 'unknown_code'], default = 'matched')
    if table == TABLES[1]: period = pd.to_datetime(raw['季度日期'], format = '%Y-%m-%d', errors = 'coerce')
    else:
        year = pd.to_numeric(raw['year' if table == TABLES[2] else '年份'], errors = 'coerce')
        year = year.where(year.eq(year.round()) & year.between(1900, 2100))
        period = pd.to_datetime(year.astype('Int64').astype('string') + '-12-31', errors = 'coerce')
    out['report_period'] = period
    out['source_sha256'], out['source_row'] = sha, np.arange(offset + 1, offset + n + 1, dtype = 'int64')
    out['mapping_version'], out['version_evidence'] = VERSION, 'final_merged_revision_unknown'
    out['available_at'] = pd.Series(pd.NaT, index = raw.index, dtype = 'datetime64[ns, Asia/Shanghai]')
    out['availability_grade'] = 'archive_only'
    malformed = []
    by_name = {r['standard_name']: r for r in dictionary if r['standard_name']}
    for name, (column, _, _, _) in CORE.items():
        if column not in raw or table == TABLES[2]: continue
        values, bad = parse_number(raw[column]); out[name] = values
        if bad.any(): malformed.append({'field': name, 'count': int(bad.sum()), 'examples': [{'source_row': int(out.loc[k, 'source_row']), 'value': raw.loc[k, column]} for k in bad[bad].index[:5]]})
        # 数值没有查证单位时只归档，不开放为标准研究字段。
        if by_name[name]['unit'] != '元': by_name[name]['time_evidence'] = 'unit_unverified_strict_blocked'
    availability = []
    for group, spec in GROUPS.items():
        cols = [raw_name for raw_name, origin, _, _ in CORE.values() if origin == group and raw_name in raw]
        if not cols or table == TABLES[2]: continue
        present = raw[cols].astype('string').apply(lambda s: ~s.str.strip().isin(MISSING) & s.notna()).any(axis = 1)
        a = out.loc[present, ['source_sha256', 'source_row', 'source_code', 'instrument', 'report_period', 'mapping_status', 'version_evidence']].copy()
        a['archive_table'], a['source_group'] = table, group
        # FAR 的 Annodt 只关联 FAR 字段；财务指标_报表类型属于另一来源，不能借给 FAR。
        a['report_type'] = raw.loc[present, spec['report_type']].astype('string').replace('', pd.NA) if group != 'far_finidx' and spec['report_type'] in raw else pd.NA
        a['correction_dates_raw'] = raw.loc[present, spec['correction']].astype('string') if spec['correction'] in raw else pd.NA
        announce = raw.loc[present, '年报公布日期'] if group == 'far_finidx' and '年报公布日期' in raw else pd.Series('', index = a.index)
        a['announcement_raw'] = announce.astype('string'); a['announcement_date'] = pd.to_datetime(announce, format = '%Y-%m-%d', errors = 'coerce')
        after = a.announcement_date > a.report_period
        a['available_at'] = next_session_available(a.announcement_date.where(after), calendar)
        a['date_evidence'] = 'far_annodt_only' if group == 'far_finidx' else 'missing_first_announcement'
        a['availability_grade'] = np.where(a.available_at.notna(), 'source_date_revision_unknown', 'unavailable')
        a['version_id'] = pd.NA             # 来源文件哈希是归档身份，不冒充历史报表版本
        a['strict_usable'] = False          # 当前最终合并值都无法证明首次发布的原始版本
        availability.append(a)
    return out, pd.concat(availability, ignore_index = True) if availability else pd.DataFrame(), malformed


def _profile(raw, dictionary, stats):
    for item in dictionary:
        column = item['column']; s = raw[column]; key = stats.setdefault(column, Counter())
        text = s.astype('string').str.strip(); missing = text.isna() | text.isin(MISSING)
        key['rows'] += len(s); key['missing'] += int(missing.sum())
        if item['numeric_candidate']:
            number, bad = parse_number(s); key['numeric_valid'] += int(number.notna().sum()); key['numeric_unparsed'] += int(bad.sum())


def _chunks(path, table, chunksize):
    if table == TABLES[2]:
        with pd.read_stata(path, iterator = True, convert_categoricals = False, convert_missing = False) as reader:
            while True:
                try: frame = reader.read(chunksize)
                except StopIteration: break
                if not len(frame): break
                yield frame.reset_index(drop = True)
    else:
        yield from pd.read_csv(path, chunksize = chunksize, dtype = str, keep_default_na = False, encoding = 'utf-8-sig')


def import_financials(root, annual, quarterly, controls, definitions_root = None, chunksize = 4000, log = None):
    """保留全部输入行、全部原字段和冲突；三套数据及字典一起发布。重复输入、映射与主数据完全一致时幂等。"""
    if chunksize < 1: raise ValueError('chunksize 必须大于 0')
    paths = dict(zip(TABLES, map(Path, (annual, quarterly, controls))))
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(p)
    store = Store(root); started = time.perf_counter()
    with operation_lock(root, DATA_WRITER):
        state = store.published(); instruments = store.load_state(state, 'instruments'); calendar = sessions(store.load_state(state, 'calendar'))
        if not len(instruments) or not calendar: raise ValueError('导入需要已发布证券主数据和交易日历')
        mapping, ambiguous = security_mapping(instruments)
        hashes = {t: file_sha(p) for t, p in paths.items()}; evidence = definitions(definitions_root)
        proof = {k: {**v, 'text_sha256': hashlib.sha256(v['text'].encode()).hexdigest()} for k, v in evidence.items()}
        basis = {'inputs': hashes, 'mapping_version': VERSION, 'metadata': {t: state['tables'].get(t, {}) for t in ('instruments', 'calendar')}, 'definitions': proof}
        iid = hashlib.sha256(json.dumps(canonical(basis), sort_keys = True, ensure_ascii = False).encode()).hexdigest()[:20]
        receipt = store.root / 'raw' / 'external_financials' / f'{iid}.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding = 'utf-8'))
            parts = previous.get('published_parts', {})
            if previous.get('status') == 'published' and parts and all(state['tables'].get(t, {}).get(k) == v for t, ps in parts.items() for k, v in ps.items()):
                if all(file_sha(store.root / f) == sha for f, sha in previous['partition_bytes'].items()):
                    return {**previous['result'], 'status': 'unchanged', 'elapsed_seconds': round(time.perf_counter() - started, 3)}
                raise ValueError('已归档财务分区损坏；拒绝幂等命中')
        staging = store.root / 'staging' / 'financials' / iid; staging.mkdir(parents = True, exist_ok = True)
        report = {'import_id': iid, 'mapping_version': VERSION, 'base_batch': state['batch_id'], 'environment': environment(), 'tables': {},
                  'definitions': proof, 'limitations': ['最终合并值缺历史版本证据，严格历史覆盖为 0', '季度无首次公告时间，FAR 年报日期不放行其他来源',
                                                     '控制变量年度汇总、everSTPT/STPT/marketvalue 不用于历史股票池或每日估值']}
        _atomic_json(receipt, {'status': 'staging', 'basis': basis, 'paths': {t: str(p.resolve()) for t, p in paths.items()}})
        parts, all_dictionary = {}, []
        try:
            for table, path in paths.items():
                if log: log(f'{table}: 分块读取 {path.name}')
                labels = variable_labels(path if table == TABLES[2] else path.with_suffix('.dta'))
                staged, avail_staged = defaultdict(list), defaultdict(list)
                dictionary, profiles, offset, malformed, code_counts = None, {}, 0, [], Counter()
                key_files = []; period_issues, mapping_counts = Counter(), Counter()
                announcement_counts, report_types = Counter(), defaultdict(Counter)
                for k, raw in enumerate(_chunks(path, table, chunksize)):
                    raw = raw.reset_index(drop = True)
                    if dictionary is None: dictionary = field_dictionary(list(raw.columns), table, hashes[table], labels, evidence)
                    _profile(raw, dictionary, profiles)
                    normalized, available, bad = _normalize(raw, table, hashes[table], offset, mapping, ambiguous, calendar, dictionary)
                    malformed += bad; offset += len(raw); code_counts.update(normalized.source_code.value_counts().to_dict()); mapping_counts.update(normalized.mapping_status.value_counts().to_dict())
                    period_issues['invalid_period'] += int(normalized.report_period.isna().sum())
                    if len(available):
                        for group, rows in available.groupby('source_group'):
                            report_types[group].update(rows.report_type.fillna('<missing>').value_counts().to_dict())
                            announcement_counts[f'{group}/valid_date'] += int(rows.announcement_date.notna().sum())
                            announcement_counts[f'{group}/before_or_at_period'] += int((rows.announcement_date <= rows.report_period).sum())
                            announcement_counts[f'{group}/calendar_available'] += int(rows.available_at.notna().sum())
                    years = normalized.report_period.dt.year.astype('Int64').astype('string').fillna('unknown')
                    for year, rows in normalized.groupby(years, sort = True):
                        f = staging / f'{table}-{year}-{k:05d}.parquet'; rows.to_parquet(f, index = False); staged[year].append(f)
                    if len(available):
                        ay = available.report_period.dt.year.astype('Int64').astype('string').fillna('unknown')
                        for year, rows in available.groupby(ay, sort = True):
                            f = staging / f'availability-{table}-{year}-{k:05d}.parquet'; rows.to_parquet(f, index = False); avail_staged[year].append(f)
                    key = normalized[['source_code', 'report_period', 'source_row', 'mapping_status']]
                    f = staging / f'keys-{table}-{k:05d}.parquet'; key.to_parquet(f, index = False); key_files.append(f)
                    if log and k % 20 == 0: log(f'{table}: 已归档 {offset:,} 行')
                if dictionary is None: raise ValueError(f'{table}: 输入为空')
                if file_sha(path) != hashes[table]: raise ValueError(f'{path}: 导入期间源文件发生变化')
                keys = pd.concat([pd.read_parquet(f) for f in key_files], ignore_index = True)
                duplicate = keys.duplicated(['source_code', 'report_period'], keep = False)
                quarantine = keys[keys.mapping_status.ne('matched') | keys.report_period.isna() | duplicate]
                quarantine.to_parquet(staging / f'{table}-quarantine.parquet', index = False)
                conflicts = keys[duplicate]
                conflicts.to_parquet(staging / f'{table}-conflicts.parquet', index = False)
                parts[table] = {}
                for year, files in sorted(staged.items()):
                    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index = True)
                    if df.duplicated(['source_sha256', 'source_row']).any(): raise ValueError(f'{table}/{year}: 来源行冲突')
                    part = f'{year}/{iid}'; entry = store.write_partition(table, part, df)
                    if entry['rows'] != len(df): raise ValueError('存储层丢行')
                    parts[table][part] = entry
                for year, files in sorted(avail_staged.items()):
                    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index = True)
                    parts.setdefault('financial_availability', {})[f'{year}/{table}-{iid}'] = store.write_partition('financial_availability', f'{year}/{table}-{iid}', df)
                if sum(v['rows'] for v in parts[table].values()) != offset: raise ValueError(f'{table}: 行数不一致')
                for item in dictionary:
                    item.update(profiles.get(item['column'], {})); item['missing_rate'] = item['missing'] / offset
                all_dictionary += dictionary
                report['tables'][table] = {'input': str(path.resolve()), 'source_sha256': hashes[table], 'input_bytes': path.stat().st_size,
                                          'rows': offset, 'columns': len(dictionary), 'securities': len(code_counts), 'mapping': dict(mapping_counts),
                                          'unmatched_codes': sorted(set(code_counts) - set(mapping)), 'period_issues': dict(period_issues),
                                          'conflicting_rows': len(conflicts), 'quarantined_rows': len(quarantine), 'quarantine': str(staging / f'{table}-quarantine.parquet'),
                                          'numeric_issues': malformed, 'source_report_types': {k: dict(v) for k, v in report_types.items()},
                                          'announcement_counts': dict(announcement_counts), 'strict_usable_rows': 0,
                                          'storage_bytes': sum((store.root / v['file']).stat().st_size for v in parts[table].values())}
                if log: log(f'{table}: {offset:,} 行核对通过，{mapping_counts["matched"]:,} 行匹配证券资料')
            df = pd.DataFrame(all_dictionary)
            parts['financial_fields'] = {iid: store.write_partition('financial_fields', iid, df)}
            _atomic_json(store.root / 'catalog' / f'financial-fields-{iid}.json', all_dictionary)
            report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            report['standard_bytes'] = sum((store.root / v['file']).stat().st_size for ps in parts.values() for v in ps.values())
            audit_path = store.root / 'audits' / 'financials' / f'{iid}.json'
            _atomic_json(audit_path, report)         # 所有审计先落盘，最后才发布
            bid = store.write_batch(parts, f'完整财务归档 {iid}；严格历史使用仍阻断')
            store.publish(bid)
            result = {'status': 'published', 'import_id': iid, 'batch_id': bid, 'audit': str(audit_path), 'rows': {t: report['tables'][t]['rows'] for t in TABLES},
                      'standard_bytes': report['standard_bytes'], 'elapsed_seconds': report['elapsed_seconds'], 'strict_usable_rows': 0}
            _atomic_json(receipt, {'status': 'published', 'basis': basis, 'paths': {t: str(p.resolve()) for t, p in paths.items()}, 'published_parts': parts,
                                   'partition_bytes': {v['file']: file_sha(store.root / v['file']) for ps in parts.values() for v in ps.values()}, 'result': result})
            return result
        except Exception as exc:
            report['failure'] = f'{type(exc).__name__}: {exc}'
            _atomic_json(store.root / 'audits' / 'financials' / f'{iid}-failed.json', report)
            raise


def read_financial_archive(root, table, snapshot, instruments = None, start = None, end = None, columns = None):
    """归档查询与历史可用性查询分开；这里返回最终整理值，不能直接当历史信号。"""
    if table not in TABLES: raise ValueError('未知财务归档表')
    store = Store(root); state = store.state(snapshot)
    files = [str(store.root / v['file']) for _, v in sorted(state['tables'].get(table, {}).items())]
    if not files: return pd.DataFrame(columns = columns)
    with duckdb.connect() as db:
        relation = db.read_parquet(files, union_by_name = True); schema = set(relation.columns)
        requested = list(columns) if columns else relation.columns
        if set(requested) - schema: raise ValueError(f'未知财务列：{sorted(set(requested) - schema)}')
        db.register('archive', relation); clauses, params = [], []
        if instruments is not None:
            if not instruments: return pd.DataFrame(columns = requested)
            clauses.append('instrument in (select unnest(?))'); params.append(list(instruments))
        for op, value in (('>=', start), ('<=', end)):
            if value is not None: clauses.append(f'report_period {op} ?'); params.append(pd.Timestamp(value).to_pydatetime())
        selected = ','.join('"' + c.replace('"', '""') + '"' for c in requested)
        return db.execute(f'select {selected} from archive' + (' where ' + ' and '.join(clauses) if clauses else '') + ' order by source_sha256,source_row', params).df()
