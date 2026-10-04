"""按决策时点查询财报与报表期算子；先过滤可用性，再选择报告期与版本。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .financial_schema import CORE
from .financials import next_session_available, read_financial_archive
from .store import Store
from ..execution import sessions


class FinancialConflict(ValueError): pass


@dataclass
class QueryResult:
    data: pd.DataFrame
    coverage: dict


def decision_timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t): raise ValueError('decision_time 不可为空')
    return t.tz_localize('Asia/Shanghai') if t.tz is None else t.tz_convert('Asia/Shanghai')


def visible_records(records, decision_time, mode = 'strict', lag_days = None, calendar = None, report_type = 'A'):
    """探索模式必须显式提供自然日滞后假设。更正日期不作首次公告，未知版本不进入严格模式。"""
    if mode not in ('strict', 'exploratory'): raise ValueError('mode 必须是 strict 或 exploratory')
    if mode == 'exploratory' and (type(lag_days) is not int or lag_days < 1 or calendar is None):
        raise ValueError('探索模式需要正整数 lag_days 和冻结交易日历；滞后不证明无前视')
    t = decision_timestamp(decision_time); frame = records.copy()
    coverage = {'mode': mode, 'input_rows': len(frame), 'lag_days': lag_days, 'decision_time': t.isoformat(), 'strict_evidence': mode == 'strict'}
    if not len(frame): return QueryResult(frame, {**coverage, 'visible_rows': 0})
    required = {'instrument', 'report_period', 'field', 'value', 'available_at', 'strict_usable', 'report_type', 'source_group'}
    if required - set(frame): raise ValueError(f'财报记录缺列：{sorted(required - set(frame))}')
    frame['report_period'] = pd.to_datetime(frame.report_period)
    available = pd.to_datetime(frame.available_at)
    frame['available_at'] = available.dt.tz_localize('Asia/Shanghai') if available.dt.tz is None else available.dt.tz_convert('Asia/Shanghai')
    frame['availability_policy'] = 'proven_version' if mode == 'strict' else 'source_date_revision_unknown'
    if mode == 'strict':
        # 字符串 "False" 不得被 bool() 当作证明；必须是真布尔值。
        strict = frame.strict_usable.map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
        coverage['excluded_unproven'] = int((~strict).sum()); frame = frame[strict].copy()
        if len(frame) and ('version_id' not in frame or frame.version_id.isna().any()): raise ValueError('strict_usable 记录缺报表版本证据')
    else:
        missing = frame.available_at.isna()
        assumed = frame.report_period + pd.to_timedelta(lag_days, unit = 'D')
        frame.loc[missing, 'available_at'] = next_session_available(assumed[missing], calendar)
        frame.loc[missing, 'availability_policy'] = f'report_period_plus_{lag_days}_days_next_session_revision_unknown'
        coverage['assumed_date_rows'] = int(missing.sum())
    finite = np.isfinite(pd.to_numeric(frame.value, errors = 'coerce').to_numpy(float))
    keep = frame.instrument.notna() & frame.available_at.notna() & (frame.available_at <= t) & finite
    # 即使可用时刻被伪填，也不允许决策时点之后的报告期。
    keep &= (frame.report_period <= t.tz_localize(None)) & frame.report_period.dt.is_quarter_end
    if report_type is not None: keep &= frame.report_type.eq(report_type).fillna(False)
    frame = frame[keep].copy()
    coverage['visible_rows'] = len(frame)
    return QueryResult(frame, coverage)


def resolve_versions(records):
    """同报告期选择当时最新版本；同可用时刻的异值/来源冲突直接报错，保留争议。"""
    if not len(records): return records.copy()
    group = ['instrument', 'field', 'report_period']
    latest = records.groupby(group, dropna = False).available_at.transform('max')
    frame = records[records.available_at.eq(latest)].copy()
    for key, rows in frame.groupby(group, dropna = False):
        if rows.value.nunique(dropna = False) > 1 or rows.source_group.nunique(dropna = False) > 1:
            raise FinancialConflict(f'同一报告期、可用时刻的财务值冲突：{key}')
        if 'version_id' in rows and rows.version_id.nunique(dropna = False) > 1:
            raise FinancialConflict(f'同一可用时刻有多个报表版本：{key}')
    return frame.sort_values(group + ['available_at']).drop_duplicates(group).reset_index(drop = True)


def latest_reports(records):
    """更正较旧报告不会替代较新的报告：先版本选择，再按报告期选择。"""
    frame = resolve_versions(records)
    if not len(frame): return frame
    latest = frame.groupby(['instrument', 'field']).report_period.transform('max')
    return frame[frame.report_period.eq(latest)].reset_index(drop = True)


def query_financial_history(root, snapshot, fields, instruments, decision_time, mode = 'strict', lag_days = None, report_type = 'A', latest = True, archive_table = 'financial_quarterly'):
    if not fields or set(fields) - set(CORE): raise ValueError(f'财务字段未登记：{sorted(set(fields) - set(CORE))}')
    if archive_table not in ('financial_annual', 'financial_quarterly'): raise ValueError('财务历史查询须明确选择年度或季度来源表')
    store = Store(root); state = store.state(snapshot)
    avail = store.load_state(state, 'financial_availability')
    dictionaries = store.load_state(state, 'financial_fields')
    records = []
    if len(avail):
        avail = avail[avail.instrument.isin(set(instruments)) & avail.archive_table.eq(archive_table)].copy()
        for field in fields:
            raw, group, kind, _ = CORE[field]
            source = avail[avail.source_group.eq(group)]
            for table, a in source.groupby('archive_table'):
                # 标准核心金额只有实际查证过元单位的来源才开放。
                reviewed = dictionaries[dictionaries.standard_name.eq(field) & dictionaries.unit.eq('元')].source_sha256
                a = a[a.source_sha256.isin(set(reviewed))]
                if not len(a): continue
                archive = read_financial_archive(root, table, snapshot, instruments, columns = ['source_sha256', 'source_row', field])
                joined = a.merge(archive, on = ['source_sha256', 'source_row'], validate = 'one_to_one').rename(columns = {field: 'value'})
                joined['field'], joined['period_kind'], joined['unit'] = field, kind, 'CNY'
                records.append(joined)
    frame = pd.concat(records, ignore_index = True) if records else pd.DataFrame()
    result = visible_records(frame, decision_time, mode, lag_days, sessions(store.load_state(state, 'calendar')), report_type)
    result.coverage.update(snapshot = snapshot, mapping_versions = sorted(set(dictionaries.mapping_version)) if len(dictionaries) else [], requested_fields = list(fields),
                           requested_instruments = len(instruments), archive_table = archive_table, report_type = report_type, note = '最终整理值的滞后探索不构成严格历史证明')
    result.data = latest_reports(result.data) if latest else resolve_versions(result.data)
    result.coverage['returned_rows'] = len(result.data)
    return result


def derive_periods(visible, field, operator):
    """对已经按同一决策时点过滤的记录算单季/TTM/同比，禁止缺期补零或累加存量。"""
    if operator not in ('single_quarter', 'ttm', 'yoy'): raise ValueError('未知财报期算子')
    selected = resolve_versions(visible[visible.field.eq(field)]) if len(visible) else visible
    if not len(selected): return pd.DataFrame(columns = ['instrument', 'report_period', 'field', 'value', 'available_at', 'reason'])
    if not selected.period_kind.eq('ytd_flow').all(): raise ValueError('报表期流量算子只支持明确登记的累计流量，不能累加资产负债存量')
    rows = []
    for instrument, data in selected.groupby('instrument'):
        by_period = {pd.Timestamp(r['report_period']).to_period('Q'): r for r in data.to_dict('records')}
        def single(q):
            current = by_period.get(q)
            if current is None: return None, []
            if q.quarter == 1: return current['value'], [current]
            previous = by_period.get(q - 1)
            if previous is None: return None, [current]
            return current['value'] - previous['value'], [current, previous]
        for q, current in sorted(by_period.items()):
            if pd.Timestamp(current['report_period']).normalize() != q.end_time.normalize():
                raise ValueError('财报期不是季度期末')
            inputs = [current]; reason = None
            if operator == 'single_quarter': value, inputs = single(q)
            elif operator == 'ttm':
                # 完整年度累计值本身就是四季度总和；中间季使用本期累计+上年年报-上年同期累计。
                if q.quarter == 4: value = current['value']
                else:
                    annual, previous = by_period.get(pd.Period(year = q.year - 1, quarter = 4, freq = 'Q')), by_period.get(q - 4)
                    value = current['value'] + annual['value'] - previous['value'] if annual is not None and previous is not None else None
                    inputs += [x for x in (annual, previous) if x is not None]
            else:
                previous = by_period.get(q - 4)
                value = current['value'] / abs(previous['value']) - np.sign(previous['value']) if previous is not None and previous['value'] != 0 else None
                if previous is not None: inputs.append(previous)
            if value is None: reason = 'missing_required_period_or_zero_base'
            rows.append({'instrument': instrument, 'report_period': current['report_period'], 'field': f'{field}_{operator}', 'value': value,
                         'available_at': max(x['available_at'] for x in inputs) if value is not None else pd.NaT, 'reason': reason,
                         'input_versions': [{'report_period': str(x['report_period']), 'version_id': x.get('version_id'), 'source_sha256': x.get('source_sha256'),
                                             'source_row': x.get('source_row'), 'available_at': str(x['available_at'])} for x in inputs]})
    return pd.DataFrame(rows)
