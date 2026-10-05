"""价格指数入库：显式区间、已发布日历、完整响应审计，整批通过后才发布。

只更新 index_1d；不改日历、证券资料、股票行情或旧快照。指数审计独立于股票日线审计。
"""
from datetime import date, datetime
import json
import re
import secrets

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import raw, standardize as std
from .audit import issue
from .locks import DATA_WRITER, operation_lock
from .sources.baostock import BaoStock
from .store import Store, _atomic_json
from ..runs import file_sha

VERSION = 'index_daily_v1'
FIELDS = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount')
ISSUE_COLUMNS = ['level', 'rule', 'date', 'instrument', 'detail']


def index_code(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{6}\.(SH|SZ)', value):
        raise ValueError('指数代码须显式指定交易所，例如 000300.SH；不推测裸代码的市场')
    return value


class IndexUpdateConfig(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    start: date
    end: date
    indices: list[str] = Field(default_factory = lambda: ['000300.SH'], min_length = 1, max_length = 50)

    @field_validator('indices')
    @classmethod
    def codes(cls, values):
        for value in values: index_code(value)
        if len(set(values)) != len(values): raise ValueError('指数代码不能重复')
        return values

    @model_validator(mode = 'after')
    def dates(self):
        if self.start > self.end: raise ValueError('start 不能晚于 end')
        return self


def standardize_index(response):
    """BaoStock 指数原始价；不按股票代码前缀推测市场，不填零、不去重。"""
    required = {'date', 'code', *FIELDS}
    if not required.issubset(response.columns): raise ValueError(f'指数响应缺少字段 {sorted(required - set(response.columns))}')
    codes = response.code.astype(str)
    if not codes.str.fullmatch(r'(sh|sz)\.\d{6}').all(): raise ValueError('指数响应的 code 格式错误')
    out = pd.DataFrame({'date': pd.to_datetime(response.date, format = '%Y-%m-%d', errors = 'coerce').dt.date, 'index': codes.map(std.instrument)})
    for name in FIELDS: out[name] = pd.to_numeric(response[name], errors = 'coerce').astype(float)
    return out.sort_values(['date', 'index']).reset_index(drop = True)


def audit_index(frame, code, days):
    """先审计主键再交给 Store；漏日、错代码、休市行、无效价格或量额均阻断。"""
    out = []
    for row in frame[frame.duplicated(['date', 'index'], keep = False)].to_dict('records'):
        out.append(issue('block', 'duplicate_key', row['date'], row['index']))
    for row in frame[frame['index'] != code].to_dict('records'):
        out.append(issue('block', 'unexpected_index', row['date'], row['index'], f'请求 {code}'))
    expected = set(days)
    for row in frame[~frame.date.isin(expected)].to_dict('records'):
        out.append(issue('block', 'unexpected_date', row['date'], code, '无效日期或不在请求区间的交易日历中'))
    for day in sorted(expected - set(frame.loc[frame['index'] == code, 'date'])):
        out.append(issue('block', 'missing_day', day, code))
    prices = frame[list(FIELDS[:5])]
    for row in frame[(~np.isfinite(prices) | (prices <= 0)).any(axis = 1)].to_dict('records'):
        out.append(issue('block', 'bad_price', row['date'], code))
    bad = (frame.low > frame[['open', 'close']].min(axis = 1)) | (frame.high < frame[['open', 'close']].max(axis = 1)) | (frame.low > frame.high)
    for row in frame[bad].to_dict('records'): out.append(issue('block', 'ohlc_order', row['date'], code))
    amounts = frame[['volume', 'amount']]
    for row in frame[(~np.isfinite(amounts) | (amounts < 0)).any(axis = 1)].to_dict('records'):
        out.append(issue('block', 'bad_volume_amount', row['date'], code))
    return out


def _inputs(store, state, cfg):
    cal = store.load_state(state, 'calendar')
    if not {'date', 'is_open'}.issubset(cal.columns): raise ValueError('先建立交易日历，再更新指数')
    cal['date'] = pd.to_datetime(cal.date).dt.date
    cal = cal[cal.date.between(cfg.start, cfg.end)].copy()
    expected = set(pd.date_range(cfg.start, cfg.end).date)
    if cal.date.duplicated().any() or set(cal.date) != expected: raise ValueError('请求区间的交易日历不完整或日期重复，不能据此审计指数覆盖')
    if cal.is_open.isna().any() or cal.is_open.astype(str).str.strip().eq('').any(): raise ValueError('交易日历的开市标记缺失，不能视作休市')
    days = sorted(cal.loc[cal.is_open.map(std.flag), 'date'])
    if not days: raise ValueError('请求区间没有交易日')
    inst = store.load_state(state, 'instruments')
    if not {'instrument', 'kind'}.issubset(inst.columns): raise ValueError('先建立证券主数据，再更新指数')
    known = set(inst.loc[inst.kind == 'index', 'instrument'])
    if set(cfg.indices) - known: raise ValueError(f'证券主数据中未确认的指数：{sorted(set(cfg.indices) - known)}')
    if set(state['tables'].get('index_1d', {})) - {'all'}: raise ValueError('index_1d 只接受 all 单文件分区，须先显式迁移旧分区布局')
    return days


def update_indices(root, start, end, indices = None, source = None):
    cfg = IndexUpdateConfig.model_validate({'start': start, 'end': end, **({'indices': indices} if indices is not None else {})})
    store = Store(root)
    with operation_lock(root, DATA_WRITER):
        state = store.published(); days = _inputs(store, state, cfg)
        aid = f'{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(4)}'
        directory = store.root / 'index_audits'; directory.mkdir(parents = True, exist_ok = True)
        report = {'audit_id': aid, 'processing_version': VERSION, 'config': cfg.model_dump(mode = 'json'), 'base_batch_id': state['batch_id'],
                  'batch_id': None, 'status': 'running', 'expected_days': len(days),
                  'indices': {code: {'expected': len(days), 'observed': 0, 'status': 'not_attempted'} for code in cfg.indices},
                  'inputs': {}, 'issues_file': str(directory / f'{aid}.issues.csv')}
        for table in ('calendar', 'instruments', 'index_1d'):
            report['inputs'][table] = {part: {**entry, 'file_sha256': file_sha(store.root / entry['file'])} for part, entry in state['tables'].get(table, {}).items()}
        _atomic_json(directory / f'{aid}.json', report)
        frames, problems = [], []
        try:
            with (source or BaoStock(root)).session() as src:
                for code in cfg.indices:
                    info = {'expected': len(days), 'observed': 0, 'status': 'failed'}; report['indices'][code] = info
                    try:
                        response = src.index_daily(std.to_baostock(code), cfg.start, cfg.end)
                        if response is None: raise ValueError('指数请求返回未知空响应')
                        path = raw.save(root, 'baostock', 'index_daily', f'{code}_{cfg.start}_{cfg.end}', response)
                        info.update(raw_file = path.relative_to(store.root).as_posix(), raw_sha256 = file_sha(path))
                        frame = standardize_index(response); found = audit_index(frame, code, days)
                        info.update(observed = len(frame), missing_days = sorted(str(d) for d in set(days) - set(frame.loc[frame['index'] == code, 'date'])),
                                    status = 'rejected' if found else 'validated')
                        problems.extend(found); frames.append(frame)
                    except Exception as exc:
                        info['error'] = f'{type(exc).__name__}: {exc}'
                        problems.append(issue('block', 'fetch_or_schema_failed', inst = code, detail = info['error']))
        except Exception as exc:
            problems.append(issue('block', 'source_session_failed', detail = f'{type(exc).__name__}: {exc}'))
        issues = pd.DataFrame(problems, columns = ISSUE_COLUMNS)
        # 审计明细先落盘；失败也留下报告，绝不把空响应解释成指数不存在。
        issues.to_csv(directory / f'{aid}.issues.csv', index = False)
        report.update(status = 'rejected', problem_count = len(issues))
        try:
            if not problems:
                old = store.load_state(state, 'index_1d')
                if len(old) and old.duplicated(['date', 'index']).any():
                    raise ValueError('已发布 index_1d 主键重复，拒绝静默去重')
                incoming = pd.concat(frames, ignore_index = True)
                merged = pd.concat([old, incoming], ignore_index = True) if len(old) else incoming
                entry = store.write_partition('index_1d', 'all', merged)
                if entry == state['tables'].get('index_1d', {}).get('all'):
                    report.update(status = 'unchanged', batch_id = state['batch_id'])
                else:
                    bid = store.write_batch({'index_1d': {'all': entry}}, note = f'价格指数审计 {aid}；{cfg.start}..{cfg.end}')
                    report.update(status = 'validated', batch_id = bid)
                    _atomic_json(directory / f'{aid}.json', report)
                    store.publish(bid)
                    report['status'] = 'published'
        except Exception as exc:
            report.update(status = 'failed', error = f'{type(exc).__name__}: {exc}')
            _atomic_json(directory / f'{aid}.json', report)
            raise
        _atomic_json(directory / f'{aid}.json', report)
        return {**report, 'report_file': str(directory / f'{aid}.json')}


def index_bars(root, index = '000300.SH', start = '1990-01-01', end = '2099-12-31', snapshot = None, limit = 1000, offset = 0):
    cfg = IndexUpdateConfig(start = start, end = end, indices = [index])
    if not 1 <= limit <= 10000 or offset < 0: raise ValueError('limit 必须在 1..10000，offset 必须非负')
    if snapshot is not None and (not re.fullmatch(r'[\w-]+', snapshot)): raise ValueError('非法快照编号')
    store = Store(root); state = store.state(snapshot)
    files = [str(store.root / entry['file']) for entry in state['tables'].get('index_1d', {}).values()]
    result = {'batch_id': state['batch_id'], 'snapshot_id': snapshot, 'index': index, 'total': 0, 'rows': [], 'limit': limit, 'offset': offset}
    if not files: return result
    with duckdb.connect() as conn:
        sql = 'from read_parquet(?) where "index" = ? and date between ? and ?'; params = [files, index, cfg.start, cfg.end]
        result['total'] = conn.execute('select count(*) ' + sql, params).fetchone()[0]
        frame = conn.execute('select * ' + sql + ' order by date limit ? offset ?', [*params, limit, offset]).df()
    result['rows'] = json.loads(frame.to_json(orient = 'records', date_format = 'iso'))
    return result
