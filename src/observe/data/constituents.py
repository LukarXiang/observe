"""带查询日期的指数成分归档。供应商按周更新，明确为探索证据，不向未知日期填充。"""
from contextlib import nullcontext
from datetime import date, datetime
import json
import secrets

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import raw, standardize as std
from .indices import _inputs as index_inputs, IndexUpdateConfig
from .locks import DATA_WRITER, operation_lock
from .sources.baostock import BaoStock
from .store import Store, _atomic_json
from ..runs import file_sha

VERSION = 'index_constituents_weekly_v1'
POLICY = 'provider_weekly_asof'
SIZES = {'000300.SH': 300, '000905.SH': 500, '000906.SH': 800}
CSI800_REFERENCE = 'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000904_Index_Methodology_cn.pdf'


class ConstituentsUpdateConfig(BaseModel):
    model_config = ConfigDict(extra = 'forbid')
    start: date
    end: date
    indices: list[str] = Field(default_factory = lambda: ['000906.SH'], min_length = 1, max_length = 3)
    max_age_days: int = Field(7, ge = 0, le = 31)
    force: bool = False

    @field_validator('indices')
    @classmethod
    def codes(cls, values):
        if set(values) - set(SIZES) or len(set(values)) != len(values): raise ValueError('成分接口仅支持不重复的 000300.SH / 000905.SH / 000906.SH')
        return values

    @model_validator(mode = 'after')
    def dates(self):
        if self.start > self.end: raise ValueError('start 不能晚于 end')
        return self


def normalize_constituents(response, index, day, master, source_sha256, max_age_days = 7):
    """不能用今天的返回值冒充历史；先检查供应商日期、全名单、重复和证券映射。"""
    if index not in ('000300.SH', '000905.SH'): raise ValueError('供应商只提供 300/500；800 须用同日期完整名单合成')
    if response is None or not {'updateDate', 'code', 'code_name'}.issubset(response): raise ValueError('成分响应缺少字段')
    if len(response) != SIZES[index]: raise ValueError(f'{index}/{day}: 名单须完整 {SIZES[index]} 行，实际 {len(response)}')
    codes = response.code.astype(str)
    if not codes.str.fullmatch(r'(sh|sz)\.\d{6}').all() or codes.duplicated().any(): raise ValueError('成分代码非法或重复')
    source_dates = pd.to_datetime(response.updateDate, format = '%Y-%m-%d', errors = 'coerce').dt.date
    if source_dates.isna().any() or source_dates.nunique() != 1: raise ValueError('供应商成分日期缺失或不一致')
    lag = (day - source_dates.iloc[0]).days
    if not 0 <= lag <= max_age_days: raise ValueError(f'成分日期晚于查询日或过期：{source_dates.iloc[0]} -> {day}')
    instruments = codes.map(std.instrument)
    if set(instruments) - set(master): raise ValueError(f'成分证券主数据未确认：{sorted(set(instruments) - set(master))}')
    return pd.DataFrame({'date': day, 'index': index, 'instrument': instruments, 'source_date': source_dates,
                         'component_index': index, 'source_name': response.code_name.astype(str), 'source_sha256': source_sha256,
                         'source_lag_days': lag, 'policy': POLICY, 'processing_version': VERSION, 'strict_usable': False}).reset_index(drop = True)


def combine_csi800(hs300, zz500):
    if len(hs300) != 300 or len(zz500) != 500 or set(hs300.instrument) & set(zz500.instrument):
        raise ValueError('中证800需要完整且无交集的300/500名单')
    if set(hs300.date) != set(zz500.date) or set(hs300.source_date) != set(zz500.source_date):
        raise ValueError('300/500成分归档日期不同，不能合成800')
    frame = pd.concat([hs300, zz500], ignore_index = True)
    frame['index'] = '000906.SH'
    return frame


def membership_records(frame, index, days, max_age_days = 7):
    """读取精确归档日期；不把最近一张名单无限向后或向前填充。"""
    required = {'date', 'index', 'instrument', 'source_date', 'component_index', 'policy', 'processing_version', 'strict_usable'}
    if not required.issubset(frame): raise ValueError('冻结快照缺少成分归档字段')
    selected = frame[frame['index'].eq(index)].copy()
    selected['date'] = pd.to_datetime(selected.date).dt.date
    selected['source_date'] = pd.to_datetime(selected.source_date).dt.date
    selected = selected[selected.date.isin(set(days))]
    if selected.duplicated(['date', 'instrument']).any(): raise ValueError('成分主键重复')
    if not selected.policy.eq(POLICY).all() or not selected.processing_version.eq(VERSION).all(): raise ValueError('未知成分归档口径')
    if selected.strict_usable.map(lambda v: not isinstance(v, bool) or v).any(): raise ValueError('周频供应商成分不具备严格历史公告证据')
    counts = selected.groupby('date').size().to_dict()
    if any(counts.get(day) != SIZES[index] for day in days): raise ValueError('成分归档缺查询日期或完整名单')
    lags = (pd.to_datetime(selected.date) - pd.to_datetime(selected.source_date)).dt.days
    if selected.source_date.isna().any() or not lags.between(0, max_age_days).all(): raise ValueError('成分日期晚于决策日期或过期')
    for day, rows in selected.groupby('date'):
        if rows.source_date.nunique() != 1: raise ValueError(f'{day}: 混合供应商成分日期')
        expected = {'000300.SH': 300, '000905.SH': 500} if index == '000906.SH' else {index: SIZES[index]}
        if rows.component_index.value_counts().to_dict() != expected: raise ValueError('派生成分来源或数量不同')
    return selected.reset_index(drop = True)


def update_constituents(root, start, end, indices = None, max_age_days = 7, force = False, source = None, log = None):
    cfg = ConstituentsUpdateConfig(start = start, end = end, **({'indices': indices} if indices is not None else {}), max_age_days = max_age_days, force = force)
    store = Store(root)
    with operation_lock(root, DATA_WRITER):
        state = store.published()
        days = index_inputs(store, state, IndexUpdateConfig(start = cfg.start, end = cfg.end, indices = cfg.indices))
        master = set(store.load_state(state, 'instruments').query("kind == 'stock'").instrument)
        old = store.load_state(state, 'index_constituents')
        if len(old):
            if old.duplicated(['date', 'index', 'instrument']).any(): raise ValueError('已发布成分主键重复，不能静默覆盖')
            old['date'] = pd.to_datetime(old.date).dt.date
        aid = f'{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}'
        report_path = store.root / 'audits' / 'constituents' / f'{aid}.json'
        report = {'audit_id': aid, 'status': 'running', 'config': cfg.model_dump(mode = 'json'), 'base_batch_id': state['batch_id'],
                  'batch_id': None, 'processing_version': VERSION, 'policy': POLICY, 'expected_days': len(days), 'requests': [], 'issues': [],
                  'derivation_reference': CSI800_REFERENCE, 'limitations': ['供应商周频历史归档，不证明调整公告时刻及临时调整日精度', '查询日期必须逐日归档，不使用今天名单倒推或无限延续旧名单'],
                  'inputs': {t: state['tables'].get(t, {}) for t in ('calendar', 'instruments', 'index_constituents')}}
        _atomic_json(report_path, report)
        frames, reuse, needed = [], 0, []
        for day in days:
            for index in cfg.indices:
                existing = old[old.date.eq(day) & old['index'].eq(index)] if len(old) else pd.DataFrame()
                if len(existing) and not cfg.force:
                    try: frames.append(membership_records(existing, index, [day], cfg.max_age_days)); reuse += 1
                    except ValueError as exc: report['issues'].append({'date': str(day), 'index': index, 'rule': 'invalid_existing', 'detail': str(exc)})
                else: needed.append((day, index))
        fetched = {}
        try:
            with (source or BaoStock(root)).session() if needed and not report['issues'] else nullcontext() as src:
                for k, (day, index) in enumerate(needed if not report['issues'] else []):
                    try:
                        bases = ('000300.SH', '000905.SH') if index == '000906.SH' else (index,)
                        for base in bases:
                            if (day, base) in fetched: continue
                            response = src.index_constituents(base, day)
                            path = raw.save(root, 'baostock', 'index_constituents', f'{base}_{day}', response) if response is not None else None
                            sha = file_sha(path) if path else None
                            request = {'date': str(day), 'index': base, 'rows': len(response) if response is not None else None, 'status': 'received',
                                       'raw_file': str(path.relative_to(store.root)) if path else None, 'raw_sha256': sha}
                            report['requests'].append(request)
                            frame = normalize_constituents(response, base, day, master, sha, cfg.max_age_days)
                            fetched[day, base] = frame
                            request.update(status = 'validated', source_date = str(frame.source_date.iloc[0]))
                        frames.append(combine_csi800(fetched[day, '000300.SH'], fetched[day, '000905.SH']) if index == '000906.SH' else fetched[day, index])
                        if log: log(f'成分归档 {k + 1}/{len(needed)} {day} {index}')
                    except Exception as exc:
                        if report['requests'] and report['requests'][-1]['status'] == 'received': report['requests'][-1]['status'] = 'rejected'
                        report['issues'].append({'date': str(day), 'index': index, 'rule': 'fetch_or_schema_failed', 'detail': f'{type(exc).__name__}: {exc}'})
                        break
                    if (k + 1) % 10 == 0: _atomic_json(report_path, report)
        except Exception as exc: report['issues'].append({'rule': 'source_session_failed', 'detail': f'{type(exc).__name__}: {exc}'})
        report.update(status = 'rejected', reused_snapshots = reuse)
        if not report['issues']:
            try:
                incoming = pd.concat(frames, ignore_index = True)
                for index in cfg.indices: membership_records(incoming, index, days, cfg.max_age_days)
                replaced = set(zip(incoming.date, incoming['index']))
                keep = old[[pair not in replaced for pair in zip(old.date, old['index'])]] if len(old) else old
                merged = pd.concat([keep, incoming], ignore_index = True) if len(keep) else incoming
                parts = {}
                for year, rows in merged.groupby(pd.to_datetime(merged.date).dt.year):
                    entry = store.write_partition('index_constituents', str(year), rows)
                    if entry != state['tables'].get('index_constituents', {}).get(str(year)): parts[str(year)] = entry
                report.update(requested_rows = len(incoming), total_rows = len(merged), strict_usable_rows = 0)
                if parts:
                    bid = store.write_batch({'index_constituents': parts}, note = f'周频历史成分归档 {aid}')
                    report.update(status = 'validated', batch_id = bid); _atomic_json(report_path, report)
                    store.publish(bid); report['status'] = 'published'
                else: report.update(status = 'unchanged', batch_id = state['batch_id'])
            except Exception as exc:
                report.update(status = 'failed', error = f'{type(exc).__name__}: {exc}'); _atomic_json(report_path, report); raise
        _atomic_json(report_path, report)
        return {**report, 'report_file': str(report_path)}


def constituents_at(root, index, day, snapshot = None, limit = 1000, offset = 0):
    if index not in SIZES: raise ValueError('成分接口不支持该指数')
    day = date.fromisoformat(str(day))
    if not 1 <= limit <= 1000 or offset < 0: raise ValueError('分页参数无效')
    store = Store(root); state = store.state(snapshot)
    files = [str(store.root / p['file']) for p in state['tables'].get('index_constituents', {}).values()]
    if not files: frame = pd.DataFrame()
    else:
        with duckdb.connect() as db: frame = db.execute('select * from read_parquet(?) where "index" = ? and date = ? order by instrument', [files, index, day]).df()
    return {'index': index, 'date': str(day), 'snapshot': snapshot, 'total': len(frame), 'limit': limit, 'offset': offset,
            'evidence': POLICY, 'strict_evidence': False, 'data': json.loads(frame.iloc[offset:offset + limit].to_json(orient = 'records', date_format = 'iso', force_ascii = False))}
