import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from observe.data.financial_history import FinancialConflict, derive_periods, latest_reports, visible_records
from observe.data.financial_schema import field_dictionary, parse_number
from observe.data.financials import _normalize, import_financials, next_session_available, read_financial_archive, security_mapping
from observe.data.store import Store


def _base(tmp_path):
    store = Store(tmp_path / 'data')
    inst = pd.DataFrame({'instrument': ['000001.SZ', '600000.SH'], 'kind': ['stock', 'stock']})
    cal = pd.DataFrame({'date': pd.date_range('2024-01-01', '2025-05-01'), 'is_open': False})
    cal.loc[cal.date.dt.weekday < 5, 'is_open'] = True
    parts = {t: {'all': store.write_partition(t, 'all', df)} for t, df in [('instruments', inst), ('calendar', cal)]}
    store.publish(store.write_batch(parts)); return store


def _inputs(tmp_path):
    annual, quarterly, controls = [tmp_path / name for name in ('annual.csv', 'quarterly.csv', 'controls.dta')]
    pd.DataFrame({'证券代码': ['000001', '000001', '099999'], '年份': ['2023', '2023', 'bad'], '净利润': ['10', '11', 'not_numeric'],
                  '报表类型': ['A', 'B', ''], '年报公布日期': ['2024-04-01'] * 3, '未映射字段': ['001', 'NULL', '']}).to_csv(annual, index = False)
    pd.DataFrame({'证券代码': ['600000'], '季度日期': ['2024-03-31'], '净利润': ['12.5'], '报表类型': ['A']}).to_csv(quarterly, index = False)
    pd.DataFrame({'stkcode': ['000001'], 'year': [2023.0], 'marketvalue': [100.0], 'everSTPT': [1]}).to_stata(controls, write_index = False, version = 118,
                                                                                                           variable_labels = {'marketvalue': '特定定义，不是总市值', 'everSTPT': '曾经ST'})
    return annual, quarterly, controls


def test_complete_archive_conflicts_unknown_codes_and_idempotence(tmp_path):
    store = _base(tmp_path); old = store.snapshot(); old_bytes = store.published_path.read_bytes(); inputs = _inputs(tmp_path)
    result = import_financials(store.root, *inputs, chunksize = 1)
    assert result['rows'] == {'financial_annual': 3, 'financial_quarterly': 1, 'company_controls_annual': 1}
    df = store.load('financial_annual')
    assert df.source_code.tolist() == ['000001', '000001', '099999']
    assert df['未映射字段'].tolist() == ['001', 'NULL', '']
    assert df.net_profit_ytd.iloc[:2].tolist() == [10, 11] and np.isnan(df.net_profit_ytd.iloc[2])
    assert df.mapping_status.tolist() == ['matched', 'matched', 'unknown_code']
    audit = json.loads(Path(result['audit']).read_text())['tables']['financial_annual']
    assert audit['conflicting_rows'] == 2 and audit['quarantined_rows'] == 3
    assert sum(i['count'] for i in audit['numeric_issues']) == 1
    assert not store.load('financial_availability').strict_usable.any()
    assert not store.state(old)['tables'].get('financial_annual')
    assert store.published_path.read_bytes() != old_bytes
    new_bytes = store.published_path.read_bytes()
    assert import_financials(store.root, *inputs, chunksize = 2)['status'] == 'unchanged'
    assert new_bytes == store.published_path.read_bytes()
    sid = store.snapshot()
    subset = read_financial_archive(store.root, 'financial_annual', sid, ['000001.SZ'], '2023-01-01', '2023-12-31', ['source_code', '净利润'])
    assert len(subset) == 2 and subset['净利润'].tolist() == ['10', '11']
    with pytest.raises(ValueError, match = '未知财务列'): read_financial_archive(store.root, 'financial_annual', sid, columns = ['fake'])


def test_failed_import_does_not_publish_or_overwrite(tmp_path, monkeypatch):
    store = _base(tmp_path); before = store.published_path.read_bytes(); inputs = _inputs(tmp_path)
    def fail(*args, **kwargs): raise RuntimeError('write failed')
    monkeypatch.setattr(Store, 'write_partition', fail)
    with pytest.raises(RuntimeError, match = 'write failed'): import_financials(store.root, *inputs, chunksize = 1)
    assert store.published_path.read_bytes() == before
    assert list((store.root / 'audits/financials').glob('*failed.json'))


def test_financial_store_never_silently_drops_versions(tmp_path):
    df = pd.DataFrame({'source_sha256': ['a', 'a'], 'source_row': [1, 1], 'value': [1, 2]})
    with pytest.raises(ValueError, match = '禁止静默覆盖'): Store(tmp_path).write_partition('financial_annual', '2024', df)


def test_dates_units_codes_and_missing_values():
    cal = ['2024-04-03', '2024-04-08', '2024-04-09']   # 清明休市；公告日不是可用日
    actual = next_session_available(pd.Series(['2024-04-03', '2024-04-05', 'bad', '2024-04-09']), cal)
    assert actual.iloc[0] == pd.Timestamp('2024-04-08 09:30', tz = 'Asia/Shanghai')
    assert actual.iloc[1] == actual.iloc[0] and actual.iloc[2:].isna().all()
    mapping, ambiguous = security_mapping(pd.DataFrame({'instrument': ['000001.SZ', '000001.SH', '600001.SH'], 'kind': ['stock', 'index', 'stock']}))
    assert mapping['000001'] == '000001.SZ' and not ambiguous
    number, bad = parse_number(pd.Series(['0', '-2', 'NULL', 'inf', 'broken', '1e3']))
    assert number.iloc[[0, 1, 5]].tolist() == [0, -2, 1000] and bad.tolist() == [False, False, False, True, True, False]
    evidence = {'FS_Comins': {'path': 'income.txt', 'fields': [{'label': '净利润', 'unit': '元', 'source_field': 'B002000000', 'description': '累计净利润'}]}}
    item = field_dictionary(['证券代码', '净利润'], 'financial_annual', 'sha', evidence = evidence)[1]
    assert item['standard_name'] == 'net_profit_ytd' and item['unit'] == '元' and item['period_kind'] == 'ytd_flow'


def _records():
    return pd.DataFrame([
        ('2023-12-31', 40, '2024-04-08 09:30', 'annual-v1'),
        ('2024-03-31', 12, '2024-04-30 09:30', 'q1-v1'),
        ('2023-12-31', 42, '2024-05-10 09:30', 'annual-v2'),
    ], columns = ['report_period', 'value', 'available_at', 'version_id']).assign(instrument = '000001.SZ', field = 'net_profit_ytd', source_group = 'income',
                                                                            strict_usable = True, report_type = 'A', period_kind = 'ytd_flow')


def test_announcement_gate_revision_order_and_strict_evidence():
    data = _records()
    assert not len(visible_records(data, '2024-04-07 16:00').data)
    assert not len(visible_records(data, '2024-04-08 09:29').data)
    assert latest_reports(visible_records(data, '2024-04-08 09:30').data).value.tolist() == [40]
    result = latest_reports(visible_records(data, '2024-05-11 16:00').data)
    assert result.value.tolist() == [12]      # 后来的年报更正不替代较新的 Q1
    data['strict_usable'] = 'False'
    assert not len(visible_records(data, '2024-05-11 16:00').data)
    with pytest.raises(ValueError, match = 'lag_days'): visible_records(data, '2024-05-11', mode = 'exploratory')
    tie = pd.concat([_records(), _records().iloc[[0]].assign(value = 99)])
    with pytest.raises(FinancialConflict): latest_reports(visible_records(tie, '2024-04-10 16:00').data)


def test_flow_hand_calculation_and_missing_quarters():
    data = pd.DataFrame({'report_period': ['2023-03-31', '2023-12-31', '2024-03-31', '2024-06-30', '2024-12-31'], 'value': [8, 40, 12, 30, 70]})
    data = data.assign(instrument = '000001.SZ', field = 'net_profit_ytd', available_at = pd.Timestamp('2025-04-01 09:30', tz = 'Asia/Shanghai'),
                       source_group = 'income', period_kind = 'ytd_flow', version_id = 'v1')
    sq = derive_periods(data, 'net_profit_ytd', 'single_quarter')
    assert sq.value.iloc[2:4].tolist() == [12, 18] and np.isnan(sq.value.iloc[-1])
    ttm = derive_periods(data, 'net_profit_ytd', 'ttm')
    assert ttm.value.iloc[2] == 44 and np.isnan(ttm.value.iloc[3]) and ttm.value.iloc[-1] == 70
    assert derive_periods(data, 'net_profit_ytd', 'yoy').value.iloc[2] == .5
    with pytest.raises(ValueError, match = '存量'): derive_periods(data.assign(period_kind = 'stock'), 'net_profit_ytd', 'ttm')


def test_merged_row_announcement_and_report_type_never_cross_sources():
    raw = pd.DataFrame({'证券代码': ['000001'], '年份': ['2023'], '净利润': ['10'], '资产总计': ['100'], '财务指标文件_总资产': ['110'],
                        '年报公布日期': ['2024-04-03'], '报表类型编码': ['A'], '报表类型': ['B'], '资产负债表_报表类型': ['A']})
    dictionary = field_dictionary(raw.columns, 'financial_annual', 'sha')
    _, available, _ = _normalize(raw, 'financial_annual', 'sha', 0, {'000001': '000001.SZ'}, set(), ['2024-04-03', '2024-04-08'], dictionary)
    rows = available.set_index('source_group')
    assert rows.loc['far_finidx', 'available_at'] == pd.Timestamp('2024-04-08 09:30', tz = 'Asia/Shanghai')
    assert pd.isna(rows.loc['income', 'available_at']) and pd.isna(rows.loc['balance', 'available_at'])
    assert rows.loc['income', 'report_type'] == 'B' and rows.loc['balance', 'report_type'] == 'A'
    assert pd.isna(rows.loc['far_finidx', 'report_type']) and not available.strict_usable.any()


def test_ttm_uses_only_versions_visible_at_the_same_decision():
    data = pd.DataFrame({'report_period': ['2023-03-31', '2023-12-31', '2024-03-31', '2023-03-31'], 'value': [20, 100, 25, 30],
                         'available_at': ['2023-05-01 09:30', '2024-04-30 09:30', '2024-05-06 09:30', '2024-05-15 09:30'], 'version_id': ['q1-v1', 'annual-v1', 'q1-v1', 'q1-v2']})
    data = data.assign(instrument = '000001.SZ', field = 'net_profit_ytd', source_group = 'income', strict_usable = True, report_type = 'A', period_kind = 'ytd_flow')
    before = derive_periods(visible_records(data, '2024-05-10 16:00').data, 'net_profit_ytd', 'ttm')
    later = derive_periods(visible_records(data, '2024-05-20 16:00').data, 'net_profit_ytd', 'ttm')
    assert before.value.iloc[-1] == 105 and later.value.iloc[-1] == 95
    assert before.available_at.iloc[-1] == pd.Timestamp('2024-05-06 09:30', tz = 'Asia/Shanghai')
    assert later.available_at.iloc[-1] == pd.Timestamp('2024-05-15 09:30', tz = 'Asia/Shanghai')
