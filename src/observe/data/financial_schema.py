"""外部合并财务表的字段证据。原字段全部保留；核心字段按来源分别解释。"""
import re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

VERSION = 'external_financials_v1'
MISSING = {'', '.', 'NULL', 'null', 'NA', 'N/A', 'nan', 'NaN', '--'}
# 名称相近的年度值、TTM、归母值从不自动合并；单位来自各原始 XLSX 第三行。
CORE = {
    'revenue_ytd': ('营业收入', 'income', 'ytd_flow', 'income.operating_revenue'),
    'net_profit_ytd': ('净利润', 'income', 'ytd_flow', 'income.net_profit'),
    'parent_net_profit_ytd': ('归属于母公司所有者的净利润', 'income', 'ytd_flow', 'income.np_parent_company_owners'),
    'operating_cashflow_ytd': ('经营活动产生的现金流量净额', 'cashflow_direct', 'ytd_flow', 'cash_flow.net_operate_cash_flow'),
    'total_assets': ('资产总计', 'balance', 'stock', 'balance.total_assets'),
    'total_liabilities': ('负债合计', 'balance', 'stock', 'balance.total_liability'),
    'parent_equity': ('归属于母公司所有者权益合计', 'balance', 'stock', None),
    'far_total_assets': ('财务指标文件_总资产', 'far_finidx', 'stock', None),
}
GROUPS = {
    'income': {'report_type': '报表类型', 'correction': '差错更正披露日期', 'definition': 'FS_Comins'},
    'cashflow_direct': {'report_type': '现金流量表直接法_报表类型', 'correction': '现金流量表直接法_差错更正披露日期', 'definition': 'FS_Comscfd'},
    'balance': {'report_type': '资产负债表_报表类型', 'correction': '资产负债表_差错更正披露日期', 'definition': 'FS_Combas'},
    'far_finidx': {'report_type': '财务指标_报表类型', 'correction': None, 'definition': 'FAR_Finidx'},
}
# 合并表中的块首列；块位置以实际表头为准，不能把首块的报表类型用于其他来源。
BLOCKS = [
    ('股票简称', 'FI_T1'), ('证券简称', 'FS_Comins'), ('发展能力_股票简称', 'FI_T8'),
    ('披露财务指标_股票简称', 'FI_T2'), ('每股指标_股票简称', 'FI_T9'), ('比率结构_股票简称', 'FI_T3'),
    ('现金流分析_股票简称', 'FI_T6'), ('现金流量表直接法_证券简称', 'FS_Comscfd'),
    ('现金流量表间接法_证券简称', 'FS_Comscfi'), ('盈利能力_股票简称', 'FI_T5'),
    ('相对价值指标_股票简称', 'FI_T10'), ('经营能力_股票简称', 'FI_T4'), ('股利分配_股票简称', 'FI_T11'),
    ('资产负债表_证券简称', 'FS_Combas'), ('风险水平_股票简称', 'FI_T7'),
    ('上市公司财务指标数据表年_证券简称', 'FI_T12'), ('财务指标_证券简称', 'FI_T13'), ('年报公布日期', 'FAR_Finidx'),
]


def read_xlsx_rows(path, count = 4):
    """只读表头、单位及抽样行，不安装 Excel 引擎，不加载整份工作簿。"""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        rows = []
        with z.open('xl/worksheets/sheet1.xml') as h:
            for _, node in ET.iterparse(h, events = ['end']):
                if not node.tag.endswith('}row'): continue
                cells = {}
                for c in node:
                    v, inline = c.find('m:v', ns), c.find('m:is', ns)
                    value = v.text if v is not None else ''.join(inline.itertext()) if inline is not None else ''
                    cells[c.get('r', '')] = strings[int(value)] if c.get('t') == 's' else value
                rows.append(list(cells.values())); node.clear()
                if len(rows) >= count: break
        return rows


def definitions(root):
    """保存原始说明原文和 XLSX 单位；同字段的多个口径均保留为候选证据。"""
    result = {}
    if root is None: return result
    for f in sorted(Path(root).rglob('*[[]DES[]]*.txt')):
        table = f.name.split('[')[0]
        if table in result: continue     # 同一说明的年度/季度副本留在 evidence_paths，不凭路径覆盖
        text = f.read_text(encoding = 'utf-8-sig')
        fields = []
        for line in text.splitlines():
            m = re.match(r'([\w]+)\s+\[([^]]+)\]\s*-\s*(.*)', line)
            if m: fields.append({'source_field': m[1], 'label': m[2], 'description': m[3], 'unit': None})
        book = f.with_name(f'{table}.xlsx')
        sample = read_xlsx_rows(book) if book.exists() else []
        if len(sample) >= 3:
            units = dict(zip(sample[0], sample[2]))
            for field in fields: field['unit'] = units.get(field['source_field'])
        result[table] = {'path': str(f), 'text': text, 'fields': fields, 'xlsx_path': str(book) if book.exists() else None,
                         'header_sample': sample, 'evidence_paths': [str(p) for p in sorted(Path(root).rglob(f.name))]}
    return result


def variable_labels(path):
    if not path or not Path(path).exists(): return {}
    with pd.read_stata(path, iterator = True, convert_categoricals = False) as r: return r.variable_labels()


def _base(label):
    return re.sub(r'^[^_]+_', '', label) if '_' in label else label


def field_dictionary(columns, table, source_sha, labels = None, evidence = None):
    labels, evidence = labels or {}, evidence or {}; rows = []; group = 'merged_unknown'
    block_map = dict(BLOCKS); core = {v[0]: (k, v) for k, v in CORE.items()}
    stata = list(labels.items())
    for k, column in enumerate(columns):
        group = block_map.get(column, group)
        matches = [f for f in evidence.get(group, {}).get('fields', []) if f['label'] in (column, _base(column))]
        # 特殊命名的核心映射须落在其原始来源块，不能从另一来源抄单位。
        mapped = core.get(column); standard = mapped[0] if mapped else None
        if mapped:
            expected = GROUPS[mapped[1][1]]['definition']
            matches = [f for f in evidence.get(expected, {}).get('fields', []) if f['label'] in (column, _base(column))]
            group = expected if len(matches) == 1 else group
        item = matches[0] if len(matches) == 1 else {}
        text_type = any(s in column for s in ('代码', '简称', '名称', '日期', '报表类型', '公告来源')) or column in ('年份', '季度日期')
        rows.append({'source_sha256': source_sha, 'table': table, 'column': column, 'position': k + 1,
                     'stata_variable': stata[k][0] if k < len(stata) else None, 'original_label': labels.get(column, stata[k][1] if k < len(stata) else column),
                     'standard_name': standard, 'source_table': group, 'source_field': item.get('source_field'),
                     'description': item.get('description', labels.get(column, '未查证，保留原字段')),
                     'unit': item.get('unit'), 'dtype': 'raw_string' if table != 'company_controls_annual' else 'original_stata_type',
                     'numeric_candidate': not text_type, 'missing_rule': '原值保留；核心数值显式识别空串/./NULL/NA/N/A/nan/NaN/--，不补零',
                     'scope': 'source_report_type_required' if mapped else 'unreviewed',
                     'period_kind': mapped[1][2] if mapped else 'unreviewed', 'formula': item.get('description'),
                     'joinquant_candidate': mapped[1][3] if mapped else None, 'joinquant_equivalence': 'not_proven',
                     'time_evidence': 'revision_unknown_strict_blocked', 'mapping_version': VERSION,
                     'evidence_path': evidence.get(group, {}).get('path')})
    return rows


def parse_number(series):
    text = series.astype('string').str.strip(); missing = text.isna() | text.isin(MISSING)
    number = pd.to_numeric(text.mask(missing), errors = 'coerce').astype(float)
    finite = number.notna() & ~number.isin([float('inf'), -float('inf')])
    bad = ~missing & ~finite
    return number.where(finite), bad
