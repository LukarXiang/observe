import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


MACRO_GAP = 'macro平台表/历史发布版本与可用时点待核实'


def scan(tmp_path, code):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'macro.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('call', [
    'macro.run_query(query(macro.MAC_MANUFACTURING_PMI.pmi))',
    'macro . run_query (query(macro.MAC_MANUFACTURING_PMI.pmi))',
    '(macro.\nrun_query(query(macro.MAC_MANUFACTURING_PMI.pmi)))',
    '# macro.run_query(query(macro.MAC_MANUFACTURING_PMI.pmi))',
])
def test_macro_query_records_historical_publication_dependency(tmp_path, call):
    item = scan(tmp_path, 'from jqdata import macro\n' + call + '\norder_value("600000.XSHG", 1000)\n')
    assert 'macro.run_query' in item['apis']
    assert MACRO_GAP in item['gaps']
    assert item['status'] == '待数据'
    assert 'finance平台表/历史披露版本与字段口径待核实' not in item['gaps']
    assert '历史财务版本/公告时间或每日市值/股本待核实' not in item['gaps']


def test_macro_and_finance_queries_have_separate_dependencies(tmp_path):
    item = scan(tmp_path, 'from jqdata import macro, finance\n'
                'macro.run_query(query(macro.MAC_MANUFACTURING_PMI.pmi))\n'
                'finance.run_query(query(finance.STK_INCOME_STATEMENT.code))\n')
    assert {'macro.run_query', 'finance.run_query'} <= set(item['apis'])
    assert MACRO_GAP in item['gaps']
    assert 'finance平台表/历史披露版本与字段口径待核实' in item['gaps']
    assert item['status'] == '非交易研究脚本'


@pytest.mark.parametrize('code', [
    'from jqdata import macro\nquery(macro.MAC_MANUFACTURING_PMI.pmi)\n',
    'run_query(query(MAC_MANUFACTURING_PMI.pmi))\n',
    'another_macro.run_query(query(MAC_MANUFACTURING_PMI.pmi))\n',
])
def test_macro_import_or_other_namespace_is_not_a_macro_query(tmp_path, code):
    item = scan(tmp_path, code)
    assert 'macro.run_query' not in item['apis']
    assert MACRO_GAP not in item['gaps']
