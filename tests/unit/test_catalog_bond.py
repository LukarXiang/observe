import json
from pathlib import Path

import pytest

from observe.strategy_catalog import REVIEWS, catalog_strategies


BOND_GAP = 'bond平台表/历史行情及事件版本与专用规则待核实'
BOND_ASSET = '可转债候选'


def scan(tmp_path, code):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'bond.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('call', [
    'bond.run_query(query(bond.CONBOND_DAILY_PRICE.close))',
    'bond . run_query (query(bond.CONBOND_DAILY_PRICE.close))',
    '(bond.\nrun_query(query(bond.CONBOND_DAILY_PRICE.close)))',
    '# bond.run_query(query(bond.CONBOND_DAILY_PRICE.close))',
])
def test_bond_query_records_data_and_asset_candidates(tmp_path, call):
    item = scan(tmp_path, 'from jqdata import bond\n' + call + '\n')
    assert 'bond.run_query' in item['apis']
    assert BOND_GAP in item['gaps']
    assert BOND_ASSET in item['asset_scope']
    assert item['status'] == '非交易研究脚本'


def test_repo_rate_dependency_does_not_imply_convertible_trading(tmp_path):
    item = scan(tmp_path, 'bond.run_query(query(bond.REPO_DAILY_PRICE.close))\n')
    assert 'bond.run_query' in item['apis']
    assert BOND_GAP in item['gaps']
    assert BOND_ASSET not in item['asset_scope']
    assert item['asset_scope'] == ['待人工识别']


def test_unknown_bond_table_keeps_query_dependency(tmp_path):
    item = scan(tmp_path, 'bond.run_query(query(bond.UNKNOWN_TABLE))\n')
    assert 'bond.run_query' in item['apis']
    assert BOND_GAP in item['gaps']
    assert BOND_ASSET not in item['asset_scope']


@pytest.mark.parametrize('code', [
    'from jqdata import bond\n',
    'query(bond.CONBOND_DAILY_PRICE.close)\n',
    'run_query(query(CONBOND_DAILY_PRICE.close))\n',
    'another_bond.run_query(query(another_bond.CONBOND_DAILY_PRICE.close))\n',
])
def test_import_table_or_other_namespace_is_not_a_bond_query(tmp_path, code):
    item = scan(tmp_path, code)
    assert 'bond.run_query' not in item['apis']
    assert BOND_GAP not in item['gaps']
    assert BOND_ASSET not in item['asset_scope']


@pytest.mark.parametrize('table', ['CONBOND_BASIC_INFO', 'CONBOND_DAILY_CONVERT', 'CONBOND_CONVERT_PRICE_ADJUST'])
def test_explicit_convertible_tables_preserve_stock_dependency(tmp_path, table):
    item = scan(tmp_path, f'bond.run_query(query(bond.{table}))\norder_value("600000.XSHG", 1000)\n')
    assert BOND_ASSET in item['asset_scope']
    assert '股票候选' in item['asset_scope']
    assert BOND_GAP in item['gaps']
    assert item['status'] == '待数据'


def test_manual_review_overrides_static_asset_gap_and_status(tmp_path, monkeypatch):
    review = {'asset_scope': ['人工资产'], 'gaps': ['人工缺口'], 'status': '暂不可复现',
              'review_status': '人工审查完成', 'rules': {'known': 'frozen'}}
    monkeypatch.setitem(REVIEWS, 'bond.py', review)
    item = scan(tmp_path, 'bond.run_query(query(bond.CONBOND_DAILY_PRICE))\n')
    assert 'bond.run_query' in item['apis']
    assert all(item[k] == v for k, v in review.items())


def test_platform_query_dependencies_remain_separate(tmp_path):
    item = scan(tmp_path, 'bond.run_query(query(bond.CONBOND_DAILY_PRICE))\n'
                'finance.run_query(query(finance.STK_INCOME_STATEMENT))\n'
                'macro.run_query(query(macro.MAC_MANUFACTURING_PMI))\n')
    assert {'bond.run_query', 'finance.run_query', 'macro.run_query'} <= set(item['apis'])
    assert BOND_GAP in item['gaps']
    assert 'finance平台表/历史披露版本与字段口径待核实' in item['gaps']
    assert 'macro平台表/历史发布版本与可用时点待核实' in item['gaps']
