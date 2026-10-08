import json
from pathlib import Path

import pytest

from observe.strategy_catalog import REVIEWS, catalog_strategies


ASSET = '衍生品候选'
GAP = '专用资产行情与交易规则'


def scan(tmp_path, code):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'contract.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('call', [
    "get_dominant_future('IF')", "jqdata.get_dominant_future('IF')",
    "get_dominant_future ('RB', date=context.previous_date)",
    "# get_dominant_future('IF')",
])
@pytest.mark.parametrize('legacy', [False, True])
def test_dominant_contract_query_records_asset_and_rule_dependency(tmp_path, call, legacy):
    code = 'from jqdata import *\n' + call + '\norder_target(code, 1)\n'
    if legacy: code += 'print code\n'
    item = scan(tmp_path, code)
    assert 'get_dominant_future' in item['apis']
    assert ASSET in item['asset_scope'] and GAP in item['gaps']
    assert item['status'] == '待数据'
    assert bool(item['syntax_error']) == legacy


@pytest.mark.parametrize('symbol', ['IF1906.CCFX', 'IH1602.CCFX', 'IC8888.CCFX', 'IM9999.CCFX'])
@pytest.mark.parametrize('quote', ["'", '"'])
def test_explicit_ccfx_contract_without_dominant_query_records_dependency(tmp_path, symbol, quote):
    item = scan(tmp_path, f'order_target({quote}{symbol}{quote}, 1)\n')
    assert ASSET in item['asset_scope'] and GAP in item['gaps']
    assert 'get_dominant_future' not in item['apis']
    assert item['status'] == '待数据'


@pytest.mark.parametrize('symbol', ['IF1906XCCFX', 'IF1906.CCFXX', 'IF1906.CCFX.other', 'IF190.CCFX'])
def test_contract_suffix_and_delimiters_are_literal(tmp_path, symbol):
    item = scan(tmp_path, f'order_target("{symbol}", 1)\n')
    assert ASSET not in item['asset_scope'] and GAP not in item['gaps']


@pytest.mark.parametrize('name', ['my_get_dominant_future', 'get_dominant_future_extra'])
def test_similar_query_names_do_not_register_dominant_api(tmp_path, name):
    item = scan(tmp_path, f'{name}("IF")\norder_target(code, 1)\n')
    assert 'get_dominant_future' not in item['apis']
    assert ASSET not in item['asset_scope'] and GAP not in item['gaps']


def test_security_info_and_stock_benchmark_do_not_imply_futures(tmp_path):
    item = scan(tmp_path, 'get_security_info("600000.XSHG")\nset_benchmark("000300.XSHG")\norder_target(code, 1)\n')
    assert ASSET not in item['asset_scope'] and GAP not in item['gaps']
    assert '股票候选' in item['asset_scope']


def test_comment_contract_is_only_a_static_candidate(tmp_path):
    item = scan(tmp_path, '# reference_security="IF8888.CCFX"\norder_target(code, 1)\n')
    assert ASSET in item['asset_scope'] and GAP in item['gaps']
    assert item['review_status'] == '静态扫描，尚未人工审查'


def test_manual_asset_rules_and_status_override_static_query_candidate(tmp_path, monkeypatch):
    review = {'asset_scope': ['人工资产'], 'gaps': ['人工缺口'], 'status': '暂不可复现',
              'review_status': '人工审查完成', 'rules': {'known': 'frozen'}}
    monkeypatch.setitem(REVIEWS, 'contract.py', review)
    item = scan(tmp_path, 'get_dominant_future("IF")\norder_target("IF1906.CCFX", 1)\n')
    assert 'get_dominant_future' in item['apis']
    assert all(item[k] == v for k, v in review.items())


def test_research_only_futures_query_has_no_invented_trading_status(tmp_path):
    item = scan(tmp_path, 'get_dominant_future("IF")\n')
    assert ASSET in item['asset_scope'] and GAP in item['gaps']
    assert item['status'] == '非交易研究脚本'


def test_stock_dependency_and_existing_contract_query_are_retained(tmp_path):
    item = scan(tmp_path, 'get_future_contracts("IF")\nget_dominant_future("IF")\norder_value("600000.XSHG", 1000)\n')
    assert {'get_dominant_future', 'get_future_contracts', 'order_value'} <= set(item['apis'])
    assert ASSET in item['asset_scope'] and '股票候选' in item['asset_scope']
    assert item['gaps'].count(GAP) == 1
