import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


def record(tmp_path, code, legacy=False):
    source = tmp_path / 'source'; source.mkdir()
    if legacy: code += '    print context.current_dt\n'
    (source / 'strategy.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('arguments', ["types='etf'", "types=['etf', 'lof']", "['fund']", "'open_fund'",
    "types=('lof',)", "types=['stock', 'etf', unknown]", "*args, types='fund'"])
def test_literal_dynamic_fund_pool_adds_asset_and_dependency(tmp_path, prefix, legacy, arguments):
    item = record(tmp_path, f'def trade(context):\n    {prefix}get_all_securities({arguments})\n    order_value(target, 1000)\n', legacy)
    assert item['asset_scope'] == ['ETF/基金候选']
    assert 'ETF行情/公司行动/执行规则' in item['gaps']
    assert item['status'] == '待数据'
    assert item['review_status'] == '静态扫描，尚未人工审查'
    assert bool(item['syntax_error']) == legacy


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('expression', ["get_all_securities(types=['stock'])", "get_all_securities(date='fund')",
    "get_all_securities(unknown_types)", "get_all_securities(*args)", "get_all_securities({'fund': 1})",
    "get_all_securities_extra(['etf'])", "my_get_all_securities(['fund'])",
    "snippet = \"get_all_securities(['fund'])\"", "# get_all_securities(['etf'])", "get_all_securities(types=['etf_extra'])"])
def test_nonfund_unknown_and_quoted_calls_do_not_add_fund_dependency(tmp_path, legacy, expression):
    item = record(tmp_path, f'def trade(context):\n    {expression}\n    order_value(target, 1000)\n', legacy)
    assert 'ETF/基金候选' not in item['asset_scope']
    assert 'ETF行情/公司行动/执行规则' not in item['gaps']


@pytest.mark.parametrize('legacy', [False, True])
def test_pool_does_not_replace_literal_stock_or_duplicate_fund_asset(tmp_path, legacy):
    item = record(tmp_path, '''def trade(context):
    get_all_securities(types=['etf', 'lof'])
    order_value('510300.XSHG', 1000)
    order_value('600000.XSHG', 1000)
''', legacy)
    assert item['asset_scope'] == ['ETF/基金候选', '股票候选']
    assert item['gaps'].count('ETF行情/公司行动/执行规则') == 1
