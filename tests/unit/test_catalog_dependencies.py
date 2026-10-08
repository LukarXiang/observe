import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_history_financial_queries_register_api_and_data_gap(tmp_path, prefix, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = ('def trade(context):\n'
            f'    {prefix}get_history_fundamentals(stocks, fields=[indicator.roe], watch_date=context.previous_date)\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'history.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_history_fundamentals' in record['apis']
    assert '历史财务版本/公告时间或每日市值/股本待核实' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['get_history_fundamentals_extra', 'my_get_history_fundamentals'])
@pytest.mark.parametrize('legacy', [False, True])
def test_history_financial_similar_names_are_not_api(tmp_path, name, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = f'def trade(context):\n    {name}(q)\n    order_value("600000.XSHG", 1000)\n'
    if legacy: code += '    print context.current_dt\n'
    (source / 'other.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_history_fundamentals' not in record['apis']
    assert '历史财务版本/公告时间或每日市值/股本待核实' not in record['gaps']


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_continuous_financial_queries_register_api_and_data_gap(tmp_path, prefix, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = ('def handle(context):\n'
            f'    {prefix}get_fundamentals_continuously(q, count=250)\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'continuous.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_fundamentals_continuously' in record['apis']
    assert '历史财务版本/公告时间或每日市值/股本待核实' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_money_flow_queries_register_historical_flow_gap(tmp_path, prefix, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = ('def trade(context):\n'
            f'    {prefix}get_money_flow(stocks, end_date=context.previous_date, fields=["net_pct_main"], count=30)\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'flow.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_money_flow' in record['apis']
    assert '历史资金流字段/供应商口径和可用时点待核实' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_valuation_queries_register_historical_valuation_gap(tmp_path, prefix, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = ('def trade(context):\n'
            f'    {prefix}get_valuation(stocks, end_date=context.previous_date, fields=["pe_ratio"], count=1)\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'value.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_valuation' in record['apis']
    assert '历史财务版本/公告时间或每日市值/股本待核实' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('prefix', ['finance.', 'finance . ', 'jqdata.finance.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_platform_finance_queries_register_table_vintage_gap(tmp_path, prefix, legacy):
    source = tmp_path / 'sources'; source.mkdir()
    code = ('def trade(context):\n'
            f'    {prefix}run_query(query(finance.STK_SHAREHOLDER_FLOATING_TOP10))\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'table.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'finance.run_query' in record['apis']
    assert 'finance平台表/历史披露版本与字段口径待核实' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['financeXrun_query', 'other_finance.run_query', 'finance.run_query_extra'])
def test_finance_qualified_name_is_literal(tmp_path, name):
    source = tmp_path / 'sources'; source.mkdir()
    (source / 'other.py').write_text(f'def trade(context):\n    {name}(q)\n    order_value("600000.XSHG", 1000)\n', encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    record = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'finance.run_query' not in record['apis']
