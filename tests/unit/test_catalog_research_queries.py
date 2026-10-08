import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


def scan(tmp_path, code):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'query.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('code', [
    'macro.run_query(query(macro.MAC_MANUFACTURING_PMI.pmi))\n',
    'finance.run_query(query(finance.STK_INCOME_STATEMENT.code))\n',
    'run_query(query(TABLE.field))\n',
    'database.run_query(query(TABLE.field))\n',
    'def research():\n    return macro.run_query(query(macro.MAC_MANUFACTURING_PMI.pmi))\n',
])
def test_queries_without_orders_or_callbacks_are_research(tmp_path, code):
    item = scan(tmp_path, code)
    assert item['syntax_error'] is None
    assert 'run_query' in item['called_functions']
    assert item['status'] == '非交易研究脚本'
    assert item['review_status'] == '静态扫描，尚未人工审查'
    if 'macro.run_query' in item['apis']:
        assert 'macro平台表/历史发布版本与可用时点待核实' in item['gaps']
    if 'finance.run_query' in item['apis']:
        assert 'finance平台表/历史披露版本与字段口径待核实' in item['gaps']


@pytest.mark.parametrize('order', ['order', 'order_value', 'order_target', 'order_target_value', 'order_target_percent'])
def test_query_with_order_remains_trading_candidate(tmp_path, order):
    item = scan(tmp_path, 'macro.run_query(query(TABLE.field))\n' + order + '("600000.XSHG", 1000)\n')
    assert item['status'] == '待数据'
    assert 'macro平台表/历史发布版本与可用时点待核实' in item['gaps']


@pytest.mark.parametrize('schedule', ['run_daily', 'run_weekly', 'run_monthly'])
def test_query_with_callback_remains_trading_candidate(tmp_path, schedule):
    item = scan(tmp_path, 'macro.run_query(query(TABLE.field))\n' + schedule + '(trade)\n')
    assert item['status'] == '待数据'


def test_unknown_run_call_keeps_existing_conservative_classification(tmp_path):
    item = scan(tmp_path, 'macro.run_query(query(TABLE.field))\nrun_custom(research)\n')
    assert item['status'] == '待数据'


def test_unparseable_source_is_not_confirmed_research(tmp_path):
    item = scan(tmp_path, 'from jqdata import macro\n'
                'macro.run_query(query(TABLE.field))\nprint context.current_dt\n')
    assert item['syntax_error'] is not None
    assert item['called_functions'] is None
    assert item['status'] == '待数据'
