import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


def row(tmp_path, text, suffix='.py'):
    source = tmp_path / 'source'; source.mkdir()
    (source / ('prefix' + suffix)).write_text(text, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('suffix', ['.py', '.txt'])
def test_parseable_module_preserves_pre_import_inputs(tmp_path, suffix):
    item = row(tmp_path, 'pool = ["510050.XSHG", "159928.XSHE"]\n'
               'data = get_price(pool, count=60)\nimport pandas as pd\n'
               'result = pd.DataFrame(data)\n', suffix)
    assert item['code_start_line'] == 1
    assert item['syntax_error'] is None
    assert 'get_price' in item['apis']
    assert 'get_price' in item['called_functions']
    assert item['asset_scope'] == ['ETF/基金候选']
    assert 'ETF行情/公司行动/执行规则' in item['gaps']


def test_parseable_module_preserves_pre_definition_schedule(tmp_path):
    item = row(tmp_path, 'run_daily(trade, time="10:00")\n'
               'def trade(context):\n    order_value("600000.XSHG", 1000)\n')
    assert item['code_start_line'] == 1
    assert item['scheduled_times'] == ['10:00']
    assert item['status'] == '待数据'
    assert '盘中事件/撮合与对应分钟或Tick股票池' in item['gaps']


@pytest.mark.parametrize('legacy', [False, True])
def test_article_header_keeps_legacy_fallback(tmp_path, legacy):
    item = row(tmp_path, '这是一段不可解析的文章正文：\n正文不能作为Python执行\n'
               'import pandas as pd\ndef trade(context):\n'
               '    get_price("600000.XSHG", count=10)\n'
               + ('    print context.current_dt\n' if legacy else ''))
    assert item['code_start_line'] == 3
    assert bool(item['syntax_error']) == legacy
    assert 'get_price' in item['apis']


def test_comments_do_not_cause_prefix_discard(tmp_path):
    item = row(tmp_path, '# research heading\nprices = history(60, "1d", "close")\n'
               'import numpy as np\nresult = np.mean(prices)\n')
    assert item['code_start_line'] == 1
    assert 'history' in item['apis']


def test_notebook_preserves_pre_import_cells(tmp_path):
    text = json.dumps({'cells': [
        {'cell_type': 'markdown', 'source': ['文章标题']},
        {'cell_type': 'code', 'source': ['pool = ["510050.XSHG"]\nget_price(pool)']},
        {'cell_type': 'code', 'source': ['import pandas as pd\npd.DataFrame()']}]})
    item = row(tmp_path, text, '.ipynb')
    assert item['code_start_line'] == 1
    assert 'get_price' in item['apis']
    assert item['asset_scope'] == ['ETF/基金候选']
