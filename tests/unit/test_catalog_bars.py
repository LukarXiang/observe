import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('unit', ['1d', '1m'])
def test_bars_queries_register_exact_api_and_frequency(tmp_path, prefix, legacy, unit):
    source = tmp_path / 'source'; source.mkdir()
    code = ('def trade(context):\n'
            f'    {prefix}get_bars("600000.XSHG", 10, "{unit}", "close", include_now=True)\n'
            '    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    (source / 'bars.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    row = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_bars' in row['apis']
    assert unit in row['frequencies']
    assert ('盘中事件/撮合与对应分钟或Tick股票池' in row['gaps']) == (unit == '1m')
    assert bool(row['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['my_get_bars', 'get_bars_extra'])
@pytest.mark.parametrize('legacy', [False, True])
def test_similar_names_are_not_bars_api(tmp_path, name, legacy):
    source = tmp_path / 'source'; source.mkdir()
    code = f'def trade(context):\n    {name}(stocks)\n    order_value("600000.XSHG", 1000)\n'
    if legacy: code += '    print context.current_dt\n'
    (source / 'other.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    row = json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]
    assert 'get_bars' not in row['apis']
