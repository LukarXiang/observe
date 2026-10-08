import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


def catalog_row(tmp_path, code, legacy):
    source = tmp_path / 'source'; source.mkdir()
    text = 'def trade(context):\n' + code + '\n    order_value("600000.XSHG", 1000)\n'
    if legacy: text += '    print context.current_dt\n'
    (source / 'source.py').write_text(text, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('api', ['marginsec_open', 'marginsec_close'])
@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
def test_margin_calls_register_exact_api(tmp_path, api, prefix, legacy):
    row = catalog_row(tmp_path, f'    {prefix}{api}("510500.XSHG", 100000)', legacy)
    assert api in row['apis']
    assert bool(row['syntax_error']) == legacy


@pytest.mark.parametrize('name', ['not_marginsec_open', 'marginsec_close_extra'])
@pytest.mark.parametrize('legacy', [False, True])
def test_similar_names_do_not_register_margin(tmp_path, name, legacy):
    row = catalog_row(tmp_path, f'    {name}("510500.XSHG", 100000)', legacy)
    assert 'marginsec_open' not in row['apis'] and 'marginsec_close' not in row['apis']


@pytest.mark.parametrize('keyword', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_weekly_frequency_is_metadata_not_intraday(tmp_path, keyword, legacy):
    code = ('    get_bars("600000.XSHG", count=48, unit="1w")' if keyword
            else '    get_bars("600000.XSHG", 48, "1w", "close")')
    row = catalog_row(tmp_path, code, legacy)
    assert '1w' in row['frequencies']
    assert '盘中事件/撮合与对应分钟或Tick股票池' not in row['gaps']


@pytest.mark.parametrize('unit', ['1d', '1m'])
@pytest.mark.parametrize('legacy', [False, True])
def test_existing_daily_minute_frequency_contract(tmp_path, unit, legacy):
    row = catalog_row(tmp_path, f'    get_bars("600000.XSHG", 10, "{unit}", "close")', legacy)
    assert unit in row['frequencies']
    assert ('盘中事件/撮合与对应分钟或Tick股票池' in row['gaps']) == (unit == '1m')
