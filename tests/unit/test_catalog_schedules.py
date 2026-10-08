import json
from pathlib import Path

import pytest

from observe.strategy_catalog import catalog_strategies


def scan(tmp_path, code):
    source = tmp_path / 'sources'; source.mkdir()
    (source / 'scheduled.py').write_text(code, encoding='utf-8')
    result = catalog_strategies(tmp_path / 'data', source)
    return json.loads((Path(result['output']) / 'catalog.json').read_text(encoding='utf-8'))[0]


@pytest.mark.parametrize('scheduler,args', [('run_daily', 'trade'), ('run_weekly', 'trade, 1'), ('run_monthly', 'trade, -1')])
@pytest.mark.parametrize('prefix', ['', 'jqdata.'])
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('time,keyword', [('09:50', False), ('every_bar', True)])
def test_scheduled_intraday_calls_register_time_and_dependency(tmp_path, scheduler, args, prefix, legacy, time, keyword):
    argument = f'time={time!r}' if keyword else repr(time)
    code = (f'def initialize(context):\n    {prefix}{scheduler}({args}, {argument})\n'
            'def trade(context):\n    order_value("600000.XSHG", 1000)\n')
    if legacy: code += '    print context.current_dt\n'
    record = scan(tmp_path, code)
    assert record['scheduled_times'] == [time]
    assert '盘中事件/撮合与对应分钟或Tick股票池' in record['gaps']
    assert record['status'] == '待数据'
    assert bool(record['syntax_error']) == legacy


@pytest.mark.parametrize('legacy', [False, True])
def test_comments_strings_unrelated_calls_and_dynamic_times_do_not_supply_literal_schedule(tmp_path, legacy):
    code = ('''def initialize(context):
    # run_daily(trade, time='09:50')
    example = "run_weekly(trade, 1, '11:28')"
    documentation = """run_daily(trade, time='every_bar')"""
    unrelated(time='14:50')
    run_daily(trade, time=context.schedule)
    run_weekly(trade, weekday=context.weekday)
def trade(context):
    order_value("600000.XSHG", 1000)
''')
    if legacy: code += '    print context.current_dt\n'
    record = scan(tmp_path, code)
    assert record['scheduled_times'] == []
    assert '盘中事件/撮合与对应分钟或Tick股票池' not in record['gaps']


@pytest.mark.parametrize('legacy', [False, True])
def test_schedule_arguments_are_parsed_without_evaluating_code(tmp_path, legacy):
    code = '''def initialize(context):
    run_weekly(make_callback("every_bar", nested(1, 2)), 1,
               time="11:28", reference_security="000300.XSHG")
    run_daily(trade, 'before_open')
def trade(context):
    order_value("600000.XSHG", 1000)
'''
    if legacy: code += '    print context.current_dt\n'
    record = scan(tmp_path, code)
    assert record['scheduled_times'] == ['11:28', 'before_open']
    assert '盘中事件/撮合与对应分钟或Tick股票池' in record['gaps']


@pytest.mark.parametrize('time', ['before_open', 'open', 'after_close', '07:00', '15:30'])
def test_non_intraday_schedule_preserves_daily_candidate(tmp_path, time):
    record = scan(tmp_path, f'def initialize(context):\n    run_daily(trade, time={time!r})\n'
                  'def trade(context):\n    order_value("600000.XSHG", 1000)\n')
    assert record['scheduled_times'] == [time]
    assert '盘中事件/撮合与对应分钟或Tick股票池' not in record['gaps']
