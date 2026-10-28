"""Synthetic original C-VIX scheduling/holding/fee callbacks; no fills or returns."""
import argparse
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import research_strategy_batch99 as stage
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-original-timing-callbacks')


def study(loader=None):
    loader = loader or stage.source_namespace; cases = []
    for signal in ('flat_buy', 'rising_sell'):
        for held in (False, True):
            ns = loader(0); calls = []; ns['g'].target_etf = '510050.XSHG'
            ns['g'].fast_N = 5; ns['g'].slow_N = 10
            def days(**kwargs):
                calls.append(['get_trade_days', kwargs])
                return pd.bdate_range(end='2018-07-20', periods=15)
            class VIX:
                def __init__(self, *args): calls.append(['CH_VIX', list(args)])
                def GetCBOE_VIX(self):
                    calls.append(['GetCBOE_VIX', 'synthetic_fixture_not_option_history'])
                    return pd.DataFrame({'CH_VIX': [20.]*15 if signal == 'flat_buy' else np.arange(15., 30.)})
            ns['get_trade_days'] = days; ns['CH_VIX'] = VIX
            ns['order_target_value'] = lambda *args: calls.append(['order_target_value', list(args)])
            ns['order_target'] = lambda *args: calls.append(['order_target', list(args)])
            context = SimpleNamespace(previous_date=dt.date(2018, 7, 20),
                current_dt=dt.datetime(2018, 7, 23, 9, 30),
                portfolio=SimpleNamespace(total_value=123456., available_cash=1000.,
                    long_positions={'510050.XSHG': object()} if held else {}))
            with redirect_stdout(StringIO()): ns['Trade'](context)
            cases.append({'case': signal+('_held' if held else '_empty'), 'calls': calls,
                          'scope': 'synthetic_original_callback_not_fill'})
    for date in ('2008-12-31', '2010-01-04', '2012-01-04', '2015-02-09'):
        ns = loader(0); calls = []; ns['datetime'] = dt
        ns['set_slippage'] = lambda value: calls.append(['slippage', value])
        ns['FixedSlippage'] = lambda value: value
        ns['set_commission'] = lambda value: calls.append(['commission', value])
        ns['PerTrade'] = lambda **kwargs: kwargs
        ns['before_trading_start'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(date)))
        cases.append({'case': 'fees_'+date, 'calls': calls,
                      'scope': 'synthetic_injected_datetime_platform_equivalence_unproven'})
    result = {'cases': cases, 'case_count': 8, 'source_sha256': stage.SOURCE_SHA[0],
              'new_trading_backtests': 0, 'real_option_history_used': False}
    require(json.loads(json.dumps(result, allow_nan=False)) == result, 'JSON callbacks changed')
    return result


def worker(mode):
    require(mode in ('study', 'replay'), 'Unexpected mode')
    frozen = read(DIRECTORY/'prepared.json')
    for name, digest in frozen['implementation_sha256'].items(): require(file_sha(name) == digest, 'Timing worker changed')
    require(not (DIRECTORY/(mode+'.json')).exists(), 'Immutable timing output exists')
    with network_blocked(): result = study()
    save(DIRECTORY/(mode+'.json'), result)
    return {'status': 'ok', 'mode': mode, 'sha256': file_sha(DIRECTORY/(mode+'.json'))}


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    paths = ('scripts/diagnose_strategy_batch99_timing.py', 'tests/unit/test_batch99_timing.py',
             'scripts/research_strategy_batch99.py', 'scripts/recover_strategy_batch94_reference.py')
    save(DIRECTORY/'prepared.json', {'implementation_sha256': {n: file_sha(n) for n in paths},
                                    'source_sha256': stage.SOURCE_SHA[0]})
    return {'status': 'prepared', 'cases': 8}


def run(mode):
    require(not (DIRECTORY/(mode+'-command.json')).exists(), 'Immutable timing command exists')
    command = [sys.executable, '-B', '-m', 'scripts.diagnose_strategy_batch99_timing', 'worker', '--mode', mode]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (DIRECTORY/(mode+'.stdout.log')).write_text(result.stdout, encoding='utf-8')
    (DIRECTORY/(mode+'.stderr.log')).write_text(result.stderr, encoding='utf-8')
    save(DIRECTORY/(mode+'-command.json'), {'command': command, 'returncode': result.returncode,
        'stdout_sha256': file_sha(DIRECTORY/(mode+'.stdout.log')),
        'stderr_sha256': file_sha(DIRECTORY/(mode+'.stderr.log'))})
    require(result.returncode == 0, 'Timing worker failed; logs retained')
    return json.loads(result.stdout.splitlines()[-1])


def offline():
    require(all(read(DIRECTORY/(mode+'-command.json'))['returncode'] == 0 for mode in ('study', 'replay')) and
            file_sha(DIRECTORY/'study.json') == file_sha(DIRECTORY/'replay.json'), 'Timing replay differs')
    result = {'status': 'match', 'differences': 0, 'fresh_process': True, 'network_blocked': True,
              'cases': 8, 'sha256': file_sha(DIRECTORY/'study.json'), 'new_trading_backtests': 0}
    save(DIRECTORY/'offline.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'replay', 'worker', 'offline'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    if args.action == 'worker': result = worker(args.mode)
    elif args.action in ('study', 'replay'): result = run(args.action)
    else: result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
