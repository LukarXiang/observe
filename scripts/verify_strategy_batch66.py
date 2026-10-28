"""Archive ETF entitlement regression checks without generating trading returns."""
import argparse
from dataclasses import asdict
from datetime import date
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

from observe.ledger import Book, Rule, RuleSet
from observe.runs import file_sha
from observe.strategies import reproduce_strategy
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect


DIRECTORY = Path('data/staging/strategies-batch66/20261008-etf-record-entitlement')
IMPLEMENTATION = ('src/observe/ledger/book.py', 'src/observe/execution.py',
                  'tests/unit/test_dividend_entitlements.py',
                  'tests/integration/test_dividend_entitlement_execution.py',
                  'scripts/verify_strategy_batch66.py')
TESTS = ('tests/unit/test_dividend_entitlements.py',
         'tests/integration/test_dividend_entitlement_execution.py',
         'tests/unit/test_ledger.py', 'tests/unit/test_conversion.py',
         'tests/unit/test_portfolio_loop.py', 'tests/unit/test_offline_integration.py',
         'tests/integration/test_conversion_execution.py',
         'tests/integration/test_public_run.py', 'tests/integration/test_strategy_scope.py')
LEGACY_RUN = '20261005-191525-e4a754-strategy-d910'


def legacy_comparison(directory):
    name = '_batch66_frozen_book'
    spec = importlib.util.spec_from_file_location(name, directory / 'book.py.before',
                                                 loader=importlib.machinery.SourceFileLoader(name, str(directory / 'book.py.before')))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    cases = []
    for year, conversion in ((2013, False), (2024, False), (2024, True)):
        days = list(pd.bdate_range(f'{year}-01-02', periods=15).date)
        rules = RuleSet([Rule(start=date(2010, 1, 1))])
        outputs = []
        for cls in (module.Book, Book):
            book = cls(10000, days)
            for k, day in enumerate(days):
                action = {'instrument': 'A', 'ex_date': days[1], 'cash_per_share': 1., 'bonus_ratio': 1.}
                if year == 2024: action.update(pay_date=days[5], bonus_list_date=days[3])
                if conversion: action = {'instrument': 'A', 'ex_date': days[1], 'convert_to': 'A',
                                         'convert_ratio': .49977589, 'convert_rounding': 'ceil'}
                price = 10 if k == 0 else 20 if conversion else 4.5
                quote = {'open': price, 'close': price, 'preclose': price}
                book.start_day(day, [action] if k == 1 else [], {'A': quote})
                if k == 0: book.execute({'instrument': 'A', 'side': 'buy', 'amount': 1000}, quote, day, rules)
                if k == 4: book.execute({'instrument': 'A', 'side': 'sell', 'qty': 'all'}, quote, day, rules)
                book.close_day(day, {'A': quote})
            outputs.append({'cash': book.cash, 'equity': book.equity_rows, 'cash_events': book.cash_events,
                            'fills': book.fills, 'assumptions': book.assumptions, 'issues': book.issues,
                            'positions': {i: asdict(p) for i, p in book.positions.items()}})
        if outputs[0] != outputs[1]: raise AssertionError('Legacy book changed')
        cases.append({'year': year, 'conversion': conversion, 'sessions': len(days), 'differences': 0})
    del sys.modules[name]
    return {'status': 'match', 'cases': cases, 'scope': 'synthetic legacy behavior; not platform equivalence'}


def verify(phase, directory=DIRECTORY):
    if (directory / f'{phase}.json').exists():
        raise FileExistsError(directory / f'{phase}.json')
    if phase == 'before':
        for name in IMPLEMENTATION[:2]:
            with (directory / (Path(name).name + '.before')).open('xb') as out:
                with Path(name).open('rb') as source:
                    shutil.copyfileobj(source, out)
    selected = TESTS[:1] if phase == 'before' else TESTS
    command = [sys.executable, '-m', 'pytest', '-q', *selected]
    process = subprocess.run(command, capture_output=True, timeout=600)
    log = directory / f'{phase}-pytest.log'
    with log.open('xb') as stream:
        stream.write(process.stdout + process.stderr)
    result = {'phase': phase, 'command': command, 'returncode': process.returncode,
              'log': str(log), 'log_sha256': file_sha(log),
              'implementation_sha256': {n: file_sha(n) for n in IMPLEMENTATION if Path(n).is_file()},
              'not_a_backtest': True, 'new_trading_backtests': 0}
    if phase == 'after':
        result['legacy_comparison'] = legacy_comparison(directory)
        result['protection'] = protect('data', directory)
    save(directory / f'{phase}.json', result)
    print(process.stdout.decode(errors='replace'), end='')
    print(json.dumps(result, ensure_ascii=False))
    if phase == 'after' and process.returncode:
        raise SystemExit(process.returncode)
    return result


def reproduce_legacy(directory=DIRECTORY):
    target = directory / 'legacy-strategy-reproduction.json'
    if target.exists(): raise FileExistsError(target)
    frozen = {n: file_sha(n) for n in IMPLEMENTATION}
    with patch('socket.create_connection', side_effect=RuntimeError('network forbidden')), \
            patch('socket.socket.connect', side_effect=RuntimeError('network forbidden')):
        result = reproduce_strategy('data', LEGACY_RUN)
    if frozen != {n: file_sha(n) for n in IMPLEMENTATION}:
        raise AssertionError('Implementation changed during reproduction')
    proof = {'source_run': LEGACY_RUN, 'network': 'forbidden', 'result': result,
             'implementation_sha256': frozen, 'new_strategy_sources': 0,
             'scope': 'existing 2005-2026 stock strategy, three-cost compatibility regression'}
    save(target, proof); print(json.dumps(proof, ensure_ascii=False))
    if result.get('reproduction', {}).get('result') != 'match':
        raise AssertionError('Legacy strategy reproduction did not match')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('before', 'after', 'reproduce'))
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.phase == 'reproduce': reproduce_legacy(args.directory)
    else: verify(args.phase, args.directory)
