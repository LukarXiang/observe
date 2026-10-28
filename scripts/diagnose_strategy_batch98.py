"""Additional original RSRS boundary: exactly-one r2 raised to NaN remains one."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from observe.runs import file_sha
from scripts import research_strategy_batch98 as stage
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch98/20261011-RSRS-perfect-fit-boundary')


def cases(namespace=None):
    rows = []
    for multiplier, seeds in ((2., [.8, 1., 1.2]), (1., [2., 3., 4.])):
        ns = namespace() if namespace else stage.source_namespace(1)
        g = ns['g']; g.N = 18; g.M = 600; g.quantile_N = g.quantile_M = 15
        g.init = False; g.ans = list(seeds); g.ans_rightdev = []; g.sell = -.4
        x = np.arange(1., 19.)
        operand = pd.DataFrame({'low': x, 'high': x*multiplier, 'close': x+2.})
        ns['attribute_history'] = lambda *a: operand.copy()
        with np.errstate(all='ignore'):
            quantile = ns['cal_ret_quantile'](operand.close.pct_change().dropna())
            score = ns['calc_zscore_Rdev']('synthetic')
        values = [*seeds, multiplier**2]; mean = math.fsum(values)/4
        sigma = math.sqrt(math.fsum((v-mean)**2 for v in values)/4)
        independent = (values[-1]-mean)/sigma
        require(math.isnan(quantile) and g.ans_rightdev[-1] == 1. and math.isfinite(score)
                and abs(score-independent) < 1e-14, 'Perfect-fit boundary differs')
        rows.append({'high_multiplier': multiplier, 'return_rows': 17, 'quantile': {'nonfinite': 'nan'},
            'r2': float(g.ans_rightdev[-1]), 'score': float(score), 'independent_score': independent,
            'exclude_below_minus_point4': bool(score < g.sell), 'shared_pool_length': len(g.ans)})
    return {'cases': rows, 'case_count': 2, 'all_synthetic': True, 'original_rule_changed': False,
        'correction': '17 returns always give NaN quantile; score is NaN for ordinary r2!=1, but exact r2==1 yields a finite score',
        'original_strategy_complete': False, 'new_trading_backtests': 0}


def run():
    require(not DIRECTORY.exists(), 'Immutable boundary directory exists'); DIRECTORY.mkdir(parents=True)
    for mode in ('study', 'replay'):
        command = [sys.executable, '-B', '-m', 'scripts.diagnose_strategy_batch98', 'worker', '--mode', mode]
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (DIRECTORY/(mode+'.log')).write_text(process.stdout, encoding='utf-8')
        (DIRECTORY/(mode+'.stderr')).write_text(process.stderr, encoding='utf-8')
        save(DIRECTORY/(mode+'-command.json'), {'command': command, 'returncode': process.returncode,
            'stdout_sha256': file_sha(DIRECTORY/(mode+'.log')), 'stderr_sha256': file_sha(DIRECTORY/(mode+'.stderr'))})
        require(process.returncode == 0, 'Boundary worker failed; actual logs retained')
    require(file_sha(DIRECTORY/'study.json') == file_sha(DIRECTORY/'replay.json'), 'Boundary replay differs')
    result = {'status': 'match', 'differences': 0, 'cases': 2, 'fresh_process': True, 'network_blocked': True,
        'output_sha256': file_sha(DIRECTORY/'study.json'),
        'implementation_sha256': file_sha('scripts/diagnose_strategy_batch98.py'),
        'source_sha256': stage.SOURCE_SHA[1], 'new_trading_backtests': 0}
    save(DIRECTORY/'offline.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['run', 'worker'])
    parser.add_argument('--mode', choices=['study', 'replay']); args = parser.parse_args()
    if args.action == 'run': result = run()
    else:
        require(read(stage.DIRECTORY/'prepared.json')['source_inventory'][1]['sha256'] == stage.SOURCE_SHA[1], 'Source binding differs')
        with network_blocked(): result = cases()
        save(DIRECTORY/(args.mode+'.json'), result)
    print(json.dumps(result, ensure_ascii=False))
