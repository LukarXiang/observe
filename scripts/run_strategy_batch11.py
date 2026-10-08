"""Frozen valuation extension, independent arithmetic and offline reproduction."""
import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
import yaml

from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.runs import file_sha, write_json
from observe.strategies import run_strategy
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import read, protect, fee_checks
from scripts.verify_strategy_batch3 import require, tree_hash

SNAPSHOT = '20261005-152153-eb05'


def run(root, directory, component, short = False):
    output = directory / f'{component}-{"short" if short else "long"}-run.json'
    parameters = directory / f'{component}-{"short" if short else "long"}-parameters.json'
    if output.exists() or parameters.exists(): raise FileExistsError(output if output.exists() else parameters)
    config = yaml.safe_load(Path(f'configs/strategies/{component}_component_v1.yaml').read_text(encoding = 'utf-8'))
    config.update(snapshot = SNAPSHOT, cache = False)
    if not short: config.update(start = '2021-02-02', end = '2026-09-29')
    write_json(parameters, config)
    started = time.monotonic()
    result = run_strategy(root, **config)
    usage = {}
    if sys.platform == 'linux':
        import resource
        usage['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(output, {**result, 'elapsed_seconds': time.monotonic() - started, **usage})
    archive(root, f'第十一批{component.upper()} {"短窗口优化前基线" if short else "长区间"}实验落盘：{result["status"]}')
    return result


def arithmetic(root, path):
    cfg = read(path / 'config.json')['config']; field = 'pb_mrq' if cfg['implementation'].startswith('bp_') else 'pe_ttm'
    store = Store(root); state = store.state(cfg['snapshot'])
    bars = store.load_state(state, 'bars_1d', columns = ['date', 'instrument', field, 'is_trading'])
    bars['date'] = pd.to_datetime(bars.date).dt.date
    factors = pd.read_parquet(path / 'factors.parquet')
    joined = factors.merge(bars, on = ['date', 'instrument'], validate = 'one_to_one')
    require(len(joined) == len(factors), 'Factor input coverage differs')
    inputs = pd.to_numeric(joined[field], errors = 'raise').where(joined.is_trading & np.isfinite(joined[field]))
    expected = (1 / inputs).where(inputs.ne(0)); expected = expected.where(np.isfinite(expected))
    np.testing.assert_allclose(joined.input_value.to_numpy(float), inputs.to_numpy(float), rtol = 0, atol = 0, equal_nan = True)
    np.testing.assert_allclose(joined.value.to_numpy(float), expected.to_numpy(float), rtol = 0, atol = 0, equal_nan = True)
    targets = pd.read_parquet(path / 'targets.parquet'); previous = {}; checks = 0
    for day, rows in targets.groupby('decision_date', sort = True):
        selected = rows[rows.instrument.ne('CASH')].set_index('instrument').weight.to_dict()
        if rows.rebalance.all():
            valid = factors[factors.date.eq(day) & factors.value.notna()]
            if cfg['parameters']['positive_only']: valid = valid[valid.value.gt(0)]
            ranked = valid.sort_values(['value', 'instrument'], ascending = [False, True])
            count = math.floor(len(ranked) * cfg['parameters']['top_fraction'])
            previous = {i: min(1 / count, cfg['portfolio']['max_weight']) for i in ranked.instrument.iloc[:count]} if count else {}
            checks += 1
        require(selected == previous, f'{day}: rank/held targets differ')
        cash = 1 - sum(previous[i] for i in previous)
        actual_cash = rows.loc[rows.instrument.eq('CASH'), 'weight'].item()
        require(math.isclose(actual_cash, cash, abs_tol = 1e-12), f'{day}: cash differs')
    return {'factor_rows_checked': len(factors), 'sessions_checked': targets.decision_date.nunique(), 'rebalance_days_checked': checks,
            'method': 'Raw valuation join, reciprocal and sorted cross-section; between-rebalance targets held'}


def verify(root, directory, component, short = False):
    period = 'short' if short else 'long'; output = directory / f'{component}-{period}-verification.json'
    if output.exists(): raise FileExistsError(output)
    original = read(directory / f'{component}-{period}-run.json'); path = Path(original['output'])
    require(original['status'] == 'success_limited', 'Original run did not finish')
    require(verify_run(root, path)['status'] == 'ok', 'Original integrity failed')
    check = arithmetic(root, path); before = tree_hash(path)
    def forbidden(*args, **kwargs): raise AssertionError('Network disabled for batch11 reproduction')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
        again = reproduce(root, original['run_id'])
    require(again['reproduction']['result'] == 'match' and again['reproduction']['differences'] == 0, 'Reproduction mismatch')
    require(tree_hash(path) == before, 'Original experiment changed')
    require(verify_run(root, again['run_id'])['status'] == 'ok', 'Reproduction integrity failed')
    result = {'status': 'ok', 'run': original, 'reproduction': again, 'hand_check': check, 'original_unchanged': True,
              'source_sha256': file_sha(path / 'source.original'), 'report': read(path / 'report.json')}
    write_json(output, result)
    proof = Path(root) / 'catalog/strategies/implementation-evidence.json'; evidence = read(proof); doc = read(path / 'config.json')
    evidence.append({'strategy_id': doc['source']['strategy_id'], 'source_sha256': result['source_sha256'],
                     'implementation': doc['config']['implementation'], 'scope': doc['source']['review']['scope'], 'run_id': original['run_id'],
                     'snapshot': doc['snapshot_id'], 'reproduction_run_id': again['run_id'], 'reproduction_result': 'match',
                     'differences': 0, 'original_strategy_complete': False, 'validation_file': str(output), 'validation_sha256': file_sha(output)})
    write_json(proof, evidence)
    archive(root, f'第十一批{component.upper()} {period}禁网验收：match / 0差异')
    return {'status': 'ok', 'run_id': original['run_id'], 'reproduction': again['reproduction'], 'hand_check': check}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('action', choices = ['run', 'verify', 'protect', 'fees'])
    parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', type = Path, default = Path('data/staging/strategies-batch11/20261005-daily-stock-reviews'))
    parser.add_argument('--component', choices = ['bp', 'ep']); parser.add_argument('--short', action = 'store_true')
    args = parser.parse_args()
    if args.action in ('run', 'verify'):
        if args.component is None: parser.error('--component is required')
        result = (run if args.action == 'run' else verify)(args.root, args.directory, args.component, args.short)
    elif args.action == 'protect': result = protect(args.root, args.directory)
    else: result = fee_checks(args.directory)
    print(json.dumps(result, ensure_ascii = False), flush = True)
