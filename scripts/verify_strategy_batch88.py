"""Independent original-definition replay and scalar checks for Batch88 research."""
import argparse
from collections import Counter
import copy
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source
from scripts import review_strategy_batch88 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch88/20261009-daily-chan-industry-acceptance')
STAGE_RECEIPT = Path('docs/handoff/2026-10-09-batch88-stage-verification.json')
FILES = tuple(sorted(set(stage.previous.FILES) | {'scripts/review_strategy_batch88.py',
    'tests/unit/test_batch88_research.py', 'scripts/verify_strategy_batch88.py', 'tests/unit/test_batch88_verification.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'independent-replay.json', 'independent-daily-values.parquet', 'momentum-diagnostics.json',
    'source-reviews/review.json', 'offline-catalog.json', 'checked-state.json'}


def bind(directory=None):
    receipt = read(STAGE_RECEIPT)
    require(receipt['status'] == 'stage_checks_passed' and not receipt['formal_acceptance'], 'Stage contract changed')
    require(all(file_sha(n) == sha for n, sha in receipt['evidence_sha256'].items()), 'Stage evidence changed')
    require(all(file_sha(n) == sha for n, sha in receipt['implementation_sha256'].items()), 'Stage entrypoint changed')
    stage.validate(stage.DIRECTORY)
    result = {'snapshot': stage.SNAPSHOT, 'stage_receipt': str(STAGE_RECEIPT),
        'stage_receipt_sha256': file_sha(STAGE_RECEIPT), 'stage_input_manifest_sha256': file_sha(stage.DIRECTORY/'input-manifest.json'),
        'stage_state_sha256': file_sha(stage.DIRECTORY/'chan-daily-state-values.parquet'), 'not_a_backtest': True}
    if directory is not None:
        require(result == read(directory/'input-binding.json'), 'Bound input changed')
    return result


def preflight(root, directory):
    binding = bind()
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checked = read(stage.previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted87 code changed')
    checkpoint(root, directory, 'Batch88 final verification starts from immutable stage; accepted87 retained')
    save(directory/'input-binding.json', binding)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    return {'status': 'ok', 'bound_stage_sha256': binding['stage_state_sha256']}


def serial(array):
    if array is None:
        return None
    columns = array.dtype.names
    output = []
    for row in array:
        values = []
        for name in columns:
            value = row[name]
            if np.issubdtype(array.dtype[name], np.datetime64):
                value = str(value)
            elif np.issubdtype(array.dtype[name], np.floating):
                value = float(value).hex()
            elif isinstance(value, np.generic):
                value = value.item()
            values.append(value)
        output.append(values)
    return {'dtype': [(name, array.dtype[name].str) for name in columns], 'rows': output}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def original_namespace():
    row = read(stage.DIRECTORY/'sources-frozen.json')[0]
    require(file_sha(row['source_copy']) == stage.SOURCE_SHA[0], 'Original class SHA changed')
    text = Path(row['source_copy']).read_text(encoding='utf-8-sig')
    require(text.count('\n# In[3]:') == 1, 'Original chart boundary changed')
    code = text.split('\n# In[3]:')[0]
    namespace = {}
    with warnings.catch_warnings(record=True):
        # Original pre-chart source, including its imports/defaults; no project AST loader.
        exec(compile(code, '<batch88-independent-original-prefix>', 'exec'), namespace)
    require('KBarChan' in namespace and 'draw_chan' not in namespace and 'stock_df' not in namespace,
            'Top-level chart/fetch executed')
    return namespace


def close(a, b):
    return abs(float(a)-float(b)) <= 1e-8 + 1e-5*abs(float(b))


def less(a, b):
    return a < b and not close(a, b)


def more(a, b):
    return a > b and not close(a, b)


def inclusion(first, second):
    high1 = first['nh'] or first['high']; high2 = second['nh'] or second['high']
    low1 = first['nl'] or first['low']; low2 = second['nl'] or second['low']
    if (high1 < high2 or close(high1, high2)) and (low1 > low2 or close(low1, low2)):
        return 2
    if (high1 > high2 or close(high1, high2)) and (low1 < low2 or close(low1, low2)):
        return 3
    return 0


def standard_rows(array):
    rows = [{'date': r['date'], 'close': float(r['close']), 'high': float(r['high']), 'low': float(r['low']),
             'real_loc': i, 'nh': 0., 'nl': 0., 'trend': 0} for i, r in enumerate(array)]
    while len(rows) > 2:
        if inclusion(rows[0], rows[1]):
            rows.pop(0)
        else:
            rows[0]['nh'] = rows[0]['high']; rows[0]['nl'] = rows[0]['low']
            break
    past, first, second = 0, 1, 2
    while second < len(rows):
        a, b = rows[first], rows[second]; kind = inclusion(a, b)
        if kind:
            trend = a['trend'] or (.5 if less(rows[past]['nh'] or rows[past]['high'], a['nh'] or a['high']) else -.5)
            choose = max if close(trend, .5) else min
            target, removed = (b, a) if kind == 2 else (a, b)
            target['nh'] = choose(a['nh'] or a['high'], b['nh'] or b['high'])
            target['nl'] = choose(a['nl'] or a['low'], b['nl'] or b['low'])
            # The original structured field is int: half-unit trends truncate to zero.
            target['trend'] = int(trend)
            removed['nh'] = removed['nl'] = 0.
            if kind == 2:
                first = second
            second += 1
        else:
            for row in (a, b):
                row['nh'] = row['nh'] or row['high']; row['nl'] = row['nl'] or row['low']
            past, first, second = first, second, second+1
    return [{'date': row['date'], 'close': row['close'], 'high': row['nh'], 'low': row['nl'],
             'real_loc': row['real_loc'], 'new_index': i}
            for i, row in enumerate(r for r in rows if r['nh'] != 0)]


def fractals(rows):
    values = [0]*len(rows)
    if len(rows) < 7:
        return values
    last = 0
    for i in range(1, len(rows)-1):
        a, b, c = rows[i-1:i+2]
        if more(b['high'], a['high']) and more(b['high'], c['high']):
            values[i] = 1
        elif less(b['low'], a['low']) and less(b['low'], c['low']):
            values[i] = -1
        if values[i]:
            last = i
    found = next((v for v in values[1:] if v), 0)
    values[0] = -found
    values[-1] = -values[last]
    return values


def independent_prices():
    manifest = read(stage.DIRECTORY/'input-manifest.json')
    data = {name: pd.read_parquet(row['file']) for name, row in manifest.items()}
    bars = data['bars_1d'].sort_values('date').reset_index(drop=True)
    coverage = data['adj_coverage']
    require(len(coverage) == 1 and coverage.status.iloc[0] == 'complete' and
            not coverage.has_gap.iloc[0] and coverage.has_start_basis.iloc[0], 'Adjustment contract changed')
    require(bars.is_trading.all() and bars.date.is_unique and
            bars.date.min() >= coverage.verified_from.iloc[0] and bars.date.max() <= coverage.verified_through.iloc[0],
            'Native daily input coverage changed')
    events = data['adj_factors'].sort_values('ex_date')
    factor = []
    for day in bars.date:
        known = events[events.ex_date <= day]
        factor.append(float(known.back_factor.iloc[-1]) if len(known) else 1.)
    require(np.isfinite(factor).all() and min(factor) > 0, 'Invalid factors')
    bars['independent_factor'] = factor
    return bars


def independent_window(bars):
    array = np.empty(len(bars), dtype=[('date', 'datetime64[ns]'), ('open', 'f8'),
        ('high', 'f8'), ('low', 'f8'), ('close', 'f8')])
    last_factor = float(bars.independent_factor.iloc[-1])
    for i, row in enumerate(bars.itertuples()):
        array['date'][i] = np.datetime64(str(row.date), 'ns')
        for name in ('open', 'high', 'low', 'close'):
            array[name][i] = float(getattr(row, name))*row.independent_factor/last_factor
    return array


def instrument(namespace, counts):
    cls = namespace['KBarChan']; original = cls.check_XD_topbot
    def check(obj, a, b, c, d, e, f):
        result, gap = original(obj, a, b, c, d, e, f)
        expected = 0; expected_gap = False
        if c['tb'] == 1:
            expected_gap = less(a['chan_price'], d['chan_price'])
            if more(c['chan_price'], a['chan_price']) and more(c['chan_price'], e['chan_price']) and more(d['chan_price'], f['chan_price']):
                expected = 1
        elif c['tb'] == -1:
            expected_gap = more(a['chan_price'], d['chan_price'])
            if less(c['chan_price'], a['chan_price']) and less(c['chan_price'], e['chan_price']) and less(d['chan_price'], f['chan_price']):
                expected = -1
        require(result.value == expected and gap == expected_gap, 'XD six-endpoint scalar predicate differs')
        counts['XD_six_endpoint_predicate_calls'] += 1
        return result, gap
    cls.check_XD_topbot = check
    return cls


def structures(obj, array, counts):
    expected = standard_rows(array); actual = obj.kDataFrame_standardized
    require(len(actual) == len(expected), 'Independent standardization length differs')
    types = fractals(expected)
    for row, exp, tb in zip(actual, expected, types, strict=True):
        require(all(row[name] == value for name, value in exp.items()) and row['tb'] == tb,
                'Independent standardization/fractal fields differ')
    counts['independent_standardized_rows'] += len(expected)
    origin = obj.kDataFrame_origin
    require(len(origin) == len(array), 'Original rows changed')
    for i, row in enumerate(origin):
        hi = float(array[i-1]['high']) if i else 0.; lo = float(array[i-1]['low']) if i else 0.
        expected_gap = 1 if more(row['low']-hi, .01) else -1 if less(row['high']-lo, -.01) else 0
        require(row['gap'] == expected_gap, 'Independent raw gap flag differs')
    counts['independent_raw_gap_rows'] += len(origin)
    lookup = {int(r['real_loc']): r for r in actual}
    marked = obj.kDataFrame_marked
    for row in marked:
        source = lookup[int(row['real_loc'])]
        require(row['tb'] in (-1, 1) and all(row[k] == source[k] for k in ('date', 'close', 'high', 'low')) and
                row['chan_price'] == row['high' if row['tb'] == 1 else 'low'], 'BI endpoint mapping differs')
    counts['BI_endpoint_mapping_rows'] += len(marked)
    xd = obj.kDataFrame_xd
    if xd is not None:
        bi_map = {int(r['real_loc']): r for r in marked}
        for row in xd:
            source = bi_map[int(row['real_loc'])]
            require(row['xd_tb'] in (-1, 1) and
                    all(row[k] == source[k] for k in ('date', 'close', 'high', 'low', 'chan_price')), 'XD endpoint mapping differs')
        counts['XD_endpoint_mapping_rows'] += len(xd)


def momentum_diagnostics():
    row = read(stage.DIRECTORY/'sources-frozen.json')[1]
    import ast
    tree = ast.parse(read_source(Path(row['source_copy']))[0])
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = {'pd': pd}; exec(compile(ast.Module(body=funcs, type_ignores=[]), '<batch88-independent-notebook>', 'exec'), ns)
    days = pd.bdate_range('2020-04-01', periods=30).strftime('%Y-%m-%d').tolist()
    rates = [-.02, 0., .005, .03, .06, .1, .15, .2, .3]
    data = {f'S{i}': [[d, float(10*(1+rate)**j), 100.] for j, d in enumerate(days)] for i, rate in enumerate(rates)}
    industries = {f'S{i}': f'I{i//3}' for i in range(9)}
    ns.update(today=days[-1], last_trade_date=days[-1], industry_level='sw_l3',
        security_industry_dict=industries, security_code_name_dict={c: c for c in data},
        industry_code_security_count_dict={f'I{i}': 3 for i in range(3)},
        industry_code_name_dict={f'I{i}': f'Industry{i}III' for i in range(3)})
    values = []
    for horizon, limit in ((1, .05), (3, .08), (5, .1), (10, .15), (20, .2)):
        got, leaders = ns['get_result'](horizon, copy.deepcopy(data))
        columns = []; expected_rows = []
        for endpoint in range(20, 30):
            counts = Counter()
            for code, points in data.items():
                growth = (points[endpoint][1]-points[endpoint-horizon][1])/points[endpoint-horizon][1]
                if growth >= limit:
                    counts[industries[code]] += 1
            expected_rows.append({key: float(format(counts[key]**2/3, '.2f')) for key in ('I0', 'I1', 'I2')})
            columns.append(days[endpoint][5:])
        order = sorted(('I0', 'I1', 'I2'), key=lambda key: expected_rows[-1][key], reverse=True)
        expected = pd.DataFrame([[r[key] for r in expected_rows] for key in order],
            columns=columns, index=[f'Industry{key[1:]}({key})' for key in order])
        pd.testing.assert_frame_equal(got, expected, check_exact=True)
        values.append({'case': f'cross-industry-horizon-{horizon}', 'values': got.to_dict(), 'leader_industries': list(leaders.index)})
    mismatch = copy.deepcopy(data)
    shifted = pd.bdate_range('2020-04-02', periods=30).strftime('%Y-%m-%d').tolist()
    for point, date in zip(mismatch['S8'], shifted, strict=True):
        point[0] = date
    ns['last_trade_date'] = shifted[-1]
    got, _ = ns['get_result'](1, mismatch)
    require(list(got.columns) == [d[5:] for d in shifted[-10:]], 'Last-security date labels changed')
    values.append({'case': 'unsynchronized-dates-use-last-security', 'last_security_dates': shifted[-10:],
                   'other_security_dates': days[-10:], 'result_columns': list(got.columns), 'synthetic': True})
    ns['last_trade_date'] = '1999-12-31'
    try:
        ns['get_result'](1, data)
    except KeyError as exc:
        values.append({'case': 'last-trade-date-column-missing', 'error': f'KeyError: {exc}'})
    else:
        raise ValueError('Missing global date defect disappeared')
    return {'cases': values, 'count': len(values), 'independent_cross_industry_cells': 150,
            'not_a_backtest': True, 'historical_original_industry_rankings': 0}


def audit(root, directory):
    bind(directory)
    original_states = pd.read_parquet(stage.DIRECTORY/'chan-daily-state-values.parquet')
    bars = independent_prices(); counts = Counter(); ns = original_namespace(); cls = instrument(ns, counts)
    require(len(bars) == len(original_states) == 1392, 'Evaluation coverage changed')
    output = []; prior = {}; observed = Counter()
    def forbidden(*a, **kw):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        diagnostics = momentum_diagnostics()
        for end, saved in enumerate(original_states.itertuples(), 1):
            window = bars.iloc[max(0, end-1000):end]; array = independent_window(window)
            obj = cls(array); stream = StringIO()
            with redirect_stdout(stream):
                obj.getFenDuan()
            state = {name: serial(getattr(obj, name)) for name in
                ('kDataFrame_origin', 'kDataFrame_standardized', 'kDataFrame_marked', 'kDataFrame_xd')}
            state.update(gap_XD=obj.gap_XD, previous_skipped_idx=obj.previous_skipped_idx,
                previous_with_xd_gap=obj.previous_with_xd_gap, stdout=stream.getvalue(), error=None)
            encoded = json.dumps(state, sort_keys=True, ensure_ascii=True)
            require(saved.state_json == encoded and saved.state_sha256 == digest(state) and
                saved.evaluation_date == str(window.date.iloc[-1]) and saved.window_start == str(window.date.iloc[0]) and
                saved.window_rows == len(window) and saved.error is None, 'Independent original-definition full state differs')
            structures(obj, array, counts)
            for field, name, column in (('kDataFrame_marked', 'bi', 'tb'), ('kDataFrame_xd', 'xd', 'xd_tb')):
                arr = getattr(obj, field)
                points = {(str(r['date'])[:10], int(r[column])) for r in arr} if arr is not None else set()
                removed = {p for p in prior.get(name, set())-points if p[0] >= saved.window_start}
                if removed:
                    observed[name+'_revision_dates'] += 1; observed[name+'_removed_endpoint_pairs'] += len(removed)
                prior[name] = points
            output.append({'evaluation_date': saved.evaluation_date, 'state_sha256': digest(state),
                'window_rows': len(window), 'bi_rows': len(obj.kDataFrame_marked),
                'xd_rows': len(obj.kDataFrame_xd) if obj.kDataFrame_xd is not None else 0})
            if end % 100 == 0:
                print(json.dumps({'independently_replayed': end, 'total': len(bars)}), flush=True)
    require(dict(observed) == {'bi_revision_dates': 439, 'bi_removed_endpoint_pairs': 478,
        'xd_revision_dates': 188, 'xd_removed_endpoint_pairs': 197}, 'Repainting observation differs')
    path = directory/'independent-daily-values.parquet'; require(not path.exists(), 'Independent replay exists')
    pd.DataFrame(output).to_parquet(path, index=False)
    save(directory/'momentum-diagnostics.json', diagnostics)
    result = {'result': 'match', 'differences': 0, 'evaluation_dates': len(output), 'all_states_exact': True,
        'socket_HTTP_disabled': True, 'independent_scalar_checks': dict(counts), 'repainting': dict(observed),
        'momentum_cells': diagnostics['independent_cross_industry_cells'], 'new_diagnostic_cases': diagnostics['count'],
        'state_values_file': str(path), 'state_values_sha256': file_sha(path), 'input_binding': bind(directory),
        'not_a_backtest': True, 'new_trading_backtests': 0, 'platform_equivalence_proven': False,
        'limits': ['BI/XD full output uses independent loading/execution of unchanged original code, not a second independently derived engine',
            'Independent scalar derivation covers default inclusion, fractals, raw gaps, endpoint mapping and six-point XD predicate only',
            'Original int trend truncation, relative tolerance, endpoint assumptions and gap-union defects retained',
            'Historical industry membership/pool and original platform date/paused/price/plot semantics not proved']}
    save(directory/'independent-replay.json', result)
    return result


def catalog(root, directory):
    bind(directory); require(read(directory/'independent-replay.json')['result'] == 'match', 'Independent replay absent')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'; old = {r['path']: r for r in read(prior)}
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False); rows = []
    for source, sha, frozen in zip(stage.SOURCES, stage.SOURCE_SHA, read(stage.DIRECTORY/'sources-frozen.json'), strict=True):
        require(old[source]['review_status'] != '人工审查完成' and old[source]['bytes_sha256'] == sha,
                'Source already accepted or changed')
        rows.append({**frozen, 'review': REVIEWS[source], 'newly_reviewed': True})
    current = catalog_strategies(root, 'repo/量化策略源代码')
    now = read(Path(current['output'])/'catalog.json')
    require(len(old) == len(now) == 695 and {r['path'] for r in now if r != old[r['path']]} == set(stage.SOURCES),
            'Unrelated source catalog changed')
    save(folder/'review.json', {'snapshot': stage.SNAPSHOT, 'sources': rows,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'catalog': current, 'not_a_backtest': True})
    return archive(root, 'Batch88 daily Chan and industry notebook source research reviewed; final checks pending; no trades')


def validate_sources(directory):
    bind(directory); review = read(directory/'source-reviews/review.json')
    require(file_sha(review['previous_catalog_file']) == review['previous_catalog_sha256'], 'Prior catalog changed')
    for source, row in zip(stage.SOURCES, review['sources'], strict=True):
        require(row['review'] == REVIEWS[source], 'Reviewed rules changed')


def offline(root, directory):
    validate_sources(directory)
    def forbidden(*a, **kw):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        result = offline_catalog(root, directory)
    save(directory/'offline-catalog.json', result); validate_catalog(root, directory)
    return result


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
        'scripts/review_strategy_batch88.py', 'scripts/verify_strategy_batch88.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch88_research.py',
            'tests/unit/test_batch88_verification.py', 'tests/unit/test_batch87_research.py',
            'tests/unit/test_research_metadata.py', 'tests/unit/test_catalog_dependencies.py',
            'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    require(CORE-{'checked-state.json'} <= {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()},
            'Core evidence missing before checks')
    before = {n: file_sha(n) for n in FILES}; logs = []; folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for i, command in enumerate(commands()):
        log = folder/f'{i}.log'
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        logs.append({'command': command, 'returncode': result.returncode, 'log': str(log), 'sha256': file_sha(log)})
        require(result.returncode == 0, f'Validation failed: {log}')
    require(before == {n: file_sha(n) for n in FILES}, 'Implementation changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': logs, 'implementation_sha256': before,
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': logs}


def finish(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); replay = read(directory/'independent-replay.json')
    require(checked['status'] == 'passed' and CORE-{'checked-state.json'} <= checked['evidence_sha256'].keys() and
            {n: file_sha(n) for n in FILES} == checked['implementation_sha256'] and
            [row['command'] for row in checked['commands']] == commands(), 'Check scope changed')
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    require(all(row['returncode'] == 0 and file_sha(row['log']) == row['sha256'] for row in checked['commands']),
            'Check command failed or log changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
            'Checked code copy changed')
    require(replay['result'] == 'match' and replay['differences'] == 0 and replay['all_states_exact'] and
            replay['evaluation_dates'] == 1392 and replay['socket_HTTP_disabled'] and replay['momentum_cells'] == 150,
            'Replay scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed during research')
    protection = protect(root, directory)
    progress = archive(root, 'Batch88 two non-trading sources accepted;1392 exact independent-definition/scalar checks; no trades')
    receipt = Path('docs/handoff/2026-10-09-batch88-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': bind(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked, 'independent_replay': replay,
        'momentum_diagnostics': read(directory/'momentum-diagnostics.json'), 'protection': protection,
        'progress': progress, 'newly_reviewed': 2, 'new_trading_backtests': 0,
        'not_a_backtest': True, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'audit', 'catalog', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
