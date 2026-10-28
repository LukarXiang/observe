"""Stage original daily Chan and industry notebook research without claiming trades."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
from enum import Enum
import hashlib
import importlib.util
import inspect
from io import StringIO
import json
from pathlib import Path
import shutil
import socket

import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import requests
from scipy.ndimage import shift
import talib
from unittest.mock import patch

from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source, strategy_id
from scripts import review_strategy_batch87 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch83 import selected_nodes
from scripts.run_strategy_batch4 import checkpoint, protect as protect_previous, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch88/20261009-daily-chan-industry-notebook')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2021年度精选策略/16.缠论工具（笔， 线段）.txt',
           '2023年度精选策略/63.A股最强板块动量趋势（最终版）.txt')
SOURCE_SHA = ('9565a07a451750a9de2381900a502a24ba13e100ecfd471848de6c887583a6ed',
              '25c512eef49a1b71cd75c05029c4cc6f26a55dd7c34bddcdecdcda3bc1e0dfad')
CONSTANTS = {'GOLDEN_RATIO', 'MIN_PRICE_UNIT', 'FEN_BI_COLUMNS', 'FEN_DUAN_COLUMNS'}
REFERENCES = ('repo/akshare/akshare/stock/stock_industry_sw.py',
              'repo/akshare/akshare/index/index_sw.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def payload(array):
    if array is None:
        return None
    def cell(value):
        if isinstance(value, np.datetime64):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return float(value).hex()
        return value.item() if isinstance(value, np.generic) else value
    # Multi-field NumPy views retain padding; hash field values, never padding bytes.
    return {'dtype': [(n, array.dtype[n].str) for n in array.dtype.names],
            'rows': [[cell(row[n]) for n in array.dtype.names] for row in array]}


def validate(directory):
    frozen = read(directory/'sources-frozen.json')
    require(len(frozen) == 2, 'Frozen source count changed')
    for row, sha in zip(frozen, SOURCE_SHA, strict=True):
        require(file_sha(row['source_copy']) == file_sha(row['source_path']) == sha, 'Source changed')
    for row in read(directory/'input-manifest.json').values():
        require(file_sha(row['file']) == row['sha256'], 'Input changed')


def namespace(directory, number):
    rows = read(directory/'sources-frozen.json'); row = rows[number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Unreviewed source')
    tree = ast.parse(read_source(Path(row['source_copy']))[0])
    if number == 1:
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        require(names == ['get_top_secutiry', 'get_momentum_result', 'get_momentum_list',
                          'recursive_join', 'get_result'], 'Notebook functions changed')
        return selected_nodes(tree, names, {'pd': pd})
    ns = {'np': np, 'copy': copy, 'talib': talib, 'Enum': Enum,
          'append_fields': append_fields, 'shift': shift}
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'draw_chan':
            nodes.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in CONSTANTS:
                ast.literal_eval(node.value)
                nodes.append(node)
    require({n.name for n in nodes if isinstance(n, ast.ClassDef)} ==
            {'InclusionType', 'TopBotType', 'KBarChan'}, 'Original classes changed')
    # Exact SHA-pinned definitions only: platform imports, fetching and chart writes stay inert.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-batch88-chan>', 'exec'), ns)
    return ns


def preflight(root, directory):
    receipt = Path('docs/handoff/2026-10-09-batch87-verification.json')
    review = read('docs/handoff/2026-10-09-batch87-final-review.json')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(file_sha(receipt) == review['receipt_sha256'] and
            file_sha(previous.DIRECTORY/'checked-state.json') == review['checked_state_sha256'] and
            file_sha(previous.DIRECTORY/'final-binding-verification.json') == review['final_binding_sha256'],
            'Accepted87 binding changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()),
            'Accepted87 evidence changed')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()),
            'Accepted implementation changed')
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['batch_id'] == '20261006-145049-5ceb' and
            state['tables'] == store.published()['tables'], 'Publication changed')
    checkpoint(root, directory, 'Batch88 stage begins; accepted87 retained; no catalog acceptance or trades')
    save(directory/'input-binding.json', {'snapshot': SNAPSHOT, 'receipt_file': str(receipt),
        'receipt_sha256': file_sha(receipt), 'checked_state_sha256': file_sha(previous.DIRECTORY/'checked-state.json'),
        'formal_acceptance': False, 'not_a_backtest': True})
    rows = []; folder = directory/'source-reviews'; folder.mkdir()
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(file_sha(path) == sha, 'Original source changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        tree = ast.parse(text)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'lines': len(text.splitlines()),
            'definitions': previous.definitions(tree),
            'top_level_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
            'formal_acceptance': False, 'has_order_calls': any(isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id.startswith('order') for n in ast.walk(tree))})
    save(directory/'sources-frozen.json', rows)
    refs = []; (directory/'references').mkdir()
    for name in REFERENCES:
        dest = directory/'references'/Path(name).name; shutil.copyfile(name, dest)
        refs.append({'file': name, 'copy': str(dest), 'sha256': file_sha(name)})
    save(directory/'references.json', refs)
    data = {}
    filters = [('instrument', '==', '300058.SZ')]
    for table in ('bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'instruments'):
        frame = store.load_state(state, table, filters=filters)
        path = directory/(table+'-input.parquet'); frame.to_parquet(path, index=False)
        data[table] = {'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
                       'columns': list(frame)}
    save(directory/'input-manifest.json', data)
    daily = pd.read_parquet(data['bars_1d']['file'])
    require(daily.instrument.eq('300058.SZ').all() and not daily.empty, 'Original stock missing')
    save(directory/'dependency-inventory.json', {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'inputs': data, 'daily_dates': [str(daily.date.min()), str(daily.date.max())],
        'installed': {n: importlib.util.find_spec(n) is not None for n in ('jqdata', 'jqdatasdk', 'pyecharts', 'talib')},
        'chan_original': {'instrument': '300058.XSHE', 'count': 1000, 'unit': '1d', 'df': False,
            'dynamic_anchor': 'fq_ref_date=end_time', 'end_time': 'pd.datetime.now(); original execution date unknown'},
        'momentum_original': {'industry': 'sw_l3', 'count': 30, 'frequency': 'daily', 'fq': 'pre',
            'skip_paused': False, 'fields': ['close', 'money'], 'fundamental_or_flow_API': False,
            'trade_callbacks_or_orders': False},
        'gates': ['No registered historical SW3 membership/name/version or all-market historical ST/delist pool',
            'Native get_bars include-current/paused semantics and original notebook execution date unproved',
            'Final extrema/segments revise earlier output; chart endpoints are not historical confirmed signals',
            'No trading/sizing/exit rules in either original source; economic trading extension requires confirmation'],
        'formal_acceptance': False, 'new_trading_backtests': 0})
    return {'status': 'ok', 'frozen_sources': 2, 'daily_rows': len(daily), 'formal_acceptance': False}


def chan_input(directory):
    files = read(directory/'input-manifest.json')
    data = {k: pd.read_parquet(v['file']) for k, v in files.items()}
    bars = with_adjusted(data['bars_1d'], data['adj_factors'], data['adj_coverage'])
    bars = bars.sort_values('date').reset_index(drop=True)
    require(bars.date.is_unique and bars.adjustment_status.eq('usable').all() and
            np.isfinite(bars[['open_adj', 'high_adj', 'low_adj', 'close_adj', 'back_factor']]).all().all(),
            'Daily adjustment unavailable')
    return bars


def window_array(bars):
    result = np.empty(len(bars), dtype=[('date', 'datetime64[ns]'), ('open', 'f8'),
                                      ('high', 'f8'), ('low', 'f8'), ('close', 'f8')])
    result['date'] = pd.to_datetime(bars.date).to_numpy(dtype='datetime64[ns]')
    anchor = bars.back_factor.iloc[-1]
    for name in ('open', 'high', 'low', 'close'):
        result[name] = bars[name+'_adj'].to_numpy()/anchor
    return result


def chan_state(ns, array):
    obj = ns['KBarChan'](array.copy()); stream = StringIO(); error = None
    with redirect_stdout(stream):
        try:
            obj.getFenDuan()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    state = {name: payload(getattr(obj, name)) for name in
             ('kDataFrame_origin', 'kDataFrame_standardized', 'kDataFrame_marked', 'kDataFrame_xd')}
    state.update(gap_XD=obj.gap_XD, previous_skipped_idx=obj.previous_skipped_idx,
                 previous_with_xd_gap=obj.previous_with_xd_gap, stdout=stream.getvalue(), error=error)
    return state


def diagnostics(directory):
    ns = namespace(directory, 1)
    days = pd.bdate_range('2020-04-01', periods=30).strftime('%Y-%m-%d').tolist()
    codes = [f'S{i}' for i in range(8)]
    ns.update(today=days[-1], last_trade_date=days[-1], industry_level='sw_l3',
        security_code_name_dict={c: c for c in codes}, security_industry_dict={c: 'I' for c in codes},
        industry_code_name_dict={'I': 'IndustryIII'}, industry_code_security_count_dict={'I': 8})
    data = {c: [[d, float(10*1.06**i), 100.] for i, d in enumerate(days)] for c in codes}
    result = []
    for horizon in (1, 3, 5, 10, 20):
        ranking, names = ns['get_result'](horizon, copy.deepcopy(data))
        require(ranking.shape == (1, 10) and (ranking.to_numpy() == 8.).all() and
                len(names.iloc[0, 0]) == 7, 'Literal notebook output changed')
        result.append({'case': f'momentum-horizon-{horizon}', 'nature': 'synthetic_original_function',
                       'momentum': 8., 'columns': list(ranking), 'leader_count': 7})
    for label, horizon, sample, kind in (
        ('unsupported-horizon', 2, data, 'NameError'),
        ('insufficient-20day-lookback', 20, {c: v[1:] for c, v in data.items()}, 'IndexError'),
        ('zero-base-close', 1, {'S0': [[d, 0., 100.] for d in days]}, 'ZeroDivisionError')):
        try:
            ns['get_result'](horizon, copy.deepcopy(sample))
        except Exception as exc:
            require(type(exc).__name__ == kind, 'Unexpected original failure')
            result.append({'case': label, 'error': f'{type(exc).__name__}: {exc}'})
        else:
            raise AssertionError('Original failure disappeared')
    cn = namespace(directory, 0)
    obj = cn['KBarChan'](np.empty(0, dtype=[('date', 'datetime64[ns]'), ('close', 'f8'), ('high', 'f8'), ('low', 'f8')]))
    require(obj.combine_gaps([(1., 10.), (2., 3.)]) == [(1., 3.)], 'Original nested gap behavior changed')
    require(not cn['float_more'](100.0005, 100.) and cn['float_more'](100.002, 100.), 'Original tolerance changed')
    result += [{'case': 'nested-gap-union-shrinks', 'input': [[1., 10.], [2., 3.]], 'output': [[1., 3.]]},
               {'case': 'isclose-relative-tolerance', '100.0005-greater-than-100': False}]
    other = copy.deepcopy(data)
    for values in other.values():
        for item in values:
            item[2] = -999999.
    require(ns['get_result'](5, other)[0].equals(ns['get_result'](5, data)[0]), 'Money affects original ranking')
    result.append({'case': 'money-column-unused', 'changed_all_money_values': True, 'ranking_changed': False})
    ns['industry_code_security_count_dict'] = {'I': 9, 'J': 9, 'K': 100000}
    ns['industry_code_name_dict'].update(J='SecondIII', K='ThirdIII')
    ns['security_industry_dict'].update(S6='J', S7='K')
    rounded = ns['get_momentum_result']([(c, .1) for c in ('S0', 'S6', 'S7')], days[-1])
    require(rounded.iloc[:, 0].tolist() == [.11, .11, 0.], 'Rounded tie changed')
    result.append({'case': 'round-before-stable-sort', 'values': rounded.iloc[:, 0].tolist(),
                   'industries': list(rounded.index)})
    tree = ast.parse(read_source(Path(read(directory/'sources-frozen.json')[1]['source_copy']))[0])
    loop = [n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and
            [a.id for a in n.target.elts] == ['security_code', 'industry_info_dict']]
    require(len(loop) == 1, 'Original industry loop changed')
    state = {'get_industry': lambda codes: {'A': {'sw_l3': {'industry_code': 'I'}}},
        'security_code_list': ['A'], 'security_code_name_dict': {'A': 'A'}, 'industry_level': 'sw_l3',
        'security_industry_dict': {}, 'industry_code_security_count_dict': {}, 'industry_code_name_dict': {}}
    try:
        exec(compile(ast.Module(body=loop, type_ignores=[]), '<frozen-batch88-industry-loop>', 'exec'), state)
    except TypeError as exc:
        result.append({'case': 'missing-industry-name-string-list-delete', 'error': f'TypeError: {exc}',
                       'industry_count_mutated_before_failure': state['industry_code_security_count_dict']})
    else:
        raise AssertionError('Original industry-name defect disappeared')
    try:
        pd.set_option('display.max_colwidth', -1)
    except ValueError as exc:
        result.append({'case': 'notebook-max-colwidth', 'error': f'ValueError: {exc}'})
    require(not hasattr(pd, 'datetime'), 'Original pd.datetime compatibility changed')
    result.append({'case': 'original-pd-datetime-missing', 'error': 'AttributeError'})
    return {'cases': result, 'count': len(result), 'not_a_backtest': True, 'formal_acceptance': False}


def probe(directory):
    import akshare as ak
    import datetime as dt
    validate(directory)
    folder = directory/'daily-history-probe'; folder.mkdir(exist_ok=False)
    fn = ak.stock_zh_a_hist; code = inspect.getsource(fn)
    save(folder/'api.json', {'name': fn.__name__, 'signature': str(inspect.signature(fn)),
        'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'akshare': ak.__version__})
    expected_url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    expected_params = {'fields1': 'f1,f2,f3,f4,f5,f6', 'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116',
        'ut': '7eea3edcaed734bea9cbfc24409ed989', 'klt': '101', 'fqt': '0', 'secid': '0.300058',
        'beg': '20100226', 'end': '20260929'}
    original = requests.sessions.Session.request
    result = {'requested_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'failed',
              'request_intents': [], 'wire_responses': [], 'new_rows': 0, 'published': False, 'not_a_backtest': True}
    def bounded(session, method, url, **kwargs):
        require(method.upper() == 'GET' and url == expected_url and kwargs.get('params') == expected_params and
                not result['request_intents'], 'Unexpected or repeated native request')
        result['request_intents'].append({'method': method, 'url': url, 'params': expected_params})
        session.trust_env = False
        kwargs.update(timeout=(8, 10), allow_redirects=False, proxies={'http': None, 'https': None})
        response = original(session, method, url, **kwargs)
        path = folder/'response.bin'; path.write_bytes(response.content)
        result['wire_responses'].append({'file': str(path), 'sha256': file_sha(path), 'status': response.status_code})
        response.raise_for_status()
        require(not response.is_redirect, 'Redirect rejected')
        return response
    try:
        with patch.object(requests.sessions.Session, 'request', bounded):
            frame = fn(symbol='300058', period='daily', start_date='20100226', end_date='20260929', adjust='', timeout=10)
        path = folder/'candidate.parquet'; frame.to_parquet(path, index=False)
        result.update(status='success' if len(frame) else 'empty', new_rows=len(frame), candidate=str(path),
                      candidate_sha256=file_sha(path), strict_usable=False)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    prior = Path('data/staging/strategies-batch85/20261009-industry-wizard-money-flow/candidate-audit.json')
    result.update(prior_industry_probe={'file': str(prior), 'sha256': file_sha(prior),
        'reason': 'Same SW2021 API previously DNS-failed; retain precise evidence, no retry or current-member substitution'},
        policy='One native raw daily GET from listing; no retry/redirect/proxy; candidate cannot silently replace frozen input')
    save(folder/'result.json', result)
    return result


def study(root, directory):
    validate(directory); bars = chan_input(directory); ns = namespace(directory, 0); rows = []
    diag = diagnostics(directory)
    for end in range(1, len(bars)+1):
        window = bars.iloc[max(0, end-1000):end]
        state = chan_state(ns, window_array(window))
        encoded = json.dumps(state, sort_keys=True, ensure_ascii=True)
        rows.append({'evaluation_date': str(window.date.iloc[-1]), 'window_start': str(window.date.iloc[0]),
            'window_rows': len(window), 'state_json': encoded, 'state_sha256': digest(state),
            'error': state['error'], 'bi_rows': len((state['kDataFrame_marked'] or {}).get('rows', [])),
            'xd_rows': len((state['kDataFrame_xd'] or {}).get('rows', []))})
        if end % 100 == 0:
            print(json.dumps({'completed_dates': end, 'total_dates': len(bars)}, ensure_ascii=True), flush=True)
    output = directory/'chan-daily-state-values.parquet'
    require(not output.exists(), 'State output already exists')
    frame = pd.DataFrame(rows); frame.to_parquet(output, index=False)
    summary = {'instrument': '300058.SZ', 'evaluation_dates': len(frame), 'dates': [rows[0]['evaluation_date'], rows[-1]['evaluation_date']],
        'policy': 'Fresh original KBarChan per date; at most1000 rows; adjusted OHLC anchored to each last row factor; no final-chart backfill',
        'state_file': str(output), 'state_sha256': file_sha(output), 'failed_dates': int(frame.error.notna().sum()),
        'error_counts': frame.error.dropna().value_counts().to_dict(), 'last_bi_rows': rows[-1]['bi_rows'],
        'last_xd_rows': rows[-1]['xd_rows'], 'nonempty_bi_dates': int(frame.bi_rows.gt(0).sum()),
        'nonempty_xd_dates': int(frame.xd_rows.gt(0).sum()), 'formal_acceptance': False, 'not_a_backtest': True,
        'new_trading_backtests': 0, 'platform_equivalence_proven': False,
        'limits': ['Historical daily frequency matches unit1d but platform inclusion/paused behavior unproved',
            'Original notebook end_time unknown; this is a daily component reference, not original chart/date reproduction',
            'Algorithm retains endpoint assumptions, approximate comparisons and nested-gap behavior',
            'Independent math/structure checks and complete forbidden-network replay still required for final acceptance']}
    save(directory/'component-research.json', summary)
    save(directory/'diagnostics.json', diag)
    return summary


def offline(directory):
    validate(directory); expected = pd.read_parquet(directory/'chan-daily-state-values.parquet')
    ns = namespace(directory, 0); bars = chan_input(directory)
    checks = []
    def forbidden(*a, **kw):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        require(diagnostics(directory) == read(directory/'diagnostics.json'), 'Diagnostics differ')
        for end in (1, 2, 3, 7, 30, 100, 250, 500, 1000, len(bars)):
            require(1 <= end <= len(bars), 'Prefix outside inputs')
            actual = chan_state(ns, window_array(bars.iloc[max(0, end-1000):end]))
            require(digest(actual) == expected.state_sha256.iloc[end-1] and
                    json.dumps(actual, sort_keys=True, ensure_ascii=True) == expected.state_json.iloc[end-1], 'Prefix differs')
            checks.append({'rows_consumed': end, 'window_rows': min(1000, end), 'state_sha256': digest(actual)})
    result = {'result': 'match', 'differences': 0, 'sampled_prefixes': checks,
              'scope': '10 selected prefix states plus all synthetic diagnostics; not full replay or independent algorithm',
              'socket_HTTP_disabled': True, 'formal_acceptance': False, 'not_a_backtest': True}
    save(directory/'stage-offline-verification.json', result)
    return result


def protect(root, directory):
    validate(directory)
    return protect_previous(root, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'probe', 'study', 'offline', 'protect', 'archive'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.action in ('offline', 'probe'):
        value = globals()[args.action](args.directory)
    elif args.action == 'archive':
        value = archive(args.root, 'Batch88 original daily Chan/notebook stage archived; accepted256 unchanged; no new trades')
    else:
        value = globals()[args.action](args.root, args.directory)
    print(json.dumps(value, ensure_ascii=False, default=str))
