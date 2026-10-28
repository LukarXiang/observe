"""Freeze corn research and IF quarterly original kernels; no invented futures NAV."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import importlib.util
from io import StringIO
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch83 import definitions, plain, selected_nodes
from scripts import review_strategy_batch70 as daily
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch91/20261010-corn-quarterly-stage')
SNAPSHOT = '20261006-145049-e06c'
SOURCES = ('2021年度精选策略/34.套利三年，我的本金翻了10倍.txt',
    '2021年度精选策略/57.股指期货套利，这段时间都没有交易分享给大家.txt',
    '2021年度精选策略/58.严格资金管理，股指期货套利.txt',
    '2022年度精选策略/25.股指期货套利策略.txt',
    '2022年度精选策略/73.股指期货分钟-高收益-低回撤，你值得拥有.txt')
SOURCE_SHA = ('19a4d5c7545bf34777f0e461e13c16764fdcf59005e08203add143707dc4e23d',
    'c39bab1c0370394034bcf2166a2b7ed65c5187a8782d40d55efde08e64f7659b',
    '2083cd4ab47c5b422b0ba74a328d8a1dae699b4a75f2807d61035169764af057',
    'aa924593450b26189e3d2d11a33be878c0446a054885e387ad24acf4c5f82528',
    '3fc5e056b0fcee0768a814b6a45099f3c432467dfb5b36a1d703e6fb32a734a8')
IF_AST = '7410afbb2e93af5da1d5927fcc855d9b65a3351b6a5f91f4b22abaa819f127c7'
PARENT = Path('docs/handoff/2026-10-10-batch90-final-review.json')
DAILY_RECEIPT = Path('docs/handoff/2026-10-08-batch70-verification.json')
URL = 'https://stock2.finance.sina.com.cn/futures/api/jsonp.php/=/InnerFuturesNewService.getFewMinLine'
SYMBOLS = ('IF2003', 'IF2006', 'IF2612', 'IF2703')
FILES = ('scripts/review_strategy_batch91.py', 'tests/unit/test_batch91_stage.py')


def source_tree(number, directory=None):
    source = Path('repo/量化策略源代码')/SOURCES[number]
    require(file_sha(source) == SOURCE_SHA[number], 'Original91 source changed')
    if directory is not None:
        row = read(directory/'sources-frozen.json')[number]
        require(row['source_path'] == str(source) and row['sha256'] == SOURCE_SHA[number] and
            file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen91 source changed')
        source = Path(row['source_copy'])
    tree = ast.parse(read_source(source)[0])
    if number:
        require(hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == IF_AST,
            'IF original bodies differ')
    return tree


def namespace(number, values=None, directory=None):
    tree = source_tree(number, directory)
    return selected_nodes(tree, [n['name'] for n in definitions(tree)], dict(values or {}))


def api_file():
    spec = importlib.util.find_spec('akshare')
    require(spec is not None, 'Installed AKShare missing')
    return Path(spec.origin).parent/'futures/futures_zh_sina.py'


def api_evidence():
    installed = api_file(); reference = Path('repo/akshare/akshare/futures/futures_zh_sina.py')
    trees = [ast.parse(p.read_text(encoding='utf-8')) for p in (installed, reference)]
    functions = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and
        n.name == 'futures_zh_minute_sina') for t in trees]
    require(ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False),
        'Installed and reference minute functions differ')
    return {'installed_file': str(installed), 'installed_sha256': file_sha(installed),
        'reference_file': str(reference), 'reference_sha256': file_sha(reference),
        'function_ast': ast.dump(functions[0], include_attributes=False), 'function_AST_equal': True,
        'url': URL, 'queries': [{'symbol': s, 'type': '1'} for s in SYMBOLS],
        'historical_date_parameter': False, 'max_attempts_per_symbol': 1,
        'network_policy': 'Sequential GET; timeout5/15; no proxy/redirect/retry;8MiB cap; stop host on transport failure',
        'limits': ['Few recent minute rows cannot prove expired contracts or continuous original300-row histories',
            'IF2003/2006 are representative historical January2020 pair, not an asserted original experiment',
            'IF2612/2703 matches the original September2026 selection, not historical membership proof']}


def binding(directory=None):
    parent = read(PARENT)
    require(parent['status'] == 'ok' and parent['accepted_checkpoint'] == '20261010-153017-ee60',
        'Accepted90 scope changed')
    for prefix in ('receipt', 'checked_state', 'final_binding'):
        require(file_sha(parent[prefix+'_file']) == parent[prefix+'_sha256'], 'Accepted90 binding changed')
    checked = read(parent['checked_state_file'])
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted90 code changed')
    require(all(file_sha(Path(parent['accepted_directory'])/n) == sha for n, sha in
        checked['evidence_sha256'].items()), 'Accepted90 evidence changed')
    receipt = read(DAILY_RECEIPT); reused = {}
    require(receipt['status'] == 'ok', 'Accepted70 missing')
    for ins in ('C', 'CS'):
        folder = daily.DIRECTORY/'probes'/ins
        for name in ('result.json', 'response.bin', 'parsed.parquet'):
            key = f'probes/{ins}/{name}'
            require(file_sha(folder/name) == receipt['checks']['evidence_sha256'][key], 'Accepted corn evidence changed')
        row = read(folder/'result.json')
        require(row['status'] == 'success' and row['parameters'] == {'symbol': ins+'0'}, 'Continuous sample changed')
        for item in row['files']+row['responses']:
            require(file_sha(item['file']) == item['sha256'], 'Supplier bytes changed')
        reused[ins] = row
    result = {'snapshot': SNAPSHOT, 'parent_review_file': str(PARENT), 'parent_review_sha256': file_sha(PARENT),
        'daily_receipt_sha256': file_sha(DAILY_RECEIPT), 'supplier_samples': reused,
        'not_a_backtest': True, 'formal_acceptance': False}
    if directory is not None:
        require(result == read(directory/'input-binding.json'), 'Stage input binding changed')
        for ins in ('C', 'CS'):
            require(file_sha(directory/(ins+'-input.parquet')) == reused[ins]['files'][0]['sha256'], 'Copied input changed')
    return result


def stage(root, directory):
    bound = binding(); api = api_evidence(); store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['batch_id'] == state['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    cat = read(Path(read(Path(root)/'catalog/strategies/latest.json')['directory'])/'catalog.json')
    for number, name in enumerate(SOURCES):
        source_tree(number); row = next(r for r in cat if r['path'] == name)
        require(row['review_status'] != '人工审查完成' and row['bytes_sha256'] == SOURCE_SHA[number], 'Scope already reviewed')
    checkpoint(root, directory, 'Batch91 five original sources freeze; corn notebook and four identical IF bodies; no trades')
    save(directory/'input-binding.json', bound); save(directory/'existing-apis.json', api)
    folder = directory/'sources'; folder.mkdir(); rows = []
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; text, encoding = read_source(source)
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(source, copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'sha256': SOURCE_SHA[number],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()),
            'definition_ast': definitions(source_tree(number)), 'full_ast_sha256': hashlib.sha256(
                ast.dump(source_tree(number), include_attributes=False).encode()).hexdigest(), 'top_level_executed': False})
    save(directory/'sources-frozen.json', rows)
    for ins in ('C', 'CS'):
        shutil.copyfile(bound['supplier_samples'][ins]['files'][0]['file'], directory/(ins+'-input.parquet'))
    refs = directory/'references'; refs.mkdir()
    for name in ('installed_file', 'reference_file'):
        shutil.copyfile(api[name], refs/(name+'.py'))
    save(directory/'dependency-inventory.json', {'snapshot': SNAPSHOT,
        'registered_table_rows': {t: sum(r['rows'] for r in entries.values()) for t, entries in state['tables'].items()},
        'statsmodels_installed': importlib.util.find_spec('statsmodels') is not None,
        'original8888_identity_proven': False, 'original_futures_1m_registered': False,
        'IF_same_AST_sources': 4, 'original_signal_rows': 0, 'new_trading_backtests': 0,
        'limits': ['No accepted originalC8888/CS8888 price construction or point-in-time identity',
            'No admitted real IF quarterly1m/expiry/settlement/contract metadata table',
            'SoleBook has no accepted futures short/margin/multiplier/daily-settlement execution',
            'statsmodels original OLS/coint not executed when absent; independent least-squares is only a diagnostic']})
    return {'status': 'stage_frozen', 'source_count': len(rows), 'IF_AST_equal': True, 'not_a_backtest': True}


def minute_function(directory):
    evidence = read(directory/'existing-apis.json')
    require(api_evidence() == evidence, 'Minute API changed')
    path = directory/'references/installed_file.py'
    require(file_sha(path) == evidence['installed_sha256'], 'Frozen minute API changed')
    return selected_nodes(ast.parse(path.read_text(encoding='utf-8')), ['futures_zh_minute_sina'],
        {'requests': requests, 'json': json, 'pd': pd})['futures_zh_minute_sina']


def probe(root, directory):
    binding(directory); function = minute_function(directory); rows = []; transport_failed = False
    for symbol in SYMBOLS:
        folder = directory/'probes'/symbol; folder.mkdir(parents=True, exist_ok=False)
        row = {'symbol': symbol, 'parameters': {'symbol': symbol, 'type': '1'}, 'attempts': [],
            'responses': [], 'files': [], 'status': 'skipped_transport_failure' if transport_failed else 'failed',
            'published': False, 'strict_usable': False}
        if not transport_failed:
            def get(url, params):
                require(url == URL and params == row['parameters'] and not row['attempts'], 'Unexpected minute request')
                row['attempts'].append({'method': 'GET', 'url': url, 'parameters': params.copy()})
                with requests.Session() as session:
                    session.trust_env = False
                    with session.get(url, params=params, timeout=(5, 15), allow_redirects=False, stream=True) as response:
                        content = bytearray()
                        for block in response.iter_content(65536):
                            require(len(content)+len(block) <= 8*1024*1024, 'Minute response exceeds8MiB')
                            content.extend(block)
                        wire = folder/'response.bin'; wire.write_bytes(content)
                        row['responses'].append({'file': str(wire), 'sha256': file_sha(wire),
                            'status_code': response.status_code, 'encoding': response.encoding or 'utf-8', 'url': response.url})
                        response._content = bytes(content)
                        require(response.status_code == 200, 'Minute HTTP not200')
                        return response
            from unittest.mock import patch
            try:
                with patch.object(requests, 'get', get):
                    frame = function(symbol=symbol, period='1')
                path = folder/'parsed.parquet'; frame.to_parquet(path, index=False)
                row['files'].append({'file': str(path), 'sha256': file_sha(path), 'rows': len(frame), 'columns': list(frame)})
                row['status'] = 'success' if len(frame) else 'empty'
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'[:1800]
                transport_failed = isinstance(exc, requests.RequestException)
        save(folder/'result.json', row); rows.append(row)
    result = {'results': rows, 'not_a_backtest': True, 'published': False,
        'request_count': sum(len(r['attempts']) for r in rows), 'parsed_rows': sum(f['rows'] for r in rows for f in r['files'])}
    save(directory/'probe-results.json', result); save(directory/'probe-audit.json', network_disabled(lambda: probe_audit(directory)))
    return result


def probe_audit(directory):
    from unittest.mock import patch
    function = minute_function(directory); entries = []
    rows = read(directory/'probe-results.json')['results']
    require([r['symbol'] for r in rows] == list(SYMBOLS), 'Probe scope changed')
    for row in rows:
        require(row == read(directory/'probes'/row['symbol']/'result.json') and
            row['parameters'] == {'symbol': row['symbol'], 'type': '1'} and len(row['attempts']) <= 1 and
            not row['published'] and not row['strict_usable'], 'Probe binding changed')
        for item in row['responses']+row['files']:
            require(file_sha(item['file']) == item['sha256'], 'Minute response/parsed changed')
        entry = {'symbol': row['symbol'], 'status': row['status'], 'requests': len(row['attempts']),
            'responses': len(row['responses']), 'rows': 0, 'admitted_original_rows': 0}
        if row['files']:
            require(len(row['files']) == len(row['responses']) == 1, 'Unexpected minute wire/file count')
            wire = row['responses'][0]; calls = []
            def get(url, params):
                require(not calls and url == URL and params == row['parameters'], 'Minute replay request changed')
                calls.append(url); response = requests.Response(); response._content = Path(wire['file']).read_bytes()
                response.encoding = wire['encoding']; response.status_code = wire['status_code']; return response
            with patch.object(requests, 'get', get):
                parsed = function(symbol=row['symbol'], period='1')
            pd.testing.assert_frame_equal(parsed, pd.read_parquet(row['files'][0]['file']), check_exact=True)
            dates = pd.to_datetime(parsed.datetime, errors='coerce')
            entry.update({'rows': len(parsed), 'datetime_min': str(dates.min()), 'datetime_max': str(dates.max()),
                'invalid_datetimes': int(dates.isna().sum()), 'duplicate_datetimes': int(dates.duplicated().sum()),
                'contract_identity_and_historical_coverage_proven': False})
        entries.append(entry)
    return {'cases': entries, 'not_a_backtest': True, 'historical_IF_coverage_admitted': 0}


def diagnostics(directory=None):
    rows = []
    for number in range(1, 5):
        orders = []; declares = []; schedules = []
        ns = namespace(number, {'np': np, 'dt': dt, 'datetime': dt, 'g': SimpleNamespace(),
            'set_benchmark': lambda code: declares.append(['benchmark', code]),
            'set_option': lambda *a: declares.append(['option', *a]),
            'set_subportfolios': lambda x: declares.append(['accounts', x]),
            'SubPortfolioConfig': lambda **kw: kw, 'OrderCost': lambda **kw: kw,
            'StepRelatedSlippage': lambda x: x,
            'set_slippage': lambda *a, **kw: declares.append(['slippage', list(a), kw]),
            'set_order_cost': lambda *a, **kw: declares.append(['cost', list(a), kw]),
            'log': SimpleNamespace(set_level=lambda *a: None),
            'run_daily': lambda fn, **kw: schedules.append([fn.__name__, kw])}, directory)
        context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 2, 9, 33),
            portfolio=SimpleNamespace(starting_cash=1e6, available_cash=1e6, positions_value=0.,
                long_positions={}, short_positions={}))
        ns['initialize'](context)
        require(schedules == [['before_market_open', {'time': 'before_open', 'reference_security': '000300.XSHG'}],
            ['market_open', {'time': 'every_bar'}]] and declares == [['benchmark', '000300.XSHG'],
            ['option', 'order_volume_ratio', .25], ['option', 'use_real_price', True],
            ['accounts', [{'cash': 1e6, 'type': 'index_futures'}]]], 'Initialization differs')
        rows.append({'source': number, 'case': 'defaults', 'schedules': schedules.copy(), 'declarations': declares.copy()})
        expected = [(2020, 3, 2020, 6)] + [(2020, 6, 2020, 9)]*3 + [(2020, 9, 2020, 12)]*3 + \
            [(2020, 12, 2021, 3)]*3 + [(2021, 3, 2021, 6)]*2
        for month, (y1, m1, y2, m2) in enumerate(expected, 1):
            context.current_dt = dt.datetime(2020, month, 2); ns['before_market_open'](context)
            codes = [ns['g'].code1, ns['g'].code2]
            require(codes == [f'IF{y1%100:02d}{m1:02d}.CCFX', f'IF{y2%100:02d}{m2:02d}.CCFX'], 'Quarterly mapping differs')
            rows.append({'source': number, 'case': 'month_'+str(month), 'context_date': str(context.current_dt.date()), 'codes': codes})
        for date, margin, today_fee in (('2017-09-17', .2, .00092), ('2017-09-18', .15, .00069),
            ('2018-12-02', .15, .00069), ('2018-12-03', .1, .00046),
            ('2019-04-21', .1, .00046), ('2019-04-22', .1, .000345)):
            declares.clear(); context.current_dt = dt.datetime.fromisoformat(date); ns['set_slip_fee'](context)
            require(ns['g'].futures_margin_rate == margin and declares == [['slippage', [2], {'type': 'futures', 'ref': 'IF'}],
                ['cost', [{'open_commission': .000023, 'close_commission': .000023, 'close_today_commission': today_fee}],
                    {'type': 'index_futures'}], ['option', 'futures_margin_rate', margin]], 'Cost/margin boundary differs')
            rows.append({'source': number, 'case': 'fees_'+date, 'margin': margin, 'declarations': declares.copy()})
        ns['g'].code1 = 'IF2003.CCFX'; ns['g'].code2 = 'IF2006.CCFX'; ns['g'].futures_margin_rate = .2
        def run(name, base, last, expected_orders, hour=9, minute=33, cash=1e6, value=0., longs=None, shorts=None,
                response='none', date_index=False, twice=False, expected_error=None):
            orders.clear(); requested = []; context.current_dt = dt.datetime(2020, 1, 2, hour, minute)
            context.portfolio.available_cash = cash; context.portfolio.positions_value = value
            context.portfolio.long_positions = dict(longs or {}); context.portfolio.short_positions = dict(shorts or {})
            index = pd.date_range('2020-01-01', periods=300, freq='min') if date_index else range(-300, 0)
            far = pd.Series(np.full(300, 100.), index=index); near = far+base; near.iloc[-1] = 100.+last
            def history(code, count, **kw):
                requested.append({'code': code, 'count': count, **kw})
                return pd.DataFrame({'close': near if code == ns['g'].code1 else far})
            def order(code, quantity, **kw):
                orders.append({'code': code, 'quantity': quantity, **kw})
                if response == 'raise': raise RuntimeError('synthetic first leg exception')
                return None
            ns['attribute_history'] = history; ns['order_target'] = order
            error = None
            try:
                ns['market_open'](context)
                if twice: ns['market_open'](context)
            except (KeyError, OverflowError, RuntimeError) as exc:
                error = type(exc).__name__
            require(error == expected_error and len(orders) == expected_orders, 'IF diagnostic differs: '+name)
            require(all(r['count'] == 300 and r['unit'] == '1m' and r['fields'] == ['close'] and r['df']
                for r in requested), 'Original history contract changed')
            if orders and value == 0:
                qty = 0 if math.isnan(cash) else int(cash/(200.+last)/300/.2/1.2)
                require(all(o['quantity'] == qty for o in orders) and orders[0]['side'] == 'short', 'Sizing/order direction differs')
                if len(orders) >= 2:
                    require(orders[1]['side'] == 'long' and all(o['close_today'] is False for o in orders), 'Two-leg sequencing differs')
            if value and orders:
                require(orders[0] == {'code': list(longs or {'': None})[-1], 'quantity': 0, 'side': 'long', 'close_today': False}
                    and orders[1] == {'code': list(shorts or {'': None})[-1], 'quantity': 0, 'side': 'short'}, 'Last-key close differs')
            rows.append({'source': number, 'case': name, 'orders': orders.copy(), 'error': error,
                'requests': requested, 'last_gap': last, 'base_gap': base, 'response_model': response})
        run('time_0932_fetches_but_no_orders', 10., 12., 0, minute=32)
        run('time_0933_opens', 10., 12., 2)
        run('positive_threshold_equal', 10., 11., 0)
        run('positive_threshold_below', 10., 10., 0)
        run('negative_threshold_equal', -10., -9., 0)
        run('negative_threshold_above', -10., -8., 2)
        run('zero_quantile_positive_last', 0., 1., 2)
        run('zero_quantile_equal_last', 0., 0., 0)
        run('small_cash_zero_targets', 10., 12., 2, cash=1.)
        run('NaN_cash_zero_targets', 10., 12., 2, cash=math.nan)
        run('negative_cash_negative_targets', 10., 12., 2, cash=-1e6)
        run('infinite_cash_fails_before_time_guard', 10., 12., 0, minute=0, cash=math.inf, expected_error='OverflowError')
        run('datetime_series_minus1_failure', 10., 12., 0, date_index=True, expected_error='KeyError')
        run('rejected_first_leg_still_submits_second', 10., 12., 2)
        run('first_leg_exception_prevents_second', 10., 12., 1, response='raise', expected_error='RuntimeError')
        run('unchanged_empty_value_duplicates_two_legs', 10., 12., 4, twice=True)
        run('close_quantile_equal', 10., 10., 0, value=1.)
        run('close_only_last_key', 10., 0., 2, value=1., longs={'old1': 1, 'old2': 2}, shorts={'old3': 1, 'old4': 2})
        run('close_empty_dicts_submits_empty_codes', 10., 0., 2, value=1.)
        run('old_month_positions_close_last_old_contract', 10., 0., 2, value=1., longs={'IF1912': 1}, shorts={'IF1909': 1})
    zscore = namespace(0, {'np': np}, directory)['zscore']
    result = zscore(pd.Series([1., 2., 3.]))
    np.testing.assert_allclose(result, [-math.sqrt(1.5), 0., math.sqrt(1.5)], rtol=1e-15)
    rows.append({'source': 0, 'case': 'population_zscore', 'actual': result.tolist()})
    require(zscore(pd.Series([1., 1., 1.])).isna().all(), 'Constant zscore differs')
    rows.append({'source': 0, 'case': 'constant_zscore_NaN', 'all_NaN': True})
    return {'cases': rows, 'case_count': len(rows), 'synthetic_only': True, 'not_a_backtest': True,
        'IF_unique_executable_bodies': 1, 'IF_source_bodies_tested': 4, 'platform_equivalence_proven': False,
        'orders_never_filled': True, 'new_trading_backtests': 0}


def regression(frame, label, zscore):
    require(len(frame) >= 3 and not frame.date.duplicated().any(), 'Invalid corn sample')
    x = frame.C.to_numpy(float); y = frame.CS.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all() and (x > 0).all() and (y > 0).all(), 'Invalid close')
    alpha, beta = map(float, np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0])
    mx = math.fsum(map(float, x))/len(x); my = math.fsum(map(float, y))/len(y)
    independent_beta = math.fsum((float(a)-mx)*(float(b)-my) for a, b in zip(x, y, strict=True))/math.fsum(
        (float(a)-mx)**2 for a in x)
    independent_alpha = my-independent_beta*mx
    require(math.isclose(beta, independent_beta, rel_tol=1e-11, abs_tol=1e-11) and
        math.isclose(alpha, independent_alpha, rel_tol=1e-11, abs_tol=1e-9), 'Independent regression differs')
    spread = pd.Series(beta*x-y); z = zscore(spread)
    mean = math.fsum(map(float, spread))/len(spread)
    sigma = math.sqrt(math.fsum((float(v)-mean)**2 for v in spread)/len(spread))
    require(sigma > 0 and math.isclose(mean, -alpha, abs_tol=1e-9), 'Intercept/spread relation differs')
    expected = np.array([(float(v)-mean)/sigma for v in spread])
    np.testing.assert_allclose(z, expected, rtol=1e-11, atol=1e-11)
    values = frame.copy(); values['range'] = label; values['fitted'] = alpha+beta*x
    values['spread'] = spread.to_numpy(); values['original_zscore_on_candidate'] = z.to_numpy()
    values['independent_zscore'] = expected
    profile = {'range': label, 'rows': len(frame), 'date_min': str(frame.date.min().date()),
        'date_max': str(frame.date.max().date()), 'intercept_numpy': alpha, 'slope_numpy': beta,
        'intercept_independent': independent_alpha, 'slope_independent': independent_beta,
        'spread_mean': mean, 'spread_population_std': sigma, 'z_above1': int((z > 1).sum()),
        'z_below_minus1': int((z < -1).sum()), 'independent_z_max_abs_error': float(np.max(np.abs(z-expected))),
        'coint_executed': False, 'original_statsmodels_OLS_executed': False,
        'retrospective_full_period_fit': True, 'not_a_backtest': True}
    return values, profile


def component(directory):
    binding(directory); zscore = namespace(0, {'np': np}, directory)['zscore']; frames = {}; profiles = []
    for ins in ('C', 'CS'):
        frame = pd.read_parquet(directory/(ins+'-input.parquet')); dates = pd.to_datetime(frame.date, errors='coerce')
        require(not dates.isna().any() and not dates.duplicated().any(), 'Supplier dates invalid or duplicated')
        frames[ins] = pd.DataFrame({'date': dates, ins: pd.to_numeric(frame.close, errors='coerce')})
        profiles.append({'symbol': ins+'0', 'parsed_rows': len(frame), 'date_min': str(dates.min().date()),
            'date_max': str(dates.max().date()), 'rows_after_cutoff_excluded': int((dates > '2026-09-29').sum())})
    joined = frames['C'].merge(frames['CS'], on='date', how='inner', validate='one_to_one').sort_values('date').reset_index(drop=True)
    outputs = []; studies = []
    for label, start, end in (('original_period_candidate', '2015-01-05', '2019-06-10'),
            ('extended_candidate', '2014-12-22', '2026-09-29')):
        subset = joined[joined.date.between(start, end)].reset_index(drop=True)
        values, summary = regression(subset, label, zscore); outputs.append(values); studies.append(summary)
    return pd.concat(outputs, ignore_index=True), {'supplier_profiles': profiles, 'studies': studies,
        'not_a_backtest': True, 'original8888_reproduced': False, 'coint_reproduced': False,
        'limits': ['C0/CS0 are supplier continuous candidates; original8888 identity/methodology unproved',
            'Two ranges overlap; no independent holdout and no forward-filled or interpolated dates',
            'NumPy interceptOLS cross-checked with centered fsum; not installed statsmodels/coint execution',
            'Pinned original population zscore only; full-period beta is retrospective, never converted to orders',
            'No literal notebook trading rules, costs, NAV, titletenfold claim or futures strategy backtest']}


def audit(root, directory):
    binding(directory)
    diag = network_disabled(lambda: diagnostics(directory)); values, summary = network_disabled(lambda: component(directory))
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', summary)
    with (directory/'corn-values.parquet').open('xb') as stream: values.to_parquet(stream, index=False)
    return {'status': 'stage_audited', 'diagnostic_cases': diag['case_count'], **summary}


def offline(root, directory):
    binding(directory)
    require(network_disabled(lambda: diagnostics(directory)) == read(directory/'diagnostics.json'), 'Diagnostic replay differs')
    values, summary = network_disabled(lambda: component(directory))
    require(summary == read(directory/'component-research.json'), 'Component summary replay differs')
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory/'corn-values.parquet'), check_exact=True)
    require(network_disabled(lambda: probe_audit(directory)) == read(directory/'probe-audit.json'), 'Minute parser replay differs')
    for row in read(directory/'input-binding.json')['supplier_samples'].values():
        replay = network_disabled(lambda row=row: daily.replay_frame(row))
        pd.testing.assert_frame_equal(replay, pd.read_parquet(row['files'][0]['file']), check_exact=True)
    result = {'status': 'match', 'differences': 0, 'socket_HTTP_disabled': True,
        'diagnostic_cases': read(directory/'diagnostics.json')['case_count'], 'candidate_rows_with_overlapping_ranges': len(values),
        'supplier_daily_wire_parsers_matched': 2, 'not_a_backtest': True, 'formal_acceptance': False}
    save(directory/'offline-verification.json', result)
    return result


def checks(root, directory):
    import subprocess
    import sys
    binding(directory); rows = []; before = {n: file_sha(n) for n in FILES}
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', *FILES],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch91_stage.py',
            'tests/unit/test_batch70_research.py', 'tests/unit/test_catalog_futures.py', 'tests/unit/test_research_metadata.py'],
        ['git', 'diff', '--check']]
    for number, command in enumerate(commands):
        path = directory/f'check-{number}.log'
        with path.open('x') as output: result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, 'Stage check failed: '+str(path))
    require(before == {n: file_sha(n) for n in FILES}, 'Checked code changed')
    protected = protect(root, directory)
    for name in FILES:
        copied = directory/'implementation-checked'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, copied)
    save(directory/'checked-state.json', {'status': 'stage_checks_passed', 'commands': rows,
        'implementation_sha256': before, 'protection': protected, 'formal_acceptance': False})
    save(directory/'stage-evidence.json', {'implementation_sha256': before,
        'files': {p.relative_to(directory).as_posix(): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
        'not_a_backtest': True, 'formal_acceptance': False})
    progress = archive(root, 'Batch91 five source stage: four identical IF kernels, daily corn candidate statistics and bounded minute coverage; not accepted/trades')
    return {'status': 'stage_checks_passed', 'checkpoint': progress['checkpoint'], 'formal_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'probe', 'audit', 'offline', 'checks'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()): result = globals()[args.action](args.root, args.directory)
    print(json.dumps(plain(result), ensure_ascii=False))
