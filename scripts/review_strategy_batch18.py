"""Freeze KD/EMA/LOF source reviews and probe existing historical interfaces."""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data import raw, standardize as std
from observe.data.prices import with_adjusted
from observe.data.sources.baostock import BaoStock
from observe.data.store import Store
from observe.execution import sessions
from observe.runs import environment, file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import FIELDS, equal_cells
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-201432-93d0'
START, END = '2005-01-05', '2026-09-29'
SOURCES = ('2020年度精选策略/29 KD指标量化交易策略.txt', '2022年度精选策略/55.【策略研发】三进兵策略（变形版）.txt',
           '2023年度精选策略/19.人人可躺平16年3400%.txt')
POOL = ('300014.SZ', '300059.SZ', '300168.SZ', '300253.SZ', '300274.SZ', '000651.SZ', '000858.SZ', '600809.SH')
STOCKS = tuple(i for i in POOL if i != '000651.SZ')
FUNDS = {'lof_eastmoney': ('fund_lof_hist_em', {'symbol': '162605', 'period': 'daily', 'start_date': '20050105', 'end_date': '20260929', 'adjust': ''}),
         'lof_sina': ('fund_etf_hist_sina', {'symbol': 'sz162605'})}
FILES = {'src/observe/strategy_catalog.py', 'scripts/review_strategy_batch18.py', 'tests/unit/test_batch18_gates.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-apis.json', 'probe-results.json', 'input-analysis.json',
        'diagnostics.json', 'ema-policy-approval.json', 'input-validation-supplement.json'}
EMA_EVIDENCE = Path('data/staging/strategies-batch11/20261005-ema-evidence')


def save(path, value):
    require(not path.exists(), f'Archive exists: {path}')
    write_json(path, value)


def start(root, directory):
    checkpoint(root, directory, 'Batch18 KD/EMA/LOF checkpoint established; no economic variant implemented')
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['tables'] == state['tables'], 'Snapshot/publication differs')
    previous_file = Path(read(Path(root) / 'catalog/strategies/latest.json')['directory']) / 'catalog.json'
    previous = read(previous_file)
    previous = {r['path']: r for r in previous}
    target = directory / 'source-reviews'; target.mkdir()
    records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name; copied = target / f'{strategy_id(name)}.source'
        require(file_sha(source) == previous[name]['bytes_sha256'], 'Source changed')
        shutil.copyfile(source, copied); text, encoding = read_source(source)
        records.append({'source_path': str(source), 'source_sha256': file_sha(source), 'source_copy': str(copied),
            'source_copy_sha256': file_sha(copied), 'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name],
            'prior_review': {k: previous[name].get(k) for k in REVIEWS[name]}})
    official = directory / 'official-reused'; official.mkdir()
    evidence = []
    for name in ('technicalanalysis.bin', 'technicalanalysis.txt', 'sdk-technical-analysis.bin', 'manifest.json'):
        source = EMA_EVIDENCE / name; copied = official / name; shutil.copyfile(source, copied)
        evidence.append({'original_file': str(source), 'copied_file': str(copied), 'sha256': file_sha(source)})
    master = store.load_state(state, 'instruments', filters=[('instrument', 'in', ['000016.SZ', *POOL])])
    require(master[master.instrument.eq('000016.SZ')].kind.tolist() == ['stock'], 'KD security identity differs')
    apis = []; import akshare as ak; import baostock as bs
    for name in [*sorted({a for a, _ in FUNDS.values()}), 'query_history_k_data_plus', 'query_adjust_factor']:
        fn = getattr(bs if name.startswith('query_') else ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'signature': str(inspect.signature(fn))})
    save(directory / 'existing-apis.json', {'akshare_version': ak.__version__, 'baostock_version': bs.__version__, 'apis': apis,
        'stock_fields': FIELDS, 'funds': FUNDS, 'stock_start': START, 'stock_end': END})
    save(directory / 'ema-policy-approval.json', {'status': 'approved_by_user', 'source_sha256': records[1]['source_sha256'],
        'variant': 'ema_slots_talib_approx_v1', 'periods': [2, 25, 60], 'pool': list(POOL), 'slots': 5, 'cash_divisor': 1.5,
        'indicator': 'Locked TA-Lib EMA, mean seed, continuous verified traded history from 2005-01-05 or listing, known pauses skipped',
        'participation': 0.25, 'slippage': 0.00246, 'execution': 'Open, sell before buy, actual Book cash/remaining slots; three cost scenarios',
        'reply': '推进TA-Lib近似变体（推荐）', 'limits': ['Fixed ex-post pool bias', 'Original platform EMA and fill equivalence unproved']})
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    save(target / 'review.json', {'snapshot': SNAPSHOT, 'sources': records, 'official_evidence': evidence,
        'master_rows': master.astype(str).to_dict('records'), 'catalog': catalog,
        'previous_catalog_file': str(previous_file), 'previous_catalog_sha256': file_sha(previous_file),
        'api_evidence_sha256': file_sha(directory / 'existing-apis.json'), 'strategy_results': [],
        'limits': ['Current KD dict semantics are compatibility evidence, not proof of original Python2 behavior',
            'EMA internals and original intraday inputs remain unproved; user approved a separate TA-Lib variant',
            'Master names are current labels; LOF inputs and rules not published']})
    archive(root, 'Batch18 one new source and two refined reviews archived, KD dict and EMA/LOF gaps explicit')
    return {'status': 'ok', 'reviewed_or_refined': 3, 'newly_reviewed': sum(r['prior_review']['review_status'] != '人工审查完成' for r in records), 'catalog': catalog['catalog_id']}


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    socket.setdefaulttimeout(15)
    result = {'endpoint': endpoint, 'status': 'failed', 'files': [], 'wire_responses': [], 'published': False}
    evidence = read(directory / 'existing-apis.json')
    def saved(frame, dataset, params):
        path = raw.save(root, 'indicator_lof_probe_batch18', dataset, directory.name, frame)
        result['files'].append({'dataset': dataset, 'file': str(path), 'sha256': file_sha(path), 'rows': len(frame),
            'columns': list(frame), 'parameters': params})
    def checked(fn):
        row = next(r for r in evidence['apis'] if r['name'] == fn.__name__)
        require(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == row['source_sha256'], 'API source changed')
    original = requests.sessions.Session.request
    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'HTTP response limit reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15)
        response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        if endpoint in STOCKS:
            import baostock as bs
            require(bs.__version__ == evidence['baostock_version'], 'BaoStock version changed')
            checked(bs.query_history_k_data_plus); checked(bs.query_adjust_factor)
            with BaoStock(root).session() as source:
                code = std.to_baostock(endpoint)
                params = {'code': code, 'fields': FIELDS, 'start': START, 'end': END, 'frequency': 'd', 'adjustflag': '3'}
                frame = source._rows('query_history_k_data_plus', params,
                    lambda: bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END, frequency='d', adjustflag='3'))
                saved(frame, f'{endpoint}_bars', params)
                saved(source.adjust_factor(code, start='1990-01-01', end=END), f'{endpoint}_factors', {'code': code, 'start': '1990-01-01', 'end': END})
        else:
            import akshare as ak
            require(ak.__version__ == evidence['akshare_version'], 'AKShare version changed')
            name, params = FUNDS[endpoint]; fn = getattr(ak, name); checked(fn)
            with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = fn(**params)
            saved(frame, endpoint, params)
        result['status'] = 'success' if all(item['rows'] for item in result['files']) else 'empty_component'
    except Exception as exc:
        result.update(status='partial' if result['files'] else 'failed', error=f'{type(exc).__name__}: {exc}'[:1200])
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    require(not (directory / 'probe-results.json').exists() and not (directory / 'probes').exists(), 'Probes exist')
    results = []
    for endpoint in [*STOCKS, *FUNDS]:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch18', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=120 if endpoint in STOCKS else 75)
            row = {'endpoint': endpoint, 'returncode': process.returncode, 'stdout': process.stdout[-1200:], 'stderr': process.stderr[-1200:]}
        except subprocess.TimeoutExpired:
            row = {'endpoint': endpoint, 'status': 'timeout'}
        terminal = directory / 'probes' / endpoint / 'result.json'
        row['result'] = read(terminal) if terminal.exists() else {'status': row.get('status', 'failed'), 'files': [], 'wire_responses': []}
        results.append(row); print({'endpoint': endpoint, 'status': row['result']['status'], 'rows': [r['rows'] for r in row['result']['files']]}, flush=True)
        archive(root, f'Batch18 {endpoint} existing-interface probe {row["result"]["status"]} archived')
    save(directory / 'probe-results.json', {'snapshot': SNAPSHOT, 'results': results, 'api_sha256': file_sha(directory / 'existing-apis.json'), 'published': False})
    return {'status': 'ok', 'probes': len(results)}


def artifacts(directory):
    result = read(directory / 'probe-results.json')
    require(result['snapshot'] == SNAPSHOT and result['api_sha256'] == file_sha(directory / 'existing-apis.json'), 'Probe input binding differs')
    require(len(result['results']) == len(STOCKS) + len(FUNDS) and {r['endpoint'] for r in result['results']} == {*STOCKS, *FUNDS}, 'Probe set differs')
    frames = {}
    for row in result['results']:
        terminal = directory / 'probes' / row['endpoint'] / 'result.json'
        if terminal.exists(): require(read(terminal) == row['result'], 'Probe terminal differs')
        for item in [*row['result']['files'], *row['result']['wire_responses']]: require(file_sha(item['file']) == item['sha256'], 'Probe artifact changed')
        for item in row['result']['files']:
            frame = pd.read_parquet(item['file']); require(len(frame) == item['rows'] and list(frame) == item['columns'], 'Probe schema differs')
            if len(frame): frames[item['dataset']] = frame
    return frames


def missing_stock_components(frames, instrument):
    return [suffix for suffix in ('bars', 'factors')
            if instrument in STOCKS and (f'{instrument}_{suffix}' not in frames or frames[f'{instrument}_{suffix}'].empty)]


def raw_quality(frame):
    flags = {c: int((~frame[c].isin(['0', '1'])).sum()) for c in ('tradestatus', 'isST')}
    flow = frame[['volume', 'amount']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    return {'invalid_raw_flags': flags, 'invalid_flow_cells': int((~np.isfinite(flow) | (flow < 0)).sum())}


def analyze(root, directory, output='input-analysis.json'):
    require(not (directory / output).exists(), 'Analysis exists')
    frames = artifacts(directory); store = Store(root); state = store.state(SNAPSHOT)
    calendar = sessions(store.load_state(state, 'calendar')); master = store.load_state(state, 'instruments').set_index('instrument')
    filters = [('instrument', 'in', list(POOL))]
    old = store.load_state(state, 'bars_1d', filters=filters); factors = store.load_state(state, 'adj_factors', filters=filters)
    coverage = store.load_state(state, 'adj_coverage', filters=filters); actions = store.load_state(state, 'corp_actions', filters=filters)
    profiles = []
    for instrument in POOL:
        frozen = old[old.instrument.eq(instrument)].copy()
        key = f'{instrument}_bars'
        missing = missing_stock_components(frames, instrument)
        if missing:
            profiles.append({'instrument': instrument, 'status': 'blocked', 'missing_components': missing}); continue
        quality = raw_quality(frames[key]) if key in frames else {'invalid_raw_flags': {}, 'invalid_flow_cells': 0}
        bars = std.daily(frames[key]) if key in frames else frozen.copy()
        require(not bars.duplicated(['date', 'instrument']).any() and set(bars.instrument) == {instrument}, 'Invalid daily keys')
        listing = pd.to_datetime(master.loc[instrument, 'list_date']).date()
        expected = {d for d in calendar if date.fromisoformat(START) <= d <= date.fromisoformat(END) and d >= listing}
        common = frozen.merge(bars, on=['date', 'instrument'], how='inner', validate='one_to_one', suffixes=('_old', '_new'))
        columns = [c for c in frozen.columns if c not in ('date', 'instrument')]
        differences = equal_cells(common[[f'{c}_old' for c in columns]].set_axis(columns, axis=1), common[[f'{c}_new' for c in columns]].set_axis(columns, axis=1), columns)
        incoming = std.adj_factors(frames[f'{instrument}_factors']) if f'{instrument}_factors' in frames else factors[factors.instrument.eq(instrument)]
        before = factors[factors.instrument.eq(instrument)]
        compared = before.merge(incoming, on=['instrument', 'ex_date'], how='outer', validate='one_to_one', suffixes=('_old', '_new'), indicator=True)
        both = compared[compared._merge.eq('both')]
        prices = with_adjusted(bars, factors, coverage); trading = bars.loc[bars.is_trading, ['open', 'high', 'low', 'close', 'preclose']].to_numpy(float)
        rights = actions[actions.instrument.eq(instrument) & actions.rights_ratio.gt(0) & actions.ex_date.ge(date.fromisoformat(START))]
        profile = {'instrument': instrument, 'rows': len(bars), 'first': str(bars.date.min()), 'last': str(bars.date.max()),
            'pre2021_rows': int(bars.date.lt(date(2021, 1, 1)).sum()), 'paused_rows': int((~bars.is_trading).sum()),
            'missing_sessions': sorted(str(d) for d in expected - set(bars.date)), 'extra_sessions': sorted(str(d) for d in set(bars.date) - expected),
            'old_rows': len(frozen), 'overlap_rows': len(common), 'exact_overlap_differences': differences,
            'factor_unmatched': int(compared._merge.ne('both').sum()), 'factor_differences': int(both.back_factor_old.ne(both.back_factor_new).sum()),
            'invalid_price_cells': int((~np.isfinite(trading) | (trading <= 0)).sum()),
            'missing_adjusted_trading_closes': int(prices.loc[prices.is_trading, 'close_adj'].isna().sum()),
            'rights_events': rights.astype(str).to_dict('records'), 'reused_frozen_history': instrument == '000651.SZ', **quality}
        profile['status'] = 'ready' if (not any(profile[k] for k in ('missing_sessions', 'extra_sessions', 'factor_unmatched', 'factor_differences', 'invalid_price_cells', 'missing_adjusted_trading_closes'))
            and not any(quality['invalid_raw_flags'].values()) and not quality['invalid_flow_cells']
            and profile['old_rows'] == profile['overlap_rows'] and not any(differences.values())) else 'blocked'
        profiles.append(profile)
    fund_profiles = []
    for name in FUNDS:
        frame = frames.get(name)
        if frame is None: fund_profiles.append({'endpoint': name, 'status': 'unavailable'}); continue
        column = 'date' if 'date' in frame else '日期'; dates = pd.to_datetime(frame[column], errors='coerce')
        fund_profiles.append({'endpoint': name, 'status': 'sample_only', 'rows': len(frame), 'first': str(dates.min()), 'last': str(dates.max()),
            'invalid_dates': int(dates.isna().sum()), 'duplicate_dates': int(dates.duplicated().sum()),
            'rows_through_frozen_end': int(dates.dt.date.le(date.fromisoformat(END)).sum()), 'columns': list(frame)})
    result = {'status': 'ready' if all(r['status'] == 'ready' for r in profiles) else 'blocked', 'snapshot': SNAPSHOT,
        'stocks': profiles, 'funds': fund_profiles, 'published': False, 'strategy_results': [],
        'limits': ['Stock availability does not prove original EMA; approved alternative must remain a distinct variant',
            'Fund chart samples do not supply LOF events, historical listing/trading flags or execution rules']}
    if output != 'input-analysis.json':
        result['prior_analysis_sha256'] = file_sha(directory / 'input-analysis.json')
        result['reason'] = 'Require both incoming components; reject unknown raw flags and invalid volume/amount before publication'
    save(directory / output, result); archive(root, 'Batch18 stock overlap/adjustments and LOF chart coverage analysis archived')
    return {'status': result['status'], 'stocks': len(profiles), 'funds': fund_profiles}


def diagnose(root, directory):
    require(not (directory / 'diagnostics.json').exists(), 'Diagnostic exists')
    reviewed = read(directory / 'source-reviews/review.json'); row = reviewed['sources'][0]
    source = Path(row['source_copy']); require(file_sha(source) == row['source_sha256'], 'Source changed')
    previous = next(r for r in read(reviewed['previous_catalog_file']) if r['path'] == SOURCES[0])
    require(previous['bytes_sha256'] == row['source_sha256'], 'Code boundary belongs to different source')
    code = '\n'.join(read_source(source)[0].splitlines()[previous['code_start_line'] - 1:])
    tree = ast.parse(code); function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'market_open')
    calls = []; orders = []; security = '000016.XSHE'
    context = SimpleNamespace(current_dt=datetime(2024, 6, 25, 9, 30), previous_date=date(2024, 6, 24), portfolio=SimpleNamespace(available_cash=100000))
    def kd(instrument, **kwargs): calls.append({'instrument': instrument, **{k: str(v) for k, v in kwargs.items()}}); return {instrument: 60.0}, {instrument: 50.0}
    namespace = {'KD': kd, 'datetime': SimpleNamespace(timedelta=timedelta), 'g': SimpleNamespace(security=security),
        'log': SimpleNamespace(info=lambda *args: None), 'order_value': lambda *a: orders.append(a), 'order_target': lambda *a: orders.append(a)}
    def forbidden(*args, **kwargs): raise AssertionError('Diagnostic attempted network')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket.socket, 'connect_ex', forbidden):
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        try: namespace['market_open'](context); error = None
        except TypeError as exc: error = f'{type(exc).__name__}: {exc}'
    require(error is not None and len(calls) == 2 and calls[1]['check_date'] == '2024-06-23' and not orders, 'Expected dict/date behavior differs')
    result = {'status': 'ok', 'source_sha256': row['source_sha256'], 'not_a_backtest': True, 'dict_comparison_error': error,
        'calls': calls, 'orders': orders, 'limits': ['Only vetted original market_open AST with local KD dict stubs',
            'Current Python compatibility and literal date behavior, not original runtime numerics or trading reproduction']}
    save(directory / 'diagnostics.json', result); return result


def checks(root, directory):
    require(CORE <= {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}, 'Evidence missing')
    require(not (directory / 'checks.json').exists(), 'Checks exist'); artifacts(directory)
    commands = [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
        'scripts/verify_financial_import.py', 'scripts/review_strategy_batch18.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch18_gates.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py'],
        ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        print(results[-1], flush=True); require(process.returncode == 0, 'Checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch18-verification.json'); require(not output.exists(), 'Receipt exists')
    checked = read(directory / 'checks.json')
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']), 'Checks incomplete')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked sets incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    review = read(directory / 'source-reviews/review.json')
    require(file_sha(review['previous_catalog_file']) == review['previous_catalog_sha256'], 'Prior review catalog changed')
    for row in review['sources']:
        name = Path(row['source_path']).relative_to('repo/量化策略源代码').as_posix()
        require(file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']) and row['review'] == REVIEWS[name], 'Reviewed source/rules changed')
    for row in review['official_evidence']: require(file_sha(row['original_file']) == row['sha256'] == file_sha(row['copied_file']), 'Official evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    artifacts(directory); protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for path, sha in checked['implementation_sha256'].items():
        target = frozen / Path(path).name; shutil.copyfile(path, target); require(file_sha(target) == sha, 'Frozen code differs')
    supplement = read(directory / 'input-validation-supplement.json')
    require(supplement['status'] == 'ready' and supplement['prior_analysis_sha256'] == file_sha(directory / 'input-analysis.json'), 'Input supplement not ready/bound')
    result = {'status': 'ok', 'environment': environment(), 'reviews': review, 'analysis': read(directory / 'input-analysis.json'), 'validated_analysis': supplement,
        'diagnostics': read(directory / 'diagnostics.json'), 'checks': checked, 'protection': protection,
        'progress': archive(root, 'Batch18 KD/EMA/LOF reviews and input probes archived; no new strategy backtest'), 'strategy_results': []}
    save(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'worker', 'probe', 'analyze', 'validate-inputs', 'diagnose', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch18/20261005-kd-ema-lof'))
    parser.add_argument('--endpoint', choices=[*STOCKS, *FUNDS])
    args = parser.parse_args()
    if args.action == 'worker':
        if args.endpoint is None: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    elif args.action == 'validate-inputs': result = analyze(args.root, args.directory, 'input-validation-supplement.json')
    else: result = {'start': start, 'probe': probe, 'analyze': analyze, 'diagnose': diagnose, 'checks': checks, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
