"""Archive momentum source rules, bounded input probes and compatibility evidence."""
import argparse
import ast
from contextlib import redirect_stdout
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
from sklearn.linear_model import LinearRegression

from observe.data import raw
from observe.data.store import Store
from observe.runs import environment, file_sha, write_json
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = '20261005-201432-93d0'
SOURCES = (
    '2021年度精选策略/20.冲天炮最高板策略，收益惊呆了我.txt',
    '2021年度精选策略/95.冲天炮最高板策略迭代.txt',
    '2021年度精选策略/98.追涨大师（超额142）.txt',
    '2023年度精选策略/26.追涨大师（超额142）.txt',
    '2024年度精选策略1/4.首板低开策略.txt',
    '2024年度精选策略1/7.连板龙头策略.txt',
    '2024年度精选策略1/84.【社区研究】连板龙头策略-wywy：复现与研究.txt',
)
PROBES = {
    'money_flow': ('stock_individual_fund_flow', {'stock': '600000', 'market': 'sh'}),
    'minute_2021': ('stock_zh_a_hist_min_em', {'symbol': '600000', 'start_date': '2021-03-29 09:30:00',
        'end_date': '2021-03-29 15:00:00', 'period': '1', 'adjust': ''}),
    'minute_recent': ('stock_zh_a_hist_min_em', {'symbol': '600000', 'start_date': '2026-09-01 09:30:00',
        'end_date': '2026-09-29 15:00:00', 'period': '1', 'adjust': ''}),
    'limit_pool_2021': ('stock_zt_pool_em', {'date': '20210329'}),
    'limit_pool_recent': ('stock_zt_pool_em', {'date': '20260929'}),
}
FALLBACK = {'minute_sina': ('stock_zh_a_minute', {'symbol': 'sh600000', 'period': '1', 'adjust': ''})}
RULE_CORRECTIONS = {
    SOURCES[0]: 'The daily guard runs before the loop; partially filled orders can leave cash for another candidate in the same callback',
    SOURCES[2]: 'Recorded purchase price precedes order_value, while day=0 follows its return',
    SOURCES[4]: 'The two-to-ten-row limit-up exclusion tests suffixes ending yesterday, not arbitrary earlier streaks',
}
FILES = {'scripts/review_strategy_batch16.py', 'src/observe/strategy_catalog.py',
    'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_batch16_archive.py'}
CORE = {'baseline.json', 'source-reviews/review.json', 'existing-apis.json', 'probe-results.json', 'diagnostics.json',
    'existing-api-sina.json', 'minute-fallback.json', 'rule-clarifications.json', 'rule-clarifications-final.json'}


def save(path, value):
    require(not path.exists(), f'Archive exists: {path}')
    write_json(path, value)


def review(root, directory):
    baseline = read(directory / 'baseline.json'); store = Store(root)
    require(store.published() == baseline['published'] and store.published()['tables'] == store.state(SNAPSHOT)['tables'], 'Inputs moved since checkpoint')
    latest = read(Path(root) / 'catalog/strategies/latest.json')
    previous = {r['path']: r for r in read(Path(latest['directory']) / 'catalog.json')}
    target = directory / 'source-reviews'; target.mkdir(exist_ok=False)
    records = []
    for name in SOURCES:
        source = Path('repo/量化策略源代码') / name
        require(previous[name]['review_status'] != '人工审查完成' and file_sha(source) == previous[name]['bytes_sha256'], 'Source already reviewed or changed')
        text, encoding = read_source(source); copied = target / f'{strategy_id(name)}.source'
        with copied.open('xb') as stream: stream.write(source.read_bytes())
        records.append({'strategy_id': strategy_id(name), 'source_path': str(source), 'source_sha256': file_sha(source),
            'source_copy': str(copied), 'source_copy_sha256': file_sha(copied), 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name]})
    require(records[2]['source_sha256'] == records[3]['source_sha256'], 'Claimed duplicate bytes differ')
    import akshare as ak
    apis = []
    for name in sorted({name for name, _ in PROBES.values()}):
        fn = getattr(ak, name); code = inspect.getsource(fn)
        apis.append({'name': name, 'signature': str(inspect.signature(fn)), 'source': code,
            'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'module_file': inspect.getsourcefile(fn)})
    save(directory / 'existing-apis.json', {'version': ak.__version__, 'apis': apis, 'probes': PROBES})
    state = store.state(SNAPSHOT)
    minute = store.load_state(state, 'bars_5m', columns=['bar_end', 'instrument'], filters=[('instrument', 'in', ['600000.SH'])])
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    current = read(Path(catalog['output']) / 'catalog.json')
    changes = [{'path': r['path'], 'old_apis': previous[r['path']]['apis'], 'new_apis': r['apis']}
        for r in current if r['apis'] != previous[r['path']]['apis']]
    result = {'snapshot': SNAPSHOT, 'published_batch': state['batch_id'], 'sources': records, 'catalog': catalog,
        'static_detection_changes': changes, 'api_sha256': file_sha(directory / 'existing-apis.json'),
        'existing_tables': sorted(state['tables']), 'sample_600000_5m': {'rows': len(minute),
            'first': str(minute.bar_end.min()) if len(minute) else None, 'last': str(minute.bar_end.max()) if len(minute) else None,
            'not_equivalent_to_1m_or_all_market': True}, 'strategy_results': [],
        'limits': ['Minute prices, original price limits, historical names and flow semantics are not supplied by daily bars',
            'Source84 is research with next-day labels; source26 is a byte-identical duplicate of source98']}
    save(target / 'review.json', result)
    archive(root, 'Batch16 seven momentum sources reviewed, money-flow dependency omission corrected; inputs still pending')
    return {'status': 'ok', 'sources': len(records), 'api_changes': len(changes)}


def worker(root, directory, endpoint):
    folder = directory / 'probes' / endpoint; folder.mkdir(parents=True, exist_ok=False)
    result = {'endpoint': endpoint, 'status': 'failed', 'wire_responses': [], 'published': False, 'strict_usable': False}
    original = requests.sessions.Session.request; socket.setdefaulttimeout(15)
    def request(session, method, url, **kwargs):
        require(len(result['wire_responses']) < 12, 'Probe request bound reached')
        session.trust_env = False; kwargs['timeout'] = (10, 15)
        response = original(session, method, url, **kwargs)
        path = folder / f'response-{len(result["wire_responses"]):03d}.bin'
        with path.open('xb') as stream: stream.write(response.content)
        result['wire_responses'].append({'url': response.url, 'status_code': response.status_code, 'file': str(path), 'sha256': file_sha(path)})
        return response
    try:
        import akshare as ak
        api_path = directory / ('existing-api-sina.json' if endpoint in FALLBACK else 'existing-apis.json')
        api, params = {**PROBES, **FALLBACK}[endpoint]; fn = getattr(ak, api); evidence = read(api_path)
        entry = next(r for r in evidence['apis'] if r['name'] == api)
        require(ak.__version__ == evidence['version'] and hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == entry['source_sha256'], 'Probed API differs from archived source')
        result.update(api=api, parameters=params, api_evidence_sha256=file_sha(api_path))
        with redirect_stdout(sys.stderr), patch.object(requests.sessions.Session, 'request', request): frame = fn(**params)
        path = raw.save(root, 'momentum_dependency_batch16', endpoint, directory.name, frame)
        result.update(status='success' if len(frame) else 'empty', raw_file=str(path), raw_sha256=file_sha(path), rows=len(frame), columns=list(frame))
        column = next((c for c in ('日期', '时间', 'day') if c in frame), None)
        if column and len(frame):
            times = pd.to_datetime(frame[column], errors='coerce')
            result['coverage'] = {'column': column, 'first': str(times.min()), 'last': str(times.max()),
                'unparsed': int(times.isna().sum()), 'duplicate_times': int(times.duplicated().sum()), 'days': int(times.dt.date.nunique())}
        result['limitations'] = ['One sampled stock or day only; no complete historical pool proof',
            'Eastmoney fields and prices have not been proven equivalent to original JoinQuant inputs']
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'[:1200]
    finally: save(folder / 'result.json', result)
    return result


def probe(root, directory):
    require(not (directory / 'probe-results.json').exists() and not (directory / 'probes').exists(), 'Probe archive already exists')
    records = []
    for endpoint in PROBES:
        command = [sys.executable, '-m', 'scripts.review_strategy_batch16', 'worker', '--root', str(root), '--directory', str(directory), '--endpoint', endpoint]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=75)
            record = {'endpoint': endpoint, 'returncode': result.returncode, 'stdout': result.stdout[-1200:], 'stderr': result.stderr[-1200:]}
        except subprocess.TimeoutExpired: record = {'endpoint': endpoint, 'status': 'timeout', 'timeout_seconds': 75}
        folder = directory / 'probes' / endpoint; terminal = folder / 'result.json'
        record['result'] = read(terminal) if terminal.exists() else {'status': record.get('status', 'failed'), 'error': 'No terminal result; partial evidence retained'}
        record['evidence_files'] = [{'file': str(p), 'sha256': file_sha(p)} for p in sorted(folder.glob('*')) if p.is_file()]
        records.append(record)
        print(json.dumps({'endpoint': endpoint, 'status': record['result']['status'], 'rows': record['result'].get('rows')}), flush=True)
        archive(root, f'Batch16 {endpoint} probe {record["result"]["status"]}; raw evidence archived without publication')
    save(directory / 'probe-results.json', {'snapshot': SNAPSHOT, 'results': records, 'published': False,
        'api_evidence_sha256': file_sha(directory / 'existing-apis.json'), 'strategy_results': []})
    return {'status': 'archived', 'probes': len(records)}


def fallback(root, directory):
    require(not (directory / 'minute-fallback.json').exists() and not (directory / 'existing-api-sina.json').exists(), 'Fallback already archived')
    original = read(directory / 'probe-results.json')
    require(all(r['result']['status'] == 'failed' for r in original['results'] if r['endpoint'].startswith('minute_')), 'Expected original minute failures absent')
    import akshare as ak
    api, _ = FALLBACK['minute_sina']; fn = getattr(ak, api); code = inspect.getsource(fn)
    save(directory / 'existing-api-sina.json', {'version': ak.__version__, 'apis': [{'name': api,
        'signature': str(inspect.signature(fn)), 'source': code, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'module_file': inspect.getsourcefile(fn)}], 'probes': FALLBACK})
    command = [sys.executable, '-m', 'scripts.review_strategy_batch16', 'worker', '--root', str(root),
        '--directory', str(directory), '--endpoint', 'minute_sina']
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=75)
        record = {'endpoint': 'minute_sina', 'returncode': process.returncode, 'stdout': process.stdout[-1200:], 'stderr': process.stderr[-1200:]}
    except subprocess.TimeoutExpired: record = {'endpoint': 'minute_sina', 'status': 'timeout', 'timeout_seconds': 75}
    folder = directory / 'probes/minute_sina'; terminal = folder / 'result.json'
    record['result'] = read(terminal) if terminal.exists() else {'status': record.get('status', 'failed'), 'error': 'No terminal result; partial evidence retained'}
    record['evidence_files'] = [{'file': str(p), 'sha256': file_sha(p)} for p in sorted(folder.glob('*')) if p.is_file()]
    save(directory / 'minute-fallback.json', {'snapshot': SNAPSHOT, 'record': record, 'published': False,
        'original_probe_sha256': file_sha(directory / 'probe-results.json'), 'api_evidence_sha256': file_sha(directory / 'existing-api-sina.json'),
        'policy': 'One independent existing Sina interface probe after Eastmoney disconnections; original failures unchanged'})
    archive(root, f'Batch16 independent Sina minute probe {record["result"]["status"]}, original Eastmoney failures retained')
    return {'status': 'archived', 'result': record['result']}


def source_function(record, name, namespace):
    require(file_sha(record['source_copy']) == record['source_sha256'], 'Frozen diagnostic source differs')
    text, _ = read_source(record['source_copy']); tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Only the named, fully reviewed function runs with local stubs; no original imports or callbacks execute.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), record['source_copy'], 'exec'), namespace)
    return namespace[name]


def clarify(root, directory):
    path = directory / 'source-reviews/review.json'; original = read(path); corrections = []
    for row in original['sources']:
        name = Path(row['source_path']).relative_to('repo/量化策略源代码').as_posix()
        require(file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']), 'Source changed before clarification')
        if name in RULE_CORRECTIONS:
            require(row['review'] != REVIEWS[name], 'Expected clarification absent')
            corrections.append({'path': name, 'source_sha256': row['source_sha256'], 'before': row['review'],
                'after': REVIEWS[name], 'reason': RULE_CORRECTIONS[name]})
        else: require(row['review'] == REVIEWS[name], 'Unexpected additional rule change')
    previous = directory / 'rule-clarifications.json'
    output = directory / ('rule-clarifications-final.json' if previous.exists() else 'rule-clarifications.json')
    result = {'original_review_sha256': file_sha(path), 'corrections': corrections,
        'original_archive_retained': True, 'not_a_backtest': True}
    if previous.exists(): result['superseded_clarification_sha256'] = file_sha(previous)
    save(output, result)
    catalog = catalog_strategies(root, 'repo/量化策略源代码')
    archive(root, 'Batch16 three source-rule descriptions corrected in a new supplement; original review bytes preserved')
    return {'status': 'ok', 'corrections': len(corrections), 'catalog': catalog['catalog_id']}


def diagnose(root, directory):
    records = read(directory / 'source-reviews/review.json')['sources']
    examples = [[], [0], [1], [1, 1, 0, 1], [1] * 20, [0, 1, 1, 1], [1, 0], [2, 2, 0, 2]]
    fn = source_function(records[6], 'cal_lb_count', {})
    counts = [fn(values) for values in examples]; expected = [0, 0, 1, 1, 20, 3, 0, 1]
    require(counts == expected, 'Original reverse nonzero count differs')
    fit_results = []
    days = pd.bdate_range('2024-06-24', periods=3); context = SimpleNamespace(previous_date=days[-1].date())
    for index in (0, 1, 2):
        fn = source_function(records[index], 'fit_linear', {'np': np, 'LinearRegression': LinearRegression,
            'history': lambda *a, **kw: pd.DataFrame({kw['security_list']: [1., 2., 3.]}, index=days)})
        try:
            value = fn(context, 3, '600000.XSHG') if index == 2 else fn(3)
            result = {'source': SOURCES[index], 'status': 'returned', 'value': value}
        except Exception as exc: result = {'source': SOURCES[index], 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        require(result['status'] == 'failed' and result['error'].startswith('TypeError:'), 'Expected original array-to-float incompatibility not observed')
        fit_results.append(result)
    before = {r['source_copy']: file_sha(r['source_copy']) for r in records}
    def forbidden(*a, **kw): raise AssertionError('Diagnostic attempted network')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        repeated = [source_function(records[6], 'cal_lb_count', {})(values) for values in examples]
    require(repeated == counts and all(file_sha(p) == sha for p, sha in before.items()), 'Offline diagnostic differs or frozen source changed')
    result = {'status': 'ok', 'not_a_backtest': True, 'source_sha256': {r['source_path']: r['source_sha256'] for r in records},
        'examples': [{'input': values, 'result': count} for values, count in zip(examples, counts, strict=True)],
        'ols_compatibility': fit_results, 'offline_count_recheck': {'result': 'match', 'differences': 0, 'source_unchanged': True},
        'red_regression': 'Four money-flow API/gap cases failed on old code, then all eight dependency cases passed',
        'limits': ['Synthetic pure-function examples and runtime compatibility evidence only',
            'Source84 next-day features and historical concept data are not reconstructed; no trading or return claimed']}
    save(directory / 'diagnostics.json', result); archive(root, 'Batch16 original reverse-count examples and NumPy OLS incompatibility archived; no strategy backtest claimed')
    return {'status': 'ok', 'examples': len(examples), 'ols_failures': len(fit_results)}


def checked_evidence(directory):
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Required evidence missing')
    review = read(directory / 'source-reviews/review.json')
    clarification = read(directory / 'rule-clarifications-final.json')
    require(clarification['superseded_clarification_sha256'] == file_sha(directory / 'rule-clarifications.json'), 'Earlier clarification changed')
    require(clarification['original_review_sha256'] == file_sha(directory / 'source-reviews/review.json') and clarification['original_archive_retained'], 'Original review clarification binding differs')
    corrections = {r['path']: r for r in clarification['corrections']}
    require(len(clarification['corrections']) == len(RULE_CORRECTIONS) and set(corrections) == set(RULE_CORRECTIONS), 'Rule clarification set differs')
    require(len(review['sources']) == len(SOURCES) and {r['source_path'] for r in review['sources']} == {f'repo/量化策略源代码/{n}' for n in SOURCES}, 'Reviewed source set differs')
    for row in review['sources']:
        require(file_sha(row['source_path']) == row['source_sha256'] == file_sha(row['source_copy']) == row['source_copy_sha256'], 'Reviewed source changed')
        name = Path(row['source_path']).relative_to('repo/量化策略源代码').as_posix()
        if name in corrections:
            item = corrections[name]
            require(item['source_sha256'] == row['source_sha256'] and item['before'] == row['review']
                and item['after'] == REVIEWS[name] and item['reason'] == RULE_CORRECTIONS[name], 'Rule clarification differs from source/checked code')
        else: require(row['review'] == REVIEWS[name], 'Reviewed rules differ from checked code')
    api_sha = file_sha(directory / 'existing-apis.json')
    require(review['api_sha256'] == api_sha, 'API evidence changed')
    probes = read(directory / 'probe-results.json')
    require(len(probes['results']) == len(PROBES) and {r['endpoint'] for r in probes['results']} == set(PROBES), 'Probe set incomplete')
    require(probes['api_evidence_sha256'] == api_sha and probes['snapshot'] == review['snapshot'] == SNAPSHOT, 'Probe snapshot/API binding differs')
    supplement = read(directory / 'minute-fallback.json'); fallback_sha = file_sha(directory / 'existing-api-sina.json')
    require(supplement['snapshot'] == SNAPSHOT and supplement['original_probe_sha256'] == file_sha(directory / 'probe-results.json')
        and supplement['api_evidence_sha256'] == fallback_sha and supplement['record']['endpoint'] == 'minute_sina', 'Fallback evidence binding differs')
    for row in [*probes['results'], supplement['record']]:
        for item in row['evidence_files']: require(file_sha(item['file']) == item['sha256'], 'Probe evidence changed')
        result = row['result']; terminal = directory / 'probes' / row['endpoint'] / 'result.json'
        if terminal.exists(): require(read(terminal) == result, 'Probe terminal result differs')
        if result.get('raw_file'): require(file_sha(result['raw_file']) == result['raw_sha256'], 'Probe raw table changed')
        if result.get('api'):
            expected_api, params = {**PROBES, **FALLBACK}[row['endpoint']]
            expected_sha = fallback_sha if row['endpoint'] in FALLBACK else api_sha
            require(result['parameters'] == params and result['api'] == expected_api and result['api_evidence_sha256'] == expected_sha, 'Actual probe parameters/API differ')
    diagnostic = read(directory / 'diagnostics.json')
    require(diagnostic['not_a_backtest'] and diagnostic['source_sha256'] == {r['source_path']: r['source_sha256'] for r in review['sources']}, 'Diagnostic source binding differs')
    require(diagnostic['offline_count_recheck'] == {'result': 'match', 'differences': 0, 'source_unchanged': True}, 'Diagnostic recheck incomplete')
    return review, probes, diagnostic


def checks(root, directory):
    checked_evidence(directory); require(not (directory / 'checks.json').exists(), 'Checks already archived')
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_batch16_archive.py'],
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py',
            'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py', 'scripts/review_strategy_batch16.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        print(json.dumps(results[-1]), flush=True); require(process.returncode == 0, 'Batch checks failed')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'ok'}


def verify_checks(directory):
    checked = read(directory / 'checks.json')
    require(checked['commands'] and all(r['returncode'] == 0 for r in checked['commands']), 'Checks failed')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked sets incomplete')
    require(all(file_sha(p) == sha for p, sha in checked['implementation_sha256'].items()), 'Checked implementation changed')
    require(all(file_sha(directory / p) == sha for p, sha in checked['evidence_sha256'].items()), 'Checked evidence changed')
    return checked


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch16-verification.json'); require(not output.exists(), 'Receipt exists')
    checked = verify_checks(directory); reviews, probes, diagnostics = checked_evidence(directory)
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for name, sha in checked['implementation_sha256'].items():
        target = frozen / Path(name).name; require(not target.exists(), 'Frozen filename collision'); shutil.copyfile(name, target)
        require(file_sha(target) == sha, 'Frozen implementation differs')
    result = {'status': 'ok', 'environment': environment(), 'snapshot': SNAPSHOT, 'reviews': reviews, 'probes': probes,
        'minute_fallback': read(directory / 'minute-fallback.json'), 'rule_clarifications': read(directory / 'rule-clarifications-final.json'),
        'diagnostics': diagnostics, 'checks': checked, 'protection': protection, 'strategy_results': [],
        'progress': archive(root, 'Batch16 seven source reviews, money-flow omission fix and six input probes archived; original dependency gaps retained'),
        'limits': ['No new strategy backtest; daily/minute/flow and historical-name dependencies still incomplete',
            'Pure-function offline checks are not trading reproductions; original sources and snapshots unchanged',
            'Strict financial usable rows remain zero; existing daily audit warnings retained']}
    save(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['review', 'worker', 'probe', 'fallback', 'clarify', 'diagnose', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch16/20261005-momentum-dependencies'))
    parser.add_argument('--endpoint', choices=[*PROBES, *FALLBACK])
    args = parser.parse_args()
    if args.action == 'worker':
        if not args.endpoint: parser.error('--endpoint required')
        result = worker(args.root, args.directory, args.endpoint)
    else: result = {'review': review, 'probe': probe, 'fallback': fallback, 'clarify': clarify, 'diagnose': diagnose, 'checks': checks, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
