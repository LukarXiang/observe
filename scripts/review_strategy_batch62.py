"""Archive original momentum arithmetic and missing multitask dependencies."""
import argparse
import ast
import base64
from contextlib import redirect_stdout
import copy
from decimal import Decimal
import hashlib
import importlib.util
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch52 import dependency_state
from scripts.review_strategy_batch58 import definitions, digest
from scripts import review_strategy_batch61 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch61/20261007-diffusion-leader-02')
RECEIPT = Path('docs/handoff/2026-10-07-batch61-verification.json')
OPERAND = previous.OPERAND
OPERAND_SHA = previous.OPERAND_SHA
CALENDAR = previous.CALENDAR
CALENDAR_SHA = previous.CALENDAR_SHA
SOURCES = ('2022年度精选策略/42.【研报复现】再论动量因子Ian666.txt',
    '2024年度精选策略1/91.【复现】多任务时序动量策略.txt')
SOURCE_SHA = ('80cdb914219c4be2ee3796f88f0650f78f06beff0cb94152a90b458d09173bfe',
    'b8a50bdda3ead04b9a561b07eb675ec0a3a2c3293ec4973e1a8fc557b22a4c61')
REFERENCES = (('repo/skfolio', 'src/skfolio/descriptor/_momentum/_rolling_momentum.py'),
    ('repo/skfolio', 'src/skfolio/preprocessing/_transformer/_cross_sectional/_cs_winsorizer.py'),
    ('repo/skfolio', 'src/skfolio/preprocessing/_transformer/_cross_sectional/_cs_standard_scaler.py'),
    ('repo/qlib', 'qlib/data/dataset/processor.py'))
METRICS = {'mom_1m': (20, None), 'mom_3m': (60, None), 'mom_6m': (120, None),
    'mom_12m': (240, None), 'mom_24m': (480, None), 'mom_1m_max': (20, 12),
    'ma_20': (20, 10), 'ma_60': (60, 40), 'ma_120': (120, 80), 'ma_240': (240, 160)}
ETFS = ['518880.SH', '513100.SH', '159915.SZ', '510300.SH']
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch62.py', 'tests/unit/test_batch62_research.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'import-failure-binding.json', 'test-failure-binding.json',
    'source-reviews/review.json', 'missing-inputs.json', 'model-parameters.json', 'existing-apis.json',
    'dependency-inventory.json', 'supplement-decision.json', 'price-input.parquet', 'calendar-input.parquet',
    'component-values.parquet', 'component-research.json', 'diagnostics.json', 'component-offline.json',
    'diagnostics-offline.json', 'offline-catalog.json', 'offline-verification.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch61.py') == proof['final_script_sha256'], 'Batch61 changed')
    accepted = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']
    require(file_sha(OPERAND) == OPERAND_SHA == accepted['price-input.parquet'] and
        file_sha(CALENDAR) == CALENDAR_SHA == accepted['calendar-input.parquet'], 'Accepted operands changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT),
        'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'), 'upstream': previous.binding(root, ACCEPTED),
        'price_file': str(OPERAND), 'price_sha256': OPERAND_SHA, 'calendar_file': str(CALENDAR),
        'calendar_sha256': CALENDAR_SHA, 'not_a_backtest': True}
    if directory is not None: require(read(directory / 'input-binding.json') == result, 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch62 momentum and multitask original-source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Source copy changed')
    return ast.parse(read_source(Path(row['source_copy']))[0])


def missing_inputs():
    trees = [ast.parse(read_source(Path('repo/量化策略源代码') / name)[0]) for name in SOURCES]
    csv = sorted({n.args[0].value for n in ast.walk(trees[0]) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'pd' and
        n.func.attr == 'read_csv' and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)})
    result = subprocess.run(['rg', '--files', 'repo', 'data'], capture_output=True, text=True, check=True)
    paths = result.stdout.splitlines(); matches = {n: sorted(p for p in paths if Path(p).name == Path(n).name) for n in csv}
    search = ['rg', '-l', r'class MTL_TSMOM|def optimize_multi_hyperparameters|class DataProcessor|def get_backtest_metrics',
        'repo', '--glob', '*.py', '--glob', '*.txt']
    found = subprocess.run(search, capture_output=True, text=True)
    require(found.returncode in (0, 1), found.stderr)
    modules = [{'module': n.module, 'symbols': [a.name for a in n.names]} for n in trees[1].body
        if isinstance(n, ast.ImportFrom) and n.module.startswith('src.')]
    return {'literal_csv': matches, 'missing_literal_csv': [n for n in csv if not matches[n]],
        'external_imports': modules, 'module_search': {'command': search, 'returncode': found.returncode, 'matches': sorted(found.stdout.splitlines())},
        'external_images': {'历史回测.png': sorted(p for p in paths if Path(p).name == '历史回测.png'),
            '19年回测.png': sorted(p for p in paths if Path(p).name == '19年回测.png')},
        'scope': 'Visible rg files and literal read_csv paths; dynamic outputs and external machines are not exhaustively searched',
        'replacement_allowed': False, 'not_a_backtest': True}


def model_parameters(tree):
    result = {}; calls = []; imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith('src.'):
            imports.append({'module': node.module, 'symbols': [n.name for n in node.names]})
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Constant):
            result[node.target.id] = node.value.value
        for call in ast.walk(node):
            if not isinstance(call, ast.Call): continue
            name = ast.unparse(call.func)
            if name not in ('dict', 'dataset.generate', 'dataset.build_dataset', 'optimize_multi_hyperparameters'): continue
            calls.append({'line': call.lineno, 'function': name, 'arguments_ast': ast.dump(call, include_attributes=False),
                'keywords': {k.arg: ast.unparse(k.value) for k in call.keywords}})
    return {'scalars': result, 'calls': calls, 'external_imports': imports, 'assets': ETFS,
        'declared_dates': ['2014-01-01', '2023-08-02'], 'price_basis': 'back-adjusted per original prose',
        'not_a_backtest': True, 'training_performed': False,
        'limits': ['No original modules, data, weights, seeds, purging or version evidence',
            'Original active target_vol=1 differs from optimization target_vol=0.5',
            'OTO future labels and 21-day future volatility are not decision-time features',
            '3bps training penalty is not actual unique-ledger costs']}


def embedded_images(text):
    rows = []
    for number, match in enumerate(re.finditer(r'data:image/png;base64,([A-Za-z0-9+/=]+)', text)):
        require(len(match[1]) < 14_000_000, 'Embedded image too large')
        raw = base64.b64decode(match[1], validate=True); require(raw.startswith(b'\x89PNG\r\n\x1a\n'), 'Bad embedded PNG')
        rows.append((f'embedded-{number}.png', raw))
    return rows


def api_evidence():
    import akshare as ak
    fn = ak.fund_etf_hist_em; code = inspect.getsource(fn)
    return {'akshare': ak.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
        'statsmodels_installed': importlib.util.find_spec('statsmodels') is not None,
        'function': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': code,
        'sha256': hashlib.sha256(code.encode()).hexdigest(),
        'pct_change_signature': str(inspect.signature(pd.DataFrame.pct_change)),
        'limits': ['Current pandas default fill_method=None differs from old pad defaults',
            'ETF daily interface does not provide original DataProcessor/MTL model semantics']}


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT); profiles = {}
    for table in ('instruments', 'bars_1d', 'bars_5m', 'corp_actions', 'adj_factors', 'instrument_status'):
        frame = store.load_state(state, table, filters=[('instrument', 'in', ETFS)])
        profiles[table] = {s: int(frame.instrument.eq(s).sum()) if 'instrument' in frame else 0 for s in ETFS}
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']), 'ETF_rows': profiles,
        'historical_equity_dependencies': dependency_state(root), 'ledger_sha256': file_sha('src/observe/ledger/book.py'),
        'not_a_backtest': True, 'limits': ['Ten stocks cannot replace whole 000985 historical universe or four ETFs',
            'Full-pool prices/ST/paused/size/ROE/BP/liquidity/industry and historical versions remain missing']}


def supplements():
    path = Path('data/staging/strategies-batch53/20261007-fund-intraday/probe-results.json')
    receipt = read('docs/handoff/2026-10-07-batch53-verification.json')
    require(file_sha(path) == receipt['checks']['evidence_sha256']['probe-results.json'], 'Accepted ETF supplement changed')
    rows = read(path)['results']; require(all(r['status'] == 'failed' and not r['files'] and not r['published'] for r in rows), 'Prior failure changed')
    return {'prior_file': str(path), 'sha256': file_sha(path), 'accepted_failed_probes': rows,
        'new_requests': 0, 'new_samples': 0, 'published': False, 'not_a_backtest': True,
        'decision': 'Reuse same-host verified ETF DNS failure; retrying symbols cannot resolve host DNS',
        'limits': ['A failed attempt does not prove provider absence; external model modules cannot be filled by prices']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    failure = Path('data/staging/strategies-batch62/20261007-momentum-import-failure')
    save(directory / 'import-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    failure = Path('data/staging/strategies-batch62/20261007-momentum-test-failure')
    save(directory / 'test-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name) + '.source'); require(not copied.exists(), 'Source copy exists'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'review': REVIEWS[name], 'newly_reviewed': True, 'definition_ast': definitions(ast.parse(text))})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Formula/missing-value/fit-interval comparison only; no original model or ledger replacement'})
    images = []
    for name, raw in embedded_images(read_source(Path(rows[1]['source_copy']))[0]):
        path = directory / name
        with path.open('xb') as out: out.write(raw)
        images.append({'file': str(path), 'bytes': len(raw), 'sha256': file_sha(path)})
    shutil.copyfile(OPERAND, directory / 'price-input.parquet'); shutil.copyfile(CALENDAR, directory / 'calendar-input.parquet')
    save(directory / 'missing-inputs.json', missing_inputs()); save(directory / 'model-parameters.json', model_parameters(ast.parse(read_source(Path(rows[1]['source_copy']))[0])))
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root))
    save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'images': images, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch62 two complete research sources frozen; original whole-pool and model inputs remain gated')


def validate_sources(directory):
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2, 'Source scope changed')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Previous catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha and
            row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Source/rule/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
        commit = subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        require(commit == row['commit'], 'Reference commit changed')
    images = embedded_images(read_source(Path(doc['sources'][1]['source_copy']))[0])
    require(len(images) == len(doc['images']), 'Image count changed')
    for (name, raw), row in zip(images, doc['images'], strict=True):
        require(row == {'file': str(directory / name), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} and
            file_sha(row['file']) == row['sha256'], 'Embedded image changed')
    require(file_sha(directory / 'price-input.parquet') == OPERAND_SHA and file_sha(directory / 'calendar-input.parquet') == CALENDAR_SHA,
        'Local operands changed')
    require(read(directory / 'missing-inputs.json') == missing_inputs() and
        read(directory / 'model-parameters.json') == model_parameters(source_tree(directory, 1)), 'Dependencies/parameters changed')
    failure = read(directory / 'import-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 2 and "No module named 'statsmodels'" in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Import failure binding changed')
    failure = read(directory / 'test-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Test failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Test failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 4 passed' in red['stdout'] and 'assert {} == []' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and
        red['test_sha256'] == failure['regression.original.py']['sha256'], 'Test failure binding changed')


def selected(directory, names, ns):
    import __future__
    nodes = [n for n in source_tree(directory, 0).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-momentum-functions>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns


def original_assignments(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in METRICS and name not in result: result[name] = node
    require(set(result) == set(METRICS), 'Original arithmetic missing')
    return result


def compute_values(close, tree):
    nodes = original_assignments(tree); ns = {'close': close}
    exec(compile(ast.Module(body=list(nodes.values()), type_ignores=[]), '<frozen-first-momentum-assignments>', 'exec'), ns)
    return {name: ns[name] for name in METRICS}


def reference_values(values):
    result = {}; arr = list(map(float, values))
    daily = [float('nan')] + [float(Decimal(str(arr[i])) / Decimal(str(arr[i-1])) - 1)
        if np.isfinite(arr[i]) and np.isfinite(arr[i-1]) else float('nan') for i in range(1, len(arr))]
    for name, (period, minimum) in METRICS.items():
        out = []
        for i, last in enumerate(arr):
            if not np.isfinite(last): value = float('nan')
            elif minimum is None:
                value = float(Decimal(str(last)) / Decimal(str(arr[i-period])) - 1) if i >= period and np.isfinite(arr[i-period]) else float('nan')
            else:
                window = (daily if name == 'mom_1m_max' else arr)[max(0, i-period+1):i+1]
                finite = [v for v in window if np.isfinite(v)]
                value = (max(finite) if name == 'mom_1m_max' else math.fsum(finite) / len(finite) / last) if len(finite) >= minimum else float('nan')
            out.append(value)
        result[name] = np.array(out)
    return result


def component(directory):
    validate_sources(directory)
    data = pd.read_parquet(directory / 'price-input.parquet'); calendar = pd.read_parquet(directory / 'calendar-input.parquet')
    data['date'] = pd.to_datetime(data.date, errors='raise'); calendar['date'] = pd.to_datetime(calendar.date, errors='raise')
    require(not data.duplicated(['date', 'instrument']).any() and not calendar.date.duplicated().any(), 'Duplicate operand keys')
    require(np.isfinite(data.close_adj.dropna()).all() and (data.close_adj.dropna() > 0).all(), 'Invalid price')
    dates = pd.DatetimeIndex(calendar.loc[calendar.is_open, 'date'].sort_values())
    require(set(data.date) == set(dates), 'Price/calendar dates differ')
    close = data.pivot(index='date', columns='instrument', values='close_adj').reindex(dates).sort_index(axis=1)
    tree = source_tree(directory, 0); studies = []; frames = []
    for label, prices in (('original_daily_2006_2019', close.loc['2006-01-01':'2019-05-31']), ('extended_2005_2026', close)):
        actual = compute_values(prices, tree); maximum = {name: 0. for name in METRICS}
        for symbol in prices:
            expected = reference_values(prices[symbol].to_numpy())
            for name in METRICS:
                observed = actual[name][symbol].to_numpy(); ref = expected[name]
                require(np.array_equal(np.isnan(observed), np.isnan(ref)), 'Original missing-value behavior differs')
                good = np.isfinite(ref); error = float(np.max(np.abs(observed[good] - ref[good]))) if good.any() else 0.
                require(np.allclose(observed[good], ref[good], rtol=1e-12, atol=1e-12), 'Independent arithmetic differs')
                maximum[name] = max(maximum[name], error)
        values = pd.DataFrame({'scope': label, 'date': np.repeat(prices.index.strftime('%Y-%m-%d'), len(prices.columns)),
            'instrument': np.tile(prices.columns, len(prices))})
        for name in METRICS: values[name] = actual[name].to_numpy().ravel()
        records = values.replace({np.nan: None}).to_dict('records'); frames.append(values)
        studies.append({'scope': label, 'first': str(prices.index.min().date()), 'last': str(prices.index.max().date()),
            'calendar_days': len(prices), 'stocks': list(prices.columns), 'slots': prices.size,
            'finite_values': {n: int(np.isfinite(actual[n]).sum().sum()) for n in METRICS},
            'unavailable_values': {n: int(actual[n].isna().sum().sum()) for n in METRICS},
            'max_abs_error': maximum, 'values_fingerprint': digest(records)})
    values = pd.concat(frames, ignore_index=True)
    return {'snapshot': SNAPSHOT, 'input_rows': len(data), 'open_days': len(dates), 'pandas': pd.__version__,
        'original_assignment_ast': {n: ast.dump(node, include_attributes=False) for n, node in original_assignments(tree).items()},
        'studies': studies, 'output_rows': len(values), 'not_a_backtest': True, 'platform_equivalent': False,
        'original_strategy_complete': False, 'cost_backtests': [],
        'limits': ['Raw first assignments only; no full-universe, winsorized, neutralized, residual or minute-bar factor claim',
            'Original interval starts fresh at 2006 rather than importing extended warmup',
            'Original pct_change executed under pandas3 fill_method=None; legacy default padding is not proved equivalent',
            'Frozen back-adjusted prices and ten-stock selection are numeric operands, not historically available full-pool versions']}, values


def constructor(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Factor_test')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')


def diagnostic_cases(directory):
    def no_OLS(*a, **kw): raise ValueError('Missing-date fixture must never fit OLS')
    ns = selected(directory, ['winsorize', 'standarize', 'changebar', 'cal_method_fac', 'neutralize_by_liqfcmc'],
        {'pd': pd, 'np': np, 'tqdm': lambda x: x,
            'sm': SimpleNamespace(add_constant=lambda frame: frame.assign(const=1.)), 'OLS': no_OLS})
    rows = []; frame = pd.DataFrame([[0., 1., 2., 100., 200.]], columns=list('ABCDE'))
    clipped = ns['winsorize'](frame)
    require(clipped is frame and frame.loc[0, 'D'] == 11.6395 and frame.loc[0, 'E'] == 12.381, 'Original MAD tail rank differs')
    rows.append({'case': 'MAD1483_mutates_input_and_ranks_extreme_tail', 'values': frame.iloc[0].tolist(), 'same_object': True})
    zero = ns['winsorize'](pd.DataFrame([[1., 1., 1., 2.]]))
    require(zero.iloc[0].tolist() == [1., 1., 1., 1.], 'Zero MAD collapse differs')
    rows.append({'case': 'zero_MAD_collapses_outlier', 'values': zero.iloc[0].tolist()})
    std = ns['standarize'](pd.DataFrame([[1., 2., 3.], [4., 4., 4.]]))
    require(std.iloc[0].tolist() == [-1., 0., 1.] and std.iloc[1].isna().all(), 'Sample standardization differs')
    rows.append({'case': 'sample_std_ddof1_constant_row_NaN', 'first': std.iloc[0].tolist(), 'constant_nan': 3})
    minutes = pd.DataFrame({'close': np.arange(10.)+1, 'volume': [9., 5., 8., 15., 11., 17., 13., 15., 12., 16.],
        'money': np.ones(10)}, index=pd.date_range('2020-01-01', periods=10, freq='min'))
    bars = ns['changebar'](minutes, 'volume', 3); extended = pd.concat([minutes, pd.DataFrame({'close': [99.], 'volume': [1000.], 'money': [1.]},
        index=[minutes.index[-1]+pd.Timedelta(minutes=1)])]); changed = ns['changebar'](extended, 'volume', 3)
    require(bars.close.tolist() == [4., 7., 10.] and changed.close.tolist() == [10., 99.], 'Full-sample bar dependence differs')
    rows.append({'case': 'future_append_changes_past_equal_volume_bar_endpoints', 'prefix_bars': bars.close.tolist(),
        'extended_bars': changed.close.tolist(), 'past_endpoint_changed': True})
    try: ns['cal_method_fac'](minutes, 'volume', 1, 3, minutes.index)
    except ValueError as exc: rows.append({'case': 'original_month_alias_M_fails', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected original M alias failure')
    dates = pd.bdate_range('2020-01-01', periods=5); fac = pd.DataFrame(1., index=dates, columns=list('ABCD'))
    missing = fac * np.nan
    try: ns['neutralize_by_liqfcmc'](missing, fac, fac)
    except KeyError as exc: rows.append({'case': 'skipped_all_missing_OLS_date_fails_reindex', 'error': type(exc).__name__, 'message': str(exc),
        'all_rows_missing_fixture': True, 'constant_column_fixture': True, 'OLS_fit_performed': False})
    else: raise ValueError('Expected skipped date failure')
    tree = source_tree(directory, 0); init = constructor(tree)
    require(not init.decorator_list and all(isinstance(n, ast.Constant) for n in init.args.defaults), 'Unsafe constructor')
    local = {'pd': pd, 'np': np, 'time': time}; exec(compile(ast.Module(body=[init], type_ignores=[]), '<original-factor-init>', 'exec'), local)
    open_ = pd.DataFrame(np.repeat(np.array([10., 12., 15., 18., 20.])[:, None], 4, axis=1), index=dates, columns=fac.columns)
    close = open_.copy(); paused = fac * 0; st = fac.astype(bool) & False; self = SimpleNamespace()
    try: local['__init__'](self, fac, 'synthetic', open_, close, paused, st, *[fac]*7)
    except ValueError as exc:
        require(hasattr(self, 'fret1') and not hasattr(self, 'factor_df'), 'Constructor failed at unexpected stage')
        rows.append({'case': 'constructor_M_fails_after_writing_future_labels', 'error': type(exc).__name__, 'message': str(exc),
            'fret1_first': float(self.fret1.iloc[0, 0]), 'factor_df_missing': True})
    else: raise ValueError('Expected constructor alias failure')
    # Execute the unchanged prefix only to isolate the next-day mask before M fails.
    prefix = copy.deepcopy(init); stop = next(k for k, n in enumerate(prefix.body) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Attribute) and t.attr == 'fret1' for t in n.targets)); prefix.body = prefix.body[:stop+1]
    exec(compile(ast.Module(body=[prefix], type_ignores=[]), '<original-factor-init-prefix>', 'exec'), local)
    first = SimpleNamespace(); second = SimpleNamespace(); local['__init__'](first, fac, 'fixture', open_, close, paused, st, *[fac]*7)
    paused2 = paused.copy(); paused2.iloc[1, 0] = 1
    local['__init__'](second, fac, 'fixture', open_, close, paused2, st, *[fac]*7)
    require(np.isfinite(first.fret1.iloc[0, 0]) and np.isnan(second.fret1.iloc[0, 0]), 'Future status dependence differs')
    rows.append({'case': 'next_day_paused_changes_prior_day_factor_label_mask', 'original': float(first.fret1.iloc[0, 0]),
        'changed': None, 'prefix_only': True, 'prefix_ast': ast.dump(prefix, include_attributes=False)})
    rows.append({'case': 'pandas3_original_pct_change_no_pad', 'values': pd.Series([1., np.nan, 2.]).pct_change().replace({np.nan: None}).tolist(),
        'legacy_default_equivalence_proved': False})
    try: pd.Grouper(freq='A')
    except ValueError as exc: rows.append({'case': 'original_annual_alias_A_fails', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected annual alias failure')
    method = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Factor_test')
    report = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == 'create_longshort_report')
    turnover = next(n for n in report.body if isinstance(n, ast.FunctionDef) and n.name == 'cal_longshort_turnover')
    exec(compile(ast.Module(body=[turnover], type_ignores=[]), '<original-turnover>', 'exec'), local)
    group = pd.DataFrame([[0., 4., 1., 2.], [0., 4., 1., 2.]], index=dates[:2], columns=list('ABCD'))
    turn = local['cal_longshort_turnover'](group, 1)
    local.update(df=group, period=1)
    exec(compile(ast.Module(body=turnover.body[:1], type_ignores=[]), '<original-turnover-selection>', 'exec'), local)
    columns = sorted(local['long_short_stocks'].iloc[0]); require(columns == ['A'], 'Original turnover selection differs')
    rows.append({'case': 'turnover_hardcodes_groups_0_and_9_for_five_groups', 'selected_columns': columns,
        'omitted_fifth_group_column': 'B', 'turnover': turn.replace({np.nan: None}).tolist(), 'function_ast': ast.dump(turnover, include_attributes=False)})
    bp = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'book_to_price_ratio_2' for t in n.targets))
    try: exec(compile(ast.Module(body=[bp], type_ignores=[]), '<original-undefined-BP>', 'exec'), {})
    except NameError as exc: rows.append({'case': 'residual_BP_variable_undefined_in_fresh_execution', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected undefined residual BP variable')
    return {'cases': rows, 'synthetic_fixture': True, 'real_historical_operand_windows': 0, 'not_a_backtest': True,
        'platform_equivalent': False, 'limits': ['Selected original functions/prefixes only; no platform runtime or trading',
            'Legacy M/A, default padding and economic future dependencies retained, not silently repaired']}


def diagnostics(directory):
    validate_sources(directory)
    with warnings.catch_warnings(), redirect_stdout(StringIO()):
        warnings.simplefilter('ignore'); return diagnostic_cases(directory)


def study(root, directory):
    binding(root, directory); before = implementation(); result, values = component(directory); diag = diagnostics(directory)
    require(before == implementation() and not (directory / 'component-values.parquet').exists(), 'Study changed or exists')
    values.to_parquet(directory / 'component-values.parquet', index=False)
    save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch62 original ten-stock arithmetic and original future/compatibility diagnostics archived; no trade backtest')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch62 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied):
        comp, values = component(directory); diag = diagnostics(directory); dep = inventory(root); catalog = offline_catalog(root, directory)
    save(directory / 'component-offline.json', comp); save(directory / 'diagnostics-offline.json', diag)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'))
    require(before == implementation() and comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'component_sha256': file_sha(directory / 'component-research.json'),
        'values_sha256': file_sha(directory / 'component-values.parquet'), 'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch62 forbidden-network arithmetic/diagnostics and three catalogs match')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch62.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch62_research.py', 'tests/unit/test_batch61_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed: {path}')
    require(before == implementation(), 'Checked implementation changed')
    for name in FILES:
        copied = directory / 'implementation-checked' / name; copied.parent.mkdir(parents=True, exist_ok=True)
        require(not copied.exists(), 'Checked copy exists'); shutil.copyfile(name, copied)
    save(directory / 'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory / 'checked-state.json'); off = read(directory / 'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == implementation() and CORE <= checked['evidence_sha256'].keys() and
        [r['command'] for r in checked['commands']] == commands() and
        all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json') and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') == file_sha(directory / 'diagnostics-offline.json') and
        off['values_sha256'] == file_sha(directory / 'component-values.parquet'), 'Offline proof changed')
    comp, values = component(directory); diag = diagnostics(directory); pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'))
    require(comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        inventory(root) == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json') and
        api_evidence() == read(directory / 'existing-apis.json'), 'Recomputed evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch62 momentum and multitask research accepted; original trading remains gated')
    receipt = Path('docs/handoff/2026-10-07-batch62-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked,
        'offline': off, 'offline_catalog': read(directory / 'offline-catalog.json'), 'component': comp, 'diagnostics': diag,
        'protection': protection, 'progress': progress, 'supplements': read(directory / 'supplement-decision.json'),
        'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory / 'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch62/20261007-momentum-multitask')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
