"""Freeze financial source arithmetic without admitting unproved historical versions."""
import argparse
import ast
from contextlib import redirect_stdout
from decimal import Decimal
import hashlib
import inspect
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from observe.data.financial_schema import parse_number
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch52 import dependency_state
from scripts.review_strategy_batch58 import definitions
from scripts import review_strategy_batch62 as previous
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

SNAPSHOT = previous.SNAPSHOT
ACCEPTED = Path('data/staging/strategies-batch62/20261007-momentum-multitask')
RECEIPT = Path('docs/handoff/2026-10-07-batch62-verification.json')
SOURCES = ('2020年度精选策略/03 机器学习多因子策略.txt', '2020年度精选策略/07 穿越牛熊基业长青的价值精选策略.txt')
SOURCE_SHA = ('77f0c7ff116e2e83699e092f7fd776667d922ad1aa98b65f92df5f7ee20a9906',
    'c478a8978b9adf6219e2e19eb32f95322b00428029907be914f9cd998bfdad2d')
REFERENCES = (('repo/baostock', 'baostock/evaluation/season_index.py'),
    ('repo/skfolio', 'src/skfolio/descriptor/_value/_cash_flow_to_price.py'),
    ('repo/skfolio', 'src/skfolio/descriptor/_profitability/_cash_flow_to_assets.py'),
    ('repo/qlib', 'qlib/contrib/model/linear.py'))
TABLES = ('financial_annual', 'financial_quarterly')
FAILED = Path('data/staging/strategies-batch63/20261007-SVR-value-financial')
OFFLINE_FAILURE = Path('data/staging/strategies-batch63/20261007-SVR-value-offline-failure')
RAW = {'log_RD': ('开发支出', 'FS_Combas', 'A001219000'),
    'investing_cashflow': ('投资活动产生的现金流量净额', 'FS_Comscfd', 'C002000000'),
    'current_assets': ('流动资产合计', 'FS_Combas', 'A001100000'),
    'current_liabilities': ('流动负债合计', 'FS_Combas', 'A002100000')}
COLUMNS = ['instrument', 'source_row', 'source_sha256', 'report_period', 'version_evidence',
    'total_assets', 'total_liabilities', 'net_profit_ytd', 'operating_cashflow_ytd', *[r[0] for r in RAW.values()]]
METRICS = ['log_NC', 'LEV', 'NI_p', 'NI_n', 'log_RD', 'CR', 'FCF']
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch63.py', 'tests/unit/test_batch63_research.py',
    'src/observe/data/financial_schema.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py', 'source-reviews/review.json',
    'fixture-failure-binding.json', 'offline-failure-binding.json',
    'existing-apis.json', 'dependency-inventory.json', 'supplement-decision.json', 'financial-fields.parquet',
    'financial-input.parquet', 'financial-input-binding.json', 'component-research.json', 'component-values.parquet',
    'diagnostics.json', 'component-offline.json', 'diagnostics-offline.json', 'offline-catalog.json', 'offline-verification.json'}


def implementation(): return {name: file_sha(name) for name in FILES}


def binding(root, directory=None):
    receipt = read(RECEIPT); proof = read(ACCEPTED / 'final-binding-verification.json')
    require(receipt['status'] == proof['status'] == 'ok' and receipt['snapshot'] == SNAPSHOT and
        file_sha(RECEIPT) == proof['receipt_sha256'] and file_sha(ACCEPTED / 'checked-state.json') == proof['checked_state_sha256'] and
        file_sha('scripts/review_strategy_batch62.py') == proof['final_script_sha256'], 'Batch62 changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(RECEIPT), 'proof_sha256': file_sha(ACCEPTED / 'final-binding-verification.json'),
        'upstream': previous.binding(root, ACCEPTED), 'not_a_backtest': True}
    if directory is not None: require(result == read(directory / 'input-binding.json'), 'Input binding changed')
    return result


def preflight(root, directory):
    bound = binding(root)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch63 SVR/value original source research started')
    save(directory / 'input-binding.json', bound)
    shutil.copyfile('src/observe/strategy_catalog.py', directory / 'strategy_catalog.before.py')
    shutil.copyfile(Path(root) / 'catalog/strategies/latest.json', directory / 'catalog-before.json')
    return {'status': 'ok'}


def recover(root, directory):
    binding(root, FAILED); validate_sources(FAILED)
    require(not directory.exists() and not (FAILED / 'offline-verification.json').exists(), 'Recovery exists or original offline passed')
    files = {str(p): file_sha(p) for folder in (FAILED, OFFLINE_FAILURE) for p in sorted(folder.rglob('*')) if p.is_file()}
    red = read(OFFLINE_FAILURE / 'red-regression.json')
    require(red['returncode'] == 1 and '1 failed' in red['stdout'] and
        red['driver_sha256'] == files[str(OFFLINE_FAILURE / 'driver.original.py')] and
        red['test_sha256'] == files[str(OFFLINE_FAILURE / 'regression.original.py')], 'Missing-metadata regression changed')
    directory.mkdir()
    for name in ('baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
            'fixture-failure-binding.json', 'existing-apis.json', 'dependency-inventory.json', 'supplement-decision.json',
            'financial-fields.parquet', 'financial-input.parquet', 'financial-input-binding.json', 'implementation-evidence.before.json'):
        shutil.copyfile(FAILED / name, directory / name)
    for name in ('source-reviews', 'references'): shutil.copytree(FAILED / name, directory / name)
    save(directory / 'offline-failure-binding.json', {'files': files, 'failed_directory': str(FAILED),
        'reason': 'Missing dictionary metadata NaN is not equal to itself; normalize new JSON evidence to null',
        'frozen_inputs_reused': True, 'not_a_backtest': True})
    return archive(root, 'Batch63 offline NaN metadata failure retained; separate null-metadata recovery begun')


def source_tree(directory, number):
    row = read(directory / 'source-reviews/review.json')['sources'][number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number] and row['code_start_line'] == 9, 'Source copy/start changed')
    return ast.parse('\n'.join(read_source(Path(row['source_copy']))[0].splitlines()[8:]))


def api_evidence():
    import baostock as bs
    import sklearn
    rows = []
    for fn in (bs.query_profit_data, bs.query_balance_data, bs.query_cash_flow_data, bs.query_growth_data, bs.query_stock_industry):
        code = inspect.getsource(fn)
        rows.append({'function': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest()})
    return {'baostock': bs.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'apis': rows,
        'SVR_parameters': SVR(kernel='rbf', gamma=.1).get_params(), 'SVR_source': inspect.getsource(SVR),
        'SVR_installed_source_sha256': file_sha(inspect.getsourcefile(SVR)),
        'limits': ['Quarterly ratios/current industry are not whole-pool historical platform fields/vintages',
            'Original 2-D target is accepted with warning; no different model or tuned parameters substituted']}


def financial_projection(root):
    store = Store(root); state = store.state(SNAPSHOT); pieces = []
    for table in TABLES:
        frame = store.load_state(state, table, columns=COLUMNS); frame.insert(0, 'archive_table', table); pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    availability = store.load_state(state, 'financial_availability', columns=['strict_usable', 'version_evidence'])
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'financial_rows': {t: sum(r['rows'] for r in state['tables'].get(t, {}).values()) for t in TABLES},
        'availability_rows': len(availability), 'strict_usable_rows': int(availability.strict_usable.sum()),
        'version_evidence': sorted(availability.version_evidence.unique()), 'historical_equity_dependencies': dependency_state(root),
        'historical_industry_table_registered': 'industry_members' in state['tables'],
        'historical_index_members_table_registered': 'index_constituents' in state['tables'],
        'ledger_sha256': file_sha('src/observe/ledger/book.py'), 'not_a_backtest': True,
        'limits': ['Annual controls market value cannot replace daily total/circulating market capitalization',
            'Final merged amounts do not establish decision-time versions or source report-type equivalence']}


def supplements():
    path = Path('data/staging/strategies-batch31/20261006-capm-weekly-rsrs-finance/probe-results.json')
    receipt = read('docs/handoff/2026-10-06-batch31-verification.json')
    require(file_sha(path) == receipt['checks']['evidence_sha256']['probe-results.json'], 'Accepted financial supplement changed')
    rows = read(path)['results']; require(all(r['status'] == 'failed' and not r['files'] and not r['published'] for r in rows), 'Prior failure changed')
    return {'prior_file': str(path), 'sha256': file_sha(path), 'accepted_failed_probes': rows, 'new_requests': 0, 'new_samples': 0,
        'published': False, 'not_a_backtest': True, 'decision': 'Reuse verified BaoStock login and same-host DNS failures; inspect existing final amounts without admitting versions',
        'limits': ['A failed attempt does not prove supplier absence; latest ratios cannot repair historical report revision/industry gaps']}


def start(root, directory):
    binding(root, directory); folder = directory / 'source-reviews'; folder.mkdir(); rows = []
    failure = Path('data/staging/strategies-batch63/20261007-SVR-value-fixture-failure')
    save(directory / 'fixture-failure-binding.json', {p.name: {'file': str(p), 'sha256': file_sha(p)} for p in sorted(failure.iterdir()) if p.is_file()})
    prior = Path(read(directory / 'catalog-before.json')['directory']) / 'catalog.json'; old = {r['path']: r for r in read(prior)}
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        path = Path('repo/量化策略源代码') / name; text, encoding = read_source(path)
        require(old[name]['review_status'] != '人工审查完成' and file_sha(path) == old[name]['bytes_sha256'] == sha, 'Source changed/reviewed')
        copied = folder / (strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha, 'encoding': encoding,
            'complete_lines_read': len(text.splitlines()), 'code_start_line': 9, 'review': REVIEWS[name], 'newly_reviewed': True,
            'definition_ast': definitions(ast.parse('\n'.join(text.splitlines()[8:])))})
    folder = directory / 'references'; folder.mkdir(); refs = []
    for repository, name in REFERENCES:
        path = Path(repository) / name; copied = folder / path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Financial field/API, ratio and train/test comparison; no platform, model or ledger replacement'})
    store = Store(root); fields = store.load('financial_fields', snapshot=SNAPSHOT)
    fields.to_parquet(directory / 'financial-fields.parquet', index=False)
    financial_projection(root).to_parquet(directory / 'financial-input.parquet', index=False)
    state = store.state(SNAPSHOT)
    save(directory / 'financial-input-binding.json', {'snapshot': SNAPSHOT, 'columns': COLUMNS,
        'projection_sha256': file_sha(directory / 'financial-input.parquet'), 'dictionary_sha256': file_sha(directory / 'financial-fields.parquet'),
        'state_entries': {t: state['tables'][t] for t in (*TABLES, 'financial_fields')},
        'partition_protection': 'Snapshot references plus baseline size/mtime; no fresh full financial partition hash',
        'strict_usable': False, 'not_a_backtest': True})
    save(directory / 'existing-apis.json', api_evidence()); save(directory / 'dependency-inventory.json', inventory(root)); save(directory / 'supplement-decision.json', supplements())
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output']) / 'catalog.json')
    require(len(current) == len(old) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog change')
    save(directory / 'source-reviews/review.json', {'snapshot': SNAPSHOT, 'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'not_a_backtest': True, 'strategy_results': []})
    return archive(root, 'Batch63 two complete SVR/value sources and final financial operands frozen; historical trading remains gated')


def validate_sources(directory):
    if (directory / 'offline-failure-binding.json').exists():
        failure = read(directory / 'offline-failure-binding.json')
        for name, sha in failure['files'].items(): require(file_sha(name) == sha, 'Original offline failure changed')
        for name in ('financial-input.parquet', 'financial-fields.parquet', 'baseline.json', 'financial-input-binding.json'):
            require(file_sha(directory / name) == failure['files'][str(FAILED / name)], 'Recovered frozen input changed')
    doc = read(directory / 'source-reviews/review.json')
    require(doc['snapshot'] == SNAPSHOT and doc['not_a_backtest'] is True and len(doc['sources']) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Source scope/catalog changed')
    for i, (name, sha, row) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], strict=True)):
        text, encoding = read_source(Path(row['source_copy']))
        require(row['source_path'] == str(Path('repo/量化策略源代码') / name) and row['review'] == REVIEWS[name] and
            row['source_sha256'] == file_sha(row['source_copy']) == file_sha(row['source_path']) == sha and
            row['encoding'] == encoding and row['complete_lines_read'] == len(text.splitlines()) and
            row['definition_ast'] == definitions(source_tree(directory, i)), 'Source/rule/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'Reference scope changed')
    for (repository, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repository) / name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
        require(subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip() == row['commit'], 'Reference HEAD changed')
    bound = read(directory / 'financial-input-binding.json')
    require(bound['snapshot'] == SNAPSHOT and bound['columns'] == COLUMNS and bound['strict_usable'] is False and
        bound['projection_sha256'] == file_sha(directory / 'financial-input.parquet') and
        bound['dictionary_sha256'] == file_sha(directory / 'financial-fields.parquet'), 'Financial projection changed')
    failure = read(directory / 'fixture-failure-binding.json')
    require(set(failure) == {'driver.original.py', 'regression.original.py', 'red-regression.json'}, 'Failure scope changed')
    for row in failure.values(): require(file_sha(row['file']) == row['sha256'], 'Failure evidence changed')
    red = read(failure['red-regression.json']['file'])
    require(red['returncode'] == 1 and '1 failed, 3 passed' in red['stdout'] and 'circulating_market_cap' in red['stdout'] and
        red['driver_sha256'] == failure['driver.original.py']['sha256'] and red['test_sha256'] == failure['regression.original.py']['sha256'], 'Red fixture binding changed')


def selected(directory, number, names, ns):
    nodes = [n for n in source_tree(directory, number).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names) and all(not n.decorator_list and
        all(isinstance(d, ast.Constant) for d in n.args.defaults) and
        all(d is None or isinstance(d, ast.Constant) for d in n.args.kw_defaults) for n in nodes), 'Unsafe or missing function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-SVR-value-functions>', 'exec'), ns)
    return ns


def trade_body(tree):
    trade = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trade')
    require(len(trade.body) == 1 and isinstance(trade.body[0], ast.If), 'Original trade shape changed')
    return trade.body[0].body


def run_nodes(nodes, ns):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-financial-arithmetic>', 'exec'), ns)
    return ns


def component_frame(frame, trees):
    body = trade_body(trees[0]); query_node = next(n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'q' for t in n.targets))
    args = query_node.value.func.value.args
    ns = {'np': np, 'balance': SimpleNamespace(total_assets=frame.total_assets, total_liability=frame.total_liabilities),
        'income': SimpleNamespace(net_profit=frame.net_profit_ytd)}
    raw = {name: eval(compile(ast.Expression(args[k]), '<original-query-arithmetic>', 'eval'), ns)
        for name, k in (('log_NC', 2), ('LEV', 3), ('NI_p', 4), ('NI_n', 5))}
    parsed = {}
    for name, (column, _, _) in RAW.items():
        value, bad = parse_number(frame[column]); require(not bad.any(), 'Unparsed raw financial amount'); parsed[name] = value
    raw['log_RD'] = parsed['log_RD']; ns['df'] = pd.DataFrame(raw)
    transforms = [n for n in body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and
        isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == 'df' and
        isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value in ('log_NC', 'NI_p', 'NI_n', 'log_RD')]
    fill = [n for n in body if isinstance(n, ast.Assign) and ast.unparse(n) in
        ('df = df.fillna(0)', 'df[df > 10000] = 10000', 'df[df < -10000] = -10000')]
    require(len(transforms) == 4 and len(fill) == 3, 'Original transformation scope changed')
    run_nodes(transforms+fill, ns); result = ns['df']
    stock = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == 'get_stock_list')
    cr = next(n for n in stock.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and ast.unparse(n.targets[0]) == "df_cr['cr']")
    cr_input = pd.DataFrame({'total_current_assets': parsed['current_assets'], 'total_current_liability': parsed['current_liabilities']})
    cr_input = cr_input[cr_input.total_current_liability != 0]
    result['CR'] = run_nodes([cr], {'df_cr': cr_input})['df_cr']['cr'].reindex(frame.index)
    loop = next(n for n in stock.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(1, 6)')
    fc = loop.body[1].body[0]
    require(isinstance(fc, ast.Assign) and ast.unparse(fc.targets[0]) == "df['FCF']", 'Original FCF shape changed')
    fc_input = pd.DataFrame({'net_operate_cash_flow': frame.operating_cashflow_ytd, 'net_invest_cash_flow': parsed['investing_cashflow']})
    result['FCF'] = run_nodes([fc], {'df': fc_input})['df']['FCF']
    flags = pd.DataFrame({f'input_missing_{n}': v.isna() for n, v in raw.items()})
    flags['input_nonpositive_log_NC'] = raw['log_NC'].le(0)
    flags['input_nonpositive_log_RD'] = raw['log_RD'].le(0)
    flags['input_NI_n_negative_branch'] = raw['NI_n'].lt(0)
    flags['input_zero_liability'] = frame.total_liabilities.eq(0)
    flags['input_zero_current_liabilities'] = parsed['current_liabilities'].eq(0)
    return result, flags, {n: ast.dump(node, include_attributes=False) for n, node in [('query', query_node), ('CR', cr), ('FCF', fc)]}, parsed


def scalar_log(value):
    if pd.isna(value) or value < 0: return 0.
    if value == 0: return -10000.
    return min(10000., max(-10000., math.log(value)))


def reference_frame(frame, parsed):
    out = {name: [] for name in METRICS}
    for a, l, p, rd, ca, cl, oc, ic in zip(frame.total_assets, frame.total_liabilities, frame.net_profit_ytd,
            parsed['log_RD'], parsed['current_assets'], parsed['current_liabilities'], frame.operating_cashflow_ytd,
            parsed['investing_cashflow'], strict=True):
        net = float(Decimal.from_float(a)-Decimal.from_float(l)) if pd.notna(a) and pd.notna(l) else float('nan')
        lev = float(Decimal.from_float(a)/Decimal.from_float(l)) if pd.notna(a) and pd.notna(l) and l != 0 else (math.copysign(float('inf'), a) if pd.notna(a) and a != 0 and l == 0 else float('nan'))
        out['log_NC'].append(scalar_log(net)); out['LEV'].append(0. if np.isnan(lev) else min(10000., max(-10000., lev)))
        out['NI_p'].append(scalar_log(abs(p)) if pd.notna(p) else 0.)
        shifted = p+1; out['NI_n'].append(scalar_log(abs(shifted)) if pd.notna(shifted) and shifted < 0 else 0.)
        out['log_RD'].append(scalar_log(rd))
        out['CR'].append(float(Decimal.from_float(ca)/Decimal.from_float(cl)) if pd.notna(ca) and pd.notna(cl) and cl != 0 else float('nan'))
        out['FCF'].append(float(Decimal.from_float(oc)-Decimal.from_float(ic)) if pd.notna(oc) and pd.notna(ic) else float('nan'))
    return pd.DataFrame(out, index=frame.index)


def frame_digest(frame):
    return hashlib.sha256(frame.to_json(orient='split', date_format='iso', double_precision=15).encode()).hexdigest()


def component(directory):
    validate_sources(directory); frame = pd.read_parquet(directory / 'financial-input.parquet')
    fields = pd.read_parquet(directory / 'financial-fields.parquet')
    require(not frame.duplicated(['archive_table', 'source_sha256', 'source_row']).any() and
        set(frame.version_evidence) == {'final_merged_revision_unknown'}, 'Financial identity/version changed')
    evidence = []
    for table in TABLES:
        for column, origin, identifier in RAW.values():
            rows = fields[(fields.table == table) & (fields.column == column)]
            require(len(rows) == 1 and rows.iloc[0].unit == '元' and rows.iloc[0].source_table == origin and
                rows.iloc[0].source_field == identifier and rows.iloc[0].joinquant_equivalence == 'not_proven', 'Financial field evidence changed')
            evidence.extend(rows.astype(object).where(rows.notna(), None).to_dict('records'))
        for name in ('total_assets', 'total_liabilities', 'net_profit_ytd', 'operating_cashflow_ytd'):
            rows = fields[(fields.table == table) & (fields.standard_name == name)]
            require(len(rows) == 1 and rows.iloc[0].unit == '元' and rows.iloc[0].joinquant_equivalence == 'not_proven', 'Alias unit/equivalence changed')
            evidence.extend(rows.astype(object).where(rows.notna(), None).to_dict('records'))
    with np.errstate(all='ignore'):
        actual, flags, nodes, parsed = component_frame(frame, [source_tree(directory, i) for i in range(2)])
        expected = reference_frame(frame, parsed)
    require(np.array_equal(actual.isna(), expected.isna()), 'Missing mask differs')
    errors = {}
    for name in METRICS:
        good = expected[name].notna(); a = actual.loc[good, name].to_numpy(); b = expected.loc[good, name].to_numpy()
        require(np.allclose(a, b, rtol=1e-12, atol=1e-12), 'Independent financial arithmetic differs')
        errors[name] = float(np.max(np.abs(a-b))) if len(a) else 0.
    identity = frame[['archive_table', 'instrument', 'source_sha256', 'source_row', 'report_period', 'version_evidence']].copy()
    values = pd.concat([identity, actual, flags], axis=1)
    studies = []
    for table in TABLES:
        mask = frame.archive_table.eq(table); part = values[mask]
        studies.append({'table': table, 'rows': int(mask.sum()), 'first_report': str(frame.loc[mask, 'report_period'].min().date()),
            'last_report': str(frame.loc[mask, 'report_period'].max().date()),
            'finite': {n: int(np.isfinite(actual.loc[mask, n]).sum()) for n in METRICS},
            'unavailable': {n: int(actual.loc[mask, n].isna().sum()) for n in METRICS},
            'input_flags': {n: int(flags.loc[mask, n].sum()) for n in flags}, 'values_fingerprint': frame_digest(part)})
    return {'snapshot': SNAPSHOT, 'rows': len(frame), 'studies': studies, 'metrics': METRICS, 'arithmetic_ast': nodes,
        'max_abs_error': errors, 'unit_evidence': evidence, 'strict_usable': False, 'not_a_backtest': True,
        'platform_equivalent': False, 'original_strategy_complete': False, 'cost_backtests': [], 'real_model_fits': 0,
        'limits': ['Final report values only; report_period is not decision availability, no strict historical records admitted',
            'Original arithmetic uses Yuan inputs; platform units, single-quarter semantics and report types remain unproved',
            'Independent Decimal operates on parsed binary floats; no original supplier decimal precision claim',
            'No market cap/growth/ROE/industry filled or invented; no full SVR vector, whole-pool selection or trading built',
            'Original NaN->0 and infinities->+/-10000 are retained only for the five observed SVR components; raw missing flags kept',
            'FCF is operating minus signed investing cash flow, not independently verified capital-expenditure FCF']}, values


def query_fixture():
    code = SimpleNamespace(in_=lambda *a: None)
    return {'query': lambda *a: SimpleNamespace(filter=lambda *a: None),
        'valuation': SimpleNamespace(code=code, market_cap=1., circulating_cap=1., circulating_market_cap=1., pe_ratio=1.),
        'balance': SimpleNamespace(total_assets=1., total_liability=1., development_expenditure=1., total_current_assets=1., total_current_liability=1.),
        'income': SimpleNamespace(net_profit=1., statDate=1., pubDate=1.),
        'indicator': SimpleNamespace(inc_revenue_year_on_year=1., roe=1., eps=1.),
        'cash_flow': SimpleNamespace(code=code, statDate=1., net_operate_cash_flow=1., net_invest_cash_flow=1.)}


def diagnostics(directory):
    validate_sources(directory); rows = []; trees = [source_tree(directory, i) for i in range(2)]
    ns = selected(directory, 0, ['trade', 'set_params'], {'pd': pd, 'np': np, 'SVR': SVR, 'g': SimpleNamespace(), **query_fixture()})
    ns['set_params'](); ns['get_index_stocks'] = lambda *a, **kw: ['A']; ns['get_industry_stocks'] = lambda *a, **kw: ['A']
    ns['get_fundamentals'] = lambda *a, **kw: pd.DataFrame([['A', 100., 50., 2., 10., 11., 20., 1.]])
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={}, cash=1000., available_cash=1000.),
        current_dt=pd.Timestamp('2024-01-01'), previous_date=pd.Timestamp('2023-12-29'))
    try: ns['trade'](context)
    except TypeError as exc: rows.append({'case': 'original_SVR_set_indexer_fails_before_fit', 'error': type(exc).__name__, 'message': str(exc), 'days': ns['g'].days})
    else: raise ValueError('Expected set indexer failure')
    body = trade_body(trees[0]); sort = next(n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'factor' for t in n.targets) and isinstance(n.value, ast.Call))
    try: run_nodes([sort], {'factor': pd.DataFrame({'log_mcap': [1., -1.]})})
    except TypeError as exc: rows.append({'case': 'original_SVR_sort_index_by_fails', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected old sort failure')
    fitting = [n for n in body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('svr', 'model')]
    residual = next(n for n in body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'factor')
    fixture = {'SVR': SVR, 'pd': pd, 'X': pd.DataFrame(np.arange(680.).reshape(20, 34)/100), 'Y': pd.DataFrame({'log_mcap': np.linspace(1., 3., 20)})}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); run_nodes(fitting+[residual], fixture)
    require(fixture['model'] is fixture['svr'], 'Original fit identity changed')
    np.testing.assert_allclose(fixture['factor'].to_numpy().ravel(), fixture['Y'].to_numpy().ravel()-fixture['svr'].predict(fixture['X']))
    rows.append({'case': 'original_SVR_in_sample_log_mcap_residual_synthetic_fit_only', 'rows': 20, 'features': 34,
        'parameters': fixture['svr'].get_params(), 'warning_classes': [type(w.message).__name__ for w in caught], 'real_model_fits': 0})
    ns['g'].days = 1; ns['trade'](context)
    rows.append({'case': 'days_count_is_callback_count_refresh_rate_unused', 'after_skipped_callback': ns['g'].days})
    value = selected(directory, 1, ['buy', 'sell', 'before_market_open', 'get_check_stocks_sort', 'get_data', 'winsorize'],
        {'pd': pd, 'np': np, 'g': SimpleNamespace(), 'log': SimpleNamespace(info=lambda *a: None), **query_fixture()})
    orders = []; value['order_value'] = lambda *a: orders.append(list(a)); value['order_target_value'] = lambda *a: orders.append(list(a))
    context.portfolio.positions = {'A': object(), 'old': object()}; value['sell'](context, ['A', 'B']); value['buy'](context, ['A', 'B'])
    require(orders == [['old', 0], ['A', 500.], ['B', 500.]], 'Original held-stock repeated cash buys changed')
    rows.append({'case': 'value_buys_held_and_new_with_equal_available_cash_budgets', 'mock_orders': orders, 'fills': 0})
    calls = []; value['get_stock_list'] = lambda *a: calls.append('selection') or ['A']; value['get_check_stocks_sort'] = lambda *a: ['A']
    value['before_market_open'](context); require(calls == ['selection', 'selection'], 'Original duplicate selection changed')
    rows.append({'case': 'value_before_open_repeats_entire_selection', 'calls': calls})
    value = selected(directory, 1, ['get_check_stocks_sort', 'get_data'], {'pd': pd, 'np': np, **query_fixture()})
    value['get_fundamentals'] = lambda *a, **kw: pd.DataFrame({'circulating_cap': [1.], 'pe_ratio': [1.], 'code': ['A']})
    try: value['get_check_stocks_sort'](context, ['A'])
    except AttributeError as exc: rows.append({'case': 'value_dataframe_sort_removed', 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected DataFrame.sort failure')
    stock = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == 'get_stock_list')
    loop = next(n for n in stock.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(1, 6)')
    local = {**query_fixture(), 'pd': pd, 'context': context, 'l4': {}, 'y': 2024, 'get_fundamentals': lambda *a, **kw:
        pd.DataFrame({'code': ['B' if kw['statDate'] == '2019' else 'A'], 'net_operate_cash_flow': [10.], 'net_invest_cash_flow': [0.]})}
    run_nodes([loop], local); require(set(local['l4']) == {'B'}, 'Original annual overwrite differs')
    rows.append({'case': 'five_year_FCF_keeps_only_oldest_nonempty_year_not_intersection', 'selected': sorted(local['l4']), 'five_year_intersection': []})
    local.update(l4={}, get_fundamentals=lambda *a, **kw: pd.DataFrame() if kw['statDate'] == '2019' else
        pd.DataFrame({'code': ['A'], 'net_operate_cash_flow': [10.], 'net_invest_cash_flow': [0.]}))
    run_nodes([loop], local); rows.append({'case': 'missing_oldest_annual_report_is_skipped', 'selected': sorted(local['l4'])})
    requests = []
    def fundamentals(*a, **kw):
        if 'statDate' in kw: requests.append(kw['statDate']); return pd.DataFrame({'code': ['A']})
        return pd.DataFrame({'code': ['A'], 'statDate': ['2025-03-31'], 'pubDate': ['2025-04-30']})
    value['get_fundamentals'] = fundamentals
    try: value['get_data'](['A'], 2)
    except AttributeError as exc:
        require(requests == ['2025q1.0', '2025q1.-1'], 'Original float quarter suffix changed')
        rows.append({'case': 'float_quarter_suffix_and_removed_Panel', 'requests': requests, 'error': type(exc).__name__, 'message': str(exc)})
    else: raise ValueError('Expected Panel failure')
    eps_loop = next(n for n in stock.body if isinstance(n, ast.For) and any(isinstance(x, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == 'df_6' for t in x.targets) for x in n.body))
    rule = eps_loop.body[1]
    result = run_nodes([rule], {'df_6': pd.DataFrame({'eps': [.08, .081, .499, .5]}, index=list('ABCD'))})['df_temp']
    require(list(result.index) == ['B', 'C'], 'Original EPS strict boundaries differ')
    rows.append({'case': 'EPS_level_not_growth_strict_008_05', 'selected': list(result.index)})
    gates = [n for n in stock.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('df_mkt', 'df_cr') for t in n.targets)]
    rows.append({'case': 'market_cap_and_current_ratio_use_slot3_legacy_Panel_order_unproved',
        'assignment_ast': [ast.dump(n, include_attributes=False) for n in gates], 'chronological_order_proved': False})
    return {'cases': rows, 'synthetic_fixture': True, 'real_historical_operand_windows': 0, 'synthetic_model_fits': 1, 'real_model_fits': 0,
        'not_a_backtest': True, 'platform_equivalent': False, 'limits': ['Original selected AST and query/order fixtures; no repaired selector or platform fills',
            'SVR fitted only to synthetic 20x34 fixture, not historical predictive or selection evidence']}


def study(root, directory):
    binding(root, directory); before = implementation(); result, values = component(directory); diag = diagnostics(directory)
    require(before == implementation() and not (directory / 'component-values.parquet').exists(), 'Study changed or exists')
    if (directory / 'offline-failure-binding.json').exists():
        pd.testing.assert_frame_equal(values, pd.read_parquet(FAILED / 'component-values.parquet'), check_exact=True)
    values.to_parquet(directory / 'component-values.parquet', index=False); save(directory / 'component-research.json', result); save(directory / 'diagnostics.json', diag)
    return archive(root, 'Batch63 real final financial arithmetic and original SVR/value defects archived; no trading or historical model fitting')


def offline(root, directory):
    binding(root, directory); before = implementation()
    def denied(*a, **kw): raise ValueError('Batch63 attempted network')
    with patch.object(socket, 'socket', denied), patch.object(socket, 'create_connection', denied), redirect_stdout(StringIO()):
        comp, values = component(directory); diag = diagnostics(directory); catalog = offline_catalog(root, directory); dep = inventory(root)
        pd.testing.assert_frame_equal(financial_projection(root), pd.read_parquet(directory / 'financial-input.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(Store(root).load('financial_fields', snapshot=SNAPSHOT), pd.read_parquet(directory / 'financial-fields.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    save(directory / 'component-offline.json', comp); save(directory / 'diagnostics-offline.json', diag)
    require(before == implementation() and comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        dep == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json'), 'Offline evidence differs')
    save(directory / 'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory / 'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True, 'implementation_sha256': before,
        'component_sha256': file_sha(directory / 'component-research.json'), 'values_sha256': file_sha(directory / 'component-values.parquet'),
        'diagnostics_sha256': file_sha(directory / 'diagnostics.json'), 'not_a_backtest': True})
    return archive(root, 'Batch63 forbidden-network amounts/values/diagnostics and three catalogs match; strict financial gate unchanged')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch63.py',
        'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch63_research.py', 'tests/unit/test_batch62_research.py',
            'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    for k, command in enumerate(commands()):
        path = directory / f'check-{k}.log'
        with path.open('x') as out: result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)}); require(result.returncode == 0, f'Check failed: {path}')
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
        [r['command'] for r in checked['commands']] == commands() and all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Checked state changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory / name) == sha, 'Evidence changed')
    for name, sha in checked['implementation_sha256'].items(): require(file_sha(directory / 'implementation-checked' / name) == sha, 'Checked copy changed')
    require(off['implementation_sha256'] == implementation() and off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] is True and
        off['component_sha256'] == file_sha(directory / 'component-research.json') == file_sha(directory / 'component-offline.json') and
        off['diagnostics_sha256'] == file_sha(directory / 'diagnostics.json') == file_sha(directory / 'diagnostics-offline.json') and off['values_sha256'] == file_sha(directory / 'component-values.parquet'), 'Offline proof changed')
    comp, values = component(directory); diag = diagnostics(directory); pd.testing.assert_frame_equal(values, pd.read_parquet(directory / 'component-values.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(financial_projection(root), pd.read_parquet(directory / 'financial-input.parquet'), check_exact=True)
    require(comp == read(directory / 'component-research.json') and diag == read(directory / 'diagnostics.json') and
        inventory(root) == read(directory / 'dependency-inventory.json') and supplements() == read(directory / 'supplement-decision.json') and
        api_evidence() == read(directory / 'existing-apis.json'), 'Recomputed evidence changed')
    require(Store(root).published() == read(directory / 'baseline.json')['published'], 'Publication changed')
    protection = protect(root, directory); progress = archive(root, 'Batch63 SVR/value financial research accepted; both original trades remain gated')
    receipt = Path('docs/handoff/2026-10-07-batch63-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory / 'source-reviews/review.json'), 'checks': checked, 'offline': off,
        'offline_catalog': read(directory / 'offline-catalog.json'), 'component': comp, 'diagnostics': diag, 'protection': protection, 'progress': progress,
        'supplements': read(directory / 'supplement-decision.json'), 'not_a_backtest': True, 'original_strategy_complete': False, 'newly_reviewed': 2})
    save(directory / 'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt), 'checked_state_sha256': file_sha(directory / 'checked-state.json'),
        'final_script_sha256': file_sha(__file__), 'catalog_sha256': file_sha(Path(read(directory / 'source-reviews/review.json')['catalog']['output']) / 'catalog.json')})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['preflight', 'start', 'recover', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', default='data/staging/strategies-batch63/20261007-SVR-value-financial-recovery')
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii=False, default=str))
