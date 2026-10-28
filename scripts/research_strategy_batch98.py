"""Review original bull/bear and northbound arithmetic; no orders, returns or NAV."""
import argparse
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.stats as st

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch59 import index_operand
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch98/20261011-original-components-v1')
PARENT = Path('docs/handoff/2026-10-11-batch97-final-review.json')
PARENT_SHA = 'e953d027c19032ef9957a48bbded6fa03aedf9cd03621b1686586643fc406ea9'
SOURCE_PATHS = (
    '2021年度精选策略/63.【复现】华泰证券-波动率和换手率构建牛熊指标.txt',
    '2023年度精选策略/58.北向RSRS与布林带择时.txt')
SOURCE_SHA = ('9dbf27618f8d0285ad9f7602954875e6dd8a6604e7c87def544b75e6055caac9',
              '1ca06cbc4d95a15dd646bb1aa73b50fbf5503252071f9ed0a1461d08aa3c1e69')
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
INDEX_SHA = '28a1ccbb565895091aa8f1cb865254f5626d3514a7ba035073113d95d8e6c406'
NORTH_RESULT = Path('data/staging/strategies-batch12/20261005-daily-candidates/probes/northbound_flow/result.json')
NORTH_SHA = '03607825ccb35da06ac8068ecdb810702a737268bf463da98e32253cde4c93d7'
OUTPUTS = ('diagnostics.json', 'index-components.parquet', 'summary.json')
FILES = ('scripts/research_strategy_batch98.py', 'tests/unit/test_batch98_research.py',
         'scripts/recover_strategy_batch94_reference.py', 'scripts/review_strategy_batch18.py',
         'scripts/review_strategy_batch59.py', 'scripts/run_strategy_batch4.py',
         'scripts/archive_strategy_progress.py', 'scripts/verify_strategy_batch3.py',
         'src/observe/data/store.py', 'src/observe/data/locks.py', 'src/observe/runs.py',
         'src/observe/strategy_catalog.py', 'pyproject.toml', 'uv.lock')
CLASS_METHODS = {
    'VT_Factor': {'__init__', 'get_turnover', '_Calc_func', '_Cal_forward_ret'},
    'BackTesting': {'__init__', 'Choose_Strategy', 'Strategy_a', 'Strategy_b', 'Strategy_c',
                    'Strategy_MA', 'Strategy_BBANDS', '_Cal_MA', '_Cal_BBANDS',
                    'back_testing', 'summary'}}


def definitions(source, number, namespace=None):
    """Compile reviewed definitions only; credential-bearing top level is never executed."""
    require(number in (0, 1) and file_sha(source) == SOURCE_SHA[number], 'Source SHA differs')
    tree = ast.parse(read_source(source)[0]); selected = []
    if number == 0:
        require(len([n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]) == 6,
                'Source class/function inventory differs')
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name in CLASS_METHODS:
                require(not n.decorator_list and not n.keywords and not n.type_params and
                        len(n.bases) == 1 and isinstance(n.bases[0], ast.Name) and
                        n.bases[0].id == 'object', 'Unreviewed class body')
                n.body = [x for x in n.body if isinstance(x, ast.FunctionDef) and
                          x.name in CLASS_METHODS[n.name]]
                require({x.name for x in n.body} == CLASS_METHODS[n.name], 'Missing selected method')
                selected.append(n)
            elif isinstance(n, ast.FunctionDef) and n.name == 'Statistical_indicator':
                selected.append(n)
    else:
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        require(len(selected) == 9, 'Northbound definitions differ')
    funcs = [f for n in selected for f in (n.body if isinstance(n, ast.ClassDef) else [n])]
    for f in funcs:
        require(not f.returns and not f.type_params and
                all(isinstance(d, ast.Name) and d.id == 'property' for d in f.decorator_list),
                'Unreviewed definition decorator/annotation')
        args = (*f.args.posonlyargs, *f.args.args, *f.args.kwonlyargs,
                *[x for x in (f.args.vararg, f.args.kwarg) if x])
        require(all(a.annotation is None for a in args), 'Executable annotation')
        for d in (*f.args.defaults, *[x for x in f.args.kw_defaults if x is not None]):
            ast.literal_eval(d)
    ns = {'np': np, 'pd': pd, 'math': math, 'st': st, 'g': SimpleNamespace()}
    ns.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def source_namespace(number, **kwargs):
    return definitions(DIRECTORY/'sources'/f'{number}.source', number, kwargs)


def inventory():
    store = Store('data'); state = store.state(SNAPSHOT)
    require(file_sha(Path('data/snapshots')/(SNAPSHOT+'.json')) == SNAPSHOT_SHA and
            state['tables'] == store.published()['tables'], 'Publication/snapshot differs')
    result = read(NORTH_RESULT)
    require(result['status'] == 'success' and result['raw_sha256'] == NORTH_SHA and
            file_sha(result['raw_file']) == NORTH_SHA, 'Archived northbound candidate changed')
    for response in result['wire_responses']:
        require(file_sha(response['file']) == response['sha256'], 'Archived response changed')
    frame = pd.read_parquet(result['raw_file']); rows = []
    for col in ('当日成交净买额', '当日资金流入', '当日余额'):
        finite = frame[col].notna()
        rows.append({'field': col, 'observations': int(finite.sum()),
            'first': str(frame.loc[finite, '日期'].min()), 'last': str(frame.loc[finite, '日期'].max())})
    present = set(state['tables'])
    return {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
        'published_batch': store.published()['batch_id'], 'registered_tables': sorted(present),
        'missing_original_tables': sorted({'index_dailybasic', 'northbound_quota',
            'northbound_holdings', 'index_members', 'fund_redemptions'} - present),
        'index_rate_provider': 'Tushare index_dailybasic turnover_rate_f; source credentials not used',
        'candidate_fields_not_equivalent': ['SW ordinary turnoverrate is not free-float index turnover',
            'NET_DEAL_AMT is not quota_daily minus quota_daily_balance',
            'MUTUAL_TYPE005 aggregate omits both link-specific quotas and quota history'],
        'northbound_candidate': {'result': str(NORTH_RESULT), 'result_sha256': file_sha(NORTH_RESULT),
            'file': result['raw_file'], 'sha256': NORTH_SHA, 'rows': len(frame), 'coverage': rows,
            'strict_usable': False, 'published': False},
        'new_trading_backtests': 0, 'source_credentials_executed_or_exported': False}


def prepare():
    require(not DIRECTORY.exists(), 'Immutable batch directory exists')
    require(file_sha(PARENT) == PARENT_SHA, 'Parent receipt changed')
    parent = read(PARENT)
    require(parent['status'] == 'ok' and parent['progress']['manually_reviewed'] == 280,
            'Parent status differs')
    for name, digest in parent['evidence_sha256'].items():
        require(file_sha(name) == digest, 'Parent evidence differs: '+name)
    sources = []
    for number, name in enumerate(SOURCE_PATHS):
        path = Path('repo/量化策略源代码')/name
        require(file_sha(path) == SOURCE_SHA[number], 'Original source differs')
        text = read_source(path)[0]; tree = ast.parse(text)
        rows = [{'name': n.name, 'kind': type(n).__name__, 'first_line': n.lineno,
                 'last_line': n.end_lineno,
                 'methods': [m.name for m in n.body if isinstance(m, ast.FunctionDef)]}
                for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        definitions(path, number)
        sources.append({'path': name, 'sha256': SOURCE_SHA[number], 'lines': len(text.splitlines()),
                        'definitions': rows, 'top_level_executed': False})
    frame, index = index_operand('data')
    require(len(index['partitions']) == 1 and index['partitions'][0]['sha256'] == INDEX_SHA,
            'Index byte identity differs')
    deps = inventory()
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch98 full-source preread/dependency investigation; not formally accepted')
        (DIRECTORY/'sources').mkdir()
        for number, name in enumerate(SOURCE_PATHS):
            shutil.copyfile(Path('repo/量化策略源代码')/name, DIRECTORY/'sources'/f'{number}.source')
        frame.to_parquet(DIRECTORY/'index.parquet', index=False)
        for name in FILES:
            target = DIRECTORY/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        save(DIRECTORY/'prepared.json', {'parent': str(PARENT), 'parent_sha256': PARENT_SHA,
            'source_inventory': sources, 'index': index,
            'operand_sha256': file_sha(DIRECTORY/'index.parquet'),
            'implementation_sha256': {n: file_sha(n) for n in FILES}})
        save(DIRECTORY/'dependencies.json', deps)
    return {'status': 'prepared', 'sources': 2, 'index_rows': len(frame), 'directory': str(DIRECTORY)}


def canonical(value):
    if isinstance(value, pd.DataFrame): return canonical(value.to_dict('records'))
    if isinstance(value, pd.Series): return canonical(value.tolist())
    if isinstance(value, np.ndarray): return canonical(value.tolist())
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else 'inf' if value > 0 else '-inf'}
    return value


def diagnostics():
    ns = source_namespace(0); vt = ns['VT_Factor']; bt = ns['BackTesting']; cases = []

    def case(name, callback):
        try:
            with redirect_stdout(StringIO()), np.errstate(all='ignore'):
                row = {'case': name, 'value': canonical(callback()), 'error': None}
        except Exception as exc:
            row = {'case': name, 'value': None, 'error': type(exc).__name__}
        cases.append(row)

    data = pd.DataFrame({'pct_chg': [1., 2., 3., 4., 5.], 'turnover_rate_f': [2.]*5,
                         'close': [100., 101., 102., 103., 104.]})
    case('kernel_population_std', lambda: vt('index', '', '', 3)._Calc_func(data))
    case('kernel_zero_denominator', lambda: vt('index', '', '', 3)._Calc_func(
        data.assign(turnover_rate_f=0.)))
    case('kernel_empty_periods', lambda: vt('index', '', '', [])._Calc_func(data))
    case('forward_labels_mixed_units', lambda: vt('index', '', '', 3, [2])._Cal_forward_ret(
        data.assign(kernel_3=1.), 3))
    case('threshold_strict', lambda: bt(data, 'a', None, value=2.).Strategy_b(
        pd.DataFrame({'SINGAL': [1., 2., 3.]})))
    case('interval_inclusive', lambda: bt(data, 'a', None, value=(1., 2.)).Strategy_c(
        pd.DataFrame({'SINGAL': [.99, 1., 2., 2.01]})))
    case('hold_signal_resets_count', lambda: bt(data, 'a', None, hold=3, value=.5).Strategy_a(
        pd.DataFrame({'SINGAL': [1., 0., 1., 0., 0., 0.]})))
    case('single_signal_summary_name_error', lambda: bt(data.assign(a=1.), 'a', None, value=0.).summary)
    case('stat_std_field_is_median', lambda: ns['Statistical_indicator'](
        pd.DataFrame({'x': [1., 2., 9.]}), 'x', False))
    for method in ('MA', 'BBANDS'):
        vals = np.r_[np.linspace(100., 10., 90), np.linspace(10., 100., 90)]
        vals[30] += 200.; vals[100] -= 200.
        frame = pd.DataFrame({'close': vals, 'kernel': vals, 'pct_chg': 0.})
        for names in (['close', 'kernel'], ['kernel', 'close']):
            case('inversion_'+method+'_'+'_'.join(names), lambda m=method, s=names:
                 bt(frame, s, m).back_testing[['close_MARK', 'kernel_MARK']].sum().to_dict())
    # Original pagination is called only against a fake provider, never Tushare or a token.
    dates = pd.date_range('2000-01-01', periods=3001).strftime('%Y%m%d').tolist()
    calls = []
    def dailybasic(**kwargs):
        calls.append([kwargs['start_date'], kwargs['end_date']])
        return pd.DataFrame({'trade_date': [kwargs['start_date'], kwargs['end_date']],
                             'turnover_rate_f': [1., 1.]})
    ns['my_ts'] = SimpleNamespace(trade_cal=lambda **kw:
        pd.DataFrame({'is_open': 1, 'cal_date': dates}), index_dailybasic=dailybasic)
    case('turnover3001_duplicate_boundary', lambda: {
        'duplicated': int(vt('index', dates[0], dates[-1], 250).get_turnover.index.duplicated().sum()),
        'calls': calls})
    north = source_namespace(1); g = north['g']
    g.quantile_N = g.quantile_M = 15; g.N = 18; g.M = 600; g.sell = -.4
    case('quantile17_always_nan', lambda: north['cal_ret_quantile'](
        pd.Series(np.linspace(.001, .02, 17), index=pd.date_range('2020-01-01', periods=17))))
    case('quantile29_current_integer_index_error', lambda: north['cal_ret_quantile'](
        pd.Series(np.linspace(.001, .02, 29))))
    g.ans = [.8, 1., 1.2]; g.ans_rightdev = [.9]*3; g.init = True
    calls_history = []
    def history(stock, *args):
        calls_history.append(stock)
        x = np.arange(1., 19.)
        return pd.DataFrame({'low': x, 'high': x+np.sin(x)*.1+1., 'close': x+2.})
    north['attribute_history'] = history
    case('first_RSRS_zero_no_stock_history', lambda: north['calc_zscore_Rdev']('A'))
    case('second_RSRS_nan_shared_state', lambda: {
        'score': north['calc_zscore_Rdev']('B'), 'shared_length': len(g.ans), 'history_calls': calls_history})
    case('constant_OLS_nonfinite', lambda: north['get_ols']([1.]*18, [2.]*18))
    case('empty_OLS_zero', lambda: north['get_ols']([], []))
    case('date_namespace_unbound', lambda: north['before_market_open'](
        SimpleNamespace(current_dt=pd.Timestamp('2020-01-02'))))
    # Query stubs inject only the already-reviewed tabular operand.
    class Column:
        def in_(self, value): return self
        def __le__(self, value): return self
    class Query:
        def filter(self, *args): return self
        def order_by(self, *args): return self
    table = SimpleNamespace(**{k: Column() for k in ('day', 'quota_daily', 'quota_daily_balance',
        'link_id', 'name', 'code', 'share_ratio')})
    north['query'] = lambda *args: Query()
    g.window = 90; g.stdev_up = g.stdev_dn = 2; g.max_stock_count = 20
    quota = pd.DataFrame({'day': ['d1', 'd1', 'd2', 'd2'], 'quota_daily': [10.]*4,
                          'quota_daily_balance': [9., 8., 7., 6.]})
    north['finance'] = SimpleNamespace(STK_ML_QUOTA=table, STK_HK_HOLD_INFO=table,
                                      run_query=lambda q: quota.copy())
    case('boll_two_days_sample_std_without90', lambda: north['get_boll']('d2'))
    north['finance'].run_query = lambda q: quota.iloc[:0].copy()
    case('boll_empty_error', lambda: north['get_boll']('d2'))
    north['finance'].run_query = lambda q: pd.DataFrame({'code': ['A']*2+[f'C{i}' for i in range(22)],
                                                     'share_ratio': list(range(24))})
    case('rank_excludes_top3_selects17', lambda: north['calc_change'](
        SimpleNamespace(previous_date='d'), []))
    selected = ['A', 'A', 'B']; intents = []
    north['calc_change'] = lambda *a: np.array(selected)
    north['get_index_stocks'] = lambda *a: selected
    north['calc_zscore_Rdev'] = lambda stock: -1.
    north['order_target'] = lambda stock, value: intents.append(['sell', stock, value])
    north['order_value'] = lambda stock, value: intents.append(['buy', stock, value])
    north['redeem'] = lambda stock, value: intents.append(['redeem', stock, value])
    g.security = 'index'; g.fund = '420002.OF'; g.mf = g.upper = 10.; g.lower = -10.
    context = SimpleNamespace(portfolio=SimpleNamespace(positions={'A': object(), 'B': object()}, total_value=1000.))
    case('same_target_set_blocks_RSRS_exit', lambda: (north['reblance'](context), list(intents)))
    north['calc_zscore_Rdev'] = lambda stock: math.nan
    context.portfolio.positions = {}
    case('nan_passes_filter_then_missing_fund_keyerror', lambda: north['reblance'](context))
    context.portfolio.positions = {g.fund: SimpleNamespace(closeable_amount=100.)}
    case('full_asset_per_new_stock_incremental_intents', lambda: (north['reblance'](context),
        sorted(intents, key=lambda x: (x[0], x[1]))))
    return {'case_count': len(cases), 'cases': cases, 'all_synthetic': True,
            'intents_are_not_orders_or_fills': True, 'new_trading_backtests': 0}


def stable_marks(values, method):
    """Independent scalar oracle retains strict previous and inclusive current crossings."""
    marks = []; prev_up = prev_low = math.nan
    for i, x in enumerate(values):
        if method == 'MA':
            up = math.nan if i < 59 else math.fsum(values[i-19:i+1])/20-math.fsum(values[i-59:i+1])/60
            low = up
        else:
            if i < 19: up = low = math.nan
            else:
                window = values[i-19:i+1]; mean = math.fsum(window)/20
                std = math.sqrt(math.fsum((v-mean)**2 for v in window)/19)
                up = x-(mean+2*std); low = x-(mean-2*std)
        marks.append(1 if prev_up < 0 and up >= 0 else -1 if prev_low > 0 and low <= 0 else 0)
        prev_up, prev_low = up, low
    return marks


def stable_ols(x, y):
    require(len(x) == len(y) and len(x) > 1, 'Invalid reference operands')
    mx = math.fsum(x)/len(x); my = math.fsum(y)/len(y)
    xx = math.fsum((a-mx)**2 for a in x); yy = math.fsum((a-my)**2 for a in y)
    xy = math.fsum((a-mx)*(b-my) for a, b in zip(x, y, strict=True))
    return xy/xx, xy*xy/(xx*yy)


def components():
    frame = pd.read_parquet(DIRECTORY/'index.parquet')
    ns = source_namespace(0); north = source_namespace(1)
    data = pd.DataFrame({'close': frame.close.to_numpy()})
    output = frame[['date', 'close']].copy(); summary = {}
    for method in ('MA', 'BBANDS'):
        obj = ns['BackTesting'](data, 'close', method)
        getter = obj._Cal_MA if method == 'MA' else obj._Cal_BBANDS
        marks = np.asarray(getter(data.rename(columns={'close': 'SINGAL'})))
        oracle = np.array(stable_marks(frame.close.to_numpy().tolist(), method))
        output[method+'_mark'] = marks
        output[method+'_independent_mark'] = oracle
        output[method+'_long_state'] = obj.back_testing.close_MARK.to_numpy()
        summary[method] = {'marks_differences': int((marks != oracle).sum()),
            'up_crossings': int((marks == 1).sum()), 'down_crossings': int((marks == -1).sum()),
            'long_state_days': int(output[method+'_long_state'].sum()),
            'complete_windows': len(frame)-(59 if method == 'MA' else 19)}
    rows = []
    for i in range(17, len(frame)):
        x = frame.low.iloc[i-17:i+1].to_numpy(); y = frame.high.iloc[i-17:i+1].to_numpy()
        row = {'date': frame.date.iloc[i]}
        for name, a, b in (('seed_unsquared', x, y), ('stock_squared', x*x, y*y)):
            beta, r2 = north['get_ols'](a, b); independent = stable_ols(a.tolist(), b.tolist())
            row.update({name+'_beta': beta, name+'_r2': r2,
                name+'_beta_error': abs(beta-independent[0]), name+'_r2_error': abs(r2-independent[1])})
        rows.append(row)
    output = output.merge(pd.DataFrame(rows), on='date', how='left')
    rsrs = {name: float(output[name].max()) for name in output if name.endswith('_error')}
    require(all(v < 1e-10 for v in rsrs.values()), 'OLS arithmetic differs')
    return output, {'index': '000300.SH', 'rows': len(frame), 'first': str(frame.date.min()),
        'last': str(frame.date.max()), 'comparison_price_leg': summary, 'OLS_windows': len(rows),
        'OLS_max_absolute_errors': rsrs, 'original_seed_skips_first_complete_window': True,
        'scopes': ['MA/BBANDS marks are original close comparison leg, not bull/bear kernel',
            'Squared OLS on index is a formula reference, not original individual stock RSRS',
            'No quota/holding ratios or per-stock shared history; no original portfolio signal',
            'No economic/numeric variant applied; independent discrepancies must remain visible'],
        'new_trading_backtests': 0, 'three_cost_scenarios': 'not_applicable_to_arithmetic',
        'original_strategy_complete': False}


def worker(mode):
    prepared = read(DIRECTORY/'prepared.json')
    require(file_sha(DIRECTORY/'index.parquet') == prepared['operand_sha256'], 'Operand differs')
    for name, digest in prepared['implementation_sha256'].items():
        require(file_sha(DIRECTORY/'implementation'/name) == digest and file_sha(name) == digest,
                'Worker code differs: '+name)
    dest = DIRECTORY/mode; require(not dest.exists(), 'Immutable worker output exists')
    with network_blocked():
        diag = diagnostics(); frame, summary = components()
        dest.mkdir()
        save(dest/'diagnostics.json', diag)
        frame.to_parquet(dest/'index-components.parquet', index=False)
        summary['synthetic_diagnostic_cases'] = diag['case_count']
        save(dest/'summary.json', summary)
    return {'status': 'ok', 'mode': mode, 'outputs_sha256': {n: file_sha(dest/n) for n in OUTPUTS}}


def run(mode):
    folder = DIRECTORY/(mode+'-command'); folder.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.research_strategy_batch98', 'worker', '--mode', mode]
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (folder/'stdout.log').write_text(process.stdout, encoding='utf-8')
    (folder/'stderr.log').write_text(process.stderr, encoding='utf-8')
    save(folder/'command.json', {'command': command, 'returncode': process.returncode,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')})
    require(process.returncode == 0, 'Worker failed; stderr retained')
    return json.loads(process.stdout.splitlines()[-1])


def offline():
    a = read(DIRECTORY/'study-command/command.json'); b = read(DIRECTORY/'replay-command/command.json')
    require(a['returncode'] == b['returncode'] == 0, 'Worker command failed')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study/index-components.parquet'),
                                  pd.read_parquet(DIRECTORY/'replay/index-components.parquet'), check_exact=True)
    require(all(file_sha(DIRECTORY/'study'/n) == file_sha(DIRECTORY/'replay'/n) for n in OUTPUTS),
            'Offline outputs differ')
    result = {'status': 'match', 'differences': 0, 'files': len(OUTPUTS), 'fresh_process': True,
        'network_blocked': True, 'outputs_sha256': {n: file_sha(DIRECTORY/'study'/n) for n in OUTPUTS},
        'prepared_sha256': file_sha(DIRECTORY/'prepared.json'), 'new_trading_backtests': 0}
    save(DIRECTORY/'offline-verification.json', result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'study', 'replay', 'worker', 'offline', 'protect'))
    parser.add_argument('--mode', choices=('study', 'replay')); args = parser.parse_args()
    if args.action == 'prepare': result = prepare()
    elif args.action == 'worker': result = worker(args.mode)
    elif args.action == 'offline': result = offline()
    elif args.action == 'protect': result = protect('data', DIRECTORY)
    else: result = run(args.action)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
